/* Copyright 2025 The MPMD Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "shardy/integrations/python/jax/mpmd/jaxlib/mpmd_program.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"
#include "shardy/dialect/mpmd/transforms/export/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"
#include "shardy/integrations/python/jax/mpmd/jaxlib/throw_error.h"

namespace mlir::mpmd {

namespace {

// Sets `jax.buffer_donor` attribute on the donated arguments that do not have
// either `jax.buffer_donor` or `tf.aliasing_output` attributes set.
// It is necessary to do this for cases when JAX does not have sufficient info
// to mark `jax.buffer_donor` attributes. It is expected that every array
// corresponding to an index in `donate_argnums` is safe to donate as these
// have been marked as donatable by users.
void SetArgDonationAttributes(func::FuncOp func_op,
                              const std::vector<int64_t>& donate_argnums) {
  for (int64_t arg_num : donate_argnums) {
    if (!func_op.getArgAttrOfType<BoolAttr>(arg_num, kBufferDonationAttrName) &&
        !func_op.getArgAttrOfType<IntegerAttr>(arg_num, kAliasingAttrName)) {
      func_op.setArgAttr(arg_num, kBufferDonationAttrName,
                         BoolAttr::get(func_op.getContext(), true));
    }
  }
}

// Verifies that only donated arguments have `jax.buffer_donor` or
// `tf.aliasing_output` attributes set.
void VerifyOnlyDonatedArgsHaveDonationAttributes(
    func::FuncOp func_op, const std::vector<int64_t>& donate_argnums) {
  std::set<int64_t> donated_argnum_set(donate_argnums.begin(),
                                       donate_argnums.end());
  for (int64_t arg_num = 0; arg_num < func_op.getNumArguments(); ++arg_num) {
    if (donated_argnum_set.count(arg_num) == 0) {
      if (func_op.getArgAttrOfType<BoolAttr>(arg_num,
                                             kBufferDonationAttrName) ||
          func_op.getArgAttrOfType<IntegerAttr>(arg_num, kAliasingAttrName)) {
        ThrowError("Argument " + std::to_string(arg_num) +
                   " that is not donated cannot have `jax.buffer_donor` "
                   "nor `tf.aliasing_output` attributes set");
      }
    }
  }
}

}  // namespace

PartitioningOptions ParsePartitioningOptions(
    std::map<std::string, std::variant<std::string, bool>> options) {
  PartitioningOptions parsed_options;
#define PARSE_BOOL_OPTION(field)                             \
  if (auto it = options.find(#field); it != options.end()) { \
    parsed_options.field = std::get<bool>(it->second);       \
  }

  PARSE_BOOL_OPTION(mpmd_infer_transfers);
  PARSE_BOOL_OPTION(mpmd_infer_cross_mesh_reductions);
  PARSE_BOOL_OPTION(mpmd_merge_inferred_with_cloning_during_import);
  PARSE_BOOL_OPTION(mpmd_fragment_remat);
  PARSE_BOOL_OPTION(mpmd_split_bwd_fragments);
  PARSE_BOOL_OPTION(mpmd_assume_homogeneous_devices);
#undef PARSE_BOOL_OPTION

  if (auto it = options.find("mpmd_pipeline_schedule"); it != options.end()) {
    std::string schedule_str = std::get<std::string>(it->second);
    if (std::optional<PipelineSchedule> parsed_schedule =
            ParsePipelineSchedule(schedule_str)) {
      parsed_options.mpmd_pipeline_schedule = *parsed_schedule;
    } else {
      SDY_LOG(FATAL) << "Invalid pipeline schedule: " << schedule_str;
    }
  }
  return parsed_options;
}

ErrorDiagnosticHandler::ErrorDiagnosticHandler(MLIRContext* context)
    : SourceMgrDiagnosticHandler(source_mgr_, context, diag_stream_),
      diag_stream_(diag_str_) {
  setHandler([&](Diagnostic& diag) { return HandleDiagnostic(diag); });
  // Set it to a large number to avoid truncation of the call stack.
  // We don't currently have a use-case where we'd prefer to truncate it.
  setCallStackLimit(100);
}

ErrorDiagnosticHandler::~ErrorDiagnosticHandler() {
  SDY_CHECK(consumed_) << "Error must be thrown before destruction";
}

void ErrorDiagnosticHandler::ConsumeStatus(LogicalResult result) {
  consumed_ = true;
  if (failed(result) && error_str_.empty()) {
    return ThrowError("Unknown MLIR failure");
  }
  if (!error_str_.empty()) {
    if (!failed(result)) {
      SDY_LOG(WARNING) << "MLIR error: " << error_str_
                       << " but MLIR returned success";
    }
    return ThrowError(error_str_);
  }
}

LogicalResult ErrorDiagnosticHandler::HandleDiagnostic(Diagnostic& diag) {
  // Emit the diagnostic and flush the stream.
  diag_str_.clear();
  emitDiagnostic(diag);
  diag_stream_.flush();

  // Emit non-errors to INFO instead of the internal status.
  if (diag.getSeverity() != DiagnosticSeverity::Error) {
    SDY_LOG(INFO) << diag_str_;
    return success();
  }

  error_str_ = diag_str_;

  // Return success to show that we consumed the diagnostic.
  return success();
}

PartitioningResult MpmdProgram::ApplyPartitioning(PartitioningPhase phases) {
  if ((phases & ~PartitioningPhase::kAll) != 0) {
    // Check that no undefined phase bits are set.
    ThrowError("Invalid PartitioningPhase: " + std::to_string(phases));
  }
  loadAllRequiredDialects(module->getContext());

  func::FuncOp main_func = GetMainFunction(module);
  SetTopology(named_meshes, main_func);
  SetArgDonationAttributes(main_func, donate_argnums);

  // It is not necessary to do this
  // validation after the export pipeline because here we're only checking that
  // the attributes set on the main func are consistent with the received donate
  // args.
  VerifyOnlyDonatedArgsHaveDonationAttributes(main_func, donate_argnums);

  SDY_LOG(INFO) << "Importing function named " << func_name
                << " for MPMD partitioning.";

  Import(module);

  SDY_LOG(INFO) << "Optimizing function named " << func_name
                << " for pipeline parallelism.";
  Optimize(module);

  SDY_LOG(INFO) << "Applying SDY propagation to function named " << func_name
                << ".";
  PropagateSharding(module);

  SDY_LOG(INFO) << "Exporting MPMD function named " << func_name << ".";
  Export(module);

  return PartitioningResult(module);
}

void MpmdProgram::Import(ModuleOp module) {
  PassManager pm(module->getName());
  pm.enableVerifier(false);

  ImportOptions import_options;
  import_options.nameToMeshAssignment = {std::move(assignment)};
  import_options.inputIndexToMeshAssignment = {
      ConvertMeshVectorToMap(input_meshes)};
  import_options.outputIndexToMeshAssignment = {
      ConvertMeshVectorToMap(output_meshes)};
  import_options.mergeAfterScheduling = options.mpmd_merge_after_scheduling;
  import_options.absorbInferredFragmentsOnEntryPointFunction =
      options.mpmd_absorb_inferred_fragments_on_entry_point_function;
  import_options.cloneInferredFragments =
      options.mpmd_merge_inferred_with_cloning_during_import;
  import_options.inferMeshOptions = {
      options.mpmd_infer_transfers,
      options.mpmd_infer_cross_mesh_reductions,
  };
  import_options.splitBwdFragments = options.mpmd_split_bwd_fragments;
  addImportPipeline(pm, import_options);
  ErrorDiagnosticHandler diagnostic_handler(module.getContext());
  return diagnostic_handler.ConsumeStatus(pm.run(module));
}

void MpmdProgram::Optimize(ModuleOp module) {
  PassManager pm(module->getName());
  pm.enableVerifier(false);

  OptimizeOptions optimize_options;
  optimize_options.fragmentMergeRules = llvm::to_vector(fragment_merge_rules);
  optimize_options.mergeAfterScheduling = options.mpmd_merge_after_scheduling;
  optimize_options.applyFragmentRemat = options.mpmd_fragment_remat;
  optimize_options.mergeRematFragments = options.mpmd_merge_remat_fragments;
  optimize_options.absorbInferredFragmentsOnEntryPointFunction =
      options.mpmd_absorb_inferred_fragments_on_entry_point_function;
  optimize_options.cloneInferredFragments =
      options.mpmd_merge_inferred_with_cloning_during_import;
  optimize_options.pipelineSchedule = options.mpmd_pipeline_schedule;
  addOptimizePipeline(pm, optimize_options);

  ErrorDiagnosticHandler diagnostic_handler(module.getContext());
  return diagnostic_handler.ConsumeStatus(pm.run(module));
}

void MpmdProgram::PropagateSharding(ModuleOp module) {
  PassManager pm(module->getName());
  pm.enableVerifier(false);

  addShardingPropagationPipeline(pm, "");
  module->setAttr(kIsSdyPartitioned, Builder(module).getBoolAttr(true));

  ErrorDiagnosticHandler diagnostic_handler(module.getContext());
  return diagnostic_handler.ConsumeStatus(pm.run(module));
}

void MpmdProgram::Export(ModuleOp module) {
  PassManager pm(module->getName());
  pm.enableVerifier(false);

  ExportOptions export_options;
  export_options.copyConstantsFromProducerToConsumer =
      options.mpmd_copy_constant_creation_from_producer_to_consumer;
  export_options.groupFragmentsAcrossMeshes =
      options.mpmd_assume_homogeneous_devices;
  export_options.applyMergeTransfers = options.mpmd_apply_merge_transfers_pass;
  export_options.verboseLogging = true;
  addExportPipeline(pm, export_options);

  ErrorDiagnosticHandler diagnostic_handler(module.getContext());
  return diagnostic_handler.ConsumeStatus(pm.run(module));
}

}  // namespace mlir::mpmd
