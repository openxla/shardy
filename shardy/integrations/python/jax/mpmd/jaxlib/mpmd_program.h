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

#ifndef SHARDY_INTEGRATIONS_PYTHON_JAX_MPMD_JAXLIB_MPMD_PROGRAM_H_
#define SHARDY_INTEGRATIONS_PYTHON_JAX_MPMD_JAXLIB_MPMD_PROGRAM_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"
#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"

namespace mlir::mpmd {

// Represents phases of the partitioning pipeline. This is used as a bitmask to
// specify which phases to run.
//
// A bitmask is chosen for extensibility and to simplify the control flow within
// `ApplyPartitioning`. With a bitmask, determining whether to run a specific
// phase is a simple bitwise AND. The trade-off is that this is less type-safe
// and does not enforce a specific ordering. This is acceptable as the Python
// API layer is responsible for enforcing a valid sequence of phases.
enum PartitioningPhase : int32_t {
  kNone = 0,
  kImport = 1 << 0,
  kPartition = 1 << 1,
  kAll = kImport | kPartition,
};

struct PartitioningResult {
  mlir::ModuleOp mpmd_module;
  // Partition specs and mesh names of each input and output of the MPMD
  // module's main function.
  mlir::mpmd::FunctionIOShardingSpecsAndMeshes
      module_io_sharding_specs_and_meshes;

  explicit PartitioningResult(mlir::ModuleOp mpmd_module)
      : mpmd_module(mpmd_module),
        module_io_sharding_specs_and_meshes(
            ExtractFunctionIOShardingSpecsAndMeshes(
                GetMainFunction(mpmd_module))) {}
};

class StatusScopedDiagnosticHandler : public mlir::SourceMgrDiagnosticHandler {
 public:
  explicit StatusScopedDiagnosticHandler(mlir::MLIRContext* context);

  // Destruction CHECK-fails if ConsumeStatus has not been called.
  ~StatusScopedDiagnosticHandler();

  // Returns the aggregate status, if it is non-OK, or an error, if `result` is
  // mlir::failed. If the aggregate status is OK and mlir::succeeded(result),
  // returns OK.
  absl::Status ConsumeStatus(mlir::LogicalResult result);

 private:
  mlir::LogicalResult HandleDiagnostic(mlir::Diagnostic& diag);

  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;
  llvm::SourceMgr source_mgr_;
  absl::Status status_;
  bool consumed_ = false;
};

// Basic options for MPMD partitioning. We should improve how this is kept in
// sync with the python version.
struct PartitioningOptions {
  bool mpmd_infer_transfers = false;
  bool mpmd_infer_cross_mesh_reductions = false;
  bool mpmd_merge_inferred_with_cloning_during_import = false;
  bool mpmd_gspmd_propagate_sharding_across_meshes = false;
  bool mpmd_allow_intra_mesh_transfer = false;
  bool mpmd_fragment_remat = false;
  bool mpmd_merge_remat_fragments = false;
  bool mpmd_split_bwd_fragments = false;
  PipelineSchedule mpmd_pipeline_schedule = PipelineSchedule::kGPipe;
  bool mpmd_assume_homogeneous_devices = false;
  bool mpmd_absorb_inferred_fragments_on_entry_point_function = false;
  bool mpmd_copy_constant_creation_from_producer_to_consumer = false;
  bool mpmd_apply_merge_transfers_pass = false;
  bool mpmd_merge_after_scheduling = false;
};

PartitioningOptions ParsePartitioningOptions(
    std::map<std::string, std::variant<std::string, bool>> options);

// Struct used for holding information needed for partitioning a MPMD program.
struct MpmdProgram {
  mlir::ModuleOp module;
  std::string func_name;
  PartitioningOptions options;
  const std::vector<std::pair<std::string, FlatMesh>>& named_meshes;
  const mlir::mpmd::UserAssignmentMap& assignment;
  const std::vector<std::optional<std::string>>& input_meshes;
  const std::vector<std::optional<std::string>>& output_meshes;
  const std::vector<int64_t>& donate_argnums;
  const mlir::mpmd::FragmentMergeRules& fragment_merge_rules;

  // Runs the PartIR MPMD partitioning passes on the MPMD program.
  absl::StatusOr<PartitioningResult> ApplyPartitioning(
      PartitioningPhase phases);

 private:
  absl::Status Import(mlir::ModuleOp module);
  absl::Status Optimize(mlir::ModuleOp module);
  absl::Status PropagateSharding(mlir::ModuleOp module);
  absl::Status Export(mlir::ModuleOp module);
};

}  // namespace mlir::mpmd

#endif  // SHARDY_INTEGRATIONS_PYTHON_JAX_MPMD_JAXLIB_MPMD_PROGRAM_H_
