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

#include <utility>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/mpmd/transforms/common/merge_fragments.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/common/scheduler_preprocess.h"
#include "shardy/dialect/mpmd/transforms/import/infer_mesh_assignment.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;

void addImportPipeline(OpPassManager& pm, ImportOptions options) {
  // Add a few passes to make the module ready for MPMD partitioning.
  pm.addNestedPass<FuncOp>(stablehlo::createChloLegalizeToStablehloPass());

  pm.addPass(createInlinerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // Canonicalization / Target Independent Optimization needed for two things:
  // 1. Aggressive constant folding: required for later fragment merging.
  // 2. Flatten concat ops: needed for mesh inference as it doesn't support
  //    nested concat ops.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addPass(createCopyTopologyFromMainPass());

  if (options.enableHeterogeneousMeshes) {
    pm.addPass(createGenerateSdyMeshesFromTopologyPass());
  }

  // Unroll mpmd.for loops as they aren't yet supported by mesh inference.
  // TODO(jupvfranco): postpone unrolling until after SPMD propagation.
  pm.addNestedPass<FuncOp>(createUnrollForLoopsPass());
  // After unrolling, we may have slice(stack(x1, ..., xn), index) in the code
  // caused by for loops with enumeration of inputs. If x1 ... xn are large
  // tensors, this could cause OOMs. However, given that index is a constant,
  // we should be able to canonicalize this pattern away.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Don't fold anything that increases file size (iota -> cst)
  stablehlo::StablehloTargetIndependentOptimizationPassOptions
      stablehlo_optimization_options;
  stablehlo_optimization_options.foldOpElementLimit = 0;
  pm.addNestedPass<FuncOp>(
      stablehlo::createStablehloTargetIndependentOptimizationPass(
          stablehlo_optimization_options));
  // Make sure the mpmd.calls include a call_counter, even if originated from a
  // for_loop.
  // TODO(jupvfranco): Consider using the unroll counter instead of the call
  // counter in the scheduler pass.
  pm.addNestedPass<FuncOp>(createFromUnrollToCallCounterPass());

  // Sink negligible ops into call_ops. This is not strictly necessary, however
  // it is advantageous for two main reasons: 1) it reduces in workload in mesh
  // inference; and 2) the sunken ops can be immediately merged into fragments.
  pm.addPass(createSinkNegligibleOpsIntoCallOpPass());

  // This pass may leave unused outputs in named computations. Thus, we apply
  // it before we simplify named_computations in order to remove those outputs.
  pm.addNestedPass<FuncOp>(createInsertNamelessCloneOfNeglibleOpsPass());

  // Needs to be applied before MPMD import passes as those replace named
  // computations with fragments that are assigned to meshes, which can cause
  // tensors to be assigned to the wrong mesh if they are just passed through a
  // named computation without being used in it (which the simplify pass will
  // eliminate).
  pm.addNestedPass<FuncOp>(createSimplifyNamedComputationsPass());

  // Map main function inputs and outputs to meshes.
  MapInputOutputToMeshPassOptions map_in_out_options;
  map_in_out_options.inputAssignment =
      std::move(options.inputIndexToMeshAssignment);
  map_in_out_options.outputAssignment =
      std::move(options.outputIndexToMeshAssignment);
  pm.addPass(createMapInputOutputToMeshPass(std::move(map_in_out_options)));

  // Inline any mpmd op nested in a named_computation, checking that its mesh
  // assignment matches that of the parent. This is needed as we only map names
  // at the top level of the program (i.e., all fragments and transfers are at
  // the top level of the functions).
  pm.addNestedPass<FuncOp>(createInlineNestedUserExposedOpsPass(
      InlineNestedUserExposedOpsPassOptions{options.nameToMeshAssignment}));

  // Validate that all named ops are only nested in mpmd functions.
  pm.addNestedPass<FuncOp>(createValidateNamedOpsInMpmdFuncPass());

  // Map named computations and named tensors to fragments, assigns/unassigns.
  pm.addNestedPass<FuncOp>(
      createMapNamedOpsToMpmdOpsPass(MapNamedOpsToMpmdOpsPassOptions{
          std::move(options.nameToMeshAssignment)}));

  // Introduce transfer ops from unassign/assign ops.
  pm.addPass(createIntroduceTransfersPass());

  // Erase unused block arguments from functions that are target of mpmd.calls.
  // We need to do this before mesh inference, which doesn't handle arguments
  // that are used by the return op very well.
  pm.addPass(createEraseUnusedCalleeBlockArgumentsPass());

  // Run infer mesh assignment passes.
  addInferMeshPipeline(pm, options.inputOutputConstraints,
                       std::move(options.inferMeshOptions));

  if (!options.mergeAfterScheduling) {
    AddMergeInferredFragmentsPasses(
        pm, options.absorbInferredFragmentsOnEntryPointFunction,
        options.cloneInferredFragments);
  }

  // Enforce the user-specified input/output equi-assignment constraints.
  pm.addNestedPass<FuncOp>(
      createEnforceEquishardingPass(EnforceEquishardingPassOptions{
          std::move(options.inputOutputConstraints)}));

  // Simplify all the fragments. We never introduce identity fragments in our
  // passes and any identity fragment that may have been created by a user
  // would have been simplified away with `simplify-named-computation-ops`.
  // Thus, we don't apply canonicalization again.
  pm.addNestedPass<FuncOp>(createFragmentDedupPass());
  pm.addNestedPass<FuncOp>(createFragmentDcePass());

  // Apply optimization passes that modify fragments so fragments are stable
  // before rule-based merging/scheduling in the partition pipeline.
  // Apply as many optimizations as possible before inlining.
  pm.addNestedPass<FuncOp>(createRemoveTransferCyclesPass());
  AddCallInliningRelatedPasses(pm);
  // Merge any inferred fragments with user-defined fragments that could not be
  // merged before because of CallOps.
  if (!options.mergeAfterScheduling) {
    pm.addNestedPass<FuncOp>(createMergeInferredFragmentsPass());
  }
  // Merge fragments into scheduling units.
  AddSchedulingPreprocessingPasses(pm, options.splitBwdFragments,
                                   options.verifyScheduleUnits);
}

namespace {

struct ImportPipelineOptions
    : public PassPipelineOptions<ImportPipelineOptions> {
  Option<UserAssignmentMapOption> nameToMeshAssignment{
      *this, "name-to-mesh-assignment",
      llvm::cl::desc(
          "Mapping between names (of computations and tensors) and mesh "
          "names, and optionally stage ids."),
      llvm::cl::init(UserAssignmentMapOption())};

  Option<bool> mergeAfterScheduling{
      *this, "merge-after-scheduling",
      llvm::cl::desc(
          "Whether to merge inferred fragments only after scheduling."),
      llvm::cl::init(false)};
};

}  // namespace

void registerImportPipeline() {
  PassPipelineRegistration<ImportPipelineOptions>(
      "mpmd-import-pipeline",
      "Run the standard set of passes to import an MPMD program with a fixed "
      "mesh assignment map.",
      [](OpPassManager& pm, const ImportPipelineOptions& pipelineOptions) {
        ImportOptions options;
        options.nameToMeshAssignment = pipelineOptions.nameToMeshAssignment;
        options.mergeAfterScheduling = pipelineOptions.mergeAfterScheduling;
        addImportPipeline(pm, std::move(options));
      });
}

}  // namespace mlir::mpmd
