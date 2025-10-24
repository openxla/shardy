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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;

void addExportPipeline(OpPassManager& pm, const ExportOptions& options) {
  // CSE the graph as this will deduplicate any duplicated transfers at the top
  // level of the function and hlo computations nested within fragments.
  // NOTE: a possible issue with applying CSE here is that in the *very
  // unlikely* scenario in which we have:
  //      fwd_fragment:N = fragment ...
  //      remat_fragment:N = fragment ...
  //      bwd_fragment = fragment (remat_fragment)
  // with fwd_fragment and remat_fragment being identical and both pure, then we
  // will deduplicate them and use fwd_fragment, which will cause a memory usage
  // regression. However, this is very unlikely to happen and we can always
  // revisit this if it does.
  pm.addNestedPass<FuncOp>(createCSEPass());

  if (options.copyConstantsFromProducerToConsumer) {
    // Apply this pass before DCE as it will leave some operations unused.
    pm.addNestedPass<FuncOp>(createCopyConstantsPass());
  }

  // Canonicalize the program and dedup operands and results of fragments.
  // TODO: jupvfranco - consider using `enabledPatterns` to apply only fragment
  // canonicalization patterns.
  pm.addPass(createCanonicalizerPass(
      GreedyRewriteConfig().setRegionSimplificationLevel(
          GreedySimplifyRegionLevel::Disabled)));
  pm.addNestedPass<FuncOp>(createFragmentDedupPass());

  // This optimization may affect certain use cases negatively. Thus, it's
  // disabled by default, but users can enable it on a per-module basis.
  if (options.applyMergeTransfers) {
    // Merges transfers that share the same producer and consumer fragments to
    // minimize the number of transfers. This pass does not cleanup unused
    // fragment results/args, so we should run it before fragment DCE.
    // We also need to run it after deduping fragment operands/results in order
    // to reduce the size of the concats (i.e., in case a fragment result is
    // duplicated with both duplicates used by the same consumer).
    // TODO: jupvfranco - consider applying this in the optimize pipeline. We
    // cannot do that yet, because we need to run it after Shardy prop, which
    // happens in between the optimization and export passes.
    pm.addNestedPass<FuncOp>(createMergeTransfersPass());

    // Run CSE and fragment dedup again after merging transfers as the
    // -mpmd-merge-transfers pass may have created duplicated concat, slice and
    // reshape ops.
    pm.addNestedPass<FuncOp>(createFragmentDedupPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // Remove any dead-code by eliminating unused fragment results and arguments
  // and by DCE'ing the fragment bodies.
  pm.addNestedPass<FuncOp>(createFragmentDcePass());

  // Must be applied after the last -mpmd-fragment-dedup, as it may add
  // duplicated fragment results and after -canonicalize, as it may add
  // identity fragments, which would be canonicalized away.
  UniquifyFunctionInputsOutputsPassOptions uniquifyOptions;
  uniquifyOptions.useTransferInsteadOfFragment = true;
  pm.addNestedPass<FuncOp>(
      createUniquifyFunctionInputsOutputsPass(uniquifyOptions));

  // The fragments created by the pass above maybe slowdown compilation (more
  // fragments to compile) and may cause performance regressions. Thus, we merge
  // them with other fragments.
  pm.addNestedPass<FuncOp>(createMergeInferredFragmentsPass());

  // Mark each fragment with the inputs and outputs which are offloaded to host
  // memory.
  pm.addNestedPass<FuncOp>(createMarkOffloadedInputOutputPass());

  // Propagate `mhlo.layout_mode` attributes from program inputs to fragments
  // that are consumers of program input, and propagate `mhlo.layout_mode`
  // attributes from program outputs to fragments that are output producers.
  pm.addNestedPass<FuncOp>(createMarkInputOutputWithLayoutsPass());

  // Before we create any dependencies between fragments, delay the
  // execution of inferred fragments to as late as possible, not to create
  // unnecessary dependencies on them, which could potentially delay the
  // execution of user-defined fragments, or even increase memory usage (the
  // produced tensors say live for longer).
  pm.addNestedPass<FuncOp>(createDelayInferredFragmentsPass());
  // Delay the execution of transfers from CPU to as late as possible to reduce
  // the amount of data in memory.
  pm.addNestedPass<FuncOp>(createDelayTransfersFromCpuPass());

  // This pass marks input and output aliasing or donation. For each fragment op
  // whose input can be aliased with an output, it adds an XLA
  // `tf.aliasing_output` attribute. Otherwise, for each input that can be
  // donated, it adds a `jax.buffer_donor` attribute. When lowering the fragment
  // op to fragment calls, the `tf.aliasing_output` and `jax.buffer_donor`
  // attributes will be set for the corresponding argument.
  pm.addNestedPass<FuncOp>(createMarkAliasingAndDonationPass());

  // Mark each fragment with how much memory should be left free to account for
  // live buffers produced by other fragments. This should be run after the
  // offloading and aliasing passes.
  pm.addNestedPass<FuncOp>(createMarkFragmentReservedMemoryPass());

  // This pass should be applied after all passes that operate on fragment ops.
  LowerToFragmentCallsPassOptions lower_to_fragment_calls_options;
  lower_to_fragment_calls_options.groupAcrossMeshes =
      options.groupFragmentsAcrossMeshes;
  lower_to_fragment_calls_options.verboseLogging = options.verboseLogging;
  pm.addPass(createLowerToFragmentCallsPass(
      std::move(lower_to_fragment_calls_options)));
}

void registerExportPipeline() {
  PassPipelineRegistration<>(
      "mpmd-export-pipeline",
      "Run the standard set of passes to export an MPMD program.",
      [](OpPassManager& pm) {
        addExportPipeline(pm);
      });
}

}  // namespace mlir::mpmd
