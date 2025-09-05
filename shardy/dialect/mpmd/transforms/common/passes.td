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

include "mlir/Pass/PassBase.td"

// TODO: b/374694825 - This pass is not complete yet. In particular, we also
// need to consider: (a) side-ways merging. We need to be careful with this as
// it may have performance and jitting time implications. (b) relax the
// condition in which we move transfers next to their producers/consumers. In
// particular, we could move any transfer (perhaps any computation) that does
// not depend on the root fragment. This should be safe and maybe we can use a
// topological sort to do so, but needs more thinking.
def AbsorbInferredFragmentsPass :
    PassBase<"mpmd-absorb-inferred-fragments", "DistributedFunctionPass"> {
  let summary = "Root fragments absorb inferred fragments.";
  let description = [{
    Makes root fragments absorb inferred fragments, i.e., by
    merging inferred producer/consumer fragments into root fragments, where a
    root fragment is any fragment that is:
    - a user fragment, or
    - not used by any other fragment (e.g. a fragment used by the return op or a
    transfer only), or
    - not a user of a value produced by any other fragment (e.g., user of block
    arguments or transfers).

    In order to do so, the pass applies the following patterns, until it reaches
    a fixed point:

    (1) Given a root fragment `rf`, if there is an inferred fragment `ipf` such
    that `ipf` is a producer of `rf` and `rf` is the closest consumer of `ipf`,
    then `ipf` is merged into `rf`.

    And dually:

    (2) Given a root fragment `rf`, if there is an inferred consumer `icf` such
    that `icf` is a consumer of `rf` and `rf` is the closest producer of `icf`,
    then `icf` is merged into `rf`.

    This means we preserve the structure/shape of the program as defined by the
    user, via named computations and stage/mesh assignment.

    Note that this pass is quite aggressive in merging inferred fragments, and in
    particular, it could cause small differences across different stages that
    could increase the number of unique fragments to compile.

    This pass will warn us if the final non entry-point functions still include
    inferred fragments, as these could cause performance issues (e.g., gradient
    accumulation gone wrong).

    WARNING: if the program doesn't have any user-defined fragment and has lots
    of inferred fragments, then this pass could be very slow.
  }];

  let options = [
    Option<"absorbOnEntryPointFunction", "absorb-on-entry-point-function",
           "bool", /*default=*/"false",
           "Whether to absorb inferred fragments into user-defined fragments "
           "on entry-point functions, in addition to targets of mpmd.calls.">
  ];
}

//===----------------------------------------------------------------------===//
// Start of - Call op passes
//===----------------------------------------------------------------------===//

def CallInlinePass : Pass<"mpmd-call-inline", "ModuleOp"> {
  let summary = "Inlines all `mpmd.call` operations.";
  let description = [{
    Inlines `mpmd.call` operations, copying their attributes to any inlined
    operations.
  }];
}

def SinkNegligibleOpsIntoCallOpPass :
    Pass<"mpmd-sink-negligible-ops-into-call-op", "ModuleOp"> {
  let summary = "Sinks negligible ops into call-ops (i.e., the called function).";
  let description = [{
    Sinks (negligible) ops into called functions: if there is an op with zero
    operands and a single result which is used as a specific operand of *all*
    call ops to the same function, then we sink it into those call ops, i.e., we
    clone it into the called function and replace all uses of the respective
    argument with the clone. Sunken ops are removed from the caller function and
    unused arguments of the callee (and operands of respective call ops)
    removed. Note that this can potentially duplicate computation across many
    microbatches, when using call ops for microbatching. Though, this
    computation is most likely negligible as it takes no operands.
  }];
}

// TODO: b/359837378 - We should erase the attribute from other ops too.
def FromUnrollToCallCounterPass :
    PassBase<"mpmd-from-unroll-to-call-counter", "DistributedFunctionPass"> {
  let summary = "Converts the unroll counter attr of a call op to a call "
                "counter attr.";
  let description = [{
    Whenever a call op has a `unroll_counter` attribute, this pass replaces it
    with a `call_counter` attribute. This is needed for cases in which a
    sequence of calls results from unrolling a loop in MLIR (e.g., via
    `-mpmd-unroll-for-loops`) instead of from unrolling a loop at the python
    level.
  }];
}

def EraseUnusedCalleeBlockArgumentsPass :
    Pass<"mpmd-erase-unused-callee-block-arguments", "ModuleOp"> {
  let summary = "Erases any mpmd callee block argument that isn't used by an "
                "(hlo) computation.";
  let description = [{
    Erases unused block arguments from functions called by mpmd.calls. We
    consider a block argument to be unused if it has no used or is used only by
    the function's terminator, i.e., if it is not used by any hlo computation.
  }];
}

//===----------------------------------------------------------------------===//
// End of - Call op passes
//===----------------------------------------------------------------------===//

// TODO: b/372460554 - Support nested mpmd.for loops.
def UnrollForLoopsPass :
    PassBase<"mpmd-unroll-for-loops", "DistributedFunctionPass"> {
  let summary = "Fully unrolls `mpmd.for` loops.";
  let description = [{
    Creates a pass that completely unrolls `mpmd.for` ops, while
    attaching an unroll_counter attribute to each unrolled op.

    Requires: the unroll factor to be equal to the number of iterations.

    NOTE: This pass does not support nested loops, nor partial unrolling.
  }];

  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def CopyConstantsPass :
    PassBase<"mpmd-copy-constants", "DistributedFunctionPass"> {
  let summary = "Copies constants produced in one fragment to their consumers.";
  let description = [{
    Copies constants from their producer fragments to their consumer fragments,
    possibly through transfers.

    Example:

    ```mlir
    %f = fragment () () {
      return constant
    }
    %t = transfer %f
    fragment (%t) (%arg) {
      op(... %arg ...)
      ...
    }

     ~~>

    %f = fragment () () {
      return constant
    }
    %t = transfer %f
    fragment (%t) (%arg) {
      %c = constant
      op(... %c ...)
      ...
    }
    ```

    Note: this pass doesn't cleanup any unused code.

    This can be beneficial for runtime performance: it enables potential
    optimizations by putting the constant together with its users and it avoids
    transfers of constants. Additionally, it will improve memory usage: we
    reduce the space needed for parameters of the computation.
  }];
}

def FragmentDcePass :
    PassBase<"mpmd-fragment-dce", "DistributedFunctionPass"> {
  let summary = "Eliminates unused fragment arguments/results and simplifies "
                "fragment regions.";
  let description = [{
    Removes unused fragment arguments and results, while simplifying fragment
    regions, essentially eliminating dead MPMD code.
  }];
}

def FragmentDedupPass :
    PassBase<"mpmd-fragment-dedup", "DistributedFunctionPass"> {
  let summary = "Removes any duplicated operands and results in fragments.";
  let description = [{
    Removes any duplicated used arguments and results in fragments. This will
    leave the duplicate arguments and results unused. Other passes should be run
    to remove the unused arguments and results.
  }];
}

//===----------------------------------------------------------------------===//
// Start of - Fragment merging passes
//===----------------------------------------------------------------------===//

// Collection of passes that merge producer-consumer fragment ops that are
// assigned to the same mesh, within the same block, and if certain conditions
// apply (c.f., each type of merge pass below).
//
// The origin of the new fragment will be the de-duplicated merge of the origins
// of the two fragments.
//
// Note that we don't merge a producer fragment with its closest consumer
// fragment if the former has other (non-fragment) users before the consumer
// fragment (e.g. a transfer op), because we don't want to reorder any
// operations in these passes (which would also require data flow dependency
// analysis).
//
// Terminology:
//   - An *inferred* fragment is a fragment created by the compiler. I.e., its
//     `origin` attribute is empty.
//   - A *user (-defined) fragment* is a fragment that resulted from a
//     named_computation or from merging another user-defined fragment with an
//     inferred fragment. I.e., its `origin` attribute is not empty.

def MergeInferredFragmentsPass :
    PassBase<"mpmd-merge-inferred-fragments", "MergeFragmentBasePass"> {
  let summary = "Merges inferred fragments with user defined fragments.";
  let description = [{
    Merges inferred with user-defined or other inferred fragments. This pass is
    useful to clean-up/simplify the module and can be useful after other
    compiler passes that introduce inferred fragments, while
    `-mpmd-transfer-aware-merge` above is more invasive and should be used for
    optimization purposes only.

    NOTE: We assume that merging an inferred fragment to any other fragment
    never delays transfers.

    When `clone_inferred_fragments=true`, then this merging pass allows for
    certain fragments to be cloned. In particular, if we encounter a pair of
    fragments f1 and f2 such that:
    - f2 uses f1, and
    - f1 is inferred, pure, and sufficiently simple (single non-return op and
      single result),
    then we merge a clone of f1 into f2, i.e., f1 itself (and other
    users) remain independent of f2. It may be undesirable to merge inferred
    producer fragments without cloning, because it can create unnecessary
    dependencies between fragments. E.g.,

    ```mlir
    %inferred = frag m1 { return stablehlo.const … }
    %frag1 = frag m1 (%inferred, …)
    %frag2 = frag m1 (%inferred, …)

    ~>

    %inferred_frag1 = frag m1 (…) { … return const_m1, … }
    %frag2 = frag m2 (inferred_frag1, …)
    ```

    So frag2 now depends on inferred_frag1 and we create a dependency.

    However, sometimes we do want to merge in place, e.g., when the inferred
    fragment has collectives inside.

    NOTE: if the `mpmd.call` ops have been inlined, doing aggressive merging
    with `merge-any-consumer=true` may create dependencies between fragments of
    different microbatches, preventing certain reschedulings.
  }];
  let options = [
    Option<"cloneInferredFragments", "clone-inferred-fragments", "bool",
           /*default=*/"false",
           "Whether to clone inferred fragments. Chains of clonable fragments "
           "are merged one-by-one into their consumers and recursively.">,
    Option<"mergeAnyConsumer", "merge-any-consumer", "bool",
           /*default=*/"false",
           "Whether to merge with any consumer or only the closest consumer.">,
    Option<"mergeSideways", "merge-sideways", "bool", /*default=*/"false",
           "Whether to merge with the next fragment in the same mesh "
           "(neighbor), even if not a consumer.">
  ];
}

def MergeForwardWithBackwardPass :
    PassBase<"mpmd-merge-forward-with-backward", "MergeFragmentBasePass"> {
  let summary = "Merge forward fragments with backward fragments.";
  let description = [{
    Merge a producer forward fragment with a consumer backward fragment, if the
    former is immediately before the latter. This is only true for the last
    stage in 1F1B schedule, so it will not merge any fragments in previous
    stages, which is the intended behavior.
  }];
}

def MergeUserDefinedFragmentsIntoSchedulingUnitsPass :
    PassBase<"mpmd-merge-user-fragments-into-scheduling-units",
             "MergeFragmentBasePass"> {
  let summary = "Merge user based fragments pre pipeline scheduling passes.";
  let description = [{
    Merges pairs of user defined fragments to be used together with pipeline scheduling
    passes.

    NOTE: this pass requires every user-defined fragment to have one and only
    one transpose count, meaning we cannot apply it after
    `-mpmd-transfer-aware-merge`, which can result in fragments with multiple
    transpose counts per fragment.
  }];
}

def VerifyStageMergingPass :
    PassBase<"mpmd-verify-stage-merging", "DistributedFunctionPass"> {
  let summary = "Verifies that merging of fragments assigned to stages succeeded.";
  let description = [{
    Verifies that fragments with stage assignment have been correctly merged.
    This means that it's not possible to have in the module any two equivalent
    fragments in terms of assignment and counters.

    Two fragments are equivalent in terms of assignment and counters _iff_
    a. they are assigned to the same mesh,
    b. they are assigned to the same stage,
    c. they have the same transpose count, and
    d. either both have the same call counter or one of them doesn't have a call
    counter defined (i.e., an undefined call counter matches any call counter).

    This is needed to guarantee to the user that any computation assigned to the
    same stage is executed contiguously.
  }];
}

def RuleBasedMergePass :
    PassBase<"mpmd-rule-based-merge", "DistributedFunctionPass"> {
  let summary = "Merges fragments based on user-defined rules.";
  let description = [{
    Merges fragments based on a specified list of rules, each specifying a list
    of source fragment to merge (by their fragment info) and the target info to
    label the merged fragment.
  }];

  let options = [
    ListOption<"rules", "rules", "FragmentMergeRule",
               "A list of fragment merge rules, each with a list of source "
               "fragment infos and a target fragment info.">
  ];
}

//===----------------------------------------------------------------------===//
// End of - Fragment merging passes
//===----------------------------------------------------------------------===//

def MergeTransfersPass :
    PassBase<"mpmd-merge-transfers", "DistributedFunctionPass"> {
  let summary = "Merges sets of transfers that share the same producer and "
                "consumer fragments.";
  let description = [{
    Merges sets of transfers of the same payload type which
    share the same producer and consumer fragments. The payload values of these
    transfers have less elements than a given threshold, are not sharded and do
    not live in pinned_host.

    Merging a set of transfers means: concatenating the transferred values at
    producer site and splitting them at consumer site.

    Note: if a producer fragment has a set of transfers that is used by distinct
    consumers, this pass will duplicate the concatenation at the producer site,
    which can cause an increase in memory footprint, and unnecessary operations.
    Applying CSE and fragment IO dedup after this pass is recommended.
  }];

  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def SplitBwdFragmentsPass : Pass<"mpmd-split-bwd-fragments", "func::FuncOp"> {
  let summary = "Splits backward fragments based on transferred results.";
  let description = [{
    Splits backwards fragments so that any computation
    that does not flow into transferred results becomes a fragment of its own.
    The original fragment will return some residual values that will be passed in
    as extra operands to the split-out fragments.

    This split allows us to transfer results into other meshes earlier. One
    canonical use of this optimization will be splitting the activation gradient
    computation in back-propagation from the parameter gradient computation.
    Note also that due to the need to potentially thread through some residual
    values in the new fragments, memory pressure will increase.

    Note: In practice we care only about transfers to different meshes, so this
    pass should be best used after we've already fused transfers and eliminated
    intra-mesh transfers.

    Note: When splitting backward fragments we add extra residual values to the
    original one and pass them as extra arguments to the split-out fragment. We
    give these residual types fully replicated mesh types, which really assumes
    that we have not run any form of SPMD propagation prior to this pass.
  }];
}

def SplitAndPrioritizeTransferIndependentComputationsPass :
    Pass<"mpmd-split-and-prioritize-transfer-independent-computations",
         "func::FuncOp"> {
  let summary = "Splits backward fragments based on transferred results.";
  let description = [{
    Splits a fragment into two fragments, so that we can start computation
    early. I.e. we split the fragment into two fragments A -> B, where A does
    not rely on any transfer result, and is maximally large, and B relies on
    transfer results.

    Note: When splitting fragments we add extra residual values to the original
    one and pass them as extra arguments to the split-out fragment. We give
    these residual types fully replicated mesh types, which really assumes that
    we have not run any form of SPMD propagation prior to this pass.
  }];
}

def RemoveTransferCyclesPass :
    PassBase<"mpmd-remove-transfer-cycles", "DistributedFunctionPass"> {
  let summary = "Removes device-only transfer cycles from the program, avoiding "
                "unnecessary transfers.";
  let description = [{
    Removes transfer cycles.

    E.g. in symbols:

      x1 = transfer(x0) : m0 -> m1
      x2 = transfer(x1) : m1 -> m2
      x3 = transfer(x2) : m2 -> m3
      x0_1 = transfer(x3) : m3 -> m0
      x1_1 = transfer(x0_1) : m0 -> m1

    ~~>

      x1 = transfer(x0) : m0 -> m1
      x2 = transfer(x1) : m1 -> m2
      x3 = transfer(x2) : m2 -> m3
      x0_1 = x0
      x1_1 = x1

    i.e. we then we break the cycle by using the existing values, removing the
    unnecessary transfers.

    Note that this could increase memory overhead, since transferring the data
    away and back again means that there's a period where the data isn't on the
    device. Thus, we only do this if the cycle only contains device-to-device
    transfers, e.g. since a `device -> host -> device` cycle could be for memory
    purposes.

    This doesn't use the MLIR Canonicalizer, because that doesn't guarantee that
    everything is canonicalized, and also it's more expensive to apply.
  }];
}

// TODO(jupvfranco): we should create these copies using ifrt_ir reshard. We are
// not ready for that yet as we need better support for donation in the
// presence of reshards.
def UniquifyFunctionInputsOutputsPass :
    PassBase<"mpmd-uniquify-function-inputs-outputs", "DistributedFunctionPass"> {
  let summary = "Uniquifies any value returned multiple times or any block "
                "argument directly returned by the function.";
  let description = [{
    If a function returns the same value multiple times, creates multiple
    versions for that value, by creating a fragment assigned to that value's
    mesh which returns the value multiple times. After this pass, each return
    operand is unique. This is important to ensure that the respective results
    are allocated in different buffers, as in the following `jax.jit` example:

    ```python
    def f(x):
      y = x + x
      return y, y

    z1, z2 = f(5)
    z1 += 1
    print(z1) ~~> 6
    print(z2) ~~> 5
    ```

    Similarly, if a function returns a block argument, this pass creates an
    identity fragment for that block argument, guaranteeing that values are
    passed by value to the function, not by reference.
  }];

  let dependentDialects = ["mlir::mpmd::MpmdDialect"];
}

def SchedulingUnitVerifierPass :
    PassBase<"mpmd-scheduling-units-verifier", "DistributedFunctionPass"> {
  let summary = "Verifies if the program contains the required scheduling units.";
}

// TODO: b/378099938 - Remove this pass once we have a better way to handle
// transfers while merging fragments. We need this now because having a transfer
// in between two fragments prevents the merge pass from merging them.
def MoveTransfersToProducerPass :
    PassBase<"mpmd-move-transfers-to-producer", "DistributedFunctionPass"> {
  let summary = "Moves transfers next to their producers.";
  let description = [{
    Moves transfers next to their producers: if the operand is a block argument,
    move the transfer to the beginning of the block, otherwise move it after the
    defining op.
  }];
}
