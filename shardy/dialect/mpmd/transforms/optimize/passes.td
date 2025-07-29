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

def RematFragmentPass : PassBase<"mpmd-remat-fragment", "DistributedFunctionPass"> {
  let summary = "Rematerializes fragments.";
  let description = [{
    Finds pairs of fragments (forward+backward) that need to be rematerialized
    and clones every forward fragment before its backward user, replace all
    backward uses of values produced by the forward fragment with the cloned
    counterparts. This can be used for activation rematerialization in pipeline
    parallelism.

    When `merge_remat_fragments` is true, then we merge the remat fragments into
    their consumer fragments.
  }];

  let options = [
    Option<"mergeRematFragments", "merge-remat-fragments", "bool",
           /*default=*/"false",
           "Whether to merge the remat fragments into their consumer "
           "fragments.">
  ];
}

def PipelineSchedulerPass :
    PassBase<"mpmd-pipeline-scheduler", "DistributedFunctionPass"> {
  let summary = "Reorders the fragments to obtain a given pipeline schedule.";
  let description = [{
    Reorders fragments according to a pipeline schedule. The scheduling
    algorithm relies on a _happens before_ function that takes two fragments
    `f1` and `f2` as arguments and checks if `f1` *must be* scheduled before
    `f2`. This function requires that:
      - `f1` and `f2` are assigned to the same mesh,
      - `f1` and `f2` are both scheduling units (i.e., user defined fragments
      with call_counters defined), and
      - `f1` does not depend on `f2` and `f2` does not depend on `f1`.
    And it should be enough to express many state-of-the-art pipeline schedules.

    For each fragment `f1` that must be scheduled before `f2`, the scheduler
    pass creates a control-dependency from `f1` to `f2`. Then, it applies a
    topological sort on the module to guarantee that all dependencies are
    respected (and the program is in a valid SSA form). Finally, the pass
    removes from the graph any control-dependency introduced.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Option<"mustHappenBefore", "must-happen-before",
           "FragmentComparatorOption", /*default=*/
           "FragmentComparatorOption::GetBuiltIn(PipelineSchedule::k1F1B)",
           "A comparator that determines whether a fragment must be scheduled "
           "before another. Can be parsed from a built-in `PipelineSchedule` "
           "as follows: `builtin:<schedule-as-string>`.">
  ];
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
