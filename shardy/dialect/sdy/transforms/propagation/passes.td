/* Copyright 2024 The Shardy Authors.

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

def BasicPropagationPass : PassBase<"sdy-basic-propagate", "BasicPropagationPassImpl"> {
  let summary = "Runs the basic sharding propagation algorithm.";
  let description = [{
    The basic propagation algorithm is the lowest strategy of propagation in the
    hierarchy, that doesn't do any conflict resolution, and instead propagates
    axes that are compatible between all operands and results.

    Options:
      * `-keep-sharding-rules` : whether to keep existing and created op
        sharding rules
      * `-module-dump-directory` : where to dump any rewritten modules for
        debugging
      * `-conservative-propagation` : whether to disallow split axes and
        non-divisible sharding axes during propagation
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def AggressivePropagationPass : PassBase<"sdy-aggressive-propagate", "AggressivePropagationPassImpl"> {
  let summary = "Runs the aggressive sharding propagation algorithm.";
  let description = [{
    Propagate shardings with an aggressive factor propagation strategy. The
    basic strategy only propagates shardings without conflicts, while the
    aggressive strategy resolves conflicts. Higher aggressiveness can reduce the
    memory footprint at the cost of potential communication.

    Options:
      * All options from `BasicPropagationPass`
      * `-propagation-strategy` : which factor propagation strategy to use
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def OpPriorityPropagationPass : PassBase<"sdy-op-priority-propagate", "OpPriorityPropagationPassImpl"> {
  let summary = "Runs the op-priority propagation algorithm.";
  let description = [{
    Propagates shardings in iterations, based on a set of op heuristics. Each
    heuristic determines whether to propagate in forward, backward, or both
    directions for a given op.

    This starts at op-priority 0 and increases until all op heuristics have been
    exhausted and the program has reached a fixed-point.

    For a priority p, all op heuristics are considered from 0 <= i < p. The most
    expressive direction is chosen for each op, with
    `BOTH > BACKWARD == FORWARD > NONE`. If during this iteration,
    `FORWARD`/`BACKWARD` is seen first then `BACKWARD`/`FORWARD` is seen, the
     direction is `BOTH`.

    This propagation strategy extends the aggressive propagation strategy, which
    means that at each op-priority iteration, a full aggressive propagation is
    applied (see `AggressivePropagationPass`).

    Options:
      * All options from `AggressivePropagationPass`
      * `-run-op-priority-propagation` : whether to run (or skip) op-priority
        propagation
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def UserPriorityPropagationPass : PassBase<"sdy-user-priority-propagate", "UserPriorityPropagationPassImpl"> {
  let summary = "Runs the user-priority propagation algorithm.";
  let description = [{
    Propagates shardings in iterations, starting from the highest priority
    (lowest number, normally p0) to the lowest priority (highest number). In
    other words, at the i-th iteration, only dimension shardings with priority
    <=i are propagated.

    This propagation strategy extends the op-priority propagation strategy,
    which means that at each user-priority iteration, a full op-priority
    propagation is applied (see `OpPriorityPropagationPass`).

    Options:
      * All options from `OpPriorityPropagationPass`
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def PopulateOpShardingRulesPass : Pass<"sdy-populate-op-sharding-rules", "func::FuncOp"> {
  let summary = "Populates all registered ops with an `OpShardingRuleAttr`.";
  let description = [{
    Populates all registered ops with an `OpShardingRuleAttr`, which is used for
    debugging/testing the registered sharding rules. Propagation already does
    this just-in-time, but this pass does it all at once.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];

  let options = [
    // TODO(b/347180954): remove conservative propagation once the cost model
    // supports split axes and padding.
    Option<"conservativePropagation", "conservative-propagation", "bool",
           /*default=*/"false",
           "whether to disllow rules that can propagate non-divisible sharding "
           "axes">
  ];
}
