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

def ShardingConstraintToReshardPass : Pass<"sdy-sharding-constraint-to-reshard", "func::FuncOp"> {
  let summary = "Converts ShardingConstraintOp into ReshardOp.";
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def SinkDataFlowEdgesPass : Pass<"sdy-sink-data-flow-edges", "func::FuncOp"> {
  let summary = "Sinks all `DataFlowEdgeOp` into their input.";
  let description = [{
    Moves the sharding of each `DataFlowEdgeOp` to its input (the root target of
    the edge), and replaces the op with its input.

    TODO(tomnatan): consider moving the sharding to all targets that can have a
    sharding attached.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}
