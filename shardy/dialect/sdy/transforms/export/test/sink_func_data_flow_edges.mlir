// RUN: sdy_opt %s -split-input-file -sdy-sink-func-data-flow-edges | FileCheck %s

// CHECK-LABEL: func private @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: return %[[NEGATE]]
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]])
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: return %[[NEGATE]]
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: @multiple_calls_on_same_func(%arg0: tensor<8xf32>)
func.func @multiple_calls_on_same_func(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[ABS0]])
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL0]]
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ABS1]])
  // CHECK-NEXT: %[[ABS2:.*]] = stablehlo.abs %[[CALL1]]
  // CHECK-NEXT: return %[[ABS2]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  %4 = call @bar(%3) : (tensor<8xf32>) -> (tensor<8xf32>)
  %5 = sdy.func_data_flow_edge %4 : tensor<8xf32>
  %6 = stablehlo.abs %5 : tensor<8xf32>
  return %6 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK-NEXT: return %[[ADD]]
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  %1 = sdy.func_data_flow_edge %arg1 : tensor<8xf32>
  %2 = stablehlo.add %0, %1: tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]], %[[ABS1]])
  // CHECK-NEXT: %[[ABS2:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS2]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = stablehlo.abs %arg0 : tensor<8xf32>
  %2 = call @bar(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %3 = sdy.func_data_flow_edge %2 : tensor<8xf32>
  %4 = stablehlo.abs %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK-NEXT: return %[[ADD]]
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  %1 = sdy.func_data_flow_edge %arg1 : tensor<8xf32>
  %2 = stablehlo.add %0, %1: tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_argument_same_operand(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_argument_same_operand(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]], %[[ABS0]])
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0, %0) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: return %[[NEGATE]]
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func @simple_call_graph_on_func_with_sharded_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_sharded_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: %[[ABS2:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  %4 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: return %[[NEGATE]]
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func @func_data_flow_edge_has_sharding_call_does_not(%arg0: tensor<8xf32>)
func.func @func_data_flow_edge_has_sharding_call_does_not(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}]>]>}
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: %[[ABS2:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  %4 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: return %[[NEGATE]]
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  %1 = stablehlo.negate %0: tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}]>]>}
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS]]
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  %1 = call @bar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK-LABEL: func @main_calls_foo_calls_bar(%arg0: tensor<8xf32>)
func.func @main_calls_foo_calls_bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[CALL]]
  // CHECK-NEXT: return %[[ABS1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = sdy.func_data_flow_edge %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}
