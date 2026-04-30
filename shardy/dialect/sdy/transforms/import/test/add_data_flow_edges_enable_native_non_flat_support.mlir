// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges='enable-native-non-flat-support=true'  | FileCheck %s

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_on_func_with_single_argument
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_on_func_multiple_users_on_func_result
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  %3 = stablehlo.negate %1 : tensor<8xf32>
  %4 = stablehlo.add %2, %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) ->(tensor<8xf32>, tensor<8xf32>) {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  %1 = stablehlo.abs %arg0: tensor<8xf32>
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// test: simple_call_graph_on_func_with_multiple_results
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %[[CALL]]#0
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]#1
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1:2 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  %2 = stablehlo.add %1#0, %1#1 : tensor<8xf32>
  %3 = stablehlo.abs %1#1 : tensor<8xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_on_func_with_sharded_argument
// CHECK-LABEL: @main(%arg0: tensor<8xf32>)
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT:      %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: multiple_calls_on_same_func
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %[[CALL0]]
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]]
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ABS0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL1]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  %3 = call @bar(%2) : (tensor<8xf32>) -> (tensor<8xf32>)
  %4 = stablehlo.abs %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %arg1
  // CHECK-NEXT:                 stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.add %arg0, %arg1: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_on_func_with_multiple_argument
// CHECK-LABEL: @main(%arg0: tensor<8xf32>)
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS0]], %[[ABS1]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = stablehlo.abs %arg0 : tensor<8xf32>
  %2 = call @bar(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %arg1
  // CHECK-NEXT:                stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.add %arg0, %arg1: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_on_func_with_multiple_argument_same_operand
// CHECK-LABEL: @main(%arg0: tensor<8xf32>)
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]], %[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0, %0) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]]
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// test: simple_chain_call_graph
// CHECK-LABEL: @main(%arg0: tensor<8xf32>)
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT:                stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]]
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// test: simple_non_flat_call_graph
// CHECK-LABEL: @main(%arg0: tensor<8xf32>)
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[NEGATE]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL0]]
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE1]]
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[NEGATE]])
  // CHECK-NEXT: %[[EDGE2:.*]] = sdy.func_data_flow_edge %[[CALL1]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE2]]
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  %3 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %4 = stablehlo.abs %3 : tensor<8xf32>
  %5 = stablehlo.add %2, %4 : tensor<8xf32>
  return %5 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]]
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// test: simple_non_flat_call_graph_one_after_the_other
// CHECK-LABEL: @main(%arg0: tensor<8xf32>)
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[NEGATE]])
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %[[CALL0]]
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[EDGE0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL1]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = call @foo(%1) : (tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]]
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// test: call_on_same_func_twice_input_of_one_is_output_of_the_other
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[NEGATE]])
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.func_data_flow_edge %[[CALL0]]
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[EDGE0]])
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.func_data_flow_edge %[[CALL1]]
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = call @bar(%1) : (tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.add %1, %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_argument_is_input_to_call
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0)
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %1 = stablehlo.abs %0 : tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_result_is_the_output_of_call
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]])
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: return %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %1 : tensor<8xf32>
}


// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_entry_contains_call_only
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0)
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: return %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %arg0
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// test: simple_call_graph_entry_contains_call_only
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0)
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]]
  // CHECK-NEXT: return %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0 : tensor<8xf32>
}

