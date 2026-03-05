// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges='enable-native-non-flat-support=true'  | FileCheck %s

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_multiple_users_on_func_result(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_multiple_users_on_func_result(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
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
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  %1 = stablehlo.abs %arg0: tensor<8xf32>
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_results(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_results(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]]:2 = call @bar(%1) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]]#1 : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
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
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 sharding=<@mesh, [{"a"}]> : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_sharded_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_sharded_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @multiple_calls_on_same_func(%arg0: tensor<8xf32>)
func.func @multiple_calls_on_same_func(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL0:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
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
  // CHECK:      %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %arg1 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.add %arg0, %arg1: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1, %2) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = stablehlo.abs %arg0 : tensor<8xf32>
  %2 = call @bar(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %arg1 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.add %arg0, %arg1: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_argument_same_operand(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_argument_same_operand(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1, %1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0, %0) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_chain_call_graph(%arg0: tensor<8xf32>)
func.func @simple_chain_call_graph(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_non_flat_call_graph(%arg0: tensor<8xf32>)
func.func @simple_non_flat_call_graph(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[NEGATE]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[NEGATE]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE2:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE2]] : tensor<8xf32>
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
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_non_flat_call_graph_one_after_the_other(%arg0: tensor<8xf32>)
func.func @simple_non_flat_call_graph_one_after_the_other(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL0:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[EDGE0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = call @foo(%1) : (tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @call_on_same_func_twice_input_of_one_is_output_of_the_other(%arg0: tensor<8xf32>)
func.func @call_on_same_func_twice_input_of_one_is_output_of_the_other(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL0:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[EDGE0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = call @bar(%1) : (tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.add %1, %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_argument_is_input_to_call(%arg0: tensor<8xf32>)
func.func @simple_call_graph_argument_is_input_to_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %1 = stablehlo.abs %0 : tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_result_is_the_output_of_call(%arg0: tensor<8xf32>)
func.func @simple_call_graph_result_is_the_output_of_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%1) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: return %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %1 : tensor<8xf32>
}


// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_entry_contains_call_only(%arg0: tensor<8xf32>)
func.func @simple_call_graph_entry_contains_call_only(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: return %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0 : tensor<8xf32>
}

