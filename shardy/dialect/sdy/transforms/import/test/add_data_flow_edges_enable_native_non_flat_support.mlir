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
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
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
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0, %0) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}
