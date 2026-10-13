// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func.func @partial_mesh_major_axis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}], unreduced={"x"}>})
func.func @partial_mesh_major_axis(
    %arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[true, true, true, true, false, false, false, false]> : tensor<8xi1>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<8xi1>, tensor<ui32>) -> tensor<1xi1>
  // CHECK-NEXT: %[[SCALAR:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.select %[[SCALAR]], %[[ARG0]], %[[ZERO]] : tensor<i1>, tensor<4x4xf32>
  %0 = sdy.replicated_to_unreduced {"x"} %arg0 out_sharding=<@mesh_2_4, [{}, {"y"}], unreduced={"x"}> : tensor<4x16xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x4xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @partial_mesh_minor_axis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<2x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}], unreduced={"y"}>})
func.func @partial_mesh_minor_axis(
    %arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[true, false, false, false, true, false, false, false]> : tensor<8xi1>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<8xi1>, tensor<ui32>) -> tensor<1xi1>
  // CHECK-NEXT: %[[SCALAR:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x16xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.select %[[SCALAR]], %[[ARG0]], %[[ZERO]] : tensor<i1>, tensor<2x16xf32>
  %0 = sdy.replicated_to_unreduced {"y"} %arg0 out_sharding=<@mesh_2_4, [{"x"}, {}], unreduced={"y"}> : tensor<4x16xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @full_mesh_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>})
// CHECK-SAME: -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x", "y"}>})
func.func @full_mesh_axes(
    %arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>})
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x", "y"}>}) {
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[true, false, false, false, false, false, false, false]> : tensor<8xi1>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<8xi1>, tensor<ui32>) -> tensor<1xi1>
  // CHECK-NEXT: %[[SCALAR:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x16xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.select %[[SCALAR]], %[[ARG0]], %[[ZERO]] : tensor<i1>, tensor<4x16xf32>
  %0 = sdy.replicated_to_unreduced {"x", "y"} %arg0 out_sharding=<@mesh_2_4, [{}, {}], unreduced={"x", "y"}> : tensor<4x16xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
