// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func.func @sharded_splat
// CHECK-SAME:    -> (tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
func.func @sharded_splat() -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<2x1xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} dense<1.0> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x1xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @sharded_dense
// CHECK-SAME:    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
func.func @sharded_dense() -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[GLOBAL_CST:.*]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C4:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // CHECK-NEXT: %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C4]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-NEXT: %[[START_0:.*]] = stablehlo.multiply %[[IDX]], %[[C2]] : tensor<i32>
  // CHECK-NEXT: %[[START_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[LOCAL_SLICE:.*]] = stablehlo.dynamic_slice %[[GLOBAL_CST]], %[[START_0]], %[[START_1]], sizes = [2, 2] : (tensor<4x2xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  // CHECK-NEXT: return %[[LOCAL_SLICE]] : tensor<2x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @unsharded_splat
// CHECK-SAME:    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>})
func.func @unsharded_splat() -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}]>]>} dense<1.0> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @unsharded_dense
// CHECK-SAME:    -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>})
func.func @unsharded_dense() -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK-SAME{LITERAL}: dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}]>]>} dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<4x2xi32>
  return %0 : tensor<4x2xi32>
}

// CHECK-LABEL: func.func @unreduced_splat
// CHECK-SAME:    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>})
func.func @unreduced_splat() -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[true, true, true, true, false, false, false, false]> : tensor<8xi1>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<8xi1>, tensor<ui32>) -> tensor<1xi1>
  // CHECK-NEXT: %[[SCALAR:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.select %[[SCALAR]], %[[CST]], %[[ZERO]] : tensor<i1>, tensor<4x4xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>} dense<1.0> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
