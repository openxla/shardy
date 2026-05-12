// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @result_indivisible
func.func @result_indivisible(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) -> tensor<3x8xf32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:4, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<3x8xf32>

  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xf32>
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<3x8xf32>

  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:3, 0:8] : (tensor<4x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: return %[[TRIM]] : tensor<3x8xf32>
  return %1 : tensor<3x8xf32>
}

// CHECK-LABEL: func @input_indivisible
func.func @input_indivisible(%arg0: tensor<4x7xi32> )
  -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>

  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xi32>) -> tensor<4x6xi32>
  %1 = stablehlo.slice %0 [0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>}
    : (tensor<4x7xi32>) -> tensor<4x6xi32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<4x6xi32>
  return %1 : tensor<4x6xi32>
}
