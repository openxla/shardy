// RUN: sdy_opt %s -sdy-pad-for-divisibility

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @expand_then_trim
func.func @expand_then_trim(%arg0: tensor<4x7xi32> ) -> tensor<4x7xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{}, {"y"}] %[[ALL_SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4, 0:7] : (tensor<4x8xi32>) -> tensor<4x7xi32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x7xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %1 = sdy.all_gather [{}, {"y"}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x7xi32>
  return %1 : tensor<4x7xi32>
}

// CHECK-LABEL: func @propagate_through_all_gather
func.func @propagate_through_all_gather(%arg0: tensor<7x7xi32>) -> tensor<7x7xi32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg0, %[[CST]]
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{"x"}, {"y"}] %[[PAD1]]
  // CHECK: %[[AG1:.*]] = sdy.all_gather [{}, {"y"}] %[[SLICE1]]
  // CHECK: %[[SLICE_AG:.*]] = stablehlo.slice %[[AG1]] [0:8, 0:7]
  // CHECK: %[[PAD2:.*]] = stablehlo.pad %[[SLICE_AG]], %[[CST]]
  // CHECK: %[[SLICE2:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD2]]
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[SLICE2]] [0:7, 0:7]
  // CHECK: return %[[TRIM]]

  %0 = sdy.all_slice [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<7x7xi32>
  %1 = sdy.all_gather [{}, {"y"}] %0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x7xi32>
  %2 = sdy.all_slice [{}, {"y"}] %1 out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<7x7xi32>
  %3 = stablehlo.slice %2 [0:7, 0:7] : (tensor<7x7xi32>) -> tensor<7x7xi32>
  return %3 : tensor<7x7xi32>
}

// CHECK-LABEL: func @all_gather_trim
func.func @all_gather_trim(%arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) -> tensor<3x8xi32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:4, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xi32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<4x8xi32>) -> tensor<3x8xi32>
  // CHECK-NEXT: return %[[TRIM]] : tensor<3x8xi32>
  %0 = stablehlo.slice %arg0 [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xi32>) -> tensor<3x8xi32>
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<3x8xi32>
  return %1 : tensor<3x8xi32>
}
