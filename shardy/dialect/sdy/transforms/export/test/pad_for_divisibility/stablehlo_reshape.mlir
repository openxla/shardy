// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file -verify-diagnostics | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reshape_pass_through_pad
func.func @reshape_pass_through_pad(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // Padding LHS input dim 0 (size 3) to 4 for x=4.
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x4xf32>
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<3x4xf32>

  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<3x4xf32>) -> tensor<3x4xf32>

  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<4x4xf32> to tensor<3x4xf32>
  // CHECK: return %[[CAST]] : tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reshape_divisible_participating
func.func @reshape_divisible_participating(%arg0: tensor<16xf32>) -> tensor<4x4xf32> {
  // All participating dimensions are divisible, no padding is added.
  // CHECK-NOT: stablehlo.pad
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %{{.*}} : (tensor<16xf32>) -> tensor<4x4xf32>
  %0 = sdy.all_slice [{"x"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}]> : tensor<16xf32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {"x"}]>]>} : (tensor<16xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reshape_mix_participating_and_passthrough
func.func @reshape_mix_participating_and_passthrough(%arg0: tensor<16x3xf32>) -> tensor<4x4x3xf32> {
  // Input:
  // - Participating dim 0 (size 16) sharded by x=4 -> divisible.
  // - Pass-through dim 1 (size 3) sharded by y=2 -> padded to 4.
  // Output:
  // - Participating dim 0 (size 4) sharded by x=4 -> divisible.
  // - Participating dim 1 (size 4) not sharded -> divisible.
  // - Pass-through dim 2 (size 3) sharded by y=2 -> padded to 4.
  //
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<16x3xf32>, tensor<f32>) -> tensor<16x4xf32>
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{"x"}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<16x4xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}, {"y"}]>]>} : (tensor<16x4xf32>) -> tensor<4x4x4xf32>
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<4x4x4xf32> to tensor<4x4x3xf32>
  // CHECK: return %[[CAST]] : tensor<4x4x3xf32>
  %0 = sdy.all_slice [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<16x3xf32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}, {"y"}]>]>} : (tensor<16x3xf32>) -> tensor<4x4x3xf32>
  return %1 : tensor<4x4x3xf32>
}

// -----

sdy.mesh @mesh_4_2_3 = <["x"=4, "y"=2, "z"=3]>

// CHECK-LABEL: func @reshape_mix_participating_middle
func.func @reshape_mix_participating_middle(%arg0: tensor<3x16x7xf32>) -> tensor<3x4x4x7xf32> {
  // Input:
  // - Pass-through dim 0 (size 3) sharded by y=2 -> padded to 4.
  // - Participating dim 1 (size 16) sharded by x=4 -> divisible.
  // - Pass-through dim 2 (size 7) sharded by z=3 -> padded to 9.
  // Output:
  // - Pass-through dim 0 (size 3) sharded by y=2 -> padded to 4.
  // - Participating dim 1 (size 4) sharded by x=4 -> divisible.
  // - Participating dim 2 (size 4) not sharded -> divisible.
  // - Pass-through dim 3 (size 7) sharded by z=3 -> padded to 9.
  //
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0, 0], high = [1, 0, 2], interior = [0, 0, 0] : (tensor<3x16x7xf32>, tensor<f32>) -> tensor<4x16x9xf32>
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{"y"}, {"x"}, {"z"}] %[[PAD]] out_sharding=<@mesh_4_2_3, [{"y"}, {"x"}, {"z"}]> : tensor<4x16x9xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2_3, [{"y"}, {"x"}, {}, {"z"}]>]>} : (tensor<4x16x9xf32>) -> tensor<4x4x4x9xf32>
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<4x4x4x9xf32> to tensor<3x4x4x7xf32>
  // CHECK: return %[[CAST]] : tensor<3x4x4x7xf32>
  %0 = sdy.all_slice [{"y"}, {"x"}, {"z"}] %arg0 out_sharding=<@mesh_4_2_3, [{"y"}, {"x"}, {"z"}]> : tensor<3x16x7xf32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2_3, [{"y"}, {"x"}, {}, {"z"}]>]>} : (tensor<3x16x7xf32>) -> tensor<3x4x4x7xf32>
  return %1 : tensor<3x4x4x7xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

func.func @reshape_split_indivisible_participating(%arg0: tensor<14xf32>) -> tensor<2x7xf32> {
  %0 = sdy.all_slice [{"x"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}]> : tensor<14xf32>
  // expected-error @+2 {{participating reshape dimensions are not divisible. Reshape sharding should have been resolved by resolve-permutation-factors.}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.reshape'}}
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {"x"}]>]>} : (tensor<14xf32>) -> tensor<2x7xf32>
  return %1 : tensor<2x7xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

func.func @reshape_combine_indivisible_participating(%arg0: tensor<2x7xf32>) -> tensor<14xf32> {
  %0 = sdy.all_slice [{"y"}, {"x"}] %arg0 out_sharding=<@mesh_4_2, [{"y"}, {"x"}]> : tensor<2x7xf32>
  // expected-error @+2 {{participating reshape dimensions are not divisible. Reshape sharding should have been resolved by resolve-permutation-factors.}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.reshape'}}
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} : (tensor<2x7xf32>) -> tensor<14xf32>
  return %1 : tensor<14xf32>
}
