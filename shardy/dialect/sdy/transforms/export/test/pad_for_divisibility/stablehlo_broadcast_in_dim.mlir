// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file -verify-diagnostics | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>

// CHECK-LABEL: func.func private @broadcast_in_dim_padded_dim
func.func private @broadcast_in_dim_padded_dim(%arg0: tensor<3x4xf32>)
    -> (tensor<2x3x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}, {}]>}) {
  // Padding LHS input dim 0 (size 3) to 4 for x=4.
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<4x4xf32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: return %[[BCAST]] : tensor<2x4x4xf32>
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<3x4xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}, {}]>]>} : (tensor<3x4xf32>) -> tensor<2x3x4xf32>
  return %1 : tensor<2x3x4xf32>
}

// -----

sdy.mesh @mesh_4 = <["x"=4]>

// CHECK-LABEL: func.func private @broadcast_in_dim_broadcasted_dim_padded
func.func private @broadcast_in_dim_broadcasted_dim_padded(%arg0: tensor<4xf32>)
    -> (tensor<3x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
  // Output dim 0 (broadcasted dim, size 3) is sharded by x=4, so output is padded to 4x4.
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{}] %arg0 out_sharding=<@mesh_4, [{}]> : tensor<4xf32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[BCAST]] : tensor<4x4xf32>
  %0 = sdy.all_slice [{}] %arg0 out_sharding=<@mesh_4, [{}]> : tensor<4xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<4xf32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}
