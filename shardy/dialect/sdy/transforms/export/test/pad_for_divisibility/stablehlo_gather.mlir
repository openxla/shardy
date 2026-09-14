// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @padded_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x4xf32>, %[[ARG1:.*]]: tensor<2x1xi64>) -> tensor<2x4xf32>
func.func @padded_gather(%arg0: tensor<3x4xf32>, %arg1: tensor<2x1xi64>) -> tensor<2x4xf32> {
  // Pad operand dimension 0 with zero from 3 to 4.
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x4xf32>

  // Clamp start indices to max valid index (3 - 1 = 2) for padded dimension.
  // CHECK: %[[CST_IDX:.*]] = stablehlo.constant dense<2> : tensor<1xi64>
  // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_IDX]], dims = [1] : (tensor<1xi64>) -> tensor<2x1xi64>
  // CHECK: %[[CLAMPED_IDX:.*]] = stablehlo.minimum %[[ARG1]], %[[BROADCAST]] : tensor<2x1xi64>

  // Perform gather with padded operand and clamped start indices.
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[SLICE]], %[[CLAMPED_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 4>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>}
  // CHECK-SAME: : (tensor<4x4xf32>, tensor<2x1xi64>) -> tensor<2x4xf32>
  // CHECK: return %[[GATHER]] : tensor<2x4xf32>

  %sliced_operand = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<3x4xf32>
  %0 = "stablehlo.gather"(%sliced_operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>
  } : (tensor<3x4xf32>, tensor<2x1xi64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
