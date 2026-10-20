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

// CHECK-LABEL: func @padded_gather_indexed_full_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x4xf32>, %[[ARG1:.*]]: tensor<2x1xi64>) -> tensor<2x3x4xf32>
func.func @padded_gather_indexed_full_slice(%arg0: tensor<3x4xf32>, %arg1: tensor<2x1xi64>) -> tensor<2x3x4xf32> {
  // Pad operand dimension 0 with zero from 3 to 4.
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  // CHECK: %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x4xf32>

  // Clamp start indices to max valid index (3 - 3 = 0) for padded dimension.
  // CHECK: %[[CST_IDX:.*]] = stablehlo.constant dense<0> : tensor<1xi64>
  // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_IDX]], dims = [1] : (tensor<1xi64>) -> tensor<2x1xi64>
  // CHECK: %[[CLAMPED_IDX:.*]] = stablehlo.minimum %[[ARG1]], %[[BROADCAST]] : tensor<2x1xi64>

  // Even though slice_sizes[0] == origOperandType.getDimSize(0) (3 == 3),
  // dim 0 is in start_index_map, so slice_sizes[0] remains 3 (not 4).
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[SLICE]], %[[CLAMPED_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 3, 4>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}]>]>}
  // CHECK-SAME: : (tensor<4x4xf32>, tensor<2x1xi64>) -> tensor<2x3x4xf32>
  // CHECK: return %[[GATHER]] : tensor<2x3x4xf32>

  %sliced_operand = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<3x4xf32>
  %0 = "stablehlo.gather"(%sliced_operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 3, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}]>]>
  } : (tensor<3x4xf32>, tensor<2x1xi64>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @padded_gather_batching_dim_size_1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8xf32>, %[[ARG1:.*]]: tensor<1x1xi64>) -> tensor<1x2xf32>
func.func @padded_gather_batching_dim_size_1(%arg0: tensor<1x8xf32>, %arg1: tensor<1x1xi64>) -> tensor<1x2xf32> {
  // Pad batching dimension 0 from 1 to 4 on both operand and start_indices.
  // Even though slice_sizes[0] == origOperandType.getDimSize(0) (1 == 1),
  // dim 0 is in operand_batching_dims, so slice_sizes[0] must remain 1.
  //
  // CHECK: %[[CST_F32:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD_OP:.*]] = stablehlo.pad %[[ARG0]], %[[CST_F32]], low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<1x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK: %[[SLICE_OP:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD_OP]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK: %[[CST_I64:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[PAD_IDX:.*]] = stablehlo.pad %[[ARG1]], %[[CST_I64]], low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<1x1xi64>, tensor<i64>) -> tensor<4x1xi64>
  // CHECK: %[[SLICE_IDX:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD_IDX]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x1xi64>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[SLICE_OP]], %[[SLICE_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 2>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>}
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<4x1xi64>) -> tensor<4x2xf32>
  // CHECK: %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[GATHER]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x2xf32>
  // CHECK: %[[UNPAD:.*]] = stablehlo.slice %[[AG]] [0:1, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<4x2xf32>) -> tensor<1x2xf32>
  // CHECK: return %[[UNPAD]] : tensor<1x2xf32>

  %sliced_operand = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<1x8xf32>
  %sliced_indices = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<1x1xi64>
  %0 = "stablehlo.gather"(%sliced_operand, %sliced_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [1], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>
  } : (tensor<1x8xf32>, tensor<1x1xi64>) -> tensor<1x2xf32>
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// CHECK-LABEL: func @padded_gather_collapsed_dim_size_1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8xf32>, %[[ARG1:.*]]: tensor<2x1xi64>) -> tensor<2x2xf32>
func.func @padded_gather_collapsed_dim_size_1(%arg0: tensor<1x8xf32>, %arg1: tensor<2x1xi64>) -> tensor<2x2xf32> {
  // Pad collapsed dimension 0 from 1 to 4 on operand.
  // Even though slice_sizes[0] == origOperandType.getDimSize(0) (1 == 1) and
  // dim 0 is not in start_index_map, dim 0 is in collapsed_slice_dims, so
  // slice_sizes[0] must remain 1.
  //
  // CHECK: %[[CST_F32:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD_OP:.*]] = stablehlo.pad %[[ARG0]], %[[CST_F32]], low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<1x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK: %[[SLICE_OP:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD_OP]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[SLICE_OP]], %[[ARG1]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [1], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 2>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>}
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
  // CHECK: return %[[GATHER]] : tensor<2x2xf32>

  %sliced_operand = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<1x8xf32>
  %0 = "stablehlo.gather"(%sliced_operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>
  } : (tensor<1x8xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
