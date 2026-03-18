// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// ([o, k, l, m], [i, j, p]) -> ([i, j, k, l, n])
// reduction={m, o} need_replication={k, n, p} blocked_propagation={k}
//
// CHECK-LABEL: func @no_shard_on_operand(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x2x5xf32>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x3x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
// CHECK-SAME: -> (tensor<1x3x2x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}, {}]>}) {
func.func @no_shard_on_operand(%arg0: tensor<3x4x2x5xf32>,
  %arg1: tensor<2x3x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
  -> (tensor<2x3x2x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}, {}]>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{dimension_numbers = #stablehlo.gather<
  // CHECK-SAME: offset_dims = [2, 3, 4],
  // CHECK-SAME: collapsed_slice_dims = [0],
  // CHECK-SAME: start_index_map = [1, 0, 3], index_vector_dim = 2>,
  // CHECK-SAME: indices_are_sorted = false, slice_sizes = array<i64: 1, 2, 2, 1>}>
  // CHECK-SAME: (tensor<3x4x2x5xf32>, tensor<1x3x3xi64>) -> tensor<1x3x2x2x1xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3, 4], collapsed_slice_dims = [0],
      start_index_map = [1, 0, 3], index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2, 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}, {}]>]>
  } : (tensor<3x4x2x5xf32>, tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32>
  // CHECK: return %[[GATHER]] : tensor<1x3x2x2x1xf32>
  return %0 : tensor<2x3x2x2x1xf32>
}

// ([i, m], [i, j]) -> ([i, j]) reduction={m}
//
// CHECK-LABEL: func @shard_batch_dim_on_operand(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<4x5xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<4x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
func.func @shard_batch_dim_on_operand(%arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<8x5xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<8x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1],
  // CHECK-SAME: operand_batching_dims = [0],
  // CHECK-SAME: start_indices_batching_dims = [0],
  // CHECK-SAME: start_index_map = [1], index_vector_dim = 2>,
  // CHECK-SAME: indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}>
  // CHECK-SAME: (tensor<4x10xf32>, tensor<4x5xi64>) -> tensor<4x5xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [1],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>
  } : (tensor<8x10xf32>, tensor<8x5xi64>) -> tensor<8x5xf32>
  // CHECK: return %[[GATHER]] : tensor<4x5xf32>
  return %0 : tensor<8x5xf32>
}

// ([m, n], [i, j]) -> ([i, j, n]) reduction={m}
//
// CHECK-LABEL: func @shard_indexed_dim_on_operand_single_indexed_dim_adjustment_size_1_vector(
// CHECK-SAME: %[[ARG0:arg[0-9]+]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}]>},
// CHECK-SAME: %[[ARG1:arg[0-9]+]]: tensor<2x3x1xi64>)
// CHECK-SAME:  -> (tensor<2x3x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}]>}) {
func.func @shard_indexed_dim_on_operand_single_indexed_dim_adjustment_size_1_vector(
  %arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}]>},
  %arg1: tensor<2x3x1xi64>)
 -> (tensor<2x3x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}]>}) {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 4, 0, 4, 0, 4, 0, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast %[[OFFSET]], sizes = [2, 3, 1] : (tensor<i64>) -> tensor<2x3x1xi64>
  // CHECK: %[[ADJ_INDICES:.*]] = stablehlo.subtract %[[ARG1]], %[[BCAST]] : tensor<2x3x1xi64>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ADJ_INDICES]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>
  // CHECK-SAME: slice_sizes = array<i64: 0, 10>
  // CHECK-SAME: (tensor<4x10xf32>, tensor<2x3x1xi64>) -> tensor<2x3x10xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 10>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {}]>]>
  } : (tensor<8x10xf32>, tensor<2x3x1xi64>) -> tensor<2x3x10xf32>
  // CHECK:    return %[[GATHER]] : tensor<2x3x10xf32>
  return %0 : tensor<2x3x10xf32>
}

// ([m, j], [i]) -> ([i, j]) reduction={m}
//
// CHECK-LABEL: func @shard_indexed_dim_on_operand_single_indexed_dim_adjustment_index_vector_dim_equal_rank(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2xi64>) -> tensor<2x4xf32> {
func.func @shard_indexed_dim_on_operand_single_indexed_dim_adjustment_index_vector_dim_equal_rank(
  %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x4xf32> {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET_CVT:.*]] = stablehlo.convert %[[OFFSET]] : tensor<i64>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast %[[OFFSET_CVT]], sizes = [2] : (tensor<i64>) -> tensor<2xi64>
  // CHECK: %[[ADJ_INDICES:.*]] = stablehlo.subtract %[[ARG1]], %[[BCAST]] : tensor<2xi64>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ADJ_INDICES]])
  // CHECK-SAME: <{dimension_numbers = #stablehlo.gather<
  // CHECK-SAME: offset_dims = [1],
  // CHECK-SAME: collapsed_slice_dims = [0],
  // CHECK-SAME: start_index_map = [0], index_vector_dim = 1>,
  // CHECK-SAME: indices_are_sorted = false,
  // CHECK-SAME: slice_sizes = array<i64: 0, 4>}>
  // CHECK-SAME: (tensor<2x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}]>]>
  } : (tensor<4x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  // CHECK: %[[RES:.*]] = sdy.all_reduce {"x"} %[[GATHER]] out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// ([o, k, l, m], [i, j, p]) -> ([i, j, k, l, n])
// reduction={m, o} need_replication={k, n, p} blocked_propagation={k}
//
// CHECK-LABEL: func.func @shard_indexed_dim_on_operand_multiple_indexed_dim_adjustment(
// CHECK-SAME: %[[ARG0:arg[0-9]+]]: tensor<3x2x2x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}, {}]>},
// CHECK-SAME: %[[ARG1:arg[0-9]+]]: tensor<2x3x3xi64>)
// CHECK-SAME: -> (tensor<2x3x1x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"x"}, {}, {}]>}) {
func.func @shard_indexed_dim_on_operand_multiple_indexed_dim_adjustment(
  %arg0: tensor<3x4x2x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}, {}]>},
  %arg1: tensor<2x3x3xi64>)
  -> (tensor<2x3x2x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"x"}, {}, {}]>}) {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFFSET_RESHAPE:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[OFFSET_RESHAPE]] : tensor<i64>
  // CHECK: %[[INDICES_SLICE:.*]] = stablehlo.slice %[[ARG1]] [0:2, 0:3, 0:1] : (tensor<2x3x3xi64>) -> tensor<2x3x1xi64>
  // CHECK: %[[BCAST_OFFSET:.*]] = stablehlo.broadcast %[[OFFSET]], sizes = [2, 3, 1] : (tensor<i64>) -> tensor<2x3x1xi64>
  // CHECK: %[[ADJ_SLICE:.*]] = stablehlo.subtract %[[INDICES_SLICE]], %[[BCAST_OFFSET]] : tensor<2x3x1xi64>
  // CHECK: %[[ADJ_INDICES:.*]] = stablehlo.dynamic_update_slice %[[ARG1]], %[[ADJ_SLICE]], %{{.*}}, %{{.*}}, %{{.*}} : (tensor<2x3x3xi64>, tensor<2x3x1xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x3x3xi64>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ADJ_INDICES]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [1, 0, 3], index_vector_dim = 2>
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 2, 1>
  // CHECK-SAME: (tensor<3x2x2x5xf32>, tensor<2x3x3xi64>) -> tensor<2x3x1x2x1xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3, 4], collapsed_slice_dims = [0],
      start_index_map = [1, 0, 3], index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2, 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {"x"}, {}, {}]>]>
  } : (tensor<3x4x2x5xf32>, tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32>
  // CHECK: return %[[GATHER]] : tensor<2x3x1x2x1xf32>
  return %0 : tensor<2x3x2x2x1xf32>
}

// ([i, k], [i, j]) -> ([i, j, k])
//
// CHECK-LABEL: func @shard_indexed_dim_and_batch_dim_on_operand(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {"x"}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x5xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
// CHECK-SAME: -> (tensor<2x5x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}, {"x"}]>}) {
func.func @shard_indexed_dim_and_batch_dim_on_operand(%arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {"x"}]>},
 %arg1: tensor<8x5xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
 -> (tensor<8x5x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}, {"x"}]>}) {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 5, 5, 5, 5]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFFSET_RAW:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[OFFSET_RAW]] : tensor<i64>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast %[[OFFSET]], sizes = [2, 5] : (tensor<i64>) -> tensor<2x5xi64>
  // CHECK: %[[ADJ_INDICES:.*]] = stablehlo.subtract %[[ARG1]], %[[BCAST]] : tensor<2x5xi64>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ADJ_INDICES]])
  // CHECK-SAME: <{dimension_numbers = #stablehlo.gather<
  // CHECK-SAME: offset_dims = [2],
  // CHECK-SAME: operand_batching_dims = [0],
  // CHECK-SAME: start_indices_batching_dims = [0],
  // CHECK-SAME: start_index_map = [1], index_vector_dim = 2>,
  // CHECK-SAME: indices_are_sorted = false,
  // CHECK-SAME: slice_sizes = array<i64: 1, 1>}>
  // CHECK-SAME: (tensor<2x5xf32>, tensor<2x5xi64>) -> tensor<2x5x1xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [1],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"y"}, {}, {"x"}]>]>
  } : (tensor<8x10xf32>, tensor<8x5xi64>) -> tensor<8x5x2xf32>
  // CHECK: return %[[GATHER]] : tensor<2x5x1xf32>
  return %0 : tensor<8x5x2xf32>
}
