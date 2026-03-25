// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// This test checks the logic to localize a gather as if it is a generic op.
//
// ([o, k, l, m], [i, j, p]) -> ([i, j, k, l, n])
// reduction={m, o} need_replication={k, n, p} blocked_propagation={k}
//
// CHECK-LABEL: func @operand_replicated
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x2x5xf32>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x3x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
// CHECK-SAME: -> (tensor<1x3x2x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}, {}]>})
func.func @operand_replicated(%arg0: tensor<3x4x2x5xf32>,
  %arg1: tensor<2x3x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
  -> (tensor<2x3x2x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}, {}]>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [1, 0, 3], index_vector_dim = 2>,
  // CHECK-SAME: slice_sizes = array<i64: 1, 2, 2, 1>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}, {}, {}]>]>}
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

// This test checks the logic to convert a gather without trivial slice dims.
//
// ([o, k, l, m], [i, j, p]) -> ([i, j, k, l, n])
// reduction={m, o} need_replication={k, n, p} blocked_propagation={k}
//
// CHECK-LABEL: func @operand_sharded_pass_through_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x1x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x3x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
// CHECK-SAME: -> (tensor<1x3x2x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {"y":(2)2}, {}]>})
func.func @operand_sharded_pass_through_dim(
  %arg0: tensor<3x4x2x5xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}, {}]>},
  %arg1: tensor<2x3x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
  -> (tensor<2x3x2x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {"y":(2)2}, {}]>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [1, 0, 3], index_vector_dim = 2>,
  // CHECK-SAME: slice_sizes = array<i64: 1, 2, 1, 1>
  // CHECK-SAME: (tensor<3x4x1x5xf32>, tensor<1x3x3xi64>) -> tensor<1x3x2x1x1xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3, 4], collapsed_slice_dims = [0],
      start_index_map = [1, 0, 3], index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2, 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {"y":(2)2}, {}]>]>
  } : (tensor<3x4x2x5xf32>, tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32>
  // CHECK: return %[[GATHER]] : tensor<1x3x2x1x1xf32>
  return %0 : tensor<2x3x2x2x1xf32>
}

// ([i, j], [k]) -> ([k, j]) reduction={i}
//
// CHECK-LABEL: func @shard_reduction_dim_is_collapsed(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2xi64>) -> tensor<2x10xf32> {
func.func @shard_reduction_dim_is_collapsed(
  %arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x10xf32> {
  // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<2xi64>
  // CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<2xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<2xi64>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1]
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i64>
  // CHECK: %[[LIMIT:.*]] = stablehlo.add %[[OFFSET]], %[[C3]] : tensor<i64>
  // CHECK: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] : (tensor<i64>) -> tensor<2xi64>
  // CHECK: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] : (tensor<i64>) -> tensor<2xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<2xi64>
  // CHECK: %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]]
  // CHECK: %[[LE:.*]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]]
  // CHECK: %[[MASK:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<2xi1>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 10>
  // CHECK: %[[MASK_BCAST:.*]] = stablehlo.broadcast_in_dim %[[MASK]], dims = [0] : (tensor<2xi1>) -> tensor<2x10xi1>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x10xf32>
  // CHECK: %[[SEL:.*]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %[[ZERO]] : tensor<2x10xi1>, tensor<2x10xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 10>
  } : (tensor<8x10xf32>, tensor<2xi64>) -> tensor<2x10xf32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x10xf32>
  // CHECK: return %[[RES]] : tensor<2x10xf32>
  return %1 : tensor<2x10xf32>
}

// ([m, n], [i]) -> ([i, n]) reduction={m}
//
// CHECK-LABEL: func @shard_reduction_dim_is_collapsed_not_in_start_index_map(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2xi64>) -> tensor<2x1xf32> {
func.func @shard_reduction_dim_is_collapsed_not_in_start_index_map(
  %arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x1xf32> {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1]
  // CHECK: %[[OFFSET:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[EQ_ZERO:.*]] = stablehlo.compare EQ, %[[OFFSET]], %[[ZERO]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [1], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 1>
  // CHECK: %[[MASK_BCAST:.*]] = stablehlo.broadcast_in_dim %[[EQ_ZERO]], dims = [] : (tensor<i1>) -> tensor<2x1xi1>
  // CHECK: %[[ZERO_F32:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1xf32>
  // CHECK: %[[SEL:.*]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %[[ZERO_F32]] : tensor<2x1xi1>, tensor<2x1xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>
  } : (tensor<8x10xf32>, tensor<2xi64>) -> tensor<2x1xf32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x1xf32>
  // CHECK: return %[[RES]] : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

// ([m, n], [i, p]) -> ([i, n]) reduction={m}, need_replication={p}
//
// CHECK-LABEL: func @shard_reduction_dim_is_collapsed_explicit_scalar_index_vector_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x1xi64>) -> tensor<2x10xf32> {
func.func @shard_reduction_dim_is_collapsed_explicit_scalar_index_vector_dim(
  %arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<2x1xi64>) -> tensor<2x10xf32> {
  // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<2x1xi64>
  // CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<2x1xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<2x1xi64>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1]
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i64>
  // CHECK: %[[LIMIT:.*]] = stablehlo.add %[[OFFSET]], %[[C3]] : tensor<i64>
  // CHECK: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] : (tensor<i64>) -> tensor<2x1xi64>
  // CHECK: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] : (tensor<i64>) -> tensor<2x1xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<2x1xi64>
  // CHECK: %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]]
  // CHECK: %[[LE:.*]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]]
  // CHECK: %[[MASK:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<2x1xi1>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 10>
  // CHECK: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK: %[[RED_MASK:.*]] = stablehlo.reduce(%[[MASK]] init: %[[TRUE]])
  // CHECK-SAME: applies stablehlo.and across dimensions = [1]
  // CHECK: %[[MASK_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RED_MASK]], dims = [0] : (tensor<2xi1>) -> tensor<2x10xi1>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x10xf32>
  // CHECK: %[[SEL:.*]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %[[ZERO]] : tensor<2x10xi1>, tensor<2x10xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 10>
  } : (tensor<8x10xf32>, tensor<2x1xi64>) -> tensor<2x10xf32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<2x10xf32>
  // CHECK: return %[[RES]] : tensor<2x10xf32>
  return %1 : tensor<2x10xf32>
}

// ([m, n], [i]) -> ([i, n]) reduction={m}
//
// CHECK-LABEL: func @shard_reduction_dim_not_collapsed(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2xi64>) -> tensor<2x1x10xf32> {
func.func @shard_reduction_dim_not_collapsed(
  %arg0: tensor<8x10xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x1x10xf32> {
  // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<2xi64>
  // CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<2xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<2xi64>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1]
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i64>
  // CHECK: %[[LIMIT:.*]] = stablehlo.add %[[OFFSET]], %[[C3]] : tensor<i64>
  // CHECK: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] : (tensor<i64>) -> tensor<2xi64>
  // CHECK: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] : (tensor<i64>) -> tensor<2xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<2xi64>
  // CHECK: %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]]
  // CHECK: %[[LE:.*]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]]
  // CHECK: %[[MASK:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<2xi1>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>
  // CHECK-SAME: slice_sizes = array<i64: 1, 10>
  // CHECK: %[[MASK_BCAST:.*]] = stablehlo.broadcast_in_dim %[[MASK]], dims = [0] : (tensor<2xi1>) -> tensor<2x1x10xi1>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1x10xf32>
  // CHECK: %[[SEL:.*]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %[[ZERO]] : tensor<2x1x10xi1>, tensor<2x1x10xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 10>
  } : (tensor<8x10xf32>, tensor<2xi64>) -> tensor<2x1x10xf32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}, {}]> : tensor<2x1x10xf32>
  // CHECK: return %[[RES]] : tensor<2x1x10xf32>
  return %1 : tensor<2x1x10xf32>
}

// CHECK-LABEL: func @shard_reduction_dim_explicit_scalar_indices(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<i64>) -> tensor<f32> {
func.func @shard_reduction_dim_explicit_scalar_indices(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>},
  %arg1: tensor<i64>) -> tensor<f32> {
  // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<i64>
  // CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<i64>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1]
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] : (tensor<i64>) -> tensor<i64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMP]], %[[BCAST_OFF]] : tensor<i64>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>
  // CHECK: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[GATHER]], %{{.*}} : tensor<i1>, tensor<f32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 0>,
    slice_sizes = array<i64: 1>
  } : (tensor<8xf32>, tensor<i64>) -> tensor<f32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, []> : tensor<f32>
  // CHECK: return %[[RES]] : tensor<f32>
  return %1 : tensor<f32>
}

// ([r1, b, r2, l, r3], [b, i, j]) -> ([b, i, r1(1), r2(1), l, r3(1)])
//
// CHECK-LABEL: func @shard_two_of_three_reduction_dims(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x3x2x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x2x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}]>})
// CHECK-SAME: -> (tensor<3x2x1x1x5x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]>}) {
func.func @shard_two_of_three_reduction_dims(
  %arg0: tensor<8x6x4x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}, {}]>},
  %arg1: tensor<6x2x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}]>})
  -> (tensor<6x2x1x1x5x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]>}) {
  // CHECK-DAG: %[[C0_IDX:.*]] = stablehlo.constant dense<0> : tensor<3x2x3xi64>
  // CHECK-DAG: %[[LIMITS:.*]] = stablehlo.constant dense<[7, 3, 2]> : tensor<3xi64>
  // CHECK: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMITS]], dims = [2]
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0_IDX]], %[[ARG1]], %[[BCAST_LIM]]

  // Offset r1 logic ("x" axis)
  // CHECK: %[[PID:.*]] = stablehlo.partition_id
  // CHECK: %[[OFF_R1_TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[OFF_R1:.*]] = stablehlo.convert %{{.*}} : tensor<i64>

  // Offset r2 logic ("y:(2)2" axis)
  // CHECK: %[[OFF_R2_TABLE:.*]] = stablehlo.constant dense<[0, 0, 2, 2, 0, 0, 2, 2]> : tensor<8xi64>
  // CHECK: %[[OFF_R2:.*]] = stablehlo.convert %{{.*}} : tensor<i64>

  // Local Coordinate Shift
  // CHECK: %[[OFF_CONCAT:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, %{{.*}}, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %{{.*}} : tensor<3x2x3xi64>

  // Local Gather
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: operand_batching_dims = [1], start_indices_batching_dims = [0], start_index_map = [0, 2, 4], index_vector_dim = 2
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 1, 5, 1>

  // Masking & Filtering
  // CHECK: %[[MASK_RED:.*]] = stablehlo.reduce(%{{.*}}) {{.*}} across dimensions = [2]
  // CHECK: %[[MASK_BCAST:.*]] = stablehlo.broadcast_in_dim %[[MASK_RED]], dims = [0, 1]
  // CHECK: %[[SELECTED:.*]] = stablehlo.select %[[MASK_BCAST]], %[[GATHER]], %{{.*}} : tensor<3x2x1x1x5x1xi1>, tensor<3x2x1x1x5x1xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3, 4, 5],
      operand_batching_dims = [1],
      start_indices_batching_dims = [0],
      start_index_map = [0, 2, 4],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 1, 5, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]>]>
  } : (tensor<8x6x4x5x3xf32>, tensor<6x2x3xi64>) -> tensor<6x2x1x1x5x1xf32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SELECTED]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 4, 5], [2, 3, 6, 7]]>
  %1 = sdy.all_reduce {"x", "y":(2)2} %0 out_sharding=<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]> : tensor<6x2x1x1x5x1xf32>
  // CHECK: return %[[RES]] : tensor<3x2x1x1x5x1xf32>
  return %1 : tensor<6x2x1x1x5x1xf32>
}

// ([r1, b, r2, l, r3], [b, i, j]) -> ([b, i, r1(1), r2(1), l, r3(1)])
//
// CHECK-LABEL: func @shard_two_of_three_reduction_dims_one_not_in_start_index_map(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x3x2x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x2x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}]>})
// CHECK-SAME: -> (tensor<3x2x1x1x5x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]>}) {
func.func @shard_two_of_three_reduction_dims_one_not_in_start_index_map(
  %arg0: tensor<8x6x4x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}, {}]>},
  %arg1: tensor<6x2x3xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}]>})
  -> (tensor<6x2x1x1x5x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]>}) {

  // Global index clamping
  // CHECK-DAG: %[[C0_IDX:.*]] = stablehlo.constant dense<0> : tensor<3x2x3xi64>
  // CHECK-DAG: %[[LIMITS:.*]] = stablehlo.constant dense<[7, 4, 2]> : tensor<3xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0_IDX]], %[[ARG1]], %{{.*}} : tensor<3x2x3xi64>

  // Indexed sharded dim 0 ("x" axis) logic
  // CHECK: %[[PID0:.*]] = stablehlo.partition_id
  // CHECK: %[[OFF_R1_TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 4, 4, 4, 4]> : tensor<8xi64>
  // CHECK: %[[OFF_R1:.*]] = stablehlo.convert %{{.*}} : tensor<i64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %{{.*}} : tensor<3x2x3xi64>
  // CHECK: %[[IDX_MASK:.*]] = stablehlo.and %{{.*}}, %{{.*}} : tensor<3x2x3xi1>

  // Unindexed sharded dim 2 ("y:(2)2" axis) logic
  // CHECK: %[[PID2:.*]] = stablehlo.partition_id
  // CHECK: %[[OFF_R2_TABLE:.*]] = stablehlo.constant dense<[0, 0, 2, 2, 0, 0, 2, 2]> : tensor<8xi64>
  // CHECK: %[[OFF_R2:.*]] = stablehlo.reshape %{{.*}} : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[EQ_ZERO:.*]] = stablehlo.compare EQ, %[[OFF_R2]], %{{.*}} : (tensor<i64>, tensor<i64>) -> tensor<i1>

  // Mask combination and local gather
  // CHECK: %[[PART_BCAST:.*]] = stablehlo.broadcast_in_dim %[[EQ_ZERO]], dims = [] : (tensor<i1>) -> tensor<3x2x3xi1>
  // CHECK: %[[COMBINED_MASK:.*]] = stablehlo.and %[[IDX_MASK]], %[[PART_BCAST]] : tensor<3x2x3xi1>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4, 5], operand_batching_dims = [1], start_indices_batching_dims = [0], start_index_map = [0, 3, 4], index_vector_dim = 2>

  // Mask reduction and filtering
  // CHECK: %[[RED_MASK:.*]] = stablehlo.reduce(%[[COMBINED_MASK]] init: %{{.*}}) applies stablehlo.and across dimensions = [2]
  // CHECK: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[GATHER]], %{{.*}}
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3, 4, 5],
      operand_batching_dims = [1],
      start_indices_batching_dims = [0],
      start_index_map = [0, 3, 4],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 1, 5, 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]>]>
  } : (tensor<8x6x4x5x3xf32>, tensor<6x2x3xi64>) -> tensor<6x2x1x1x5x1xf32>
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SEL]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 4, 5], [2, 3, 6, 7]]>
  %1 = sdy.all_reduce {"x", "y":(2)2} %0 out_sharding=<@mesh_2_4, [{"y":(1)2}, {}, {}, {}, {}, {}]> : tensor<6x2x1x1x5x1xf32>
  // CHECK: return %[[RES]] : tensor<3x2x1x1x5x1xf32>
  return %1 : tensor<6x2x1x1x5x1xf32>
}
