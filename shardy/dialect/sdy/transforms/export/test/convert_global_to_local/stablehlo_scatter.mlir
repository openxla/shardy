// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// This test checks the logic to localize a scatter as if it is a generic op.
//
// ([k, l, m], [i, j, n], [i, j, o]) -> ([k, l, m])
// reductioin={i, j} need_replication= {n, o}
//
// CHECK-LABEL: func @input_not_sharded_scatter_indices_update_sharded_on_implicit_batch_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x2xf32>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x3x2xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
// CHECK-SAME: %[[ARG2:.*]]: tensor<1x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) -> tensor<3x4x2xf32> {
func.func @input_not_sharded_scatter_indices_update_sharded_on_implicit_batch_dim(
    %arg0: tensor<3x4x2xf32>,
    %arg1: tensor<2x3x2xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
    %arg2: tensor<2x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
 -> tensor<3x4x2xf32> {
  // CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
  // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>
  // CHECK: (tensor<3x4x2xf32>, tensor<1x3x2xi64>, tensor<1x3x1xf32>) -> tensor<3x4x2xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2
    >
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>, tensor<2x3x1xf32>) -> tensor<3x4x2xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SCATTER]])
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>
  // CHECK: (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}, {}]> : tensor<3x4x2xf32>

  // CHECK: return %[[RES]] : tensor<3x4x2xf32>
  return %1 : tensor<3x4x2xf32>
}

// This test checks that even though operand is sharded, the op is also
// converted like a generic op.
//
// ([k, l, m], [i, j, n], [i, j, o]) -> ([k, l, m])
// reductioin={i, j} need_replication= {n, o}
//
// CHECK-LABEL: func @input_sharded_not_on_indexed_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x3x2xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
// CHECK-SAME: %[[ARG2:.*]]: tensor<1x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
// CHECK-SAME: -> (tensor<3x4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>}) {
func.func @input_sharded_not_on_indexed_dim(
    %arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>},
    %arg1: tensor<2x3x2xi64> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
    %arg2: tensor<2x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
 -> (tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>}) {
  // CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
  // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>
  // CHECK: (tensor<3x4x1xf32>, tensor<1x3x2xi64>, tensor<1x3x1xf32>) -> tensor<3x4x1xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>]>
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>, tensor<2x3x1xf32>) -> tensor<3x4x2xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[SCATTER]])
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>
  // CHECK: (tensor<3x4x1xf32>) -> tensor<3x4x1xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}, {"y":(2)2}]> : tensor<3x4x2xf32>

  // CHECK: return %[[RES]] : tensor<3x4x1xf32>
  return %1 : tensor<3x4x2xf32>
}

// This test checks the scatter_indice adjustment logic when the operand is
// sharded on the inserted window dim.
//
// ([k, l, m], [i, j, n], [i, j, o]) -> ([k, l, m])
// reductioin={i, j} need_replication= {n, o}
//
// CHECK-LABEL: func @input_sharded_on_indexed_inserted__window_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x3x2xi64>,
// CHECK-SAME: %[[ARG2:.*]]: tensor<2x3x1xf32>)
// CHECK-SAME: -> (tensor<3x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>}) {
func.func @input_sharded_on_indexed_inserted__window_dim(
    %arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>},
    %arg1: tensor<2x3x2xi64>,
    %arg2: tensor<2x3x1xf32>)
 -> (tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>}) {
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPE_OFF:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE_OFF]] : tensor<i64>
  // CHECK: %[[OFF0:.*]] = stablehlo.reshape %[[C0]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[OFFSET]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate %[[OFF0]], %[[OFF1]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  // CHECK: %[[OFF_VEC:.*]] = stablehlo.broadcast_in_dim %[[CONCAT]], dims = [2] : (tensor<2xi64>) -> tensor<2x3x2xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[ARG1]], %[[OFF_VEC]] : tensor<2x3x2xi64>
  // CHECK: %[[RES:.*]] = "stablehlo.scatter"(%[[ARG0]], %[[LOCAL_IDX]], %[[ARG2]])
  // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>
  // CHECK: (tensor<3x2x2xf32>, tensor<2x3x2xi64>, tensor<2x3x1xf32>) -> tensor<3x2x2xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {"x"}, {}]>]>
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>, tensor<2x3x1xf32>) -> tensor<3x4x2xf32>

  // CHECK: return %[[RES]] : tensor<3x2x2xf32>
  return %0 : tensor<3x4x2xf32>
}

// This test checks the scatter_indice adjustment logic when the operand is
// sharded on the non-inserted window dim.
//
// ([k, l, m], [i, j, n], [i, j, o]) -> ([k, l, m])
// reductioin={i, j} need_replication= {n, o}
//
// CHECK-LABEL: func @input_sharded_on_indexed_but_non_inserted_window_dim(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x3x3xi64>,
// CHECK-SAME: %[[ARG2:.*]]: tensor<2x3x1xf32>)
// CHECK-SAME: -> (tensor<3x4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>}) {
func.func @input_sharded_on_indexed_but_non_inserted_window_dim(
    %arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>},
    %arg1: tensor<2x3x3xi64>,
    %arg2: tensor<2x3x1xf32>)
 -> (tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>}) {

  // Base offset logic for unsharded dimensions 0 and 1
  // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-DAG: %[[C0_1:.*]] = stablehlo.constant dense<0> : tensor<i64>

  // Shard offset calculation logic for sharded dimension 2 (axis "y:(2)2")
  // CHECK-DAG: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-DAG: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPE_OFF:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE_OFF]] : tensor<i64>

  // Index vector construction for [dim0_offset, dim1_offset, dim2_offset]
  // CHECK: %[[OFF0:.*]] = stablehlo.reshape %[[C0]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[C0_1]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF2:.*]] = stablehlo.reshape %[[OFFSET]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate %[[OFF0]], %[[OFF1]], %[[OFF2]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  // CHECK: %[[OFF_VEC:.*]] = stablehlo.broadcast_in_dim %[[CONCAT]], dims = [2] : (tensor<3xi64>) -> tensor<2x3x3xi64>

  // Coordinate shift and local scatter execution
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[ARG1]], %[[OFF_VEC]] : tensor<2x3x3xi64>
  // CHECK: %[[RES:.*]] = "stablehlo.scatter"(%[[ARG0]], %[[LOCAL_IDX]], %[[ARG2]])
  // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 2>
  // CHECK: (tensor<3x4x1xf32>, tensor<2x3x3xi64>, tensor<2x3x1xf32>) -> tensor<3x4x1xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1, 2],
      index_vector_dim = 2
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {"y":(2)2}]>]>
  } : (tensor<3x4x2xf32>, tensor<2x3x3xi64>, tensor<2x3x1xf32>) -> tensor<3x4x2xf32>

  // CHECK: return %[[RES]] : tensor<3x4x1xf32>
  return %0 : tensor<3x4x2xf32>
}

// Sharding Rule:
// ([k], [], []) -> ([k])
//
// CHECK-LABEL: func @shard_indexd_dim_scalar_scatter_indices(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<i64>,
// CHECK-SAME: %[[ARG2:.*]]: tensor<i32>)
// CHECK-SAME: -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>}) {
func.func @shard_indexd_dim_scalar_scatter_indices(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
  %arg1: tensor<i64>,
  %arg2: tensor<i32>)
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>}) {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 4, 0, 4, 0, 4, 0, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[OFF_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] : (tensor<i64>) -> tensor<i64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[ARG1]], %[[OFF_BCAST]] : tensor<i64>
  // CHECK: %[[RES:.*]] = "stablehlo.scatter"(%[[ARG0]], %[[LOCAL_IDX]], %[[ARG2]])
  // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>
  // CHECK: (tensor<4xi32>, tensor<i64>, tensor<i32>) -> tensor<4xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %1 = stablehlo.and %arg3, %arg4 : tensor<i32>
    stablehlo.return %1 : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"y":(1)2}]>]>
  } : (tensor<8xi32>, tensor<i64>, tensor<i32>) -> tensor<8xi32>

  // CHECK: return %[[RES]] : tensor<4xi32>
  return %0 : tensor<8xi32>
}

// Sharding Rule:
// ([k], [], []) -> ([k])
//
// Note that I have to use %arg0 in this test instead of using variable name
// matching %[[ARG0:.*]], because %arg0 and %arg1 have the same type and confuse
// FileCheck, and causes it to match ARGO all the way to the ":" right after
// %arg1.
//
// CHECK-LABEL: func.func @shard_indexd_dim_scalar_scatter_indices_variadic(
// CHECK-SAME: %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
// CHECK-SAME: %[[ARG2:.*]]: tensor<i64>,
// CHECK-SAME: %[[ARG3:.*]]: tensor<i32>,
// CHECK-SAME: %[[ARG4:.*]]: tensor<i32>)
// CHECK-SAME: -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
// CHECK-SAME: {{.*}}tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>}) {
func.func @shard_indexd_dim_scalar_scatter_indices_variadic(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
  %arg1: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>},
  %arg2: tensor<i64>,
  %arg3: tensor<i32>,
  %arg4: tensor<i32>)
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>}, tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>}) {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT_PID:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 4, 0, 4, 0, 4, 0, 4]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT_PID]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i64>
  // CHECK: %[[OFF_BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] : (tensor<i64>) -> tensor<i64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[ARG2]], %[[OFF_BCAST]] : tensor<i64>
  // CHECK: %[[RES:.*]]:2 = "stablehlo.scatter"(%arg0, %[[ARG1]], %[[LOCAL_IDX]], %[[ARG3]], %[[ARG4]])
  // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>
  // CHECK: (tensor<4xi32>, tensor<4xi32>, tensor<i64>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<4xi32>)
  %0:2 = "stablehlo.scatter"(%arg0, %arg1, %arg2, %arg3, %arg4) ({
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<i32>, %arg8: tensor<i32>):
    %1 = stablehlo.and %arg5, %arg6 : tensor<i32>
    %2 = stablehlo.and %arg7, %arg8 : tensor<i32>
    stablehlo.return %1, %2 : tensor<i32>, tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"y":(1)2}]>, #sdy.sharding<@mesh_2_4, [{"y":(1)2}]>]>
  } : (tensor<8xi32>, tensor<8xi32>, tensor<i64>, tensor<i32>, tensor<i32>) -> (tensor<8xi32>, tensor<8xi32>)

  // CHECK: return %[[RES]]#0, %[[RES]]#1 : tensor<4xi32>, tensor<4xi32>
  return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
}
