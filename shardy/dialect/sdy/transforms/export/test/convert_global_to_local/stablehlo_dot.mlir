// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

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
