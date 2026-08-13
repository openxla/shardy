// RUN: sdy_opt %s -sdy-convert-global-to-local="replica-count=8 partition-count=1" | FileCheck %s

sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func.func @sharded_dense_replica_id
// CHECK-SAME:    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
func.func @sharded_dense_replica_id() -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[GLOBAL_CST:.*]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>
  // CHECK-NEXT: %[[RID:.*]] = stablehlo.replica_id : tensor<ui32>
  // CHECK-NEXT: %[[RID_I64:.*]] = stablehlo.convert %[[RID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[RID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[START_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: %[[START_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[LOCAL_SLICE:.*]] = stablehlo.dynamic_slice %[[GLOBAL_CST]], %[[START_0]], %[[START_1]], sizes = [2, 2] : (tensor<4x2xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  // CHECK-NEXT: return %[[LOCAL_SLICE]] : tensor<2x2xf32>
  return %0 : tensor<4x2xf32>
}
