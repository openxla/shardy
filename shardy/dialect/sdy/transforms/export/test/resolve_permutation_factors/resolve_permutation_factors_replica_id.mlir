// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true replica-count=8 partition-count=1" | FileCheck %s

sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @pad_single_left_hop_replica_id
func.func @pad_single_left_hop_replica_id(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>})
  -> tensor<7x8xi32> {
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK: %[[MC:.*]] = sdy.manual_computation
  // CHECK: %[[RID:.*]] = stablehlo.replica_id : tensor<ui32>
  // CHECK: %[[CONVERT:.*]] = stablehlo.convert %[[RID]] : (tensor<ui32>) -> tensor<i64>
  %0 = stablehlo.pad %arg0, %c, low = [3, 0], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(2)2}, {"b"}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<7x8xi32>

  return %0 : tensor<7x8xi32>
}
