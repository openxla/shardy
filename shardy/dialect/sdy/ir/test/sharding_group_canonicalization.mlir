// RUN: sdy_opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @duplicate_shard_group_ops
// CHECK-DAG: sdy.sharding_group %arg0 group_id=123
// CHECK-NEXT: return
func.func @duplicate_shard_group_ops(%arg0: tensor<8xf32>) {
  sdy.sharding_group %arg0 group_id=123 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=123 : tensor<8xf32>
  func.return
}

// CHECK-LABEL: func @duplicate_multiple_shard_group_ops
// CHECK-DAG: sdy.sharding_group %arg0 group_id=52
// CHECK-DAG: sdy.sharding_group %arg0 group_id=1117
// CHECK-NEXT: return
func.func @duplicate_multiple_shard_group_ops(%arg0: tensor<8xf32>) {
  sdy.sharding_group %arg0 group_id=1117 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=52 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=1117 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=52 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=52 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=1117 : tensor<8xf32>
  func.return
}

// CHECK-LABEL: func @duplicate_multiple_shard_group_ops_different_args
// CHECK-DAG: sdy.sharding_group %arg0 group_id=19
// CHECK-DAG: sdy.sharding_group %arg1 group_id=19
// CHECK-DAG: sdy.sharding_group %arg1 group_id=217
// CHECK-NEXT: return
func.func @duplicate_multiple_shard_group_ops_different_args(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  sdy.sharding_group %arg0 group_id=19 : tensor<8xf32>
  sdy.sharding_group %arg1 group_id=19 : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=19 : tensor<8xf32>
  sdy.sharding_group %arg1 group_id=217 : tensor<8xf32>
  sdy.sharding_group %arg1 group_id=217 : tensor<8xf32>
  sdy.sharding_group %arg1 group_id=19 : tensor<8xf32>
  func.return
}

// CHECK-LABEL: func @shard_group_no_dup_with_other_op
// CHECK-DAG: sdy.sharding_group %arg0 group_id=19
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg0
// CHECK-NEXT: return
func.func @shard_group_no_dup_with_other_op(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  sdy.sharding_group %arg0 group_id=19 : tensor<16x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  func.return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @shard_group_dup_with_mult_other_op
// CHECK-DAG: sdy.sharding_group %arg0 group_id=19
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg0
// CHECK-NEXT: %1 = stablehlo.add %arg0, %0
// CHECK-NEXT: return
func.func @shard_group_dup_with_mult_other_op(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  sdy.sharding_group %arg0 group_id=19 : tensor<16x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  sdy.sharding_group %arg0 group_id=19 : tensor<16x32xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<16x32xf32>
  func.return %1 : tensor<16x32xf32>
}
