// RUN: sdy_opt -split-input-file %s -sdy-group-canonicalization | FileCheck %s

// CHECK-LABEL: sharding_groups_no_overlap
func.func @sharding_groups_no_overlap(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=2 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_all_overlap
func.func @sharding_groups_all_overlap(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 3 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_overlap_min_id_used
func.func @sharding_groups_overlap_min_id_used(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 3 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 3 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_mixed_overlaps
func.func @sharding_groups_mixed_overlaps(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=3 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=3 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 3 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 4 : tensor<4xf32>
  func.return
}
