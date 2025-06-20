// RUN: sdy_opt -split-input-file %s -sdy-sharding-group-import | FileCheck %s

// CHECK-LABEL: sharding_groups_no_overlap
func.func @sharding_groups_no_overlap(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 1 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_all_overlap
func.func @sharding_groups_all_overlap(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_overlap_min_id_used
func.func @sharding_groups_overlap_min_id_used(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_mixed_overlaps
func.func @sharding_groups_mixed_overlaps(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 3 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_reindexes_ids
func.func @sharding_groups_reindexes_ids(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 12 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 89 : tensor<4xf32>
  func.return
}

// -----

// CHECK-LABEL: sharding_groups_reindex_ordering_matches_min_element_ordering
func.func @sharding_groups_reindex_ordering_matches_min_element_ordering(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) {
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg2 group_id=2 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 567 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 23 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg2 group_id = 123456 : tensor<4xf32>
  func.return
}
