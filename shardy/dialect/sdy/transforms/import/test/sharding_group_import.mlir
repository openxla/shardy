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
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=0 : tensor<4xf32>
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
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
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
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
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
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg0 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg2 group_id=2 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 567 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 23 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg2 group_id = 123456 : tensor<4xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: set_existing_shardings_for_sharding_group_members
func.func @set_existing_shardings_for_sharding_group_members(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>

  sdy.sharding_group %arg0 group_id = 43210 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 43210 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 43210 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: transitively_update_shardings_for_sharding_group_members
func.func @transitively_update_shardings_for_sharding_group_members(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK: %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  // CHECK: %cst_0 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<8x8xf32>

  sdy.sharding_group %arg0 group_id = 10 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 10 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 20 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 20 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 30 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 30 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: set_existing_shards_for_disjoint_groups
// CHECK-SAMEL    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
// CHECK-SAMEL    %arg3: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
func.func @set_existing_shards_for_disjoint_groups(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32>,
    %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
    %arg3: tensor<8x8xf32>) {
  // CHECK: %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>
  // CHECK: %cst_0 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<8x8xf32>
  // CHECK: %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf32>
  %2 = stablehlo.constant dense<0.0> : tensor<8x8xf32>

  sdy.sharding_group %arg0 group_id = 11111 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 11111 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 11111 : tensor<8x8xf32>

  sdy.sharding_group %arg2 group_id = 22222 : tensor<8x8xf32>
  sdy.sharding_group %arg3 group_id = 22222 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 22222 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: set_existing_shardings_in_manual_computation_op
func.func @set_existing_shardings_in_manual_computation_op(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={} (%arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) {
    // CHECK: %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
    %1 = stablehlo.add %arg2, %arg2 : tensor<8x8xf32>
    // CHECK: %2 = stablehlo.add %arg3, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
    %2 = stablehlo.add %arg3, %arg3 : tensor<8x8xf32>

    sdy.sharding_group %1 group_id = 1000 : tensor<8x8xf32>
    sdy.sharding_group %2 group_id = 1000 : tensor<8x8xf32>
    sdy.sharding_group %arg2 group_id = 1000 : tensor<8x8xf32>
    sdy.sharding_group %arg3 group_id = 1000 : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @set_existing_shardings_in_groups_with_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  // CHECK: %2 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
  %2 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 1000 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 1000 : tensor<8x8xf32>
  sdy.sharding_group %2 group_id = 1000 : tensor<8x8xf32>
  func.return
}
