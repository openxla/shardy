// RUN: sdy_opt -split-input-file %s -sdy-sharding-group-import | FileCheck %s
// RUN: sdy_opt %s -split-input-file -sdy-sharding-group-import -verify-diagnostics

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

// Emit warning as well for sharding groups which have incompatible shardings
// inferred from initial constraints.
// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_shardings_in_sharding_group
func.func @add_extra_sharding_constraint_for_incompatible_shardings_in_sharding_group(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  // Sharding Group and Sharding Constraint compatibility checks happend after
  // unification + canonicalization of group ids.
  // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
  sdy.sharding_group %arg0 group_id = 555 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 555 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Emit warning as well for sharding groups which have incompatible shardings
// inferred from initial constraints.
// CHECK-LABEL: add_extra_sharding_constraint_for_transitively_inferred_incompatible_shardings_in_unified_sharding_group
func.func @add_extra_sharding_constraint_for_transitively_inferred_incompatible_shardings_in_unified_sharding_group(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_2:.*]] = sdy.sharding_constraint %[[CST_0]] <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_3:.*]] = sdy.sharding_constraint %[[CST_1]] <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_2]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_3]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<8x8xf32>

  // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
  sdy.sharding_group %arg0 group_id = 10 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 10 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 20 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 20 : tensor<8x8xf32>

  // The shard group below will cause the above sharding groups to be merged
  // by transitivity this implies that all of {%arg0, %arg1, 0, 1} should have
  // the same sharding. Note that %0 and %1 are compatible by them selves but
  // %arg0 and %arg1 are not due to their initial shardings.
  sdy.sharding_group %1 group_id = 30 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 30 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_shardings_in_manual_computation
func.func @add_extra_sharding_constraint_for_incompatible_shardings_in_manual_computation(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:  sdy.manual_computation
  // CHECK-NEXT:    %[[WSC_0:.*]] = sdy.sharding_constraint %arg3 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:    %[[WSC_1:.*]] = sdy.sharding_constraint %arg2 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:    sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:    sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"b"}, {}]>] manual_axes={} (%arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) {
    // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
    sdy.sharding_group %arg2 group_id = 8675 : tensor<8x8xf32>
    sdy.sharding_group %arg3 group_id = 8675 : tensor<8x8xf32>
    sdy.return %arg2 : tensor<8x8xf32>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_shardings_with_sharding_constraint
func.func @add_extra_sharding_constraint_for_incompatible_shardings_with_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[WSC_0]], %[[WSC_0]] : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_2]] group_id=0 : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
  sdy.sharding_group %arg0 group_id = 1000 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 1000 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_sharding_shardings
func.func @add_extra_sharding_constraint_for_incompatible_sharding_shardings(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_2:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_3:.*]] = sdy.sharding_constraint %[[WSC_2]] <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_3]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[WSC_1]], %[[WSC_3]] : tensor<8x8xf32>
  // CHECK-NEXT:  return %[[ADD]] : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {?}]> : tensor<8x8xf32>
  // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
  sdy.sharding_group %0 group_id=1183 : tensor<8x8xf32>

  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %1 group_id=1183 : tensor<8x8xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x8xf32>
  func.return %2: tensor<8x8xf32>
}
