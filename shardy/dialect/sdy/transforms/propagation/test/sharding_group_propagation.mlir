// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate -verify-diagnostics | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2, "d"=2]>

// A propagation barrier prevents the sharding from arg0 to propagate to %1 but
// %1 still receives the sharding due to the sharding group.
// CHECK-LABEL: func @shard_as_applies_despite_barrier
func.func @shard_as_applies_despite_barrier(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %0 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
  %1 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
  // CHECK: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  %2 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  // CHECK-NEXT: sdy.propagation_barrier %1 allowed_direction=NONE {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  %3 = sdy.propagation_barrier %1 allowed_direction=NONE : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %2, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  %4 = stablehlo.add %2, %3 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id=0 : tensor<8x8xf32>
  sdy.sharding_group %4 group_id=0 : tensor<8x8xf32>
  return %4 : tensor<8x8xf32>
}

// Validate that sharding groups are propagated across function calls. %0 will
// receive a sharding from %arg0 which should then be propagated across function
// calls to %1 (which otherwise would not receive a sharding).
// CHECK-LABEL: func @shard_as_applies_across_functionA
func.func @shard_as_applies_across_functionA(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: func @shard_as_applies_across_functionB
func.func @shard_as_applies_across_functionB(%arg0: tensor<8x8xf32>)
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 1 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// Verifies sharding group constraint propagation reflects user priority.
// Without the sharding group %1 would have %arg1's sharding, so we validate
// that because of the group, all intermediate values in this function call have
// the same sharding (which had the highest priority).
// CHECK-LABEL: func @shard_as_reflects_user_priority
func.func @shard_as_reflects_user_priority(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}p0, {"c", ?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", "a",?}p1, {"b", ?}]>})
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 2 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 2 : tensor<8x8xf32>
  sdy.sharding_group %2 group_id = 2 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// Check sharding group constraint is propagated across a DataFlowEdgeOp. %3
// will receive the sharding from %arg0 at which point it will set the sharding
// on %4 (which otherwise would not have a sharding propagated).
// CHECK-LABEL: func @shard_as_across_dataflow_edge
func.func @shard_as_across_dataflow_edge(
  %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"d"}]>})
  -> tensor<16x16xf32> {
  // CHECK: stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"d", ?}]>]>} dense<1.000000e+00> : tensor<16x16xf32>
  %4 = stablehlo.constant dense<1.0> : tensor<16x16xf32>
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %inc = stablehlo.constant dense<1> : tensor<i32>
  %comp = stablehlo.constant dense<32> : tensor<i32>
  // CHECK: %[[RESULT:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<16x16xf32>, tensor<i32>
  %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<16x16xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %comp : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %inc : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"d", ?}]>]>} : tensor<16x16xf32>
    %3 = stablehlo.add %iterArg, %iterArg : tensor<16x16xf32>
    sdy.sharding_group %3 group_id = 3 : tensor<16x16xf32>
    stablehlo.return %3, %2 : tensor<16x16xf32>, tensor<i32>
  }
  // CHECK: sdy.data_flow_edge %[[RESULT]]#0 sharding=<@mesh, [{?}, {"d", ?}]> : tensor<16x16xf32>
  %5 = sdy.data_flow_edge %1#0 : tensor<16x16xf32>
  %6 = sdy.data_flow_edge %1#1 : tensor<i32>
  sdy.sharding_group %4 group_id = 3 : tensor<16x16xf32>
  return %5 : tensor<16x16xf32>
}

// CHECK-LABEL: func @different_group_ids_greater_group_id_first
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:    %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>},
// CHECK-SAME:    %arg3: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>},
// CHECK-SAME:        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>})
func.func @different_group_ids_greater_group_id_first(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
  %arg1: tensor<8x8xf32>,
  %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>},
  %arg3: tensor<8x8xf32>)
   -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  sdy.sharding_group %arg0 group_id=5 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id=5 : tensor<8x8xf32>
  sdy.sharding_group %arg2 group_id=4 : tensor<8x8xf32>
  sdy.sharding_group %arg3 group_id=4 : tensor<8x8xf32>
  return %arg0, %arg1, %arg2, %arg3 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// -----

// Tests for initial member sharding sync

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: set_existing_shardings_for_sharding_group_members
func.func @set_existing_shardings_for_sharding_group_members(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>

  sdy.sharding_group %arg0 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 0 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Emit warning as well for sharding groups which have incompatible shardings
// inferred from initial constraints.
// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_shardings_in_sharding_group
// expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
func.func @add_extra_sharding_constraint_for_incompatible_shardings_in_sharding_group(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{"b", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"b", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  // Sharding Group and Sharding Constraint compatibility checks happend after
  // unification + canonicalization of group ids.
  sdy.sharding_group %arg0 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 0 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_shardings_in_manual_computation
func.func @add_extra_sharding_constraint_for_incompatible_shardings_in_manual_computation(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:  sdy.manual_computation
  // CHECK-NEXT:    %[[WSC_0:.*]] = sdy.sharding_constraint %arg3 <@mesh, [{"b", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:    %[[WSC_1:.*]] = sdy.sharding_constraint %arg2 <@mesh, [{"b", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:    sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:    sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"b"}, {}]>] manual_axes={} (%arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) {
    sdy.sharding_group %arg2 group_id = 0 : tensor<8x8xf32>
    sdy.sharding_group %arg3 group_id = 0 : tensor<8x8xf32>
    sdy.return %arg2 : tensor<8x8xf32>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_shardings_with_sharding_constraint
// expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
func.func @add_extra_sharding_constraint_for_incompatible_shardings_with_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a", ?}, {"b", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[WSC_0]], %[[WSC_0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{"a", ?}, {"b", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_2]] group_id=0 : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 0 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 0 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_partially_closed_replicated_sharding
func.func @add_extra_sharding_constraint_for_partially_closed_replicated_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_2:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_3:.*]] = sdy.sharding_constraint %[[WSC_2]] <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_3]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  return %[[WSC_1]], %[[WSC_3]]
  // expected-warning@below {{The initial operand shardings on the sharding groups of groupID: 0}}
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %0 group_id=0 : tensor<8x8xf32>

  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %1 group_id=0 : tensor<8x8xf32>
  func.return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_fully_open_replicated_sharding
func.func @add_extra_sharding_constraint_for_fully_open_replicated_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT:  %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_0]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  %[[WSC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  // CHECK-NEXT:  sdy.sharding_group %[[WSC_1]] group_id=0 : tensor<8x8xf32>
  // CHECK-NEXT:  return %[[WSC_0]], %[[WSC_1]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %0 group_id=0 : tensor<8x8xf32>

  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %1 group_id=0 : tensor<8x8xf32>
  func.return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}
