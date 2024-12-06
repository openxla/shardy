// RUN: sdy_opt %s -split-input-file -sdy-import-pipeline 2>&1 | FileCheck %s

// Verifies that function `-inliner` pass is applied
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_2]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_0]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_1]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = call @add_matmul_to_lhs(%0, %arg0) : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-NOT: @add_matmul_to_lhs
func.func private @add_matmul_to_lhs(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// Verifies that `-apply-sharding-constraints` pass is applied after
// `-add-data_flow_edges` pass
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]] = stablehlo.optimization_barrier %arg0
  // CHECK-NEXT: sdy.data_flow_edge %[[OPT_BARRIER]] : tensor<32x96xf32>
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.optimization_barrier %arg0 : tensor<32x96xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"a"}]> :  tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}

// -----

// Verifies that both the -sdy-sharding-group-unification pass and sharding
// group canonicalizer pass are applied in order. This is checked by asserting
// group merging, reindexing and deduplication of ops are all applied.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
  // CHECK-DAG: sdy.sharding_group %arg0 group_id=0 : tensor<8x8xf32>
  // CHECK-DAG: sdy.sharding_group %arg1 group_id=0 : tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 1234 : tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 2345 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 1234 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 3456 : tensor<8x8xf32>
  func.return
}

// -----

// Verifies that the `-apply-sharding-constraints` pass is applied before the
// `-sharding-group-import` pass. This is validated by asserting that members
// of a sharding group pick up the sharding of a group member with a sharding
// constraint (the constraint needs to be added to the value in order for it to
// be applied to other group members).
sdy.mesh @mesh = <["a"=2]>
// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // CHECK: %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x16xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"a"}]> :  tensor<16x16xf32>
  // CHECK: %2 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %2 = stablehlo.add %arg0, %arg0 : tensor<16x16xf32>
  sdy.sharding_group %0 group_id = 32 : tensor<16x16xf32>
  sdy.sharding_group %2 group_id = 32 : tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// -----

// Verifies that the `-sdy-add-data-flow-edges` pass is applied before the
// `-sharding-group-import` pass. This is validated by adding a block argument
// of a while op to a sharding group which has a sharding constraint. This
// should be applied to other members of the group but can only happen if the
// `-sdy-add-data-flow-edges` pass is applied first.

sdy.mesh @mesh = <["a"=2]>

// CHECK: func.func @main
// CHECK-SAME %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
func.func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %inc = stablehlo.constant dense<1> : tensor<i32>
  %comp = stablehlo.constant dense<32> : tensor<i32>
  %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<16x16xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %comp : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %inc : tensor<i32>
    // Add a value with an explicit sharding to group_id=50 which will apply an
    // initial sharding to the result of the WhileOp outside of the loop.
    %3 = stablehlo.add %iterArg, %iterArg : tensor<16x16xf32>
    %4 = sdy.sharding_constraint %3 <@mesh, [{"a"}, {}]> :  tensor<16x16xf32>
    sdy.sharding_group %3 group_id = 50 : tensor<16x16xf32>
    stablehlo.return %3, %2 : tensor<16x16xf32>, tensor<i32>
  }

  // CHECK: sdy.data_flow_edge %3#0 sharding=<@mesh, [{"a"}, {}]> : tensor<16x16xf32>
  sdy.sharding_group %1#0 group_id = 50 : tensor<16x16xf32>
  return %1#0 : tensor<16x16xf32>
}

// -----

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>

// CHECK-LABEL: @add_manual_axes_to_replicated
func.func @add_manual_axes_to_replicated(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}:  out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}]>] out_shardings=[<@mesh, [{"c", ?}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>

// Due to the in_sharding being fully closed, the in_sharding is added to the
// func arg but with the manual axis added as replicated.
// CHECK-LABEL: @add_manual_axes_to_replicated_applied_constraint
// CHECK-SAME     %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}], replicated={"a"}>}
// CHECK-SAME     -> tensor<16x16xf32> {
func.func @add_manual_axes_to_replicated_applied_constraint(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c"}]>] out_shardings=[<@mesh, [{"c"}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    sdy.return %1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
