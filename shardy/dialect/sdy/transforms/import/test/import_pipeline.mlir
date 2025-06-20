// RUN: sdy_opt %s -split-input-file -sdy-import-pipeline 2>&1 | FileCheck %s

// Verifies that function `-inliner` pass is applied
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
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
