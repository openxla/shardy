// RUN: sdy_opt %s -split-input-file -sdy-import-pipeline 2>&1 | FileCheck %s

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

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}:  out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
  // CHECK-NEXT:   %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{?}]>
  // CHECK-NEXT:   sdy.return %2
  // CHECK-NEXT: }
  // CHECK-NEXT: %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{"c", ?}], replicated={"a"}>
  // CHECK-NEXT: return %1
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}]>] out_shardings=[<@mesh, [{"c", ?}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>

// Due to the in_sharding being fully closed, the in_sharding is added to the
// func arg but with the manual axis added as replicated.
// CHECK-LABEL: func @main
// CHECK-SAME     %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {}], replicated={"a"}>}
// CHECK-SAME     -> tensor<16x16xf32> {
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c"}], replicated={"a"}>] out_shardings=[<@mesh, [{"c"}], replicated={"a"}>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
  // CHECK-NEXT:   %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{}]>
  // CHECK-NEXT:   %3 = stablehlo.add %2, %2
  // CHECK-NEXT:   sdy.return %3
  // CHECK-NEXT: }
  // CHECK-NEXT: %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{"c"}], replicated={"a"}>
  // CHECK-NEXT: return %1
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c"}]>] out_shardings=[<@mesh, [{"c"}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    sdy.return %1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// This test verifies that the manual axes are cleaned up before adding data
// flow edges.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}], replicated={"a"}>] out_shardings=[<@mesh, [{?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{?}]>
  // CHECK-NEXT:   %3 = stablehlo.add %2, %2
  // CHECK-NEXT:   sdy.return %3
  // CHECK-NEXT: }
  // CHECK-NEXT: %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{?}], replicated={"a"}>
  // CHECK-NEXT: return %1
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}]>] out_shardings=[<@mesh, [{?}]>] manual_axes={"a"} (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// test: single_call
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.func_data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: return %[[EDGE]] : tensor<8xf32>
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
  // CHECK-NEXT: return %0
  return %arg0 : tensor<8xf32>
}
