// RUN: sdy_opt %s -sdy-sink-data-flow-edges | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @other_mesh = <["c"=4]>

// TODO(tomnatan): once ops like while are allowed to have shardings with
// different meshes, add a test that verifies that the first mesh name is used
// for missing shardings.

// CHECK-LABEL: func @data_flow_edge_on_block_arg
func.func @data_flow_edge_on_block_arg(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK:      %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK:      %[[C1:.*]] = stablehlo.constant dense<1>
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.constant dense<32> : tensor<i32>
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %[[C0]])
  // CHECK:      } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg_2, %[[C1]]
  // CHECK-NEXT:   %[[ADD_2:.*]] = stablehlo.add %iterArg, %iterArg
  // CHECK-NEXT:   stablehlo.return %[[ADD_2]], %[[ADD_1]]
  // CHECK-NEXT: }
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: return %[[WHILE]]#0
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = sdy.data_flow_edge %iterArg sharding=<@mesh, [{"a"}, {}]> : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %6 = stablehlo.add %4, %4 : tensor<32x96xf32>
    stablehlo.return %6, %5 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// CHECK-LABEL: func @data_flow_edge_on_op_result
func.func @data_flow_edge_on_op_result(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK:      %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK:      %[[C1:.*]] = stablehlo.constant dense<1>
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.constant dense<32> : tensor<i32>
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %[[C0]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, []>]>}
  // CHECK:      } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg_2, %[[C1]]
  // CHECK-NEXT:   %[[ADD_2:.*]] = stablehlo.add %iterArg, %iterArg
  // CHECK-NEXT:   stablehlo.return %[[ADD_2]], %[[ADD_1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE]]#0
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %5 = stablehlo.compare  LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %5 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %6 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %6, %5 : tensor<32x96xf32>, tensor<i32>
  }
  %4 = sdy.data_flow_edge %3#0 sharding=<@mesh, [{"a"}, {}]> : tensor<32x96xf32>
  return %4 : tensor<32x96xf32>
}

// CHECK-LABEL: func @no_shardings
func.func @no_shardings(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier %arg0, %arg1
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#0 : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %0#1 : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @some_edges_have_sharding
func.func @some_edges_have_sharding(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}]>, <@mesh, [{"a"}, {?}]>]>}
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#0 : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"a"}, {?}]> : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @all_edges_have_sharding
func.func @all_edges_have_sharding(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>, <@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"b", ?}, {?}]> : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"a"}, {}]> : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @missing_edge
func.func @missing_edge(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {?}]>, <@mesh, [{"a", ?}, {}]>]>}
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {?}]>, <@mesh, [{?}, {}]>]>}
    %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"a", ?}, {}]> : tensor<32x96xf32>
  return %0#0, %1 : tensor<32x96xf32>, tensor<32x96xf32>
}

// This use case shouldn't happen as `sdy-add-data-flow-edges` would copy the
// sharding of the result to the edge.
// CHECK-LABEL: func @sharding_overrided
func.func @sharding_overrided(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>, <@mesh, [{"a", ?}, {}]>]>}
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>, <@mesh, [{?}, {}]>]>}
    %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"b"}, {}]> : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"a", ?}, {}]> : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}

// This use case shouldn't happen as `sdy-add-data-flow-edges` would copy the
// sharding of the result to the edge.
// CHECK-LABEL: func @edge_missing_sharding
func.func @edge_missing_sharding(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {?}]>, <@mesh, [{"a", ?}, {}]>]>}
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {?}]>, <@mesh, [{?}, {}]>]>}
    %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#0 : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"a", ?}, {}]> : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}
