// RUN: sdy_opt %s -sdy-sink-data-flow-edges | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
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
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}]>, <@mesh, [{"a", ?}, {}]>]>}
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
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}]>, <@mesh, [{"a", ?}, {}]>]>}
  // CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {?}]>, <@mesh, [{?}, {}]>]>}
    %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %1 = sdy.data_flow_edge %0#0 : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"a", ?}, {}]> : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @named_computation_multiple_inputs_outputs
func.func @named_computation_multiple_inputs_outputs(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func">(%arg0, %arg1)
  // CHECK-SAME:   out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>]
  // CHECK-SAME:   (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
  // CHECK-NEXT:   sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: return %[[NC]]#0, %[[NC]]#1 : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
    %3 = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
    %4 = sdy.data_flow_edge %arg3 : tensor<4x2xi32>
    sdy.return %3, %4 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{?}, {}]> : tensor<4x2xi32>
  return %1, %2 : tensor<8x2xi32>, tensor<4x2xi32>
}

// This use case shouldn't happen as `sdy-add-data-flow-edges` would copy the
// sharding of the out_sharding/in_sharding to the edge.
// CHECK-LABEL: func @named_computation_multiple_inputs_outputs_override
func.func @named_computation_multiple_inputs_outputs_override(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func">(%arg0, %arg1)
  // CHECK-SAME: in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, [{?}, {?}]>]
  // CHECK-SAME: out_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, [{?}, {}]>]
  // CHECK-SAME: (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
  // CHECK-NEXT:   sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: return %[[NC]]#0, %[[NC]]#1 : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = sdy.named_computation<"my_func">(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
    %3 = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"b"}, {}]>  : tensor<8x2xi32>
    %4 = sdy.data_flow_edge %arg3 : tensor<4x2xi32>
    sdy.return %3, %4 : tensor<8x2xi32>, tensor<4x2xi32>
  } {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{?}, {}]> : tensor<4x2xi32>
  return %1, %2 : tensor<8x2xi32>, tensor<4x2xi32>
}


// CHECK-LABEL: func @manual_computation_with_manual_axes
func.func @manual_computation_with_manual_axes(
    %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME:     manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:   %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.return %1 : tensor<16x32xf32>
  // CHECK-NEXT: } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NEXT: return %0 : tensor<32x32xf32>
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh, [{"b", ?}, {?}]>]
      out_shardings=[<@mesh, [{"b", ?}, {?}]>]
      manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<16x32xf32>
    %3 = stablehlo.add %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %3 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{"b", "a", ?}, {?}]> : tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: func @manual_computation_update_in_out_shardings
func.func @manual_computation_update_in_out_shardings(
    %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {"c", ?}]>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {"c", ?}]>}) {
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a", "b", ?}, {"c", ?}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"a", "b", ?}, {"c", ?}]>]
  // CHECK-SAME:     manual_axes={} (%arg1: tensor<32x32xf32>) {
  // CHECK-NEXT:   sdy.return %arg1 : tensor<32x32xf32>
  // CHECK-NEXT: } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<32x32xf32>
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh, [{"a", ?}, {?}]>]
      out_shardings=[<@mesh, [{"a", ?}, {?}]>]
      manual_axes={} (%arg1: tensor<32x32xf32>) {
    %2 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{"a", "b", ?}, {"c", ?}]> : tensor<32x32xf32>
    sdy.return %2 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{"a", "b", ?}, {"c", ?}]> : tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Don't need to preserve any of the debug info on the edges as they will be
// on the ops that used them.
//
// CHECK-LABEL: func @manual_computation_origin_debug_info
func.func @manual_computation_origin_debug_info(%arg0: tensor<32x32x32xf32>) -> tensor<32x32x32xf32> {
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:     manual_axes={} (%arg1: tensor<32x32x32xf32>) {
  // CHECK-NEXT:   sdy.return %arg1 : tensor<32x32x32xf32>
  // CHECK-NEXT: } {sdy.origin_sharding_name = "mc_0"} : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<32x32x32xf32>
  %1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a", ?}, {?}, {?}]>] out_shardings=[<@mesh, [{?}, {"b", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32x32xf32>) {
    %3 = sdy.data_flow_edge %arg1 sharding=<@mesh, [{"a", ?}, {"b", ?}, {?}]> {sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"}} : tensor<32x32x32xf32>
    sdy.return %3 : tensor<32x32x32xf32>
  } {sdy.origin_sharding_name = "mc_0"} : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  %2 = sdy.data_flow_edge %1 sharding=<@mesh, [{"a", ?}, {"b", ?}, {?}]> {sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"}} : tensor<32x32x32xf32>
  return %2 : tensor<32x32x32xf32>
}
