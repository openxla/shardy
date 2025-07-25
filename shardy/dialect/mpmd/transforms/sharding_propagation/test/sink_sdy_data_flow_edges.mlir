// RUN: mpmd_opt %s -sdy-sink-data-flow-edges 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
// CHECK-LABEL: func @one_edge_for_arg_and_one_for_result
func.func @one_edge_for_arg_and_one_for_result(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
  // CHECK-NOT: sdy.data_flow_edge
  // CHECK:  mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{"x"}, {}]>], out_shardings=[<@mesh, [{"x", ?}, {?}]>]> (%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %2 = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"x"}, {}]> : tensor<4x8xf32>
    %3 = stablehlo.add %2, %2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %1 = sdy.data_flow_edge %0 sharding=<@mesh, [{"x", ?}, {?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// CHECK-LABEL: func @multiple_edges_for_args_and_results
func.func @multiple_edges_for_args_and_results(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
  // CHECK-NOT: sdy.data_flow_edge
  // CHECK: mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"x"}]>], out_shardings=[<@mesh, [{"x", ?}, {?}]>, <@mesh, [{?}, {"x", ?}]>]> (%arg0, %arg1)
  %0:2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"x"}, {}]> : tensor<4x8xf32>
    %4 = sdy.data_flow_edge %arg3 sharding=<@mesh, [{}, {"x"}]> : tensor<4x8xf32>
    %5 = stablehlo.add %3, %4 : tensor<4x8xf32>
    mpmd.return %5, %5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"x", ?}, {?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{?}, {"x", ?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// CHECK-LABEL: func @some_edges_have_shardings
func.func @some_edges_have_shardings(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
  // CHECK-NOT: sdy.data_flow_edge
  // CHECK: mpmd.fragment<mesh="m1", origin=[], out_shardings=[<@mesh, [{"x", ?}, {?}]>, <@mesh, [{?}, {"x", ?}]>]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>)
  %0:2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = sdy.data_flow_edge %arg2 : tensor<4x8xf32>
    %4 = sdy.data_flow_edge %arg3 : tensor<4x8xf32>
    %5 = stablehlo.add %3, %4 : tensor<4x8xf32>
    mpmd.return %5, %5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"x", ?}, {?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{?}, {"x", ?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// CHECK-LABEL: func @not_all_args_have_an_edge
func.func @not_all_args_have_an_edge(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
  // CHECK-NOT: sdy.data_flow_edge
  // CHECK: mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{?}, {?}]>], out_shardings=[<@mesh, [{"x", ?}, {?}]>, <@mesh, [{?}, {"x", ?}]>]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>)
  %0:2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"x"}, {}]> : tensor<4x8xf32>
    %5 = stablehlo.add %3, %arg3 : tensor<4x8xf32>
    mpmd.return %5, %5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"x", ?}, {?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{?}, {"x", ?}]> : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// CHECK-LABEL: func @no_shardings_on_edges
func.func @no_shardings_on_edges(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
  // CHECK-NOT: sdy.data_flow_edge
  // CHECK-NOT: sdy.sharding
  // CHECK: mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1)
  %0:2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = sdy.data_flow_edge %arg2 : tensor<4x8xf32>
    %4 = sdy.data_flow_edge %arg3 : tensor<4x8xf32>
    %5 = stablehlo.add %3, %4 : tensor<4x8xf32>
    mpmd.return %5, %5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
  %1 = sdy.data_flow_edge %0#0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %2 = sdy.data_flow_edge %0#1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// CHECK-LABEL: func @for_loop
func.func @for_loop(%arg0: tensor<10xui32>, %arg1: tensor<10xui32>) -> (tensor<10xui32>, tensor<10xui32>) attributes {topology = #mpmd.topology<<"m1" : <["x"=4, "y"=2]>>>} {
  // CHECK: mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}]>, <@mesh, [{"y", ?}]>]>, unroll_factor = 3 : ui32}
  %0:2 = mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}]>, <@mesh, [{"x", ?}]>]>, unroll_factor = 3 : ui32} (%arg2: tensor<10xui32>, %arg3: tensor<10xui32>, %index: tensor<ui32>) {
    %3 = stablehlo.broadcast_in_dim %index, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %4 = stablehlo.add %arg2, %3 : tensor<10xui32>
    %5 = stablehlo.add %arg2, %arg3 : tensor<10xui32>
    mpmd.return %4, %5 : tensor<10xui32>, tensor<10xui32>
  } : tensor<10xui32>, tensor<10xui32>
  // CHECK-NOT: sdy.data_flow_edge
  %1 = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"x", ?}]> : tensor<10xui32>
  %2 = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"y", ?}]> : tensor<10xui32>
  return %1, %2 : tensor<10xui32>, tensor<10xui32>
}
