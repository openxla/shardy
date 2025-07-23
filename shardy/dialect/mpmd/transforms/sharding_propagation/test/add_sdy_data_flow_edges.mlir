// RUN: mpmd_opt %s -sdy-add-data-flow-edges 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @fragment_data_flow_edges_are_added_correctly
func.func @fragment_data_flow_edges_are_added_correctly(
   %arg0: !mesh_1_tensor_4_8_f32
     {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
   %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {"topology"=#mpmd.topology<<"m1": <["x"=4]>>>} {
    %0 = mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{"x"}, {}]>], out_shardings=[<@mesh, [{"x", ?}, {?}]>]>
      (%arg0) (%arg2: tensor<4x8xf32>) {
      // CHECK: sdy.data_flow_edge %arg2 sharding=<@mesh, [{"x"}, {}]> : tensor<4x8xf32>
      %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %3 : tensor<4x8xf32>
    } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
    // CHECK: %[[RESULT:.*]] = sdy.data_flow_edge %0 sharding=<@mesh, [{"x", ?}, {?}]>
    // CHECK: return %[[RESULT]]
    func.return %0 : !mesh_1_tensor_4_8_f32
  }

// CHECK-LABEL: func @fragment_with_multiple_args_and_results
func.func @fragment_with_multiple_args_and_results(
   %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
    %0:2 = mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"x"}]>]>
      (%arg0, %arg1)
      (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
      // CHECK: sdy.data_flow_edge %arg2 sharding=<@mesh, [{"x"}, {}]>
      // CHECK: sdy.data_flow_edge %arg3 sharding=<@mesh, [{}, {"x"}]>
      %3 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
      mpmd.return %3, %3 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
    // CHECK-NOT: sharding=
    // CHECK: %[[RESULT:.*]] = sdy.data_flow_edge %0#0
    // CHECK: %[[RESULT1:.*]] = sdy.data_flow_edge %0#1
    func.return %0#0 : !mesh_1_tensor_4_8_f32
  }

// CHECK-LABEL: @for_loop_with_sharding
func.func @for_loop_with_sharding(%arg0: tensor<10xui32>, %arg1: tensor<10xui32>)
  -> (tensor<10xui32>, tensor<10xui32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2, "y"=2]>>
    >} {
    %0:2 = mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}]>, <@mesh, [{"y", ?}]>]>, unroll_factor = 3 : ui32}
      (%arg2: tensor<10xui32>, %arg3: tensor<10xui32>, %index: tensor<ui32>) {
      %1 = stablehlo.broadcast_in_dim %index, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %2 = stablehlo.add %arg2, %1 : tensor<10xui32>
      %3 = stablehlo.add %arg2, %arg3 : tensor<10xui32>
      mpmd.return %2, %3 : tensor<10xui32>, tensor<10xui32>
  } : tensor<10xui32>, tensor<10xui32>
  // CHECK %[[RESULT:.*]] = sdy.data_flow_edge %0#0 sharding=<@mesh, [{"x", ?}]> : tensor<10xui32>
  // CHECK %[[RESULT1:.*]] = sdy.data_flow_edge %0#1 sharding=<@mesh, [{"y", ?}]> : tensor<10xui32>
  func.return %0#0, %0#1 : tensor<10xui32>, tensor<10xui32>
}

// CHECK-LABEL: @for_loop_without_sharding
func.func @for_loop_without_sharding(%arg0: tensor<10xui32>, %arg1: tensor<10xui32>)
  -> (tensor<10xui32>, tensor<10xui32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
    %0:2 = mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, unroll_factor = 3 : ui32}
      (%arg2: tensor<10xui32>, %arg3: tensor<10xui32>, %index: tensor<ui32>) {
      %1 = stablehlo.broadcast_in_dim %index, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %2 = stablehlo.add %arg2, %1 : tensor<10xui32>
      %3 = stablehlo.add %arg2, %arg3 : tensor<10xui32>
      mpmd.return %2, %3 : tensor<10xui32>, tensor<10xui32>
  } : tensor<10xui32>, tensor<10xui32>
  // CHECK %[[RESULT:.*]] = sdy.data_flow_edge %0#0 : tensor<10xui32>
  // CHECK %[[RESULT1:.*]] = sdy.data_flow_edge %0#1 : tensor<10xui32>
  func.return %0#0, %0#1 : tensor<10xui32>, tensor<10xui32>
}
