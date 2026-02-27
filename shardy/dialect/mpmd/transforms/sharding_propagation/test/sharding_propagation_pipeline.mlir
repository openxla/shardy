// RUN: mpmd_opt %s -mpmd-sharding-propagation-pipeline -split-input-file -mlir-diagnostic-verbosity-level=errors 2>&1 | FileCheck %s

module {
sdy.mesh @mesh = <["x"=4]>

// CHECK-LABEL: @simple_propagation_within_fragment
func.func public @simple_propagation_within_fragment(
  %arg0: !mpmd.mesh_tensor<"mesh1", tensor<16x3x5xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
  %arg1: !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>})
  -> (!mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
  // CHECK: %[[FRAG1:.*]] = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg0, %arg1) (%arg2: tensor<16x3x5xf32>, %arg3: tensor<16x10x3xf32>) {
  // CHECK:   %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<16x3x5xf32>
  // CHECK:   %[[ADD:.*]] = stablehlo.add %arg2, %[[CST]]
  // CHECK:   %[[DOT:.*]] = stablehlo.dot_general
  // CHECK:   mpmd.return %[[DOT]] : tensor<16x10x5xf32>
  // CHECK: }
  // CHECK: (!mpmd.mesh_tensor<"mesh1", tensor<16x3x5xf32>, sharding=<@mesh, [{"x"}, {}, {}]>>,
  // CHECK: !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>, sharding=<@mesh, [{"x"}, {}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>, sharding=<@mesh, [{"x"}, {}, {}]>>
  %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg0, %arg1) (%arg2: tensor<16x3x5xf32>, %arg3: tensor<16x10x3xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<16x3x5xf32>
    %1 = stablehlo.add %arg2, %cst : tensor<16x3x5xf32>
    %4 = stablehlo.dot_general %arg3, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x10x3xf32>, tensor<16x3x5xf32>) -> tensor<16x10x5xf32>
    mpmd.return %4 : tensor<16x10x5xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<16x3x5xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>>
}

}

// -----
module {
sdy.mesh @mesh = <["x"=4]>

// CHECK-LABEL: @simple_propagation_within_fragment_simplified
func.func public @simple_propagation_within_fragment_simplified(
  %arg0: !mpmd.mesh_tensor<"mesh1", tensor<16x3x5xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
  %arg1: !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>})
  -> (!mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
  // CHECK-NEXT: %[[FRAG1:.*]] = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg0, %arg1) (%arg2: tensor<16x3x5xf32>, %arg3: tensor<16x10x3xf32>) {
  // CHECK-NEXT:   %[[DOT:.*]] = stablehlo.dot_general %arg3, %arg2, batching_dims = [0] x [0]
  // CHECK-NEXT:   mpmd.return %[[DOT]] : tensor<16x10x5xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: (!mpmd.mesh_tensor<"mesh1", tensor<16x3x5xf32>, sharding=<@mesh, [{"x"}, {}, {}]>>,
  // CHECK-SAME: !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>, sharding=<@mesh, [{"x"}, {}, {}]>>) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>, sharding=<@mesh, [{"x"}, {}, {}]>>
  %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg0, %arg1) (%arg2: tensor<16x3x5xf32>, %arg3: tensor<16x10x3xf32>) {
    %cst = stablehlo.constant dense<0.0> : tensor<16x3x5xf32>
    %cst_false = stablehlo.constant dense<false> : tensor<i1>
    %extra_select = stablehlo.select %cst_false, %arg2, %cst : tensor<i1>, tensor<16x3x5xf32>
    %1 = stablehlo.add %arg2, %extra_select : tensor<16x3x5xf32>
    %4 = stablehlo.dot_general %arg3, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x10x3xf32>, tensor<16x3x5xf32>) -> tensor<16x10x5xf32>
    mpmd.return %4 : tensor<16x10x5xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<16x3x5xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh1", tensor<16x10x5xf32>>
}

}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=4]>

// CHECK-LABEL: @two_fragments_one_producer_one_consumer
func.func @two_fragments_one_producer_one_consumer(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
   %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
  // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %2 = mpmd.fragment<mesh="m1", origin=["consumer"]> (%0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %2 : !mesh_1_tensor_4_8_f32
}
}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=4]>

// CHECK-LABEL: @fragment_with_transfer
func.func @fragment_with_transfer(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
   %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_2_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: %1 = mpmd.transfer %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  // CHECK: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  func.return %1 : !mesh_2_tensor_4_8_f32
}
}


// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=4]>

// CHECK-LABEL: @fragment_with_transfer_backward_propagation
func.func @fragment_with_transfer_backward_propagation(
  // CHECK: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  // CHECK: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
   %arg0: !mesh_1_tensor_4_8_f32,
   %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_2_tensor_4_8_f32  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["producer"]>
  // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: mpmd.transfer %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
  // CHECK: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  func.return %1 : !mesh_2_tensor_4_8_f32
}
}


// -----

#homogenous_topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>
!mesh_2_tensor_16_32_f32 = !mpmd.mesh_tensor<"m2", tensor<16x32xf32>>

module {
sdy.mesh @mesh = <["x"=8]>

// CHECK-LABEL: func @identify_function_with_arg_sharding
// CHECK: (%arg0: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {}]>>,
// CHECK:  %arg1: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
// CHECK: (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>>,
// CHECK:  !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>)
func.func @identify_function_with_arg_sharding(
  %arg0: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}], replicated = {"x"}>},
  %arg1: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>}
)
  -> (!mesh_1_tensor_16_32_f32,
      !mesh_1_tensor_16_32_f32)
  attributes {topology=#homogenous_topology} {
  func.return %arg0, %arg1 : !mesh_1_tensor_16_32_f32, !mesh_1_tensor_16_32_f32
}
}

// -----

#homogenous_topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>
!mesh_2_tensor_16_32_f32 = !mpmd.mesh_tensor<"m2", tensor<16x32xf32>>

module {
sdy.mesh @mesh = <["x"=8]>

// CHECK-LABEL: func @identify_function_with_result_sharding
// CHECK: (%arg0: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {"x"}]>>,
// CHECK: %arg1: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
// CHECK: (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {"x"}]>>,
// CHECK: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>)
func.func @identify_function_with_result_sharding(
  %arg0: !mesh_1_tensor_16_32_f32,
  %arg1: !mesh_1_tensor_16_32_f32
)
  -> (!mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x"}]>},
      !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  attributes {topology=#homogenous_topology} {
  func.return %arg0, %arg1 : !mesh_1_tensor_16_32_f32, !mesh_1_tensor_16_32_f32
}
}

// -----
#homogenous_topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>
!mesh_2_tensor_16_32_f32 = !mpmd.mesh_tensor<"m2", tensor<16x32xf32>>

module {
sdy.mesh @mesh = <["x"=8]>

// CHECK-LABEL: @propagates_to_func_args_and_inter_mesh_transfers
func.func @propagates_to_func_args_and_inter_mesh_transfers(%arg0: !mesh_1_tensor_16_32_f32)
  -> !mesh_2_tensor_16_32_f32 attributes {topology=#homogenous_topology} {
  // CHECK-NEXT: %[[FRAG1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:   mpmd.return %[[ADD]] : tensor<16x32xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[FRAG1]] : {{.*}}"m1"{{.*}}<@mesh, [{"x"}, {}]>{{.*}} -> {{.*}}"m2"{{.*}}<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[FRAG2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER]]) (%arg1
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:   mpmd.return %[[ADD]] : tensor<16x32xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAG2]] : {{.*}}"m2"{{.*}}<@mesh, [{"x"}, {}]>
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<16x32xf32>) {
    %4 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<16x32xf32>
    mpmd.return %4 : tensor<16x32xf32>
  } : (!mesh_1_tensor_16_32_f32) -> (!mesh_1_tensor_16_32_f32)
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_16_32_f32) -> !mesh_2_tensor_16_32_f32
  %2 = mpmd.fragment<mesh="m2", origin=[]> (%1) (%arg1: tensor<16x32xf32>) {
    %5 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    mpmd.return %5 : tensor<16x32xf32>
  } : (!mesh_2_tensor_16_32_f32) -> (!mesh_2_tensor_16_32_f32)
  func.return %2 : !mesh_2_tensor_16_32_f32
}
}

// -----

#homogenous_topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>
!mesh_2_tensor_16_32_f32 = !mpmd.mesh_tensor<"m2", tensor<16x32xf32>>

module {
sdy.mesh @mesh = <["x"=8]>

// CHECK-LABEL: func @pass_func_arg_shardings_into_frag_through_transfers
func.func @pass_func_arg_shardings_into_frag_through_transfers(
  %arg0: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (!mesh_2_tensor_16_32_f32) attributes {topology=#homogenous_topology} {
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>) -> !mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>
  // CHECK-NEXT: mpmd.fragment<mesh="m2", origin=[]>
  // CHECK: } : (!mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>) -> !mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %1 = mpmd.transfer %arg0 : (!mesh_1_tensor_16_32_f32) -> !mesh_2_tensor_16_32_f32
  %2 = mpmd.fragment<mesh="m2", origin=[]> (%1) (%arg1: tensor<16x32xf32>) {
    %5 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    mpmd.return %5 : tensor<16x32xf32>
  } : (!mesh_2_tensor_16_32_f32) -> (!mesh_2_tensor_16_32_f32)
  func.return %2 : !mesh_2_tensor_16_32_f32
}
}

// -----

#homogenous_topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>
!mesh_2_tensor_16_32_f32 = !mpmd.mesh_tensor<"m2", tensor<16x32xf32>>

module {
sdy.mesh @mesh = <["x"=8]>

// CHECK-LABEL: func @preserves_replicated_through_transfer(
// CHECK-SAME:      %arg0: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {}]>>
// CHECK-SAME:      %arg1: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>
// CHECK-SAME:      -> (!mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME:          !mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{}, {}]>>)
func.func @preserves_replicated_through_transfer(
  %arg0: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}], replicated = {"x"}>},
  %arg1: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>}
)
  -> (!mesh_2_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>},
      !mesh_2_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}], replicated = {"x"}>})
  attributes {topology=#homogenous_topology} {
// CHECK-NEXT: %[[RESHARD1:.*]] = mpmd.fragment
// CHECK: } : (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {}]>>) -> !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>
// CHECK-NEXT: mpmd.transfer %[[RESHARD1]] :
// CHECK-NEXT: mpmd.transfer %arg1 :
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>
// CHECK-NEXT: %[[RESHARD2:.*]] = mpmd.fragment
// CHECK: } : (!mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>) -> !mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{}, {}]>>
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor_16_32_f32) -> !mesh_2_tensor_16_32_f32
  %1 = mpmd.transfer %arg1 : (!mesh_1_tensor_16_32_f32) -> !mesh_2_tensor_16_32_f32
  func.return %0, %1 : !mesh_2_tensor_16_32_f32, !mesh_2_tensor_16_32_f32
}

}


// -----

!mesh_1_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>>
!mesh_2_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>>
#topology = #mpmd.topology<<"mesh1" : <["devices"=4]>>, <"mesh2" : <["devices"=4]>>>
module {
sdy.mesh @mesh = <["devices"=4]>

// CHECK-LABEL: func @introduce_reshard_for_transfer_operand_and_result_with_different_sharding
func.func @introduce_reshard_for_transfer_operand_and_result_with_different_sharding(
  %arg0: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>},
  %arg1: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>}) ->
  (!mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices"}, {}]>},
  !mesh_2_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) attributes {topology = #topology} {
  %0 = mpmd.fragment<mesh="mesh1", origin=["add"]> (%arg0, %arg1) (%arg2: tensor<8x2xf32>, %arg3: tensor<8x2xf32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x2xf32>
    mpmd.return %2 : tensor<8x2xf32>
  } : (!mesh_1_tensor_8_2_f32, !mesh_1_tensor_8_2_f32) -> !mesh_1_tensor_8_2_f32
  // CHECK: %[[FRAG:.*]] = mpmd.fragment<mesh="mesh1", origin=["add"]> (%arg0, %arg1) (%arg2: tensor<8x2xf32>, %arg3: tensor<8x2xf32>) {
  // CHECK: %[[TRANSFER:.*]] = mpmd.transfer %0 : (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>) -> !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>
  // CHECK: %[[RESHARD:.*]] = mpmd.fragment<mesh="mesh2", origin=[]> (%[[TRANSFER]]) (%arg2: tensor<8x2xf32>) {
  // CHECK: mpmd.return %arg2 : tensor<8x2xf32>
  // CHECK: } : (!mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>) -> !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{}, {}]>>
  // CHECK: return %[[FRAG]], %[[RESHARD]] : !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>, !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{}, {}]>>
  %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} %0 : (!mesh_1_tensor_8_2_f32) -> !mesh_2_tensor_8_2_f32
  return %0, %1 : !mesh_1_tensor_8_2_f32, !mesh_2_tensor_8_2_f32
}

}

// -----

!mesh_1_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>>
!mesh_2_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>>
#topology = #mpmd.topology<<"mesh1" : <["devices"=4]>>, <"mesh2" : <["devices"=4]>>>

module {
sdy.mesh @mesh = <["devices"=4]>

// CHECK-LABEL: func @transfer_result_less_sharded_than_operand
func.func @transfer_result_less_sharded_than_operand(
  %arg0: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>},
  %arg1: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>})
  ->
  (!mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices"}, {}]>}, !mesh_2_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
  attributes {topology = #topology}
{
// CHECK-NEXT: %[[ADD:.*]] = mpmd.fragment<mesh="mesh1", origin=["add"]> (%arg0, %arg1)
// CHECK-NEXT:   add
// CHECK-NEXT:   mpmd.return
// CHECK-NEXT: }
// CHECK-SAME: (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>,
// CHECK-SAME: !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>

// CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[ADD]]
// CHECK-SAME:  (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>)
// CHECK-SAME:  -> !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>

// CHECK-NEXT: %[[RESHARD:.*]] = mpmd.fragment<mesh="mesh2", origin=[]> (%[[TRANSFER]])
// CHECK: } : (!mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>)
// CHECK-SAME: -> !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{}, {}]>>

// CHECK-NEXT: %[[ADD]], %[[RESHARD]]

  %0 = mpmd.fragment<mesh="mesh1", origin=["add"], in_shardings=[<@mesh, [{"devices", ?}, {?}]>, <@mesh, [{"devices", ?}, {?}]>], out_shardings=[<@mesh, [{"devices", ?}, {?}]>]> (%arg0, %arg1) (%arg2: tensor<8x2xf32>, %arg3: tensor<8x2xf32>) {
    %2 = stablehlo.add %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} : tensor<8x2xf32>
    mpmd.return %2 : tensor<8x2xf32>
  } : (!mesh_1_tensor_8_2_f32, !mesh_1_tensor_8_2_f32) -> !mesh_1_tensor_8_2_f32
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_8_2_f32) -> !mesh_2_tensor_8_2_f32
  return %0, %1 : !mesh_1_tensor_8_2_f32, !mesh_2_tensor_8_2_f32
}

}

// -----

!mesh_1_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>>
!mesh_2_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>>
#topology = #mpmd.topology<<"mesh1" : <["devices"=4]>>, <"mesh2" : <["devices"=4]>>>

module {
sdy.mesh @mesh = <["devices"=4]>

// CHECK-LABEL: func @transfer_result_more_sharded_than_operand
func.func @transfer_result_more_sharded_than_operand(
  %arg0: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>},
  %arg1: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>})
  ->
  (!mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, !mesh_2_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices"}, {}]>})
  attributes {topology = #topology}
{
// CHECK-NEXT: %[[ADD:.*]] = mpmd.fragment<mesh="mesh1", origin=["add"]> (%arg0, %arg1)
// CHECK-NEXT:   add
// CHECK-NEXT:   mpmd.return
// CHECK-NEXT: }
// CHECK-SAME: (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>,
// CHECK-SAME: !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>

// CHECK-NEXT: %[[RESHARD:.*]] = mpmd.fragment<mesh="mesh1", origin=[]> (%[[ADD]])
// CHECK: } : (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>)
// CHECK-SAME: -> !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{}, {}]>>

// CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[ADD]]
// CHECK-SAME:  (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>)
// CHECK-SAME:  -> !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>, sharding=<@mesh, [{"devices"}, {}]>>

// CHECK-NEXT: %[[RESHARD]], %[[TRANSFER]]

  %0 = mpmd.fragment<mesh="mesh1", origin=["add"], in_shardings=[<@mesh, [{"devices", ?}, {?}]>, <@mesh, [{"devices", ?}, {?}]>]> (%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} (%arg2: tensor<8x2xf32>, %arg3: tensor<8x2xf32>) {
    %2 = stablehlo.add %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} : tensor<8x2xf32>
    mpmd.return %2 : tensor<8x2xf32>
  } : (!mesh_1_tensor_8_2_f32, !mesh_1_tensor_8_2_f32) -> !mesh_1_tensor_8_2_f32
  %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} %0 : (!mesh_1_tensor_8_2_f32) -> !mesh_2_tensor_8_2_f32
  return %0, %1 : !mesh_1_tensor_8_2_f32, !mesh_2_tensor_8_2_f32
}

}


// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: @for_loop_with_fragment_nested
func.func @for_loop_with_fragment_nested(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>},
   %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
    // CHECK: mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, unroll_factor = 3 : ui32}
    // CHECK-SAME: (%arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
    // CHECK-SAME: %arg3: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>, %index: tensor<ui32>)
    %0:2 = mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, unroll_factor = 3 : ui32} (
        %arg2: !mesh_1_tensor_4_8_f32, %arg3: !mesh_1_tensor_4_8_f32, %index: tensor<ui32>) {
        // CHECK: mpmd.fragment<mesh="m1", origin=["producer"]>
        // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
        // CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
        %fragment_result:2 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg2)
        (%arg4: tensor<4x8xf32>) {
          %3 = stablehlo.add %arg4, %arg4 : tensor<4x8xf32>
          mpmd.return %3, %3 : tensor<4x8xf32>, tensor<4x8xf32>
        } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
        mpmd.return %fragment_result#0, %fragment_result#1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
  } : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
  %1 = mpmd.transfer %0#0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %1, %0#1 : !mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: @for_loop_sharding_from_op_within_loop
func.func @for_loop_sharding_from_op_within_loop(
// CHECK: (%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
// CHECK-SAME: (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>)
   %arg0: !mesh_1_tensor_4_8_f32,
   %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
    // CHECK: mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, unroll_factor = 3 : ui32}
    // CHECK-SAME: (%arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
    // CHECK-SAME: %arg3: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>, %index: tensor<ui32>)
    %0:2 = mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, unroll_factor = 3 : ui32} (
        %arg2: !mesh_1_tensor_4_8_f32, %arg3: !mesh_1_tensor_4_8_f32, %index: tensor<ui32>) {
        // CHECK: mpmd.fragment<mesh="m1", origin=["producer"]>
        // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
        // CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
        %fragment_result:2 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg2)
        (%arg4: tensor<4x8xf32>) {
          %3 = stablehlo.add %arg4, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
          mpmd.return %3, %3 : tensor<4x8xf32>, tensor<4x8xf32>
        } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
        mpmd.return %fragment_result#0, %fragment_result#1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
  } : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
  %1 = mpmd.transfer %0#0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %1, %0#1 : !mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
}

// -----

#homogenous_topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>
!mesh_2_tensor_16_32_f32 = !mpmd.mesh_tensor<"m2", tensor<16x32xf32>>

module {
sdy.mesh @mesh = <["x"=8]>

// CHECK-LABEL: func @introduce_reshard_for_arg
func.func @introduce_reshard_for_arg(
  %arg0: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
  -> (!mesh_2_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {topology=#homogenous_topology} {
  // CHECK:  %[[RESHARD:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:    mpmd.return %arg1 : tensor<16x32xf32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {}]>>) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[RESHARD]] : (!mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>)
  // CHECK-SAME: -> !mpmd.mesh_tensor<"m2", tensor<16x32xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %1 = mpmd.transfer %arg0 : (!mesh_1_tensor_16_32_f32) -> !mesh_2_tensor_16_32_f32
  %2 = mpmd.fragment<mesh="m2", origin=[]> (%1) (%arg1: tensor<16x32xf32>) {
    mpmd.return %arg1 : tensor<16x32xf32>
  } : (!mesh_2_tensor_16_32_f32) -> (!mesh_2_tensor_16_32_f32)
  func.return %2 : !mesh_2_tensor_16_32_f32
}
}

// -----

#homogenous_topology = #mpmd.topology<<"m1": <["x"=8, "y"=1]>>>

!mesh_1_tensor_16_32_f32 = !mpmd.mesh_tensor<"m1", tensor<16x32xf32>>

module {
// CHECK: sdy.mesh @mesh = <["x"=8, "y"=1]>
sdy.mesh @mesh = <["x"=8, "y"=1]>

// CHECK-LABEL: func @axis_sized_one_should_be_removed_from_sharding_not_mesh
func.func @axis_sized_one_should_be_removed_from_sharding_not_mesh(
// CHECK: (%arg0: !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {"x"}]>>)
// CHECK-SAME: -> !mpmd.mesh_tensor<"m1", tensor<16x32xf32>, sharding=<@mesh, [{}, {"x"}]>>
  %arg0: !mesh_1_tensor_16_32_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}
)
  -> (!mesh_1_tensor_16_32_f32)
  attributes {topology=#homogenous_topology} {
  func.return %arg0 : !mesh_1_tensor_16_32_f32
}
}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=5]>

// CHECK-LABEL: @non_divisible_sharding_after_propagation_turns_to_replicated
func.func @non_divisible_sharding_after_propagation_turns_to_replicated(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
   %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_2_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=5]>>,
      <"m2": <["x"=5]>>
    >} {
  // CHECK: %[[FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]>
  // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{}, {}]>>)
  // CHECK-SAME: -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{}, {}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %1 : !mesh_2_tensor_4_8_f32
}
}

// -----


!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=16]>

// CHECK-LABEL: @partly_divisible_sharding_after_propagation_turns_to_subaxis_sharding
func.func @partly_divisible_sharding_after_propagation_turns_to_subaxis_sharding(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
   %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_2_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=16]>>,
      <"m2": <["x"=16]>>
    >} {
  // CHECK: %[[FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]>
  // CHECK: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{}, {}]>>) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x":(1)4}, {}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: %[[TRANSFER_RESULT:.*]] = mpmd.transfer %[[FRAGMENT_RESULT]] :
  // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x":(1)4}, {}]>>) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x":(1)4}, {}]>>
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  // CHECK:  %[[RESHARD_RESULT:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER_RESULT]])
  // CHECK: (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x":(1)4}, {}]>>) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{}, {}]>>
  func.return %1 : !mesh_2_tensor_4_8_f32
}
}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: @sharding_constraint_preserved
func.func @sharding_constraint_preserved(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> !mesh_2_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0) (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:     %[[CONSTRAINT:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{}, {"x"}]>
  // CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[CONSTRAINT]], %[[CONSTRAINT]]
  // CHECK-NEXT:     mpmd.return %[[ADD]]
  // CHECK-NEXT:   } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>)
  // CHECK-SAME:   -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{}, {"x"}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    %2 = sdy.sharding_constraint %arg1 <@mesh, [{}, {"x"}]> : tensor<4x8xf32>
    %3 = stablehlo.add %2, %2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %1 : !mesh_2_tensor_4_8_f32
}
}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: @sharding_group_and_propagation_barrier_preserved
func.func @sharding_group_and_propagation_barrier_preserved(
   %arg0: !mesh_1_tensor_4_8_f32
   {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> !mesh_2_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0) (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:     sdy.sharding_group %arg1 group_id=0
  // CHECK-NEXT:     %[[BARRIER:.*]] = sdy.propagation_barrier %arg1 allowed_direction=NONE
  // CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[BARRIER]], %[[BARRIER]]
  // CHECK-NEXT:     sdy.sharding_group %[[ADD]] group_id=0
  // CHECK-NEXT:     mpmd.return %[[ADD]]
  // CHECK-NEXT:   } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>)
  // CHECK-SAME:   -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    sdy.sharding_group %arg1 group_id=0 : tensor<4x8xf32>
    %2 = sdy.propagation_barrier %arg1 allowed_direction=NONE : tensor<4x8xf32>
    %3 = stablehlo.add %2, %2 : tensor<4x8xf32>
    sdy.sharding_group %3 group_id=0 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %1 : !mesh_2_tensor_4_8_f32
}
}

// -----
!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4xf32>>

module {
// CHECK: sdy.mesh @mesh = <["x"=2, "y"=4]>
sdy.mesh @mesh = <["x"=2, "y"=4]>
// CHECK-LABEL: func @return_value_used_in_another_fragment
func.func @return_value_used_in_another_fragment(%arg0: !mesh_1_tensor {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg1: !mesh_2_tensor) -> (!mesh_2_tensor {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, !mesh_2_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>
} {
  // CHECK: %[[TRANSFER_RESULT:.*]] = mpmd.transfer %arg0 :
  // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4xf32>, sharding=<@mesh, [{"y"}]>>)
  // CHECK-SAME: -> !mpmd.mesh_tensor<"m2", tensor<4xf32>, sharding=<@mesh, [{"y"}]>>
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor // %arg0 is sharded by y axis
  // CHECK: mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER_RESULT]]) (%arg2: tensor<4xf32>) {
  // CHECK-NEXT:     mpmd.return %arg2 : tensor<4xf32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m2", tensor<4xf32>, sharding=<@mesh, [{"y"}]>>)
  // CHECK-SAME: -> !mpmd.mesh_tensor<"m2", tensor<4xf32>, sharding=<@mesh, [{"x"}]>>
  %1 = mpmd.fragment<mesh="m2", origin=["f2"]> (%0, %arg1) (%arg2: tensor<4xf32>, %arg3: tensor<4xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor
  return %0, %1 : !mesh_2_tensor, !mesh_2_tensor // %0 is sharded by x
}
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2, "y"=2]>>,<"m2": <["x"=2, "y"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @transfer_result_used_in_fragment_and_returned_with_user_specified_sharding(
func.func @transfer_result_used_in_fragment_and_returned_with_user_specified_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) attributes {topology = #topology} {
    // CHECK: %[[TRANSFER_RESULT:.*]] = mpmd.transfer %arg0 : ({{.*}}sharding=<@mesh, [{"x"}, {}]>>) ->
    // CHECK-SAME: {{.*}}sharding=<@mesh, [{"x"}, {}]>>
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    // CHECK: %[[FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]>
    // CHECK: ({{.*}}sharding=<@mesh, [{"x"}, {}]>>) ->
    // CHECK-SAME: {{.*}}sharding=<@mesh, [{"y"}, {}]>>
    %1 = mpmd.fragment<mesh="m1", origin=["producer"]> (%0) (%arg2: tensor<4x8xf32>) {
            %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2: tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    // CHECK: return %[[TRANSFER_RESULT]], %[[FRAGMENT_RESULT]] : {{.*}}sharding=<@mesh, [{"x"}, {}]>>,
    // CHECK-SAME: {{.*}}sharding=<@mesh, [{"y"}, {}]>>
    return %0, %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2, "y"=2]>>,<"m2": <["x"=2, "y"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @fragment_result_used_in_fragment_and_returned_with_user_specified_sharding(
// CHECK: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>)
// CHECK-SAME: -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {}]>>,
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>)
func.func @fragment_result_used_in_fragment_and_returned_with_user_specified_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
  !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {topology = #topology} {
    // CHECK: %[[PRODUCER_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]>
    %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
      mpmd.return %2: tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    // CHECK: {{.*}}sharding=<@mesh, [{"x"}, {}]>>,
    // CHECK-SAME: {{.*}}sharding=<@mesh, [{"x"}, {}]>>)
    // CHECK-SAME: -> {{.*}}sharding=<@mesh, [{"y"}, {}]>>
    // CHECK: %[[CONSUMER_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["consumer"]> (%[[PRODUCER_RESULT]])
    %1 = mpmd.fragment<mesh="m1", origin=["consumer"]> (%0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2: tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    // CHECK: ({{.*}}sharding=<@mesh, [{"y"}, {}]>>) ->
    // CHECK-SAME: {{.*}}sharding=<@mesh, [{"x"}, {}]>>
    // CHECK: return %[[PRODUCER_RESULT]], %[[CONSUMER_RESULT]] :
    // CHECK-SAME: {{.*}}sharding=<@mesh, [{"y"}, {}]>>, {{.*}}sharding=<@mesh, [{"x"}, {}]>>
    return %0, %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    }
}

// -----

sdy.mesh @mesh = <["x"=4]>

// CHECK-LABEL: @simple_propagation_within_fragment_inputs_simplified
func.func public @simple_propagation_within_fragment_inputs_simplified(
  %arg0: !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
  %arg1: !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>})
  -> (!mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
  // CHECK: %[[FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="mesh1", origin=["stage1"]>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.multiply %arg2, %arg2
  // CHECK-NEXT: mpmd.return %[[RESULT]] : tensor<16x10x3xf32>
  // CHECK-NEXT: } :
  // CHECK-NEXT: return %[[FRAGMENT_RESULT]], %arg0 :
  %0:2 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg0, %arg1) (%arg2: tensor<16x10x3xf32>, %arg3: tensor<16x10x3xf32>) {
    %cst = stablehlo.constant dense<0.0> : tensor<16x10x3xf32>
    %cst_false = stablehlo.constant dense<false> : tensor<i1>
    %extra_select = stablehlo.select %cst_false, %arg2, %cst : tensor<i1>, tensor<16x10x3xf32>
    %1 = stablehlo.add %arg3, %extra_select : tensor<16x10x3xf32>
    %4 = stablehlo.multiply %arg3, %1 : (tensor<16x10x3xf32>, tensor<16x10x3xf32>) -> tensor<16x10x3xf32>
    mpmd.return %4, %arg2 : tensor<16x10x3xf32>, tensor<16x10x3xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>) -> (!mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>)
  return %0#0, %0#1 : !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
}

// -----

module {
sdy.mesh @tpu = <["tpu_dim0"=8]>
sdy.mesh @cpu = <["cpu_dim0"=1]>

// CHECK-LABEL: @heterogeneous_inter_mesh_propagation_blocked
func.func @heterogeneous_inter_mesh_propagation_blocked(
  %arg0: !mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>
  {sdy.sharding = #sdy.sharding<@tpu, [{"tpu_dim0"}, {}]>})
  -> !mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>
  attributes {topology = #mpmd.topology<<"tpu" : <["tpu_dim0"=8]>>, <"cpu" : <["cpu_dim0"=1]>>>} {
  // CHECK: mpmd.fragment<mesh="tpu", origin=["producer"]
  %0 = mpmd.fragment<mesh="tpu", origin=["producer"]> (%arg0) (%arg1: tensor<16x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    mpmd.return %3 : tensor<16x32xf32>
  } : (!mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>) -> !mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>
  // The transfer operand retains @tpu sharding but the result should NOT.
  // CHECK: mpmd.transfer %0 :
  // CHECK-SAME: sharding=<@tpu, [{"tpu_dim0"}, {}]>
  // CHECK-SAME: -> !mpmd.mesh_tensor<"cpu", tensor<16x32xf32>>
  %1 = mpmd.transfer %0 : (!mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>) -> !mpmd.mesh_tensor<"cpu", tensor<16x32xf32>>
  // CHECK: mpmd.fragment<mesh="cpu", origin=["consumer"]>
  %2 = mpmd.fragment<mesh="cpu", origin=["consumer"]> (%1) (%arg1: tensor<16x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    mpmd.return %3 : tensor<16x32xf32>
  } : (!mpmd.mesh_tensor<"cpu", tensor<16x32xf32>>) -> !mpmd.mesh_tensor<"cpu", tensor<16x32xf32>>
  %3 = mpmd.transfer %2 : (!mpmd.mesh_tensor<"cpu", tensor<16x32xf32>>) -> !mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>
  func.return %3 : !mpmd.mesh_tensor<"tpu", tensor<16x32xf32>>
}
}
