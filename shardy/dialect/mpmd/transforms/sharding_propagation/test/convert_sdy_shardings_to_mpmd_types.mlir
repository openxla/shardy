// RUN: mpmd_opt %s -mpmd-convert-sdy-shardings-to-mpmd-types 2>&1 | FileCheck -implicit-check-not sdy.sharding -implicit-check-not in_shardings %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @fragment_with_input_and_result_shardings
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
func.func @fragment_with_input_and_result_shardings(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x", ?}, {?}]>}) attributes {topology = #topology} {
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>], out_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }

// CHECK-LABEL: func @fragment_with_only_input_sharding
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
func.func @fragment_with_only_input_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }


// CHECK-LABEL: func @fragment_with_only_result_shardings
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
func.func @fragment_with_only_result_shardings(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x", ?}, {?}]>}) attributes {topology = #topology} {
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }

// CHECK-LABEL: func @fragment_without_input_or_result_shardings
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
func.func @fragment_without_input_or_result_shardings(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }

// CHECK-LABEL: func @transfer_has_no_sharding
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
func.func @transfer_has_no_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  }

  // CHECK-LABEL: func @single_transfer_has_sharding
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
func.func @single_transfer_has_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.transfer %arg0
    // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
    %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  }

// CHECK-LABEL: func @fragment_with_transfer_fragment_result_has_multiple_uses
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
// CHECK-SAME: -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
// CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
func.func @fragment_with_transfer_fragment_result_has_multiple_uses(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
   !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x", ?}, {?}]>}) attributes {topology = #topology} {
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    // CHECK: mpmd.transfer %0
    // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>
    %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    return %0, %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  }

// CHECK-LABEL: func @for_loop_with_fragment_nested
func.func @for_loop_with_fragment_nested(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>} {
  // CHECK: mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, unroll_factor = 3 : ui32}
  // CHECK-SAME: (%arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>,
  // CHECK-SAME: %arg3: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {}]>>, %index: tensor<ui32>)
  %0:2 = mpmd.for (%arg0, %arg1) {iterations = 12 : ui32, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]>, unroll_factor = 3 : ui32} (%arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg3: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %index: tensor<ui32>) {
    %2:2 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>], out_shardings = [<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]> (%arg2) (%arg4: tensor<4x8xf32>) {
      %3 = stablehlo.add %arg4, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
      mpmd.return %3, %3 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    mpmd.return %2#0, %2#1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  } : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %0#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  return %1, %0#1 : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

}
