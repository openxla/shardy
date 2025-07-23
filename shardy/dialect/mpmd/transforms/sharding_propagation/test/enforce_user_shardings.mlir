// RUN: mpmd_opt %s -mpmd-enforce-user-shardings -verify-diagnostics -split-input-file | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @enforce_user_arg_sharding
// The user specified sharding for %arg0 but not for %arg1.
// The fragment should get the user specified sharding for %arg0 but keep the sharding for %arg1.
// The transfer op should get the user specified sharding.
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
func.func @enforce_user_arg_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{?}, {?}]>]>
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{}, {"x"}]>, <@mesh, [{?}, {?}]>]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %arg0 :
    // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %arg0  : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    return %1 : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  }
}


// -----
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @no_user_arg_sharding_should_keep_fragment_in_sharding
func.func @no_user_arg_sharding_should_keep_fragment_in_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0)
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2: tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}


// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>

module {
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @enforce_user_sharding_for_fragment_result
func.func @enforce_user_sharding_for_fragment_result(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{?}, {?}]>]> (%arg0)
    %0:2 = mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{}, {"x"}]>, <@mesh, [{?}, {?}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    return %0#0, %0#1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,  !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @enforce_user_sharding_for_transfer_result_if_no_original_sharding
func.func @enforce_user_sharding_for_transfer_result_if_no_original_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {topology = #topology} {
    // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %arg0 :
    // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    %1 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @enforce_user_sharding_for_transfer_result
func.func @enforce_user_sharding_for_transfer_result(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {topology = #topology} {
    // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %arg0 :
    // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
module {
  sdy.mesh @mesh = <["x"=2]>

  // CHECK-LABEL: func @no_user_result_sharding_should_keep_fragment_result_sharding
func.func @no_user_result_sharding_should_keep_fragment_result_sharding(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0)
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{"x"}, {}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2: tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @only_enforce_user_sharding_once_for_transitive_transfer
func.func @only_enforce_user_sharding_once_for_transitive_transfer(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {topology = #topology} {
    // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
    // CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
    // CHECK-SAME: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    %2 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    return %2 : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  }
}


// -----

!mesh_1_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>>
!mesh_2_tensor_8_2_f32 = !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>>
#topology = #mpmd.topology<<"mesh1" : <["devices"=4]>>, <"mesh2" : <["devices"=4]>>>


module {
  sdy.mesh @mesh = <["devices"=4]>

// CHECK-LABEL: func @transfer_operand_and_result_with_different_sharding
func.func @transfer_operand_and_result_with_different_sharding(
  %arg0: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>},
  %arg1: !mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices", ?}, {?}]>}) ->
  (!mesh_1_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"devices"}, {}]>},
  !mesh_2_tensor_8_2_f32 {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) attributes {topology = #topology} {
  %0 = mpmd.fragment<mesh="mesh1", origin=["add"], in_shardings=[<@mesh, [{"devices", ?}, {?}]>, <@mesh, [{"devices", ?}, {?}]>]> (%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} (%arg2: tensor<8x2xf32>, %arg3: tensor<8x2xf32>) {
    %2 = stablehlo.add %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} : tensor<8x2xf32>
    mpmd.return %2 : tensor<8x2xf32>
  } : (!mesh_1_tensor_8_2_f32, !mesh_1_tensor_8_2_f32) -> !mesh_1_tensor_8_2_f32
  // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} %0
  // CHECK-SAME: (!mpmd.mesh_tensor<"mesh1", tensor<8x2xf32>>) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"mesh2", tensor<8x2xf32>>
  %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"devices", ?}, {?}]>]>} %0 : (!mesh_1_tensor_8_2_f32) -> !mesh_2_tensor_8_2_f32
  return %0, %1 : !mesh_1_tensor_8_2_f32, !mesh_2_tensor_8_2_f32
}

}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=4, "y"=2]>>,<"m2": <["x"=4, "y"=2]>>>

module {
sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @enforce_user_arg_sharding_with_sub_axes
// After propagation, the sharding of %arg0 has sub-axes.
// Replace the sub-axes with empty axes.
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}
func.func @enforce_user_arg_sharding_with_sub_axes(
  // expected-warning@below {{Sub-axes sharding found for arg 0}}
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2}, {}]>},
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {}]>}) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"y"}, {}]>, <@mesh, [{}, {}]>]>
    %0 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{}, {"x"}]>, <@mesh, [{?}, {?}]>]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=4, "y"=2]>>,<"m2": <["x"=4, "y"=2]>>>

module {
sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @enforce_user_result_sharding_with_sub_axes
func.func @enforce_user_result_sharding_with_sub_axes(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
  %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{?}, {?}]>]> (%arg0)
    %0:2 = mpmd.fragment<mesh="m1", origin=["producer"], out_shardings=[<@mesh, [{}, {"x"}]>, <@mesh, [{?}, {?}]>]> (%arg0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    // expected-warning@below {{Sub-axes sharding found for result 0}}
    return %0#0, %0#1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,  !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2, "y"=2]>>,<"m2": <["x"=2, "y"=2]>>>
module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @enforce_result_sharding_on_transfer_result_used_in_fragment(
func.func @enforce_result_sharding_on_transfer_result_used_in_fragment(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) ->
  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) attributes {topology = #topology} {
    // CHECK: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %arg0 :
    // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    // CHECK: mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"x"}, {}]>]>
    %1 = mpmd.fragment<mesh="m1", origin=["producer"], in_shardings=[<@mesh, [{"y"}, {}]>]> (%0) (%arg2: tensor<4x8xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %2: tensor<4x8xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
    return %0, %1 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  }
}
