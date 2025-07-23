// RUN: mpmd_opt %s -mpmd-convert-sdy-shardings-to-mpmd-types -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>

module {
sdy.mesh @mesh = <["x"=2]>
// CHECK-LABEL: func @transfer_with_different_tensor_and_result_shardings
func.func @transfer_with_different_tensor_and_result_shardings(
  %arg0: !mesh_1_tensor_4_8_f32 {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) ->
  (!mesh_2_tensor_4_8_f32) attributes {topology = #topology} {
    // expected-error @+1: Transfer op has different shardings for the tensor and result, tensor sharding: #sdy.sharding<@mesh, [{}, {"x"}]>, result sharding: #sdy.sharding<@mesh, [{"x"}, {}]>
    %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
    return %0 : !mesh_2_tensor_4_8_f32
  }
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2, "y"=2]>>,<"m2": <["x"=2, "y"=2]>>>

module {
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @transfer_result_has_different_sharding_at_fragment
func.func @transfer_result_has_different_sharding_at_fragment(
  %arg0: !mesh_1_tensor_4_8_f32 {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) ->
  (!mesh_2_tensor_4_8_f32) attributes {topology = #topology} {
    // expected-error @+1: Transfer op has different shardings for the tensor and result, tensor sharding: #sdy.sharding<@mesh, [{"y"}, {}]>, result sharding: #sdy.sharding<@mesh, [{}, {"x"}]>
    %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
    %1 = mpmd.fragment<mesh="m2", origin=["f"],  in_shardings=[<@mesh, [{}, {"x"}]>]> (%0) (%arg2: tensor<4x8xf32>) {
      mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
    return %0 : !mesh_2_tensor_4_8_f32
  }
}
