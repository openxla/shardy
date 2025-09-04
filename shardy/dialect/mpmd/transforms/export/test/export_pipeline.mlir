// RUN: mpmd_opt %s -mpmd-export-pipeline -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 0: i32}, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[CALLEE:.*]](%arg0, %arg1)
// CHECK: func.func @[[CALLEE]](%arg0: tensor<4x8xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 0 : i64}
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  func.return %0 : !mesh_1_tensor_4_8_f32
}


// -----
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 0: i32}, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[CALLEE:.*]](%arg0, %arg1)
// CHECK: func.func @[[CALLEE]](%arg0: tensor<4x8xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 0 : i64}
// Note the has_side_effect = true attribute has been dropped from the custom_call.
// CHECK: %[[FISRT_CUSTOM_CALL:.*]] = stablehlo.custom_call @Sharding(%arg0) {mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[SECOND_CUSTOM_CALL:.*]] = stablehlo.custom_call @Sharding(%arg1) {mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: return %[[FISRT_CUSTOM_CALL]], %[[SECOND_CUSTOM_CALL]] : tensor<4x8xf32>, tensor<4x8xf32>
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.custom_call @Sharding(%arg2) {has_side_effect = true, mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.custom_call @Sharding(%arg3) {has_side_effect = true, mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  func.return %0#0, %0#1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
