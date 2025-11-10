// RUN: mpmd_opt %s -mpmd-rule-based-schedule='rules=FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["f"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["g"],mesh_name="m1")])' -verify-diagnostics | FileCheck %s

// To make sure each test is run in isolation, we place each test case and
// scheduling rule on its own file.

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @schedule_two_fragments
func.func @schedule_two_fragments
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK: %[[F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) {call_counter = 0 : ui32} (%arg2: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg2, %arg2 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: %[[G:.*]] = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) (%arg2: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg2, %arg2 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: return %[[F]], %[[G]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %g = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %f = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  return %f, %g : !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32
}
