// RUN: mpmd_opt %s -mpmd-rule-based-schedule='rules=FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["x"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["y"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["z"],mesh_name="m1",call_counter=0)])' -verify-diagnostics | FileCheck %s

// To make sure each test is run in isolation, we place each test case and
// scheduling rule on its own file.

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @schedule_with_missing_fragment
func.func @schedule_with_missing_fragment
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // Test case: rule specifies x->y->z but fragment y doesn't exist
  // The system should still be able to schedule x->z
  // CHECK: %[[X:.*]] = mpmd.fragment<mesh="m1", origin=["x"]> (%arg0) {call_counter = 0 : ui32} (%arg2: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg2, %arg2 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: %[[Z:.*]] = mpmd.fragment<mesh="m1", origin=["z"]> (%arg1) {call_counter = 0 : ui32} (%arg2: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg2, %arg2 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: return %[[X]], %[[Z]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %z = mpmd.fragment<mesh="m1", origin=["z"]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %x = mpmd.fragment<mesh="m1", origin=["x"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  return %x, %z : !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32
}
