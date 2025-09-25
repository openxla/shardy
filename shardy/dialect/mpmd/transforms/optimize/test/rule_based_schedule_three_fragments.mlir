// RUN: mpmd_opt %s -mpmd-rule-based-schedule='rules=FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["i"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["j"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["k"],mesh_name="m1",call_counter=0)])' -verify-diagnostics | FileCheck %s

// To make sure each test is run in isolation, we place each test case and
// scheduling rule on its own file.

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @schedule_three_fragments
func.func @schedule_three_fragments
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32, %arg2: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK: %[[I:.*]] = mpmd.fragment<mesh="m1", origin=["i"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
  // CHECK-NEXT:   %3 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %3 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: %[[J:.*]] = mpmd.fragment<mesh="m1", origin=["j"]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
  // CHECK-NEXT:   %3 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %3 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: %[[K:.*]] = mpmd.fragment<mesh="m1", origin=["k"]> (%arg2) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
  // CHECK-NEXT:   %3 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %3 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: return %[[I]], %[[J]], %[[K]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %k = mpmd.fragment<mesh="m1", origin=["k"]> (%arg2) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %j = mpmd.fragment<mesh="m1", origin=["j"]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %i = mpmd.fragment<mesh="m1", origin=["i"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  return %i, %j, %k : !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32
}
