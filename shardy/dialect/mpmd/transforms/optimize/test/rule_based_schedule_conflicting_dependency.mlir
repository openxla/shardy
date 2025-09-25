// RUN: mpmd_opt %s -mpmd-rule-based-schedule='rules=FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["q"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["p"],mesh_name="m1",call_counter=0)])' -verify-diagnostics | FileCheck %s

// To make sure each test is run in isolation, we place each test case and
// scheduling rule on its own file.

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @conflicting_dependency
// expected-warning@+1 {{Scheduling rule conflicts with existing dataflow dependency}}
func.func @conflicting_dependency
(%arg0: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK: %[[P:.*]] = mpmd.fragment<mesh="m1", origin=["p"]> (%arg0) {call_counter = 0 : ui32} (%arg1: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg1, %arg1 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: %[[Q:.*]] = mpmd.fragment<mesh="m1", origin=["q"]> (%[[P]]) {call_counter = 0 : ui32} (%arg1: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg1, %arg1 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: return %[[Q]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %p = mpmd.fragment<mesh="m1", origin=["p"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %q = mpmd.fragment<mesh="m1", origin=["q"]> (%p) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  return %q : !mesh_1_tensor_2_2_f32
}
