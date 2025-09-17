// RUN: mpmd_opt %s -mpmd-rule-based-schedule='rules=FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["f"],mesh_name="m1",call_counter=0)->FragmentInfo(origins=["g"],mesh_name="m1",call_counter=0)]),FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["i"],mesh_name="m2",call_counter=0)->FragmentInfo(origins=["j"],mesh_name="m2",call_counter=0)->FragmentInfo(origins=["k"],mesh_name="m2",call_counter=0)]),FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["q"],mesh_name="m3",call_counter=0)->FragmentInfo(origins=["p"],mesh_name="m3",call_counter=0)])' -verify-diagnostics | FileCheck %s

// To make sure each test is run in isolation, we place each test case and
// scheduling rule on its own mesh

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
!mesh_2_tensor_2_2_f32 = !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>
!mesh_3_tensor_2_2_f32 = !mpmd.mesh_tensor<"m3", tensor<2x2xf32>>

// CHECK-LABEL: func @schedule_two_fragments
func.func @schedule_two_fragments
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK: %[[F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) {call_counter = 0 : ui32} (%arg2: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg2, %arg2 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: %[[G:.*]] = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) {call_counter = 0 : ui32} (%arg2: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg2, %arg2 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  // CHECK: return %[[F]], %[[G]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %g = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %f = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  return %f, %g : !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32
}

// CHECK-LABEL: func @schedule_three_fragments
func.func @schedule_three_fragments
(%arg0: !mesh_2_tensor_2_2_f32, %arg1: !mesh_2_tensor_2_2_f32, %arg2: !mesh_2_tensor_2_2_f32)
 -> (!mesh_2_tensor_2_2_f32, !mesh_2_tensor_2_2_f32, !mesh_2_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m2" : <["x"=1]>>>} {
  // CHECK: %[[I:.*]] = mpmd.fragment<mesh="m2", origin=["i"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
  // CHECK-NEXT:   %3 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %3 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m2", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>
  // CHECK: %[[J:.*]] = mpmd.fragment<mesh="m2", origin=["j"]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
  // CHECK-NEXT:   %3 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %3 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m2", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>
  // CHECK: %[[K:.*]] = mpmd.fragment<mesh="m2", origin=["k"]> (%arg2) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
  // CHECK-NEXT:   %3 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %3 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m2", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>
  // CHECK: return %[[I]], %[[J]], %[[K]] : !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m2", tensor<2x2xf32>>
  %k = mpmd.fragment<mesh="m2", origin=["k"]> (%arg2) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_2_tensor_2_2_f32) -> !mesh_2_tensor_2_2_f32
  %j = mpmd.fragment<mesh="m2", origin=["j"]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_2_tensor_2_2_f32) -> !mesh_2_tensor_2_2_f32
  %i = mpmd.fragment<mesh="m2", origin=["i"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_2_tensor_2_2_f32) -> !mesh_2_tensor_2_2_f32
  return %i, %j, %k : !mesh_2_tensor_2_2_f32, !mesh_2_tensor_2_2_f32, !mesh_2_tensor_2_2_f32
}

// CHECK-LABEL: func @conflicting_dependency
// expected-warning@+1 {{Scheduling rule conflicts with existing dataflow dependency}}
func.func @conflicting_dependency
(%arg0: !mesh_3_tensor_2_2_f32)
 -> (!mesh_3_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m3" : <["x"=1]>>>} {
  // CHECK: %[[P:.*]] = mpmd.fragment<mesh="m3", origin=["p"]> (%arg0) {call_counter = 0 : ui32} (%arg1: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg1, %arg1 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m3", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m3", tensor<2x2xf32>>
  // CHECK: %[[Q:.*]] = mpmd.fragment<mesh="m3", origin=["q"]> (%[[P]]) {call_counter = 0 : ui32} (%arg1: tensor<2x2xf32>) {
  // CHECK-NEXT:   %{{.*}} = stablehlo.add %arg1, %arg1 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %{{.*}} : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m3", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m3", tensor<2x2xf32>>
  // CHECK: return %[[Q]] : !mpmd.mesh_tensor<"m3", tensor<2x2xf32>>
  %p = mpmd.fragment<mesh="m3", origin=["p"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_3_tensor_2_2_f32) -> !mesh_3_tensor_2_2_f32
  %q = mpmd.fragment<mesh="m3", origin=["q"]> (%p) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_3_tensor_2_2_f32) -> !mesh_3_tensor_2_2_f32
  return %q : !mesh_3_tensor_2_2_f32
}
