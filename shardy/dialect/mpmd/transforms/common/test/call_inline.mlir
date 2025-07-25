// RUN: mpmd_opt %s -mpmd-call-inline -symbol-dce -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func @test_no_inline_func_call
func.func @test_no_inline_func_call(%arg0 : !mesh_1_tensor) -> !mesh_1_tensor attributes {
  topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>>}
{
  // The mpmd-custom-inline pass must not inline call ops from other dialects.
  // CHECK-NEXT: %[[CALL:.*]] = call @f(%arg0)
  // CHECK-NEXT: return %[[CALL]]
  %0 = call @f(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

func.func @f(%arg0 : !mesh_1_tensor) -> !mesh_1_tensor
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  func.return %arg0 : !mesh_1_tensor
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func @test_no_inline_of_fragment_calls
func.func @test_no_inline_of_fragment_calls(%arg0 : !mesh_tensor) -> !mesh_tensor attributes {
  topology=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  // The mpmd-custom-inline pass must not inline fragment_calls.
  // CHECK-NEXT: %[[FC:.*]] = mpmd.fragment_call<mesh="m1", origin=[]> @f(%arg0)
  // CHECK-NEXT: return %[[FC]]
  %0 = mpmd.fragment_call<mesh="m1", origin=[]> @f(%arg0) : (!mesh_tensor) -> !mesh_tensor
  func.return %0 : !mesh_tensor
}

func.func @f(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  func.return %arg0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: module
// CHECK: func.func @test_inline_of_mpmd_calls
func.func @test_inline_of_mpmd_calls(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>>}
{
  // All mpmd.calls must be inlined. Only fragments and fragment_calls
  // will get a call_counter.
  // CHECK-NOT: mpmd.call
  // CHECK-NEXT: %[[ASSIGN0:.*]] = mpmd.assign %arg0
  // CHECK-NEXT: %[[FRAG0:.*]] = mpmd.fragment<mesh="mesh1", origin=["f"]> (%[[ASSIGN0]]) {call_counter = 0 : ui32} (%arg1: tensor<3x5xf32>) {
  // CHECK-NEXT:   stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[UNASSIGN0:.*]] = mpmd.unassign %[[FRAG0]]
  %0 = mpmd.call @f(%arg0) {call_counter = 0 : ui32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: %[[ASSIGN1:.*]] = mpmd.assign %[[UNASSIGN0]]
  // CHECK-NEXT: %[[FRAG1:.*]] = mpmd.fragment<mesh="mesh1", origin=["f"]> (%[[ASSIGN1]]) {call_counter = 1 : ui32} (%arg1: tensor<3x5xf32>) {
  // CHECK-NEXT:   stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[UNASSIGN1:.*]] = mpmd.unassign %[[FRAG1]]
  %1 = mpmd.call @f(%0) {call_counter = 1 : ui32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // This call doesn't have a call_counter attribute. When we inline it's body,
  // the inlined ops will not have the attribute either.
  // CHECK-NEXT: %[[ASSIGN2:.*]] = mpmd.assign %[[UNASSIGN1]]
  // CHECK-NEXT: %[[FRAG2:.*]] = mpmd.fragment<mesh="mesh1", origin=["f"]> (%[[ASSIGN2]]) (%arg1: tensor<3x5xf32>) {
  // CHECK-NEXT:   stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[UNASSIGN2:.*]] = mpmd.unassign %[[FRAG2]]
  %2 = mpmd.call @f(%1) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: return %[[UNASSIGN2]]
  return %2 : tensor<3x5xf32>
}

!mesh_tensor = !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>

// CHECK-NOT: func.func @f
func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>>}
{
  %0 = mpmd.assign %arg0 : (tensor<3x5xf32>) -> !mesh_tensor
  %1 = mpmd.fragment<mesh="mesh1", origin=["f"]> (%0) (%arg1: tensor<3x5xf32>) {
    %a = stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
    mpmd.return %a : tensor<3x5xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  %2 = mpmd.unassign %1 : (!mesh_tensor) -> tensor<3x5xf32>
  return %2 : tensor<3x5xf32>
}
