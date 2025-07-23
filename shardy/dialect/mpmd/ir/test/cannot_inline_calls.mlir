// RUN: mpmd_opt %s -inline -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func @test_inline_func_call
func.func @test_inline_func_call(%arg0 : !mesh_1_tensor) -> !mesh_1_tensor attributes {
  topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>>}
{
  // A call from the func dialect can still be inlined, even if the function
  // contains fragment_calls. I.e., we cannot inline a fragment call, but we can
  // inline a call of a function that includes a fragment_call (or any MPMD op).
  // CHECK-NEXT: mpmd.fragment_call<mesh="m1", origin=[]> @f(%arg0)
  // CHECK-NOT: call @func_to_inline(%arg0)
  %0 = call @func_to_inline(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

func.func @func_to_inline(%arg0 : !mesh_1_tensor) -> !mesh_1_tensor attributes {
  topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>>}
{
  %0 = mpmd.fragment_call<mesh="m1", origin=[]> @f(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

func.func @f(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  func.return %arg0 : tensor<4x8xf32>
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func @test_no_inline_of_fragment_calls
func.func @test_no_inline_of_fragment_calls(%arg0 : !mesh_tensor) -> !mesh_tensor attributes {
  topology=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  // fragment_call ops cannot be inlined.
  // CHECK-NEXT: mpmd.fragment_call<mesh="m1", origin=[]> @fragment(%arg0)
  %0 = mpmd.fragment_call<mesh="m1", origin=[]> @fragment(%arg0) : (!mesh_tensor) -> !mesh_tensor
  func.return %0 : !mesh_tensor
}

func.func @fragment(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  func.return %arg0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: module
// CHECK: func.func @test_no_inline_of_mpmd_calls
func.func @test_no_inline_of_mpmd_calls(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["y"=1]>>>}
{
  // mpmd.call ops cannot be inlined.
  // CHECK-NEXT: %[[C1:.*]] = mpmd.call @f(%arg0) {call_counter = 0 : ui32}
  // CHECK-NEXT: %[[C2:.*]] = mpmd.call @f(%[[C1]]) {call_counter = 1 : ui32}
  // CHECK-NEXT: return %[[C2]]
  %0 = mpmd.call @f(%arg0) {call_counter = 0 : ui32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %1 = mpmd.call @f(%0) {call_counter = 1 : ui32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %1 : tensor<3x5xf32>
}

func.func @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  return %arg0 : tensor<3x5xf32>
}

// -----

// CHECK-LABEL: module
// CHECK: func.func @test_assign_op_is_inlined
func.func @test_assign_op_is_inlined(%arg0: tensor<4x8xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["y"=1]>>>}
{
  // An AssignOp can always be inlined.
  // CHECK-NEXT: mpmd.assign %arg
  // CHECK-NOT: call @assign_fn
  // CHECK-NEXT: return
  %0 = call @assign_fn(%arg0) : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
}

func.func @assign_fn(%arg0: tensor<4x8xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["y"=1]>>>}
{
  %0 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
}
