// RUN: mpmd_opt %s -mpmd-from-unroll-to-call-counter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @only_calls_are_annotated
func.func @only_calls_are_annotated(%arg0: tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>}
{

  // CHECK-NEXT: %[[C1:.*]] = mpmd.call @f(%arg0) {call_counter = 0 : ui32} :
  %0 = mpmd.call @f(%arg0) {unroll_counter = 0 : ui32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: %[[C2:.*]] = mpmd.call @f(%[[C1]]) {call_counter = 1 : ui32} :
  %1 = mpmd.call @f(%0) {unroll_counter = 1 : ui32} : (tensor<3x5xf32>) -> tensor<3x5xf32>

  // This call doesn't have an unroll_counter, so it doesn't get a call_counter
  // either.
  // CHECK-NEXT: %[[C3:.*]] = mpmd.call @f(%[[C2]]) :
  %2 = mpmd.call @f(%1) : (tensor<3x5xf32>) -> tensor<3x5xf32>

  // Any other op that is not a call doesn't get a call_counter either, even if
  // it has an unroll_counter.
  // CHECK-NEXT: mpmd.assign {unroll_counter = 2 : ui32}
  %3 = mpmd.assign {unroll_counter = 2 : ui32} %2 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
  return %3 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
}

func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>}
{
  return %arg0 : tensor<3x5xf32>
}
