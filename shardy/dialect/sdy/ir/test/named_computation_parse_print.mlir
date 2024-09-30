// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @one_input_output
func.func @one_input_output(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   sdy.return %arg1 : tensor<8x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %0 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}


// CHECK-LABEL: func @two_inputs_outputs
func.func @two_inputs_outputs(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %0:2 = sdy.named_computation<"named_computation">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
  //CHECK-NEXT:   sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %0:2 = sdy.named_computation<"named_computation">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}
