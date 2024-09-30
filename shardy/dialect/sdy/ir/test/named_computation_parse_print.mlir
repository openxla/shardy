// RUN: sdy_opt %s 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

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

// CHECK-LABEL: func @in_out_shardings
func.func @in_out_shardings(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   sdy.return %arg1 : tensor<8x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}


// CHECK-LABEL: func @token_type_with_sharding
func.func @token_type_with_sharding(%arg0: tensor<8x2xi32>, %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
  // CHECK-NEXT: %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, []>] (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
  // CHECK-NEXT:   sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  // CHECK-NEXT: } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, []>] (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  return %0#0, %0#1 : tensor<8x2xi32>, !stablehlo.token
}
