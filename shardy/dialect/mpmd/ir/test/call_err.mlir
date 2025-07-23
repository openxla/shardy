// RUN: mpmd_opt %s 2>&1 -split-input-file -verify-diagnostics


func.func public @main(%arg0: tensor<i32>) -> (tensor<i32>) {
  // The function expects an integer, not a float.
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  // expected-error@+1 {{Type mismatch. Expected call operand to have type 'tensor<i32>' but got 'tensor<f32>'}}
  %1 = mpmd.call @fn(%0, %arg0) : (tensor<f32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func public @main(%arg0: tensor<i32>) -> (tensor<f32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  // expected-error@+1 {{Type mismatch. Expected call result to have type 'tensor<i32>' but got 'tensor<f32>'}}
  %1 = mpmd.call @fn(%0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<f32>
  return %1 : tensor<f32>
}

func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func public @main(%arg0: tensor<i32>) -> (tensor<f32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  // expected-error@+1 {{call_counter must be an uint32, got 'i64'}}
  %1 = mpmd.call @fn(%0, %arg0) {call_counter = 1} : (tensor<i32>, tensor<i32>) -> tensor<f32>
  return %1 : tensor<f32>
}

func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
