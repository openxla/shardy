// RUN: sdy_opt %s -split-input-file -verify-diagnostics

func.func @invalid_operand_type(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{expected the type of the 0'th block argument to match the type of the corresponding operand: 'tensor<4x2xi32>' vs 'tensor<8x2xi32>'}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<4x2xi32>) {
    %1 = stablehlo.custom_call @foo(%arg1) : (tensor<4x2xi32>) -> tensor<8x2xi32>
    sdy.return %1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @operand_count_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{number of block arguments must match the number of operands: 2 != 1}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @invalid_result_type(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{expected the type of the 0'th returned value to match the type of the corresponding result: 'tensor<4x2xi32>' vs 'tensor<8x2xi32>'}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    %1 = stablehlo.custom_call @foo(%arg1) : (tensor<8x2xi32>) -> tensor<4x2xi32>
    sdy.return %1 : tensor<4x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @result_count_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{number of returned values must match the number of results: 2 != 1}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1, %arg1 : tensor<8x2xi32>, tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
