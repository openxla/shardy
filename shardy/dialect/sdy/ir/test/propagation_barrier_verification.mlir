// RUN: sdy_opt %s -split-input-file -verify-diagnostics

func.func @none(%arg0 : tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{cannot specify `BOTH` as the direction}}
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BOTH : tensor<8xf32>
  return %0 : tensor<8xf32>
}
