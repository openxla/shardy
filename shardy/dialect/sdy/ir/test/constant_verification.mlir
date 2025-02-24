// RUN: sdy_opt %s -split-input-file -verify-diagnostics

func.func @dynamic_constant() -> tensor<8x?xf32> {
  // expected-error @below {{elements literal type must have static shape}}
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x?xf32>
  func.return %0 : tensor<8x?xf32>
}

// -----

func.func @unranked_constant() -> tensor<*xf32> {
  // expected-error @below {{elements literal type must have static shape}}
  %0 = sdy.constant dense<1.000000e+00> : tensor<*xf32>
  func.return %0 : tensor<8x?xf32>
}
