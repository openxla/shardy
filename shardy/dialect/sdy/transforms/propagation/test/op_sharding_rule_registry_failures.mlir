// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics

// CHECK-LABEL: func @unknown_custom_op
func.func @unknown_custom_op(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
  // expected-warning@+1 {{custom call @unknown_custom_op is unknown to SDY sharding rule registry}}
  %0 = stablehlo.custom_call @unknown_custom_op(%arg0, %arg1) : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}
