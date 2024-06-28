// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @constant
func.func @constant() {
  // CHECK-NEXT: sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  func.return
}

// CHECK-LABEL: func @constant_different_types
func.func @constant_different_types() {
  // CHECK-NEXT: stablehlo.constant() <{value = dense<[1, 512, 4]> : tensor<3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>
  %0 = stablehlo.constant() {value = dense<[1, 512, 4]> : tensor<3xi32>} : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>
  func.return
}
