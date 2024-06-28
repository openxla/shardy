// RUN: sdy_opt %s -sdy-import-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = call @add_matmul_to_lhs(%0, %arg0) : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-NOT: @add_matmul_to_lhs
func.func private @add_matmul_to_lhs(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
