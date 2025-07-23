// RUN: mpmd_opt %s -- 2>&1 | FileCheck %s

// Just verifying no errors in parsing and printing, and giving examples of
// how the attributes are printed.

// CHECK-LABEL: func @simple_example(%arg0
func.func @simple_example(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.add %arg0, %arg0  {mpmd.use_set = #mpmd.meshes_with_origins<"m3">}: tensor<4x8xf32>
  %1 = stablehlo.add %arg0, %arg0  {mpmd.use_set = #mpmd.meshes_with_origins<"m3"["origin1","origin2"]>}: tensor<4x8xf32>
  %2 = stablehlo.add %arg0, %arg0  {mpmd.use_set = #mpmd.meshes_with_origins<>}: tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
