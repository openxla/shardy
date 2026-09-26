// RUN: mpmd_opt %s -mpmd-add-side-effect-to-avoid-cse | FileCheck %s

// CHECK-LABEL: func @custom_call_with_no_cse_should_add_side_effect
// CHECK-SAME: (%arg0: tensor<f32>) -> tensor<f32>
func.func @custom_call_with_no_cse_should_add_side_effect(%arg0: tensor<f32>) -> tensor<f32>  {
  // CHECK: %[[RES0:.*]] = stablehlo.custom_call @Sharding(%arg0)
  // CHECK-SAME: has_side_effect = true
  // CHECK-SAME: mhlo.no_cse
  // CHECK-SAME: : (tensor<f32>) -> tensor<f32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.no_cse} : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @custom_call_without_no_cse_should_not_add_side_effect
// CHECK-SAME: (%arg0: tensor<f32>) -> tensor<f32>
func.func @custom_call_without_no_cse_should_not_add_side_effect(%arg0: tensor<f32>) -> tensor<f32>  {
  // CHECK-NOT: has_side_effect
  // CHECK: stablehlo.custom_call @Sharding(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = stablehlo.custom_call @Sharding(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @other_op_with_no_cse_should_not_add_side_effect
// CHECK-SAME: (%arg0: tensor<f32>) -> tensor<f32>
func.func @other_op_with_no_cse_should_not_add_side_effect(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: has_side_effect
  // CHECK: stablehlo.add %arg0, %arg0 {mhlo.no_cse} : tensor<f32>
  %0 = stablehlo.add %arg0, %arg0 {mhlo.no_cse} : tensor<f32>
  func.return %0 : tensor<f32>
}
