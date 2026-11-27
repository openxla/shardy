// RUN: mpmd_opt %s -mpmd-add-side-effect-to-avoid-cse -cse -mpmd-remove-side-effect-after-cse | FileCheck %s

// CHECK-LABEL: func @duplicate_custom_call_with_no_cse_should_be_csed
// CHECK-SAME: (%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>)
func.func @duplicate_custom_call_with_no_cse_should_be_csed(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>)  {
  // CHECK: %[[RES0:.*]] = stablehlo.custom_call @Sharding(%arg0)
  // CHECK-NOT: has_side_effect
  // CHECK-SAME: mhlo.no_cse
  // CHECK: %[[RES1:.*]] = stablehlo.custom_call @Sharding(%arg0)
  // CHECK-NOT: has_side_effect
  // CHECK-SAME: mhlo.no_cse
  // CHECK: return %[[RES0]], %[[RES1]]
  %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.no_cse} : (tensor<f32>) -> tensor<f32>
  %1 = stablehlo.custom_call @Sharding(%arg0) {mhlo.no_cse} : (tensor<f32>) -> tensor<f32>
  func.return %0, %1 : tensor<f32>, tensor<f32>
}
