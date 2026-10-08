// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func private @pad_sdy_constant
func.func private @pad_sdy_constant() -> (tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 0.000000e+00]> : tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func private @pad_stablehlo_constant
func.func private @pad_stablehlo_constant() -> (tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 0.000000e+00]> : tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func private @pad_sdy_splat_constant
func.func private @pad_sdy_splat_constant() -> (tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<1.000000e+00> : tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<1.0> : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func private @divisible_sdy_constant
func.func private @divisible_sdy_constant() -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}
