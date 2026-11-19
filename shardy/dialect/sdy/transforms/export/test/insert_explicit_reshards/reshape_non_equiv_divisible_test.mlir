// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reshape_overflow_axes_non_equivalent_factors
// CHECK-SAME:      (%[[ARG0:.*]]: tensor<3x32x160xf32>
func.func @reshape_overflow_axes_non_equivalent_factors(
    %arg0: tensor<3x32x160xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>})
    -> (tensor<2x7680xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK:      %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}]> : tensor<3x32x160xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x32x160xf32>) -> tensor<2x7680xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<2x7680xf32>
  %0 = stablehlo.reshape %arg0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  } : (tensor<3x32x160xf32>) -> tensor<2x7680xf32>
  return %0 : tensor<2x7680xf32>
}

// CHECK-LABEL: func @reshape_overflow_axes_preserves_trailing_batching_factor
// CHECK-SAME:      (%[[ARG0:.*]]: tensor<2x15x32xf32>
func.func @reshape_overflow_axes_preserves_trailing_batching_factor(
    %arg0: tensor<2x15x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>})
    -> (tensor<3x10x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
  // CHECK:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {"x", "y"}]> : tensor<2x15x32xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"x", "y"}]>]>} : (tensor<2x15x32xf32>) -> tensor<3x10x32xf32>
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {}, {}]> : tensor<3x10x32xf32>
  // CHECK-NEXT: return %[[RESHARD_OUT]] : tensor<3x10x32xf32>
  %0 = stablehlo.reshape %arg0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>
  } : (tensor<2x15x32xf32>) -> tensor<3x10x32xf32>
  return %0 : tensor<3x10x32xf32>
}
