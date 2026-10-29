// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @result_indivisible
func.func @result_indivisible(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) -> tensor<3x8xf32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:4, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<3x8xf32>

  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xf32>
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<3x8xf32>

  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:3, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<4x8xf32>) -> tensor<3x8xf32>
  // CHECK-NEXT: return %[[TRIM]] : tensor<3x8xf32>
  return %1 : tensor<3x8xf32>
}

// CHECK-LABEL: func @input_indivisible
func.func @input_indivisible(%arg0: tensor<4x7xi32> )
  -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>

  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x8xi32>) -> tensor<4x6xi32>
  %1 = stablehlo.slice %0 [0:4, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>}
    : (tensor<4x7xi32>) -> tensor<4x6xi32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<4x6xi32>
  return %1 : tensor<4x6xi32>
}

// When stablehlo.slice slices the full extent of all padded dimensions with
// start = 0 and stride = 1 (here dim 0 is [0:7] on tensor<7x8xf32> sharded on
// "x"=4, while unpadded dim 1 is sliced to [0:4]), PaddingCache is preserved
// on the slice result so downstream reduce_add does not emit a select mask.
// CHECK-LABEL: func @full_dim_slice_preserves_padding_cache
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<7x8xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %[[ZERO0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[ZERO0]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:8, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:    %[[ZERO1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NOT:     stablehlo.select
// CHECK-NEXT:    %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %[[ZERO1]]) applies stablehlo.add across dimensions = [0]
// CHECK-SAME:      : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[REDUCE]] : tensor<4xf32>
// CHECK-NEXT:  }
func.func @full_dim_slice_preserves_padding_cache(%arg0: tensor<7x8xf32>) -> tensor<4xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  %1 = stablehlo.slice %0 [0:7, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<7x4xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}]>]>} : (tensor<7x4xf32>, tensor<f32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// Negative case: when stablehlo.slice does not slice the full operand extent
// along a padded dimension (here dim 0 is [0:1] on tensor<7x8xf32>, which is
// communication-free and expanded to [0:4] for divisibility), elements 1..3 of
// the expanded slice come from %arg0 rather than the zero-padded region, so
// PaddingCache must not be propagated and reduce_add must mask with select.
// CHECK-LABEL: func @partial_dim_slice_does_not_preserve_padding_cache
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<7x8xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %[[ZERO0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[ZERO0]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[ZERO1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[IOTA:.*]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<4x4xi32>
// CHECK-NEXT:    %[[LIMIT:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<4x4xi32>
// CHECK-NEXT:    %[[MASK:.*]] = stablehlo.compare LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
// CHECK-NEXT:    %[[BCAST_ZERO:.*]] = stablehlo.broadcast_in_dim %[[ZERO1]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<f32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_ZERO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK-NEXT:    %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %[[ZERO1]]) applies stablehlo.add across dimensions = [0]
// CHECK-SAME:      : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[REDUCE]] : tensor<4xf32>
// CHECK-NEXT:  }
func.func @partial_dim_slice_does_not_preserve_padding_cache(%arg0: tensor<7x8xf32>) -> tensor<4xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  %1 = stablehlo.slice %0 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<1x4xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}]>]>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// When an unpadded dimension has a stride > 1, its limit index must not be
// rewritten to start + paddedShape (which only equals the original limit when
// stride is 1). Here dim 0 is padded 7 -> 8 along "x"=4, while unpadded dim 1
// has stride = 2 and keeps its original limit [0:10:2] -> tensor<8x5xf32>.
// CHECK-LABEL: func @strided_slice_unpadded_dim
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<7x10xf32>) -> tensor<7x5xf32> {
// CHECK-NEXT:    %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[ZERO]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x10xf32>, tensor<f32>) -> tensor<8x10xf32>
// CHECK-NEXT:    %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x10xf32>
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ALL_SLICE]] [0:8, 0:10:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x10xf32>) -> tensor<8x5xf32>
// CHECK-NEXT:    %[[AG:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x5xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:7, 0:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x5xf32>) -> tensor<7x5xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x5xf32>
// CHECK-NEXT:  }
func.func @strided_slice_unpadded_dim(%arg0: tensor<7x10xf32>) -> tensor<7x5xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x10xf32>
  %1 = stablehlo.slice %0 [0:7, 0:10:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x10xf32>) -> tensor<7x5xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x5xf32>
  return %2 : tensor<7x5xf32>
}
