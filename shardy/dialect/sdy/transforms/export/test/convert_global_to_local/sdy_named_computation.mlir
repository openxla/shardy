// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @flat
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
func.func @flat(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RES:.*]] = sdy.named_computation<"my_comp">(%[[ARG0]])
  // CHECK-SAME: in_shardings=[<@mesh_2_4, [{"x"}, {}]>]
  // CHECK-SAME: out_shardings=[<@mesh_2_4, [{"x"}, {}]>]
  // CHECK-SAME: (%[[INNER_ARG:.*]]: tensor<8x32xf32>) {
  %0 = sdy.named_computation<"my_comp">(%arg0) in_shardings=[<@mesh_2_4, [{"x"}, {}]>] out_shardings=[<@mesh_2_4, [{"x"}, {}]>] (%arg1: tensor<16x32xf32>) {
    // CHECK-NEXT:   %[[TANH:.*]] = stablehlo.tanh %[[INNER_ARG]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} : tensor<8x32xf32>
    %1 = stablehlo.tanh %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} : tensor<16x32xf32>
    // CHECK-NEXT:   sdy.return %[[TANH]] : tensor<8x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  // CHECK-NEXT: } : (tensor<8x32xf32>) -> tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @two_nested
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
// CHECK-SAME: -> (tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
func.func @two_nested(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
  // CHECK: %[[OUTER:.*]] = sdy.named_computation<"outer">(%[[ARG0]])
  // CHECK-SAME: in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
  // CHECK-SAME: out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
  // CHECK-SAME: (%[[OUTER_ARG:.*]]: tensor<2x32xf32>) {
  %0 = sdy.named_computation<"outer">(%arg0)
      in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
      out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
      (%arg1: tensor<16x32xf32>) {
    // CHECK: %[[INNER:.*]] = sdy.named_computation<"inner">(%[[OUTER_ARG]])
    // CHECK-SAME: in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    // CHECK-SAME: out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    // CHECK-SAME: (%[[INNER_ARG:.*]]: tensor<2x32xf32>) {
    %1 = sdy.named_computation<"inner">(%arg1)
        in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
        out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
        (%arg2: tensor<16x32xf32>) {
      // CHECK: %[[TANH:.*]] = stablehlo.tanh %[[INNER_ARG]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x", "y"}, {}]>]>} : tensor<2x32xf32>
      %2 = stablehlo.tanh %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x", "y"}, {}]>]>} : tensor<16x32xf32>
      // CHECK: sdy.return %[[TANH]] : tensor<2x32xf32>
      sdy.return %2 : tensor<16x32xf32>
    } : (tensor<16x32xf32>) -> tensor<16x32xf32>
    // CHECK: sdy.return %[[INNER]] : tensor<2x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>

  // CHECK: return %[[OUTER]] : tensor<2x32xf32>
  return %0 : tensor<16x32xf32>
}
