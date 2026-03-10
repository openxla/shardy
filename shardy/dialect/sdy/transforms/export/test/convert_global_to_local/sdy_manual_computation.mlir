// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func @no_free_axes_two_manual_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>}) {
func.func @no_free_axes_two_manual_axes(%arg0 : tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"x"}, {"z"}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ABS]], %[[ABS]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[ADD]] : tensor<8x16xf32>
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"x"}, {"z"}]>]>} : tensor<16x32xf32>
  %1 = sdy.manual_computation(%0)
    in_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    out_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    manual_axes={"x", "z"}
    (%arg1: tensor<8x16xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>
    %3 = stablehlo.tanh %2  : tensor<8x16xf32>
    // CHECK-NEXT: return %[[TANH]] : tensor<8x16xf32>
    sdy.return %3 : tensor<8x16xf32>
  } : (tensor<16x32xf32>) -> (tensor<16x32xf32>)
  func.return %1 : tensor<16x32xf32>
}

// CHECK-LABEL: func @one_free_axis_one_manual_axis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
// CHECK-SAME: -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
func.func @one_free_axis_one_manual_axis(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
  // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y"}, {}]>]>} : tensor<2x8xf32>
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.tanh %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y"}, {}]>]>} : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: return %[[TANH]] : tensor<2x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @nested_manual_computations
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>}) {
func.func @nested_manual_computations(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_2_4_2, [{"x", "z"}, {"y"}]>]
    out_shardings=[<@mesh_2_4_2, [{"x", "z"}, {"y"}]>]
    manual_axes={"x"} (%arg1: tensor<8x32xf32>) {
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"z"}, {"y"}]>]>} : tensor<4x8xf32>
    %1 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"z"}, {"y"}]>]>}: tensor<8x32xf32>
    %2 = sdy.manual_computation(%1)
      in_shardings=[<@mesh_2_4_2, [{"z"}, {"y"}]>]
      out_shardings=[<@mesh_2_4_2, [{"z"}, {"y"}]>]
      manual_axes={"z"} (%arg2: tensor<4x32xf32>) {
      // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ABS]], %[[ABS]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{}, {"y"}]>]>} : tensor<4x8xf32>
      %3 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{}, {"y"}]>]>}: tensor<4x32xf32>
      sdy.return %3 : tensor<4x32xf32>
    } : (tensor<8x32xf32>) -> tensor<8x32xf32>
    sdy.return %2 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<4x8xf32>
  return %0 : tensor<16x32xf32>
}
