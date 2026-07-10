// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @divisible(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<7x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : tensor<8x8xf32>
// CHECK-NEXT:    %[[REDUCE_SCATTER:.*]] = sdy.reduce_scatter [{"x"}, {}] %[[TANH]] out_sharding=<@mesh_4_2, [{"y", "x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"y", "x"}, {}] %[[REDUCE_SCATTER]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x8xf32>
// CHECK-NEXT:  }
func.func @divisible(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<7x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %1 = stablehlo.tanh %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"y"}, {}]>]>} : tensor<7x8xf32>
  %2 = sdy.reduce_scatter [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{"y", "x"}, {}]> : tensor<7x8xf32>
  %3 = sdy.all_gather [{"y", "x"}, {}] %2 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %3 : tensor<7x8xf32>
}

// CHECK-LABEL: func @non_divisible(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] : tensor<7x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[TANH]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[REDUCE_SCATTER:.*]] = sdy.reduce_scatter [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[REDUCE_SCATTER]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x8xf32>
// CHECK-NEXT:  }
func.func @non_divisible(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = stablehlo.tanh %arg0 : tensor<7x8xf32>
  %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}

// Tests that we don't insert a select if the padding kind is already known (kZero).
// CHECK-LABEL: func @known_padding_no_select
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x7xf32>, %[[ARG1:.*]]: tensor<7x5xf32>
// CHECK-NOT:     stablehlo.select
// CHECK:         stablehlo.dot_general
func.func @known_padding_no_select(%arg0: tensor<7x7xf32>, %arg1: tensor<7x5xf32>) -> tensor<7x5xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x7xf32>
  %1 = sdy.reduce_scatter [{}, {"y"}] %0 out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<7x7xf32>
  %2 = sdy.all_gather [{}, {"y"}] %1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x7xf32>
  %arg1_sharded = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x5xf32>
  %3 = stablehlo.dot_general %2, %arg1_sharded, contracting_dims = [0] x [0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}, {}]>]>} : (tensor<7x7xf32>, tensor<7x5xf32>) -> tensor<7x5xf32>
  return %3 : tensor<7x5xf32>
}

// Tests that we insert a select to enforce kZero if the padding kind is unknown.
// CHECK-LABEL: func @unknown_padding_requires_select
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x7xf32>, %[[ARG1:.*]]: tensor<7x7xf32>, %[[ARG2:.*]]: tensor<7x5xf32>
// CHECK:         %[[ADD:.*]] = stablehlo.add
// CHECK:         %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[PAD:.*]] = stablehlo.pad %[[ADD]], %[[CST]]
// CHECK:         %[[REDUCE_SCATTER:.*]] = sdy.reduce_scatter
// CHECK:         %[[ALL_GATHER:.*]] = sdy.all_gather
// CHECK:         %[[SELECT:.*]] = stablehlo.select
// CHECK:         stablehlo.dot_general %[[SELECT]]
func.func @unknown_padding_requires_select(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x5xf32>) -> tensor<7x5xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x7xf32>
  %arg1_sharded = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x7xf32>
  %1 = stablehlo.add %0, %arg1_sharded {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {}]>]>} : tensor<7x7xf32>
  %2 = sdy.reduce_scatter [{}, {"y"}] %1 out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<7x7xf32>
  %3 = sdy.all_gather [{}, {"y"}] %2 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x7xf32>
  %arg2_sharded = sdy.all_slice [{"x"}, {}] %arg2 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x5xf32>
  %4 = stablehlo.dot_general %3, %arg2_sharded, contracting_dims = [0] x [0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}, {}]>]>} : (tensor<7x7xf32>, tensor<7x5xf32>) -> tensor<7x5xf32>
  return %4 : tensor<7x5xf32>
}
