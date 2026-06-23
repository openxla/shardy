// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reduce_scatter_then_gather(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<7x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : tensor<8x8xf32>
// CHECK-NEXT:    %[[REDUCE_SCATTER:.*]] = sdy.reduce_scatter [{"x"}, {}] %[[TANH]] out_sharding=<@mesh_4_2, [{"y", "x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"y", "x"}, {}] %[[REDUCE_SCATTER]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x8xf32>
// CHECK-NEXT:  }
func.func @reduce_scatter_then_gather(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<7x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %1 = stablehlo.tanh %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"y"}, {}]>]>} : tensor<7x8xf32>
  %2 = sdy.reduce_scatter [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{"y", "x"}, {}]> : tensor<7x8xf32>
  %3 = sdy.all_gather [{"y", "x"}, {}] %2 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %3 : tensor<7x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_introduce_padding(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] : tensor<7x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[TANH]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[REDUCE_SCATTER:.*]] = sdy.reduce_scatter [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[REDUCE_SCATTER]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x8xf32>
// CHECK-NEXT:  }
func.func @reduce_scatter_introduce_padding(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = stablehlo.tanh %arg0 : tensor<7x8xf32>
  %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}
