// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>

// CHECK-LABEL: func @all_to_all_then_gather(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x7xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<8x7xf32> {
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x7xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_TO_ALL:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[PAD]] out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{}, {"x"}] %[[ALL_TO_ALL]] out_sharding=<@mesh_4, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<8x7xf32>
// CHECK-NEXT:  }
func.func @all_to_all_then_gather(%arg0: tensor<8x7xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<8x7xf32> {
  %0 = sdy.all_to_all [{"x"}: 0->1] %arg0 out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x7xf32>
  %1 = sdy.all_gather [{}, {"x"}] %0 out_sharding=<@mesh_4, [{}, {}]> : tensor<8x7xf32>
  return %1 : tensor<8x7xf32>
}

// CHECK-LABEL: func @all_to_all_internal_pad(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) -> tensor<8x7xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_TO_ALL:.*]] = sdy.all_to_all [{"x"}: 1->0] %[[TANH]] out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_TO_ALL]] [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<8x7xf32>
// CHECK-NEXT:  }
func.func @all_to_all_internal_pad(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) -> tensor<8x7xf32> {
  %0 = stablehlo.slice %arg0 [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
  %1 = stablehlo.tanh %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {"x"}]>]>} : tensor<8x7xf32>
  %2 = sdy.all_to_all [{"x"}: 1->0] %1 out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<8x7xf32>
  return %2 : tensor<8x7xf32>
}

// CHECK-LABEL: func @all_to_all_both_non_divisible(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<7x7xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : tensor<8x7xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[TANH]], %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x7xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_TO_ALL:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[PAD]] out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM_ALL_TO_ALL:.*]] = stablehlo.slice %[[ALL_TO_ALL]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{}, {"x"}] %[[TRIM_ALL_TO_ALL]] out_sharding=<@mesh_4, [{}, {}]> : tensor<7x8xf32>
// CHECK-NEXT:    %[[TRIM_ALL_GATHER:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<7x8xf32>) -> tensor<7x7xf32>
// CHECK-NEXT:    return %[[TRIM_ALL_GATHER]] : tensor<7x7xf32>
// CHECK-NEXT:  }
func.func @all_to_all_both_non_divisible(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<7x7xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:7] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x7xf32>
  %1 = stablehlo.tanh %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : tensor<7x7xf32>
  %2 = sdy.all_to_all [{"x"}: 0->1] %1 out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<7x7xf32>
  %3 = sdy.all_gather [{}, {"x"}] %2 out_sharding=<@mesh_4, [{}, {}]> : tensor<7x7xf32>
  return %3 : tensor<7x7xf32>
}

// CHECK-LABEL: func @all_to_all_both_divisible(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<8x8xf32> {
// CHECK-NEXT:    %[[ALL_TO_ALL:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[ARG0]] out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x8xf32>
// CHECK-NEXT:    return %[[ALL_TO_ALL]] : tensor<8x8xf32>
// CHECK-NEXT:  }
func.func @all_to_all_both_divisible(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  %0 = sdy.all_to_all [{"x"}: 0->1] %arg0 out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @all_to_all_multiple_params(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>}) -> tensor<7x7x7x7xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8, 0:7, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<8x8x8x8xf32>) -> tensor<8x8x7x7xf32>
// CHECK-NEXT:    %[[TANH:.*]] = stablehlo.tanh %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>]>} : tensor<8x8x7x7xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[TANH]], %[[CST]], low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<8x8x7x7xf32>, tensor<f32>) -> tensor<8x8x8x8xf32>
// CHECK-NEXT:    %[[ALL_TO_ALL:.*]] = sdy.all_to_all [{"x"}: 0->2, {"y"}: 1->3] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<8x8x8x8xf32>
// CHECK-NEXT:    %[[TRIM_ALL_TO_ALL:.*]] = stablehlo.slice %[[ALL_TO_ALL]] [0:7, 0:7, 0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {"x"}, {"y"}]>]>} : (tensor<8x8x8x8xf32>) -> tensor<7x7x8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{}, {}, {"x"}, {"y"}] %[[TRIM_ALL_TO_ALL]] out_sharding=<@mesh_4_2, [{}, {}, {}, {}]> : tensor<7x7x8x8xf32>
// CHECK-NEXT:    %[[TRIM_ALL_GATHER:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:7, 0:7, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<7x7x8x8xf32>) -> tensor<7x7x7x7xf32>
// CHECK-NEXT:    return %[[TRIM_ALL_GATHER]] : tensor<7x7x7x7xf32>
// CHECK-NEXT:  }
func.func @all_to_all_multiple_params(%arg0: tensor<8x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>}) -> tensor<7x7x7x7xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:7, 0:7, 0:7] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<8x8x8x8xf32>) -> tensor<7x7x7x7xf32>
  %1 = stablehlo.tanh %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {"y"}, {}, {}]>]>} : tensor<7x7x7x7xf32>
  %2 = sdy.all_to_all [{"x"}: 0->2, {"y"}: 1->3] %1 out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<7x7x7x7xf32>
  %3 = sdy.all_gather [{}, {}, {"x"}, {"y"}] %2 out_sharding=<@mesh_4_2, [{}, {}, {}, {}]> : tensor<7x7x7x7xf32>
  return %3 : tensor<7x7x7x7xf32>
}
