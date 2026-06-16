// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>

// CHECK-LABEL: func @all_to_all_then_gather
func.func @all_to_all_then_gather(%arg0: tensor<8x7xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<8x7xf32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x7xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[PAD]] out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{}, {"x"}] %[[A2A]] out_sharding=<@mesh_4, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
  // CHECK-NEXT: return %[[TRIM]] : tensor<8x7xf32>
  %0 = sdy.all_to_all [{"x"}: 0->1] %arg0 out_sharding=<@mesh_4, [{}, {"x"}]> : tensor<8x7xf32>
  %1 = sdy.all_gather [{}, {"x"}] %0 out_sharding=<@mesh_4, [{}, {}]> : tensor<8x7xf32>
  return %1 : tensor<8x7xf32>
}

// CHECK-LABEL: func @all_to_all_internal_pad
func.func @all_to_all_internal_pad(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) -> tensor<8x7xf32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[SLICE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[A2A:.*]] = sdy.all_to_all [{"x"}: 1->0] %[[TANH]] out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[A2A]] [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
  // CHECK-NEXT: return %[[TRIM]] : tensor<8x7xf32>
  %0 = stablehlo.slice %arg0 [0:8, 0:7] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<8x7xf32>
  %1 = stablehlo.tanh %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {"x"}]>]>} : tensor<8x7xf32>
  %2 = sdy.all_to_all [{"x"}: 1->0] %1 out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<8x7xf32>
  return %2 : tensor<8x7xf32>
}
