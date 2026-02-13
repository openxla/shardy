// RUN: sdy_opt %s -sdy-remove-sub-axes-in-input-output-shardings | FileCheck %s

sdy.mesh @mesh = <["x"=8, "y"=8, "z"=8]>

// This test checks that:
// 1. We remove sub-axes and the trailing axes in input and output shardings.
// 2. We do not modify the shardings for intermediate tensors.

// CHECK-LABEL: func @update_in_out_shardings
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "z"}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
func.func @update_in_out_shardings(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2}, {"x", "z"}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  // CHECK: %0 = stablehlo.add      %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2, "y"}, {"z"}]>]>} : tensor<64x64xf32>
  // CHECK: %1 = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {}]>]>} : tensor<64x64xf32>
  %0 = stablehlo.add      %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2, "y"}, {"z"}]>]>} : tensor<64x64xf32>
  %1 = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {}]>]>} : tensor<64x64xf32>
  return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
}
