// RUN: sdy_opt %s -sdy-drop-sharding-and-mesh | FileCheck %s

// CHECK-NOT: sdy.mesh
sdy.mesh @mesh_2 = <["x"=2]>

// CHECK-LABEL: func @drop_sharding
// CHECK-SAME:    %arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT:     sdy.sharding
func.func @drop_sharding(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @drop_multiple_sharding
// CHECK-SAME:    %arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
// CHECK-NOT:     sdy.sharding
func.func @drop_multiple_sharding(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
    %arg1: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) -> tensor<2x4xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

