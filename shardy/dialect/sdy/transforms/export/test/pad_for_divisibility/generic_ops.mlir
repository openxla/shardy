// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

// CHECK:  sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>
sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @no_pad
func.func @no_pad(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]] : tensor<4x8xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
