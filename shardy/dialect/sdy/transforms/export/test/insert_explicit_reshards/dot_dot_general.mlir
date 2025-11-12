// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>

// CHECK-LABEL: func @dot_test
func.func @dot_test(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
