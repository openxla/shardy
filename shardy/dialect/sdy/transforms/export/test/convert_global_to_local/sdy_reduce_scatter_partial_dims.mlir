// RUN: sdy_opt %s -sdy-convert-global-to-local='combine-multi-dimension-reduce-scatter=true' | FileCheck %s

sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// Regression test: a reduce-scatter that scatters two dimensions of a rank-3
// tensor while leaving the third unscattered. The combining lowering previously
// fell through the "not scattered" branch (missing `continue`) and read past the
// end of the scattering-factors vector for the unscattered dimension. Verify the
// pass lowers it to a single reduce-scatter without crashing.
// CHECK-LABEL: func @reduce_scatter_partial_multi_dim
func.func @reduce_scatter_partial_multi_dim(
    %arg0: tensor<16x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(1)2}, {}, {}]>})
    -> (tensor<16x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "z"}, {"x"}, {}]>}) {
  // CHECK: stablehlo.reduce_scatter
  %0 = sdy.reduce_scatter [{"y":(2)2, "z"}, {"x"}, {}] %arg0
      out_sharding=<@mesh_2_4_2, [{"y", "z"}, {"x"}, {}]> : tensor<16x8x4xf32>
  return %0 : tensor<16x8x4xf32>
}
