// RUN: sdy_opt %s -sdy-reshard-to-collectives | FileCheck %s

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh3d = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh2d_4x2 = <["x"=4, "y"=2]>
sdy.mesh @mesh2d_2x8 = <["x"=2, "y"=8]>

// CHECK-LABEL: func @reshard_to_all_gather_single_axis
func.func @reshard_to_all_gather_single_axis(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh2d, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh2d, [{"y"}, {}]> :  tensor<16x2xf32>
  %0 = sdy.reshard %arg0 <@mesh2d, [{"y"}, {}]> : tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}


// CHECK-LABEL: func @reshard_to_all_gather_multiple_axes
func.func @reshard_to_all_gather_multiple_axes(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh3d, [{"x", "y", "z"}, {}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_gather [{"y", "z"}, {}] %arg0 out_sharding=<@mesh3d, [{"x"}, {}]> :  tensor<16x2xf32>
  %0 = sdy.reshard %arg0 <@mesh3d, [{"x"}, {}]> : tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}


// CHECK-LABEL: func @reshard_to_all_gather_multiple_dims
func.func @reshard_to_all_gather_multiple_dims(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh3d, [{"y", "z"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_gather [{"z"}, {}] %arg0 out_sharding=<@mesh3d, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  %0 = sdy.reshard %arg0 <@mesh3d, [{"y"}, {"x"}]> : tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}


// LABEL: func @reshard_to_all_gather_with_subaxis
// func.func @reshard_to_all_gather_with_subaxis(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh2d_4x2, [{"y":(1)4}, {"x"}]>}) -> tensor<16x2xf32> {
//  // NEXT: sdy.all_gather [{"y":(4)1}, {}] %arg0 out_sharding=<@mesh2d_4x2, [{"y":(1)4}, {}]> :  tensor<16x2xf32>
//  %0 = sdy.reshard %arg0 <@mesh2d_4x2, [{"y":(1)4}, {"x"}]> :  tensor<16x2xf32>
//  return %0 : tensor<16x2xf32>
// }
