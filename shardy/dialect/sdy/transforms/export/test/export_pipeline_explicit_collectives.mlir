// RUN: sdy_opt %s -sdy-export-pipeline='enable-insert-explicit-collectives=true' 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @reduce_scatter_fusion
func.func @reduce_scatter_fusion(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %2 = sdy.all_slice [{}, {"y"}] %1 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %2 : tensor<16x8xf32>
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{"x"}, {"y"}]> : tensor<16x8xf32>
  return %2 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_all_gather
func.func @all_slice_all_gather(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: %1 = sdy.all_gather [{"y"}, {}] %0 out_sharding=<@mesh, [{}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: return %1 : tensor<16x2xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"x"}]> : tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @reshard_of_reshard
func.func @reshard_of_reshard(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: return %0 : tensor<16x2xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"x"}]> : tensor<16x2xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  return %1 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_to_all_fusion
func.func @all_to_all_fusion(%arg0 : tensor<64x16x8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>}) -> (tensor<64x16x8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}, {"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = sdy.all_to_all [{"x"}: 0->2, {"y"}: 1->3] %arg0 out_sharding=<@mesh, [{}, {}, {"x"}, {"y"}]> : tensor<64x16x8x8xf32>
  // CHECK-NEXT: return %0 : tensor<64x16x8x8xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {"x"}, {}]> : tensor<64x16x8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {"x"}, {"y"}]> : tensor<64x16x8x8xf32>
  return %1 : tensor<64x16x8x8xf32>
}

// TODO: Add more tests.
