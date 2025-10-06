// RUN: sdy_opt %s -sdy-reshard-to-collectives='keep-redundant-reshards=true' | FileCheck %s

sdy.mesh @mesh2d = <["x"=2, "y"=2]>

// CHECK-LABEL: func @redundant_reshard
func.func @redundant_reshard(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2d, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh2d, [{"x"}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh2d, [{"x"}, {"y"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @redundant_reshard_on_fully_replicated
func.func @redundant_reshard_on_fully_replicated(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2d, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh2d, [{}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh2d, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @non_redundant_reshard
func.func @non_redundant_reshard(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2d, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh2d, [{"y"}, {"x"}]>
  // CHECK-NEXT: return %[[CP]]
  %0 = sdy.reshard %arg0 <@mesh2d, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
