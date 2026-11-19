// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=false' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>

// CHECK-LABEL: func @reshard_cross_mesh_tiled_preserved_when_false
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
func.func @reshard_cross_mesh_tiled_preserved_when_false(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}]>}) {
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[ARG0]] <@mesh_xt, [{"t"}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_xt, [{"t"}]> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @reshard_cross_mesh_replicated_preserved_when_false
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
func.func @reshard_cross_mesh_replicated_preserved_when_false(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>}) {
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[ARG0]] <@mesh_xt, [{}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_xt, [{}]> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @add_cross_mesh_replicated_not_pivoted_when_false
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>})
func.func @add_cross_mesh_replicated_not_pivoted_when_false(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
    %arg1: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{}]>]>} : tensor<24xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<24xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{}]>]>} : tensor<24xf32>
  return %0 : tensor<24xf32>
}
