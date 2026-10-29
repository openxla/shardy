// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @select_and_scatter_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x8xi32>)
// CHECK-SAME: -> (tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
func.func @select_and_scatter_permutation(
    %arg0: tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
    %arg1: tensor<1x8xi32>)
    -> (tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<1x16xi32>
  // CHECK: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_OP]], %[[ARG1]], %[[CST]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{}, {"b"}]> : tensor<1x16xi32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {
    window_dimensions = array<i64: 1, 2>,
    window_strides = array<i64: 1, 2>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {"b"}]>]>
  } : (tensor<1x16xi32>, tensor<1x8xi32>, tensor<i32>) -> tensor<1x16xi32>
  // CHECK: return %[[RES]] : tensor<1x16xi32>
  return %0 : tensor<1x16xi32>
}
