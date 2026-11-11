// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @reduce_window_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME: -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
func.func @reduce_window_permutation(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
    // CHECK: %[[CST:.*]] = stablehlo.constant
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // CHECK: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // CHECK: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>
  // CHECK: return %[[RES]] : tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// CHECK-LABEL: func @reduce_window_base_dilation_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME: -> (tensor<10x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @reduce_window_base_dilation_permutation(%arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<10x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{base_dilations = array<i64: 3, 1>, padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 1, 1>, window_strides = array<i64: 1, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<i32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // CHECK-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<10x8xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<10x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    base_dilations = array<i64: 3, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<4x8xi32>, tensor<i32>) -> tensor<10x8xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<10x8xi32>
  return %0 : tensor<10x8xi32>
}

// CHECK-LABEL: func @reduce_window_window_dilation_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME: -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @reduce_window_window_dilation_permutation(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // CHECK-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 2, 1>, window_dimensions = array<i64: 1, 1>, window_strides = array<i64: 1, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<i32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // CHECK-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    window_dilations = array<i64: 2, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}
