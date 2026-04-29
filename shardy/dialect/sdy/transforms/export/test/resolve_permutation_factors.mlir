// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false" | FileCheck %s --check-prefixes=CHECK,REPL

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @reduce_window_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME: -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
func.func @reduce_window_permutation(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
    // CHECK: %[[CST:.*]] = stablehlo.constant
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // REPL: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>
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
  // REPL: return %[[RES]] : tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// CHECK-LABEL: func @reverse_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
// CHECK-SAME: -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @reverse_permutation(
  %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  // REPL: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<16xf32>
  // REPL: %[[REV:.*]] = stablehlo.reverse %[[RESHARD_IN]], dims = [0]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[REV]] <@mesh, [{"a"}]> : tensor<16xf32>
  %0 = stablehlo.reverse %arg0, dims = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
    : tensor<16xf32>
    //CHECK: return %[[RES]] : tensor<16xf32>
  return %0 : tensor<16xf32>
}


// CHECK-LABEL: func @convolution_spatial_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %arg1: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_spatial_permutation(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
  // REPL: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>
  // REPL: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x14x14xf32>
   // CHECK: return %[[RES]] : tensor<1x1x14x14xf32>
  return %0 : tensor<1x1x14x14xf32>
}

// CHECK-LABEL: func @select_and_scatter_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x8xi32>)
// CHECK-SAME: -> (tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
func.func @select_and_scatter_permutation(
    %arg0: tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
    %arg1: tensor<1x8xi32>)
    -> (tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
  // REPL: %[[CST:.*]] = stablehlo.constant
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // REPL: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<1x16xi32>
  // REPL: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_OP]], %[[ARG1]], %[[CST]])
  // REPL: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{}, {"b"}]> : tensor<1x16xi32>
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
