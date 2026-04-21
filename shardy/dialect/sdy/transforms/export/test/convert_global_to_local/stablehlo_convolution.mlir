// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// ([i, jk, mn, o], [k, n, o, p])->([i, j, m, p])
// {i=2, j=112, k=2, m=112, n=2, o=3, p=64}
// reduction={k, n, o} permutation={j, m}>
//
// CHECK-LABEL: func @shard_batch
// CHECK-SAME: (%arg0: tensor<1x224x224x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>},
// CHECK-SAME: %arg1: tensor<3x3x3x64xf32>)
// CHECK-SAME: -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>})
func.func @shard_batch(
  %arg0: tensor<2x224x224x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>},
  %arg1: tensor<3x3x3x64xf32>)
  -> (tensor<2x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>}) {

  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<1x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x112x112x64xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}, {}]>]>
    } : (tensor<2x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<2x112x112x64xf32>

  // CHECK: return %[[CONV]] : tensor<1x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// ([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip])
// {i=2, j=1, k=112, l=2, m=112, n=2, o=3, p=32}
// reduction={l, n, o} permutation={k, m}
//
// CHECK-LABEL: func @shard_batch_group
// CHECK-SAME: (%arg0: tensor<1x224x224x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>},
// CHECK-SAME:  %arg1: tensor<3x3x3x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
// CHECK-SAME:  -> (tensor<1x112x112x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
func.func @shard_batch_group(
  %arg0: tensor<2x224x224x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}, {}]>},
  %arg1: tensor<3x3x3x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
  -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>}) {

  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 2 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>]>
    } : (tensor<2x224x224x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x112x112x64xf32>

  // CHECK: return %[[CONV]] : tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x64xf32>
}

// ([i, jk, lm, n], [k, m, n, o])->([i, j, l, o])
// {i=2, j=112, k=2, l=112, m=2, n=4, o=64}
// reduction={k, m, n} permutation={j, l}

// CHECK-LABEL: func @shard_feature
// CHECK-SAME: (%arg0: tensor<2x224x224x4xf32>,
// CHECK-SAME:  %arg1: tensor<3x3x4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
// CHECK-SAME: -> (tensor<2x112x112x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
func.func @shard_feature(
  %arg0: tensor<2x224x224x4xf32>,
  %arg1: tensor<3x3x4x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
  -> (tensor<2x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>}) {

  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<2x224x224x4xf32>, tensor<3x3x4x32xf32>) -> tensor<2x112x112x32xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}, {}, {"x"}]>]>
    } : (tensor<2x224x224x4xf32>, tensor<3x3x4x64xf32>) -> tensor<2x112x112x64xf32>

  // CHECK: return %[[CONV]] : tensor<2x112x112x32xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// ([i, jk, lm, no], [k, m, o, np])->([i, j, l, np])
// {i=2, j=112, k=2, l=112, m=2, n=2, o=2, p=32}
// reduction={k, m, o} permutation={j, l}
//
// CHECK-LABEL: func @shard_feature_group
// CHECK-SAME: (%arg0: tensor<2x224x224x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>},
// CHECK-SAME:  %arg1: tensor<3x3x2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
// CHECK-SAME: -> (tensor<2x112x112x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
func.func @shard_feature_group(
  %arg0: tensor<2x224x224x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>},
  %arg1: tensor<3x3x2x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>})
  -> (tensor<2x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>}) {

  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<2x224x224x2xf32>, tensor<3x3x2x32xf32>) -> tensor<2x112x112x32xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 2 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{}, {}, {}, {"x"}]>]>
    } : (tensor<2x224x224x4xf32>, tensor<3x3x2x64xf32>) -> tensor<2x112x112x64xf32>

  // CHECK: return %[[CONV]] : tensor<2x112x112x32xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// ([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip])
// {i=2, j=1, k=112, l=2, m=112, n=2, o=4, p=32}
// reduction={l, n, o} permutation={k, m}>

// CHECK-LABEL: func @shard__reduction_factors
// CHECK-SAME: (%arg0: tensor<2x224x224x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"y":(2)2}]>},
// CHECK-SAME:  %arg1: tensor<1x1x2x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}]>})
// CHECK-SAME: -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {}]>})
func.func @shard__reduction_factors(
  %arg0: tensor<2x224x224x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {"y":(2)2}]>},
  %arg1: tensor<2x2x4x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(1)2}, {"y":(2)2}, {}]>})
    -> (tensor<1x112x112x64xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}, {}, {}]>}) {

  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
  // CHECK-SAME: {batch_group_count = 2 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<2x224x224x2xf32>, tensor<1x1x2x64xf32>) -> tensor<1x112x112x64xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 2 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}, {}, {}]>]>
    } : (tensor<2x224x224x4xf32>, tensor<2x2x4x64xf32>) -> tensor<1x112x112x64xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[CONV]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
  // CHECK: (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %1 = sdy.all_reduce {"x", "y"} %0 out_sharding=<@mesh_2_4, [{}, {}, {}, {}]> : tensor<1x112x112x64xf32>

  // CHECK: return %[[RES]] : tensor<1x112x112x64xf32>
  return %1 : tensor<1x112x112x64xf32>
}
