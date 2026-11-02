// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @convolution_spatial_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_spatial_permutation(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x14x14xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {}, {"a"}, {}]>, <@mesh, [{?}, {?}, {?}, {?}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {}]>] manual_axes={"a"} (%arg2: tensor<1x1x8x16xf32>, %arg3: tensor<3x3x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 7:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 0:1, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg2, %[[CP2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x10x16xf32>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ID_IN_GRP]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[C_ZERO]], %[[C_ZERO]], %[[ADD]], %[[C_ZERO]], sizes = [1, 1, 9, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x9x16xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x9x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x7x14xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<1x1x7x14xf32>
  // HALO-NEXT: } : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>, tensor<f32>) -> tensor<1x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x14x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x14x14xf32>
  return %0 : tensor<1x1x14x14xf32>
}

// CHECK-LABEL: func @convolution_rhs_spatial_permutation_lhs_size_one
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16x1xf32>, %[[ARG1:.*]]: tensor<16x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<8x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
func.func @convolution_rhs_spatial_permutation_lhs_size_one(
    %arg0: tensor<8x16x1xf32>,
    %arg1: tensor<16x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
    -> (tensor<8x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {}]> : tensor<16x10x4xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[RESHARD_RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, f, 0]x[i, o, 0]->[b, f, 0], window = {stride = [1], pad = [[3, 3]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<8x16x1xf32>, tensor<16x10x4xf32>) -> tensor<8x10x4xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}]> : tensor<8x10x4xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x10x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0] x [i, o, 0] -> [b, f, 0],
    window = {stride = [1], pad = [[3, 3]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}]>]>
    }
    : (tensor<8x16x1xf32>, tensor<16x10x4xf32>) -> tensor<8x10x4xf32>
  return %0 : tensor<8x10x4xf32>
}

// CHECK-LABEL: func @convolution_batch_group_count_per_group_size_one_kept
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {}, {}]>}, %[[ARG1:.*]]: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
func.func @convolution_batch_group_count_per_group_size_one_kept(
    %arg0: tensor<3x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 3 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>} : (tensor<3x1x1x2xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x1x3xf32>
  // CHECK-NEXT: return %[[CONV]] : tensor<1x1x1x3xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 3 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>
    }
    : (tensor<3x1x1x2xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x1x3xf32>
  return %0 : tensor<1x1x1x3xf32>
}

// CHECK-LABEL: func @convolution_feature_group_count_per_group_size_one_kept
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}, %[[ARG1:.*]]: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
func.func @convolution_feature_group_count_per_group_size_one_kept(
    %arg0: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>},
    %arg1: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>} : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
  // CHECK-NEXT: return %[[CONV]] : tensor<1x1x1x3xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 3 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>
    }
    : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
  return %0 : tensor<1x1x1x3xf32>
}

// CHECK-LABEL: func @convolution_batch_group_count_per_group_batch_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {}, {}]>}, %[[ARG1:.*]]: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<2x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
func.func @convolution_batch_group_count_per_group_batch_replicated(
    %arg0: tensor<6x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<2x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHARD_LHS:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {}, {}, {}]> : tensor<6x1x1x2xf32>
  // CHECK-NEXT: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x2x3xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 3 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {}]>]>} : (tensor<6x1x1x2xf32>, tensor<1x1x2x3xf32>) -> tensor<2x1x1x3xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{}, {}, {}, {"a"}]> : tensor<2x1x1x3xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x1x1x3xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 3 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>
    }
    : (tensor<6x1x1x2xf32>, tensor<1x1x2x3xf32>) -> tensor<2x1x1x3xf32>
  return %0 : tensor<2x1x1x3xf32>
}

// CHECK-LABEL: func @convolution_feature_group_count_per_group_feature_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x1x6xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}, %[[ARG1:.*]]: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
func.func @convolution_feature_group_count_per_group_feature_replicated(
    %arg0: tensor<1x1x1x6xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHARD_LHS:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x1x6xf32>
  // CHECK-NEXT: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x2x3xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x6xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x1x3xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{}, {}, {}, {"a"}]> : tensor<1x1x1x3xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x1x3xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 3 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>
    }
    : (tensor<1x1x1x6xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x1x3xf32>
  return %0 : tensor<1x1x1x3xf32>
}

// CHECK-LABEL: func @convolution_feature_group_count_per_group_reduction_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a", "b"}]>}, %[[ARG1:.*]]: tensor<1x1x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}, {"a"}]>})
// CHECK-SAME: -> (tensor<1x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
func.func @convolution_feature_group_count_per_group_reduction_replicated(
    %arg0: tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a", "b"}]>},
    %arg1: tensor<1x1x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}, {"a"}]>})
    -> (tensor<1x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHARD_LHS:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {"a"}]> : tensor<1x1x1x4xf32>
  // CHECK-NEXT: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {}, {"a"}]> : tensor<1x1x2x2xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 2 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"a"}]>]>} : (tensor<1x1x1x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x1x2xf32>
  // CHECK-NEXT: return %[[CONV]] : tensor<1x1x1x2xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"a"}], unreduced={"b"}>]>
    }
    : (tensor<1x1x1x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x1x2xf32>
  %1 = sdy.all_reduce {"b"} %0 out_sharding=<@mesh, [{}, {}, {}, {"a"}]> : tensor<1x1x1x2xf32>
  return %1 : tensor<1x1x1x2xf32>
}

// CHECK-LABEL: func @convolution_batch_group_count_divisible_per_group_kept
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>}, %[[ARG1:.*]]: tensor<1x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<2x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
func.func @convolution_batch_group_count_divisible_per_group_kept(
    %arg0: tensor<4x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<1x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
    -> (tensor<2x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 2 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"a"}]>]>} : (tensor<4x1x1x2xf32>, tensor<1x1x2x4xf32>) -> tensor<2x1x1x4xf32>
  // CHECK-NEXT: return %[[CONV]] : tensor<2x1x1x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 2 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"a"}]>]>
    }
    : (tensor<4x1x1x2xf32>, tensor<1x1x2x4xf32>) -> tensor<2x1x1x4xf32>
  return %0 : tensor<2x1x1x4xf32>
}

// CHECK-LABEL: func @convolution_feature_group_count_divisible_per_group_kept
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>}, %[[ARG1:.*]]: tensor<1x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
func.func @convolution_feature_group_count_divisible_per_group_kept(
    %arg0: tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>},
    %arg1: tensor<1x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 2 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"a"}]>]>} : (tensor<1x1x1x4xf32>, tensor<1x1x2x4xf32>) -> tensor<1x1x1x4xf32>
  // CHECK-NEXT: return %[[CONV]] : tensor<1x1x1x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"a"}]>]>
    }
    : (tensor<1x1x1x4xf32>, tensor<1x1x2x4xf32>) -> tensor<1x1x1x4xf32>
  return %0 : tensor<1x1x1x4xf32>
}

// CHECK-LABEL: func @convolution_batch_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<2x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>})
func.func @convolution_batch_sharded(
    %arg0: tensor<2x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<2x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) {
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}, {}, {}]>]>} : (tensor<2x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<2x1x14x14xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {}, {}]> : tensor<2x1x14x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}, {}, {}]>]>
    }
    : (tensor<2x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<2x1x14x14xf32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {}, {}]> : tensor<2x1x14x14xf32>
  return %1 : tensor<2x1x14x14xf32>
}

// CHECK-LABEL: func @convolution_features_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>}, %[[ARG1:.*]]: tensor<3x3x2x2xf32>)
// CHECK-SAME: -> (tensor<1x2x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>})
func.func @convolution_features_sharded(
    %arg0: tensor<1x2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>},
    %arg1: tensor<3x3x2x2xf32>)
    -> (tensor<1x2x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) {
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {}, {}]>]>} : (tensor<1x2x16x16xf32>, tensor<3x3x2x2xf32>) -> tensor<1x2x14x14xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {}, {}]> : tensor<1x2x14x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x2x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {}, {}]>]>
    }
    : (tensor<1x2x16x16xf32>, tensor<3x3x2x2xf32>) -> tensor<1x2x14x14xf32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {}, {}]> : tensor<1x2x14x14xf32>
  return %1 : tensor<1x2x14x14xf32>
}

// CHECK-LABEL: func @convolution_spatial_stride_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x8x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_spatial_stride_padding(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x8x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[1, 1], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x14xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x8x14xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {}, {"a"}, {}]>, <@mesh, [{?}, {?}, {?}, {?}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {}]>] manual_axes={"a"} (%arg2: tensor<1x1x8x16xf32>, %arg3: tensor<3x3x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 7:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[C_ONE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP:.*]] = stablehlo.compare GE, %[[ID_IN_GRP]], %[[C_ONE]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[BCAST_CMP:.*]] = stablehlo.broadcast_in_dim %[[CMP]], dims = [] : (tensor<i1>) -> tensor<1x1x1x16xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[BCAST_CMP]], %[[CP1]], %[[BCAST_PAD]] : tensor<1x1x1x16xi1>, tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %arg2, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x16xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x9x16xf32>
  // HALO-NEXT:   %[[C_BASE:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[C_ZERO]], %[[C_ZERO]], %[[C_BASE]], %[[C_ZERO]], sizes = [1, 1, 9, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x9x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x9x16xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x9x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x4x14xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<1x1x4x14xf32>
  // HALO-NEXT: } : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>, tensor<f32>) -> tensor<1x1x8x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [2, 1], pad = [[1, 1], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x8x14xf32>
  return %0 : tensor<1x1x8x14xf32>
}

// CHECK-LABEL: func @convolution_dilations
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_dilations(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x12x14xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x12x14xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {}, {"a"}, {}]>, <@mesh, [{?}, {?}, {?}, {?}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {}]>] manual_axes={"a"} (%arg2: tensor<1x1x8x16xf32>, %arg3: tensor<3x3x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 6:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x2x16xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x2x16xf32>) -> tensor<1x1x2x16xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 0:2, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x2x16xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x1x2x16xf32>) -> tensor<1x1x2x16xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg2, %[[CP2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x2x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x2x16xf32>) -> tensor<1x1x12x16xf32>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ID_IN_GRP]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[C_ZERO]], %[[C_ZERO]], %[[ADD]], %[[C_ZERO]], sizes = [1, 1, 10, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x12x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x10x16xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x6x14xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<1x1x6x14xf32>
  // HALO-NEXT: } : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>, tensor<f32>) -> tensor<1x1x12x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x12x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x12x14xf32>
  return %0 : tensor<1x1x12x14xf32>
}

// CHECK-LABEL: func @convolution_2d_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {"b"}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {"b"}]>})
func.func @convolution_2d_sharded(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {"b"}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {"b"}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x16x16xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {"b"}]> : tensor<1x1x16x16xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {}, {"a"}, {"b"}]>, <@mesh, [{?}, {?}, {?}, {?}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<1x1x8x8xf32>, %arg3: tensor<3x3x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV0:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV0:.*]] = stablehlo.divide %[[PID0]], %[[C_DIV0]] : tensor<ui32>
  // HALO-NEXT:   %[[ID0:.*]] = stablehlo.convert %[[DIV0]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L0:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 7:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L0]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[C_ONE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[ID0]], %[[C_ONE1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD1:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[BCAST_CMP1:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<1x1x1x8xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_CMP1]], %[[CP1]], %[[BCAST_PAD1]] : tensor<1x1x1x8xi1>, tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[SLICE_R0:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 0:1, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R0]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[C_ONE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[ID0]], %[[C_ONE2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD2:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[BCAST_CMP2:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<1x1x1x8xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BCAST_CMP2]], %[[CP2]], %[[BCAST_PAD2]] : tensor<1x1x1x8xi1>, tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[CONCAT0:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x1x10x8xf32>
  // HALO-NEXT:   %[[C_BASE0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE0:.*]] = stablehlo.dynamic_slice %[[CONCAT0]], %[[C_ZERO0]], %[[C_ZERO0]], %[[C_BASE0]], %[[C_ZERO0]], sizes = [1, 1, 10, 8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x10x8xf32>
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_REM1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM1:.*]] = stablehlo.remainder %[[PID1]], %[[C_REM1]] : tensor<ui32>
  // HALO-NEXT:   %[[ID1:.*]] = stablehlo.convert %[[REM1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L1:.*]] = stablehlo.slice %[[SLICE0]] [0:1, 0:1, 0:10, 7:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x8xf32>) -> tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x10x1xf32>) -> tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[C_ONE3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[ID1]], %[[C_ONE3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD3:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[BCAST_CMP3:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x1x10x1xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BCAST_CMP3]], %[[CP3]], %[[BCAST_PAD3]] : tensor<1x1x10x1xi1>, tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[SLICE_R1:.*]] = stablehlo.slice %[[SLICE0]] [0:1, 0:1, 0:10, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x8xf32>) -> tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}> : (tensor<1x1x10x1xf32>) -> tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[C_ONE4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[ID1]], %[[C_ONE4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD4:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[BCAST_CMP4:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x1x10x1xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BCAST_CMP4]], %[[CP4]], %[[BCAST_PAD4]] : tensor<1x1x10x1xi1>, tensor<1x1x10x1xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL3]], %[[SLICE0]], %[[SEL4]], dim = 3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x1xf32>, tensor<1x1x10x8xf32>, tensor<1x1x10x1xf32>) -> tensor<1x1x10x10xf32>
  // HALO-NEXT:   %[[C_BASE1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[C_ZERO1]], %[[C_ZERO1]], %[[C_ZERO1]], %[[C_BASE1]], sizes = [1, 1, 10, 10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x10x10xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE1]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x10xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x8xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<1x1x8x8xf32>
  // HALO-NEXT: } : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>, tensor<f32>) -> tensor<1x1x16x16xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {"b"}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x16x16xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x16x16xf32>
  return %0 : tensor<1x1x16x16xf32>
}

// CHECK-LABEL: func @convolution_subaxis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<2x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>})
func.func @convolution_subaxis(
    %arg0: tensor<2x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<2x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{"a":(1)2}, {}, {}, {}]> : tensor<2x1x16x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<2x1x14x14xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]> : tensor<2x1x14x14xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>, <@mesh_a4, [{?}, {?}, {?}, {?}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>] manual_axes={"a":(2)2} (%arg2: tensor<2x1x8x16xf32>, %arg3: tensor<3x3x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[C_REM:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.remainder %[[DIV]], %[[C_REM]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[REM]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:2, 0:1, 7:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x8x16xf32>) -> tensor<2x1x1x16xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]> : tensor<4x2xi64>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x1x16xf32>) -> tensor<2x1x1x16xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:1, 0:1, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x8x16xf32>) -> tensor<2x1x1x16xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [6, 4], [7, 5]]> : tensor<4x2xi64>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x1x16xf32>) -> tensor<2x1x1x16xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg2, %[[CP2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x1x16xf32>, tensor<2x1x8x16xf32>, tensor<2x1x1x16xf32>) -> tensor<2x1x10x16xf32>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ID_IN_GRP]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[C_ZERO]], %[[C_ZERO]], %[[ADD]], %[[C_ZERO]], sizes = [2, 1, 9, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x10x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x9x16xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {}, {}]>]>} : (tensor<2x1x9x16xf32>, tensor<3x3x1x1xf32>) -> tensor<2x1x7x14xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<2x1x7x14xf32>
  // HALO-NEXT: } : (tensor<2x1x16x16xf32>, tensor<3x3x1x1xf32>, tensor<f32>) -> tensor<2x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {}, {"a":(2)2}, {}]>]>
    }
    : (tensor<2x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<2x1x14x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x1x14x14xf32>
  return %0 : tensor<2x1x14x14xf32>
}

// CHECK-LABEL: func @convolution_lhs_and_rhs_dilations_low_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x1x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_lhs_and_rhs_dilations_low_pad(
    %arg0: tensor<1x1x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x1x1x1xf32>)
    -> (tensor<1x1x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x4x8xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[4, 6], [0, 0]], lhs_dilate = [3, 1], rhs_dilate = [2, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x4x8xf32>, tensor<3x1x1x1xf32>) -> tensor<1x1x16x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x16x8xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {}, {"a"}, {}]>, <@mesh, [{?}, {?}, {?}, {?}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {}]>] manual_axes={"a"} (%arg2: tensor<1x1x2x8xf32>, %arg3: tensor<3x1x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 1:2, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x2x8xf32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[C_ONE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[ID_IN_GRP]], %[[C_ONE1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD1:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[BCAST_CMP1:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<1x1x1x8xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_CMP1]], %[[CP1]], %[[BCAST_PAD1]] : tensor<1x1x1x8xi1>, tensor<1x1x1x8xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 0:2, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
  // HALO-NEXT:   %[[C_ONE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[ID_IN_GRP]], %[[C_ONE2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD2:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x1x2x8xf32>
  // HALO-NEXT:   %[[BCAST_CMP2:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<1x1x2x8xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BCAST_CMP2]], %[[CP2]], %[[BCAST_PAD2]] : tensor<1x1x2x8xi1>, tensor<1x1x2x8xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x8xf32>, tensor<1x1x2x8xf32>, tensor<1x1x2x8xf32>) -> tensor<1x1x5x8xf32>
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg4, low = [0, 0, 2, 0], high = [0, 0, 3, 0], interior = [0, 0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x5x8xf32>, tensor<f32>) -> tensor<1x1x10x8xf32>
  // HALO-NEXT:   %[[C_DIFF:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[C_BASE:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ID_IN_GRP]], %[[C_DIFF]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_BASE]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIL:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[UNDIL:.*]] = stablehlo.divide %[[ADD]], %[[C_DIL]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_ZERO]], %[[C_ZERO]], %[[UNDIL]], %[[C_ZERO]], sizes = [1, 1, 6, 8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x8xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [3, 1], rhs_dilate = [2, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x6x8xf32>, tensor<3x1x1x1xf32>) -> tensor<1x1x12x8xf32>
  // HALO-NEXT:   %[[C_POST_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[POST_PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_POST_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[POST_DIV:.*]] = stablehlo.divide %[[POST_PID]], %[[C_POST_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[POST_ID:.*]] = stablehlo.convert %[[POST_DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_POST_DIFF:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[C_POST_BASE:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[POST_MUL:.*]] = stablehlo.multiply %[[POST_ID]], %[[C_POST_DIFF]] : tensor<i32>
  // HALO-NEXT:   %[[POST_ADD:.*]] = stablehlo.add %[[POST_MUL]], %[[C_POST_BASE]] : tensor<i32>
  // HALO-NEXT:   %[[C_POST_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[POST_MAX:.*]] = stablehlo.maximum %[[POST_ADD]], %[[C_POST_ZERO2]] : tensor<i32>
  // HALO-NEXT:   %[[C_POST_DIL:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[POST_REM:.*]] = stablehlo.remainder %[[POST_MAX]], %[[C_POST_DIL]] : tensor<i32>
  // HALO-NEXT:   %[[POST_SLICE:.*]] = stablehlo.dynamic_slice %[[LOCAL_CONV]], %[[C_POST_ZERO]], %[[C_POST_ZERO]], %[[POST_REM]], %[[C_POST_ZERO]], sizes = [1, 1, 8, 8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x12x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x8x8xf32>
  // HALO-NEXT:   sdy.return %[[POST_SLICE]] : tensor<1x1x8x8xf32>
  // HALO-NEXT: } : (tensor<1x1x4x8xf32>, tensor<3x1x1x1xf32>, tensor<f32>) -> tensor<1x1x16x8xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[4, 6], [0, 0]], lhs_dilate = [3, 1], rhs_dilate = [2, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x4x8xf32>, tensor<3x1x1x1xf32>) -> tensor<1x1x16x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x16x8xf32>
  return %0 : tensor<1x1x16x8xf32>
}

// CHECK-LABEL: func @convolution_spatial_reduction_window_reversal_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x8x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<8x1x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
// CHECK-SAME: -> (tensor<1x1x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>})
func.func @convolution_spatial_reduction_window_reversal_replicated(
    %arg0: tensor<1x1x8x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<8x1x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
    -> (tensor<1x1x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_LHS:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x1xf32>
  // CHECK-NEXT: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {}, {}]> : tensor<8x1x1x1xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], reverse = [true, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x1xf32>, tensor<8x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  // CHECK-NEXT: return %[[CONV]] : tensor<1x1x1x1xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]], reverse = [1, 0]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}], unreduced={"a"}>]>
    }
    : (tensor<1x1x8x1xf32>, tensor<8x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %1 = sdy.all_reduce {"a"} %0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x1x1x1xf32>
  return %1 : tensor<1x1x1x1xf32>
}

// CHECK-LABEL: func @convolution_indivisible_group_and_per_group_reduction_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x1x6xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a", "b"}]>}, %[[ARG1:.*]]: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {"b"}, {"a"}]>})
// CHECK-SAME: -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
func.func @convolution_indivisible_group_and_per_group_reduction_replicated(
    %arg0: tensor<1x1x1x6xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a", "b"}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {"b"}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NEXT: %[[RESHARD_LHS:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x1x6xf32>
  // CHECK-NEXT: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x2x3xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x6xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x1x3xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{}, {}, {}, {"a"}]> : tensor<1x1x1x3xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x1x3xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 3 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}], unreduced={"b"}>]>
    }
    : (tensor<1x1x1x6xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x1x3xf32>
  %1 = sdy.all_reduce {"b"} %0 out_sharding=<@mesh_a4, [{}, {}, {}, {"a"}]> : tensor<1x1x1x3xf32>
  return %1 : tensor<1x1x1x3xf32>
}

// CHECK-LABEL: func @convolution_spatial_halo_with_indivisible_non_spatial_dim_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<3x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}, {"a"}, {}]>})
func.func @convolution_spatial_halo_with_indivisible_non_spatial_dim_replicated(
    %arg0: tensor<3x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<3x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}, {"a"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"b"}, {}, {}, {}]> : tensor<3x1x16x16xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}, {}, {}]>]>} : (tensor<3x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<3x1x14x14xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{"b"}, {}, {"a"}, {}]> : tensor<3x1x14x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<3x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}, {"a"}, {}]>]>
    }
    : (tensor<3x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<3x1x14x14xf32>
  return %0 : tensor<3x1x14x14xf32>
}

// CHECK-LABEL: func @convolution_spatial_halo_with_batch_group_per_group_out_sharded_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x2xf32>)
// CHECK-SAME: -> (tensor<2x2x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}, {"a"}, {}]>})
func.func @convolution_spatial_halo_with_batch_group_per_group_out_sharded_replicated(
    %arg0: tensor<4x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x2xf32>)
    -> (tensor<2x2x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}, {"a"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<4x1x16x16xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 2 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<4x1x16x16xf32>, tensor<3x3x1x2xf32>) -> tensor<2x2x14x14xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{"b"}, {}, {"a"}, {}]> : tensor<2x2x14x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x2x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 2 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}, {"a"}, {}]>]>
    }
    : (tensor<4x1x16x16xf32>, tensor<3x3x1x2xf32>) -> tensor<2x2x14x14xf32>
  return %0 : tensor<2x2x14x14xf32>
}

// CHECK-LABEL: func @convolution_indivisible_spatial
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x15x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x13x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_indivisible_spatial(
    %arg0: tensor<1x1x15x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x13x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x15x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x15x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x13x14xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x13x14xf32>

  // HALO-NEXT: %[[PAD_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[PAD_IN:.*]] = stablehlo.pad %[[ARG0]], %[[PAD_ZERO]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>} : (tensor<1x1x15x16xf32>, tensor<f32>) -> tensor<1x1x16x16xf32>
  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD_IN]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {}, {"a"}, {}]>, <@mesh, [{?}, {?}, {?}, {?}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {}]>] manual_axes={"a"} (%arg2: tensor<1x1x8x16xf32>, %arg3: tensor<3x3x1x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 7:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:1, 0:1, 0:1, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg2, %[[CP2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x1x16xf32>, tensor<1x1x8x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x10x16xf32>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ID_IN_GRP]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[C_ZERO]], %[[C_ZERO]], %[[ADD]], %[[C_ZERO]], sizes = [1, 1, 9, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x10x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x9x16xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<1x1x9x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x7x14xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<1x1x7x14xf32>
  // HALO-NEXT: } : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>, tensor<f32>) -> tensor<1x1x14x14xf32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1, 0:1, 0:13, 0:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>} : (tensor<1x1x14x14xf32>) -> tensor<1x1x13x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x15x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x13x14xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1x1x13x14xf32>
  return %0 : tensor<1x1x13x14xf32>
}

// CHECK-LABEL: func @convolution_spatial_halo_and_reduction_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}, {"a"}, {}]>}, %[[ARG1:.*]]: tensor<3x3x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}, {}]>})
// CHECK-SAME: -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_spatial_halo_and_reduction_sharded(
    %arg0: tensor<1x2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}, {"a"}, {}]>},
    %arg1: tensor<3x3x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}, {}]>})
    -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}, {}, {}]> : tensor<1x2x16x16xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}], unreduced={"b"}>]>} : (tensor<1x2x16x16xf32>, tensor<3x3x2x1xf32>) -> tensor<1x1x14x14xf32>
  // REPL-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}], unreduced={"b"}> : tensor<1x1x14x14xf32>
  // REPL-NEXT: %[[AR:.*]] = sdy.all_reduce {"b"} %[[RESHARD_OUT]] out_sharding=<@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[CST]]) in_shardings=[<@mesh, [{}, {"b"}, {"a"}, {}]>, <@mesh, [{}, {}, {"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {}, {"a"}, {}], unreduced={"b"}>] manual_axes={"a"} (%arg2: tensor<1x2x8x16xf32>, %arg3: tensor<3x3x2x1xf32>, %arg4: tensor<f32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_IN_GRP:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [0:1, 0:2, 7:8, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}, {}]>]>} : (tensor<1x2x8x16xf32>) -> tensor<1x2x1x16xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}, {}]>]>} : (tensor<1x2x1x16xf32>) -> tensor<1x2x1x16xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:1, 0:2, 0:1, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}, {}]>]>} : (tensor<1x2x8x16xf32>) -> tensor<1x2x1x16xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}, {}]>]>} : (tensor<1x2x1x16xf32>) -> tensor<1x2x1x16xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg2, %[[CP2]], dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}, {}]>]>} : (tensor<1x2x1x16xf32>, tensor<1x2x8x16xf32>, tensor<1x2x1x16xf32>) -> tensor<1x2x10x16xf32>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ID_IN_GRP]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[C_ZERO]], %[[C_ZERO]], %[[ADD]], %[[C_ZERO]], sizes = [1, 2, 9, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}, {}]>]>} : (tensor<1x2x10x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x9x16xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICE]], %arg3)
  // HALO-SAME{LITERAL}: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}], unreduced={"b"}>]>} : (tensor<1x2x9x16xf32>, tensor<3x3x2x1xf32>) -> tensor<1x1x7x14xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<1x1x7x14xf32>
  // HALO-NEXT: } : (tensor<1x2x16x16xf32>, tensor<3x3x2x1xf32>, tensor<f32>) -> tensor<1x1x14x14xf32>
  // HALO-NEXT: %[[AR:.*]] = sdy.all_reduce {"b"} %[[MC]] out_sharding=<@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}], unreduced={"b"}>]>
    }
    : (tensor<1x2x16x16xf32>, tensor<3x3x2x1xf32>) -> tensor<1x1x14x14xf32>
  %1 = sdy.all_reduce {"b"} %0 out_sharding=<@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<1x1x14x14xf32>
  return %1 : tensor<1x1x14x14xf32>
}

// CHECK-LABEL: func @convolution_spatial_halo_skipped_when_lhs_resharded_from_other_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x16x16x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>}, %[[ARG1:.*]]: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<2x14x14x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>})
func.func @convolution_spatial_halo_skipped_when_lhs_resharded_from_other_dim(
    %arg0: tensor<2x16x16x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<2x14x14x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"a"}, {}, {}]> : tensor<2x16x16x1xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh, [{}, {}, {}, {}]> : tensor<2x16x16x1xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD1]], %[[ARG1]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<2x16x16x1xf32>, tensor<3x3x1x1xf32>) -> tensor<2x14x14x1xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {"a"}, {}, {}]> : tensor<2x14x14x1xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<2x14x14x1xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"a"}, {}, {}]> : tensor<2x16x16x1xf32>
  %1 = stablehlo.convolution(%0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {}, {}]>]>
    }
    : (tensor<2x16x16x1xf32>, tensor<3x3x1x1xf32>) -> tensor<2x14x14x1xf32>
  return %1 : tensor<2x14x14x1xf32>
}

// CHECK-LABEL: func @conv_dual_semantics_only_spatial_reduction
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
func.func @conv_dual_semantics_only_spatial_reduction(
  %arg0: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>},
  %arg1: tensor<8x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
    -> (tensor<1x3x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) {
  // CHECK-NEXT: %[[LHS_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x2xf32>
  // CHECK-NEXT: %[[RHS_REPL:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {}, {}]> : tensor<8x1x2x4xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[LHS_REPL]], %[[RHS_REPL]])
  // CHECK-SAME:   {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>}
  // CHECK-NOT:  sdy.all_reduce
  // CHECK-NEXT: return %[[CONV]] : tensor<1x3x8x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[1, 1], [0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {}, {}, {}], unreduced={"a"}>]>
    } : (tensor<1x8x8x2xf32>, tensor<8x1x2x4xf32>) -> tensor<1x3x8x4xf32>
  %1 = sdy.all_reduce {"a"} %0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x3x8x4xf32>
  return %1 : tensor<1x3x8x4xf32>
}

// CHECK-LABEL: func @conv_dual_semantics_with_other_reduction_axis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {"b"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {"b"}, {}]>})
func.func @conv_dual_semantics_with_other_reduction_axis(
  %arg0: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {"b"}]>},
  %arg1: tensor<8x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {"b"}, {}]>})
    -> (tensor<1x3x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) {
  // CHECK-NEXT: %[[LHS_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {"b"}]> : tensor<1x8x8x2xf32>
  // CHECK-NEXT: %[[RHS_REPL:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {"b"}, {}]> : tensor<8x1x2x4xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[LHS_REPL]], %[[RHS_REPL]])
  // CHECK-SAME:   {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}], unreduced={"b"}>]>}
  // CHECK-NEXT: %[[AR:.*]] = sdy.all_reduce {"b"} %[[CONV]] out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x3x8x4xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<1x3x8x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[1, 1], [0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {}, {}, {}], unreduced={"a", "b"}>]>
    } : (tensor<1x8x8x2xf32>, tensor<8x1x2x4xf32>) -> tensor<1x3x8x4xf32>
  %1 = sdy.all_reduce {"a", "b"} %0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x3x8x4xf32>
  return %1 : tensor<1x3x8x4xf32>
}

// CHECK-LABEL: func @conv_pure_reduction_1x1_unpadded_unchanged
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
func.func @conv_pure_reduction_1x1_unpadded_unchanged(
  %arg0: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}, {}]>},
  %arg1: tensor<8x1x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
    -> (tensor<1x1x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) {
  // CHECK-NOT:  sdy.reshard
  // CHECK:      %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME:   {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}], unreduced={"a"}>]>}
  // CHECK-NEXT: %[[AR:.*]] = sdy.all_reduce {"a"} %[[CONV]] out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x4xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<1x1x8x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {}, {}, {}], unreduced={"a"}>]>
    } : (tensor<1x8x8x2xf32>, tensor<8x1x2x4xf32>) -> tensor<1x1x8x4xf32>
  %1 = sdy.all_reduce {"a"} %0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x4xf32>
  return %1 : tensor<1x1x8x4xf32>
}

// CHECK-LABEL: func @conv_dual_semantics_with_partitioned_output_spatial_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x3x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
func.func @conv_dual_semantics_with_partitioned_output_spatial_dim(
  %arg0: tensor<1x8x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>},
  %arg1: tensor<8x3x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}, {}]>})
    -> (tensor<1x3x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}, {}]>}) {
  // CHECK-NEXT: %[[LHS_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x2xf32>
  // CHECK-NEXT: %[[RHS_REPL:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {}, {}]> : tensor<8x3x2x4xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[LHS_REPL]], %[[RHS_REPL]])
  // CHECK-SAME:   {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>}
  // CHECK-NEXT: %[[OUT_RESHARD:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"b"}, {}]> : tensor<1x3x8x4xf32>
  // CHECK-NEXT: return %[[OUT_RESHARD]] : tensor<1x3x8x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {}, {"b"}, {}], unreduced={"a"}>]>
    } : (tensor<1x8x8x2xf32>, tensor<8x3x2x4xf32>) -> tensor<1x3x8x4xf32>
  %1 = sdy.all_reduce {"a"} %0 out_sharding=<@mesh, [{}, {}, {"b"}, {}]> : tensor<1x3x8x4xf32>
  return %1 : tensor<1x3x8x4xf32>
}

// CHECK-LABEL: func @convolution_padded_spatial_sharded_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x64x64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {}, {"a"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<3x3x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {}]>})
// CHECK-SAME: -> (tensor<4x64x64x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {}, {"a"}, {}]>})
func.func @convolution_padded_spatial_sharded_all_gather(
    %arg0: tensor<4x64x64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {}]>})
    -> (tensor<4x64x64x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{"b"}, {}, {}, {}]> : tensor<4x64x64x8xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME:   pad = {{\[\[}}32, 32], [32, 32{{\]\]}}, rhs_dilate = [32, 32]
  // REPL-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}, {}]>]>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{"b"}, {}, {"a"}, {}]> : tensor<4x64x64x12xf32>
  // REPL-NEXT: return %[[RES]] : tensor<4x64x64x12xf32>

  // HALO-NEXT: %[[GATHERED:.*]] = sdy.all_gather [{}, {}, {"a"}, {}] %[[ARG0]] out_sharding=<@mesh_a4, [{"b"}, {}, {}, {}]> : tensor<4x64x64x8xf32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[GATHERED]], %[[ARG1]])
  // HALO-SAME:   in_shardings=[<@mesh_a4, [{"b"}, {}, {}, {}]>, <@mesh_a4, [{}, {}, {}, {}]>]
  // HALO-SAME:   out_shardings=[<@mesh_a4, [{"b"}, {}, {"a"}, {}]>]
  // HALO-SAME:   manual_axes={"a"} (%[[MC_LHS:.*]]: tensor<4x64x64x8xf32>, %[[MC_RHS:.*]]: tensor<3x3x8x12xf32>) {
  // HALO-NEXT:   %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT:   %[[PADDED:.*]] = stablehlo.pad %[[MC_LHS]], %[[ZERO]], low = [0, 0, 32, 0], high = [0, 0, 32, 0], interior = [0, 0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}, {}]>]>} : (tensor<4x64x64x8xf32>, tensor<f32>) -> tensor<4x64x128x8xf32>
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 0, 16, 16, 32, 32, 48, 48]> : tensor<8xi32>
  // HALO-NEXT:   %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<8xi32>, tensor<ui32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[OFFSET:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICED:.*]] = stablehlo.dynamic_slice %[[PADDED]], %[[C0]], %[[C0]], %[[OFFSET]], %[[C0]], sizes = [4, 64, 80, 8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}, {}]>]>} : (tensor<4x64x128x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x64x80x8xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICED]], %[[MC_RHS]])
  // HALO-SAME:     pad = {{\[\[}}32, 32], [0, 0{{\]\]}}, rhs_dilate = [32, 32]
  // HALO-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}, {}]>]>
  // HALO-SAME:     : (tensor<4x64x80x8xf32>, tensor<3x3x8x12xf32>) -> tensor<4x64x16x12xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<4x64x16x12xf32>
  // HALO-NEXT: } : (tensor<4x64x64x8xf32>, tensor<3x3x8x12xf32>) -> tensor<4x64x64x12xf32>
  // HALO-NEXT: return %[[MC]] : tensor<4x64x64x12xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
    window = {stride = [1, 1], pad = [[32, 32], [32, 32]], rhs_dilate = [32, 32]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {"a"}, {}]>]>
    }
    : (tensor<4x64x64x8xf32>, tensor<3x3x8x12xf32>) -> tensor<4x64x64x12xf32>
  return %0 : tensor<4x64x64x12xf32>
}
