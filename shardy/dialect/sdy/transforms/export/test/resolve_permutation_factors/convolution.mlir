// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @convolution_spatial_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %arg1: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_spatial_permutation(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // CHECK: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
  // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>
  // CHECK: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>
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

// CHECK-LABEL: func @convolution_rhs_spatial_permutation_lhs_size_one
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16x1xf32>, %[[ARG1:.*]]: tensor<16x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
// CHECK-SAME: -> (tensor<8x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
func.func @convolution_rhs_spatial_permutation_lhs_size_one(
    %arg0: tensor<8x16x1xf32>,
    %arg1: tensor<16x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
    -> (tensor<8x10x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>}) {
  // CHECK: %[[RESHARD_RHS:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}, {}]> : tensor<16x10x4xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[RESHARD_RHS]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>
  // CHECK: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}]> : tensor<8x10x4xf32>
  // CHECK: return %[[RES]] : tensor<8x10x4xf32>
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
func.func @convolution_batch_group_count_per_group_size_one_kept(
    %arg0: tensor<3x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: batch_group_count = 3 : i64
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>
  // CHECK-NEXT: return %[[CONV]]
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
func.func @convolution_feature_group_count_per_group_size_one_kept(
    %arg0: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>},
    %arg1: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: feature_group_count = 3 : i64
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {"a"}]>]>
  // CHECK-NEXT: return %[[CONV]]
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
func.func @convolution_batch_group_count_per_group_batch_replicated(
    %arg0: tensor<6x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {}, {}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<2x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK: %[[RESHARD_LHS:.*]] = sdy.reshard %arg0 <@mesh_a4, [{}, {}, {}, {}]> : tensor<6x1x1x2xf32>
  // CHECK: %[[RESHARD_RHS:.*]] = sdy.reshard %arg1 <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x2x3xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME: batch_group_count = 3 : i64
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {}]>]>
  // CHECK: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{}, {}, {}, {"a"}]> : tensor<2x1x1x3xf32>
  // CHECK: return %[[RES]] : tensor<2x1x1x3xf32>
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
func.func @convolution_feature_group_count_per_group_feature_replicated(
    %arg0: tensor<1x1x1x6xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>},
    %arg1: tensor<1x1x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>})
    -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}, {"a"}]>}) {
  // CHECK: %[[RESHARD_LHS:.*]] = sdy.reshard %arg0 <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x1x6xf32>
  // CHECK: %[[RESHARD_RHS:.*]] = sdy.reshard %arg1 <@mesh_a4, [{}, {}, {}, {}]> : tensor<1x1x2x3xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME: feature_group_count = 3 : i64
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}, {}]>]>
  // CHECK: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{}, {}, {}, {"a"}]> : tensor<1x1x1x3xf32>
  // CHECK: return %[[RES]] : tensor<1x1x1x3xf32>
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
func.func @convolution_feature_group_count_per_group_reduction_replicated(
    %arg0: tensor<1x1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a", "b"}]>},
    %arg1: tensor<1x1x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}, {"a"}]>})
    -> (tensor<1x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"a"}]>}) {
  // CHECK: %[[RESHARD_LHS:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x1x4xf32>
  // CHECK: %[[RESHARD_RHS:.*]] = sdy.reshard %arg1 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x2x2xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_LHS]], %[[RESHARD_RHS]])
  // CHECK-SAME: feature_group_count = 2 : i64
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>
  // CHECK-NOT: sdy.all_reduce
  // CHECK: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {}, {"a"}]> : tensor<1x1x1x2xf32>
  // CHECK: return %[[RES]] : tensor<1x1x1x2xf32>
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

// CHECK-LABEL: func @convolution_spatial_sharded_stride_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<3x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}]>})
// CHECK-SAME: -> (tensor<4x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>})
func.func @convolution_spatial_sharded_stride_all_gather(
    %arg0: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>},
    %arg1: tensor<3x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}]>})
    -> (tensor<4x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{"b"}, {}, {}]> : tensor<4x16x8xf32>
  // REPL-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %[[ARG1]])
  // REPL-SAME:   stride = [2], pad = {{\[\[}}1, 1{{\]\]}}
  // REPL-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}]>]>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{"b"}, {"a"}, {}]> : tensor<4x8x12xf32>
  // REPL-NEXT: return %[[RES]] : tensor<4x8x12xf32>

  // HALO-NEXT: %[[GATHERED:.*]] = sdy.all_gather [{}, {"a"}, {}] %[[ARG0]] out_sharding=<@mesh_a4, [{"b"}, {}, {}]> : tensor<4x16x8xf32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[GATHERED]], %[[ARG1]])
  // HALO-SAME:   in_shardings=[<@mesh_a4, [{"b"}, {}, {}]>, <@mesh_a4, [{}, {}, {}]>]
  // HALO-SAME:   out_shardings=[<@mesh_a4, [{"b"}, {"a"}, {}]>]
  // HALO-SAME:   manual_axes={"a"} (%[[MC_LHS:.*]]: tensor<4x16x8xf32>, %[[MC_RHS:.*]]: tensor<3x8x12xf32>) {
  // HALO-NEXT:   %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT:   %[[PADDED:.*]] = stablehlo.pad %[[MC_LHS]], %[[ZERO]], low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}]>]>} : (tensor<4x16x8xf32>, tensor<f32>) -> tensor<4x18x8xf32>
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 0, 4, 4, 8, 8, 12, 12]> : tensor<8xi32>
  // HALO-NEXT:   %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<8xi32>, tensor<ui32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[OFFSET:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICED:.*]] = stablehlo.dynamic_slice %[[PADDED]], %[[C0]], %[[OFFSET]], %[[C0]], sizes = [4, 5, 8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}]>]>} : (tensor<4x18x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x5x8xf32>
  // HALO-NEXT:   %[[LOCAL_CONV:.*]] = stablehlo.convolution(%[[SLICED]], %[[MC_RHS]])
  // HALO-SAME:     stride = [2], pad = {{\[\[}}0, 0{{\]\]}}
  // HALO-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}]>]>
  // HALO-SAME:     : (tensor<4x5x8xf32>, tensor<3x8x12xf32>) -> tensor<4x2x12xf32>
  // HALO-NEXT:   sdy.return %[[LOCAL_CONV]] : tensor<4x2x12xf32>
  // HALO-NEXT: } : (tensor<4x16x8xf32>, tensor<3x8x12xf32>) -> tensor<4x8x12xf32>
  // HALO-NEXT: return %[[MC]] : tensor<4x8x12xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f] x [0, i, o] -> [b, 0, f],
    window = {stride = [2], pad = [[1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {"a"}, {}]>]>
    }
    : (tensor<4x16x8xf32>, tensor<3x8x12xf32>) -> tensor<4x8x12xf32>
  return %0 : tensor<4x8x12xf32>
}

// Negative test: non-trivial lhs_dilate > 1 falls back to replicating the
// spatial dimension.
// CHECK-LABEL: func @convolution_spatial_sharded_lhs_dilation_replicated
func.func @convolution_spatial_sharded_lhs_dilation_replicated(
    %arg0: tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>},
    %arg1: tensor<3x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}]>})
    -> (tensor<4x16x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %arg0 <@mesh_a4, [{"b"}, {}, {}]> : tensor<4x8x8xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
  // CHECK-SAME:   lhs_dilate = [2]
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}]>]>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{"b"}, {"a"}, {}]> : tensor<4x16x12xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x16x12xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f] x [0, i, o] -> [b, 0, f],
    window = {stride = [1], pad = [[1, 2]], lhs_dilate = [2]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {"a"}, {}]>]>
    }
    : (tensor<4x8x8xf32>, tensor<3x8x12xf32>) -> tensor<4x16x12xf32>
  return %0 : tensor<4x16x12xf32>
}

// Negative test: non-trivial stride producing an indivisible output spatial
// dimension falls back to replicating the spatial dimension.
// CHECK-LABEL: func @convolution_spatial_sharded_stride_indivisible_output_replicated
func.func @convolution_spatial_sharded_stride_indivisible_output_replicated(
    %arg0: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>},
    %arg1: tensor<3x8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}]>})
    -> (tensor<4x6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"b"}, {"a"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %arg0 <@mesh_a4, [{"b"}, {}, {}]> : tensor<4x16x8xf32>
  // CHECK-NEXT: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
  // CHECK-SAME:   stride = [3]
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {}, {}]>]>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh_a4, [{"b"}, {"a"}, {}]> : tensor<4x6x12xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x6x12xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f] x [0, i, o] -> [b, 0, f],
    window = {stride = [3], pad = [[1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"b"}, {"a"}, {}]>]>
    }
    : (tensor<4x16x8xf32>, tensor<3x8x12xf32>) -> tensor<4x6x12xf32>
  return %0 : tensor<4x6x12xf32>
}
