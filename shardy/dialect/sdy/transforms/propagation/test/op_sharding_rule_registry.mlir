// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics 2>&1 | FileCheck %s

// CHECK-LABEL: func @pointwise_op
func.func @pointwise_op(%arg0: tensor<2x1x4xf32>) -> tensor<2x1x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=3, j=1, k=4}>
  %0 = stablehlo.add %arg0, %arg0: tensor<2x1x4xf32>
  return %0 : tensor<2x1x4xf32>
}

// CHECK-LABEL: func @pointwise_op_size_zero_dim
func.func @pointwise_op_size_zero_dim(%arg0: tensor<2x0x4xf32>) -> tensor<2x0x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=2, j=0, k=4}>
  %0 = stablehlo.add %arg0, %arg0: tensor<2x0x4xf32>
  return %0 : tensor<2x0x4xf32>
}

// CHECK-LABEL: func @scalar_op
func.func @scalar_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([], [])->([])>}
  %0 = stablehlo.add %arg0, %arg0:tensor<f32>
  return %0 : tensor<f32>
}

//===----------------------------------------------------------------------===//
// NOTE: Please keep the order of ops alphabetical.
//===----------------------------------------------------------------------===//


// CHECK-LABEL: func @all_gather
func.func @all_gather(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, k], [i, k]) {i=2, j=2, k=4} need_replication={j, k}>
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    // channel_id = 0
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    // use_global_device_ids = false
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
  return %0#0, %0#1 :  tensor<2x4xi64>, tensor<2x4xi64>
}

// CHECK-LABEL: func @all_to_all_different_dimensions
func.func @all_to_all_different_dimensions(%arg0: tensor<2x4xi64>, %arg1: tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([k, i], [k, i])->([l, j], [l, j]) {i=4, j=2, k=2, l=4} need_replication={i, j, k, l}>
  %0:2 = "stablehlo.all_to_all"(%arg0, %arg1) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 2 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    // channel_id = 0
  } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
  return %0#0, %0#1 :  tensor<4x2xi64>, tensor<4x2xi64>
}

// CHECK-LABEL: func @all_to_all_same_dimension
func.func @all_to_all_same_dimension(%arg0: tensor<2x4xi64>, %arg1: tensor<2x4xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([j, i], [j, i])->([k, i], [k, i]) {i=4, j=2, k=2} need_replication={j, k}>
  %0:2 = "stablehlo.all_to_all"(%arg0, %arg1) {
    split_dimension = 0 : i64,
    concat_dimension = 0 : i64,
    split_count = 2 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    // channel_id = 0
  } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
  return %0#0, %0#1 :  tensor<2x4xi64>, tensor<2x4xi64>
}

// CHECK-LABEL: func @bitcast_convert_upcast
func.func @bitcast_convert_upcast(%arg0: tensor<4x2x2xui32>) -> tensor<4x2xui64> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j]) {i=4, j=2, k=2} need_replication={k}>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  return %0 :  tensor<4x2xui64>
}

// CHECK-LABEL: func @bitcast_convert_equal
func.func @bitcast_convert_equal(%arg0: tensor<4x2x2xui32>) -> tensor<4x2x2xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=4, j=2, k=2}>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4x2x2xui32>) ->tensor<4x2x2xui32>
  return %0 : tensor<4x2x2xui32>
}

// CHECK-LABEL: func @bitcast_convert_downcast
func.func @bitcast_convert_downcast(%arg0: tensor<4x2xui64>) -> tensor<4x2x2xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j, k]) {i=4, j=2, k=2} need_replication={k}>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  return %0 :  tensor<4x2x2xui32>
}

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<2x13x1xf32>) -> tensor<2x64x13x1xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, l])->([i, j, k, l]) {i=2, j=64, k=13, l=1}>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<2x13x1xf32>) -> tensor<2x64x13x1xf32>
  return %0 :  tensor<2x64x13x1xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_size_zero_dim
func.func @broadcast_in_dim_size_zero_dim(%arg0: tensor<2x13x0xf32>) -> tensor<2x64x13x0xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, l])->([i, j, k, l]) {i=2, j=64, k=13, l=0}>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<2x13x0xf32>) -> tensor<2x64x13x0xf32>
  return %0 :  tensor<2x64x13x0xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_no_broadcast_dims
func.func @broadcast_in_dim_no_broadcast_dims(%arg0: tensor<f32>) -> tensor<2x1x13xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([])->([i, j, k]) {i=2, j=1, k=13}>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x1x13xf32>
  return %0 :  tensor<2x1x13xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_input_dim_expanded
func.func @broadcast_in_dim_input_dim_expanded(%arg0: tensor<2x1x13xf32>) -> tensor<2x64x13xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l])->([i, k, l]) {i=2, j=1, k=64, l=13}>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x13xf32>) -> tensor<2x64x13xf32>
  return %0 :  tensor<2x64x13xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_all_dim_types_in_one
func.func @broadcast_in_dim_all_dim_types_in_one(%arg0: tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, j, l, n])->([i, j, k, m, n, o]) {i=2, j=5, k=3, l=1, m=11, n=7, o=13}>}
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}

// CHECK-LABEL: func @cholesky
func.func @cholesky(%arg0: tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, j, k, l]) {i=2, j=4, k=8, l=8} need_replication={k, l}>
  %0 = stablehlo.cholesky %arg0, lower = true : (tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xf32>
  return %0 :  tensor<2x4x8x8xf32>
}

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [i, j])->([i, j]) {i=4, j=8}>
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @clamp_scalar_min_max
func.func @clamp_scalar_min_max(%arg0: tensor<f32>, %arg1: tensor<4x8xf32>, %arg2: tensor<f32>) -> tensor<4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([], [i, j], [])->([i, j]) {i=4, j=8}>
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @concat_operands_dim_size_one
func.func @concat_operands_dim_size_one(%arg0: tensor<4x1x256xf32>, %arg1: tensor<4x1x256xf32>, %arg2: tensor<4x1x256xf32>) -> tensor<4x3x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [i, j, k])->([i, j, k]) {i=4, j=3, k=256} need_replication={j}>
 %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 1 : (tensor<4x1x256xf32>, tensor<4x1x256xf32>, tensor<4x1x256xf32>) -> tensor<4x3x256xf32>
 return %0 : tensor<4x3x256xf32>
}

// CHECK-LABEL: func @concat_gcd_is_one
func.func @concat_gcd_is_one(%arg0: tensor<4x3x256xf32>, %arg1: tensor<4x5x256xf32>) -> tensor<4x8x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=4, j=8, k=256} need_replication={j}>
 %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x3x256xf32>, tensor<4x5x256xf32>) -> tensor<4x8x256xf32>
 return %0 : tensor<4x8x256xf32>
}

// CHECK-LABEL: func @concat_operands_with_same_shape
func.func @concat_operands_with_same_shape(%arg0: tensor<4x16x256xf32>, %arg1: tensor<4x16x256xf32>) -> tensor<4x32x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=4, j=32, k=256} need_replication={j}>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x16x256xf32>, tensor<4x16x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

// CHECK-LABEL: func @concat_gcd_is_equal_to_operand_dim
func.func @concat_gcd_is_equal_to_operand_dim(%arg0: tensor<4x32x256xf32>, %arg1: tensor<4x16x256xf32>) -> tensor<4x48x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=4, j=48, k=256} need_replication={j}>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x16x256xf32>) -> tensor<4x48x256xf32>
  return %0 : tensor<4x48x256xf32>
}

// CHECK-LABEL: func @concat_gcd_is_greater_than_all_operand_dims
func.func @concat_gcd_is_greater_than_all_operand_dims(%arg0: tensor<4x32x256xf32>, %arg1: tensor<4x48x256xf32>) -> tensor<4x80x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=4, j=80, k=256} need_replication={j}>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concat_operands_are_results_of_slices
func.func @concat_operands_are_results_of_slices(%arg0: tensor<4x40x256xf32>, %arg1: tensor<4x60x256xf32>) -> tensor<4x80x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=4, j=80, k=256} permutation={j}>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concat_operands_are_from_slices_of_the_same_tensor
func.func @concat_operands_are_from_slices_of_the_same_tensor(%arg0: tensor<4x40x256xf32>) -> tensor<4x96x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg0 [0:4, 0:24, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x24x256xf32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [i, j, k])->([i, j, k]) {i=4, j=96, k=256} permutation={j}>
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x40x256xf32>, tensor<4x24x256xf32>) -> tensor<4x96x256xf32>
  return %2 : tensor<4x96x256xf32>
}

// CHECK-LABEL: func @concat_not_all_operands_are_results_of_slices
func.func @concat_not_all_operands_are_results_of_slices(%arg0: tensor<4x40x256xf32>, %arg1: tensor<4x60x256xf32>) -> tensor<4x92x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.negate %arg1 : tensor<4x60x256xf32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k]) {i=4, j=92, k=256} need_replication={j}>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x60x256xf32>) -> tensor<4x92x256xf32>
  return %2 : tensor<4x92x256xf32>
}

// CHECK-LABEL: func @concat_not_all_operands_are_from_slices_of_the_same_tensor
func.func @concat_not_all_operands_are_from_slices_of_the_same_tensor(%arg0: tensor<4x40x256xf32>, %arg1: tensor<4x60x256xf32>) -> tensor<4x96x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:24, 0:256] : (tensor<4x60x256xf32>) -> tensor<4x24x256xf32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [i, j, k])->([i, j, k]) {i=4, j=96, k=256} need_replication={j}>
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x40x256xf32>, tensor<4x24x256xf32>) -> tensor<4x96x256xf32>
  return %2 : tensor<4x96x256xf32>
}

// CHECK-LABEL: func @conv_simple
func.func @conv_simple(%arg0 : tensor<2x224x224x192xf32>, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk, lm, n], [k, m, n, o])->([i, j, l, o]) {i=2, j=112, k=2, l=112, m=2, n=192, o=64} reduction={k, m, n} permutation={j, l}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// CHECK-LABEL: func @conv_window_size_greater_than_num_windows
func.func @conv_window_size_greater_than_num_windows(%arg0: tensor<2x224x224x192xf32>, %arg1: tensor<112x112x192x64xf32>) -> tensor<2x57x57x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk, lm, n], [j, l, n, o])->([i, k, m, o]) {i=2, j=112, k=2, l=112, m=2, n=192, o=64} reduction={j, l, n} permutation={k, m}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<112x112x192x64xf32>) -> tensor<2x57x57x64xf32>
  return %0 : tensor<2x57x57x64xf32>
}

// CHECK-LABEL: func @conv_batch_group_count
func.func @conv_batch_group_count(%arg0: tensor<8x224x224x192xf32>, %arg1: tensor<3x3x192x256xf32>) -> tensor<2x112x112x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip]) {i=4, j=2, k=112, l=2, m=112, n=2, o=192, p=64} reduction={l, n, o} permutation={k, m}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 4 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<8x224x224x192xf32>, tensor<3x3x192x256xf32>) -> tensor<2x112x112x256xf32>
  return %0 : tensor<2x112x112x256xf32>
}

// CHECK-LABEL: func @conv_per_group_batch_size_one
func.func @conv_per_group_batch_size_one(%arg0: tensor<4x224x224x192xf32>, %arg1: tensor<3x3x192x256xf32>) -> tensor<1x112x112x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, kl, mn, o], [l, n, o, ip])->([j, k, m, ip]) {i=4, j=1, k=112, l=2, m=112, n=2, o=192, p=64} reduction={l, n, o} permutation={k, m}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 4 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<4x224x224x192xf32>, tensor<3x3x192x256xf32>) -> tensor<1x112x112x256xf32>
  return %0 : tensor<1x112x112x256xf32>
}

// CHECK-LABEL: func @conv_feature_group_count
func.func @conv_feature_group_count(%arg0: tensor<8x224x224x192xf32>, %arg1: tensor<3x3x12x256xf32>) -> tensor<8x112x112x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk, lm, no], [k, m, o, np])->([i, j, l, np]) {i=8, j=112, k=2, l=112, m=2, n=16, o=12, p=16} reduction={k, m, o} permutation={j, l}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 16 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<8x224x224x192xf32>, tensor<3x3x12x256xf32>) -> tensor<8x112x112x256xf32>
  return %0 : tensor<8x112x112x256xf32>
}

// CHECK-LABEL: func @conv_per_group_input_feature_size_one
func.func @conv_per_group_input_feature_size_one(%arg0: tensor<8x224x224x16xf32>, %arg1: tensor<3x3x1x256xf32>) -> tensor<8x112x112x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk, lm, n], [k, m, o, np])->([i, j, l, np]) {i=8, j=112, k=2, l=112, m=2, n=16, o=1, p=16} reduction={k, m, o} permutation={j, l}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 16 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<8x224x224x16xf32>, tensor<3x3x1x256xf32>) -> tensor<8x112x112x256xf32>
  return %0 : tensor<8x112x112x256xf32>
}

// CHECK-LABEL: func @conv_output_feature_size_one
func.func @conv_output_feature_size_one(%arg0: tensor<8x224x224x192xf32>, %arg1: tensor<3x3x12x16xf32>) -> tensor<8x112x112x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk, lm, no], [k, m, o, n])->([i, j, l, n]) {i=8, j=112, k=2, l=112, m=2, n=16, o=12} reduction={k, m, o} permutation={j, l}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 16 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<8x224x224x192xf32>, tensor<3x3x12x16xf32>) -> tensor<8x112x112x16xf32>
  return %0 : tensor<8x112x112x16xf32>
}

// CHECK-LABEL: func @custom_call_compact_wy_helper
func.func @custom_call_compact_wy_helper(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=128, j=128}>
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @custom_call_inspect_sharding
func.func @custom_call_inspect_sharding(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_x64_combine
func.func @custom_call_x64_combine(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) {backend_config = ""} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}

// CHECK-LABEL: func @custom_call_x64_combine_used_by_rng_bit_generator
func.func @custom_call_x64_combine_used_by_rng_bit_generator(%arg0: tensor<2xui32>, %arg1: tensor<2xui32>) -> (tensor<4x1000xui32>, tensor<2xui64>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [i])->([i]) {i=2} blocked_propagation={i}>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui64>
  %output_state, %output = stablehlo.rng_bit_generator %0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  return %output, %0 : tensor<4x1000xui32>, tensor<2xui64>
}

// CHECK-LABEL: func @custom_call_x64_split_high
func.func @custom_call_x64_split_high(%arg0: tensor<8x2xui64>) -> tensor<8x2xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = ""} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_x64_split_low
func.func @custom_call_x64_split_low(%arg0: tensor<8x2xui64>) -> tensor<8x2xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = ""} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_xla_megascale_provide_metadata
func.func @custom_call_xla_megascale_provide_metadata(%arg0: tensor<8x2xbf16>) -> tensor<8x2xbf16> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  %0 = stablehlo.custom_call @xla.megascale.provide_metadata(%arg0) {backend_config = ""} : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
  return %0 : tensor<8x2xbf16>
}

// CHECK-LABEL: func @custom_call_move_to_device
func.func @custom_call_move_to_device(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  %0 = stablehlo.custom_call @MoveToDevice(%arg0) {backend_config = ""} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_move_to_host
func.func @custom_call_move_to_host(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  %0 = stablehlo.custom_call @MoveToHost(%arg0) {backend_config = ""} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_layout_constraint
func.func @custom_call_layout_constraint(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {backend_config = ""} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_eigh
func.func @custom_call_eigh(%arg0: tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, k]) {i=8, j=4, k=4}>
  %0:2 = stablehlo.custom_call @Eigh(%arg0) {backend_config = "1,1,100,1e-6"} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
  return %0#0, %0#1 : tensor<8x4x4xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_qr
func.func @custom_call_qr(%arg0: tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  %0:2 = stablehlo.custom_call @Qr(%arg0) : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_qr_decomposition_block
func.func @custom_call_qr_decomposition_block(%arg0: tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  %0:2 = stablehlo.custom_call @QrDecompositionBlock(%arg0) : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_householder_product
func.func @custom_call_householder_product(%arg0: tensor<8x12x16xf32>, %arg1: tensor<8x5xf32>) -> tensor<8x12x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, j, k]) {i=8, j=12, k=16, l=5} need_replication={j, k, l}>
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<8x12x16xf32>, tensor<8x5xf32>) -> tensor<8x12x16xf32>
  return %0 : tensor<8x12x16xf32>
}

// CHECK-LABEL: func @custom_call_erf
func.func @custom_call_erf(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  %0 = stablehlo.custom_call @mhlo.erf (%arg0) {backend_config = ""} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_topk_of_1d
func.func @custom_call_topk_of_1d(%arg0: tensor<16xf32>) -> (tensor<1xf32>, tensor<1xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i], [i]) {i=16} need_replication={i} blocked_propagation={i}>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 1 : i64,
        largest = true},
    mhlo.version = 1 : i64}
    : (tensor<16xf32>) -> (tensor<1xf32>, tensor<1xi32>)
  return %0#0, %0#1 : tensor<1xf32>, tensor<1xi32>
}

// CHECK-LABEL: func @custom_call_topk_of_2d
func.func @custom_call_topk_of_2d(%arg0: tensor<16x8xf32>) -> (tensor<16x1xf32>, tensor<16x1xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} need_replication={j} blocked_propagation={j}>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 1 : i64,
        largest = true},
    mhlo.version = 1 : i64}
    : (tensor<16x8xf32>) -> (tensor<16x1xf32>, tensor<16x1xi32>)
  return %0#0, %0#1 : tensor<16x1xf32>, tensor<16x1xi32>
}

// CHECK-LABEL: func @custom_call_top2_of_2d
func.func @custom_call_top2_of_2d(%arg0: tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} need_replication={j} blocked_propagation={j}>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 2 : i64,
        largest = true},
    mhlo.version = 1 : i64}
    : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xi32>
}

// CHECK-LABEL: func @custom_call_approx_topk
func.func @custom_call_approx_topk(%arg0: tensor<16x4xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator]} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce
func.func @custom_call_partial_reduce(%arg0: tensor<16x4xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator]} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce_string_backend_config
func.func @custom_call_partial_reduce_string_backend_config(%arg0: tensor<16x4xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    backend_config = "{\22log2_reduction\22: 5, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 2, \22recall_target\22: 0.950000}",
    called_computations = [@top_k_gt_f32_comparator]} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @unregisterd_custom_call_with_existing_rule
func.func @unregisterd_custom_call_with_existing_rule(%arg0: tensor<4x2xf32>) -> tensor<2x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @dot_vector_vector
func.func @dot_vector_vector(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<f32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i], [i])->([]) {i=32} reduction={i}>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @dot_vector_matrix
func.func @dot_vector_matrix(%arg0: tensor<32xf32>, %arg1: tensor<32x16xf32>) -> tensor<16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([j], [j, i])->([i]) {i=16, j=32} reduction={j}>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<32xf32>, tensor<32x16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @dot_matrix_vector
func.func @dot_matrix_vector(%arg0: tensor<8x32xf32>, %arg1: tensor<32xf32>) -> tensor<8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [j])->([i]) {i=8, j=32} reduction={j}>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x32xf32>, tensor<32xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @dot_matrix_matrix
func.func @dot_matrix_matrix(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_general_no_batching_dims
func.func @dot_general_no_batching_dims(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_general_batching_dims
func.func @dot_general_batching_dims(%arg0: tensor<4x8x32xf32>, %arg1: tensor<4x32x16xf32>) -> tensor<4x8x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32} reduction={l}>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_general_many_mixed_dims
func.func @dot_general_many_mixed_dims(%arg0: tensor<2x4x8x4x64x32xf32>, %arg1: tensor<16x32x64x4x2xf32>) -> tensor<2x4x8x4x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l, o, n], [m, n, o, j, i])->([i, j, k, l, m]) {i=2, j=4, k=8, l=4, m=16, n=32, o=64} reduction={n, o}>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [4, 3], contracting_dims = [5, 4] x [1, 2] : (tensor<2x4x8x4x64x32xf32>, tensor<16x32x64x4x2xf32>) -> tensor<2x4x8x4x16xf32>
  return %0 : tensor<2x4x8x4x16xf32>
}

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<32x4x8xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<32x1x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [], [], [])->([i, j, k]) {i=32, j=4, k=8} need_replication={j, k} blocked_propagation={j, k}>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2] : (tensor<32x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @dynamic_update_slice
func.func @dynamic_update_slice(%arg0: tensor<32x4x8xf32>, %arg1: tensor<32x1x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<32x4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, k, m], [], [], [])->([i, j, l]) {i=32, j=4, k=1, l=8, m=2} need_replication={k, m}>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @fft
func.func @fft(%arg0: tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=64} need_replication={j, k}>
  %0  = stablehlo.fft %arg0, type = FFT, length = [32, 64] : (tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>>
  return %0 : tensor<8x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_inverse
func.func @fft_inverse(%arg0: tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=64} need_replication={j, k}>
  %0  = stablehlo.fft %arg0, type = IFFT, length = [32, 64] : (tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>>
  return %0 : tensor<8x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_real_truncated_result
func.func @fft_real_truncated_result(%arg0: tensor<8x32x64xf32>) -> tensor<8x32x33xcomplex<f32>> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=64} need_replication={j, k} blocked_propagation={k}>
  %0  = stablehlo.fft %arg0, type = RFFT, length = [32, 64] : (tensor<8x32x64xf32>) -> tensor<8x32x33xcomplex<f32>>
  return %0 : tensor<8x32x33xcomplex<f32>>
}

// CHECK-LABEL: func @fft_inverse_real_expanded_result
func.func @fft_inverse_real_expanded_result(%arg0: tensor<8x32x33xcomplex<f32>>) -> tensor<8x32x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=33} need_replication={j, k} blocked_propagation={k}>
  %0  = stablehlo.fft %arg0, type = IRFFT, length = [32, 64] : (tensor<8x32x33xcomplex<f32>>) -> tensor<8x32x64xf32>
  return %0 : tensor<8x32x64xf32>
}

// CHECK-LABEL: func @fft_single_fft_dimension
func.func @fft_single_fft_dimension(%arg0: tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=64} need_replication={k}>
  %0  = stablehlo.fft %arg0, type = FFT, length = [64] : (tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>>
  return %0 : tensor<8x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension_inverse
func.func @fft_single_fft_dimension_inverse(%arg0: tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=64} need_replication={k}>
  %0  = stablehlo.fft %arg0, type = IFFT, length = [64] : (tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>>
  return %0 : tensor<8x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension_real_truncated_result
func.func @fft_single_fft_dimension_real_truncated_result(%arg0: tensor<8x32x64xf32>) -> tensor<8x32x33xcomplex<f32>> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=64} need_replication={k} blocked_propagation={k}>
  %0  = stablehlo.fft %arg0, type = RFFT, length = [64] : (tensor<8x32x64xf32>) -> tensor<8x32x33xcomplex<f32>>
  return %0 : tensor<8x32x33xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension_inverse_real_expanded_result
func.func @fft_single_fft_dimension_inverse_real_expanded_result(%arg0: tensor<8x32x33xcomplex<f32>>) -> tensor<8x32x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=8, j=32, k=33} need_replication={k} blocked_propagation={k}>
  %0  = stablehlo.fft %arg0, type = IRFFT, length = [64] : (tensor<8x32x33xcomplex<f32>>) -> tensor<8x32x64xf32>
  return %0 : tensor<8x32x64xf32>
}

// CHECK-LABEL: @gather
func.func @gather(%arg0: tensor<3x4x2x5xf32>, %arg1: tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([o, k, l, m], [i, j, p])->([i, j, k, l, n]) {i=2, j=3, k=4, l=2, m=5, n=1, o=3, p=3} reduction={m, o} need_replication={k, n, p} blocked_propagation={k}>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3, 4],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0, 3],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2, 1>,
    indices_are_sorted = false
  } : (tensor<3x4x2x5xf32>, tensor<2x3x3xi64>) -> tensor<2x3x2x2x1xf32>
  return %0 : tensor<2x3x2x2x1xf32>
}

// CHECK-LABEL: @gather_implicit_index_vector_dim
func.func @gather_implicit_index_vector_dim(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([m, n, l], [i, j, k])->([i, j, k, l]) {i=2, j=3, k=2, l=2, m=3, n=4} reduction={m, n}>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [1],
      index_vector_dim = 3>,
    slice_sizes = array<i64: 1, 1, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  return %0 : tensor<2x3x2x2xf32>
}

// CHECK-LABEL: @gather_batching_dims
func.func @gather_batching_dims(%arg0: tensor<5x3x7x4xf32>, %arg1: tensor<7x5x3x2xi64>) -> tensor<7x5x3x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([j, m, i, l], [i, j, k, n])->([i, j, k, l]) {i=7, j=5, k=3, l=4, m=3, n=2} reduction={m} need_replication={l, n} blocked_propagation={l}>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0, 2],
      start_indices_batching_dims = [1, 0],
      start_index_map = [1, 3],
      index_vector_dim = 3>,
    slice_sizes = array<i64: 1, 1, 1, 2>,
    indices_are_sorted = false
  } : (tensor<5x3x7x4xf32>, tensor<7x5x3x2xi64>) -> tensor<7x5x3x2xf32>
  return %0 : tensor<7x5x3x2xf32>
}

// CHECK-LABEL: @gather_index_vector_dim_before_batching_dim
func.func @gather_index_vector_dim_before_batching_dim(%arg0: tensor<5x3x7x4xf32>, %arg1: tensor<7x2x5x3xi64>) -> tensor<7x5x3x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([j, m, i, l], [i, n, j, k])->([i, j, k, l]) {i=7, j=5, k=3, l=4, m=3, n=2} reduction={m} need_replication={l, n} blocked_propagation={l}>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0, 2],
      start_indices_batching_dims = [2, 0],
      start_index_map = [1, 3],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1, 1, 2>,
    indices_are_sorted = false
  } : (tensor<5x3x7x4xf32>, tensor<7x2x5x3xi64>) -> tensor<7x5x3x2xf32>
  return %0 : tensor<7x5x3x2xf32>
}

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<28x28x16xf32>, %arg1: tensor<f32>) -> tensor<30x26x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [])->([i, j, k]) {i=28, j=28, k=16} permutation={i, j}>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @reduce_single_result
func.func @reduce_single_result(%arg0: tensor<2x64x13xf32>) -> tensor<2x13xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [])->([i, k]) {i=2, j=64, k=13} reduction={j}>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_multiple_results
func.func @reduce_multiple_results(%arg0: tensor<2x64x13xf32>, %arg1: tensor<2x64x13xi32>)
    -> (tensor<64xf32>, tensor<64xi32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [], [])->([j], [j]) {i=2, j=64, k=13} reduction={i, k}>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0, 2] :
    (tensor<2x64x13xf32>, tensor<2x64x13xi32>, tensor<f32>, tensor<i32>) -> (tensor<64xf32>, tensor<64xi32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %3 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %4 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f32>, tensor<i32>
    }
  return %2#0, %2#1 : tensor<64xf32>, tensor<64xi32>
}

// CHECK-LABEL: func @reduce_size_one_dim
func.func @reduce_size_one_dim(%arg0: tensor<2x64x1x13xf32>) -> tensor<2x1x13xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l], [])->([i, k, l]) {i=2, j=64, k=1, l=13} reduction={j}>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x1x13xf32>, tensor<f32>) -> tensor<2x1x13xf32>
  return %1 : tensor<2x1x13xf32>
}

// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%arg0: tensor<2x64x1x13xi64>) -> tensor<2x32x1x13xi64> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, l, j, k])->([i, m, j, k]) {i=2, j=1, k=13, l=64, m=32} need_replication={l, m}>
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
    %0 = "stablehlo.add"(%arg3, %arg4) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
  }) {
    scatter_dimension = 1 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<2x64x1x13xi64>) -> tensor<2x32x1x13xi64>
  return %0 : tensor<2x32x1x13xi64>
}

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<48x48x3xf32>, %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
    -> (tensor<16x48x3xf32>, tensor<16x48x3xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [], [])->([i, j, k], [i, j, k]) {i=16, j=48, k=3} permutation={i, j}>
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<i32>, %arg6: tensor<f32>, %arg7 : tensor<i32>):
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {window_dimensions = array<i64: 3, 1, 1>,
      window_strides = array<i64: 3, 1, 1>,
      padding = dense<[[0, 0], [2, -2], [0, 0]]> : tensor<3x2xi64>}
      : (tensor<48x48x3xf32>, tensor<48x48x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<16x48x3xf32>, tensor<16x48x3xi32>)
  func.return %0#0, %0#1 : tensor<16x48x3xf32>, tensor<16x48x3xi32>
}

// CHECK-LABEL: func @reduce_window_permute_only
func.func @reduce_window_permute_only(%arg0: tensor<8x16xf32>, %arg1: tensor<f32>)
    -> tensor<8x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [])->([i, j]) {i=8, j=16} permutation={i, j}>
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<f32>):
    %1 = stablehlo.maximum %arg4, %arg5 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {window_dimensions = array<i64: 1, 1>,
      padding = dense<[[-1, 1], [2, -2]]> : tensor<2x2xi64>}
      : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @reduce_window_passthrough_dim
func.func @reduce_window_passthrough_dim(%arg0: tensor<8x16xf32>, %arg1: tensor<f32>)
    -> tensor<8x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [])->([i, j]) {i=8, j=16}
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<f32>):
    %1 = stablehlo.maximum %arg4, %arg5 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {window_dimensions = array<i64: 1, 1>,
      padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>}
      : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @reshape_scalar
func.func @reshape_scalar(%arg0: tensor<1x1xf32>) -> tensor<f32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([]) {i=1, j=1}>
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @reshape_size_zero_dim
func.func @reshape_size_zero_dim(%arg0: tensor<4x0xf32>) -> tensor<0x8xf32> {
  // CHECK: stablehlo.reshape %arg0
  // CHECK-NOT: sdy.sharding_rule
  %0 = stablehlo.reshape %arg0 : (tensor<4x0xf32>) -> tensor<0x8xf32>
  return %0 : tensor<0x8xf32>
}

// CHECK-LABEL: func @reshape_merge_dim
func.func @reshape_merge_dim(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @reshape_split_dim
func.func @reshape_split_dim(%arg0: tensor<8xf32>) -> tensor<2x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ij])->([i, j]) {i=2, j=4}>
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @reshape_split_dim_three_way
func.func @reshape_split_dim_three_way(%arg0: tensor<4x12xf32>) -> tensor<4x2x3x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, jkl])->([i, j, k, l]) {i=4, j=2, k=3, l=2}>
  %0 = stablehlo.reshape %arg0 : (tensor<4x12xf32>) -> tensor<4x2x3x2xf32>
  return %0 : tensor<4x2x3x2xf32>
}

// CHECK-LABEL: func @reshape_split_and_merge_dims
func.func @reshape_split_and_merge_dims(%arg0: tensor<8x4x5xf32>) -> tensor<2x16x5xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k, l])->([i, jk, l]) {i=2, j=4, k=4, l=5}>
  %0 = stablehlo.reshape %arg0 : (tensor<8x4x5xf32>) -> tensor<2x16x5xf32>
  return %0 : tensor<2x16x5xf32>
}

// CHECK-LABEL: func @reshape_swap_dims_no_common_divisor
func.func @reshape_swap_dims_no_common_divisor(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // CHECK: #sdy.op_sharding_rule<([i, l])->([j, k]) {i=3, j=2, k=3, l=2}>
  %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @reshape_swap_dims_with_common_divisor
func.func @reshape_swap_dims_with_common_divisor(%arg0: tensor<6x4xf32>) -> tensor<4x6xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ij, mn])->([ik, ln]) {i=2, j=3, k=2, l=3, m=2, n=2}>
  %0 = stablehlo.reshape %arg0 : (tensor<6x4xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

// CHECK-LABEL: func @reshape_swap_with_dim_before_and_in_between
func.func @reshape_swap_with_dim_before_and_in_between(%arg0: tensor<5x2x5x3xf32>) -> tensor<5x3x5x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l, n])->([i, k, m, o]) {i=5, j=2, k=3, l=5, m=5, n=3, o=2}>
  %0 = stablehlo.reshape %arg0 : (tensor<5x2x5x3xf32>) -> tensor<5x3x5x2xf32>
  return %0 : tensor<5x3x5x2xf32>
}

// CHECK-LABEL: func @reshape_swap_and_merge
func.func @reshape_swap_and_merge(%arg0: tensor<2x3x8xf32>) -> tensor<3x4x4xf32> {
  // CHECK: dy.sharding_rule = #sdy.op_sharding_rule<([i, k, mn])->([j, lm, n]) {i=2, j=3, k=3, l=2, m=2, n=4}>
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x8xf32>) -> tensor<3x4x4xf32>
  return %0 : tensor<3x4x4xf32>
}

// CHECK-LABEL: func @reshape_split_and_swap
func.func @reshape_split_and_swap(%arg0: tensor<3x4x4xf32>) -> tensor<2x3x8xf32> {
  // CHECK: dy.sharding_rule = #sdy.op_sharding_rule<([i, lm, n])->([j, k, mn]) {i=3, j=2, k=3, l=2, m=2, n=4}>
  %0 = stablehlo.reshape %arg0 : (tensor<3x4x4xf32>) -> tensor<2x3x8xf32>
  return %0 : tensor<2x3x8xf32>
}

// CHECK-LABEL: func @reshape_split_swap_and_merge
func.func @reshape_split_swap_and_merge(%arg0: tensor<4x7x5x8xf32>) -> tensor<14x10x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ij, l, n, o])->([ik, mn, o]) {i=2, j=2, k=7, l=7, m=2, n=5, o=8}>
  %0 = stablehlo.reshape %arg0 : (tensor<4x7x5x8xf32>) -> tensor<14x10x8xf32>
  return %0 : tensor<14x10x8xf32>
}

// CHECK-LABEL: func @reshape_size_one_dims
func.func @reshape_size_one_dims(%arg0: tensor<1x8x4xf32>) -> tensor<8x1x4x1xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l])->([j, k, l, m]) {i=1, j=8, k=1, l=4, m=1}>
  %0 = stablehlo.reshape %arg0 : (tensor<1x8x4xf32>) -> tensor<8x1x4x1xf32>
  return %0 : tensor<8x1x4x1xf32>
}

// CHECK-LABEL: func @reshape_merge_dim_then_size_one
func.func @reshape_merge_dim_then_size_one(%arg0: tensor<8x4xf32>) -> tensor<32x1xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij, k]) {i=8, j=4, k=1}>
  %0 = stablehlo.reshape %arg0 : (tensor<8x4xf32>) -> tensor<32x1xf32>
  return %0 : tensor<32x1xf32>
}

// CHECK-LABEL: func @reshape_split_dim_then_size_one
func.func @reshape_split_dim_then_size_one(%arg0: tensor<32x1xf32>) -> tensor<8x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k])->([i, j]) {i=8, j=4, k=1}>
  %0 = stablehlo.reshape %arg0 : (tensor<32x1xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @reshape_split_dim_with_intermediate_one
func.func @reshape_split_dim_with_intermediate_one(%arg0: tensor<32xf32>) -> tensor<8x1x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([ik])->([i, j, k]) {i=8, j=1, k=4}>
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<8x1x4xf32>
  return %0 : tensor<8x1x4xf32>
}

// CHECK-LABEL: func @reverse
func.func @reverse(%arg0: tensor<4x32x8x2xf32>) -> tensor<4x32x8x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, j, k, l]) {i=4, j=32, k=8, l=2} permutation={j, l}>
  %0 = stablehlo.reverse %arg0, dims = [1, 3] : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @rng_bit_generator
func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> tensor<4x1000xui32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i], [j, k]) {i=2, j=4, k=1000} need_replication={i} blocked_propagation={i}>
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  return %output : tensor<4x1000xui32>
}

// CHECK-LABEL: @scatter_single_input
func.func @scatter_single_input(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x3x2xi64>, %arg2: tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>{
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([n, k, m], [i, j, o], [i, j, l, m])->([n, k, m]) {i=2, j=3, k=4, l=2, m=2, n=3, o=2} reduction={i, j} need_replication={l, o}>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>, tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>
  return %0 : tensor<3x4x2xf32>
}

// CHECK-LABEL: @scatter_single_input_with_implicit_index_vector_dim
func.func @scatter_single_input_with_implicit_index_vector_dim(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x3xi64>, %arg2: tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>{
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([n, k, m], [i, j], [i, j, l, m])->([n, k, m]) {i=2, j=3, k=4, l=2, m=2, n=3} reduction={i, j} need_replication={l}>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf32>, tensor<2x3xi64>, tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>
  return %0 : tensor<3x4x2xf32>
}

// CHECK-LABEL: @scatter_inserted_window_dim_is_last_one
func.func @scatter_inserted_window_dim_is_last_one(%arg0: tensor<4x2x3xf32>, %arg1: tensor<2x3x2xi64>, %arg2: tensor<2x3x2x2xf32>) -> tensor<4x2x3xf32>{
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([k, m, n], [i, j, o], [i, j, l, m])->([k, m, n]) {i=2, j=3, k=4, l=2, m=2, n=3, o=2} reduction={i, j} need_replication={l, o}>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [2],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<4x2x3xf32>, tensor<2x3x2xi64>, tensor<2x3x2x2xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// CHECK-LABEL: @scatter_batching_dims
func.func @scatter_batching_dims(%arg0: tensor<5x3x7x4xf32>, %arg1: tensor<7x5x3x2xi64>, %arg2: tensor<7x5x3x2xf32>) -> tensor<5x3x7x4xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([j, n, i, l], [i, j, k, o], [i, j, k, m])->([j, n, i, l]) {i=7, j=5, k=3, l=4, m=2, n=3, o=2} reduction={k} need_replication={m, o}>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [3],
      inserted_window_dims = [1],
      input_batching_dims = [0, 2],
      scatter_indices_batching_dims = [1, 0],
      scatter_dims_to_operand_dims = [1, 3],
      index_vector_dim = 3>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<5x3x7x4xf32>, tensor<7x5x3x2xi64>, tensor<7x5x3x2xf32>) -> tensor<5x3x7x4xf32>
  return %0 : tensor<5x3x7x4xf32>
}

// CHECK-LABEL: @scatter_multiple_input
func.func @scatter_multiple_input(%arg0: tensor<3x4x2xi32>,
                                  %arg1: tensor<3x4x2xf32>,
                                  %arg2: tensor<2x3x2xi64>,
                                  %arg3: tensor<2x3x2x2xi32>,
                                  %arg4: tensor<2x3x2x2xf32>)
    -> (tensor<3x4x2xi32>, tensor<3x4x2xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([n, k, m], [n, k, m], [i, j, o], [i, j, l, m], [i, j, l, m])->([n, k, m], [n, k, m]) {i=2, j=3, k=4, l=2, m=2, n=3, o=2} reduction={i, j} need_replication={l, o}>
  %0:2 = "stablehlo.scatter"(%arg0, %arg1, %arg2, %arg3, %arg4) ({
    ^bb0(%arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>, %arg8: tensor<f32>):
      %1 = stablehlo.add %arg5, %arg7 : tensor<i32>
      %2 = stablehlo.add %arg6, %arg8 : tensor<f32>
      stablehlo.return %1, %2 : tensor<i32>, tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xi32>, tensor<3x4x2xf32>, tensor<2x3x2xi64>,
      tensor<2x3x2x2xi32>, tensor<2x3x2x2xf32>)
      -> (tensor<3x4x2xi32>, tensor<3x4x2xf32>)
  return %0#0, %0#1 : tensor<3x4x2xi32>, tensor<3x4x2xf32>
}

// CHECK-LABEL: func @select
func.func @select(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [i, j])->([i, j]) {i=4, j=8}>
  %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<4x8xi1>, tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @select_scalar_pred
func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([], [i, j], [i, j])->([i, j]) {i=4, j=8}>
  %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @select_and_scatter
func.func @select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<9x12x12x64xf32>, %arg2: tensor<f32>)
   -> tensor<10x24x24x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([j, l, n, o], [i, k, m, o], [])->([j, l, n, o]) {i=9, j=10, k=12, l=24, m=12, n=24, o=64} need_replication={i, k, m} permutation={j, l, n}>
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.compare GT, %arg3, %arg4 :(tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    padding = dense<[[-1, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_strides = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64xf32>, tensor<9x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32>) -> tensor<32x1x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=32, j=4, k=8} permutation={j, k}>
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @sort
func.func @sort(%arg0: tensor<4x32x8xi32>, %arg1: tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k])->([i, j, k], [i, j, k]) {i=4, j=32, k=8} need_replication={i}>
  %0:2 = "stablehlo.sort"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>, tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>)
  return %0#0, %0#1 : tensor<4x32x8xi32>, tensor<4x32x8xf32>
}

// CHECK-LABEL: func @sort_all_other_dims_size_one
func.func @sort_all_other_dims_size_one(%arg0: tensor<1x4x1xi32>) -> tensor<1x4x1xi32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=1, j=4, k=1} need_replication={j} blocked_propagation={j}>
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
  return %0 : tensor<1x4x1xi32>
}

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([k, j, l, i])->([i, j, k, l]) {i=100, j=32, k=256, l=64}>
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}

// CHECK-LABEL: func @triangular_solve_left_side_no_transpose
func.func @triangular_solve_left_side_no_transpose(%arg0: tensor<8x3x3xf32>, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, j, k])->([i, l, k]) {i=8, j=3, k=5, l=3} need_replication={j, k, l}>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}

// CHECK-LABEL: func @triangular_solve_right_side_no_transpose
func.func @triangular_solve_right_side_no_transpose(%arg0: tensor<8x3x3xf32>, %arg1: tensor<8x5x3xf32>) -> tensor<8x5x3xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, l, k], [i, j, k])->([i, j, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = false,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<8x3x3xf32>, tensor<8x5x3xf32>) -> tensor<8x5x3xf32>
  return %0 : tensor<8x5x3xf32>
}

// CHECK-LABEL: func @triangular_solve_left_side_transpose
func.func @triangular_solve_left_side_transpose(%arg0: tensor<8x3x3xf32>, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, l, j], [i, j, k])->([i, l, k]) {i=8, j=3, k=5, l=3} need_replication={j, k, l}>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose TRANSPOSE>
  } : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}

// CHECK-LABEL: func @triangular_solve_right_side_transpose
func.func @triangular_solve_right_side_transpose(%arg0: tensor<8x3x3xf32>, %arg1: tensor<8x5x3xf32>) -> tensor<8x5x3xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, l], [i, j, k])->([i, j, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = false,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose TRANSPOSE>
  } : (tensor<8x3x3xf32>, tensor<8x5x3xf32>) -> tensor<8x5x3xf32>
  return %0 : tensor<8x5x3xf32>
}
