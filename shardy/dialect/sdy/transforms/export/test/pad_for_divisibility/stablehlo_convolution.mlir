// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @padded_conv_dims
func.func @padded_conv_dims(%arg0: tensor<1x4x7x3xf32>, %arg1: tensor<3x3x3x2xf32>) -> tensor<1x2x5x2xf32> {
  // Pad LHS input feature (dim 3) and spatial2 (dim 2) with zero.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<1x4x7x3xf32>, tensor<f32>) -> tensor<1x4x8x4xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {}, {"x"}, {"y"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x8x4xf32>

  // Pad RHS input feature (dim 2) with zero.
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<3x3x3x2xf32>, tensor<f32>) -> tensor<3x3x4x2xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {}, {"y"}, {}] %[[PAD1]] out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x4x2xf32>

  // Perform convolution (result is padded on spatial2 due to input padding).
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>}
  // CHECK-SAME: : (tensor<1x4x8x4xf32>, tensor<3x3x4x2xf32>) -> tensor<1x2x6x2xf32>

  // Trim the result back to original shape on spatial2 (dim 2).
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:2, 0:5, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<1x2x6x2xf32>) -> tensor<1x2x5x2xf32>
  // CHECK: return %[[TRIM]] : tensor<1x2x5x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {}, {"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x7x3xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {"y"}, {}] %arg1 out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x3x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>}
    : (tensor<1x4x7x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x2x5x2xf32>
  return %conv_out : tensor<1x2x5x2xf32>
}

// CHECK-LABEL: func @no_op_conv
func.func @no_op_conv(%arg0: tensor<1x4x8x4xf32>, %arg1: tensor<3x3x4x2xf32>) -> tensor<1x2x6x2xf32> {
  // Verify no padding is added when sharding divides cleanly.
  // CHECK-NOT: stablehlo.pad
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%{{.*}}, %{{.*}})
  // CHECK-NOT: stablehlo.slice
  // CHECK: return %[[CONV]]

  %sliced_lhs = sdy.all_slice [{}, {}, {"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x8x4xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {"y"}, {}] %arg1 out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x4x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>}
    : (tensor<1x4x8x4xf32>, tensor<3x3x4x2xf32>) -> tensor<1x2x6x2xf32>
  return %conv_out : tensor<1x2x6x2xf32>
}

// CHECK-LABEL: func @padded_conv_dilation
func.func @padded_conv_dilation(%arg0: tensor<1x6x7x3xf32>, %arg1: tensor<3x3x3x2xf32>) -> tensor<1x2x3x2xf32> {
  // Pad LHS input feature (dim 3) and spatial2 (dim 2) with zero.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<1x6x7x3xf32>, tensor<f32>) -> tensor<1x6x8x4xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {}, {"x"}, {"y"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x6x8x4xf32>

  // Pad RHS input feature (dim 2) with zero.
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<3x3x3x2xf32>, tensor<f32>) -> tensor<3x3x4x2xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {}, {"y"}, {}] %[[PAD1]] out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x4x2xf32>

  // Perform convolution with kernel dilation = [2, 2].
  // Padded input spatial2 (8) - dilated kernel spatial2 (5) + 1 = 4.
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: rhs_dilate = [2, 2]
  // CHECK-SAME: : (tensor<1x6x8x4xf32>, tensor<3x3x4x2xf32>) -> tensor<1x2x4x2xf32>

  // Trim output from 4 back to original size 3.
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:2, 0:3, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<1x2x4x2xf32>) -> tensor<1x2x3x2xf32>
  // CHECK: return %[[TRIM]] : tensor<1x2x3x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {}, {"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x6x7x3xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {"y"}, {}] %arg1 out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x3x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [2, 2],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>}
    : (tensor<1x6x7x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x2x3x2xf32>
  return %conv_out : tensor<1x2x3x2xf32>
}

sdy.mesh @mesh_4_4 = <["x"=4, "y"=4]>

// CHECK-LABEL: func @padded_conv_multi_spatial
func.func @padded_conv_multi_spatial(%arg0: tensor<1x5x7x3xf32>, %arg1: tensor<3x3x3x2xf32>) -> tensor<1x3x5x2xf32> {
  // Pad LHS spatial1 (dim 1) and spatial2 (dim 2) with zero.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 3, 1, 0], interior = [0, 0, 0, 0] : (tensor<1x5x7x3xf32>, tensor<f32>) -> tensor<1x8x8x3xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {"y"}, {"x"}, {}] %[[PAD0]] out_sharding=<@mesh_4_4, [{}, {"y"}, {"x"}, {}]> : tensor<1x8x8x3xf32>

  // RHS is replicated, no padding added.
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {}, {}, {}] %arg1 out_sharding=<@mesh_4_4, [{}, {}, {}, {}]> : tensor<3x3x3x2xf32>

  // Perform convolution.
  // Output spatial1: 8 - 3 + 1 = 6 (original 5 - 3 + 1 = 3).
  // Output spatial2: 8 - 3 + 1 = 6 (original 7 - 3 + 1 = 5).
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: : (tensor<1x8x8x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x6x6x2xf32>

  // Trim both output spatial dimensions (dim 1 and dim 2) back to original
  // sizes.
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:3, 0:5, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_4, [{}, {}, {}, {}]>]>} : (tensor<1x6x6x2xf32>) -> tensor<1x3x5x2xf32>
  // CHECK: return %[[TRIM]] : tensor<1x3x5x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {"y"}, {"x"}, {}] %arg0 out_sharding=<@mesh_4_4, [{}, {"y"}, {"x"}, {}]> : tensor<1x5x7x3xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {}, {}] %arg1 out_sharding=<@mesh_4_4, [{}, {}, {}, {}]> : tensor<3x3x3x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_4, [{}, {}, {}, {}]>]>}
    : (tensor<1x5x7x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x3x5x2xf32>
  return %conv_out : tensor<1x3x5x2xf32>
}

sdy.mesh @mesh_3_2 = <["x"=3, "y"=2]>

// CHECK-LABEL: func @padded_conv_stride
func.func @padded_conv_stride(%arg0: tensor<1x4x7x3xf32>, %arg1: tensor<3x3x3x2xf32>) -> tensor<1x2x3x2xf32> {
  // Pad LHS input feature (dim 3) with high = 1 (to 4), and spatial2 (dim 2)
  // with high = 2 (to 9).
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 0, 2, 1], interior = [0, 0, 0, 0] : (tensor<1x4x7x3xf32>, tensor<f32>) -> tensor<1x4x9x4xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {}, {"x"}, {"y"}] %[[PAD0]] out_sharding=<@mesh_3_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x9x4xf32>

  // Pad RHS input feature (dim 2) with high = 1 (to 4).
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<3x3x3x2xf32>, tensor<f32>) -> tensor<3x3x4x2xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {}, {"y"}, {}] %[[PAD1]] out_sharding=<@mesh_3_2, [{}, {}, {"y"}, {}]> : tensor<3x3x4x2xf32>

  // Perform convolution with stride [1, 2].
  // Padded input spatial2 (9) -> output spatial2 becomes (9-3)/2 + 1 = 4.
  // Original input spatial2 (7) -> output spatial2 was (7-3)/2 + 1 = 3.
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: stride = [1, 2]
  // CHECK-SAME: : (tensor<1x4x9x4xf32>, tensor<3x3x4x2xf32>) -> tensor<1x2x4x2xf32>

  // Trim output spatial2 from 4 back to 3.
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:2, 0:3, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3_2, [{}, {}, {}, {}]>]>} : (tensor<1x2x4x2xf32>) -> tensor<1x2x3x2xf32>
  // CHECK: return %[[TRIM]] : tensor<1x2x3x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {}, {"x"}, {"y"}] %arg0 out_sharding=<@mesh_3_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x7x3xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {"y"}, {}] %arg1 out_sharding=<@mesh_3_2, [{}, {}, {"y"}, {}]> : tensor<3x3x3x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 2],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_3_2, [{}, {}, {}, {}]>]>}
    : (tensor<1x4x7x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x2x3x2xf32>
  return %conv_out : tensor<1x2x3x2xf32>
}

// CHECK-LABEL: func @padded_conv_with_pad
func.func @padded_conv_with_pad(%arg0: tensor<1x4x7x3xf32>, %arg1: tensor<3x3x3x2xf32>) -> tensor<1x2x7x2xf32> {
  // Pad LHS input feature (dim 3) and spatial2 (dim 2) with zero.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<1x4x7x3xf32>, tensor<f32>) -> tensor<1x4x8x4xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {}, {"x"}, {"y"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x8x4xf32>

  // Pad RHS input feature (dim 2) with zero.
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<3x3x3x2xf32>, tensor<f32>) -> tensor<3x3x4x2xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {}, {"y"}, {}] %[[PAD1]] out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x4x2xf32>

  // Perform convolution with its own padding [[0, 0], [1, 1]].
  // Padded input spatial2 (8) -> output spatial2 becomes (8+1+1-3)/1 + 1 = 8.
  // Original input spatial2 (7) -> output spatial2 was (7+1+1-3)/1 + 1 = 7.
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: pad = {{\[\[}}0, 0], [1, 1]]
  // CHECK-SAME: : (tensor<1x4x8x4xf32>, tensor<3x3x4x2xf32>) -> tensor<1x2x8x2xf32>

  // Trim output spatial2 from 8 back to original 7.
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:2, 0:7, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<1x2x8x2xf32>) -> tensor<1x2x7x2xf32>
  // CHECK: return %[[TRIM]] : tensor<1x2x7x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {}, {"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x7x3xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {"y"}, {}] %arg1 out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x3x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [1, 1]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>}
    : (tensor<1x4x7x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x2x7x2xf32>
  return %conv_out : tensor<1x2x7x2xf32>
}

// CHECK-LABEL: func @padded_conv_nchw_layout
func.func @padded_conv_nchw_layout(%arg0: tensor<1x3x7x4xf32>, %arg1: tensor<2x3x3x3xf32>) -> tensor<1x2x5x2xf32> {
  // dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // LHS: Batch=0, Feature=1, Spatial1=2, Spatial2=3
  // RHS: OutputFeature=0, InputFeature=1, Spatial1=2, Spatial2=3

  // Pad LHS feature (dim 1) to 4, and Spatial1 (dim 2) to 8.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0] : (tensor<1x3x7x4xf32>, tensor<f32>) -> tensor<1x4x8x4xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {"y"}, {"x"}, {}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {"y"}, {"x"}, {}]> : tensor<1x4x8x4xf32>

  // Pad RHS input feature (dim 1) to 4.
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0, 0, 0], high = [0, 1, 0, 0], interior = [0, 0, 0, 0] : (tensor<2x3x3x3xf32>, tensor<f32>) -> tensor<2x4x3x3xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {"y"}, {}, {}] %[[PAD1]] out_sharding=<@mesh_4_2, [{}, {"y"}, {}, {}]> : tensor<2x4x3x3xf32>

  // Perform convolution. Output spatial1 grows from 5 to 6.
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // CHECK-SAME: : (tensor<1x4x8x4xf32>, tensor<2x4x3x3xf32>) -> tensor<1x2x6x2xf32>

  // Trim output spatial1 (dim 2) from 6 back to 5.
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:2, 0:5, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<1x2x6x2xf32>) -> tensor<1x2x5x2xf32>
  // CHECK: return %[[TRIM]] : tensor<1x2x5x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {"y"}, {"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}, {"x"}, {}]> : tensor<1x3x7x4xf32>
  %sliced_rhs = sdy.all_slice [{}, {"y"}, {}, {}] %arg1 out_sharding=<@mesh_4_2, [{}, {"y"}, {}, {}]> : tensor<2x3x3x3xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>}
    : (tensor<1x3x7x4xf32>, tensor<2x3x3x3xf32>) -> tensor<1x2x5x2xf32>
  return %conv_out : tensor<1x2x5x2xf32>
}

// CHECK-LABEL: func @padded_conv_sharded_output
func.func @padded_conv_sharded_output(%arg0: tensor<1x4x7x3xf32>, %arg1: tensor<3x3x3x2xf32>) -> tensor<1x2x5x2xf32> {
  // Pad LHS input feature (dim 3) and spatial2 (dim 2) with zero.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<1x4x7x3xf32>, tensor<f32>) -> tensor<1x4x8x4xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {}, {"x"}, {"y"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x8x4xf32>

  // Pad RHS input feature (dim 2) with zero.
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0, 0, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<3x3x3x2xf32>, tensor<f32>) -> tensor<3x3x4x2xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{}, {}, {"y"}, {}] %[[PAD1]] out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x4x2xf32>

  // Perform convolution with replicated sharding.
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[SLICE0]], %[[SLICE1]])
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<1x4x8x4xf32>, tensor<3x3x4x2xf32>) -> tensor<1x2x6x2xf32>

  // Trim output spatial2 (dim 2) back to original size (5).
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[CONV]] [0:1, 0:2, 0:5, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {}, {}]>]>} : (tensor<1x2x6x2xf32>) -> tensor<1x2x5x2xf32>

  // Pad the output spatial2 (dim 2) from 5 to 8 for divisibility.
  // CHECK: %[[CST2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD_OUT:.*]] = stablehlo.pad %[[TRIM]], %[[CST2]], low = [0, 0, 0, 0], high = [0, 0, 3, 0], interior = [0, 0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {"x"}, {}]>]>} : (tensor<1x2x5x2xf32>, tensor<f32>) -> tensor<1x2x8x2xf32>
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[PAD_OUT]] : tensor<1x2x8x2xf32> to tensor<1x2x5x2xf32>
  // CHECK: return %[[CAST]] : tensor<1x2x5x2xf32>

  %sliced_lhs = sdy.all_slice [{}, {}, {"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {}, {"x"}, {"y"}]> : tensor<1x4x7x3xf32>
  %sliced_rhs = sdy.all_slice [{}, {}, {"y"}, {}] %arg1 out_sharding=<@mesh_4_2, [{}, {}, {"y"}, {}]> : tensor<3x3x3x2xf32>
  %conv_out = stablehlo.convolution(%sliced_lhs, %sliced_rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}, {"x"}, {}]>]>}
    : (tensor<1x4x7x3xf32>, tensor<3x3x3x2xf32>) -> tensor<1x2x5x2xf32>
  return %conv_out : tensor<1x2x5x2xf32>
}
