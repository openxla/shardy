// RUN: rm -rf %t && mkdir -p %t && split-file %s %t
// RUN: sdy_opt %t/part1.mlir --sdy-pad-for-divisibility --sdy-resolve-single-device-sharding="partition-count=4" --sdy-convert-global-to-local="partition-count=4" --sdy-inline-meshes --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_processed.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>

// Batch dim 0 has size 3 (indivisible by "x"=2, padded 3->4) and gets non-zero
// runtime values in its padded row from stablehlo.cosine. Spatial dim 1 has
// size 3 on LHS (padded 3->4 across "y"=2) and size 6 on RHS (sharded across
// "y"=2), with pad=[2, 2] and stride=[2], so the padded convolution produces
// spatial size 2 which trimOutputForDims slices back to 1. The sliced
// convolution output must not be marked kZero in PaddingCache, so the
// downstream sum reduction across batch dim 0 masks out the padded batch row 3
// before reducing.
func.func @parallel_conv_non_zero_batch_padding(
    %arg0: tensor<3x3x1xf32>,
    %arg1: tensor<6x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}]>})
    -> (tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %lhs0 = sdy.all_slice [{"x"}, {"y"}, {}] %arg0 out_sharding=<@mesh, [{"x"}, {"y"}, {}]> : tensor<3x3x1xf32>
  %lhs = stablehlo.cosine %lhs0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : tensor<3x3x1xf32>
  %conv = stablehlo.convolution(%lhs, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [2],
      pad = [[2, 2]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
       sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>}
    : (tensor<3x3x1xf32>, tensor<6x1x1xf32>) -> tensor<3x1x1xf32>
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %red = stablehlo.reduce(%conv init: %c0) applies stablehlo.add across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
    : (tensor<3x1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  return %red : tensor<1x1xf32>
}

//--- part2.mlir
func.func @sequential_conv_non_zero_batch_padding(
    %arg0: tensor<3x3x1xf32>, %arg1: tensor<6x1x1xf32>) -> tensor<1x1xf32> {
  %lhs = stablehlo.cosine %arg0 : tensor<3x3x1xf32>
  %conv = stablehlo.convolution(%lhs, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [2],
      pad = [[2, 2]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<3x3x1xf32>, tensor<6x1x1xf32>) -> tensor<3x1x1xf32>
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %red = stablehlo.reduce(%conv init: %c0) applies stablehlo.add across dimensions = [0]
    : (tensor<3x1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  return %red : tensor<1x1xf32>
}

func.func @main() {
  %lhs = stablehlo.constant dense<0.0> : tensor<3x3x1xf32>
  %rhs = stablehlo.constant dense<[
    [[0.0]],
    [[0.0]],
    [[1.0]],
    [[0.0]],
    [[1.0]],
    [[1.0]]
  ]> : tensor<6x1x1xf32>

  %expected = func.call @sequential_conv_non_zero_batch_padding(%lhs, %rhs)
      : (tensor<3x3x1xf32>, tensor<6x1x1xf32>) -> tensor<1x1xf32>

  %rhs_y0 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 3, 1, 1>, strides = array<i64: 1, 1, 1>} : (tensor<6x1x1xf32>) -> tensor<3x1x1xf32>
  %rhs_y1 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 3, 0, 0>, limit_indices = array<i64: 6, 1, 1>, strides = array<i64: 1, 1, 1>} : (tensor<6x1x1xf32>) -> tensor<3x1x1xf32>

  %res:4 = "interpreter.run_parallel"(
      %lhs, %rhs_y0,
      %lhs, %rhs_y1,
      %lhs, %rhs_y0,
      %lhs, %rhs_y1) {
    programs = [[
      @parallel_conv_non_zero_batch_padding,
      @parallel_conv_non_zero_batch_padding,
      @parallel_conv_non_zero_batch_padding,
      @parallel_conv_non_zero_batch_padding
    ]]
  } : (tensor<3x3x1xf32>, tensor<3x1x1xf32>,
       tensor<3x3x1xf32>, tensor<3x1x1xf32>,
       tensor<3x3x1xf32>, tensor<3x1x1xf32>,
       tensor<3x3x1xf32>, tensor<3x1x1xf32>) ->
      (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<1x1xf32>, tensor<1x1xf32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<1x1xf32>, tensor<1x1xf32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<1x1xf32>, tensor<1x1xf32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<1x1xf32>, tensor<1x1xf32>) -> ()
  func.return
}
