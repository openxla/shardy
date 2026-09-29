// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_x2 = <["x"=2]>

// Feature group count is 3, LHS input feature is 3 (padded 3->4), RHS kernel output feature is 3 (padded 3->4).
// feature_group_count is updated from 3 to 4, and result output feature is trimmed from 4 to 3.
func.func @parallel_conv_feature_group(
  %arg0: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>},
  %arg1: tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>})
  -> (tensor<1x1x1x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_x2, [{}, {}, {}, {"x"}]> : tensor<1x1x1x3xf32>
  %1 = sdy.reshard %arg1 <@mesh_x2, [{}, {}, {}, {"x"}]> : tensor<1x1x1x3xf32>
  %2 = stablehlo.convolution(%0, %1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 3 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {}, {}, {"x"}]>]>}
    : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
  %3 = sdy.reshard %2 <@mesh_x2, [{}, {}, {}, {}]> : tensor<1x1x1x3xf32>
  return %3 : tensor<1x1x1x3xf32>
}

func.func @sequential_conv_feature_group(%arg0: tensor<1x1x1x3xf32>, %arg1: tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 3 : i64}
    : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
  return %0 : tensor<1x1x1x3xf32>
}

//--- part2.mlir

func.func @main() {
  %lhs = stablehlo.constant dense<[
    [[[1.0, 2.0, 3.0]]]
  ]> : tensor<1x1x1x3xf32>

  %rhs = stablehlo.constant dense<[
    [[
      [2.0, 3.0, 4.0]
    ]]
  ]> : tensor<1x1x1x3xf32>

  %seq = func.call @sequential_conv_feature_group(%lhs, %rhs) : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>

  %res:2 = "interpreter.run_parallel"(
    %lhs, %rhs,
    %lhs, %rhs
  ) {
    programs = [[
      @parallel_conv_feature_group,
      @parallel_conv_feature_group
    ]]
  } : (
    tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>,
    tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>
  ) -> (
    tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>
  )

  "check.expect_eq"(%res#0, %seq) : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> ()

  return
}
