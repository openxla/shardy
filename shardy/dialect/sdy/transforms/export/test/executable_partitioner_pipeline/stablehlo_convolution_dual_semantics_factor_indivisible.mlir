// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

// Spatial dimension 0 has size 7 (indivisible by mesh axis "a"=2) and is a
// kReduction factor with dual kPermutation semantics. Batch dimension 0 has
// size 2 and is partitioned along mesh axis "b"=2.
func.func @parallel_conv(
  %arg0: tensor<2x7x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>},
  %arg1: tensor<1x7x3x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>})
  -> (tensor<2x1x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_a2_b2, [{"b"}, {"a"}, {}, {}]> : tensor<2x7x3x1xf32>
  %1 = sdy.reshard %arg1 <@mesh_a2_b2, [{}, {"a"}, {}, {}]> : tensor<1x7x3x1xf32>
  %2 = stablehlo.convolution(%0, %1)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[3, 3], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 1]}
    {batch_group_count = 1 : i64,
     feature_group_count = 1 : i64,
     sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"b"}, {}, {}, {}]>]>}
    : (tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>) -> tensor<2x1x1x1xf32>
  %3 = sdy.reshard %2 <@mesh_a2_b2, [{}, {}, {}, {}]> : tensor<2x1x1x1xf32>
  return %3 : tensor<2x1x1x1xf32>
}

func.func @sequential_conv(
  %arg0: tensor<2x7x3x1xf32>,
  %arg1: tensor<1x7x3x1xf32>) -> tensor<2x1x1x1xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[3, 3], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 1]}
    {batch_group_count = 1 : i64,
     feature_group_count = 1 : i64}
    : (tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>) -> tensor<2x1x1x1xf32>
  return %0 : tensor<2x1x1x1xf32>
}

//--- part2.mlir

func.func @main() {
  %lhs_iota = stablehlo.iota dim = 0 : tensor<42xf32>
  %lhs = stablehlo.reshape %lhs_iota : (tensor<42xf32>) -> tensor<2x7x3x1xf32>

  %rhs_iota = stablehlo.iota dim = 0 : tensor<21xf32>
  %c1 = stablehlo.constant dense<1.0> : tensor<21xf32>
  %rhs_plus = stablehlo.add %rhs_iota, %c1 : tensor<21xf32>
  %rhs = stablehlo.reshape %rhs_plus : (tensor<21xf32>) -> tensor<1x7x3x1xf32>

  %seq = func.call @sequential_conv(%lhs, %rhs)
    : (tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>) -> tensor<2x1x1x1xf32>

  %res:4 = "interpreter.run_parallel"(
    %lhs, %rhs,
    %lhs, %rhs,
    %lhs, %rhs,
    %lhs, %rhs
  ) {
    programs = [[@parallel_conv, @parallel_conv, @parallel_conv, @parallel_conv
    ]]
  } : (
    tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>,
    tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>,
    tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>,
    tensor<2x7x3x1xf32>, tensor<1x7x3x1xf32>
  ) -> (
    tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>
  )

  "check.expect_eq"(%res#0, %seq) : (tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<2x1x1x1xf32>, tensor<2x1x1x1xf32>) -> ()

  return
}
