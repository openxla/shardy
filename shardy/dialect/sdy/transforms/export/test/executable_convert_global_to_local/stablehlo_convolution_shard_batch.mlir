// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_conv(
  %arg0: tensor<2x4x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}, {}, {}]>},
  %arg1: tensor<3x3x2x4xi32>
) -> (tensor<2x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}, {}, {}]>]>
    } : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> tensor<2x2x2x4xi32>
  return %0 : tensor<2x2x2x4xi32>
}

//--- part2.mlir

func.func @main() {
  %iota64 = stablehlo.iota dim = 0 : tensor<64xi32>
  %lhs = stablehlo.reshape %iota64 : (tensor<64xi32>) -> tensor<2x4x4x2xi32>

  %iota72 = stablehlo.iota dim = 0 : tensor<72xi32>
  %rhs = stablehlo.reshape %iota72 : (tensor<72xi32>) -> tensor<3x3x2x4xi32>

  %expected = func.call @sequential_conv(%lhs, %rhs) : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> tensor<2x2x2x4xi32>
  %l0 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 1,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<1x4x4x2xi32>
  %l1 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 1,0,0,0>, limit_indices=array<i64: 2,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<1x4x4x2xi32>

  %res:2 = "interpreter.run_parallel"(%l0, %rhs, %l1, %rhs) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<1x4x4x2xi32>, tensor<3x3x2x4xi32>, tensor<1x4x4x2xi32>, tensor<3x3x2x4xi32>) -> (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>)

  %actual = "stablehlo.concatenate"(%res#0, %res#1) {dimension = 0 : i64} : (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>) -> tensor<2x2x2x4xi32>
  "check.expect_eq"(%actual, %expected) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()
  return
}

