// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sequential_conv(%arg0: tensor<2x4x4x4xi32>, %arg1: tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64
    } : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  return %0 : tensor<2x2x2x4xi32>
}

func.func @parallel_conv(
  %arg0: tensor<2x4x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>},
  %arg1: tensor<3x3x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {"x"}, {}]>}
) -> (tensor<2x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64
    } : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}, {}, {}]> : tensor<2x2x2x4xi32>
  return %1 : tensor<2x2x2x4xi32>
}

//--- part2.mlir

func.func @main() {
  %iota128 = stablehlo.iota dim = 0 : tensor<128xi32>
  %lhs_base = stablehlo.reshape %iota128 : (tensor<128xi32>) -> tensor<2x4x4x4xi32>
  %l0 = "stablehlo.slice"(%lhs_base) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 2,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x4xi32>) -> tensor<2x4x4x2xi32>
  %l1_unscaled = "stablehlo.slice"(%lhs_base) {start_indices=array<i64: 0,0,0,2>, limit_indices=array<i64: 2,4,4,4>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x4xi32>) -> tensor<2x4x4x2xi32>
  // Scale Shard 1 by 1000 to make its contribution identifiable in the sum
  %c1000 = stablehlo.constant dense<1000> : tensor<2x4x4x2xi32>
  %l1 = stablehlo.multiply %l1_unscaled, %c1000 : tensor<2x4x4x2xi32>
  %lhs = "stablehlo.concatenate"(%l0, %l1) {dimension = 3 : i64} : (tensor<2x4x4x2xi32>, tensor<2x4x4x2xi32>) -> tensor<2x4x4x4xi32>

  %iota144 = stablehlo.iota dim = 0 : tensor<144xi32>
  %rhs = stablehlo.reshape %iota144 : (tensor<144xi32>) -> tensor<3x3x4x4xi32>

  %seq = func.call @sequential_conv(%lhs, %rhs) : (tensor<2x4x4x4xi32>, tensor<3x3x4x4xi32>) -> tensor<2x2x2x4xi32>

  %r0 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 3,3,2,4>, strides=array<i64: 1,1,1,1>} : (tensor<3x3x4x4xi32>) -> tensor<3x3x2x4xi32>
  %r1 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0,0,2,0>, limit_indices=array<i64: 3,3,4,4>, strides=array<i64: 1,1,1,1>} : (tensor<3x3x4x4xi32>) -> tensor<3x3x2x4xi32>
  %pars:2 = "interpreter.run_parallel"(%l0, %r0, %l1, %r1) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>, tensor<2x4x4x2xi32>, tensor<3x3x2x4xi32>) -> (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>)
  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()
  return
}
