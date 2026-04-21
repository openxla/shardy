// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_conv(
  %arg0: tensor<2x4x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>},
  %arg1: tensor<3x3x1x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>}
) -> (tensor<2x2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}, {"x"}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]}
    {
      feature_group_count = 2 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}, {}, {"x"}]>]>
    } : (tensor<2x4x4x2xi32>, tensor<3x3x1x4xi32>) -> tensor<2x2x2x4xi32>
  return %0 : tensor<2x2x2x4xi32>
}

//--- part2.mlir
func.func @main() {
  %iota64 = stablehlo.iota dim = 0 : tensor<64xi32>
  %lhs_base = stablehlo.reshape %iota64 : (tensor<64xi32>) -> tensor<2x4x4x2xi32>
  %iota36 = stablehlo.iota dim = 0 : tensor<36xi32>
  %rhs = stablehlo.reshape %iota36 : (tensor<36xi32>) -> tensor<3x3x1x4xi32>

  // Prepare shards: Sharding the "groups" means slicing LHS feature and RHS output feature.
  %l0 = "stablehlo.slice"(%lhs_base) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 2,4,4,1>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<2x4x4x1xi32>
  %l1 = "stablehlo.slice"(%lhs_base) {start_indices=array<i64: 0,0,0,1>, limit_indices=array<i64: 2,4,4,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x4x4x2xi32>) -> tensor<2x4x4x1xi32>

  %r0 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 3,3,1,2>, strides=array<i64: 1,1,1,1>} : (tensor<3x3x1x4xi32>) -> tensor<3x3x1x2xi32>
  %r1 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0,0,0,2>, limit_indices=array<i64: 3,3,1,4>, strides=array<i64: 1,1,1,1>} : (tensor<3x3x1x4xi32>) -> tensor<3x3x1x2xi32>

  %expected = func.call @sequential_conv(%lhs_base, %rhs) : (tensor<2x4x4x2xi32>, tensor<3x3x1x4xi32>) -> tensor<2x2x2x4xi32>

  %res:2 = "interpreter.run_parallel"(%l0, %r0, %l1, %r1) {
    programs = [[@parallel_conv, @parallel_conv]]
  } : (tensor<2x4x4x1xi32>, tensor<3x3x1x2xi32>, tensor<2x4x4x1xi32>, tensor<3x3x1x2xi32>) -> (tensor<2x2x2x2xi32>, tensor<2x2x2x2xi32>)

  %actual = "stablehlo.concatenate"(%res#0, %res#1) {dimension = 3 : i64} : (tensor<2x2x2x2xi32>, tensor<2x2x2x2xi32>) -> tensor<2x2x2x4xi32>
  "check.expect_eq"(%actual, %expected) : (tensor<2x2x2x4xi32>, tensor<2x2x2x4xi32>) -> ()
  return
}
