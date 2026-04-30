// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// Padding on replicated Dim 1, sharded on Dim 0.
func.func @pad_uniform(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c0, low = [0, 1], high = [0, 1], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<4x2xi32>, tensor<i32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2,0>, limit_indices=array<i64: 4,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@pad_uniform, @pad_uniform]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e0 = stablehlo.constant dense<[[0, 1, 2, 0], [0, 3, 4, 0]]> : tensor<2x4xi32>
  "check.expect_eq"(%res#0, %e0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  return
}

