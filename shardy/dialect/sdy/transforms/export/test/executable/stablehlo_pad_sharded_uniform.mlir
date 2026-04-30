// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// Padding on sharded Dim 0, with pLow + pHigh = pInt, which is uniform.
func.func @pad_sharded_uniform(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c0, low = [1], high = [0], interior = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0>, limit_indices=array<i64: 2>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2>, limit_indices=array<i64: 4>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@pad_sharded_uniform, @pad_sharded_uniform]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %e0 = stablehlo.constant dense<[0, 1, 0, 2]> : tensor<4xi32>
  %e1 = stablehlo.constant dense<[0, 3, 0, 4]> : tensor<4xi32>

  "check.expect_eq"(%res#0, %e0) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<4xi32>, tensor<4xi32>) -> ()
  return
}
