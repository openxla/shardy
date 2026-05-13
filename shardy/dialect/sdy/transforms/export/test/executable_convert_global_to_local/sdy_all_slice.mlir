// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @all_slice(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %cst = stablehlo.constant dense<[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<4x4xi32>

  %res:2 = "interpreter.run_parallel"(%cst, %cst) {
    programs = [[@all_slice, @all_slice]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e0 = "stablehlo.slice"(%cst) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>
  } : (tensor<4x4xi32>) -> tensor<2x4xi32>
  %e1 = "stablehlo.slice"(%cst) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>
  } : (tensor<4x4xi32>) -> tensor<2x4xi32>

  // Check element-wise correctness for both devices.
  "check.expect_eq"(%res#0, %e0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()

  return
}

