// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_sort(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %p = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%p) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[
    [4, 1, 3, 2], [8, 5, 7, 6],
    [400, 100, 300, 200], [800, 500, 700, 600]
  ]> : tensor<4x4xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 2, 0>, limit_indices=array<i64: 4, 4>, strides=array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x4xi32>

  %seq = func.call @sequential_sort(%input) : (tensor<4x4xi32>) -> tensor<4x4xi32>

  %pars:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_sort, @parallel_sort]]
  } : (tensor<2x4xi32>, tensor<2x4xi32>) -> (tensor<2x4xi32>, tensor<2x4xi32>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {dimension = 0 : i64}
    : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<4x4xi32>

  // Verify that all 4 rows were sorted correctly and stayed within their shards.
  "check.expect_eq"(%seq, %par) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()

  return
}
