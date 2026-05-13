// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sequential_dot(%arg0: tensor<2x4xi32>, %arg1: tensor<4x2xi32>) -> tensor<2x2xi32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

func.func @parallel_dot(
  %arg0: tensor<2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
  %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}
) -> (tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}

//--- part2.mlir

func.func @main() {
  %lhs = stablehlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
  %rhs = stablehlo.constant dense<[[1, 16], [2, 32], [4, 64], [8, 128]]> : tensor<4x2xi32>

  %seq = func.call @sequential_dot(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>

  // Device 0: lhs[:, 0:2], rhs[0:2, :]
  %lhs0 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 2>, strides=array<i64: 1, 1>} : (tensor<2x4xi32>) -> tensor<2x2xi32>
  %rhs0 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 2>, strides=array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  // Device 1: lhs[:, 2:4], rhs[2:4, :]
  %lhs1 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0, 2>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xi32>) -> tensor<2x2xi32>
  %rhs1 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 2, 0>, limit_indices=array<i64: 4, 2>, strides=array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %pars:2 = "interpreter.run_parallel"(%lhs0, %rhs0, %lhs1, %rhs1) {
    programs = [[@parallel_dot, @parallel_dot]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi32>, tensor<2x2xi32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()

  return
}

