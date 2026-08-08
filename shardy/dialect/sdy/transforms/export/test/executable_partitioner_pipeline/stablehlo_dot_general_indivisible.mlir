// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh_x2_y2 = <["x"=2, "y"=2]>

// Contracting dimension is sharded and indivisible (padded 3->4).
func.func @parallel_dot_contracting_indivisible(
  %arg0: tensor<3x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
  %arg1: tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
  -> (tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}, {"y"}]> : tensor<3x3xf32>
  %1 = sdy.reshard %arg1 <@mesh_x2_y2, [{"y"}, {}]> : tensor<3x5xf32>
  %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {}]>]>} : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  %3 = sdy.reshard %2 <@mesh_x2_y2, [{}, {}]> : tensor<3x5xf32>
  return %3 : tensor<3x5xf32>
}

func.func @sequential_dot_contracting_indivisible(%arg0: tensor<3x3xf32>, %arg1: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

//--- part2.mlir

func.func @main() {
  %lhs_contracting = stablehlo.constant dense<[
    [1.12, -2.45, 3.89],
    [-4.03, 5.76, 0.12],
    [7.34, -8.91, 9.54]
  ]> : tensor<3x3xf32>

  %rhs_contracting = stablehlo.constant dense<[
    [-1.87, 2.34, -3.15, 4.56, 5.12],
    [6.02, -7.89, 8.43, -9.12, 0.54],
    [-11.45, 12.78, -13.01, 14.67, -15.89]
  ]> : tensor<3x5xf32>

  %seq = func.call @sequential_dot_contracting_indivisible(%lhs_contracting, %rhs_contracting) : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  %res:4 = "interpreter.run_parallel"(
    %lhs_contracting, %rhs_contracting,
    %lhs_contracting, %rhs_contracting,
    %lhs_contracting, %rhs_contracting,
    %lhs_contracting, %rhs_contracting
  ) {
    programs = [[
      @parallel_dot_contracting_indivisible,
      @parallel_dot_contracting_indivisible,
      @parallel_dot_contracting_indivisible,
      @parallel_dot_contracting_indivisible
    ]]
  } : (
    tensor<3x3xf32>, tensor<3x5xf32>,
    tensor<3x3xf32>, tensor<3x5xf32>,
    tensor<3x3xf32>, tensor<3x5xf32>,
    tensor<3x3xf32>, tensor<3x5xf32>
  ) -> (
    tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
  )

  "check.expect_eq"(%res#0, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()

  return
}
