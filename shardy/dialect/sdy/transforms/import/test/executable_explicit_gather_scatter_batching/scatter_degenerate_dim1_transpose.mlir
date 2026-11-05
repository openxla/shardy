// RUN: %S/run_sdy_interpreter_test.sh %s %t

// Degenerate scatter with operandDim = 1 rewritten to transpose + add.

//--- part1.mlir

func.func @transformed_scatter(
    %operand: tensor<2x4xi32>, %update: tensor<4x2xi32>) -> tensor<2x4xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %result = "stablehlo.scatter"(%operand, %iota, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 1>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) : (tensor<2x4xi32>, tensor<4x1xi32>, tensor<4x2xi32>) -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.iota dim = 0 : tensor<2x4xi32>
  %update = stablehlo.iota dim = 1 : tensor<4x2xi32>

  %transformed = func.call @transformed_scatter(%operand, %update)
      : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x4xi32>
  %original = func.call @original_scatter(%operand, %update)
      : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x4xi32>

  "check.expect_eq"(%transformed, %original)
      : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  return
}
