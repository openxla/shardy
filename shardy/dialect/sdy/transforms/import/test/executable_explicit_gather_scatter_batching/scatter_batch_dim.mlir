// RUN: %S/run_sdy_interpreter_test.sh %s %t

// Scatter with implicit batch dim via iota concat.

//--- part1.mlir

func.func @transformed_scatter(%operand: tensor<4x8xi32>,
    %offset: tensor<4x1xi32>, %updates: tensor<4xi32>) -> tensor<4x8xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    stablehlo.return %arg1 : tensor<i32>
  }) : (tensor<4x8xi32>, tensor<4x2xi32>, tensor<4xi32>) -> tensor<4x8xi32>
  return %result : tensor<4x8xi32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.constant dense<0> : tensor<4x8xi32>
  %offset = stablehlo.constant dense<[[2], [5], [1], [7]]> : tensor<4x1xi32>
  %updates = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi32>

  %transformed = func.call @transformed_scatter(%operand, %offset, %updates)
      : (tensor<4x8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<4x8xi32>

  %original = func.call @original_scatter(%operand, %offset, %updates)
      : (tensor<4x8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<4x8xi32>

  "check.expect_eq"(%transformed, %original)
      : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  return
}
