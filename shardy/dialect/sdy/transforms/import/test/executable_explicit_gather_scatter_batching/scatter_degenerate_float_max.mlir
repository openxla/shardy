// RUN: %S/run_sdy_interpreter_test.sh %s %t

// Degenerate scatter with float maximum reduction rewritten to reduce + maximum.

//--- part1.mlir

func.func @transformed_scatter(
    %operand: tensor<8x2xf32>, %update: tensor<8x4x2xf32>) -> tensor<8x2xf32> {
  %iota = stablehlo.iota dim = 1 : tensor<1x8x4xi32>
  %result = "stablehlo.scatter"(%operand, %iota, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    stablehlo.return %max : tensor<f32>
  }) : (tensor<8x2xf32>, tensor<1x8x4xi32>, tensor<8x4x2xf32>) -> tensor<8x2xf32>
  return %result : tensor<8x2xf32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.iota dim = 0 : tensor<8x2xf32>
  %update = stablehlo.iota dim = 1 : tensor<8x4x2xf32>

  %transformed = func.call @transformed_scatter(%operand, %update)
      : (tensor<8x2xf32>, tensor<8x4x2xf32>) -> tensor<8x2xf32>
  %original = func.call @original_scatter(%operand, %update)
      : (tensor<8x2xf32>, tensor<8x4x2xf32>) -> tensor<8x2xf32>

  "check.expect_eq"(%transformed, %original)
      : (tensor<8x2xf32>, tensor<8x2xf32>) -> ()
  return
}
