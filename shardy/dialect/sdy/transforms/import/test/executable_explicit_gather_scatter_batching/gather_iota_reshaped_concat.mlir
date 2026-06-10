// RUN: %S/run_sdy_interpreter_test.sh %s %t

// Gather with iota -> reshape -> concat.

//--- part1.mlir

func.func @transformed_gather(%operand: tensor<3x4xi32>,
    %other: tensor<3x1xi32>) -> tensor<3x1xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<3xi32>
  %reshaped = stablehlo.reshape %iota : (tensor<3xi32>) -> tensor<3x1xi32>
  %indices = stablehlo.concatenate %reshaped, %other, dim = 1
      : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<3x4xi32>, tensor<3x2xi32>) -> tensor<3x1xi32>
  return %result : tensor<3x1xi32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.constant dense<[
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33]]> : tensor<3x4xi32>
  %other = stablehlo.constant dense<[[3], [0], [2]]> : tensor<3x1xi32>

  %transformed = func.call @transformed_gather(%operand, %other)
      : (tensor<3x4xi32>, tensor<3x1xi32>) -> tensor<3x1xi32>
  %original = func.call @original_gather(%operand, %other)
      : (tensor<3x4xi32>, tensor<3x1xi32>) -> tensor<3x1xi32>

  "check.expect_eq"(%transformed, %original)
      : (tensor<3x1xi32>, tensor<3x1xi32>) -> ()
  return
}
