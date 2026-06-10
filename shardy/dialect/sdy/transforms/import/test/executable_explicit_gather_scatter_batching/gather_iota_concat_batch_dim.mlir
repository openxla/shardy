// RUN: %S/run_sdy_interpreter_test.sh %s %t

// Basic gather with iota in concatenated indices.

//--- part1.mlir

func.func @transformed_gather(%operand: tensor<4x8xi32>,
    %offset: tensor<4x1xi32>) -> tensor<4x1xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xi32>, tensor<4x2xi32>) -> tensor<4x1xi32>
  return %result : tensor<4x1xi32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.constant dense<[
    [0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
  %offset = stablehlo.constant dense<[[2], [5], [1], [7]]> : tensor<4x1xi32>

  %transformed = func.call @transformed_gather(%operand, %offset)
      : (tensor<4x8xi32>, tensor<4x1xi32>) -> tensor<4x1xi32>
  %original = func.call @original_gather(%operand, %offset)
      : (tensor<4x8xi32>, tensor<4x1xi32>) -> tensor<4x1xi32>

  "check.expect_eq"(%transformed, %original)
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> ()
  return
}
