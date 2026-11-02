// RUN: %S/run_sdy_interpreter_test.sh %s %t

// Degenerate gather with identity iota rewritten to broadcast_in_dim.

//--- part1.mlir

func.func @transformed_gather(%operand: tensor<4xi32>) -> tensor<4x8xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x8xi32>
  %result = "stablehlo.gather"(%operand, %iota) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1>,
    indices_are_sorted = false
  } : (tensor<4xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %result : tensor<4x8xi32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi32>

  %transformed = func.call @transformed_gather(%operand)
      : (tensor<4xi32>) -> tensor<4x8xi32>
  %original = func.call @original_gather(%operand)
      : (tensor<4xi32>) -> tensor<4x8xi32>

  "check.expect_eq"(%transformed, %original)
      : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  return
}
