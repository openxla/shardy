// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: @gather
func.func @gather(%arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"x"}]>}, %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  return %0 : tensor<2x3x2x2xf32>
}

// CHECK-LABEL: @scatter_single_input
func.func @scatter_single_input(%arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<2x3x2xi64>, %arg2: tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>{
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>, tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>
  return %0 : tensor<3x4x2xf32>
}
