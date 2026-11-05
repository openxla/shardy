// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Batched scatter lowered from vmap of dynamic_update_slice where operand and
// updates are sharded along an unsliced pass through window dimension (dim 2)
// that also appears in scatter_dims_to_operand_dims with start index 0.
func.func @parallel_batched_scatter_passthrough_window(
  %arg0: tensor<2x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>},
  %arg1: tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
  %arg2: tensor<2x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>})
  -> (tensor<2x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1, 2],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}]>]>
  } : (tensor<2x4x4xi32>, tensor<2x2xi32>, tensor<2x2x4xi32>) -> tensor<2x4x4xi32>
  return %0 : tensor<2x4x4xi32>
}

// Scatter where operand dim 0 is a sharded inserted window dimension (trivial
// slice) on axis "a" and operand dim 2 is a sharded pass through update window
// dimension on axis "b".
func.func @parallel_scatter_mixed_trivial_and_passthrough_dims(
  %arg0: tensor<4x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {"b"}]>},
  %arg1: tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
  %arg2: tensor<2x1x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"b"}]>})
  -> (tensor<4x2x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0, 1, 2],
      index_vector_dim = 1
    >,
    unique_indices = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}, {"b"}]>]>
  } : (tensor<4x2x4xi32>, tensor<2x3xi32>, tensor<2x1x4xi32>) -> tensor<4x2x4xi32>
  return %0 : tensor<4x2x4xi32>
}

//--- part2.mlir

func.func @main() {
  %in = stablehlo.constant dense<0> : tensor<2x4x4xi32>
  %idx = stablehlo.constant dense<[[1, 0], [2, 0]]> : tensor<2x2xi32>
  %upd = stablehlo.constant dense<[
    [[11, 12, 13, 14], [21, 22, 23, 24]],
    [[31, 32, 33, 34], [41, 42, 43, 44]]
  ]> : tensor<2x2x4xi32>

  %expected = func.call @sequential_batched_scatter_passthrough_window(%in, %idx, %upd)
    : (tensor<2x4x4xi32>, tensor<2x2xi32>, tensor<2x2x4xi32>) -> tensor<2x4x4xi32>

  // @mesh = <["a"=2, "b"=2]>: "a"=0 on devices 0, 1; "a"=1 on devices 2, 3.
  %in_a0 = stablehlo.slice %in [0:2, 0:4, 0:2] : (tensor<2x4x4xi32>) -> tensor<2x4x2xi32>
  %in_a1 = stablehlo.slice %in [0:2, 0:4, 2:4] : (tensor<2x4x4xi32>) -> tensor<2x4x2xi32>
  %upd_a0 = stablehlo.slice %upd [0:2, 0:2, 0:2] : (tensor<2x2x4xi32>) -> tensor<2x2x2xi32>
  %upd_a1 = stablehlo.slice %upd [0:2, 0:2, 2:4] : (tensor<2x2x4xi32>) -> tensor<2x2x2xi32>

  %res:4 = "interpreter.run_parallel"(
    %in_a0, %idx, %upd_a0,
    %in_a0, %idx, %upd_a0,
    %in_a1, %idx, %upd_a1,
    %in_a1, %idx, %upd_a1
  ) {
    programs = [[
      @parallel_batched_scatter_passthrough_window,
      @parallel_batched_scatter_passthrough_window,
      @parallel_batched_scatter_passthrough_window,
      @parallel_batched_scatter_passthrough_window
    ]]
  } : (tensor<2x4x2xi32>, tensor<2x2xi32>, tensor<2x2x2xi32>,
       tensor<2x4x2xi32>, tensor<2x2xi32>, tensor<2x2x2xi32>,
       tensor<2x4x2xi32>, tensor<2x2xi32>, tensor<2x2x2xi32>,
       tensor<2x4x2xi32>, tensor<2x2xi32>, tensor<2x2x2xi32>)
      -> (tensor<2x4x4xi32>, tensor<2x4x4xi32>, tensor<2x4x4xi32>, tensor<2x4x4xi32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<2x4x4xi32>, tensor<2x4x4xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<2x4x4xi32>, tensor<2x4x4xi32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<2x4x4xi32>, tensor<2x4x4xi32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<2x4x4xi32>, tensor<2x4x4xi32>) -> ()

  %m_in = stablehlo.constant dense<1> : tensor<4x2x4xi32>
  %m_idx = stablehlo.constant dense<[[1, 0, 0], [3, 1, 0]]> : tensor<2x3xi32>
  %m_upd = stablehlo.constant dense<[
    [[10, 20, 30, 40]],
    [[50, 60, 70, 80]]
  ]> : tensor<2x1x4xi32>

  %m_expected = func.call @sequential_scatter_mixed_trivial_and_passthrough_dims(%m_in, %m_idx, %m_upd)
    : (tensor<4x2x4xi32>, tensor<2x3xi32>, tensor<2x1x4xi32>) -> tensor<4x2x4xi32>

  %m_in_00 = stablehlo.slice %m_in [0:2, 0:2, 0:2] : (tensor<4x2x4xi32>) -> tensor<2x2x2xi32>
  %m_in_01 = stablehlo.slice %m_in [0:2, 0:2, 2:4] : (tensor<4x2x4xi32>) -> tensor<2x2x2xi32>
  %m_in_10 = stablehlo.slice %m_in [2:4, 0:2, 0:2] : (tensor<4x2x4xi32>) -> tensor<2x2x2xi32>
  %m_in_11 = stablehlo.slice %m_in [2:4, 0:2, 2:4] : (tensor<4x2x4xi32>) -> tensor<2x2x2xi32>
  %m_upd_b0 = stablehlo.slice %m_upd [0:2, 0:1, 0:2] : (tensor<2x1x4xi32>) -> tensor<2x1x2xi32>
  %m_upd_b1 = stablehlo.slice %m_upd [0:2, 0:1, 2:4] : (tensor<2x1x4xi32>) -> tensor<2x1x2xi32>

  %m_res:4 = "interpreter.run_parallel"(
    %m_in_00, %m_idx, %m_upd_b0,
    %m_in_01, %m_idx, %m_upd_b1,
    %m_in_10, %m_idx, %m_upd_b0,
    %m_in_11, %m_idx, %m_upd_b1
  ) {
    programs = [[
      @parallel_scatter_mixed_trivial_and_passthrough_dims,
      @parallel_scatter_mixed_trivial_and_passthrough_dims,
      @parallel_scatter_mixed_trivial_and_passthrough_dims,
      @parallel_scatter_mixed_trivial_and_passthrough_dims
    ]]
  } : (tensor<2x2x2xi32>, tensor<2x3xi32>, tensor<2x1x2xi32>,
       tensor<2x2x2xi32>, tensor<2x3xi32>, tensor<2x1x2xi32>,
       tensor<2x2x2xi32>, tensor<2x3xi32>, tensor<2x1x2xi32>,
       tensor<2x2x2xi32>, tensor<2x3xi32>, tensor<2x1x2xi32>)
      -> (tensor<4x2x4xi32>, tensor<4x2x4xi32>, tensor<4x2x4xi32>, tensor<4x2x4xi32>)

  "check.expect_eq"(%m_res#0, %m_expected) : (tensor<4x2x4xi32>, tensor<4x2x4xi32>) -> ()
  "check.expect_eq"(%m_res#1, %m_expected) : (tensor<4x2x4xi32>, tensor<4x2x4xi32>) -> ()
  "check.expect_eq"(%m_res#2, %m_expected) : (tensor<4x2x4xi32>, tensor<4x2x4xi32>) -> ()
  "check.expect_eq"(%m_res#3, %m_expected) : (tensor<4x2x4xi32>, tensor<4x2x4xi32>) -> ()
  return
}
