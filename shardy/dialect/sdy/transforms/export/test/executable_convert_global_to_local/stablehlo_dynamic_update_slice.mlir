// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// Single-element update (slice size 1 along partitioned dim 0) with a dynamic
// start index.
func.func @parallel_dynamic_index_single_element(
    %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<1x2xf32>,
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<4x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>)
      -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// Multi-element update (slice size 2 along partitioned dim 0) with an
// out-of-bounds constant start index (20), which clamps to 2 (`4 - 2`) and
// lies entirely within shard 1 (`[2, 4)`).
func.func @parallel_static_index_single_shard_with_clamping(
    %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2x2xf32>)
    -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %c20 = stablehlo.constant dense<20> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c20, %c0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>)
      -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 2, 0>,
    limit_indices = array<i64: 4, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  // 1. Test dynamic index single-element update on shard 1 (row 2).
  %update_1x2 = stablehlo.constant dense<[[90.0, 91.0]]> : tensor<1x2xf32>
  %idx_row2 = stablehlo.constant dense<2> : tensor<i32>
  %idx_col0 = stablehlo.constant dense<0> : tensor<i32>

  %seq_dyn = func.call @sequential_dynamic_index_single_element(
      %input, %update_1x2, %idx_row2, %idx_col0)
      : (tensor<4x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>

  %pars_dyn:2 = "interpreter.run_parallel"(
      %shard0, %update_1x2, %idx_row2, %idx_col0,
      %shard1, %update_1x2, %idx_row2, %idx_col0) {
    programs = [[@parallel_dynamic_index_single_element,
                 @parallel_dynamic_index_single_element]]
  } : (tensor<2x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>,
       tensor<2x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>)

  %par_dyn = "stablehlo.concatenate"(%pars_dyn#0, %pars_dyn#1) {
    dimension = 0 : i64
  } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  check.expect_eq %seq_dyn, %par_dyn : tensor<4x2xf32>

  // 2. Test static index single-shard update with clamping (index 20 -> 2, shard 1).
  %update_2x2 = stablehlo.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf32>
  %seq_clamp = func.call @sequential_static_index_single_shard_with_clamping(
      %input, %update_2x2) : (tensor<4x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>

  %pars_clamp:2 = "interpreter.run_parallel"(
      %shard0, %update_2x2, %shard1, %update_2x2) {
    programs = [[@parallel_static_index_single_shard_with_clamping,
                 @parallel_static_index_single_shard_with_clamping]]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>)

  %par_clamp = "stablehlo.concatenate"(%pars_clamp#0, %pars_clamp#1) {
    dimension = 0 : i64
  } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  check.expect_eq %seq_clamp, %par_clamp : tensor<4x2xf32>

  return
}
