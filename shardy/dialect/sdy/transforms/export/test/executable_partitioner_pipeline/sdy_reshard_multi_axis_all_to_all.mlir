// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_2d = <["a"=2, "b"=2]>

func.func @parallel_reshard_multi_axis_all_to_all(
    %arg0: tensor<4x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "z", "y"}]>})
    -> (tensor<4x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {"z"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {"z"}]>]>} : tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

func.func @parallel_reshard_multi_axis_all_to_all_uneven_padded_mismatch(
    %arg0: tensor<5x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"b", "a"}]>})
    -> (tensor<5x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"a", "b"}, {}]>]>} : tensor<5x4xi32>
  return %0 : tensor<5x4xi32>
}

func.func @parallel_reshard_multi_axis_all_to_all_uneven_same_padded_size(
    %arg0: tensor<7x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"b", "a"}]>})
    -> (tensor<7x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"a", "b"}, {}]>]>} : tensor<7x4xi32>
  return %0 : tensor<7x4xi32>
}

//--- part2.mlir
func.func @main() {
  %flat = stablehlo.iota dim = 0 : tensor<64xi32>
  %input = stablehlo.reshape %flat : (tensor<64xi32>) -> tensor<4x16xi32>
  %expected = func.call @sequential_reshard_multi_axis_all_to_all(%input)
      : (tensor<4x16xi32>) -> tensor<4x16xi32>

  %in0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in2 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 4, 6>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in3 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 6>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in4 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 8>, limit_indices = array<i64: 4, 10>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in5 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 10>, limit_indices = array<i64: 4, 12>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in6 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 12>, limit_indices = array<i64: 4, 14>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>
  %in7 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 14>, limit_indices = array<i64: 4, 16>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<4x2xi32>

  // Device (x, y, z) has linear index 4*x + 2*y + z, while dim 1 sharded by
  // {"x", "z", "y"} assigns slice 4*x + 2*z + y.
  %res:8 = "interpreter.run_parallel"(%in0, %in2, %in1, %in3, %in4, %in6, %in5, %in7) {
    programs = [[
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all,
      @parallel_reshard_multi_axis_all_to_all
    ]]
  } : (tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>,
       tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>) ->
      (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>,
       tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>)

  %exp0 = "stablehlo.slice"(%expected) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 8>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp1 = "stablehlo.slice"(%expected) {start_indices = array<i64: 0, 8>, limit_indices = array<i64: 1, 16>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp2 = "stablehlo.slice"(%expected) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp3 = "stablehlo.slice"(%expected) {start_indices = array<i64: 1, 8>, limit_indices = array<i64: 2, 16>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp4 = "stablehlo.slice"(%expected) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 8>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp5 = "stablehlo.slice"(%expected) {start_indices = array<i64: 2, 8>, limit_indices = array<i64: 3, 16>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp6 = "stablehlo.slice"(%expected) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>
  %exp7 = "stablehlo.slice"(%expected) {start_indices = array<i64: 3, 8>, limit_indices = array<i64: 4, 16>, strides = array<i64: 1, 1>} : (tensor<4x16xi32>) -> tensor<1x8xi32>

  "check.expect_eq"(%res#0, %exp0) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#1, %exp1) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#2, %exp2) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#3, %exp3) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#4, %exp4) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#5, %exp5) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#6, %exp6) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()
  "check.expect_eq"(%res#7, %exp7) : (tensor<1x8xi32>, tensor<1x8xi32>) -> ()

  %flat_u = stablehlo.iota dim = 0 : tensor<20xi32>
  %c1_u = stablehlo.constant dense<1> : tensor<20xi32>
  %flat1_u = stablehlo.add %flat_u, %c1_u : tensor<20xi32>
  %in_u = stablehlo.reshape %flat1_u : (tensor<20xi32>) -> tensor<5x4xi32>
  %exp_u = func.call @sequential_reshard_multi_axis_all_to_all_uneven_padded_mismatch(%in_u)
      : (tensor<5x4xi32>) -> tensor<5x4xi32>
  %u0 = "stablehlo.slice"(%in_u) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 5, 1>, strides = array<i64: 1, 1>} : (tensor<5x4xi32>) -> tensor<5x1xi32>
  %u1 = "stablehlo.slice"(%in_u) {start_indices = array<i64: 0, 1>, limit_indices = array<i64: 5, 2>, strides = array<i64: 1, 1>} : (tensor<5x4xi32>) -> tensor<5x1xi32>
  %u2 = "stablehlo.slice"(%in_u) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 5, 3>, strides = array<i64: 1, 1>} : (tensor<5x4xi32>) -> tensor<5x1xi32>
  %u3 = "stablehlo.slice"(%in_u) {start_indices = array<i64: 0, 3>, limit_indices = array<i64: 5, 4>, strides = array<i64: 1, 1>} : (tensor<5x4xi32>) -> tensor<5x1xi32>
  %res_u:4 = "interpreter.run_parallel"(%u0, %u2, %u1, %u3) {
    programs = [[
      @parallel_reshard_multi_axis_all_to_all_uneven_padded_mismatch,
      @parallel_reshard_multi_axis_all_to_all_uneven_padded_mismatch,
      @parallel_reshard_multi_axis_all_to_all_uneven_padded_mismatch,
      @parallel_reshard_multi_axis_all_to_all_uneven_padded_mismatch
    ]]
  } : (tensor<5x1xi32>, tensor<5x1xi32>, tensor<5x1xi32>, tensor<5x1xi32>) ->
      (tensor<5x4xi32>, tensor<5x4xi32>, tensor<5x4xi32>, tensor<5x4xi32>)
  "check.expect_eq"(%res_u#0, %exp_u) : (tensor<5x4xi32>, tensor<5x4xi32>) -> ()
  "check.expect_eq"(%res_u#1, %exp_u) : (tensor<5x4xi32>, tensor<5x4xi32>) -> ()
  "check.expect_eq"(%res_u#2, %exp_u) : (tensor<5x4xi32>, tensor<5x4xi32>) -> ()
  "check.expect_eq"(%res_u#3, %exp_u) : (tensor<5x4xi32>, tensor<5x4xi32>) -> ()

  %flat_s = stablehlo.iota dim = 0 : tensor<28xi32>
  %c1_s = stablehlo.constant dense<1> : tensor<28xi32>
  %flat1_s = stablehlo.add %flat_s, %c1_s : tensor<28xi32>
  %in_s = stablehlo.reshape %flat1_s : (tensor<28xi32>) -> tensor<7x4xi32>
  %exp_s = func.call @sequential_reshard_multi_axis_all_to_all_uneven_same_padded_size(%in_s)
      : (tensor<7x4xi32>) -> tensor<7x4xi32>
  %s0 = "stablehlo.slice"(%in_s) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 7, 1>, strides = array<i64: 1, 1>} : (tensor<7x4xi32>) -> tensor<7x1xi32>
  %s1 = "stablehlo.slice"(%in_s) {start_indices = array<i64: 0, 1>, limit_indices = array<i64: 7, 2>, strides = array<i64: 1, 1>} : (tensor<7x4xi32>) -> tensor<7x1xi32>
  %s2 = "stablehlo.slice"(%in_s) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 7, 3>, strides = array<i64: 1, 1>} : (tensor<7x4xi32>) -> tensor<7x1xi32>
  %s3 = "stablehlo.slice"(%in_s) {start_indices = array<i64: 0, 3>, limit_indices = array<i64: 7, 4>, strides = array<i64: 1, 1>} : (tensor<7x4xi32>) -> tensor<7x1xi32>
  %res_s:4 = "interpreter.run_parallel"(%s0, %s2, %s1, %s3) {
    programs = [[
      @parallel_reshard_multi_axis_all_to_all_uneven_same_padded_size,
      @parallel_reshard_multi_axis_all_to_all_uneven_same_padded_size,
      @parallel_reshard_multi_axis_all_to_all_uneven_same_padded_size,
      @parallel_reshard_multi_axis_all_to_all_uneven_same_padded_size
    ]]
  } : (tensor<7x1xi32>, tensor<7x1xi32>, tensor<7x1xi32>, tensor<7x1xi32>) ->
      (tensor<7x4xi32>, tensor<7x4xi32>, tensor<7x4xi32>, tensor<7x4xi32>)
  "check.expect_eq"(%res_s#0, %exp_s) : (tensor<7x4xi32>, tensor<7x4xi32>) -> ()
  "check.expect_eq"(%res_s#1, %exp_s) : (tensor<7x4xi32>, tensor<7x4xi32>) -> ()
  "check.expect_eq"(%res_s#2, %exp_s) : (tensor<7x4xi32>, tensor<7x4xi32>) -> ()
  "check.expect_eq"(%res_s#3, %exp_s) : (tensor<7x4xi32>, tensor<7x4xi32>) -> ()
  return
}
