// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>

func.func @parallel_reshard_multi_axis_all_to_all(
    %arg0: tensor<4x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "z", "y"}]>})
    -> (tensor<4x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {"z"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {"z"}]>]>} : tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
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
  return
}
