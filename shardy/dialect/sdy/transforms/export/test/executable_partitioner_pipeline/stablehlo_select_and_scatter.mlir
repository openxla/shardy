// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_x2 = <["x"=2]>
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

func.func @parallel_select_and_scatter(
    %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %init) <{
    window_dimensions = array<i64: 1, 2>,
    window_strides = array<i64: 1, 2>,
    padding = dense<0> : tensor<2x2xi64>
  }> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %cmp = stablehlo.compare GE, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }, {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %add = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<4x4xf32>, tensor<4x2xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @parallel_select_and_scatter_passthrough(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"b"}]>},
  %arg1: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"b"}]>})
  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}]>}) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    window_dimensions = array<i64: 4>,
    window_strides = array<i64: 4>,
    padding = dense<0> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"b"}]>]>
  } : (tensor<8xf32>, tensor<2xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

func.func @parallel_select_and_scatter_overlapping(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>},
  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 1>,
    padding = dense<[[1, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

func.func @parallel_select_and_scatter_stride_padding_int(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>},
  %arg1: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 2>,
    padding = dense<[[1, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

func.func @parallel_select_and_scatter_2d(
  %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>},
  %arg1: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>]>
  } : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @parallel_select_and_scatter_lt_min(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>},
  %arg1: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare LT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 1>,
    padding = dense<[[1, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<8xi32>, tensor<8xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

func.func @parallel_select_and_scatter_passthrough_pad_high(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>},
  %arg1: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    window_dimensions = array<i64: 4>,
    window_strides = array<i64: 4>,
    padding = dense<[[0, 2]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<8xf32>, tensor<2xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

func.func @parallel_select_and_scatter_unsigned_i32(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>},
  %arg1: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %c) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 1>,
    padding = dense<[[1, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<8xi32>, tensor<8xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

//--- part2.mlir

func.func @main() {
  %op_f32 = stablehlo.constant dense<[1.0, 7.0, 3.0, 6.0, 4.0, 8.0, 5.0, 2.0]> : tensor<8xf32>
  %src_2_f32 = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %src_8_f32 = stablehlo.constant dense<[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]> : tensor<8xf32>

  %op_i32 = stablehlo.constant dense<[1, 7, 3, 6, 4, 8, 5, 2]> : tensor<8xi32>
  %src_4_i32 = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi32>
  %src_8_i32 = stablehlo.constant dense<[10, 20, 30, 40, 50, 60, 70, 80]> : tensor<8xi32>

  %o_f32_0 = "stablehlo.slice"(%op_f32) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xf32>) -> tensor<4xf32>
  %o_f32_1 = "stablehlo.slice"(%op_f32) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xf32>) -> tensor<4xf32>

  %o_i32_0 = "stablehlo.slice"(%op_i32) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %o_i32_1 = "stablehlo.slice"(%op_i32) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>

  // 1. Passthrough (non-overlapping window [4], stride [4], pad 0).
  %seq_pass = func.call @sequential_select_and_scatter_passthrough(%op_f32, %src_2_f32) : (tensor<8xf32>, tensor<2xf32>) -> tensor<8xf32>
  %s2_0 = "stablehlo.slice"(%src_2_f32) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf32>) -> tensor<1xf32>
  %s2_1 = "stablehlo.slice"(%src_2_f32) {start_indices = array<i64: 1>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<2xf32>) -> tensor<1xf32>
  %res_pass:4 = "interpreter.run_parallel"(%o_f32_0, %s2_0, %o_f32_1, %s2_1, %o_f32_0, %s2_0, %o_f32_1, %s2_1) {
    programs = [[@parallel_select_and_scatter_passthrough, @parallel_select_and_scatter_passthrough, @parallel_select_and_scatter_passthrough, @parallel_select_and_scatter_passthrough]]
  } : (tensor<4xf32>, tensor<1xf32>, tensor<4xf32>, tensor<1xf32>, tensor<4xf32>, tensor<1xf32>, tensor<4xf32>, tensor<1xf32>) ->
      (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  "check.expect_eq"(%res_pass#0, %seq_pass) : (tensor<8xf32>, tensor<8xf32>) -> ()
  "check.expect_eq"(%res_pass#1, %seq_pass) : (tensor<8xf32>, tensor<8xf32>) -> ()
  "check.expect_eq"(%res_pass#2, %seq_pass) : (tensor<8xf32>, tensor<8xf32>) -> ()
  "check.expect_eq"(%res_pass#3, %seq_pass) : (tensor<8xf32>, tensor<8xf32>) -> ()

  // 2. Overlapping window (window [3], stride [1], pad [[1, 1]]).
  %seq_ov = func.call @sequential_select_and_scatter_overlapping(%op_f32, %src_8_f32) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %s8_f32_0 = "stablehlo.slice"(%src_8_f32) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xf32>) -> tensor<4xf32>
  %s8_f32_1 = "stablehlo.slice"(%src_8_f32) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xf32>) -> tensor<4xf32>
  %res_ov:2 = "interpreter.run_parallel"(%o_f32_0, %s8_f32_0, %o_f32_1, %s8_f32_1) {
    programs = [[@parallel_select_and_scatter_overlapping, @parallel_select_and_scatter_overlapping]]
  } : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  "check.expect_eq"(%res_ov#0, %seq_ov) : (tensor<8xf32>, tensor<8xf32>) -> ()
  "check.expect_eq"(%res_ov#1, %seq_ov) : (tensor<8xf32>, tensor<8xf32>) -> ()

  // 3. Stride + padding with integer tensor (window [3], stride [2], pad [[1, 1]]).
  %seq_sp = func.call @sequential_select_and_scatter_stride_padding_int(%op_i32, %src_4_i32) : (tensor<8xi32>, tensor<4xi32>) -> tensor<8xi32>
  %s4_i32_0 = "stablehlo.slice"(%src_4_i32) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s4_i32_1 = "stablehlo.slice"(%src_4_i32) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %res_sp:2 = "interpreter.run_parallel"(%o_i32_0, %s4_i32_0, %o_i32_1, %s4_i32_1) {
    programs = [[@parallel_select_and_scatter_stride_padding_int, @parallel_select_and_scatter_stride_padding_int]]
  } : (tensor<4xi32>, tensor<2xi32>, tensor<4xi32>, tensor<2xi32>) -> (tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_sp#0, %seq_sp) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_sp#1, %seq_sp) : (tensor<8xi32>, tensor<8xi32>) -> ()

  // 4. 2D sharding with halo exchange on both dims (window [3, 3], stride [1, 1], pad [[1, 1], [1, 1]]).
  %op_2d = stablehlo.constant dense<[
    [ 1.0,  2.0,  3.0,  4.0],
    [ 5.0, 16.0,  7.0,  8.0],
    [ 9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0,  6.0]
  ]> : tensor<4x4xf32>
  %src_2d = stablehlo.constant dense<[
    [ 1.0,  2.0,  3.0,  4.0],
    [ 5.0,  6.0,  7.0,  8.0],
    [ 9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
  ]> : tensor<4x4xf32>
  %seq_2d = func.call @sequential_select_and_scatter_2d(%op_2d, %src_2d) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %o2d_00 = "stablehlo.slice"(%op_2d) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %o2d_01 = "stablehlo.slice"(%op_2d) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %o2d_10 = "stablehlo.slice"(%op_2d) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %o2d_11 = "stablehlo.slice"(%op_2d) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %s2d_00 = "stablehlo.slice"(%src_2d) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %s2d_01 = "stablehlo.slice"(%src_2d) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %s2d_10 = "stablehlo.slice"(%src_2d) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %s2d_11 = "stablehlo.slice"(%src_2d) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %res_2d:4 = "interpreter.run_parallel"(%o2d_00, %s2d_00, %o2d_01, %s2d_01, %o2d_10, %s2d_10, %o2d_11, %s2d_11) {
    programs = [[@parallel_select_and_scatter_2d, @parallel_select_and_scatter_2d, @parallel_select_and_scatter_2d, @parallel_select_and_scatter_2d]]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) ->
      (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
  "check.expect_eq"(%res_2d#0, %seq_2d) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()
  "check.expect_eq"(%res_2d#1, %seq_2d) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()
  "check.expect_eq"(%res_2d#2, %seq_2d) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()
  "check.expect_eq"(%res_2d#3, %seq_2d) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  // 5. LT (min) comparison direction (window [3], stride [1], pad [[1, 1]]).
  %seq_lt = func.call @sequential_select_and_scatter_lt_min(%op_i32, %src_8_i32) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %s8_i32_0 = "stablehlo.slice"(%src_8_i32) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %s8_i32_1 = "stablehlo.slice"(%src_8_i32) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %res_lt:2 = "interpreter.run_parallel"(%o_i32_0, %s8_i32_0, %o_i32_1, %s8_i32_1) {
    programs = [[@parallel_select_and_scatter_lt_min, @parallel_select_and_scatter_lt_min]]
  } : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> (tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_lt#0, %seq_lt) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_lt#1, %seq_lt) : (tensor<8xi32>, tensor<8xi32>) -> ()

  // 6. Passthrough with unused pad_high > 0 (window [4], stride [4], pad [[0, 2]]).
  %seq_pass_pad = func.call @sequential_select_and_scatter_passthrough_pad_high(%op_f32, %src_2_f32) : (tensor<8xf32>, tensor<2xf32>) -> tensor<8xf32>
  %res_pass_pad:2 = "interpreter.run_parallel"(%o_f32_0, %s2_0, %o_f32_1, %s2_1) {
    programs = [[@parallel_select_and_scatter_passthrough_pad_high, @parallel_select_and_scatter_passthrough_pad_high]]
  } : (tensor<4xf32>, tensor<1xf32>, tensor<4xf32>, tensor<1xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  "check.expect_eq"(%res_pass_pad#0, %seq_pass_pad) : (tensor<8xf32>, tensor<8xf32>) -> ()
  "check.expect_eq"(%res_pass_pad#1, %seq_pass_pad) : (tensor<8xf32>, tensor<8xf32>) -> ()

  // 7. UNSIGNED i32 GT comparison direction (window [3], stride [1], pad [[1, 1]]).
  %seq_u32 = func.call @sequential_select_and_scatter_unsigned_i32(%op_i32, %src_8_i32) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %res_u32:2 = "interpreter.run_parallel"(%o_i32_0, %s8_i32_0, %o_i32_1, %s8_i32_1) {
    programs = [[@parallel_select_and_scatter_unsigned_i32, @parallel_select_and_scatter_unsigned_i32]]
  } : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> (tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_u32#0, %seq_u32) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_u32#1, %seq_u32) : (tensor<8xi32>, tensor<8xi32>) -> ()

  // 8. 2D tensor sharded on non-windowed batch dim (window [1, 2], stride [1, 2], pad 0).
  %operand = stablehlo.constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [100.0, 200.0, 300.0, 400.0],
    [500.0, 600.0, 700.0, 800.0]
  ]> : tensor<4x4xf32>
  %source = stablehlo.constant dense<[
    [10.0, 20.0],
    [30.0, 40.0],
    [10.0, 20.0],
    [30.0, 40.0]
  ]> : tensor<4x2xf32>

  %seq = func.call @sequential_select_and_scatter(%operand, %source) : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>

  %operand_shard0 = "stablehlo.slice"(%operand) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %operand_shard1 = "stablehlo.slice"(%operand) {
    start_indices = array<i64: 2, 0>,
    limit_indices = array<i64: 4, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<4x4xf32>) -> tensor<2x4xf32>

  %source_shard0 = "stablehlo.slice"(%source) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %source_shard1 = "stablehlo.slice"(%source) {
    start_indices = array<i64: 2, 0>,
    limit_indices = array<i64: 4, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%operand_shard0, %source_shard0, %operand_shard1, %source_shard1) {
    programs = [[@parallel_select_and_scatter, @parallel_select_and_scatter]]
  } : (tensor<2x4xf32>, tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>

  check.expect_eq %seq, %par : tensor<4x4xf32>

  return
}
