// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_x2 = <["x"=2]>
sdy.mesh @mesh_x4 = <["x"=4]>
sdy.mesh @mesh_x8 = <["x"=8]>
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

// 2D sum-pooling (3x3 window, stride 1, symmetric padding 1) on i32 tensor
// sharded 2x2 across both spatial dimensions.
func.func @parallel_reduce_window_2d(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// 2D sum-pooling (3x3 window, stride 1, symmetric padding 1) on f32 tensor
// sharded 2x2 across both spatial dimensions.
func.func @parallel_reduce_window_f32(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %add : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// 2D max-pooling (3x3 window, stride 1, symmetric padding 1) with a non-zero
// init value (-1000) to verify HALO border padding uses the init value.
func.func @parallel_reduce_window_max(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<-1000> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %max = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %max : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// Mixed sharding: dim 0 has a 3-element window (HALO exchange along "a") while
// dim 1 has a 1-element passthrough window (sharded along "b" without HALO).
func.func @parallel_reduce_window_mixed_sharding(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// Variadic 2D reduce_window over two inputs and two init values, sharded 2x2.
func.func @parallel_reduce_window_variadic(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>},
  %arg1: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>},
      tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %c0, %c1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %add0 = stablehlo.add %arg2, %arg4 : tensor<i32>
      %add1 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %add0, %add1 : tensor<i32>, tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>, <@mesh_a2_b2, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<i32>, tensor<i32>) -> (tensor<8x8xi32>, tensor<8x8xi32>)
  return %0#0, %0#1 : tensor<8x8xi32>, tensor<8x8xi32>
}

// 1D window (size 3, stride 2, padding [1, 1]) on tensor<8xi32> ->
// tensor<4xi32>, requiring 1 left-hop HALO exchange.
func.func @parallel_reduce_window_padding_and_stride(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"b"}]>})
  -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 2>,
    padding = dense<[[1, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"b"}]>]>
  } : (tensor<8xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 1D window with window_dilations = [2] (dilated window size 5) and padding
// [2, 2] on tensor<8xi32> -> tensor<8xi32>.
func.func @parallel_reduce_window_window_dilations(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 1>,
    window_dilations = array<i64: 2>,
    padding = dense<[[2, 2]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<8xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// 1D window with base_dilations = [2], stride 1, and padding [1, 2] where
// sOut * stride = 4 is divisible by baseDilation = 2 (phase-aligned).
func.func @parallel_reduce_window_base_dilations(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 1>,
    base_dilations = array<i64: 2>,
    padding = dense<[[1, 2]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// 1D window with both base_dilations = [3] and window_dilations = [2], stride
// 1, and padding [4, 4] on tensor<4xi32> -> tensor<14xi32> where sOut = 7 is
// not divisible by baseDilation = 3, exercising post-op dynamic slicing.
func.func @parallel_reduce_window_complex_dilations(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}]>})
  -> (tensor<14xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 1>,
    window_dilations = array<i64: 2>,
    base_dilations = array<i64: 3>,
    padding = dense<[[4, 4]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>
  } : (tensor<4xi32>, tensor<i32>) -> tensor<14xi32>
  return %0 : tensor<14xi32>
}

// Asymmetric causal padding [[2, 0], [0, 0]] with a 3x1 window (leftHops = 1,
// rightHops = 0).
func.func @parallel_reduce_window_asymmetric_padding(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}, {}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// Negative high padding [[1, -1], [0, 0]] with a 3x1 window on tensor<8x8xi32>
// -> tensor<6x8xi32>, trimming 1 element from the high end before reduction.
func.func @parallel_reduce_window_negative_padding(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}, {}]>})
  -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, -1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// Indivisible input and output dimension (7x8 sharded 2-way on dim 0) with a
// 3x1 window, stride 1, and padding [[1, 1], [0, 0]], requiring high-side
// input padding and output slicing.
func.func @parallel_reduce_window_indivisible(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}, {}]>})
  -> (tensor<7x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 7, 8>,
    strides = array<i64: 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>
  } : (tensor<8x8xi32>) -> tensor<7x8xi32>

  %1 = "stablehlo.reduce_window"(%0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>
  } : (tensor<7x8xi32>, tensor<i32>) -> tensor<7x8xi32>
  return %1 : tensor<7x8xi32>
}

// Indivisible input dimension (7x8 sharded 2-way on dim 0) with stride 2 and
// padding [[1, 1], [0, 0]], producing a divisible 4x8 output.
func.func @parallel_reduce_window_indivisible_padding_stride(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{"x"}, {}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 7, 8>,
    strides = array<i64: 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>
  } : (tensor<8x8xi32>) -> tensor<7x8xi32>

  %1 = "stablehlo.reduce_window"(%0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>
  } : (tensor<7x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  return %1 : tensor<4x8xi32>
}

// Multi-hop HALO exchange (window size 5, padding [2, 2] on tensor<8x8xi32>
// sharded 8-way with sIn = 1, requiring leftHops = 2 and rightHops = 2).
func.func @parallel_reduce_window_multi_hop(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{"x"}, {}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 5, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x8, [{"x"}, {}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// Phase-aligned stride > 1 with base_dilations = [3] (sOut * stride = 3 * 2 =
// 6 is divisible by baseDilation = 3), using residual local padding [[2, 1]].
func.func @parallel_reduce_window_base_dilation_phase_aligned_stride(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{"x"}]>})
  -> (tensor<12xi32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %add = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %add : tensor<i32>
  }) {
    window_dimensions = array<i64: 6>,
    window_strides = array<i64: 2>,
    base_dilations = array<i64: 3>,
    padding = dense<[[5, 2]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{"x"}]>]>
  } : (tensor<8xi32>, tensor<i32>) -> tensor<12xi32>
  return %0 : tensor<12xi32>
}

//--- part2.mlir

func.func @main() {
  %input_2d_i32 = stablehlo.constant dense<[[1, 2, 3, 4, 5, 6, 7, 8],
                                            [2, 3, 4, 5, 6, 7, 8, 9],
                                            [3, 4, 5, 6, 7, 8, 9, 10],
                                            [4, 5, 6, 7, 8, 9, 10, 11],
                                            [5, 6, 7, 8, 9, 10, 11, 12],
                                            [6, 7, 8, 9, 10, 11, 12, 13],
                                            [7, 8, 9, 10, 11, 12, 13, 14],
                                            [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<8x8xi32>
  %input_2d_i32_b = stablehlo.constant dense<[[10, 20, 30, 40, 50, 60, 70, 80],
                                              [20, 30, 40, 50, 60, 70, 80, 90],
                                              [30, 40, 50, 60, 70, 80, 90, 100],
                                              [40, 50, 60, 70, 80, 90, 100, 110],
                                              [50, 60, 70, 80, 90, 100, 110, 120],
                                              [60, 70, 80, 90, 100, 110, 120, 130],
                                              [70, 80, 90, 100, 110, 120, 130, 140],
                                              [80, 90, 100, 110, 120, 130, 140, 150]]> : tensor<8x8xi32>
  %input_2d_f32 = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                                            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                                            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                                            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                                            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                                            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]> : tensor<8x8xf32>
  %input_8_i32 = stablehlo.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  %input_4_i32 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  // 2x2 grid slices of %input_2d_i32 (4x4).
  %s00 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %s01 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %s10 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %s11 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>

  // 2x2 grid slices of %input_2d_i32_b (4x4).
  %sb00 = "stablehlo.slice"(%input_2d_i32_b) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %sb01 = "stablehlo.slice"(%input_2d_i32_b) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %sb10 = "stablehlo.slice"(%input_2d_i32_b) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %sb11 = "stablehlo.slice"(%input_2d_i32_b) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>

  // 2x2 grid slices of %input_2d_f32 (4x4).
  %sf00 = "stablehlo.slice"(%input_2d_f32) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xf32>) -> tensor<4x4xf32>
  %sf01 = "stablehlo.slice"(%input_2d_f32) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xf32>) -> tensor<4x4xf32>
  %sf10 = "stablehlo.slice"(%input_2d_f32) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xf32>) -> tensor<4x4xf32>
  %sf11 = "stablehlo.slice"(%input_2d_f32) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xf32>) -> tensor<4x4xf32>

  // 2x1 slices of %input_2d_i32 (4x8).
  %s2d_0 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x8xi32>
  %s2d_1 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x8xi32>

  // 1D slices of %input_8_i32 (4) and %input_4_i32 (2).
  %s8_0 = "stablehlo.slice"(%input_8_i32) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %s8_1 = "stablehlo.slice"(%input_8_i32) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %s4_0 = "stablehlo.slice"(%input_4_i32) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s4_1 = "stablehlo.slice"(%input_4_i32) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>

  // 1. 2D i32 reduce_window.
  %seq_2d = func.call @sequential_reduce_window_2d(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %res_2d:4 = "interpreter.run_parallel"(%s00, %s01, %s10, %s11) {
    programs = [[@parallel_reduce_window_2d, @parallel_reduce_window_2d, @parallel_reduce_window_2d, @parallel_reduce_window_2d]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)
  "check.expect_eq"(%res_2d#0, %seq_2d) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_2d#1, %seq_2d) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_2d#2, %seq_2d) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_2d#3, %seq_2d) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  // 2. 2D f32 reduce_window.
  %seq_f32 = func.call @sequential_reduce_window_f32(%input_2d_f32) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %res_f32:4 = "interpreter.run_parallel"(%sf00, %sf01, %sf10, %sf11) {
    programs = [[@parallel_reduce_window_f32, @parallel_reduce_window_f32, @parallel_reduce_window_f32, @parallel_reduce_window_f32]]
  } : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>)
  "check.expect_eq"(%res_f32#0, %seq_f32) : (tensor<8x8xf32>, tensor<8x8xf32>) -> ()
  "check.expect_eq"(%res_f32#1, %seq_f32) : (tensor<8x8xf32>, tensor<8x8xf32>) -> ()
  "check.expect_eq"(%res_f32#2, %seq_f32) : (tensor<8x8xf32>, tensor<8x8xf32>) -> ()
  "check.expect_eq"(%res_f32#3, %seq_f32) : (tensor<8x8xf32>, tensor<8x8xf32>) -> ()

  // 3. 2D max reduce_window.
  %seq_max = func.call @sequential_reduce_window_max(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %res_max:4 = "interpreter.run_parallel"(%s00, %s01, %s10, %s11) {
    programs = [[@parallel_reduce_window_max, @parallel_reduce_window_max, @parallel_reduce_window_max, @parallel_reduce_window_max]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)
  "check.expect_eq"(%res_max#0, %seq_max) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_max#1, %seq_max) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_max#2, %seq_max) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_max#3, %seq_max) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  // 4. Mixed sharding reduce_window.
  %seq_mixed = func.call @sequential_reduce_window_mixed_sharding(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %res_mixed:4 = "interpreter.run_parallel"(%s00, %s01, %s10, %s11) {
    programs = [[@parallel_reduce_window_mixed_sharding, @parallel_reduce_window_mixed_sharding, @parallel_reduce_window_mixed_sharding, @parallel_reduce_window_mixed_sharding]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)
  "check.expect_eq"(%res_mixed#0, %seq_mixed) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mixed#1, %seq_mixed) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mixed#2, %seq_mixed) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mixed#3, %seq_mixed) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  // 5. Variadic reduce_window.
  %seq_var:2 = func.call @sequential_reduce_window_variadic(%input_2d_i32, %input_2d_i32_b) : (tensor<8x8xi32>, tensor<8x8xi32>) -> (tensor<8x8xi32>, tensor<8x8xi32>)
  %res_var:8 = "interpreter.run_parallel"(%s00, %sb00, %s01, %sb01, %s10, %sb10, %s11, %sb11) {
    programs = [[@parallel_reduce_window_variadic, @parallel_reduce_window_variadic, @parallel_reduce_window_variadic, @parallel_reduce_window_variadic]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)
    -> (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)
  "check.expect_eq"(%res_var#0, %seq_var#0) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#1, %seq_var#1) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#2, %seq_var#0) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#3, %seq_var#1) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#4, %seq_var#0) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#5, %seq_var#1) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#6, %seq_var#0) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_var#7, %seq_var#1) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  // 6. Padding and stride reduce_window.
  %seq_ps = func.call @sequential_reduce_window_padding_and_stride(%input_8_i32) : (tensor<8xi32>) -> tensor<4xi32>
  %res_ps:4 = "interpreter.run_parallel"(%s8_0, %s8_1, %s8_0, %s8_1) {
    programs = [[@parallel_reduce_window_padding_and_stride, @parallel_reduce_window_padding_and_stride, @parallel_reduce_window_padding_and_stride, @parallel_reduce_window_padding_and_stride]]
  } : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
  "check.expect_eq"(%res_ps#0, %seq_ps) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res_ps#1, %seq_ps) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res_ps#2, %seq_ps) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res_ps#3, %seq_ps) : (tensor<4xi32>, tensor<4xi32>) -> ()

  // 7. Window dilations reduce_window.
  %seq_wd = func.call @sequential_reduce_window_window_dilations(%input_8_i32) : (tensor<8xi32>) -> tensor<8xi32>
  %res_wd:2 = "interpreter.run_parallel"(%s8_0, %s8_1) {
    programs = [[@parallel_reduce_window_window_dilations, @parallel_reduce_window_window_dilations]]
  } : (tensor<4xi32>, tensor<4xi32>) -> (tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_wd#0, %seq_wd) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_wd#1, %seq_wd) : (tensor<8xi32>, tensor<8xi32>) -> ()

  // 8. Base dilations reduce_window.
  %seq_bd = func.call @sequential_reduce_window_base_dilations(%input_4_i32) : (tensor<4xi32>) -> tensor<8xi32>
  %res_bd:2 = "interpreter.run_parallel"(%s4_0, %s4_1) {
    programs = [[@parallel_reduce_window_base_dilations, @parallel_reduce_window_base_dilations]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_bd#0, %seq_bd) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_bd#1, %seq_bd) : (tensor<8xi32>, tensor<8xi32>) -> ()

  // 9. Complex dilations (base + window dilations) reduce_window.
  %seq_cd = func.call @sequential_reduce_window_complex_dilations(%input_4_i32) : (tensor<4xi32>) -> tensor<14xi32>
  %res_cd:2 = "interpreter.run_parallel"(%s4_0, %s4_1) {
    programs = [[@parallel_reduce_window_complex_dilations, @parallel_reduce_window_complex_dilations]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<14xi32>, tensor<14xi32>)
  "check.expect_eq"(%res_cd#0, %seq_cd) : (tensor<14xi32>, tensor<14xi32>) -> ()
  "check.expect_eq"(%res_cd#1, %seq_cd) : (tensor<14xi32>, tensor<14xi32>) -> ()

  // 10. Asymmetric padding reduce_window.
  %seq_asym = func.call @sequential_reduce_window_asymmetric_padding(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %res_asym:2 = "interpreter.run_parallel"(%s2d_0, %s2d_1) {
    programs = [[@parallel_reduce_window_asymmetric_padding, @parallel_reduce_window_asymmetric_padding]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>) -> (tensor<8x8xi32>, tensor<8x8xi32>)
  "check.expect_eq"(%res_asym#0, %seq_asym) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_asym#1, %seq_asym) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  // 11. Negative padding reduce_window.
  %seq_neg = func.call @sequential_reduce_window_negative_padding(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<6x8xi32>
  %res_neg:2 = "interpreter.run_parallel"(%s2d_0, %s2d_1) {
    programs = [[@parallel_reduce_window_negative_padding, @parallel_reduce_window_negative_padding]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>) -> (tensor<6x8xi32>, tensor<6x8xi32>)
  "check.expect_eq"(%res_neg#0, %seq_neg) : (tensor<6x8xi32>, tensor<6x8xi32>) -> ()
  "check.expect_eq"(%res_neg#1, %seq_neg) : (tensor<6x8xi32>, tensor<6x8xi32>) -> ()

  // 12. Indivisible tensor shape reduce_window.
  %seq_indiv = func.call @sequential_reduce_window_indivisible(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<7x8xi32>
  %res_indiv:2 = "interpreter.run_parallel"(%s2d_0, %s2d_1) {
    programs = [[@parallel_reduce_window_indivisible, @parallel_reduce_window_indivisible]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>) -> (tensor<7x8xi32>, tensor<7x8xi32>)
  "check.expect_eq"(%res_indiv#0, %seq_indiv) : (tensor<7x8xi32>, tensor<7x8xi32>) -> ()
  "check.expect_eq"(%res_indiv#1, %seq_indiv) : (tensor<7x8xi32>, tensor<7x8xi32>) -> ()

  // 13. Indivisible tensor shape with padding and stride reduce_window.
  %seq_indiv_ps = func.call @sequential_reduce_window_indivisible_padding_stride(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<4x8xi32>
  %res_indiv_ps:2 = "interpreter.run_parallel"(%s2d_0, %s2d_1) {
    programs = [[@parallel_reduce_window_indivisible_padding_stride, @parallel_reduce_window_indivisible_padding_stride]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>) -> (tensor<4x8xi32>, tensor<4x8xi32>)
  "check.expect_eq"(%res_indiv_ps#0, %seq_indiv_ps) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res_indiv_ps#1, %seq_indiv_ps) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

  // 14. Multi-hop reduce_window across 8 shards.
  %seq_mh = func.call @sequential_reduce_window_multi_hop(%input_2d_i32) : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %sm0 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm1 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm2 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm3 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm4 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 5, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm5 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 5, 0>, limit_indices = array<i64: 6, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm6 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 7, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %sm7 = "stablehlo.slice"(%input_2d_i32) {start_indices = array<i64: 7, 0>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<1x8xi32>
  %res_mh:8 = "interpreter.run_parallel"(%sm0, %sm1, %sm2, %sm3, %sm4, %sm5, %sm6, %sm7) {
    programs = [[@parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop, @parallel_reduce_window_multi_hop]]
  } : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>)
    -> (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)
  "check.expect_eq"(%res_mh#0, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#1, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#2, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#3, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#4, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#5, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#6, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res_mh#7, %seq_mh) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  // 15. Phase-aligned base_dilations and window_strides with residual local padding across 4 shards.
  %seq_bd_pas = func.call @sequential_reduce_window_base_dilation_phase_aligned_stride(%input_8_i32) : (tensor<8xi32>) -> tensor<12xi32>
  %s8_4_0 = "stablehlo.slice"(%input_8_i32) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s8_4_1 = "stablehlo.slice"(%input_8_i32) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s8_4_2 = "stablehlo.slice"(%input_8_i32) {start_indices = array<i64: 4>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s8_4_3 = "stablehlo.slice"(%input_8_i32) {start_indices = array<i64: 6>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %res_bd_pas:4 = "interpreter.run_parallel"(%s8_4_0, %s8_4_1, %s8_4_2, %s8_4_3) {
    programs = [[@parallel_reduce_window_base_dilation_phase_aligned_stride, @parallel_reduce_window_base_dilation_phase_aligned_stride, @parallel_reduce_window_base_dilation_phase_aligned_stride, @parallel_reduce_window_base_dilation_phase_aligned_stride]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<12xi32>, tensor<12xi32>, tensor<12xi32>, tensor<12xi32>)
  "check.expect_eq"(%res_bd_pas#0, %seq_bd_pas) : (tensor<12xi32>, tensor<12xi32>) -> ()
  "check.expect_eq"(%res_bd_pas#1, %seq_bd_pas) : (tensor<12xi32>, tensor<12xi32>) -> ()
  "check.expect_eq"(%res_bd_pas#2, %seq_bd_pas) : (tensor<12xi32>, tensor<12xi32>) -> ()
  "check.expect_eq"(%res_bd_pas#3, %seq_bd_pas) : (tensor<12xi32>, tensor<12xi32>) -> ()

  return
}
