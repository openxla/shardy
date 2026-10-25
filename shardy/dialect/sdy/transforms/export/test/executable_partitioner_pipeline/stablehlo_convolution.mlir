// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_x2 = <["x"=2]>
sdy.mesh @mesh_x4 = <["x"=4]>
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

// 1D spatially sharded convolution with padding and rhs_dilate = [2] on a
// 4-device mesh (window footprint = 5 > shard size = 2).
func.func @parallel_convolution_padded_rhs_dilation(
  %arg0: tensor<1x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {"x"}, {}]>},
  %arg1: tensor<3x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}]>})
  -> (tensor<1x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f] x [0, i, o] -> [b, 0, f],
    window = {stride = [1], pad = [[2, 2]], rhs_dilate = [2]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{}, {"x"}, {}]>]>
    }
    : (tensor<1x8x2xf32>, tensor<3x2x2xf32>) -> tensor<1x8x2xf32>
  return %0 : tensor<1x8x2xf32>
}

func.func @parallel_convolution_1d_sharded(
  %arg0: tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {"x":(2)2}, {}]>},
  %arg1: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}, {}]>})
  -> (tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{}, {}, {"x":(2)2}, {}]>]>
    }
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x8xf32>
  return %0 : tensor<1x1x8x8xf32>
}

func.func @parallel_convolution_2d_sharded(
  %arg0: tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {"a"}, {"b"}]>},
  %arg1: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>})
  -> (tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{}, {}, {"a"}, {"b"}]>]>
    }
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x8xf32>
  return %0 : tensor<1x1x8x8xf32>
}

func.func @parallel_convolution_stride_padding(
  %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {"x"}, {}]>},
  %arg1: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>})
  -> (tensor<1x1x8x14xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [2, 1], pad = [[1, 1], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {}, {"x"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x14xf32>
  return %0 : tensor<1x1x8x14xf32>
}

func.func @parallel_convolution_rhs_dilation(
  %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {"x"}, {}]>},
  %arg1: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>})
  -> (tensor<1x1x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {}, {"x"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x12x14xf32>
  return %0 : tensor<1x1x12x14xf32>
}

func.func @parallel_convolution_lhs_and_rhs_dilations_low_pad(
  %arg0: tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {"x"}, {}]>},
  %arg1: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>})
  -> (tensor<1x1x28x6xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[4, 6], [0, 0]], lhs_dilate = [3, 1], rhs_dilate = [2, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {}, {"x"}, {}]>]>
    }
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x28x6xf32>
  return %0 : tensor<1x1x28x6xf32>
}

func.func @parallel_convolution_feature_group_per_group_and_spatial(
  %arg0: tensor<1x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {"x":(1)2}, {"x":(2)2}, {}]>},
  %arg1: tensor<3x3x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}, {"x":(1)2}]>})
  -> (tensor<1x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{}, {"x":(1)2}, {"x":(2)2}, {}]>]>
    }
    : (tensor<1x4x8x8xf32>, tensor<3x3x2x4xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

func.func @parallel_convolution_indivisible_spatial(
  %arg0: tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {"x"}, {}]>},
  %arg1: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>})
  -> (tensor<1x1x7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}, {}, {}]>}) {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0, 0, 0, 0>,
    limit_indices = array<i64: 1, 1, 7, 8>,
    strides = array<i64: 1, 1, 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {}, {"x"}, {}]>]>
  } : (tensor<1x1x8x8xf32>) -> tensor<1x1x7x8xf32>
  %1 = stablehlo.convolution(%0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {}, {"x"}, {}]>]>
    }
    : (tensor<1x1x7x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x7x8xf32>
  return %1 : tensor<1x1x7x8xf32>
}

func.func @parallel_convolution_spatial_and_reduction_sharded(
  %arg0: tensor<1x2x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {"x":(1)2}, {"x":(2)2}, {}]>},
  %arg1: tensor<3x3x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {"x":(1)2}, {}]>})
  -> (tensor<1x1x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{}, {}, {"x":(2)2}, {}]>]>
    }
    : (tensor<1x2x8x8xf32>, tensor<3x3x2x1xf32>) -> tensor<1x1x8x8xf32>
  return %0 : tensor<1x1x8x8xf32>
}

//--- part2.mlir

func.func @main() {
  // 0. 1D spatially sharded convolution with padding and rhs_dilate = [2] on 4 devices.
  %iota_16 = stablehlo.iota dim = 0 : tensor<16xf32>
  %one_16 = stablehlo.constant dense<1.000000e+00> : tensor<16xf32>
  %seq_16 = stablehlo.add %iota_16, %one_16 : tensor<16xf32>
  %input_1x8x2 = stablehlo.reshape %seq_16 : (tensor<16xf32>) -> tensor<1x8x2xf32>

  %iota_12 = stablehlo.iota dim = 0 : tensor<12xf32>
  %one_12 = stablehlo.constant dense<1.000000e+00> : tensor<12xf32>
  %seq_12 = stablehlo.add %iota_12, %one_12 : tensor<12xf32>
  %filter_3x2x2 = stablehlo.reshape %seq_12 : (tensor<12xf32>) -> tensor<3x2x2xf32>

  %seq_dil = func.call @sequential_convolution_padded_rhs_dilation(%input_1x8x2, %filter_3x2x2)
    : (tensor<1x8x2xf32>, tensor<3x2x2xf32>) -> tensor<1x8x2xf32>
  %d0 = "stablehlo.slice"(%input_1x8x2) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 1, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<1x8x2xf32>) -> tensor<1x2x2xf32>
  %d1 = "stablehlo.slice"(%input_1x8x2) {start_indices = array<i64: 0, 2, 0>, limit_indices = array<i64: 1, 4, 2>, strides = array<i64: 1, 1, 1>} : (tensor<1x8x2xf32>) -> tensor<1x2x2xf32>
  %d2 = "stablehlo.slice"(%input_1x8x2) {start_indices = array<i64: 0, 4, 0>, limit_indices = array<i64: 1, 6, 2>, strides = array<i64: 1, 1, 1>} : (tensor<1x8x2xf32>) -> tensor<1x2x2xf32>
  %d3 = "stablehlo.slice"(%input_1x8x2) {start_indices = array<i64: 0, 6, 0>, limit_indices = array<i64: 1, 8, 2>, strides = array<i64: 1, 1, 1>} : (tensor<1x8x2xf32>) -> tensor<1x2x2xf32>
  %res_dil:4 = "interpreter.run_parallel"(%d0, %filter_3x2x2, %d1, %filter_3x2x2, %d2, %filter_3x2x2, %d3, %filter_3x2x2) {
    programs = [[@parallel_convolution_padded_rhs_dilation, @parallel_convolution_padded_rhs_dilation, @parallel_convolution_padded_rhs_dilation, @parallel_convolution_padded_rhs_dilation]]
  } : (tensor<1x2x2xf32>, tensor<3x2x2xf32>, tensor<1x2x2xf32>, tensor<3x2x2xf32>, tensor<1x2x2xf32>, tensor<3x2x2xf32>, tensor<1x2x2xf32>, tensor<3x2x2xf32>)
    -> (tensor<1x8x2xf32>, tensor<1x8x2xf32>, tensor<1x8x2xf32>, tensor<1x8x2xf32>)
  "check.expect_eq"(%res_dil#0, %seq_dil) : (tensor<1x8x2xf32>, tensor<1x8x2xf32>) -> ()
  "check.expect_eq"(%res_dil#1, %seq_dil) : (tensor<1x8x2xf32>, tensor<1x8x2xf32>) -> ()
  "check.expect_eq"(%res_dil#2, %seq_dil) : (tensor<1x8x2xf32>, tensor<1x8x2xf32>) -> ()
  "check.expect_eq"(%res_dil#3, %seq_dil) : (tensor<1x8x2xf32>, tensor<1x8x2xf32>) -> ()

  %iota_64 = stablehlo.iota dim = 0 : tensor<64xf32>
  %one_64 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
  %seq_64 = stablehlo.add %iota_64, %one_64 : tensor<64xf32>
  %input_8x8 = stablehlo.reshape %seq_64 : (tensor<64xf32>) -> tensor<1x1x8x8xf32>

  %iota_256 = stablehlo.iota dim = 0 : tensor<256xf32>
  %one_256 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
  %seq_256 = stablehlo.add %iota_256, %one_256 : tensor<256xf32>
  %input_16x16 = stablehlo.reshape %seq_256 : (tensor<256xf32>) -> tensor<1x1x16x16xf32>

  %iota_9 = stablehlo.iota dim = 0 : tensor<9xf32>
  %one_9 = stablehlo.constant dense<1.000000e+00> : tensor<9xf32>
  %seq_9 = stablehlo.add %iota_9, %one_9 : tensor<9xf32>
  %filter = stablehlo.reshape %seq_9 : (tensor<9xf32>) -> tensor<3x3x1x1xf32>

  // 1. 1D sharded convolution on 1D mesh of size 4 (sharded along sub-axis "x":(2)2).
  %seq_1d = func.call @sequential_convolution_1d_sharded(%input_8x8, %filter)
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x8xf32>
  %s8_0 = "stablehlo.slice"(%input_8x8) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 1, 4, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x4x8xf32>
  %s8_1 = "stablehlo.slice"(%input_8x8) {start_indices = array<i64: 0, 0, 4, 0>, limit_indices = array<i64: 1, 1, 8, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x4x8xf32>
  %res_1d:4 = "interpreter.run_parallel"(%s8_0, %filter, %s8_1, %filter, %s8_0, %filter, %s8_1, %filter) {
    programs = [[@parallel_convolution_1d_sharded, @parallel_convolution_1d_sharded, @parallel_convolution_1d_sharded, @parallel_convolution_1d_sharded]]
  } : (tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>)
  "check.expect_eq"(%res_1d#0, %seq_1d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_1d#1, %seq_1d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_1d#2, %seq_1d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_1d#3, %seq_1d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()

  // 2. 2D sharded convolution on 2x2 mesh (sharded along "a" and "b").
  %seq_2d = func.call @sequential_convolution_2d_sharded(%input_8x8, %filter)
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x8xf32>
  %s8_00 = "stablehlo.slice"(%input_8x8) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 1, 4, 4>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x4x4xf32>
  %s8_01 = "stablehlo.slice"(%input_8x8) {start_indices = array<i64: 0, 0, 0, 4>, limit_indices = array<i64: 1, 1, 4, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x4x4xf32>
  %s8_10 = "stablehlo.slice"(%input_8x8) {start_indices = array<i64: 0, 0, 4, 0>, limit_indices = array<i64: 1, 1, 8, 4>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x4x4xf32>
  %s8_11 = "stablehlo.slice"(%input_8x8) {start_indices = array<i64: 0, 0, 4, 4>, limit_indices = array<i64: 1, 1, 8, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x8x8xf32>) -> tensor<1x1x4x4xf32>
  %res_2d:4 = "interpreter.run_parallel"(%s8_00, %filter, %s8_01, %filter, %s8_10, %filter, %s8_11, %filter) {
    programs = [[@parallel_convolution_2d_sharded, @parallel_convolution_2d_sharded, @parallel_convolution_2d_sharded, @parallel_convolution_2d_sharded]]
  } : (tensor<1x1x4x4xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x4xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x4xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x4xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>)
  "check.expect_eq"(%res_2d#0, %seq_2d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_2d#1, %seq_2d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_2d#2, %seq_2d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_2d#3, %seq_2d) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()

  // 3. Stride and padding convolution on 1D mesh.
  %seq_sp = func.call @sequential_convolution_stride_padding(%input_16x16, %filter)
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x8x14xf32>
  %s16_0 = "stablehlo.slice"(%input_16x16) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 1, 8, 16>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x16x16xf32>) -> tensor<1x1x8x16xf32>
  %s16_1 = "stablehlo.slice"(%input_16x16) {start_indices = array<i64: 0, 0, 8, 0>, limit_indices = array<i64: 1, 1, 16, 16>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x1x16x16xf32>) -> tensor<1x1x8x16xf32>
  %res_sp:2 = "interpreter.run_parallel"(%s16_0, %filter, %s16_1, %filter) {
    programs = [[@parallel_convolution_stride_padding, @parallel_convolution_stride_padding]]
  } : (tensor<1x1x8x16xf32>, tensor<3x3x1x1xf32>, tensor<1x1x8x16xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x8x14xf32>, tensor<1x1x8x14xf32>)
  "check.expect_eq"(%res_sp#0, %seq_sp) : (tensor<1x1x8x14xf32>, tensor<1x1x8x14xf32>) -> ()
  "check.expect_eq"(%res_sp#1, %seq_sp) : (tensor<1x1x8x14xf32>, tensor<1x1x8x14xf32>) -> ()

  // 4. RHS dilation convolution on 1D mesh.
  %seq_rhs_dil = func.call @sequential_convolution_rhs_dilation(%input_16x16, %filter)
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x12x14xf32>
  %res_rhs_dil:2 = "interpreter.run_parallel"(%s16_0, %filter, %s16_1, %filter) {
    programs = [[@parallel_convolution_rhs_dilation, @parallel_convolution_rhs_dilation]]
  } : (tensor<1x1x8x16xf32>, tensor<3x3x1x1xf32>, tensor<1x1x8x16xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x12x14xf32>, tensor<1x1x12x14xf32>)
  "check.expect_eq"(%res_rhs_dil#0, %seq_rhs_dil) : (tensor<1x1x12x14xf32>, tensor<1x1x12x14xf32>) -> ()
  "check.expect_eq"(%res_rhs_dil#1, %seq_rhs_dil) : (tensor<1x1x12x14xf32>, tensor<1x1x12x14xf32>) -> ()

  // 5. LHS and RHS dilations with low pad > 0 (exercises post-op dynamic slice).
  %seq_lhs_rhs_dil = func.call @sequential_convolution_lhs_and_rhs_dilations_low_pad(%input_8x8, %filter)
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x28x6xf32>
  %res_lhs_rhs_dil:2 = "interpreter.run_parallel"(%s8_0, %filter, %s8_1, %filter) {
    programs = [[@parallel_convolution_lhs_and_rhs_dilations_low_pad, @parallel_convolution_lhs_and_rhs_dilations_low_pad]]
  } : (tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x28x6xf32>, tensor<1x1x28x6xf32>)
  "check.expect_eq"(%res_lhs_rhs_dil#0, %seq_lhs_rhs_dil) : (tensor<1x1x28x6xf32>, tensor<1x1x28x6xf32>) -> ()
  "check.expect_eq"(%res_lhs_rhs_dil#1, %seq_lhs_rhs_dil) : (tensor<1x1x28x6xf32>, tensor<1x1x28x6xf32>) -> ()

  // 6. Grouped convolution with per_group_size > 1 co-sharded on feature groups ("x":(1)2) and spatial ("x":(2)2).
  %input_1x4x8x8 = stablehlo.reshape %seq_256 : (tensor<256xf32>) -> tensor<1x4x8x8xf32>
  %iota_72 = stablehlo.iota dim = 0 : tensor<72xf32>
  %one_72 = stablehlo.constant dense<1.000000e+00> : tensor<72xf32>
  %seq_72 = stablehlo.add %iota_72, %one_72 : tensor<72xf32>
  %filter_3x3x2x4 = stablehlo.reshape %seq_72 : (tensor<72xf32>) -> tensor<3x3x2x4xf32>
  %seq_grp = func.call @sequential_convolution_feature_group_per_group_and_spatial(%input_1x4x8x8, %filter_3x3x2x4)
    : (tensor<1x4x8x8xf32>, tensor<3x3x2x4xf32>) -> tensor<1x4x8x8xf32>
  %in_g0_s0 = "stablehlo.slice"(%input_1x4x8x8) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 2, 4, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x4x8x8xf32>) -> tensor<1x2x4x8xf32>
  %in_g0_s1 = "stablehlo.slice"(%input_1x4x8x8) {start_indices = array<i64: 0, 0, 4, 0>, limit_indices = array<i64: 1, 2, 8, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x4x8x8xf32>) -> tensor<1x2x4x8xf32>
  %in_g1_s0 = "stablehlo.slice"(%input_1x4x8x8) {start_indices = array<i64: 0, 2, 0, 0>, limit_indices = array<i64: 1, 4, 4, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x4x8x8xf32>) -> tensor<1x2x4x8xf32>
  %in_g1_s1 = "stablehlo.slice"(%input_1x4x8x8) {start_indices = array<i64: 0, 2, 4, 0>, limit_indices = array<i64: 1, 4, 8, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x4x8x8xf32>) -> tensor<1x2x4x8xf32>
  %flt_g0 = "stablehlo.slice"(%filter_3x3x2x4) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 3, 3, 2, 2>, strides = array<i64: 1, 1, 1, 1>} : (tensor<3x3x2x4xf32>) -> tensor<3x3x2x2xf32>
  %flt_g1 = "stablehlo.slice"(%filter_3x3x2x4) {start_indices = array<i64: 0, 0, 0, 2>, limit_indices = array<i64: 3, 3, 2, 4>, strides = array<i64: 1, 1, 1, 1>} : (tensor<3x3x2x4xf32>) -> tensor<3x3x2x2xf32>
  %res_grp:4 = "interpreter.run_parallel"(%in_g0_s0, %flt_g0, %in_g0_s1, %flt_g0, %in_g1_s0, %flt_g1, %in_g1_s1, %flt_g1) {
    programs = [[@parallel_convolution_feature_group_per_group_and_spatial, @parallel_convolution_feature_group_per_group_and_spatial, @parallel_convolution_feature_group_per_group_and_spatial, @parallel_convolution_feature_group_per_group_and_spatial]]
  } : (tensor<1x2x4x8xf32>, tensor<3x3x2x2xf32>, tensor<1x2x4x8xf32>, tensor<3x3x2x2xf32>, tensor<1x2x4x8xf32>, tensor<3x3x2x2xf32>, tensor<1x2x4x8xf32>, tensor<3x3x2x2xf32>)
    -> (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>)
  "check.expect_eq"(%res_grp#0, %seq_grp) : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> ()
  "check.expect_eq"(%res_grp#1, %seq_grp) : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> ()
  "check.expect_eq"(%res_grp#2, %seq_grp) : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> ()
  "check.expect_eq"(%res_grp#3, %seq_grp) : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> ()

  // 7. Indivisible spatial dimension shape (exercises high-side pad and slice around manual_computation).
  %seq_indiv = func.call @sequential_convolution_indivisible_spatial(%input_8x8, %filter)
    : (tensor<1x1x8x8xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x7x8xf32>
  %res_indiv:2 = "interpreter.run_parallel"(%s8_0, %filter, %s8_1, %filter) {
    programs = [[@parallel_convolution_indivisible_spatial, @parallel_convolution_indivisible_spatial]]
  } : (tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x7x8xf32>, tensor<1x1x7x8xf32>)
  "check.expect_eq"(%res_indiv#0, %seq_indiv) : (tensor<1x1x7x8xf32>, tensor<1x1x7x8xf32>) -> ()
  "check.expect_eq"(%res_indiv#1, %seq_indiv) : (tensor<1x1x7x8xf32>, tensor<1x1x7x8xf32>) -> ()

  // 8. Spatial HALO exchange ("x":(2)2) combined with sharded input feature reduction dimension ("x":(1)2).
  %iota_128 = stablehlo.iota dim = 0 : tensor<128xf32>
  %one_128 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
  %seq_128 = stablehlo.add %iota_128, %one_128 : tensor<128xf32>
  %input_1x2x8x8 = stablehlo.reshape %seq_128 : (tensor<128xf32>) -> tensor<1x2x8x8xf32>
  %iota_18 = stablehlo.iota dim = 0 : tensor<18xf32>
  %one_18 = stablehlo.constant dense<1.000000e+00> : tensor<18xf32>
  %seq_18 = stablehlo.add %iota_18, %one_18 : tensor<18xf32>
  %filter_3x3x2x1 = stablehlo.reshape %seq_18 : (tensor<18xf32>) -> tensor<3x3x2x1xf32>
  %seq_red = func.call @sequential_convolution_spatial_and_reduction_sharded(%input_1x2x8x8, %filter_3x3x2x1)
    : (tensor<1x2x8x8xf32>, tensor<3x3x2x1xf32>) -> tensor<1x1x8x8xf32>
  %in_f0_s0 = "stablehlo.slice"(%input_1x2x8x8) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 1, 4, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x2x8x8xf32>) -> tensor<1x1x4x8xf32>
  %in_f0_s1 = "stablehlo.slice"(%input_1x2x8x8) {start_indices = array<i64: 0, 0, 4, 0>, limit_indices = array<i64: 1, 1, 8, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x2x8x8xf32>) -> tensor<1x1x4x8xf32>
  %in_f1_s0 = "stablehlo.slice"(%input_1x2x8x8) {start_indices = array<i64: 0, 1, 0, 0>, limit_indices = array<i64: 1, 2, 4, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x2x8x8xf32>) -> tensor<1x1x4x8xf32>
  %in_f1_s1 = "stablehlo.slice"(%input_1x2x8x8) {start_indices = array<i64: 0, 1, 4, 0>, limit_indices = array<i64: 1, 2, 8, 8>, strides = array<i64: 1, 1, 1, 1>} : (tensor<1x2x8x8xf32>) -> tensor<1x1x4x8xf32>
  %flt_f0 = "stablehlo.slice"(%filter_3x3x2x1) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 3, 3, 1, 1>, strides = array<i64: 1, 1, 1, 1>} : (tensor<3x3x2x1xf32>) -> tensor<3x3x1x1xf32>
  %flt_f1 = "stablehlo.slice"(%filter_3x3x2x1) {start_indices = array<i64: 0, 0, 1, 0>, limit_indices = array<i64: 3, 3, 2, 1>, strides = array<i64: 1, 1, 1, 1>} : (tensor<3x3x2x1xf32>) -> tensor<3x3x1x1xf32>
  %res_red:4 = "interpreter.run_parallel"(%in_f0_s0, %flt_f0, %in_f0_s1, %flt_f0, %in_f1_s0, %flt_f1, %in_f1_s1, %flt_f1) {
    programs = [[@parallel_convolution_spatial_and_reduction_sharded, @parallel_convolution_spatial_and_reduction_sharded, @parallel_convolution_spatial_and_reduction_sharded, @parallel_convolution_spatial_and_reduction_sharded]]
  } : (tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>, tensor<1x1x4x8xf32>, tensor<3x3x1x1xf32>)
    -> (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>)
  "check.expect_eq"(%res_red#0, %seq_red) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_red#1, %seq_red) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_red#2, %seq_red) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  "check.expect_eq"(%res_red#3, %seq_red) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()

  return
}
