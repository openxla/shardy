// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sequential_gather_i32(%arg0: tensor<200x2xf32>, %arg1: tensor<3xi32>) -> tensor<3x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<200x2xf32>, tensor<3xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @parallel_gather_i8(
    %arg0: tensor<200x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<3xi8>) -> tensor<3x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<200x2xf32>, tensor<3xi8>) -> tensor<3x2xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

func.func @sequential_gather_512_i32(%arg0: tensor<512x2xf32>, %arg1: tensor<3xi32>) -> tensor<3x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<512x2xf32>, tensor<3xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @parallel_gather_ui8(
    %arg0: tensor<512x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<3xui8>) -> tensor<3x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<512x2xf32>, tensor<3xui8>) -> tensor<3x2xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

func.func @parallel_gather_ui32(
    %arg0: tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2xui32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<16x2xf32>, tensor<2xui32>) -> tensor<2x2xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

func.func @sequential_scatter_512_i32(
    %arg0: tensor<512x2xf32>,
    %arg1: tensor<2x1xi32>,
    %arg2: tensor<2x2xf32>) -> tensor<512x2xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<512x2xf32>, tensor<2x1xi32>, tensor<2x2xf32>) -> tensor<512x2xf32>
  return %0 : tensor<512x2xf32>
}

func.func @parallel_scatter_ui8(
    %arg0: tensor<512x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2x1xui8>,
    %arg2: tensor<2x2xf32>)
    -> (tensor<512x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<512x2xf32>, tensor<2x1xui8>, tensor<2x2xf32>) -> tensor<512x2xf32>
  return %0 : tensor<512x2xf32>
}

func.func @sequential_scatter_240_i32(
    %arg0: tensor<240x2xf32>,
    %arg1: tensor<2x1xi32>,
    %arg2: tensor<2x2xf32>) -> tensor<240x2xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<240x2xf32>, tensor<2x1xi32>, tensor<2x2xf32>) -> tensor<240x2xf32>
  return %0 : tensor<240x2xf32>
}

func.func @parallel_scatter_i8_oob(
    %arg0: tensor<240x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2x1xi8>,
    %arg2: tensor<2x2xf32>)
    -> (tensor<240x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<240x2xf32>, tensor<2x1xi8>, tensor<2x2xf32>) -> tensor<240x2xf32>
  return %0 : tensor<240x2xf32>
}

//--- part2.mlir

func.func @main() {
  // Case 1: i8 gather on 200x2 table (200 - 1 = 199 wraps to -57 in i8).
  %iota200 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<200x2xf32>
  %one200 = stablehlo.constant dense<1.0> : tensor<200x2xf32>
  %tab200 = stablehlo.add %iota200, %one200 : tensor<200x2xf32>
  %t200_0 = stablehlo.slice %tab200 [0:100, 0:2] : (tensor<200x2xf32>) -> tensor<100x2xf32>
  %t200_1 = stablehlo.slice %tab200 [100:200, 0:2] : (tensor<200x2xf32>) -> tensor<100x2xf32>
  %idx_i8 = stablehlo.constant dense<[0, 10, 110]> : tensor<3xi8>
  %idx_i32 = stablehlo.constant dense<[0, 10, 110]> : tensor<3xi32>
  %exp_g1 = func.call @sequential_gather_i32(%tab200, %idx_i32)
    : (tensor<200x2xf32>, tensor<3xi32>) -> tensor<3x2xf32>
  %res_g1:2 = "interpreter.run_parallel"(%t200_0, %idx_i8, %t200_1, %idx_i8) {
    programs = [[@parallel_gather_i8, @parallel_gather_i8]]
  } : (tensor<100x2xf32>, tensor<3xi8>, tensor<100x2xf32>, tensor<3xi8>)
      -> (tensor<3x2xf32>, tensor<3x2xf32>)
  "check.expect_eq"(%res_g1#0, %exp_g1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res_g1#1, %exp_g1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  // Case 2: ui8 gather on 512x2 table (shard 1 offset 256 truncates to 0 in ui8).
  %iota512 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<512x2xf32>
  %one512 = stablehlo.constant dense<1.0> : tensor<512x2xf32>
  %tab512 = stablehlo.add %iota512, %one512 : tensor<512x2xf32>
  %t512_0 = stablehlo.slice %tab512 [0:256, 0:2] : (tensor<512x2xf32>) -> tensor<256x2xf32>
  %t512_1 = stablehlo.slice %tab512 [256:512, 0:2] : (tensor<512x2xf32>) -> tensor<256x2xf32>
  %idx_u8 = stablehlo.constant dense<[0, 5, 200]> : tensor<3xui8>
  %idx_u8_i32 = stablehlo.constant dense<[0, 5, 200]> : tensor<3xi32>
  %exp_g2 = func.call @sequential_gather_512_i32(%tab512, %idx_u8_i32)
    : (tensor<512x2xf32>, tensor<3xi32>) -> tensor<3x2xf32>
  %res_g2:2 = "interpreter.run_parallel"(%t512_0, %idx_u8, %t512_1, %idx_u8) {
    programs = [[@parallel_gather_ui8, @parallel_gather_ui8]]
  } : (tensor<256x2xf32>, tensor<3xui8>, tensor<256x2xf32>, tensor<3xui8>)
      -> (tensor<3x2xf32>, tensor<3x2xf32>)
  "check.expect_eq"(%res_g2#0, %exp_g2) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  // Case 3: ui32 gather with > INT32_MAX index clamps to row 15 without sign wrap.
  %iota16 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<16x2xf32>
  %t16_0 = stablehlo.slice %iota16 [0:8, 0:2] : (tensor<16x2xf32>) -> tensor<8x2xf32>
  %t16_1 = stablehlo.slice %iota16 [8:16, 0:2] : (tensor<16x2xf32>) -> tensor<8x2xf32>
  %idx_u32 = stablehlo.constant dense<[2, 3000000000]> : tensor<2xui32>
  %exp_g3 = stablehlo.constant dense<[[2.0, 2.0], [15.0, 15.0]]> : tensor<2x2xf32>
  %res_g3:2 = "interpreter.run_parallel"(%t16_0, %idx_u32, %t16_1, %idx_u32) {
    programs = [[@parallel_gather_ui32, @parallel_gather_ui32]]
  } : (tensor<8x2xf32>, tensor<2xui32>, tensor<8x2xf32>, tensor<2xui32>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>)
  "check.expect_eq"(%res_g3#0, %exp_g3) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()

  // Case 4: ui8 scatter on 512x2 table (shard 1 offset 256 truncates to 0 in ui8).
  %z512 = stablehlo.constant dense<0.0> : tensor<512x2xf32>
  %z512_0 = stablehlo.slice %z512 [0:256, 0:2] : (tensor<512x2xf32>) -> tensor<256x2xf32>
  %z512_1 = stablehlo.slice %z512 [256:512, 0:2] : (tensor<512x2xf32>) -> tensor<256x2xf32>
  %sidx_u8 = stablehlo.constant dense<[[3], [200]]> : tensor<2x1xui8>
  %sidx_u8_i32 = stablehlo.constant dense<[[3], [200]]> : tensor<2x1xi32>
  %upd2 = stablehlo.constant dense<[[7.0, 9.0], [11.0, 13.0]]> : tensor<2x2xf32>
  %exp_s1 = func.call @sequential_scatter_512_i32(%z512, %sidx_u8_i32, %upd2)
    : (tensor<512x2xf32>, tensor<2x1xi32>, tensor<2x2xf32>) -> tensor<512x2xf32>
  %res_s1:2 = "interpreter.run_parallel"(
      %z512_0, %sidx_u8, %upd2, %z512_1, %sidx_u8, %upd2) {
    programs = [[@parallel_scatter_ui8, @parallel_scatter_ui8]]
  } : (tensor<256x2xf32>, tensor<2x1xui8>, tensor<2x2xf32>,
       tensor<256x2xf32>, tensor<2x1xui8>, tensor<2x2xf32>)
      -> (tensor<256x2xf32>, tensor<256x2xf32>)
  %cat_s1 = "stablehlo.concatenate"(%res_s1#0, %res_s1#1) {dimension = 0 : i64}
    : (tensor<256x2xf32>, tensor<256x2xf32>) -> tensor<512x2xf32>
  "check.expect_eq"(%cat_s1, %exp_s1) : (tensor<512x2xf32>, tensor<512x2xf32>) -> ()

  // Case 5: i8 negative OOB scatter index -20 on 240x2 table (-20 - 120 wraps to +116 in i8).
  %z240 = stablehlo.constant dense<0.0> : tensor<240x2xf32>
  %z240_0 = stablehlo.slice %z240 [0:120, 0:2] : (tensor<240x2xf32>) -> tensor<120x2xf32>
  %z240_1 = stablehlo.slice %z240 [120:240, 0:2] : (tensor<240x2xf32>) -> tensor<120x2xf32>
  %sidx_i8 = stablehlo.constant dense<[[-20], [5]]> : tensor<2x1xi8>
  %sidx_i8_i32 = stablehlo.constant dense<[[-20], [5]]> : tensor<2x1xi32>
  %exp_s2 = func.call @sequential_scatter_240_i32(%z240, %sidx_i8_i32, %upd2)
    : (tensor<240x2xf32>, tensor<2x1xi32>, tensor<2x2xf32>) -> tensor<240x2xf32>
  %res_s2:2 = "interpreter.run_parallel"(
      %z240_0, %sidx_i8, %upd2, %z240_1, %sidx_i8, %upd2) {
    programs = [[@parallel_scatter_i8_oob, @parallel_scatter_i8_oob]]
  } : (tensor<120x2xf32>, tensor<2x1xi8>, tensor<2x2xf32>,
       tensor<120x2xf32>, tensor<2x1xi8>, tensor<2x2xf32>)
      -> (tensor<120x2xf32>, tensor<120x2xf32>)
  %cat_s2 = "stablehlo.concatenate"(%res_s2#0, %res_s2#1) {dimension = 0 : i64}
    : (tensor<120x2xf32>, tensor<120x2xf32>) -> tensor<240x2xf32>
  "check.expect_eq"(%cat_s2, %exp_s2) : (tensor<240x2xf32>, tensor<240x2xf32>) -> ()

  return
}
