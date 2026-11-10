// RUN: %S/run_sdy_interpreter_test.sh %s %t --partition_count=6

//--- part1.mlir

sdy.mesh @mesh_6 = <["x"=6]>
sdy.mesh @mesh_3_2 = <["x"=3, "y"=2]>

func.func @sequential_gather_201(%arg0: tensor<204x2xf32>, %arg1: tensor<3x1xi32>) -> tensor<3x2xf32> {
  %tbl = stablehlo.slice %arg0 [0:201, 0:2] : (tensor<204x2xf32>) -> tensor<201x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<201x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @parallel_gather_i8_201(
    %arg0: tensor<204x2xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"x"}, {}]>},
    %arg1: tensor<3x1xi8>)
    -> (tensor<3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{}, {}]>}) {
  %tbl = stablehlo.slice %arg0 [0:201, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{"x"}, {}]>]>}
    : (tensor<204x2xf32>) -> tensor<201x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {}]>]>
  } : (tensor<201x2xf32>, tensor<3x1xi8>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @sequential_gather_301(%arg0: tensor<306x2xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x2xf32> {
  %tbl = stablehlo.slice %arg0 [0:301, 0:2] : (tensor<306x2xf32>) -> tensor<301x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<301x2xf32>, tensor<2x1xi32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @parallel_gather_ui8_301(
    %arg0: tensor<306x2xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"x"}, {}]>},
    %arg1: tensor<2x1xui8>)
    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{}, {}]>}) {
  %tbl = stablehlo.slice %arg0 [0:301, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{"x"}, {}]>]>}
    : (tensor<306x2xf32>) -> tensor<301x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {}]>]>
  } : (tensor<301x2xf32>, tensor<2x1xui8>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @sequential_gather_255(%arg0: tensor<258x2xf32>, %arg1: tensor<3x1xi32>) -> tensor<3x2xf32> {
  %tbl = stablehlo.slice %arg0 [0:255, 0:2] : (tensor<258x2xf32>) -> tensor<255x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<255x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @parallel_gather_ui8_255_padded_to_258(
    %arg0: tensor<258x2xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"x"}, {}]>},
    %arg1: tensor<3x1xui8>)
    -> (tensor<3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{}, {}]>}) {
  %tbl = stablehlo.slice %arg0 [0:255, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{"x"}, {}]>]>}
    : (tensor<258x2xf32>) -> tensor<255x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {}]>]>
  } : (tensor<255x2xf32>, tensor<3x1xui8>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @sequential_gather_while_202(%arg0: tensor<204x2xf32>, %arg1: tensor<4x1xi32>) -> tensor<4x2xf32> {
  %tbl = stablehlo.slice %arg0 [0:202, 0:2] : (tensor<204x2xf32>) -> tensor<202x2xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<202x2xf32>, tensor<4x1xi32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

func.func @parallel_gather_while_sharded_i8_202(
    %arg0: tensor<204x2xf32> {sdy.sharding = #sdy.sharding<@mesh_3_2, [{"x"}, {}]>},
    %arg1: tensor<4x1xi8> {sdy.sharding = #sdy.sharding<@mesh_3_2, [{"y"}, {}]>})
    -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_3_2, [{"y"}, {}]>}) {
  %tbl = stablehlo.slice %arg0 [0:202, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3_2, [{"x"}, {}]>]>}
    : (tensor<204x2xf32>) -> tensor<202x2xf32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %init = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3_2, [{"y"}, {}]>]>} dense<0.0> : tensor<4x2xf32>
  %w:3 = stablehlo.while(%iter_idx = %arg1, %iter_acc = %init, %iter_i = %c0)
      : tensor<4x1xi8>, tensor<4x2xf32>, tensor<i32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[
      <@mesh_3_2, [{"y"}, {}]>, <@mesh_3_2, [{"y"}, {}]>, <@mesh_3_2, []>]>}
    cond {
      %cond = stablehlo.compare LT, %iter_i, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %g = "stablehlo.gather"(%tbl, %iter_idx) {
        dimension_numbers = #stablehlo.gather<
          offset_dims = [1], collapsed_slice_dims = [0],
          start_index_map = [0], index_vector_dim = 1>,
        slice_sizes = array<i64: 1, 2>,
        sdy.sharding = #sdy.sharding_per_value<[<@mesh_3_2, [{"y"}, {}]>]>
      } : (tensor<202x2xf32>, tensor<4x1xi8>) -> tensor<4x2xf32>
      %next_acc = stablehlo.add %iter_acc, %g
        {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3_2, [{"y"}, {}]>]>} : tensor<4x2xf32>
      %next_i = stablehlo.add %iter_i, %c1 : tensor<i32>
      stablehlo.return %iter_idx, %next_acc, %next_i : tensor<4x1xi8>, tensor<4x2xf32>, tensor<i32>
    }
  return %w#1 : tensor<4x2xf32>
}

func.func @sequential_gather_indexed_full_slice(%arg0: tensor<6x4xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x5x4xf32> {
  %tbl = stablehlo.slice %arg0 [0:5, 0:4] : (tensor<6x4xf32>) -> tensor<5x4xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 5, 4>
  } : (tensor<5x4xf32>, tensor<2x1xi32>) -> tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
}

func.func @parallel_gather_indexed_full_slice_sharded_output(
    %arg0: tensor<6x4xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"x"}, {}]>},
    %arg1: tensor<2x1xi32>)
    -> (tensor<2x5x4xf32> {sdy.sharding = #sdy.sharding<@mesh_6, [{}, {}, {}]>}) {
  %tbl = stablehlo.slice %arg0 [0:5, 0:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{"x"}, {}]>]>}
    : (tensor<6x4xf32>) -> tensor<5x4xf32>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 5, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {"x"}, {}]>]>
  } : (tensor<5x4xf32>, tensor<2x1xi32>) -> tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
}

//--- part2.mlir

func.func @main() {
  // Case 1: i8 indices [0, 10, 110] into padded 201x2 table (maxIdx=200 wraps to -56 in i8).
  %iota204 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<204x2xf32>
  %one204 = stablehlo.constant dense<1.0> : tensor<204x2xf32>
  %tab204 = stablehlo.add %iota204, %one204 : tensor<204x2xf32>
  %t204_0 = stablehlo.slice %tab204 [0:34, 0:2] : (tensor<204x2xf32>) -> tensor<34x2xf32>
  %t204_1 = stablehlo.slice %tab204 [34:68, 0:2] : (tensor<204x2xf32>) -> tensor<34x2xf32>
  %t204_2 = stablehlo.slice %tab204 [68:102, 0:2] : (tensor<204x2xf32>) -> tensor<34x2xf32>
  %t204_3 = stablehlo.slice %tab204 [102:136, 0:2] : (tensor<204x2xf32>) -> tensor<34x2xf32>
  %t204_4 = stablehlo.slice %tab204 [136:170, 0:2] : (tensor<204x2xf32>) -> tensor<34x2xf32>
  %t204_5 = stablehlo.slice %tab204 [170:204, 0:2] : (tensor<204x2xf32>) -> tensor<34x2xf32>
  %idx_i8 = stablehlo.constant dense<[[0], [10], [110]]> : tensor<3x1xi8>
  %idx_i32 = stablehlo.constant dense<[[0], [10], [110]]> : tensor<3x1xi32>
  %exp1 = func.call @sequential_gather_201(%tab204, %idx_i32)
    : (tensor<204x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
  %res1:6 = "interpreter.run_parallel"(
      %t204_0, %idx_i8, %t204_1, %idx_i8, %t204_2, %idx_i8,
      %t204_3, %idx_i8, %t204_4, %idx_i8, %t204_5, %idx_i8) {
    programs = [[
      @parallel_gather_i8_201, @parallel_gather_i8_201, @parallel_gather_i8_201,
      @parallel_gather_i8_201, @parallel_gather_i8_201, @parallel_gather_i8_201
    ]]
  } : (tensor<34x2xf32>, tensor<3x1xi8>, tensor<34x2xf32>, tensor<3x1xi8>,
       tensor<34x2xf32>, tensor<3x1xi8>, tensor<34x2xf32>, tensor<3x1xi8>,
       tensor<34x2xf32>, tensor<3x1xi8>, tensor<34x2xf32>, tensor<3x1xi8>)
      -> (tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>,
          tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>)
  "check.expect_eq"(%res1#0, %exp1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res1#5, %exp1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  // Case 2: ui8 indices [50, 200] into padded 301x2 table (maxIdx=300 wraps to 44 in ui8).
  %iota306 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<306x2xf32>
  %one306 = stablehlo.constant dense<1.0> : tensor<306x2xf32>
  %tab306 = stablehlo.add %iota306, %one306 : tensor<306x2xf32>
  %t306_0 = stablehlo.slice %tab306 [0:51, 0:2] : (tensor<306x2xf32>) -> tensor<51x2xf32>
  %t306_1 = stablehlo.slice %tab306 [51:102, 0:2] : (tensor<306x2xf32>) -> tensor<51x2xf32>
  %t306_2 = stablehlo.slice %tab306 [102:153, 0:2] : (tensor<306x2xf32>) -> tensor<51x2xf32>
  %t306_3 = stablehlo.slice %tab306 [153:204, 0:2] : (tensor<306x2xf32>) -> tensor<51x2xf32>
  %t306_4 = stablehlo.slice %tab306 [204:255, 0:2] : (tensor<306x2xf32>) -> tensor<51x2xf32>
  %t306_5 = stablehlo.slice %tab306 [255:306, 0:2] : (tensor<306x2xf32>) -> tensor<51x2xf32>
  %idx_u8 = stablehlo.constant dense<[[50], [200]]> : tensor<2x1xui8>
  %idx_u8_i32 = stablehlo.constant dense<[[50], [200]]> : tensor<2x1xi32>
  %exp2 = func.call @sequential_gather_301(%tab306, %idx_u8_i32)
    : (tensor<306x2xf32>, tensor<2x1xi32>) -> tensor<2x2xf32>
  %res2:6 = "interpreter.run_parallel"(
      %t306_0, %idx_u8, %t306_1, %idx_u8, %t306_2, %idx_u8,
      %t306_3, %idx_u8, %t306_4, %idx_u8, %t306_5, %idx_u8) {
    programs = [[
      @parallel_gather_ui8_301, @parallel_gather_ui8_301, @parallel_gather_ui8_301,
      @parallel_gather_ui8_301, @parallel_gather_ui8_301, @parallel_gather_ui8_301
    ]]
  } : (tensor<51x2xf32>, tensor<2x1xui8>, tensor<51x2xf32>, tensor<2x1xui8>,
       tensor<51x2xf32>, tensor<2x1xui8>, tensor<51x2xf32>, tensor<2x1xui8>,
       tensor<51x2xf32>, tensor<2x1xui8>, tensor<51x2xf32>, tensor<2x1xui8>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
          tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  "check.expect_eq"(%res2#0, %exp2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res2#5, %exp2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()

  // Case 3: ui8 indices [10, 200, 255] into 255x2 table padded to 258x2 across 6 shards
  // (unpadded maxIdx=254 fits in ui8, whereas padded max 257 overflows ui8 to 1).
  %iota258 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<258x2xf32>
  %one258 = stablehlo.constant dense<1.0> : tensor<258x2xf32>
  %tab258 = stablehlo.add %iota258, %one258 : tensor<258x2xf32>
  %t258_0 = stablehlo.slice %tab258 [0:43, 0:2] : (tensor<258x2xf32>) -> tensor<43x2xf32>
  %t258_1 = stablehlo.slice %tab258 [43:86, 0:2] : (tensor<258x2xf32>) -> tensor<43x2xf32>
  %t258_2 = stablehlo.slice %tab258 [86:129, 0:2] : (tensor<258x2xf32>) -> tensor<43x2xf32>
  %t258_3 = stablehlo.slice %tab258 [129:172, 0:2] : (tensor<258x2xf32>) -> tensor<43x2xf32>
  %t258_4 = stablehlo.slice %tab258 [172:215, 0:2] : (tensor<258x2xf32>) -> tensor<43x2xf32>
  %t258_5 = stablehlo.slice %tab258 [215:258, 0:2] : (tensor<258x2xf32>) -> tensor<43x2xf32>
  %idx_u8_255 = stablehlo.constant dense<[[10], [200], [255]]> : tensor<3x1xui8>
  %idx_i32_255 = stablehlo.constant dense<[[10], [200], [255]]> : tensor<3x1xi32>
  %exp3 = func.call @sequential_gather_255(%tab258, %idx_i32_255)
    : (tensor<258x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
  %res3:6 = "interpreter.run_parallel"(
      %t258_0, %idx_u8_255, %t258_1, %idx_u8_255, %t258_2, %idx_u8_255,
      %t258_3, %idx_u8_255, %t258_4, %idx_u8_255, %t258_5, %idx_u8_255) {
    programs = [[
      @parallel_gather_ui8_255_padded_to_258, @parallel_gather_ui8_255_padded_to_258,
      @parallel_gather_ui8_255_padded_to_258, @parallel_gather_ui8_255_padded_to_258,
      @parallel_gather_ui8_255_padded_to_258, @parallel_gather_ui8_255_padded_to_258
    ]]
  } : (tensor<43x2xf32>, tensor<3x1xui8>, tensor<43x2xf32>, tensor<3x1xui8>,
       tensor<43x2xf32>, tensor<3x1xui8>, tensor<43x2xf32>, tensor<3x1xui8>,
       tensor<43x2xf32>, tensor<3x1xui8>, tensor<43x2xf32>, tensor<3x1xui8>)
      -> (tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>,
          tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>)
  "check.expect_eq"(%res3#0, %exp3) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res3#5, %exp3) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  // Case 4: sharded i8 while-loop BlockArgument indices [0, 10, 50, 110] on mesh_3_2
  // (202x2 sliced from 204x2, padded back to 204x2 across "x"=3 while indices are sharded on "y"=2).
  %t204_x0 = stablehlo.slice %tab204 [0:68, 0:2] : (tensor<204x2xf32>) -> tensor<68x2xf32>
  %t204_x1 = stablehlo.slice %tab204 [68:136, 0:2] : (tensor<204x2xf32>) -> tensor<68x2xf32>
  %t204_x2 = stablehlo.slice %tab204 [136:204, 0:2] : (tensor<204x2xf32>) -> tensor<68x2xf32>
  %idx4_i8 = stablehlo.constant dense<[[0], [10], [50], [110]]> : tensor<4x1xi8>
  %idx4_i8_y0 = stablehlo.slice %idx4_i8 [0:2, 0:1] : (tensor<4x1xi8>) -> tensor<2x1xi8>
  %idx4_i8_y1 = stablehlo.slice %idx4_i8 [2:4, 0:1] : (tensor<4x1xi8>) -> tensor<2x1xi8>
  %idx4_i32 = stablehlo.constant dense<[[0], [10], [50], [110]]> : tensor<4x1xi32>
  %exp4 = func.call @sequential_gather_while_202(%tab204, %idx4_i32)
    : (tensor<204x2xf32>, tensor<4x1xi32>) -> tensor<4x2xf32>
  %res4:6 = "interpreter.run_parallel"(
      %t204_x0, %idx4_i8_y0, %t204_x0, %idx4_i8_y1,
      %t204_x1, %idx4_i8_y0, %t204_x1, %idx4_i8_y1,
      %t204_x2, %idx4_i8_y0, %t204_x2, %idx4_i8_y1) {
    programs = [[
      @parallel_gather_while_sharded_i8_202, @parallel_gather_while_sharded_i8_202,
      @parallel_gather_while_sharded_i8_202, @parallel_gather_while_sharded_i8_202,
      @parallel_gather_while_sharded_i8_202, @parallel_gather_while_sharded_i8_202
    ]]
  } : (tensor<68x2xf32>, tensor<2x1xi8>, tensor<68x2xf32>, tensor<2x1xi8>,
       tensor<68x2xf32>, tensor<2x1xi8>, tensor<68x2xf32>, tensor<2x1xi8>,
       tensor<68x2xf32>, tensor<2x1xi8>, tensor<68x2xf32>, tensor<2x1xi8>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
          tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  %out4_x0 = stablehlo.concatenate %res4#0, %res4#1, dim = 0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  %out4_x2 = stablehlo.concatenate %res4#4, %res4#5, dim = 0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  "check.expect_eq"(%out4_x0, %exp4) : (tensor<4x2xf32>, tensor<4x2xf32>) -> ()
  "check.expect_eq"(%out4_x2, %exp4) : (tensor<4x2xf32>, tensor<4x2xf32>) -> ()

  // Case 5: full-slice indexed dimension (size 5 padded to 6 across 6 shards)
  // sharded on both operand dim 0 and result offset dim 1.
  %iota6x4 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<6x4xf32>
  %one6x4 = stablehlo.constant dense<1.0> : tensor<6x4xf32>
  %tab6x4 = stablehlo.add %iota6x4, %one6x4 : tensor<6x4xf32>
  %t6_0 = stablehlo.slice %tab6x4 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
  %t6_1 = stablehlo.slice %tab6x4 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
  %t6_2 = stablehlo.slice %tab6x4 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
  %t6_3 = stablehlo.slice %tab6x4 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
  %t6_4 = stablehlo.slice %tab6x4 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
  %t6_5 = stablehlo.slice %tab6x4 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
  %idx5 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
  %exp5 = func.call @sequential_gather_indexed_full_slice(%tab6x4, %idx5)
    : (tensor<6x4xf32>, tensor<2x1xi32>) -> tensor<2x5x4xf32>
  %res5:6 = "interpreter.run_parallel"(
      %t6_0, %idx5, %t6_1, %idx5, %t6_2, %idx5,
      %t6_3, %idx5, %t6_4, %idx5, %t6_5, %idx5) {
    programs = [[
      @parallel_gather_indexed_full_slice_sharded_output,
      @parallel_gather_indexed_full_slice_sharded_output,
      @parallel_gather_indexed_full_slice_sharded_output,
      @parallel_gather_indexed_full_slice_sharded_output,
      @parallel_gather_indexed_full_slice_sharded_output,
      @parallel_gather_indexed_full_slice_sharded_output
    ]]
  } : (tensor<1x4xf32>, tensor<2x1xi32>, tensor<1x4xf32>, tensor<2x1xi32>,
       tensor<1x4xf32>, tensor<2x1xi32>, tensor<1x4xf32>, tensor<2x1xi32>,
       tensor<1x4xf32>, tensor<2x1xi32>, tensor<1x4xf32>, tensor<2x1xi32>)
      -> (tensor<2x5x4xf32>, tensor<2x5x4xf32>, tensor<2x5x4xf32>,
          tensor<2x5x4xf32>, tensor<2x5x4xf32>, tensor<2x5x4xf32>)
  "check.expect_eq"(%res5#0, %exp5) : (tensor<2x5x4xf32>, tensor<2x5x4xf32>) -> ()
  "check.expect_eq"(%res5#5, %exp5) : (tensor<2x5x4xf32>, tensor<2x5x4xf32>) -> ()
  return
}
