// RUN: %S/run_sdy_interpreter_test.sh %s %t --partition_count=4

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

func.func @parallel_gather_f8e8m0fnu_replicated(
    %arg0: tensor<8xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x", "y"}]>},
    %arg1: tensor<4x1xi32>)
    -> (tensor<4xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>}) {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}]>]>
  } : (tensor<8xf8E8M0FNU>, tensor<4x1xi32>) -> tensor<4xf8E8M0FNU>
  return %0 : tensor<4xf8E8M0FNU>
}

func.func @parallel_gather_f8e8m0fnu_padded_sharded_out(
    %arg0: tensor<8x2xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>},
    %arg1: tensor<4x1xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {}]>})
    -> (tensor<4x2xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {}]>}) {
  %tbl = stablehlo.slice %arg0 [0:5, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {}]>]>}
    : (tensor<8x2xf8E8M0FNU>) -> tensor<5x2xf8E8M0FNU>
  %0 = "stablehlo.gather"(%tbl, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>
  } : (tensor<5x2xf8E8M0FNU>, tensor<4x1xi32>) -> tensor<4x2xf8E8M0FNU>
  return %0 : tensor<4x2xf8E8M0FNU>
}

//--- part2.mlir

func.func @main() {
  %raw8 = stablehlo.constant dense<[0, 1, 2, 3, 0, 127, 254, 255]> : tensor<8xui8>
  %in8 = stablehlo.bitcast_convert %raw8 : (tensor<8xui8>) -> tensor<8xf8E8M0FNU>
  %idx4 = stablehlo.constant dense<[[4], [5], [6], [7]]> : tensor<4x1xi32>
  %exp_rep = func.call @sequential_gather_f8e8m0fnu_replicated(%in8, %idx4)
    : (tensor<8xf8E8M0FNU>, tensor<4x1xi32>) -> tensor<4xf8E8M0FNU>
  %s0 = stablehlo.slice %in8 [0:2] : (tensor<8xf8E8M0FNU>) -> tensor<2xf8E8M0FNU>
  %s1 = stablehlo.slice %in8 [2:4] : (tensor<8xf8E8M0FNU>) -> tensor<2xf8E8M0FNU>
  %s2 = stablehlo.slice %in8 [4:6] : (tensor<8xf8E8M0FNU>) -> tensor<2xf8E8M0FNU>
  %s3 = stablehlo.slice %in8 [6:8] : (tensor<8xf8E8M0FNU>) -> tensor<2xf8E8M0FNU>
  %res_rep:4 = "interpreter.run_parallel"(%s0, %idx4, %s1, %idx4, %s2, %idx4, %s3, %idx4) {
    programs = [[
      @parallel_gather_f8e8m0fnu_replicated,
      @parallel_gather_f8e8m0fnu_replicated,
      @parallel_gather_f8e8m0fnu_replicated,
      @parallel_gather_f8e8m0fnu_replicated
    ]]
  } : (tensor<2xf8E8M0FNU>, tensor<4x1xi32>, tensor<2xf8E8M0FNU>, tensor<4x1xi32>,
       tensor<2xf8E8M0FNU>, tensor<4x1xi32>, tensor<2xf8E8M0FNU>, tensor<4x1xi32>)
      -> (tensor<4xf8E8M0FNU>, tensor<4xf8E8M0FNU>, tensor<4xf8E8M0FNU>, tensor<4xf8E8M0FNU>)
  %exp_rep_u8 = stablehlo.bitcast_convert %exp_rep : (tensor<4xf8E8M0FNU>) -> tensor<4xui8>
  %res_rep0_u8 = stablehlo.bitcast_convert %res_rep#0 : (tensor<4xf8E8M0FNU>) -> tensor<4xui8>
  %res_rep3_u8 = stablehlo.bitcast_convert %res_rep#3 : (tensor<4xf8E8M0FNU>) -> tensor<4xui8>
  "check.expect_eq"(%res_rep0_u8, %exp_rep_u8) : (tensor<4xui8>, tensor<4xui8>) -> ()
  "check.expect_eq"(%res_rep3_u8, %exp_rep_u8) : (tensor<4xui8>, tensor<4xui8>) -> ()

  %raw8x2 = stablehlo.constant dense<[[0, 1], [2, 3], [0, 127], [128, 254], [0, 255], [4, 5], [6, 7], [8, 9]]> : tensor<8x2xui8>
  %in8x2 = stablehlo.bitcast_convert %raw8x2 : (tensor<8x2xui8>) -> tensor<8x2xf8E8M0FNU>
  %idx_sh = stablehlo.constant dense<[[4], [2], [3], [0]]> : tensor<4x1xi32>
  %exp_sh = func.call @sequential_gather_f8e8m0fnu_padded_sharded_out(%in8x2, %idx_sh)
    : (tensor<8x2xf8E8M0FNU>, tensor<4x1xi32>) -> tensor<4x2xf8E8M0FNU>
  %p_top = stablehlo.slice %in8x2 [0:4, 0:2] : (tensor<8x2xf8E8M0FNU>) -> tensor<4x2xf8E8M0FNU>
  %p_bot = stablehlo.slice %in8x2 [4:8, 0:2] : (tensor<8x2xf8E8M0FNU>) -> tensor<4x2xf8E8M0FNU>
  %i_lo = stablehlo.slice %idx_sh [0:2, 0:1] : (tensor<4x1xi32>) -> tensor<2x1xi32>
  %i_hi = stablehlo.slice %idx_sh [2:4, 0:1] : (tensor<4x1xi32>) -> tensor<2x1xi32>
  %res_sh:4 = "interpreter.run_parallel"(%p_top, %i_lo, %p_top, %i_hi, %p_bot, %i_lo, %p_bot, %i_hi) {
    programs = [[
      @parallel_gather_f8e8m0fnu_padded_sharded_out,
      @parallel_gather_f8e8m0fnu_padded_sharded_out,
      @parallel_gather_f8e8m0fnu_padded_sharded_out,
      @parallel_gather_f8e8m0fnu_padded_sharded_out
    ]]
  } : (tensor<4x2xf8E8M0FNU>, tensor<2x1xi32>, tensor<4x2xf8E8M0FNU>, tensor<2x1xi32>,
       tensor<4x2xf8E8M0FNU>, tensor<2x1xi32>, tensor<4x2xf8E8M0FNU>, tensor<2x1xi32>)
      -> (tensor<2x2xf8E8M0FNU>, tensor<2x2xf8E8M0FNU>, tensor<2x2xf8E8M0FNU>, tensor<2x2xf8E8M0FNU>)
  %full_sh = "stablehlo.concatenate"(%res_sh#0, %res_sh#1) {dimension = 0 : i64}
    : (tensor<2x2xf8E8M0FNU>, tensor<2x2xf8E8M0FNU>) -> tensor<4x2xf8E8M0FNU>
  %exp_sh_u8 = stablehlo.bitcast_convert %exp_sh : (tensor<4x2xf8E8M0FNU>) -> tensor<4x2xui8>
  %full_sh_u8 = stablehlo.bitcast_convert %full_sh : (tensor<4x2xf8E8M0FNU>) -> tensor<4x2xui8>
  "check.expect_eq"(%full_sh_u8, %exp_sh_u8) : (tensor<4x2xui8>, tensor<4x2xui8>) -> ()
  return
}
