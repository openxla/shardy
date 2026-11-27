// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_x8 = <["x"=8]>

// Shard t of the result reads shards 2t and 2t+1 of the operand.
func.func @parallel_pad_double_length(
  %arg0: tensor<16x3xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{"x"}, {}]>})
  -> (tensor<32x3xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{}, {}]>}) {
  %c = stablehlo.constant dense<-1> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [16, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x8, [{"x"}, {}]>]>} : (tensor<16x3xi32>, tensor<i32>) -> tensor<32x3xi32>
  %1 = sdy.reshard %0 <@mesh_x8, [{}, {}]> : tensor<32x3xi32>
  return %1 : tensor<32x3xi32>
}

func.func @sequential_pad_double_length(%arg0: tensor<16x3xi32>, %arg1: tensor<i32>) -> tensor<32x3xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 0], high = [16, 0], interior = [0, 0] : (tensor<16x3xi32>, tensor<i32>) -> tensor<32x3xi32>
  return %0 : tensor<32x3xi32>
}

// Each result shard reads halves of two operand shards 4 and 3 shards away.
func.func @parallel_pad_shift_by_seven_rows(
  %arg0: tensor<16x3xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{"x"}, {}]>})
  -> (tensor<16x3xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{}, {}]>}) {
  %c = stablehlo.constant dense<-1> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [7, 0], high = [-7, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x8, [{"x"}, {}]>]>} : (tensor<16x3xi32>, tensor<i32>) -> tensor<16x3xi32>
  %1 = sdy.reshard %0 <@mesh_x8, [{}, {}]> : tensor<16x3xi32>
  return %1 : tensor<16x3xi32>
}

func.func @sequential_pad_shift_by_seven_rows(%arg0: tensor<16x3xi32>, %arg1: tensor<i32>) -> tensor<16x3xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [7, 0], high = [-7, 0], interior = [0, 0] : (tensor<16x3xi32>, tensor<i32>) -> tensor<16x3xi32>
  return %0 : tensor<16x3xi32>
}

// Non-uniform intra-shard offsets (sIn = 3, sOut = 8): exercises both kUnused
// in the last round and dynamic_slice over concatenated source rounds.
func.func @parallel_pad_uneven_offsets(
  %arg0: tensor<24x3xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{"x"}, {}]>})
  -> (tensor<64x3xi32> {sdy.sharding = #sdy.sharding<@mesh_x8, [{}, {}]>}) {
  %c = stablehlo.constant dense<-1> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [40, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x8, [{"x"}, {}]>]>} : (tensor<24x3xi32>, tensor<i32>) -> tensor<64x3xi32>
  %1 = sdy.reshard %0 <@mesh_x8, [{}, {}]> : tensor<64x3xi32>
  return %1 : tensor<64x3xi32>
}

func.func @sequential_pad_uneven_offsets(%arg0: tensor<24x3xi32>, %arg1: tensor<i32>) -> tensor<64x3xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 0], high = [40, 0], interior = [0, 0] : (tensor<24x3xi32>, tensor<i32>) -> tensor<64x3xi32>
  return %0 : tensor<64x3xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<-1> : tensor<i32>
  %input_seq = stablehlo.iota dim = 0 : tensor<48xi32>
  %input = stablehlo.reshape %input_seq : (tensor<48xi32>) -> tensor<16x3xi32>

  // The input is sharded into 8 sub-tensors of size 2x3.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 6, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 8, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s4 = "stablehlo.slice"(%input) {start_indices = array<i64: 8, 0>, limit_indices = array<i64: 10, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s5 = "stablehlo.slice"(%input) {start_indices = array<i64: 10, 0>, limit_indices = array<i64: 12, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s6 = "stablehlo.slice"(%input) {start_indices = array<i64: 12, 0>, limit_indices = array<i64: 14, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>
  %s7 = "stablehlo.slice"(%input) {start_indices = array<i64: 14, 0>, limit_indices = array<i64: 16, 3>, strides = array<i64: 1, 1>} : (tensor<16x3xi32>) -> tensor<2x3xi32>

  %seq_double = func.call @sequential_pad_double_length(%input, %c) : (tensor<16x3xi32>, tensor<i32>) -> tensor<32x3xi32>
  %res_double:8 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7) {
    programs = [[@parallel_pad_double_length, @parallel_pad_double_length, @parallel_pad_double_length, @parallel_pad_double_length, @parallel_pad_double_length, @parallel_pad_double_length, @parallel_pad_double_length, @parallel_pad_double_length]]
  } : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<32x3xi32>, tensor<32x3xi32>, tensor<32x3xi32>, tensor<32x3xi32>, tensor<32x3xi32>, tensor<32x3xi32>, tensor<32x3xi32>, tensor<32x3xi32>)
  "check.expect_eq"(%res_double#0, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#1, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#2, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#3, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#4, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#5, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#6, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()
  "check.expect_eq"(%res_double#7, %seq_double) : (tensor<32x3xi32>, tensor<32x3xi32>) -> ()

  %seq_shift = func.call @sequential_pad_shift_by_seven_rows(%input, %c) : (tensor<16x3xi32>, tensor<i32>) -> tensor<16x3xi32>
  %res_shift:8 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7) {
    programs = [[@parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows, @parallel_pad_shift_by_seven_rows]]
  } : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>)
  "check.expect_eq"(%res_shift#0, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#1, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#2, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#3, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#4, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#5, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#6, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()
  "check.expect_eq"(%res_shift#7, %seq_shift) : (tensor<16x3xi32>, tensor<16x3xi32>) -> ()

  %input24_seq = stablehlo.iota dim = 0 : tensor<72xi32>
  %input24 = stablehlo.reshape %input24_seq : (tensor<72xi32>) -> tensor<24x3xi32>
  %u0 = "stablehlo.slice"(%input24) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 3, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u1 = "stablehlo.slice"(%input24) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 6, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u2 = "stablehlo.slice"(%input24) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 9, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u3 = "stablehlo.slice"(%input24) {start_indices = array<i64: 9, 0>, limit_indices = array<i64: 12, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u4 = "stablehlo.slice"(%input24) {start_indices = array<i64: 12, 0>, limit_indices = array<i64: 15, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u5 = "stablehlo.slice"(%input24) {start_indices = array<i64: 15, 0>, limit_indices = array<i64: 18, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u6 = "stablehlo.slice"(%input24) {start_indices = array<i64: 18, 0>, limit_indices = array<i64: 21, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>
  %u7 = "stablehlo.slice"(%input24) {start_indices = array<i64: 21, 0>, limit_indices = array<i64: 24, 3>, strides = array<i64: 1, 1>} : (tensor<24x3xi32>) -> tensor<3x3xi32>

  %seq_uneven = func.call @sequential_pad_uneven_offsets(%input24, %c) : (tensor<24x3xi32>, tensor<i32>) -> tensor<64x3xi32>
  %res_uneven:8 = "interpreter.run_parallel"(%u0, %u1, %u2, %u3, %u4, %u5, %u6, %u7) {
    programs = [[@parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets, @parallel_pad_uneven_offsets]]
  } : (tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>) -> (tensor<64x3xi32>, tensor<64x3xi32>, tensor<64x3xi32>, tensor<64x3xi32>, tensor<64x3xi32>, tensor<64x3xi32>, tensor<64x3xi32>, tensor<64x3xi32>)
  "check.expect_eq"(%res_uneven#0, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#1, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#2, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#3, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#4, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#5, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#6, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()
  "check.expect_eq"(%res_uneven#7, %seq_uneven) : (tensor<64x3xi32>, tensor<64x3xi32>) -> ()

  return
}
