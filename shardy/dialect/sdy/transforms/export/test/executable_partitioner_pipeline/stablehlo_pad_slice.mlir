// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_a4 = <["a"=4]>
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

func.func @parallel_pad_slice_right_shift(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [3], high = [1], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<8xi32>, tensor<i32>) -> tensor<12xi32>
  %1 = stablehlo.slice %0 [1:9]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<12xi32>) -> tensor<8xi32>
  return %1 : tensor<8xi32>
}

func.func @parallel_pad_slice_left_shift(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [1], high = [3], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<8xi32>, tensor<i32>) -> tensor<12xi32>
  %1 = stablehlo.slice %0 [2:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<12xi32>) -> tensor<8xi32>
  return %1 : tensor<8xi32>
}

func.func @parallel_pad_slice_noop_cancel(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [2], high = [3], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<8xi32>, tensor<i32>) -> tensor<13xi32>
  %1 = stablehlo.slice %0 [2:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<13xi32>) -> tensor<8xi32>
  return %1 : tensor<8xi32>
}

func.func @parallel_pad_slice_all_padding(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<-7> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [3], high = [2], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<8xi32>, tensor<i32>) -> tensor<13xi32>
  %1 = stablehlo.slice %0 [0:3]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<13xi32>) -> tensor<3xi32>
  return %1 : tensor<3xi32>
}

func.func @parallel_pad_slice_all_padding_high(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<-7> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [1], high = [4], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<8xi32>, tensor<i32>) -> tensor<13xi32>
  %1 = stablehlo.slice %0 [9:12]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<13xi32>) -> tensor<3xi32>
  return %1 : tensor<3xi32>
}

func.func @parallel_pad_slice_interior_other_dim(
  %arg0: tensor<4x3xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {}]>})
  -> (tensor<4x7xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [2, 1], high = [2, 1], interior = [0, 1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {}]>]>} : (tensor<4x3xi32>, tensor<i32>) -> tensor<8x7xi32>
  %1 = stablehlo.slice %0 [3:7, 0:7]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {}]>]>} : (tensor<8x7xi32>) -> tensor<4x7xi32>
  return %1 : tensor<4x7xi32>
}

func.func @parallel_pad_slice_indivisible(
  %arg0: tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<15xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<-3> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [1], high = [2], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<16xi32>, tensor<i32>) -> tensor<19xi32>
  %1 = stablehlo.slice %0 [0:15]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<19xi32>) -> tensor<15xi32>
  return %1 : tensor<15xi32>
}

//--- part2.mlir

func.func @main() {
  %input8 = stablehlo.iota dim = 0 : tensor<8xi32>
  %s8_0 = "stablehlo.slice"(%input8) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s8_1 = "stablehlo.slice"(%input8) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s8_2 = "stablehlo.slice"(%input8) {start_indices = array<i64: 4>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s8_3 = "stablehlo.slice"(%input8) {start_indices = array<i64: 6>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>

  %seq_right = func.call @sequential_pad_slice_right_shift(%input8) : (tensor<8xi32>) -> tensor<8xi32>
  %res_right:4 = "interpreter.run_parallel"(%s8_0, %s8_1, %s8_2, %s8_3) {
    programs = [[@parallel_pad_slice_right_shift, @parallel_pad_slice_right_shift, @parallel_pad_slice_right_shift, @parallel_pad_slice_right_shift]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_right#0, %seq_right) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_right#1, %seq_right) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_right#2, %seq_right) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_right#3, %seq_right) : (tensor<8xi32>, tensor<8xi32>) -> ()

  %seq_left = func.call @sequential_pad_slice_left_shift(%input8) : (tensor<8xi32>) -> tensor<8xi32>
  %res_left:4 = "interpreter.run_parallel"(%s8_0, %s8_1, %s8_2, %s8_3) {
    programs = [[@parallel_pad_slice_left_shift, @parallel_pad_slice_left_shift, @parallel_pad_slice_left_shift, @parallel_pad_slice_left_shift]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_left#0, %seq_left) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_left#1, %seq_left) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_left#2, %seq_left) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_left#3, %seq_left) : (tensor<8xi32>, tensor<8xi32>) -> ()

  %seq_noop = func.call @sequential_pad_slice_noop_cancel(%input8) : (tensor<8xi32>) -> tensor<8xi32>
  %res_noop:4 = "interpreter.run_parallel"(%s8_0, %s8_1, %s8_2, %s8_3) {
    programs = [[@parallel_pad_slice_noop_cancel, @parallel_pad_slice_noop_cancel, @parallel_pad_slice_noop_cancel, @parallel_pad_slice_noop_cancel]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res_noop#0, %seq_noop) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_noop#1, %seq_noop) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_noop#2, %seq_noop) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res_noop#3, %seq_noop) : (tensor<8xi32>, tensor<8xi32>) -> ()

  %seq_all_pad = func.call @sequential_pad_slice_all_padding(%input8) : (tensor<8xi32>) -> tensor<3xi32>
  %res_all_pad:4 = "interpreter.run_parallel"(%s8_0, %s8_1, %s8_2, %s8_3) {
    programs = [[@parallel_pad_slice_all_padding, @parallel_pad_slice_all_padding, @parallel_pad_slice_all_padding, @parallel_pad_slice_all_padding]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)
  "check.expect_eq"(%res_all_pad#0, %seq_all_pad) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res_all_pad#1, %seq_all_pad) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res_all_pad#2, %seq_all_pad) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res_all_pad#3, %seq_all_pad) : (tensor<3xi32>, tensor<3xi32>) -> ()

  %seq_all_pad_high = func.call @sequential_pad_slice_all_padding_high(%input8) : (tensor<8xi32>) -> tensor<3xi32>
  %res_all_pad_high:4 = "interpreter.run_parallel"(%s8_0, %s8_1, %s8_2, %s8_3) {
    programs = [[@parallel_pad_slice_all_padding_high, @parallel_pad_slice_all_padding_high, @parallel_pad_slice_all_padding_high, @parallel_pad_slice_all_padding_high]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)
  "check.expect_eq"(%res_all_pad_high#0, %seq_all_pad_high) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res_all_pad_high#1, %seq_all_pad_high) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res_all_pad_high#2, %seq_all_pad_high) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res_all_pad_high#3, %seq_all_pad_high) : (tensor<3xi32>, tensor<3xi32>) -> ()

  %input2d_seq = stablehlo.iota dim = 0 : tensor<12xi32>
  %input2d = stablehlo.reshape %input2d_seq : (tensor<12xi32>) -> tensor<4x3xi32>
  %seq_interior = func.call @sequential_pad_slice_interior_other_dim(%input2d) : (tensor<4x3xi32>) -> tensor<4x7xi32>
  %s2d_0 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 3>, strides = array<i64: 1, 1>} : (tensor<4x3xi32>) -> tensor<2x3xi32>
  %s2d_1 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 3>, strides = array<i64: 1, 1>} : (tensor<4x3xi32>) -> tensor<2x3xi32>
  %res_interior:4 = "interpreter.run_parallel"(%s2d_0, %s2d_0, %s2d_1, %s2d_1) {
    programs = [[@parallel_pad_slice_interior_other_dim, @parallel_pad_slice_interior_other_dim, @parallel_pad_slice_interior_other_dim, @parallel_pad_slice_interior_other_dim]]
  } : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<4x7xi32>, tensor<4x7xi32>, tensor<4x7xi32>, tensor<4x7xi32>)
  "check.expect_eq"(%res_interior#0, %seq_interior) : (tensor<4x7xi32>, tensor<4x7xi32>) -> ()
  "check.expect_eq"(%res_interior#1, %seq_interior) : (tensor<4x7xi32>, tensor<4x7xi32>) -> ()
  "check.expect_eq"(%res_interior#2, %seq_interior) : (tensor<4x7xi32>, tensor<4x7xi32>) -> ()
  "check.expect_eq"(%res_interior#3, %seq_interior) : (tensor<4x7xi32>, tensor<4x7xi32>) -> ()

  %input16 = stablehlo.iota dim = 0 : tensor<16xi32>
  %seq_indiv = func.call @sequential_pad_slice_indivisible(%input16) : (tensor<16xi32>) -> tensor<15xi32>
  %s16_0 = "stablehlo.slice"(%input16) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<16xi32>) -> tensor<4xi32>
  %s16_1 = "stablehlo.slice"(%input16) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<16xi32>) -> tensor<4xi32>
  %s16_2 = "stablehlo.slice"(%input16) {start_indices = array<i64: 8>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<16xi32>) -> tensor<4xi32>
  %s16_3 = "stablehlo.slice"(%input16) {start_indices = array<i64: 12>, limit_indices = array<i64: 16>, strides = array<i64: 1>} : (tensor<16xi32>) -> tensor<4xi32>
  %res_indiv:4 = "interpreter.run_parallel"(%s16_0, %s16_1, %s16_2, %s16_3) {
    programs = [[@parallel_pad_slice_indivisible, @parallel_pad_slice_indivisible, @parallel_pad_slice_indivisible, @parallel_pad_slice_indivisible]]
  } : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> (tensor<15xi32>, tensor<15xi32>, tensor<15xi32>, tensor<15xi32>)
  "check.expect_eq"(%res_indiv#0, %seq_indiv) : (tensor<15xi32>, tensor<15xi32>) -> ()
  "check.expect_eq"(%res_indiv#1, %seq_indiv) : (tensor<15xi32>, tensor<15xi32>) -> ()
  "check.expect_eq"(%res_indiv#2, %seq_indiv) : (tensor<15xi32>, tensor<15xi32>) -> ()
  "check.expect_eq"(%res_indiv#3, %seq_indiv) : (tensor<15xi32>, tensor<15xi32>) -> ()

  return
}
