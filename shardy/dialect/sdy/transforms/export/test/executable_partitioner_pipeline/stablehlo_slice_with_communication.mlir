// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>
sdy.mesh @mesh_2d = <["a"=2, "b"=2]>

func.func @parallel_slice_with_communication(
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [1:5, 0:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x4xi32>) -> tensor<4x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

func.func @sequential_slice_with_communication(%arg0: tensor<8x4xi32>) -> tensor<4x4xi32> {
  %0 = stablehlo.slice %arg0 [1:5, 0:4] : (tensor<8x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

func.func @parallel_slice_comm_free_to_one_and_comm(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"a"}, {"b"}]>})
  -> (tensor<1x3xi32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:1, 1:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"a"}, {"b"}]>]>} : (tensor<4x4xi32>) -> tensor<1x3xi32>
  %1 = sdy.reshard %0 <@mesh_2d, [{}, {}]> : tensor<1x3xi32>
  return %1 : tensor<1x3xi32>
}

func.func @sequential_slice_comm_free_to_one_and_comm(%arg0: tensor<4x4xi32>) -> tensor<1x3xi32> {
  %0 = stablehlo.slice %arg0 [0:1, 1:4] : (tensor<4x4xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input = stablehlo.reshape %input_seq : (tensor<32xi32>) -> tensor<8x4xi32>

  %seq = func.call @sequential_slice_with_communication(%input) : (tensor<8x4xi32>) -> tensor<4x4xi32>

  // The input is sharded into 2 sub-tensors of size 4x4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<4x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<4x4xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_slice_with_communication, @parallel_slice_with_communication]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<4x4xi32>, tensor<4x4xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()

  %input2_seq = stablehlo.iota dim = 0 : tensor<16xi32>
  %input2 = stablehlo.reshape %input2_seq : (tensor<16xi32>) -> tensor<4x4xi32>
  %seq2 = func.call @sequential_slice_comm_free_to_one_and_comm(%input2) : (tensor<4x4xi32>) -> tensor<1x3xi32>

  %s2_0 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %s2_1 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %s2_2 = "stablehlo.slice"(%input2) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %s2_3 = "stablehlo.slice"(%input2) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>

  %res2:4 = "interpreter.run_parallel"(%s2_0, %s2_1, %s2_2, %s2_3) {
    programs = [[@parallel_slice_comm_free_to_one_and_comm, @parallel_slice_comm_free_to_one_and_comm, @parallel_slice_comm_free_to_one_and_comm, @parallel_slice_comm_free_to_one_and_comm]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<1x3xi32>, tensor<1x3xi32>, tensor<1x3xi32>, tensor<1x3xi32>)

  "check.expect_eq"(%res2#0, %seq2) : (tensor<1x3xi32>, tensor<1x3xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<1x3xi32>, tensor<1x3xi32>) -> ()
  "check.expect_eq"(%res2#2, %seq2) : (tensor<1x3xi32>, tensor<1x3xi32>) -> ()
  "check.expect_eq"(%res2#3, %seq2) : (tensor<1x3xi32>, tensor<1x3xi32>) -> ()

  return
}
