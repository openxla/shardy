// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_slice_strided(
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<3x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [1:7:2, 0:4]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<8x4xi32>) -> tensor<3x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<3x4xi32>
  return %1 : tensor<3x4xi32>
}

func.func @sequential_slice_strided(%arg0: tensor<8x4xi32>) -> tensor<3x4xi32> {
  %0 = stablehlo.slice %arg0 [1:7:2, 0:4] : (tensor<8x4xi32>) -> tensor<3x4xi32>
  return %0 : tensor<3x4xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input = stablehlo.reshape %input_seq : (tensor<32xi32>) -> tensor<8x4xi32>

  %seq = func.call @sequential_slice_strided(%input) : (tensor<8x4xi32>) -> tensor<3x4xi32>

  // The input is sharded into 2 sub-tensors of size 4x4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<4x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<4x4xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_slice_strided, @parallel_slice_strided]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<3x4xi32>, tensor<3x4xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()

  return
}
