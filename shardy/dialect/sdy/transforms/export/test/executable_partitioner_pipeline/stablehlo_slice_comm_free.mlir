// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_slice_comm_free(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<4x4xi32>) -> tensor<4x2xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<4x2xi32>
  return %1 : tensor<4x2xi32>
}

func.func @sequential_slice_comm_free(%arg0: tensor<4x4xi32>) -> tensor<4x2xi32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:2] : (tensor<4x4xi32>) -> tensor<4x2xi32>
  return %0 : tensor<4x2xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<16xi32>
  %input = stablehlo.reshape %input_seq : (tensor<16xi32>) -> tensor<4x4xi32>

  %seq = func.call @sequential_slice_comm_free(%input) : (tensor<4x4xi32>) -> tensor<4x2xi32>

  // Slice input into 2 shards of size 2x4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x4xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_slice_comm_free, @parallel_slice_comm_free]]
  } : (tensor<2x4xi32>, tensor<2x4xi32>) -> (tensor<4x2xi32>, tensor<4x2xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()

  return
}
