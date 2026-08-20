// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_slice_replicated(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %0 = stablehlo.slice %arg0 [0:3]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}]>]>} : (tensor<8xi32>) -> tensor<3xi32>
  %1 = sdy.all_gather [{"x"}] %0 out_sharding=<@mesh, [{}]> : tensor<3xi32>
  return %1 : tensor<3xi32>
}

func.func @sequential_slice_replicated(%arg0: tensor<8xi32>) -> tensor<3xi32> {
  %0 = stablehlo.slice %arg0 [0:3] : (tensor<8xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>

  %seq = func.call @sequential_slice_replicated(%input) : (tensor<8xi32>) -> tensor<3xi32>

  // Slice input into 2 shards of size 4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_slice_replicated, @parallel_slice_replicated]]
  } : (tensor<4xi32>, tensor<4xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3xi32>, tensor<3xi32>) -> ()

  return
}
