// RUN: %S/run_sdy_interpreter_test.sh %s %t


//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// Performs a slice op on a single device.
//
func.func @sequential_slice(%arg0: tensor<2x4xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.slice %arg0 [0:2, 0:2] : (tensor<2x4xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// Performs the same slice above, but on 2 devices in parallel.
//
// We will use sdy-opt to convert this to a device local program that
// stablehlo interpreter can execute.
//
func.func @parallel_slice(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:2, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
    : (tensor<2x4xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

//--- part2.mlir

// Main Orchestrator: executes the sequential and parallel iota and checks that
// they are equivalent.
func.func @main() {
  // Define a global 2D input: 2 rows, 4 columns.
  %input = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

  // Sequential Run
  // This produces the full 2x2 slice: [[1.0, 2.0], [5.0, 6.0]]
  %seq = func.call @sequential_slice(%input) : (tensor<2x4xf32>) -> tensor<2x2xf32>

  // Parallel Run
  // We manually shard the input along dimension 0 (rows) for 2 virtual devices.
  // Shard 0: Row 0 (tensor<1x4xf32>)
  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 1, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<2x4xf32>) -> tensor<1x4xf32>

  // Shard 1: Row 1 (tensor<1x4xf32>)
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<2x4xf32>) -> tensor<1x4xf32>

  // Execute the localized program in parallel across 2 devices.
  // Device 0 computes [[1.0, 2.0]], Device 1 computes [[5.0, 6.0]].
  %pars:2 = "interpreter.run_parallel"(%shard0, %shard1) {
    programs = [[@parallel_slice], [@parallel_slice]]
  } : (tensor<1x4xf32>, tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)

  // Concatenate the local shards along the sharded dimension (dimension 0).
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>

  check.expect_eq %seq, %par : tensor<2x2xf32>

  return
}
