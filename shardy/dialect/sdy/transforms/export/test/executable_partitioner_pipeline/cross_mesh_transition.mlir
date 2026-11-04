// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2x2 = <["x"=2, "y"=2]>
sdy.mesh @mesh_4 = <["z"=4]>

func.func @parallel_cross_mesh(
    %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_2x2, [{"x"}, {"y"}]>})
    -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  // Compute elementwise addition on @mesh_2x2.
  %0 = stablehlo.add %arg0, %arg0 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2x2, [{"x"}, {"y"}]>]>
  } : tensor<8x8xi32>

  // Transition from 2x2 grid (@mesh_2x2) to 1D 4-device mesh (@mesh_4)
  // via the 3-step replicated pivot:
  //   Tiled(@mesh_2x2) -> Replicated(@mesh_2x2) -> Replicated(@mesh_4) -> Tiled(@mesh_4).
  %1 = sdy.reshard %0 <@mesh_4, [{"z"}, {}]> : tensor<8x8xi32>

  // Compute elementwise multiplication on @mesh_4.
  %c3 = stablehlo.constant dense<3> : tensor<8x8xi32>
  %2 = stablehlo.multiply %1, %c3 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"z"}, {}]>]>
  } : tensor<8x8xi32>

  // Replicate output on @mesh_4 so all devices obtain the full result tensor.
  %3 = sdy.reshard %2 <@mesh_4, [{}, {}]> : tensor<8x8xi32>
  return %3 : tensor<8x8xi32>
}

func.func @sequential_cross_mesh(%arg0: tensor<8x8xi32>) -> tensor<8x8xi32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xi32>
  %c3 = stablehlo.constant dense<3> : tensor<8x8xi32>
  %1 = stablehlo.multiply %0, %c3 : tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<64xi32>
  %input = stablehlo.reshape %input_seq : (tensor<64xi32>) -> tensor<8x8xi32>

  %seq = func.call @sequential_cross_mesh(%input) : (tensor<8x8xi32>) -> tensor<8x8xi32>

  // Slice input into 4 local shards of size 4x4 matching @mesh_2x2 [{"x"}, {"y"}]:
  // Device 0 (x=0, y=0): rows [0:4], cols [0:4]
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  // Device 1 (x=0, y=1): rows [0:4], cols [4:8]
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  // Device 2 (x=1, y=0): rows [4:8], cols [0:4]
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  // Device 3 (x=1, y=1): rows [4:8], cols [4:8]
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>

  // Execute 4 partitions in parallel on the StableHLO interpreter.
  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_cross_mesh, @parallel_cross_mesh, @parallel_cross_mesh, @parallel_cross_mesh]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)

  // Verify that all 4 parallel device results match the sequential unpartitioned result.
  "check.expect_eq"(%res#0, %seq) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<8x8xi32>, tensor<8x8xi32>) -> ()

  return
}
