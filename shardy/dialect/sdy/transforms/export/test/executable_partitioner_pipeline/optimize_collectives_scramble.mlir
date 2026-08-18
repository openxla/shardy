// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @parallel_optimize_collectives_scramble(%arg0: tensor<8x8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}, {}]>}) -> (tensor<8x8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {"x"}]> : tensor<8x8x8xi32>
  return %0 : tensor<8x8x8xi32>
}

func.func @sequential_optimize_collectives_scramble(%arg0: tensor<8x8x8xi32>) -> tensor<8x8x8xi32> {
  return %arg0 : tensor<8x8x8xi32>
}

//--- part2.mlir
func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<8x8x8xi32>
  %c1 = stablehlo.constant dense<1> : tensor<8x8x8xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<8x8x8xi32>
  %seq = func.call @sequential_optimize_collectives_scramble(%input) : (tensor<8x8x8xi32>) -> tensor<8x8x8xi32>

  // Input slices (dim 0 sharded by {"x", "y"}):
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 2, 8, 8>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<2x8x8xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0, 0>, limit_indices = array<i64: 4, 8, 8>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<2x8x8xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0, 0>, limit_indices = array<i64: 6, 8, 8>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<2x8x8xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 0, 0>, limit_indices = array<i64: 8, 8, 8>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<2x8x8xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_optimize_collectives_scramble, @parallel_optimize_collectives_scramble, @parallel_optimize_collectives_scramble, @parallel_optimize_collectives_scramble]]
  } : (tensor<2x8x8xi32>, tensor<2x8x8xi32>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) ->
      (tensor<8x4x4xi32>, tensor<8x4x4xi32>, tensor<8x4x4xi32>, tensor<8x4x4xi32>)

  // Expected output slices (dim 1 sharded by "y", dim 2 sharded by "x"):
  // Device 0 (x=0, y=0): [0:8, 0:4, 0:4]
  %exp0 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 8, 4, 4>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<8x4x4xi32>
  // Device 1 (x=0, y=1): [0:8, 4:8, 0:4]
  %exp1 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 4, 0>, limit_indices = array<i64: 8, 8, 4>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<8x4x4xi32>
  // Device 2 (x=1, y=0): [0:8, 0:4, 4:8]
  %exp2 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 0, 4>, limit_indices = array<i64: 8, 4, 8>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<8x4x4xi32>
  // Device 3 (x=1, y=1): [0:8, 4:8, 4:8]
  %exp3 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 4, 4>, limit_indices = array<i64: 8, 8, 8>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x8xi32>) -> tensor<8x4x4xi32>

  "check.expect_eq"(%res#0, %exp0) : (tensor<8x4x4xi32>, tensor<8x4x4xi32>) -> ()
  "check.expect_eq"(%res#1, %exp1) : (tensor<8x4x4xi32>, tensor<8x4x4xi32>) -> ()
  "check.expect_eq"(%res#2, %exp2) : (tensor<8x4x4xi32>, tensor<8x4x4xi32>) -> ()
  "check.expect_eq"(%res#3, %exp3) : (tensor<8x4x4xi32>, tensor<8x4x4xi32>) -> ()
  return
}
