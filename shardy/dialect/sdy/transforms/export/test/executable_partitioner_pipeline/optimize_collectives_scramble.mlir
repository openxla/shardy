// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @parallel_scramble(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {"x"}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  // Construct tensor<4x4xi32> where row i has elements [10*(i+1), 10*(i+1), 10*(i+1), 10*(i+1)]
  // B=0 (Dev 0,0): [10, 10, 10, 10]
  // B=1 (Dev 0,1): [20, 20, 20, 20]
  // B=2 (Dev 1,0): [30, 30, 30, 30]
  // B=3 (Dev 1,1): [40, 40, 40, 40]
  %input = stablehlo.constant dense<[[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30], [40, 40, 40, 40]]> : tensor<4x4xi32>

  // Expected output is identical to input because resharding does not change logical tensor values.
  %expected = stablehlo.constant dense<[[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30], [40, 40, 40, 40]]> : tensor<4x4xi32>

  // Initial slices for sharding [{"x", "y"}, {}] (B = dev_x * 2 + dev_y)
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<1x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<1x4xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<1x4xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<1x4xi32>

  // Run parallel program
  %pars:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_scramble, @parallel_scramble, @parallel_scramble, @parallel_scramble]]
  } : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) ->
      (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>)

  // Check output slices against target sharding [{"y"}, {"x"}]
  // dev_x=0, dev_y=0 (Device 0): B in {0, 1}, L in {0, 1} -> [[10, 10], [20, 20]]
  %exp0 = "stablehlo.slice"(%expected) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  check.expect_eq %exp0, %pars#0 : tensor<2x2xi32>

  // dev_x=0, dev_y=1 (Device 1): B in {2, 3}, L in {0, 1} -> [[30, 30], [40, 40]]
  %exp1 = "stablehlo.slice"(%expected) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  check.expect_eq %exp1, %pars#1 : tensor<2x2xi32>

  // dev_x=1, dev_y=0 (Device 2): B in {0, 1}, L in {2, 3} -> [[10, 10], [20, 20]]
  %exp2 = "stablehlo.slice"(%expected) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  check.expect_eq %exp2, %pars#2 : tensor<2x2xi32>

  // dev_x=1, dev_y=1 (Device 3): B in {2, 3}, L in {2, 3} -> [[30, 30], [40, 40]]
  %exp3 = "stablehlo.slice"(%expected) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  check.expect_eq %exp3, %pars#3 : tensor<2x2xi32>

  return
}
