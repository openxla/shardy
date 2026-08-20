// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_a4 = <["a"=4]>

func.func @parallel_pad_interior(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [1], high = [0], interior = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}]>]>} : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}]> : tensor<8xi32>
  return %1 : tensor<8xi32>
}

func.func @sequential_pad_interior(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> tensor<8xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [1], high = [0], interior = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  %seq = func.call @sequential_pad_interior(%input, %c) : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>

  // Slice input into 4 shards of size 1.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_pad_interior, @parallel_pad_interior, @parallel_pad_interior, @parallel_pad_interior]]
  } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<8xi32>, tensor<8xi32>) -> ()

  return
}
