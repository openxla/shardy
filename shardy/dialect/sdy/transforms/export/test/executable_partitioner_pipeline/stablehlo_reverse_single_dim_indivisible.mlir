// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=4]>

func.func @parallel_reverse(
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  -> tensor<4x6x5xi32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : (tensor<4x6x8xi32>) -> tensor<4x6x5xi32>
  %1 = stablehlo.reverse %0, dims = [0, 2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : tensor<4x6x5xi32>
  %2 = sdy.reshard %1 <@mesh_abc, [{}, {}, {}]> : tensor<4x6x5xi32>
  return %2 : tensor<4x6x5xi32>
}

func.func @sequential_reverse(%arg0: tensor<4x6x8xi32>) -> tensor<4x6x5xi32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
    : (tensor<4x6x8xi32>) -> tensor<4x6x5xi32>
  %1 = stablehlo.reverse %0, dims = [0, 2]
    : tensor<4x6x5xi32>
  return %1 : tensor<4x6x5xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<192xi32>
  %input = stablehlo.reshape %input_seq : (tensor<192xi32>) -> tensor<4x6x8xi32>

  %seq = func.call @sequential_reverse(%input) : (tensor<4x6x8xi32>) -> tensor<4x6x5xi32>

  %s000 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 2, 3, 2>, strides = array<i64: 1, 1, 1>
  } : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>

  // The input tensor<4x6x8xi32> is sharded [{"b"}, {"a"}, {"c"}] on mesh a=2, b=2, c=4.
  // This results in 16 shards of shape 2x3x2.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 2, 3, 2>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 2>, limit_indices = array<i64: 2, 3, 4>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 4>, limit_indices = array<i64: 2, 3, 6>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 6>, limit_indices = array<i64: 2, 3, 8>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s4 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0, 0>, limit_indices = array<i64: 4, 3, 2>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s5 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0, 2>, limit_indices = array<i64: 4, 3, 4>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s6 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0, 4>, limit_indices = array<i64: 4, 3, 6>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s7 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0, 6>, limit_indices = array<i64: 4, 3, 8>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s8 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 3, 0>, limit_indices = array<i64: 2, 6, 2>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %s9 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 3, 2>, limit_indices = array<i64: 2, 6, 4>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %sa = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 3, 4>, limit_indices = array<i64: 2, 6, 6>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %sb = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 3, 6>, limit_indices = array<i64: 2, 6, 8>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %sc = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 3, 0>, limit_indices = array<i64: 4, 6, 2>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %sd = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 3, 2>, limit_indices = array<i64: 4, 6, 4>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %se = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 3, 4>, limit_indices = array<i64: 4, 6, 6>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>
  %sf = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 3, 6>, limit_indices = array<i64: 4, 6, 8>, strides = array<i64: 1, 1, 1>} : (tensor<4x6x8xi32>) -> tensor<2x3x2xi32>

  %pars:16 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %sa, %sb, %sc, %sd, %se, %sf) {
    programs = [[@parallel_reverse, @parallel_reverse, @parallel_reverse, @parallel_reverse,
      @parallel_reverse, @parallel_reverse, @parallel_reverse, @parallel_reverse,
      @parallel_reverse, @parallel_reverse, @parallel_reverse, @parallel_reverse,
      @parallel_reverse, @parallel_reverse, @parallel_reverse, @parallel_reverse
    ]]
  } : (tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>,
       tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>,
       tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>,
       tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>, tensor<2x3x2xi32>) ->
      (tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>,
       tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>,
       tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>,
       tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>, tensor<4x6x5xi32>)

  check.expect_eq %seq, %pars#0 : tensor<4x6x5xi32>

  return
}
