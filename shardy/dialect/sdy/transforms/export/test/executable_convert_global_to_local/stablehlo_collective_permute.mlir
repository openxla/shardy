// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_4 = <["x"=4]>

// This function takes a 16x8 global tensor sharded by "x" (4x8 per device) and
// performs a cyclic shift (0->1, 1->2, 2->3, 3->0).
func.func @manual_collective_permute(
  %arg0: tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{"x"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xi32>) {
      %1 = "stablehlo.collective_permute"(%arg1) {
        source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
      } : (tensor<4x8xi32>) -> tensor<4x8xi32>
      sdy.return %1 : tensor<4x8xi32>
  } : (tensor<16x8xi32>) -> (tensor<16x8xi32>)
  return %0 : tensor<16x8xi32>
}

//--- part2.mlir

func.func @main() {
  %s0 = stablehlo.constant dense<1> : tensor<4x8xi32>
  %s1 = stablehlo.constant dense<10> : tensor<4x8xi32>
  %s2 = stablehlo.constant dense<100> : tensor<4x8xi32>
  %s3 = stablehlo.constant dense<1000> : tensor<4x8xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@manual_collective_permute, @manual_collective_permute, @manual_collective_permute, @manual_collective_permute]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) ->
      (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)

  "check.expect_eq"(%res#0, %s3) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#1, %s0) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#2, %s1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#3, %s2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

  return
}

