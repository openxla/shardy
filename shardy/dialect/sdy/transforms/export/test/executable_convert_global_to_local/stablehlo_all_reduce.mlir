// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_4 = <["x"=4]>

func.func @manual_all_reduce(
  %arg0: tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xi32>) {
      %1 = "stablehlo.all_reduce"(%arg1) ({
        ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
          %2 = stablehlo.add %arg2, %arg3 : tensor<i32>
          stablehlo.return %2 : tensor<i32>
      }) {
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        use_global_device_ids
      } : (tensor<4x8xi32>) -> tensor<4x8xi32>
      sdy.return %1 : tensor<4x8xi32>
  } : (tensor<16x8xi32>) -> (tensor<4x8xi32>)
  return %0 : tensor<4x8xi32>
}

//--- part2.mlir

func.func @main() {
  // Create distinct 4x8 local shards for the 4 virtual devices.
  %s0 = stablehlo.constant dense<1> : tensor<4x8xi32>
  %s1 = stablehlo.constant dense<10> : tensor<4x8xi32>
  %s2 = stablehlo.constant dense<100> : tensor<4x8xi32>
  %s3 = stablehlo.constant dense<1000> : tensor<4x8xi32>

  // Expected global result for a sum reduction: 1 + 10 + 100 + 1000 = 1111.
  %expected = stablehlo.constant dense<1111> : tensor<4x8xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@manual_all_reduce, @manual_all_reduce, @manual_all_reduce, @manual_all_reduce]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) ->
      (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

  return
}
