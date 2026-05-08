// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_4 = <["x"=4]>

func.func @manual_all_gather(
  %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xf32>) {
      %1 = "stablehlo.all_gather"(%arg1) {
        all_gather_dim = 0 : i64,
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        use_global_device_ids
      } : (tensor<4x8xf32>) -> tensor<16x8xf32>
      sdy.return %1 : tensor<16x8xf32>
  } : (tensor<16x8xf32>) -> (tensor<16x8xf32>)
  return %0 : tensor<16x8xf32>
}

//--- part2.mlir

func.func @main() {
  // Create distinct 4x8 local shards for the 4 devices.
  %s0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %s1 = stablehlo.constant dense<10.0> : tensor<4x8xf32>
  %s2 = stablehlo.constant dense<100.0> : tensor<4x8xf32>
  %s3 = stablehlo.constant dense<1000.0> : tensor<4x8xf32>

  // Create the expected global tensor for verification.
  %input = "stablehlo.concatenate"(%s0, %s1, %s2, %s3) {dimension = 0 : i64}
    : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<16x8xf32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@manual_all_gather, @manual_all_gather, @manual_all_gather, @manual_all_gather]]
  } : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>, tensor<16x8xf32>, tensor<16x8xf32>)

  "check.expect_eq"(%res#0, %input) : (tensor<16x8xf32>, tensor<16x8xf32>) -> ()
  "check.expect_eq"(%res#1, %input) : (tensor<16x8xf32>, tensor<16x8xf32>) -> ()
  "check.expect_eq"(%res#2, %input) : (tensor<16x8xf32>, tensor<16x8xf32>) -> ()
  "check.expect_eq"(%res#3, %input) : (tensor<16x8xf32>, tensor<16x8xf32>) -> ()

  return
}
