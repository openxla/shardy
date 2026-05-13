// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_4 = <["x"=4]>

// This function takes a 16x8 global tensor sharded by "x" on dim 0
// (4x8 per device) and returns a 16x8 global tensor sharded by "x" on dim 1
// (16x2 per device).
func.func @manual_all_to_all(
  %arg0: tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {"x"}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xi32>) {
      %1 = "stablehlo.all_to_all"(%arg1) {
        split_dimension = 1 : i64,
        concat_dimension = 0 : i64,
        split_count = 4 : i64,
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        use_global_device_ids
      } : (tensor<4x8xi32>) -> tensor<16x2xi32>
      sdy.return %1 : tensor<16x2xi32>
  } : (tensor<16x8xi32>) -> (tensor<16x8xi32>)
  return %0 : tensor<16x8xi32>
}

//--- part2.mlir

func.func @main() {
  // Create distinct 4x8 local shards for the 4 virtual devices.
  %s0 = stablehlo.constant dense<1> : tensor<4x8xi32>
  %s1 = stablehlo.constant dense<10> : tensor<4x8xi32>
  %s2 = stablehlo.constant dense<100> : tensor<4x8xi32>
  %s3 = stablehlo.constant dense<1000> : tensor<4x8xi32>

  // Expected local result on every device (since input shards were uniform).
  // The result on every device will contain segments of 4x2 from all devices
  // concatenated on dim 0.
  %p0 = stablehlo.constant dense<1> : tensor<4x2xi32>
  %p1 = stablehlo.constant dense<10> : tensor<4x2xi32>
  %p2 = stablehlo.constant dense<100> : tensor<4x2xi32>
  %p3 = stablehlo.constant dense<1000> : tensor<4x2xi32>
  %expected = "stablehlo.concatenate"(%p0, %p1, %p2, %p3) {dimension = 0 : i64}
    : (tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<16x2xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@manual_all_to_all, @manual_all_to_all, @manual_all_to_all, @manual_all_to_all]]
  } : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) ->
      (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<16x2xi32>, tensor<16x2xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<16x2xi32>, tensor<16x2xi32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<16x2xi32>, tensor<16x2xi32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<16x2xi32>, tensor<16x2xi32>) -> ()

  return
}
