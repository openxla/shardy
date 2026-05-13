// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_a = <["x"=4]>
sdy.mesh @mesh_b = <["x"=4], device_ids=[1, 2, 3, 0]>

// This function performs a collective permute by moving data
// between two meshes with different device orders.
func.func @collective_permute(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_a, [{"x"}]>})
  -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_b, [{"x"}]>}) {
  // Shardy identifies that Mesh A and Mesh B have different device layouts
  // for axis "x" and inserts a cyclic collective permute.
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_b, [{"x"}]> : tensor<4xi32>
  return %0 : tensor<4xi32>
}

//--- part2.mlir

func.func @main() {
  %cst = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  // Prepare 4 Shards (1 element each).
  %s0 = "stablehlo.slice"(%cst) {start_indices=array<i64: 0>, limit_indices=array<i64: 1>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s1 = "stablehlo.slice"(%cst) {start_indices=array<i64: 1>, limit_indices=array<i64: 2>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s2 = "stablehlo.slice"(%cst) {start_indices=array<i64: 2>, limit_indices=array<i64: 3>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s3 = "stablehlo.slice"(%cst) {start_indices=array<i64: 3>, limit_indices=array<i64: 4>, strides=array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>

  // Run the parallel programs.
  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@collective_permute, @collective_permute,
                 @collective_permute, @collective_permute]]
  } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) ->
      (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)

  // Verify the cyclic shift. The new data in the devices are [4, 1, 2, 3]
  %e0 = stablehlo.constant dense<[4]> : tensor<1xi32>
  %e1 = stablehlo.constant dense<[1]> : tensor<1xi32>
  %e2 = stablehlo.constant dense<[2]> : tensor<1xi32>
  %e3 = stablehlo.constant dense<[3]> : tensor<1xi32>

  "check.expect_eq"(%res#0, %e0) : (tensor<1xi32>, tensor<1xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<1xi32>, tensor<1xi32>) -> ()
  "check.expect_eq"(%res#2, %e2) : (tensor<1xi32>, tensor<1xi32>) -> ()
  "check.expect_eq"(%res#3, %e3) : (tensor<1xi32>, tensor<1xi32>) -> ()

  return
}

