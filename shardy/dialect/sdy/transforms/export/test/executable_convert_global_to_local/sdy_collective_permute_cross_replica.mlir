// RUN: split-file %s %t
// RUN: sdy_opt %t/part1.mlir --sdy-convert-global-to-local="replica-count=4 partition-count=1" \
// RUN:         --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_processed.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir

sdy.mesh @mesh_a = <["x"=4]>
sdy.mesh @mesh_b = <["x"=4], device_ids=[1, 2, 3, 0]>

// This function performs a collective permute across 4 replicas.
func.func @collective_permute(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_a, [{"x"}]>})
  -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_b, [{"x"}]>}) {
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

  // Run the 4 replicas in parallel (4 replicas, 1 partition each).
  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@collective_permute],
                [@collective_permute],
                [@collective_permute],
                [@collective_permute]]
  } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) ->
      (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)

  // Verify the cyclic shift. The new data in the devices are [4, 1, 2, 3].
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
