// RUN: split-file %s %t

// Convert the routine using per-dim-all-gather=false.
// RUN: sdy_opt %t/part1.mlir --sdy-convert-global-to-local="per-dim-all-gather=false" \
// RUN:         --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
// RUN: sed 's/all_gather_test/combined_gather/g' > %t/combined_gather.mlir

// Convert the routine using per-dim-all-gather=true.
// RUN: sdy_opt %t/part1.mlir --sdy-convert-global-to-local="per-dim-all-gather=true" \
// RUN:         --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
// RUN: sed 's/all_gather_test/per_dim_gather/g' > %t/per_dim_gather.mlir

// Assemble the final module by unwrapping the module bodies then concatenating.
// RUN: sed '1d; /^}/,$d' %t/combined_gather.mlir > %t/combined.mlir
// RUN: sed '1d; /^}/,$d' %t/per_dim_gather.mlir >> %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir

// Execute and verify results.
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// This function gathers a 4x4 tensor sharded 2x2 using i32 data.
func.func @all_gather_test(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}
  ) -> tensor<4x4xi32> {
  %0 = sdy.all_gather [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  // Define a global 4x4 integer tensor.
  %cst = stablehlo.constant dense<[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<4x4xi32>

  // Prepare Shards (x=2, y=2):
  %s0 = "stablehlo.slice"(%cst) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %s1 = "stablehlo.slice"(%cst) {start_indices=array<i64: 0,2>, limit_indices=array<i64: 2,4>, strides=array<i64: 1,1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %s2 = "stablehlo.slice"(%cst) {start_indices=array<i64: 2,0>, limit_indices=array<i64: 4,2>, strides=array<i64: 1,1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %s3 = "stablehlo.slice"(%cst) {start_indices=array<i64: 2,2>, limit_indices=array<i64: 4,4>, strides=array<i64: 1,1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>

  // Run the Combined version.
  %res_comb:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@combined_gather, @combined_gather, @combined_gather, @combined_gather]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) ->
      (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)

  // Run the Per-Dim version.
  %res_pdim:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@per_dim_gather, @per_dim_gather, @per_dim_gather, @per_dim_gather]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) ->
      (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)

  // Verify consistency and correctness against original constant.
  "check.expect_eq"(%res_comb#0, %res_pdim#0) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_comb#0, %cst) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()

  return
}
