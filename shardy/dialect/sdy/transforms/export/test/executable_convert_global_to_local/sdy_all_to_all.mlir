// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

func.func @per_dim_all_to_all(
  %arg0: tensor<2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}, {"y"}, {}]>})
  -> (tensor<2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"x"}, {}, {"y"}]>}) {
  %0 = sdy.all_to_all [{"x"}: 0->1] %arg0
       out_sharding=<@mesh_2_2, [{}, {"x"}, {"y"}, {}]> : tensor<2x2x2x2xi32>
  %1 = sdy.all_to_all [{"y"}: 2->3] %0
       out_sharding=<@mesh_2_2, [{}, {"x"}, {}, {"y"}]> : tensor<2x2x2x2xi32>
  return %1 : tensor<2x2x2x2xi32>
}

func.func @combined_all_to_all(
  %arg0: tensor<2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}, {"y"}, {}]>})
  -> (tensor<2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"x"}, {}, {"y"}]>}) {
  %0 = sdy.all_to_all [{"x"}: 0->1, {"y"}: 2->3] %arg0
       out_sharding=<@mesh_2_2, [{}, {"x"}, {}, {"y"}]> : tensor<2x2x2x2xi32>
  return %0 : tensor<2x2x2x2xi32>
}

func.func @per_dim_all_to_all_non_zero_min_dim(
  %arg0: tensor<2x2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"x"}, {}, {"y"}, {}]>})
  -> (tensor<2x2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}, {"x"}, {}, {"y"}]>}) {
  %0 = sdy.all_to_all [{"x"}: 1->2] %arg0
       out_sharding=<@mesh_2_2, [{}, {}, {"x"}, {"y"}, {}]> : tensor<2x2x2x2x2xi32>
  %1 = sdy.all_to_all [{"y"}: 3->4] %0
       out_sharding=<@mesh_2_2, [{}, {}, {"x"}, {}, {"y"}]> : tensor<2x2x2x2x2xi32>
  return %1 : tensor<2x2x2x2x2xi32>
}

func.func @combined_all_to_all_non_zero_min_dim(
  %arg0: tensor<2x2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"x"}, {}, {"y"}, {}]>})
  -> (tensor<2x2x2x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}, {"x"}, {}, {"y"}]>}) {
  %0 = sdy.all_to_all [{"x"}: 1->2, {"y"}: 3->4] %arg0
       out_sharding=<@mesh_2_2, [{}, {}, {"x"}, {}, {"y"}]> : tensor<2x2x2x2x2xi32>
  return %0 : tensor<2x2x2x2x2xi32>
}

//--- part2.mlir

func.func @main() {
  // Global Input (2x2x2x2): Values 1 to 16.
  %cst = stablehlo.constant dense<[
    [[ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ]],
    [[ [9, 10], [11, 12] ], [ [13, 14], [15, 16] ]]
  ]> : tensor<2x2x2x2xi32>

  // Prepare Shards for Input Sharding [{"x"}, {}, {"y"}, {}]:
  // Each device owns a 1x2x1x2 physical quadrant.
  %s0 = "stablehlo.slice"(%cst) {start_indices=array<i64: 0,0,0,0>, limit_indices=array<i64: 1,2,1,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x2x2x2xi32>) -> tensor<1x2x1x2xi32>
  %s1 = "stablehlo.slice"(%cst) {start_indices=array<i64: 0,0,1,0>, limit_indices=array<i64: 1,2,2,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x2x2x2xi32>) -> tensor<1x2x1x2xi32>
  %s2 = "stablehlo.slice"(%cst) {start_indices=array<i64: 1,0,0,0>, limit_indices=array<i64: 2,2,1,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x2x2x2xi32>) -> tensor<1x2x1x2xi32>
  %s3 = "stablehlo.slice"(%cst) {start_indices=array<i64: 1,0,1,0>, limit_indices=array<i64: 2,2,2,2>, strides=array<i64: 1,1,1,1>} : (tensor<2x2x2x2xi32>) -> tensor<1x2x1x2xi32>

  %res_comb:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@combined_all_to_all, @combined_all_to_all, @combined_all_to_all, @combined_all_to_all]]
  } : (tensor<1x2x1x2xi32>, tensor<1x2x1x2xi32>, tensor<1x2x1x2xi32>, tensor<1x2x1x2xi32>) ->
      (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>)

  %res_seq:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@per_dim_all_to_all, @per_dim_all_to_all, @per_dim_all_to_all, @per_dim_all_to_all]]
  } : (tensor<1x2x1x2xi32>, tensor<1x2x1x2xi32>, tensor<1x2x1x2xi32>, tensor<1x2x1x2xi32>) ->
      (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>)

  "check.expect_eq"(%res_seq#0, %res_comb#0) : (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>) -> ()
  "check.expect_eq"(%res_seq#1, %res_comb#1) : (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>) -> ()
  "check.expect_eq"(%res_seq#2, %res_comb#2) : (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>) -> ()
  "check.expect_eq"(%res_seq#3, %res_comb#3) : (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>) -> ()

  // Re-assemble Global Result from Shards of [{}, {"x"}, {}, {"y"}] by
  // concatinating y-shards along Dim 3, then x-shards along Dim 1.
  %r0 = "stablehlo.concatenate"(%res_comb#0, %res_comb#1) {dimension = 3 : i64} : (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>) -> tensor<2x1x2x2xi32>
  %r1 = "stablehlo.concatenate"(%res_comb#2, %res_comb#3) {dimension = 3 : i64} : (tensor<2x1x2x1xi32>, tensor<2x1x2x1xi32>) -> tensor<2x1x2x2xi32>
  %total = "stablehlo.concatenate"(%r0, %r1) {dimension = 1 : i64} : (tensor<2x1x2x2xi32>, tensor<2x1x2x2xi32>) -> tensor<2x2x2x2xi32>
  "check.expect_eq"(%total, %cst) : (tensor<2x2x2x2xi32>, tensor<2x2x2x2xi32>) -> ()

  // Test all-to-all on non-zero min dim (dims 1->2 and 3->4, leaving dim 0 untouched).
  %iota = stablehlo.iota dim = 0 : tensor<32xi32>
  %cst_5d = stablehlo.reshape %iota : (tensor<32xi32>) -> tensor<2x2x2x2x2xi32>
  %s0_5d = stablehlo.slice %cst_5d [0:2, 0:1, 0:2, 0:1, 0:2] : (tensor<2x2x2x2x2xi32>) -> tensor<2x1x2x1x2xi32>
  %s1_5d = stablehlo.slice %cst_5d [0:2, 0:1, 0:2, 1:2, 0:2] : (tensor<2x2x2x2x2xi32>) -> tensor<2x1x2x1x2xi32>
  %s2_5d = stablehlo.slice %cst_5d [0:2, 1:2, 0:2, 0:1, 0:2] : (tensor<2x2x2x2x2xi32>) -> tensor<2x1x2x1x2xi32>
  %s3_5d = stablehlo.slice %cst_5d [0:2, 1:2, 0:2, 1:2, 0:2] : (tensor<2x2x2x2x2xi32>) -> tensor<2x1x2x1x2xi32>

  %res_comb_5d:4 = "interpreter.run_parallel"(%s0_5d, %s1_5d, %s2_5d, %s3_5d) {
    programs = [[@combined_all_to_all_non_zero_min_dim, @combined_all_to_all_non_zero_min_dim, @combined_all_to_all_non_zero_min_dim, @combined_all_to_all_non_zero_min_dim]]
  } : (tensor<2x1x2x1x2xi32>, tensor<2x1x2x1x2xi32>, tensor<2x1x2x1x2xi32>, tensor<2x1x2x1x2xi32>) ->
      (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>)

  %res_seq_5d:4 = "interpreter.run_parallel"(%s0_5d, %s1_5d, %s2_5d, %s3_5d) {
    programs = [[@per_dim_all_to_all_non_zero_min_dim, @per_dim_all_to_all_non_zero_min_dim, @per_dim_all_to_all_non_zero_min_dim, @per_dim_all_to_all_non_zero_min_dim]]
  } : (tensor<2x1x2x1x2xi32>, tensor<2x1x2x1x2xi32>, tensor<2x1x2x1x2xi32>, tensor<2x1x2x1x2xi32>) ->
      (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>)

  "check.expect_eq"(%res_seq_5d#0, %res_comb_5d#0) : (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>) -> ()
  "check.expect_eq"(%res_seq_5d#1, %res_comb_5d#1) : (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>) -> ()
  "check.expect_eq"(%res_seq_5d#2, %res_comb_5d#2) : (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>) -> ()
  "check.expect_eq"(%res_seq_5d#3, %res_comb_5d#3) : (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>) -> ()

  %r0_5d = "stablehlo.concatenate"(%res_comb_5d#0, %res_comb_5d#1) {dimension = 4 : i64} : (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>) -> tensor<2x2x1x2x2xi32>
  %r1_5d = "stablehlo.concatenate"(%res_comb_5d#2, %res_comb_5d#3) {dimension = 4 : i64} : (tensor<2x2x1x2x1xi32>, tensor<2x2x1x2x1xi32>) -> tensor<2x2x1x2x2xi32>
  %total_5d = "stablehlo.concatenate"(%r0_5d, %r1_5d) {dimension = 2 : i64} : (tensor<2x2x1x2x2xi32>, tensor<2x2x1x2x2xi32>) -> tensor<2x2x2x2x2xi32>
  "check.expect_eq"(%total_5d, %cst_5d) : (tensor<2x2x2x2x2xi32>, tensor<2x2x2x2x2xi32>) -> ()

  return
}
