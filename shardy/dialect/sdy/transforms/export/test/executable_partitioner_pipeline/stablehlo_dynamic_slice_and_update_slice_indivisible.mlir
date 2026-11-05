// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_x2 = <["x"=2]>

// The operand dimension 0 has size 5 (padded to 6 across 2 shards). A
// dynamic_update_slice of 1x2 at an out-of-bounds dynamic start index (5) must
// clamp to the unpadded bound (5 - 1 = 4) rather than the padded bound
// (6 - 1 = 5), which would write into the padding slot on shard 1 and be
// trimmed away.
func.func @parallel_dus_oob_dynamic_index(
  %arg0: tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>},
  %upd: tensor<1x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>},
  %idx: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh_x2, []>})
  -> (tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %sharded = sdy.reshard %arg0 <@mesh_x2, [{"x"}, {}]> : tensor<5x2xi32>
  %0 = stablehlo.dynamic_update_slice %sharded, %upd, %idx, %c0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>}
    : (tensor<5x2xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>) -> tensor<5x2xi32>
  %neg = stablehlo.negate %0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>}
    : tensor<5x2xi32>
  %1 = sdy.reshard %neg <@mesh_x2, [{}, {}]> : tensor<5x2xi32>
  return %1 : tensor<5x2xi32>
}

func.func @sequential_dus_oob_dynamic_index(
  %arg0: tensor<5x2xi32>, %upd: tensor<1x2xi32>, %idx: tensor<i32>) -> tensor<5x2xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %upd, %idx, %c0
    : (tensor<5x2xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>) -> tensor<5x2xi32>
  %neg = stablehlo.negate %0 : tensor<5x2xi32>
  return %neg : tensor<5x2xi32>
}

// Constant start index (4) on operand dimension 0 of size 5 with update of
// size 2x2 must clamp to 5 - 2 = 3 (updating rows 3 and 4 on shard 1) rather
// than 6 - 2 = 4 (updating row 4 and padding row 5).
func.func @parallel_dus_oob_constant_index(
  %arg0: tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>},
  %upd: tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>})
  -> (tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %sharded = sdy.reshard %arg0 <@mesh_x2, [{"x"}, {}]> : tensor<5x2xi32>
  %0 = stablehlo.dynamic_update_slice %sharded, %upd, %c4, %c0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>}
    : (tensor<5x2xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<5x2xi32>
  %neg = stablehlo.negate %0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}, {}]>]>}
    : tensor<5x2xi32>
  %1 = sdy.reshard %neg <@mesh_x2, [{}, {}]> : tensor<5x2xi32>
  return %1 : tensor<5x2xi32>
}

func.func @sequential_dus_oob_constant_index(
  %arg0: tensor<5x2xi32>, %upd: tensor<2x2xi32>) -> tensor<5x2xi32> {
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %upd, %c4, %c0
    : (tensor<5x2xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<5x2xi32>
  %neg = stablehlo.negate %0 : tensor<5x2xi32>
  return %neg : tensor<5x2xi32>
}

// A dynamic_slice that slices the full extent of an indivisible sharded
// dimension (dim 1 of size 5 sharded on "x"=2) must expand slice_sizes[1] to
// the padded size (6) during pad-for-divisibility.
func.func @parallel_ds_full_sharded_indivisible_dim(
  %arg0: tensor<2x5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>},
  %row: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh_x2, []>})
  -> (tensor<1x5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %sharded = sdy.reshard %arg0 <@mesh_x2, [{}, {"x"}]> : tensor<2x5xi32>
  %0 = stablehlo.dynamic_slice %sharded, %row, %c0, sizes = [1, 5]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{}, {"x"}]>]>}
    : (tensor<2x5xi32>, tensor<i32>, tensor<i32>) -> tensor<1x5xi32>
  %1 = sdy.reshard %0 <@mesh_x2, [{}, {}]> : tensor<1x5xi32>
  return %1 : tensor<1x5xi32>
}

func.func @sequential_ds_full_sharded_indivisible_dim(
  %arg0: tensor<2x5xi32>, %row: tensor<i32>) -> tensor<1x5xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_slice %arg0, %row, %c0, sizes = [1, 5]
    : (tensor<2x5xi32>, tensor<i32>, tensor<i32>) -> tensor<1x5xi32>
  return %0 : tensor<1x5xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]> : tensor<5x2xi32>

  // Test 1: dynamic out-of-bounds start_index = 5 with 1x2 update.
  %upd1 = stablehlo.constant dense<[[99, 98]]> : tensor<1x2xi32>
  %idx5 = stablehlo.constant dense<5> : tensor<i32>
  %seq1 = func.call @sequential_dus_oob_dynamic_index(%input, %upd1, %idx5)
    : (tensor<5x2xi32>, tensor<1x2xi32>, tensor<i32>) -> tensor<5x2xi32>
  %res1:2 = "interpreter.run_parallel"(%input, %upd1, %idx5, %input, %upd1, %idx5) {
    programs = [[@parallel_dus_oob_dynamic_index, @parallel_dus_oob_dynamic_index]]
  } : (tensor<5x2xi32>, tensor<1x2xi32>, tensor<i32>, tensor<5x2xi32>, tensor<1x2xi32>, tensor<i32>)
      -> (tensor<5x2xi32>, tensor<5x2xi32>)
  "check.expect_eq"(%res1#0, %seq1) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
  "check.expect_eq"(%res1#1, %seq1) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()

  // Test 2: constant out-of-bounds start_index = 4 with 2x2 update.
  %upd2 = stablehlo.constant dense<[[88, 87], [77, 76]]> : tensor<2x2xi32>
  %seq2 = func.call @sequential_dus_oob_constant_index(%input, %upd2)
    : (tensor<5x2xi32>, tensor<2x2xi32>) -> tensor<5x2xi32>
  %res2:2 = "interpreter.run_parallel"(%input, %upd2, %input, %upd2) {
    programs = [[@parallel_dus_oob_constant_index, @parallel_dus_oob_constant_index]]
  } : (tensor<5x2xi32>, tensor<2x2xi32>, tensor<5x2xi32>, tensor<2x2xi32>)
      -> (tensor<5x2xi32>, tensor<5x2xi32>)
  "check.expect_eq"(%res2#0, %seq2) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()

  // Test 3: dynamic_slice with full slice along indivisible sharded dim 1.
  %input3 = stablehlo.constant dense<[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]> : tensor<2x5xi32>
  %row1 = stablehlo.constant dense<1> : tensor<i32>
  %seq3 = func.call @sequential_ds_full_sharded_indivisible_dim(%input3, %row1)
    : (tensor<2x5xi32>, tensor<i32>) -> tensor<1x5xi32>
  %res3:2 = "interpreter.run_parallel"(%input3, %row1, %input3, %row1) {
    programs = [[@parallel_ds_full_sharded_indivisible_dim, @parallel_ds_full_sharded_indivisible_dim]]
  } : (tensor<2x5xi32>, tensor<i32>, tensor<2x5xi32>, tensor<i32>)
      -> (tensor<1x5xi32>, tensor<1x5xi32>)
  "check.expect_eq"(%res3#0, %seq3) : (tensor<1x5xi32>, tensor<1x5xi32>) -> ()
  "check.expect_eq"(%res3#1, %seq3) : (tensor<1x5xi32>, tensor<1x5xi32>) -> ()

  return
}
