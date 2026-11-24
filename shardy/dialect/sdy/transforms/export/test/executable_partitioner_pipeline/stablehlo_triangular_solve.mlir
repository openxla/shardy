// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Test 1: Matrix dimension sharded on a 1D mesh, forcing kNeedReplication
// resharding (all-gather inputs before triangular_solve, all-slice result).
func.func @parallel_triangular_solve_sharded_matrix_dim(
    %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// Test 2: 2D mesh with batch dims sharded across {"x"} and {"y"} and
// non-default attributes (left_side = false, lower = false, transpose_a = TRANSPOSE).
func.func @parallel_triangular_solve_2d_batch_right_upper_transpose(
    %arg0: tensor<2x2x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}, {}, {}]>},
    %arg1: tensor<2x2x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}, {}, {}]>})
    -> (tensor<2x2x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}, {}, {}]>}) {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = false,
    lower = false,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose TRANSPOSE>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}, {}, {}]>]>
  } : (tensor<2x2x3x3xf32>, tensor<2x2x2x3xf32>) -> tensor<2x2x2x3xf32>
  return %0 : tensor<2x2x2x3xf32>
}

// Test 3: Indivisible batch dim (3 on mesh_2) exercising insert-explicit-reshards
// and pad-for-divisibility with unit_diagonal = true (diagonal entries ignored).
func.func @parallel_triangular_solve_indivisible_batch(
    %arg0: tensor<3x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}]>},
    %arg1: tensor<3x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}]>})
    -> (tensor<3x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}, {}]>}) {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}, {}]>]>
  } : (tensor<3x2x2xf32>, tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>
}

//--- part2.mlir

func.func @main() {
  // === Test 1: Matrix dim sharded (kNeedReplication) ===
  %a1 = stablehlo.constant dense<[
    [ 1.0,  9.0, -3.0,  7.0],
    [ 2.0,  2.0, -5.0,  4.0],
    [-1.0,  1.0,  4.0, -2.0],
    [ 3.0, -2.0,  2.0,  2.0]
  ]> : tensor<4x4xf32>
  %b1 = stablehlo.constant dense<[
    [1.0, 2.0],
    [4.0, 0.0],
    [3.0, 5.0],
    [1.0, 7.0]
  ]> : tensor<4x2xf32>

  %seq1 = func.call @sequential_triangular_solve_sharded_matrix_dim(%a1, %b1)
    : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>

  %a1_s0 = "stablehlo.slice"(%a1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %a1_s1 = "stablehlo.slice"(%a1) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %b1_s0 = "stablehlo.slice"(%b1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %b1_s1 = "stablehlo.slice"(%b1) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars1:2 = "interpreter.run_parallel"(%a1_s0, %b1_s0, %a1_s1, %b1_s1) {
    programs = [[@parallel_triangular_solve_sharded_matrix_dim,
                 @parallel_triangular_solve_sharded_matrix_dim]]
  } : (tensor<2x4xf32>, tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<2x2xf32>)
  %par1 = "stablehlo.concatenate"(%pars1#0, %pars1#1) {dimension = 0 : i64}
    : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>

  check.expect_eq %seq1, %par1 : tensor<4x2xf32>

  // === Test 2: 2D mesh batch-sharded + left_side=false, lower=false, TRANSPOSE ===
  %a2 = stablehlo.constant dense<[
    [
      [[ 1.0,  2.0, -1.0], [ 9.0,  2.0,  3.0], [-8.0,  7.0,  4.0]],
      [[ 2.0, -1.0,  2.0], [ 5.0,  1.0,  1.0], [-4.0,  3.0,  2.0]]
    ],
    [
      [[ 4.0,  3.0, -2.0], [ 6.0,  2.0,  4.0], [-3.0,  2.0,  1.0]],
      [[ 2.0,  1.0,  3.0], [ 4.0,  4.0, -2.0], [-5.0,  1.0,  2.0]]
    ]
  ]> : tensor<2x2x3x3xf32>
  %b2 = stablehlo.constant dense<[
    [
      [[ 4.0,  2.0,  8.0], [ 8.0, -4.0,  4.0]],
      [[ 2.0,  6.0,  4.0], [-4.0,  2.0,  6.0]]
    ],
    [
      [[ 8.0,  4.0,  2.0], [ 4.0, -2.0,  3.0]],
      [[ 6.0,  8.0,  4.0], [-2.0,  4.0,  2.0]]
    ]
  ]> : tensor<2x2x2x3xf32>

  %seq2 = func.call @sequential_triangular_solve_2d_batch_right_upper_transpose(%a2, %b2)
    : (tensor<2x2x3x3xf32>, tensor<2x2x2x3xf32>) -> tensor<2x2x2x3xf32>

  %a2_00 = "stablehlo.slice"(%a2) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 1, 3, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x3x3xf32>) -> tensor<1x1x3x3xf32>
  %a2_01 = "stablehlo.slice"(%a2) {start_indices = array<i64: 0, 1, 0, 0>, limit_indices = array<i64: 1, 2, 3, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x3x3xf32>) -> tensor<1x1x3x3xf32>
  %a2_10 = "stablehlo.slice"(%a2) {start_indices = array<i64: 1, 0, 0, 0>, limit_indices = array<i64: 2, 1, 3, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x3x3xf32>) -> tensor<1x1x3x3xf32>
  %a2_11 = "stablehlo.slice"(%a2) {start_indices = array<i64: 1, 1, 0, 0>, limit_indices = array<i64: 2, 2, 3, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x3x3xf32>) -> tensor<1x1x3x3xf32>

  %b2_00 = "stablehlo.slice"(%b2) {start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: 1, 1, 2, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x2x3xf32>) -> tensor<1x1x2x3xf32>
  %b2_01 = "stablehlo.slice"(%b2) {start_indices = array<i64: 0, 1, 0, 0>, limit_indices = array<i64: 1, 2, 2, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x2x3xf32>) -> tensor<1x1x2x3xf32>
  %b2_10 = "stablehlo.slice"(%b2) {start_indices = array<i64: 1, 0, 0, 0>, limit_indices = array<i64: 2, 1, 2, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x2x3xf32>) -> tensor<1x1x2x3xf32>
  %b2_11 = "stablehlo.slice"(%b2) {start_indices = array<i64: 1, 1, 0, 0>, limit_indices = array<i64: 2, 2, 2, 3>, strides = array<i64: 1, 1, 1, 1>} : (tensor<2x2x2x3xf32>) -> tensor<1x1x2x3xf32>

  %pars2:4 = "interpreter.run_parallel"(
      %a2_00, %b2_00, %a2_01, %b2_01, %a2_10, %b2_10, %a2_11, %b2_11) {
    programs = [[@parallel_triangular_solve_2d_batch_right_upper_transpose,
                 @parallel_triangular_solve_2d_batch_right_upper_transpose,
                 @parallel_triangular_solve_2d_batch_right_upper_transpose,
                 @parallel_triangular_solve_2d_batch_right_upper_transpose]]
  } : (tensor<1x1x3x3xf32>, tensor<1x1x2x3xf32>,
       tensor<1x1x3x3xf32>, tensor<1x1x2x3xf32>,
       tensor<1x1x3x3xf32>, tensor<1x1x2x3xf32>,
       tensor<1x1x3x3xf32>, tensor<1x1x2x3xf32>)
    -> (tensor<1x1x2x3xf32>, tensor<1x1x2x3xf32>,
        tensor<1x1x2x3xf32>, tensor<1x1x2x3xf32>)

  %row0 = "stablehlo.concatenate"(%pars2#0, %pars2#1) {dimension = 1 : i64}
    : (tensor<1x1x2x3xf32>, tensor<1x1x2x3xf32>) -> tensor<1x2x2x3xf32>
  %row1 = "stablehlo.concatenate"(%pars2#2, %pars2#3) {dimension = 1 : i64}
    : (tensor<1x1x2x3xf32>, tensor<1x1x2x3xf32>) -> tensor<1x2x2x3xf32>
  %par2 = "stablehlo.concatenate"(%row0, %row1) {dimension = 0 : i64}
    : (tensor<1x2x2x3xf32>, tensor<1x2x2x3xf32>) -> tensor<2x2x2x3xf32>

  check.expect_eq %seq2, %par2 : tensor<2x2x2x3xf32>

  // === Test 3: Indivisible batch dim (3 on mesh_2) with unit_diagonal=true ===
  %a3 = stablehlo.constant dense<[
    [[ 5.0,  9.0], [ 2.0, -3.0]],
    [[ 7.0,  4.0], [-3.0,  8.0]],
    [[-2.0,  6.0], [ 4.0,  1.0]]
  ]> : tensor<3x2x2xf32>
  %b3 = stablehlo.constant dense<[
    [[ 1.0,  2.0,  3.0], [ 4.0,  5.0,  6.0]],
    [[-1.0,  3.0,  0.5], [ 2.0, -4.0,  1.5]],
    [[ 3.0, -2.0,  4.0], [ 1.0,  5.0, -3.0]]
  ]> : tensor<3x2x3xf32>

  %seq3 = func.call @sequential_triangular_solve_indivisible_batch(%a3, %b3)
    : (tensor<3x2x2xf32>, tensor<3x2x3xf32>) -> tensor<3x2x3xf32>

  %pars3:2 = "interpreter.run_parallel"(%a3, %b3, %a3, %b3) {
    programs = [[@parallel_triangular_solve_indivisible_batch,
                 @parallel_triangular_solve_indivisible_batch]]
  } : (tensor<3x2x2xf32>, tensor<3x2x3xf32>,
       tensor<3x2x2xf32>, tensor<3x2x3xf32>)
    -> (tensor<3x2x3xf32>, tensor<3x2x3xf32>)

  check.expect_eq %seq3, %pars3#0 : tensor<3x2x3xf32>
  check.expect_eq %seq3, %pars3#1 : tensor<3x2x3xf32>

  return
}
