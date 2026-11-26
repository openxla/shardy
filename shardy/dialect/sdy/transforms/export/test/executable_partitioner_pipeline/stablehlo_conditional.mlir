// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Test 1: stablehlo.if with cross-branch resharding and captured sharded values
// on a 1D mesh. True branch produces [{}, {"x"}] (all-to-all to [{"x"}, {}]),
// false branch produces [{}, {}] (all-slice to [{"x"}, {}]).
func.func @parallel_if_cross_branch_reshard(
    %pred: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh_2, []>},
    %x: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %y: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>})
    -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.if"(%pred) ({
    %neg = stablehlo.negate %y {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"x"}]>]>} : tensor<4x2xf32>
    stablehlo.return %neg : tensor<4x2xf32>
  }, {
    %add = stablehlo.add %x, %y {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}]>]>} : tensor<4x2xf32>
    stablehlo.return %add : tensor<4x2xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<i1>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

func.func @sequential_if_cross_branch_reshard(
    %pred: tensor<i1>, %x: tensor<4x2xf32>, %y: tensor<4x2xf32>)
    -> tensor<4x2xf32> {
  %0 = "stablehlo.if"(%pred) ({
    %neg = stablehlo.negate %y : tensor<4x2xf32>
    stablehlo.return %neg : tensor<4x2xf32>
  }, {
    %add = stablehlo.add %x, %y : tensor<4x2xf32>
    stablehlo.return %add : tensor<4x2xf32>
  }) : (tensor<i1>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// Test 2: stablehlo.if on a 2D mesh with multi-result branches, captured outer
// sharded tensors, and cross-branch resharding on both results.
func.func @parallel_if_2d_multi_result_captured(
    %pred: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh_2_2, []>},
    %a: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
    %b: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
        tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>}) {
  %0:2 = "stablehlo.if"(%pred) ({
    %r0 = stablehlo.add %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : tensor<4x4xf32>
    %r1 = stablehlo.subtract %b, %a {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {"x"}]>]>} : tensor<4x4xf32>
    stablehlo.return %r0, %r1 : tensor<4x4xf32>, tensor<4x4xf32>
  }, {
    %r0 = stablehlo.multiply %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {"x"}]>]>} : tensor<4x4xf32>
    %r1 = stablehlo.subtract %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : tensor<4x4xf32>
    stablehlo.return %r0, %r1 : tensor<4x4xf32>, tensor<4x4xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>, <@mesh_2_2, [{"y"}, {"x"}]>]>}
    : (tensor<i1>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

func.func @sequential_if_2d_multi_result_captured(
    %pred: tensor<i1>, %a: tensor<4x4xf32>, %b: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:2 = "stablehlo.if"(%pred) ({
    %r0 = stablehlo.add %a, %b : tensor<4x4xf32>
    %r1 = stablehlo.subtract %b, %a : tensor<4x4xf32>
    stablehlo.return %r0, %r1 : tensor<4x4xf32>, tensor<4x4xf32>
  }, {
    %r0 = stablehlo.multiply %a, %b : tensor<4x4xf32>
    %r1 = stablehlo.subtract %a, %b : tensor<4x4xf32>
    stablehlo.return %r0, %r1 : tensor<4x4xf32>, tensor<4x4xf32>
  }) : (tensor<i1>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// Test 3: stablehlo.if with indivisible sharded intermediate values (3x4 on
// mesh_2), exercising pad-for-divisibility on IfOp / padGenericOp.
func.func @parallel_if_indivisible(
    %pred: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh_2, []>},
    %a: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %b: tensor<3x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>})
    -> (tensor<3x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %sa = stablehlo.slice %a [0:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<4x4xf32>) -> tensor<3x4xf32>
  %sb = sdy.reshard %b <@mesh_2, [{"x"}, {}]> : tensor<3x4xf32>
  %0 = "stablehlo.if"(%pred) ({
    %sum = stablehlo.add %sa, %sb {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<3x4xf32>
    stablehlo.return %sum : tensor<3x4xf32>
  }, {
    %diff = stablehlo.subtract %sa, %sb {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<3x4xf32>
    stablehlo.return %diff : tensor<3x4xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<i1>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

func.func @sequential_if_indivisible(
    %pred: tensor<i1>, %a: tensor<4x4xf32>, %b: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  %sa = stablehlo.slice %a [0:3, 0:4] : (tensor<4x4xf32>) -> tensor<3x4xf32>
  %0 = "stablehlo.if"(%pred) ({
    %sum = stablehlo.add %sa, %b : tensor<3x4xf32>
    stablehlo.return %sum : tensor<3x4xf32>
  }, {
    %diff = stablehlo.subtract %sa, %b : tensor<3x4xf32>
    stablehlo.return %diff : tensor<3x4xf32>
  }) : (tensor<i1>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// Test 4: stablehlo.case with 3 branches, captured sharded values, multi-result
// branches, and cross-branch resharding on a 2D mesh.
func.func @parallel_case_multi_branch_reshard_2d(
    %idx: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh_2_2, []>},
    %a: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
    %b: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
        tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"y"}]>}) {
  %0:2 = "stablehlo.case"(%idx) ({
    %sum = stablehlo.add %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : tensor<4x4xf32>
    %neg = stablehlo.negate %a {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}, {"y"}]>]>} : tensor<4x4xf32>
    stablehlo.return %sum, %neg : tensor<4x4xf32>, tensor<4x4xf32>
  }, {
    %sub = stablehlo.subtract %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {"x"}]>]>} : tensor<4x4xf32>
    %mul = stablehlo.multiply %a, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {}]>]>} : tensor<4x4xf32>
    stablehlo.return %sub, %mul : tensor<4x4xf32>, tensor<4x4xf32>
  }, {
    %c2 = stablehlo.constant dense<2.0> : tensor<4x4xf32>
    %scaled = stablehlo.multiply %b, %c2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}, {}]>]>} : tensor<4x4xf32>
    stablehlo.return %scaled, %b : tensor<4x4xf32>, tensor<4x4xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>, <@mesh_2_2, [{}, {"y"}]>]>}
    : (tensor<i32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

func.func @sequential_case_multi_branch_reshard_2d(
    %idx: tensor<i32>, %a: tensor<4x4xf32>, %b: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:2 = "stablehlo.case"(%idx) ({
    %sum = stablehlo.add %a, %b : tensor<4x4xf32>
    %neg = stablehlo.negate %a : tensor<4x4xf32>
    stablehlo.return %sum, %neg : tensor<4x4xf32>, tensor<4x4xf32>
  }, {
    %sub = stablehlo.subtract %a, %b : tensor<4x4xf32>
    %mul = stablehlo.multiply %a, %b : tensor<4x4xf32>
    stablehlo.return %sub, %mul : tensor<4x4xf32>, tensor<4x4xf32>
  }, {
    %c2 = stablehlo.constant dense<2.0> : tensor<4x4xf32>
    %scaled = stablehlo.multiply %b, %c2 : tensor<4x4xf32>
    stablehlo.return %scaled, %b : tensor<4x4xf32>, tensor<4x4xf32>
  }) : (tensor<i32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// Test 5: stablehlo.case with 3 branches and indivisible sharded results
// (3x4 on mesh_2), exercising pad-for-divisibility on CaseOp.
func.func @parallel_case_indivisible(
    %idx: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh_2, []>},
    %a: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %b: tensor<3x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>})
    -> (tensor<3x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %sa = stablehlo.slice %a [0:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<4x4xf32>) -> tensor<3x4xf32>
  %sb = sdy.reshard %b <@mesh_2, [{"x"}, {}]> : tensor<3x4xf32>
  %0 = "stablehlo.case"(%idx) ({
    %sum = stablehlo.add %sa, %sb {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<3x4xf32>
    stablehlo.return %sum : tensor<3x4xf32>
  }, {
    %diff = stablehlo.subtract %sa, %sb {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}]>]>} : tensor<3x4xf32>
    stablehlo.return %diff : tensor<3x4xf32>
  }, {
    %prod = stablehlo.multiply %sa, %sb {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<3x4xf32>
    stablehlo.return %prod : tensor<3x4xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

func.func @sequential_case_indivisible(
    %idx: tensor<i32>, %a: tensor<4x4xf32>, %b: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  %sa = stablehlo.slice %a [0:3, 0:4] : (tensor<4x4xf32>) -> tensor<3x4xf32>
  %0 = "stablehlo.case"(%idx) ({
    %sum = stablehlo.add %sa, %b : tensor<3x4xf32>
    stablehlo.return %sum : tensor<3x4xf32>
  }, {
    %diff = stablehlo.subtract %sa, %b : tensor<3x4xf32>
    stablehlo.return %diff : tensor<3x4xf32>
  }, {
    %prod = stablehlo.multiply %sa, %b : tensor<3x4xf32>
    stablehlo.return %prod : tensor<3x4xf32>
  }) : (tensor<i32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

//--- part2.mlir

func.func @main() {
  %true = stablehlo.constant dense<true> : tensor<i1>
  %false = stablehlo.constant dense<false> : tensor<i1>

  // === Test 1: stablehlo.if cross-branch reshard (true and false) ===
  %x1 = stablehlo.constant dense<[
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
  ]> : tensor<4x2xf32>
  %y1 = stablehlo.constant dense<[
    [10.0, 20.0],
    [30.0, 40.0],
    [50.0, 60.0],
    [70.0, 80.0]
  ]> : tensor<4x2xf32>

  %x1_s0 = stablehlo.slice %x1 [0:2, 0:2] : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %x1_s1 = stablehlo.slice %x1 [2:4, 0:2] : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %y1_s0 = stablehlo.slice %y1 [0:4, 0:1] : (tensor<4x2xf32>) -> tensor<4x1xf32>
  %y1_s1 = stablehlo.slice %y1 [0:4, 1:2] : (tensor<4x2xf32>) -> tensor<4x1xf32>

  %seq1_t = func.call @sequential_if_cross_branch_reshard(%true, %x1, %y1)
    : (tensor<i1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %pars1_t:2 = "interpreter.run_parallel"(%true, %x1_s0, %y1_s0, %true, %x1_s1, %y1_s1) {
    programs = [[@parallel_if_cross_branch_reshard, @parallel_if_cross_branch_reshard]]
  } : (tensor<i1>, tensor<2x2xf32>, tensor<4x1xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<4x1xf32>)
    -> (tensor<2x2xf32>, tensor<2x2xf32>)
  %par1_t = "stablehlo.concatenate"(%pars1_t#0, %pars1_t#1) {dimension = 0 : i64}
    : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  check.expect_eq %seq1_t, %par1_t : tensor<4x2xf32>

  %seq1_f = func.call @sequential_if_cross_branch_reshard(%false, %x1, %y1)
    : (tensor<i1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %pars1_f:2 = "interpreter.run_parallel"(%false, %x1_s0, %y1_s0, %false, %x1_s1, %y1_s1) {
    programs = [[@parallel_if_cross_branch_reshard, @parallel_if_cross_branch_reshard]]
  } : (tensor<i1>, tensor<2x2xf32>, tensor<4x1xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<4x1xf32>)
    -> (tensor<2x2xf32>, tensor<2x2xf32>)
  %par1_f = "stablehlo.concatenate"(%pars1_f#0, %pars1_f#1) {dimension = 0 : i64}
    : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  check.expect_eq %seq1_f, %par1_f : tensor<4x2xf32>

  // === Shared 2D inputs for Tests 2 & 4 ===
  %a2 = stablehlo.constant dense<[
    [ 1.0,  2.0,  3.0,  4.0],
    [ 5.0,  6.0,  7.0,  8.0],
    [ 9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
  ]> : tensor<4x4xf32>
  %b2 = stablehlo.constant dense<[
    [ 2.0,  1.0,  4.0,  3.0],
    [-1.0,  3.0, -2.0,  5.0],
    [ 6.0, -4.0,  8.0, -2.0],
    [ 3.0,  7.0, -5.0,  9.0]
  ]> : tensor<4x4xf32>

  // %a2 sharded [{"x"}, {"y"}] on 2x2: (0,0), (0,1), (1,0), (1,1).
  %a2_00 = stablehlo.slice %a2 [0:2, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %a2_01 = stablehlo.slice %a2 [0:2, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %a2_10 = stablehlo.slice %a2 [2:4, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %a2_11 = stablehlo.slice %a2 [2:4, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>

  // %b2 sharded [{"y"}, {"x"}] on 2x2: device (x, y) gets slice (y, x).
  %b2_00 = stablehlo.slice %b2 [0:2, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %b2_01 = stablehlo.slice %b2 [2:4, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %b2_10 = stablehlo.slice %b2 [0:2, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %b2_11 = stablehlo.slice %b2 [2:4, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>

  // === Test 2: stablehlo.if 2D multi-result + captured values (true and false) ===
  %seq2_t:2 = func.call @sequential_if_2d_multi_result_captured(%true, %a2, %b2)
    : (tensor<i1>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %pars2_t:8 = "interpreter.run_parallel"(
      %true, %a2_00, %b2_00, %true, %a2_01, %b2_01,
      %true, %a2_10, %b2_10, %true, %a2_11, %b2_11) {
    programs = [[@parallel_if_2d_multi_result_captured,
                 @parallel_if_2d_multi_result_captured,
                 @parallel_if_2d_multi_result_captured,
                 @parallel_if_2d_multi_result_captured]]
  } : (tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  %p2t_0_r0 = "stablehlo.concatenate"(%pars2_t#0, %pars2_t#2) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p2t_0_r1 = "stablehlo.concatenate"(%pars2_t#4, %pars2_t#6) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par2_t0 = "stablehlo.concatenate"(%p2t_0_r0, %p2t_0_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %p2t_1_r0 = "stablehlo.concatenate"(%pars2_t#1, %pars2_t#5) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p2t_1_r1 = "stablehlo.concatenate"(%pars2_t#3, %pars2_t#7) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par2_t1 = "stablehlo.concatenate"(%p2t_1_r0, %p2t_1_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  check.expect_eq %seq2_t#0, %par2_t0 : tensor<4x4xf32>
  check.expect_eq %seq2_t#1, %par2_t1 : tensor<4x4xf32>

  %seq2_f:2 = func.call @sequential_if_2d_multi_result_captured(%false, %a2, %b2)
    : (tensor<i1>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %pars2_f:8 = "interpreter.run_parallel"(
      %false, %a2_00, %b2_00, %false, %a2_01, %b2_01,
      %false, %a2_10, %b2_10, %false, %a2_11, %b2_11) {
    programs = [[@parallel_if_2d_multi_result_captured,
                 @parallel_if_2d_multi_result_captured,
                 @parallel_if_2d_multi_result_captured,
                 @parallel_if_2d_multi_result_captured]]
  } : (tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  %p2f_0_r0 = "stablehlo.concatenate"(%pars2_f#0, %pars2_f#2) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p2f_0_r1 = "stablehlo.concatenate"(%pars2_f#4, %pars2_f#6) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par2_f0 = "stablehlo.concatenate"(%p2f_0_r0, %p2f_0_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %p2f_1_r0 = "stablehlo.concatenate"(%pars2_f#1, %pars2_f#5) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p2f_1_r1 = "stablehlo.concatenate"(%pars2_f#3, %pars2_f#7) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par2_f1 = "stablehlo.concatenate"(%p2f_1_r0, %p2f_1_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  check.expect_eq %seq2_f#0, %par2_f0 : tensor<4x4xf32>
  check.expect_eq %seq2_f#1, %par2_f1 : tensor<4x4xf32>

  // === Shared indivisible inputs for Tests 3 & 5 ===
  %a_ind_s0 = stablehlo.slice %a2 [0:2, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %a_ind_s1 = stablehlo.slice %a2 [2:4, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %b_ind = stablehlo.constant dense<[
    [ 0.5, -1.5,  2.5, -3.5],
    [ 4.5, -5.5,  6.5, -7.5],
    [ 8.5, -9.5, 10.5, -11.5]
  ]> : tensor<3x4xf32>

  // === Test 3: stablehlo.if indivisible (true and false) ===
  %seq3_t = func.call @sequential_if_indivisible(%true, %a2, %b_ind)
    : (tensor<i1>, tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %pars3_t:2 = "interpreter.run_parallel"(%true, %a_ind_s0, %b_ind, %true, %a_ind_s1, %b_ind) {
    programs = [[@parallel_if_indivisible, @parallel_if_indivisible]]
  } : (tensor<i1>, tensor<2x4xf32>, tensor<3x4xf32>,
       tensor<i1>, tensor<2x4xf32>, tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xf32>)
  check.expect_eq %seq3_t, %pars3_t#0 : tensor<3x4xf32>
  check.expect_eq %seq3_t, %pars3_t#1 : tensor<3x4xf32>

  %seq3_f = func.call @sequential_if_indivisible(%false, %a2, %b_ind)
    : (tensor<i1>, tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %pars3_f:2 = "interpreter.run_parallel"(%false, %a_ind_s0, %b_ind, %false, %a_ind_s1, %b_ind) {
    programs = [[@parallel_if_indivisible, @parallel_if_indivisible]]
  } : (tensor<i1>, tensor<2x4xf32>, tensor<3x4xf32>,
       tensor<i1>, tensor<2x4xf32>, tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xf32>)
  check.expect_eq %seq3_f, %pars3_f#0 : tensor<3x4xf32>
  check.expect_eq %seq3_f, %pars3_f#1 : tensor<3x4xf32>

  // === Test 4: stablehlo.case 3 branches on 2D mesh (idx = 0, 1, 2) ===
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>

  // Branch 0
  %seq4_0:2 = func.call @sequential_case_multi_branch_reshard_2d(%c0, %a2, %b2)
    : (tensor<i32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %pars4_0:8 = "interpreter.run_parallel"(
      %c0, %a2_00, %b2_00, %c0, %a2_01, %b2_01,
      %c0, %a2_10, %b2_10, %c0, %a2_11, %b2_11) {
    programs = [[@parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d]]
  } : (tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>,
        tensor<2x2xf32>, tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>)
  %p4_0_r0 = "stablehlo.concatenate"(%pars4_0#0, %pars4_0#2) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p4_0_r1 = "stablehlo.concatenate"(%pars4_0#4, %pars4_0#6) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par4_0_a = "stablehlo.concatenate"(%p4_0_r0, %p4_0_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %par4_0_b0 = "stablehlo.concatenate"(%pars4_0#1, %pars4_0#3) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
  %par4_0_b1 = "stablehlo.concatenate"(%pars4_0#5, %pars4_0#7) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
  check.expect_eq %seq4_0#0, %par4_0_a : tensor<4x4xf32>
  check.expect_eq %seq4_0#1, %par4_0_b0 : tensor<4x4xf32>
  check.expect_eq %seq4_0#1, %par4_0_b1 : tensor<4x4xf32>

  // Branch 1
  %seq4_1:2 = func.call @sequential_case_multi_branch_reshard_2d(%c1, %a2, %b2)
    : (tensor<i32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %pars4_1:8 = "interpreter.run_parallel"(
      %c1, %a2_00, %b2_00, %c1, %a2_01, %b2_01,
      %c1, %a2_10, %b2_10, %c1, %a2_11, %b2_11) {
    programs = [[@parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d]]
  } : (tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>,
        tensor<2x2xf32>, tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>)
  %p4_1_r0 = "stablehlo.concatenate"(%pars4_1#0, %pars4_1#2) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p4_1_r1 = "stablehlo.concatenate"(%pars4_1#4, %pars4_1#6) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par4_1_a = "stablehlo.concatenate"(%p4_1_r0, %p4_1_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %par4_1_b0 = "stablehlo.concatenate"(%pars4_1#1, %pars4_1#3) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
  %par4_1_b1 = "stablehlo.concatenate"(%pars4_1#5, %pars4_1#7) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
  check.expect_eq %seq4_1#0, %par4_1_a : tensor<4x4xf32>
  check.expect_eq %seq4_1#1, %par4_1_b0 : tensor<4x4xf32>
  check.expect_eq %seq4_1#1, %par4_1_b1 : tensor<4x4xf32>

  // Branch 2
  %seq4_2:2 = func.call @sequential_case_multi_branch_reshard_2d(%c2, %a2, %b2)
    : (tensor<i32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %pars4_2:8 = "interpreter.run_parallel"(
      %c2, %a2_00, %b2_00, %c2, %a2_01, %b2_01,
      %c2, %a2_10, %b2_10, %c2, %a2_11, %b2_11) {
    programs = [[@parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d,
                 @parallel_case_multi_branch_reshard_2d]]
  } : (tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>,
       tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>,
        tensor<2x2xf32>, tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>)
  %p4_2_r0 = "stablehlo.concatenate"(%pars4_2#0, %pars4_2#2) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %p4_2_r1 = "stablehlo.concatenate"(%pars4_2#4, %pars4_2#6) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %par4_2_a = "stablehlo.concatenate"(%p4_2_r0, %p4_2_r1) {dimension = 0 : i64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %par4_2_b0 = "stablehlo.concatenate"(%pars4_2#1, %pars4_2#3) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
  %par4_2_b1 = "stablehlo.concatenate"(%pars4_2#5, %pars4_2#7) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
  check.expect_eq %seq4_2#0, %par4_2_a : tensor<4x4xf32>
  check.expect_eq %seq4_2#1, %par4_2_b0 : tensor<4x4xf32>
  check.expect_eq %seq4_2#1, %par4_2_b1 : tensor<4x4xf32>

  // === Test 5: stablehlo.case indivisible (idx = 0, 1, 2) ===
  %seq5_0 = func.call @sequential_case_indivisible(%c0, %a2, %b_ind)
    : (tensor<i32>, tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %pars5_0:2 = "interpreter.run_parallel"(%c0, %a_ind_s0, %b_ind, %c0, %a_ind_s1, %b_ind) {
    programs = [[@parallel_case_indivisible, @parallel_case_indivisible]]
  } : (tensor<i32>, tensor<2x4xf32>, tensor<3x4xf32>,
       tensor<i32>, tensor<2x4xf32>, tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xf32>)
  check.expect_eq %seq5_0, %pars5_0#0 : tensor<3x4xf32>
  check.expect_eq %seq5_0, %pars5_0#1 : tensor<3x4xf32>

  %seq5_1 = func.call @sequential_case_indivisible(%c1, %a2, %b_ind)
    : (tensor<i32>, tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %pars5_1:2 = "interpreter.run_parallel"(%c1, %a_ind_s0, %b_ind, %c1, %a_ind_s1, %b_ind) {
    programs = [[@parallel_case_indivisible, @parallel_case_indivisible]]
  } : (tensor<i32>, tensor<2x4xf32>, tensor<3x4xf32>,
       tensor<i32>, tensor<2x4xf32>, tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xf32>)
  check.expect_eq %seq5_1, %pars5_1#0 : tensor<3x4xf32>
  check.expect_eq %seq5_1, %pars5_1#1 : tensor<3x4xf32>

  %seq5_2 = func.call @sequential_case_indivisible(%c2, %a2, %b_ind)
    : (tensor<i32>, tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %pars5_2:2 = "interpreter.run_parallel"(%c2, %a_ind_s0, %b_ind, %c2, %a_ind_s1, %b_ind) {
    programs = [[@parallel_case_indivisible, @parallel_case_indivisible]]
  } : (tensor<i32>, tensor<2x4xf32>, tensor<3x4xf32>,
       tensor<i32>, tensor<2x4xf32>, tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xf32>)
  check.expect_eq %seq5_2, %pars5_2#0 : tensor<3x4xf32>
  check.expect_eq %seq5_2, %pars5_2#1 : tensor<3x4xf32>

  return
}
