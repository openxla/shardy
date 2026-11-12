// RUN: %S/run_sdy_interpreter_test.sh %s %t
// RUN: FileCheck %s < %t/part1_processed.mlir

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Three contracting-dim sharded dot_generals combined via add and subtract:
//   (dot(a0, b0) + dot(a1, b1)) - dot(a2, b2)
// OptimizeCollectivesPass commutes stablehlo.add and stablehlo.subtract before
// sdy.all_reduce {"x"}, collapsing three all-reduces into one.
// CHECK-LABEL: func.func @parallel_dot_add_and_subtract
// CHECK-SAME:    (%[[A0:.*]]: tensor<2x2xi32>, %[[B0:.*]]: tensor<2x2xi32>, %[[A1:.*]]: tensor<2x2xi32>, %[[B1:.*]]: tensor<2x2xi32>, %[[A2:.*]]: tensor<2x2xi32>, %[[B2:.*]]: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK-NEXT:    %[[D0:.*]] = stablehlo.dot_general %[[A0]], %[[B0]], contracting_dims = [1] x [0] : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK-NEXT:    %[[D1:.*]] = stablehlo.dot_general %[[A1]], %[[B1]], contracting_dims = [1] x [0] : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK-NEXT:    %[[D2:.*]] = stablehlo.dot_general %[[A2]], %[[B2]], contracting_dims = [1] x [0] : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK-NEXT:    %[[SUM:.*]] = stablehlo.add %[[D0]], %[[D1]] : tensor<2x2xi32>
// CHECK-NEXT:    %[[DIFF:.*]] = stablehlo.subtract %[[SUM]], %[[D2]] : tensor<2x2xi32>
// CHECK-NEXT:    %[[AR:.*]] = "stablehlo.all_reduce"(%[[DIFF]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes=[<name = "x", size = 2>, <name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> ({
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: tensor<i32>, %[[RHS:.*]]: tensor<i32>):
// CHECK-NEXT:      %[[RED:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[RED]] : tensor<i32>
// CHECK-NEXT:    }) : (tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK-NEXT:    return %[[AR]] : tensor<2x2xi32>
// CHECK-NEXT:  }
func.func @parallel_dot_add_and_subtract(
  %a0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>},
  %b0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>},
  %a1: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>},
  %b1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>},
  %a2: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>},
  %b2: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>})
  -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {}]>}) {
  %d0 = stablehlo.dot_general %a0, %b0, contracting_dims = [1] x [0] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>
  } : (tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
  %d1 = stablehlo.dot_general %a1, %b1, contracting_dims = [1] x [0] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>
  } : (tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
  %d2 = stablehlo.dot_general %a2, %b2, contracting_dims = [1] x [0] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>
  } : (tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
  %sum = stablehlo.add %d0, %d1 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>
  } : tensor<4x2xi32>
  %diff = stablehlo.subtract %sum, %d2 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>
  } : tensor<4x2xi32>
  return %diff : tensor<4x2xi32>
}

// Two sharded max-reductions combined via stablehlo.maximum:
// OptimizeCollectivesPass commutes stablehlo.maximum before sdy.all_reduce
// (with unreduced=max{"x"}).
// CHECK-LABEL: func.func @parallel_reduce_max
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<2x2xi32>, %[[ARG1:.*]]: tensor<2x2xi32>) -> tensor<2xi32> {
// CHECK-NEXT:    %[[INIT:.*]] = stablehlo.constant dense<-1000> : tensor<i32>
// CHECK-NEXT:    %[[R0:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xi32>, tensor<i32>) -> tensor<2xi32>
// CHECK-NEXT:    %[[R1:.*]] = stablehlo.reduce(%[[ARG1]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xi32>, tensor<i32>) -> tensor<2xi32>
// CHECK-NEXT:    %[[MAX:.*]] = stablehlo.maximum %[[R0]], %[[R1]] : tensor<2xi32>
// CHECK-NEXT:    %[[AR_MAX:.*]] = "stablehlo.all_reduce"(%[[MAX]]) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes=[<name = "x", size = 2>, <name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> ({
// CHECK-NEXT:    ^bb0(%[[LHS_MAX:.*]]: tensor<i32>, %[[RHS_MAX:.*]]: tensor<i32>):
// CHECK-NEXT:      %[[RED_MAX:.*]] = stablehlo.maximum %[[LHS_MAX]], %[[RHS_MAX]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[RED_MAX]] : tensor<i32>
// CHECK-NEXT:    }) : (tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:    return %[[AR_MAX]] : tensor<2xi32>
// CHECK-NEXT:  }
func.func @parallel_reduce_max(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>},
  %arg1: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>})
  -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}]>}) {
  %init = stablehlo.constant dense<-1000> : tensor<i32>
  %r0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>
  } : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<i32>
      stablehlo.return %m : tensor<i32>
    }
  %r1 = stablehlo.reduce(%arg1 init: %init) across dimensions = [1] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>
  } : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<i32>
      stablehlo.return %m : tensor<i32>
    }
  %out = stablehlo.maximum %r0, %r1 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>
  } : tensor<4xi32>
  return %out : tensor<4xi32>
}

// Two sharded min-reductions combined via stablehlo.minimum:
// OptimizeCollectivesPass commutes stablehlo.minimum before sdy.all_reduce
// (with unreduced=min{"x"}).
// CHECK-LABEL: func.func @parallel_reduce_min
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<2x2xi32>, %[[ARG1:.*]]: tensor<2x2xi32>) -> tensor<2xi32> {
// CHECK-NEXT:    %[[INIT:.*]] = stablehlo.constant dense<1000> : tensor<i32>
// CHECK-NEXT:    %[[R0:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[INIT]]) applies stablehlo.minimum across dimensions = [1] : (tensor<2x2xi32>, tensor<i32>) -> tensor<2xi32>
// CHECK-NEXT:    %[[R1:.*]] = stablehlo.reduce(%[[ARG1]] init: %[[INIT]]) applies stablehlo.minimum across dimensions = [1] : (tensor<2x2xi32>, tensor<i32>) -> tensor<2xi32>
// CHECK-NEXT:    %[[MIN:.*]] = stablehlo.minimum %[[R0]], %[[R1]] : tensor<2xi32>
// CHECK-NEXT:    %[[AR_MIN:.*]] = "stablehlo.all_reduce"(%[[MIN]]) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes=[<name = "x", size = 2>, <name = "y", size = 2>]>, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> ({
// CHECK-NEXT:    ^bb0(%[[LHS_MIN:.*]]: tensor<i32>, %[[RHS_MIN:.*]]: tensor<i32>):
// CHECK-NEXT:      %[[RED_MIN:.*]] = stablehlo.minimum %[[LHS_MIN]], %[[RHS_MIN]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[RED_MIN]] : tensor<i32>
// CHECK-NEXT:    }) : (tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:    return %[[AR_MIN]] : tensor<2xi32>
// CHECK-NEXT:  }
func.func @parallel_reduce_min(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>},
  %arg1: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {"x"}]>})
  -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}]>}) {
  %init = stablehlo.constant dense<1000> : tensor<i32>
  %r0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>
  } : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %m = stablehlo.minimum %lhs, %rhs : tensor<i32>
      stablehlo.return %m : tensor<i32>
    }
  %r1 = stablehlo.reduce(%arg1 init: %init) across dimensions = [1] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>
  } : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %m = stablehlo.minimum %lhs, %rhs : tensor<i32>
      stablehlo.return %m : tensor<i32>
    }
  %out = stablehlo.minimum %r0, %r1 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>
  } : tensor<4xi32>
  return %out : tensor<4xi32>
}

//--- part2.mlir

func.func @main() {
  %a0 = stablehlo.constant dense<[
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<4x4xi32>
  %b0 = stablehlo.constant dense<[
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
  ]> : tensor<4x2xi32>
  %a1 = stablehlo.constant dense<[
    [ 2, -1,  4, -3],
    [-5,  6, -7,  8],
    [ 3,  1, -2,  5],
    [-4,  2,  6, -1]
  ]> : tensor<4x4xi32>
  %b1 = stablehlo.constant dense<[
    [ 2, -1],
    [-3,  4],
    [ 1,  5],
    [-2,  3]
  ]> : tensor<4x2xi32>
  %a2 = stablehlo.constant dense<[
    [10, 20,  5, 15],
    [ 4,  8, 12, 16],
    [ 7,  3,  9, 11],
    [ 2, 14,  6, 18]
  ]> : tensor<4x4xi32>
  %b2 = stablehlo.constant dense<[
    [1, 0],
    [0, 1],
    [2, 1],
    [1, 2]
  ]> : tensor<4x2xi32>

  // 1. Test dot add & subtract across 2x2 mesh:
  // Device grid (x, y):
  //   dev 0: (x=0, y=0) -> rows [0:2], cols [0:2]
  //   dev 1: (x=0, y=1) -> rows [2:4], cols [0:2]
  //   dev 2: (x=1, y=0) -> rows [0:2], cols [2:4]
  //   dev 3: (x=1, y=1) -> rows [2:4], cols [2:4]
  %seq_dot = func.call @sequential_dot_add_and_subtract(%a0, %b0, %a1, %b1, %a2, %b2)
    : (tensor<4x4xi32>, tensor<4x2xi32>, tensor<4x4xi32>, tensor<4x2xi32>, tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>

  %a0_00 = "stablehlo.slice"(%a0) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a0_01 = "stablehlo.slice"(%a0) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a0_10 = "stablehlo.slice"(%a0) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a0_11 = "stablehlo.slice"(%a0) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>

  %a1_00 = "stablehlo.slice"(%a1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a1_01 = "stablehlo.slice"(%a1) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a1_10 = "stablehlo.slice"(%a1) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a1_11 = "stablehlo.slice"(%a1) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>

  %a2_00 = "stablehlo.slice"(%a2) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a2_01 = "stablehlo.slice"(%a2) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a2_10 = "stablehlo.slice"(%a2) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>
  %a2_11 = "stablehlo.slice"(%a2) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<2x2xi32>

  %b0_0 = "stablehlo.slice"(%b0) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %b0_1 = "stablehlo.slice"(%b0) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %b1_0 = "stablehlo.slice"(%b1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %b1_1 = "stablehlo.slice"(%b1) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %b2_0 = "stablehlo.slice"(%b2) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %b2_1 = "stablehlo.slice"(%b2) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %res_dot:4 = "interpreter.run_parallel"(
    %a0_00, %b0_0, %a1_00, %b1_0, %a2_00, %b2_0,
    %a0_01, %b0_0, %a1_01, %b1_0, %a2_01, %b2_0,
    %a0_10, %b0_1, %a1_10, %b1_1, %a2_10, %b2_1,
    %a0_11, %b0_1, %a1_11, %b1_1, %a2_11, %b2_1
  ) {
    programs = [[
      @parallel_dot_add_and_subtract,
      @parallel_dot_add_and_subtract,
      @parallel_dot_add_and_subtract,
      @parallel_dot_add_and_subtract
    ]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>)
    -> (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>)

  %par_dot_x0 = stablehlo.concatenate %res_dot#0, %res_dot#1, dim = 0
    : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  %par_dot_x1 = stablehlo.concatenate %res_dot#2, %res_dot#3, dim = 0
    : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  "check.expect_eq"(%par_dot_x0, %seq_dot) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%par_dot_x1, %seq_dot) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()

  // 2. Test max reduction + maximum commutation across 2x2 mesh:
  %seq_max = func.call @sequential_reduce_max(%a1, %a2)
    : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4xi32>
  %res_max:4 = "interpreter.run_parallel"(
    %a1_00, %a2_00,
    %a1_01, %a2_01,
    %a1_10, %a2_10,
    %a1_11, %a2_11
  ) {
    programs = [[
      @parallel_reduce_max,
      @parallel_reduce_max,
      @parallel_reduce_max,
      @parallel_reduce_max
    ]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>)
    -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)

  %par_max_x0 = stablehlo.concatenate %res_max#0, %res_max#1, dim = 0
    : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  %par_max_x1 = stablehlo.concatenate %res_max#2, %res_max#3, dim = 0
    : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  "check.expect_eq"(%par_max_x0, %seq_max) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%par_max_x1, %seq_max) : (tensor<4xi32>, tensor<4xi32>) -> ()

  // 3. Test min reduction + minimum commutation across 2x2 mesh:
  %seq_min = func.call @sequential_reduce_min(%a1, %a2)
    : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4xi32>
  %res_min:4 = "interpreter.run_parallel"(
    %a1_00, %a2_00,
    %a1_01, %a2_01,
    %a1_10, %a2_10,
    %a1_11, %a2_11
  ) {
    programs = [[
      @parallel_reduce_min,
      @parallel_reduce_min,
      @parallel_reduce_min,
      @parallel_reduce_min
    ]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>)
    -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)

  %par_min_x0 = stablehlo.concatenate %res_min#0, %res_min#1, dim = 0
    : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  %par_min_x1 = stablehlo.concatenate %res_min#2, %res_min#3, dim = 0
    : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  "check.expect_eq"(%par_min_x0, %seq_min) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%par_min_x1, %seq_min) : (tensor<4xi32>, tensor<4xi32>) -> ()

  return
}
