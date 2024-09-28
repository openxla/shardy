// RUN: sdy_opt %s -split-input-file -sdy-sharding-group-import -verify-diagnostics

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Allow sharding groups where group values don't cross ManualComputationOps
// barrier.
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    sdy.sharding_group %1 group_id = 8675 : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  sdy.sharding_group %0 group_id = 309 : tensor<8x8xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Don't permit the creation of sharding groups which mix values who have parent
// ManualComputationOps with those that don't.
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    sdy.sharding_group %1 group_id = 90210 : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 90210}}
  sdy.sharding_group %0 group_id = 90210 : tensor<8x8xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Don't permit the creation of sharding groups which have different manual
// computation op parents
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    sdy.sharding_group %1 group_id = 44094 : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %4 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%2: tensor<8x8xf32>) {
    %3 = stablehlo.add %2, %2 : tensor<8x8xf32>
    // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 44094}}
    sdy.sharding_group %3 group_id = 44094 : tensor<8x8xf32>
    sdy.return %3 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  func.return %4: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Allow sharding groups with values that remain in the same level of a nested
// manual computation
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>

    %3 = sdy.manual_computation(%2) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg2: tensor<8x8xf32>) {
      %4 = stablehlo.add %arg2, %arg2 : tensor<8x8xf32>
      %5 = stablehlo.add %4, %arg2 : tensor<8x8xf32>
      sdy.sharding_group %4 group_id = 1881 : tensor<8x8xf32>
      sdy.sharding_group %5 group_id = 1881 : tensor<8x8xf32>
      sdy.return %5 : tensor<8x8xf32>
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>

    sdy.sharding_group %1 group_id = 8008 : tensor<8x8xf32>
    sdy.sharding_group %2 group_id = 8008 : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Don't allow sharding groups with values at different levels of blocks within
// a nested manual computation.
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg2: tensor<8x8xf32>) {
      %3 = stablehlo.add %arg2, %arg2 : tensor<8x8xf32>
      sdy.sharding_group %3 group_id = 4311 : tensor<8x8xf32>
      sdy.return %3 : tensor<8x8xf32>
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>

    // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 4311}}
    sdy.sharding_group %1 group_id = 4311 : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Allow sharding groups within ManualComputationOp which has a WhileOp and
// 1) Some group members are outside of the WhileOp and some are inside
// 2) All ops have the same parent ManualComputationOp
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
    sdy.sharding_group %1 group_id = 1337 : tensor<8x8xf32>
    sdy.sharding_group %2 group_id = 1337 : tensor<8x8xf32>

    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<32> : tensor<i32>
    %6:2 = stablehlo.while(%iterArg = %2, %iterArg_2 = %3) : tensor<8x8xf32>, tensor<i32>
      cond {
      %7 = stablehlo.compare  LT, %iterArg_2, %5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7 : tensor<i1>
    } do {
      %7 = sdy.data_flow_edge %iterArg sharding=<@mesh, [{"a"}, {}]> : tensor<8x8xf32>
      %8 = stablehlo.add %iterArg_2, %4 : tensor<i32>
      %9 = stablehlo.add %7, %7 : tensor<8x8xf32>
      sdy.sharding_group %9 group_id = 1337 : tensor<8x8xf32>
      stablehlo.return %9, %8 : tensor<8x8xf32>, tensor<i32>
    }
    sdy.return %2 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Disallow sharding groups which cross the barrier of a ManualComputationOp
// (and also a while op).
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<32> : tensor<i32>
    %6:2 = stablehlo.while(%iterArg = %2, %iterArg_2 = %3) : tensor<8x8xf32>, tensor<i32>
      cond {
      %7 = stablehlo.compare  LT, %iterArg_2, %5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7 : tensor<i1>
    } do {
      %7 = sdy.data_flow_edge %iterArg sharding=<@mesh, [{"a"}, {}]> : tensor<8x8xf32>
      %8 = stablehlo.add %iterArg_2, %4 : tensor<i32>
      %9 = stablehlo.add %7, %7 : tensor<8x8xf32>
      sdy.sharding_group %9 group_id = 7331 : tensor<8x8xf32>
      stablehlo.return %9, %8 : tensor<8x8xf32>, tensor<i32>
    }
    sdy.return %2 : tensor<8x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 7331}}
  sdy.sharding_group %0 group_id = 7331 : tensor<8x8xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Disallow creation of sharding groups which have values with different shapes.
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<8x8x1xf32>
  sdy.sharding_group %arg0 group_id = 23 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 23 : tensor<8x8xf32>
  // expected-error@below {{ShardingGroupOps values must have the same shape for groupId: 23}}
  sdy.sharding_group %1 group_id = 23 : tensor<8x8x1xf32>
  func.return %0: tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Throw error for sharding groups which have incompatible shardings inferred
// from initial constraints.
func.func @error_for_incompatible_shardings_in_sharding_group(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
  // Sharding Group and Sharding Constraint compatibility checks happend after
  // unification + canonicalization of group ids, which is why the group id
  // below (555) corresponds to group id: 0 in the check-error.
  sdy.sharding_group %arg0 group_id = 555 : tensor<8x8xf32>
  // expected-error@below {{Inconsistent shardings prior to propagation for ShardingGroupOps with canonicalized groupId: 0}}
  sdy.sharding_group %arg1 group_id = 555 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Throw error for sharding groups which have incompatible shardings inferred
// from initial constraints.
func.func @error_for_transitively_inferred_incompatible_shardings_in_unified_sharding_group(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {

  %0 = stablehlo.constant dense<0.0> : tensor<8x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<8x8xf32>

  sdy.sharding_group %arg0 group_id = 10 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 10 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 20 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 20 : tensor<8x8xf32>

  // The shard group below will cause the above sharding groups to be merged
  // by transitivity this implies that all of {%arg0, %arg1, 0, 1} should have
  // the same sharding. Note that %0 and %1 are compatible by them selves but
  // %arg0 and %arg1 are not due to their initial shardings.
  sdy.sharding_group %1 group_id = 30 : tensor<8x8xf32>
  // expected-error@below {{Inconsistent shardings prior to propagation for ShardingGroupOps with canonicalized groupId: 0}}
  sdy.sharding_group %arg1 group_id = 30 : tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @error_for_incompatible_shardings_in_manual_computation(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"b"}, {}]>] manual_axes={} (%arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) {
    sdy.sharding_group %arg2 group_id = 8675 : tensor<8x8xf32>
    // expected-error@below {{Inconsistent shardings prior to propagation for ShardingGroupOps with canonicalized groupId: 0}}
    sdy.sharding_group %arg3 group_id = 8675 : tensor<8x8xf32>
    sdy.return %arg2 : tensor<8x8xf32>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @error_for_incompatible_shardings_with_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 1000 : tensor<8x8xf32>
  // expected-error@below {{Inconsistent shardings prior to propagation for ShardingGroupOps with canonicalized groupId: 0}}
  sdy.sharding_group %1 group_id = 1000 : tensor<8x8xf32>
  func.return
}

