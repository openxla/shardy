// RUN: sdy_opt %s -sdy-propagation-pipeline='dedup-functions-fully=true' -split-input-file 2>&1 | FileCheck %s

// Test Case 1 (shmap_ag): Async compute-on with a manual all-gather inside
// @compute_on.2 and compute-on-done wrapped in a manual_computation on a 2D
// mesh. Verifies that:
// 1. %arg2 starts without a sharding attribute and gets [{"x"}, {"y"}]
//    back-propagated from @compute_on.2 across compute-on-start.
// 2. @compute_on.2's [{}, {"y"}] result sharding propagates out of
//    compute-on-start, into and across compute-on-done (closing its open
//    out_shardings=[<@mesh, [{}, {?}], replicated={"x"}>]), and through the
//    final dot to @main's return sharding [{"x"}, {"y"}].
sdy.mesh @mesh = <["x"=8, "y"=2]>

// The manual_computation expects [{"x"}, {"y"}] on input and all-gathers dim 0
// across manual axis "x" to produce a [{}, {"y"}] result.
func.func private @compute_on.2(%arg0: tensor<256x128xf32>)
    -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh, [{"x"}, {"y"}]>]
      out_shardings=[<@mesh, [{}, {"y"}], replicated={"x"}>]
      manual_axes={"x"} (%arg1: tensor<32x128xf32>) {
    %2 = "stablehlo.all_gather"(%arg1) <{
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
      use_global_device_ids
    }> : (tensor<32x128xf32>) -> tensor<256x128xf32>
    sdy.return %2 : tensor<256x128xf32>
  } : (tensor<256x128xf32>) -> tensor<256x128xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"y"}]> : tensor<256x128xf32>
  return %1 : tensor<256x128xf32>
}

// Note that %arg2 and @main's result begin without shardings. After the
// propagation pipeline, [{"x"}, {"y"}] is back-propagated from @compute_on.2 to
// %arg2, and [{"x"}, {"y"}] is forward-propagated to @main's result.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:      %arg1: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:      %arg2: tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
func.func @main(
    %arg0: tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg2: tensor<256x128xf32>)
    -> tensor<1024x128xf32> {

  // 1. "compute-on-start" (rewritten to func.call) back-propagates
  // @compute_on.2's [{"x"}, {"y"}] parameter sharding to %arg2 and receives
  // @compute_on.2's [{}, {"y"}] result sharding.
  // CHECK-NEXT: %[[START:.*]] = call @compute_on.2(%arg2) {mhlo.frontend_attributes = {xla.sdy.is_compute_on_start = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
  // CHECK-NEXT: %[[BARRIER_0:.*]]:3 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>, <@mesh, [{"x"}, {}]>, <@mesh, [{}, {}]>]>} %[[START]], %arg0, %arg1 : tensor<256x128xf32>, tensor<1024x128xf32>, tensor<128x256xf32>
  //
  // 2. The overlapping compute propagates [{"x"}, {}] from %arg0.
  // CHECK-NEXT: %[[DOT_0:.*]] = stablehlo.dot %[[BARRIER_0]]#1, %[[BARRIER_0]]#2, precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
  // CHECK-NEXT: %[[BARRIER_1:.*]]:2 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} %[[DOT_0]], %[[BARRIER_0]]#0 : tensor<1024x256xf32>, tensor<256x128xf32>
  //
  // 3. "compute-on-done" (wrapped in manual_computation with manual_axes={"x"})
  // propagates free axis "y" from %arg3 to its result and out_shardings.
  // CHECK-NEXT: %[[MANUAL:.*]] = sdy.manual_computation(%[[BARRIER_1]]#1) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x"} (%arg3: tensor<256x128xf32>) {
  // CHECK-NEXT:   %[[DONE:.*]] = stablehlo.custom_call @"compute-on-done"(%arg3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
  // CHECK-NEXT:   sdy.return %[[DONE]] : tensor<256x128xf32>
  // CHECK-NEXT: } : (tensor<256x128xf32>) -> tensor<256x128xf32>
  //
  // 4. The final compute combines [{"x"}, {}] from the overlapping dot and
  // [{}, {"y"}] from compute-on-done into [{"x"}, {"y"}].
  // CHECK-NEXT: %[[BARRIER_2:.*]]:2 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} %[[BARRIER_1]]#0, %[[MANUAL]] : tensor<1024x256xf32>, tensor<256x128xf32>
  // CHECK-NEXT: %[[DOT_1:.*]] = stablehlo.dot %[[BARRIER_2]]#0, %[[BARRIER_2]]#1, precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
  // CHECK-NEXT: return %[[DOT_1]] : tensor<1024x128xf32>

  // compute-on-start (rewritten to func.call) with no initial sharding
  // attribute on result %0.
  %0 = call @compute_on.2(%arg2) {
    mhlo.frontend_attributes = {xla.sdy.is_compute_on_start = "true"}
  } : (tensor<256x128xf32>) -> tensor<256x128xf32>
  %1:3 = stablehlo.optimization_barrier %0, %arg0, %arg1 :
      tensor<256x128xf32>, tensor<1024x128xf32>, tensor<128x256xf32>

  // Overlapping compute
  %2 = stablehlo.dot %1#1, %1#2, precision = [DEFAULT, DEFAULT] :
      (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
  %3:2 = stablehlo.optimization_barrier %2, %1#0 :
      tensor<1024x256xf32>, tensor<256x128xf32>

  // compute-on-done wrapped in manual_computation with open dim 1 on
  // in_shardings and out_shardings.
  %4 = sdy.manual_computation(%3#1)
      in_shardings=[<@mesh, [{}, {?}], replicated={"x"}>]
      out_shardings=[<@mesh, [{}, {?}], replicated={"x"}>]
      manual_axes={"x"} (%arg3: tensor<256x128xf32>) {
    %7 = stablehlo.custom_call @"compute-on-done"(%arg3) :
        (tensor<256x128xf32>) -> tensor<256x128xf32>
    sdy.return %7 : tensor<256x128xf32>
  } : (tensor<256x128xf32>) -> tensor<256x128xf32>

  // Final compute that consumes compute-on-done result.
  %5:2 = stablehlo.optimization_barrier %3#0, %4 :
      tensor<1024x256xf32>, tensor<256x128xf32>
  %6 = stablehlo.dot %5#0, %5#1, precision = [DEFAULT, DEFAULT] :
      (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
  return %6 : tensor<1024x128xf32>
}

// Checks for @compute_on.2: %arg0 receives [{"x"}, {"y"}] propagated from
// manual_computation's in_shardings (which also propagates across
// compute-on-start to @main's %arg2), and the function returns [{}, {"y"}]
// after the manual all_gather.
// CHECK-LABEL: func private @compute_on.2(
// CHECK-SAME:      %arg0: tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
// CHECK-NEXT:    %[[RESHARD_ARG:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<256x128xf32>
// CHECK-NEXT:    %[[MANUAL_AG:.*]] = sdy.manual_computation(%[[RESHARD_ARG]]) in_shardings=[<@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x"} (%arg1: tensor<32x128xf32>) {
// CHECK-NEXT:      %[[AG:.*]] = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<{{\[\[0, 1, 2, 3, 4, 5, 6, 7\]\]}}> : tensor<1x8xi64>, use_global_device_ids}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<32x128xf32>) -> tensor<256x128xf32>
// CHECK-NEXT:      sdy.return %[[AG]] : tensor<256x128xf32>
// CHECK-NEXT:    } : (tensor<256x128xf32>) -> tensor<256x128xf32>
// CHECK-NEXT:    %[[RESHARD_OUT:.*]] = sdy.reshard %[[MANUAL_AG]] <@mesh, [{}, {"y"}]> : tensor<256x128xf32>
// CHECK-NEXT:    return %[[RESHARD_OUT]] : tensor<256x128xf32>
// CHECK-NEXT:  }

// -----

// Test Case 2 (reshard_ag): Async compute-on with a sharding_constraint inside
// @compute_on.1 and compute-on-done wrapped in a manual_computation on a 2D
// mesh. Verifies that:
// 1. %arg2's [{"x"}, {"y"}] sharding forward-propagates across compute-on-start
//    into @compute_on.1's %arg0.
// 2. @compute_on.1's [{}, {"y"}] result sharding propagates out of
//    compute-on-start, into and across compute-on-done (closing its open
//    out_shardings=[<@mesh, [{}, {?}], replicated={"x"}>]), and through the
//    final dot to @main's return sharding [{"x"}, {"y"}].
sdy.mesh @mesh = <["x"=2, "y"=2]>

// %arg0 begins without a sharding and the sharding_constraint unshards dim 0
// across "x" while keeping dim 1 sharded on "y" to produce a [{}, {"y"}]
// result.
func.func private @compute_on.1(%arg0: tensor<256x128xf32>)
    -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"y"}]> : tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// Note that %arg2 begins with a [{"x"}, {"y"}] sharding and @main's result
// begins without a sharding. After the propagation pipeline, [{"x"}, {"y"}] is
// forward-propagated from %arg2 to @compute_on.1, and [{"x"}, {"y"}] is
// propagated to @main's result.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:      %arg1: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:      %arg2: tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
func.func @main(
    %arg0: tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg2: tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
    -> tensor<1024x128xf32> {

  // 1. "compute-on-start" (rewritten to func.call) propagates %arg2's
  // [{"x"}, {"y"}] sharding into @compute_on.1's %arg0 and receives
  // @compute_on.1's [{}, {"y"}] result sharding.
  // CHECK-NEXT: %[[START:.*]] = call @compute_on.1(%arg2) {mhlo.frontend_attributes = {xla.sdy.is_compute_on_start = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
  // CHECK-NEXT: %[[BARRIER_0:.*]]:3 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>, <@mesh, [{"x"}, {}]>, <@mesh, [{}, {}]>]>} %[[START]], %arg0, %arg1 : tensor<256x128xf32>, tensor<1024x128xf32>, tensor<128x256xf32>
  //
  // 2. The overlapping compute propagates [{"x"}, {}] from %arg0.
  // CHECK-NEXT: %[[DOT_0:.*]] = stablehlo.dot %[[BARRIER_0]]#1, %[[BARRIER_0]]#2, precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
  // CHECK-NEXT: %[[BARRIER_1:.*]]:2 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} %[[DOT_0]], %[[BARRIER_0]]#0 : tensor<1024x256xf32>, tensor<256x128xf32>
  //
  // 3. "compute-on-done" (wrapped in manual_computation with manual_axes={"x"})
  // propagates free axis "y" from %arg3 to its result and out_shardings.
  // CHECK-NEXT: %[[MANUAL:.*]] = sdy.manual_computation(%[[BARRIER_1]]#1) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x"} (%arg3: tensor<256x128xf32>) {
  // CHECK-NEXT:   %[[DONE:.*]] = stablehlo.custom_call @"compute-on-done"(%arg3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
  // CHECK-NEXT:   sdy.return %[[DONE]] : tensor<256x128xf32>
  // CHECK-NEXT: } : (tensor<256x128xf32>) -> tensor<256x128xf32>
  //
  // 4. The final compute combines [{"x"}, {}] from the overlapping dot and
  // [{}, {"y"}] from compute-on-done into [{"x"}, {"y"}].
  // CHECK-NEXT: %[[BARRIER_2:.*]]:2 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} %[[BARRIER_1]]#0, %[[MANUAL]] : tensor<1024x256xf32>, tensor<256x128xf32>
  // CHECK-NEXT: %[[DOT_1:.*]] = stablehlo.dot %[[BARRIER_2]]#0, %[[BARRIER_2]]#1, precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
  // CHECK-NEXT: return %[[DOT_1]] : tensor<1024x128xf32>

  // compute-on-start (rewritten to func.call) with no initial sharding
  // attribute on result %0.
  %0 = call @compute_on.1(%arg2) {
    mhlo.frontend_attributes = {xla.sdy.is_compute_on_start = "true"}
  } : (tensor<256x128xf32>) -> tensor<256x128xf32>
  %1:3 = stablehlo.optimization_barrier %0, %arg0, %arg1 :
      tensor<256x128xf32>, tensor<1024x128xf32>, tensor<128x256xf32>

  // Overlapping compute
  %2 = stablehlo.dot %1#1, %1#2, precision = [DEFAULT, DEFAULT] :
      (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
  %3:2 = stablehlo.optimization_barrier %2, %1#0 :
      tensor<1024x256xf32>, tensor<256x128xf32>

  // compute-on-done wrapped in manual_computation with open dim 1 on
  // in_shardings and out_shardings.
  %4 = sdy.manual_computation(%3#1)
      in_shardings=[<@mesh, [{}, {?}], replicated={"x"}>]
      out_shardings=[<@mesh, [{}, {?}], replicated={"x"}>]
      manual_axes={"x"} (%arg3: tensor<256x128xf32>) {
    %7 = stablehlo.custom_call @"compute-on-done"(%arg3) :
        (tensor<256x128xf32>) -> tensor<256x128xf32>
    sdy.return %7 : tensor<256x128xf32>
  } : (tensor<256x128xf32>) -> tensor<256x128xf32>

  // Final compute that consumes compute-on-done result.
  %5:2 = stablehlo.optimization_barrier %3#0, %4 :
      tensor<1024x256xf32>, tensor<256x128xf32>
  %6 = stablehlo.dot %5#0, %5#1, precision = [DEFAULT, DEFAULT] :
      (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
  return %6 : tensor<1024x128xf32>
}

// Checks for @compute_on.1: %arg0 receives [{"x"}, {"y"}] propagated from
// @main's %arg2 across compute-on-start, and the function reshards %arg0 to
// [{}, {"y"}] for the return value.
// CHECK-LABEL: func private @compute_on.1(
// CHECK-SAME:      %arg0: tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
// CHECK-NEXT:    %[[RESHARD_0:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<256x128xf32>
// CHECK-NEXT:    %[[RESHARD_1:.*]] = sdy.reshard %[[RESHARD_0]] <@mesh, [{}, {"y"}]> : tensor<256x128xf32>
// CHECK-NEXT:    return %[[RESHARD_1]] : tensor<256x128xf32>
// CHECK-NEXT:  }
