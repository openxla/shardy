// RUN: sdy_opt %s -sdy-convert-global-to-local -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @propagates_caller_loc_to_body_and_nested_regions
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>} loc({{[^)]*}}))
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
func.func @propagates_caller_loc_to_body_and_nested_regions(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xf32>) {
    // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG0]] : tensor<4x8xf32> loc(callsite("add:"("inner/add"("body"("body.py":20:1))) at "call:"("jit(run)/outer/shard_map"("run"("main.py":10:1)))))
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32> loc("add:"("inner/add"("body"("body.py":20:1))))
    // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[ADD]]) <{
    // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
    // CHECK-SAME: }> ({
    // CHECK-NEXT: ^bb0(%[[RED_ARG0:.*]]: tensor<f32> loc(callsite("red_arg"("body.py":21:3) at "call:"("jit(run)/outer/shard_map"("run"("main.py":10:1))))), %[[RED_ARG1:.*]]: tensor<f32> loc(callsite("red_arg"("body.py":21:3) at "call:"("jit(run)/outer/shard_map"("run"("main.py":10:1)))))):
    // CHECK-NEXT:   %[[RED_SUM:.*]] = stablehlo.add %[[RED_ARG0]], %[[RED_ARG1]] : tensor<f32> loc(callsite("add:"("inner/red_add"("body"("body.py":22:5))) at "call:"("jit(run)/outer/shard_map"("run"("main.py":10:1)))))
    // CHECK-NEXT:   stablehlo.return %[[RED_SUM]] : tensor<f32> loc("call:"("jit(run)/outer/shard_map"("run"("main.py":10:1))))
    // CHECK-NEXT: }) : (tensor<4x8xf32>) -> tensor<4x8xf32> loc(callsite("all_reduce:"("inner/psum"("body"("body.py":21:1))) at "call:"("jit(run)/outer/shard_map"("run"("main.py":10:1)))))
    %2 = "stablehlo.all_reduce"(%1) ({
      ^bb0(%arg2: tensor<f32> loc("red_arg"("body.py":21:3)), %arg3: tensor<f32> loc("red_arg"("body.py":21:3))):
        %3 = stablehlo.add %arg2, %arg3 : tensor<f32> loc("add:"("inner/red_add"("body"("body.py":22:5))))
        stablehlo.return %3 : tensor<f32> loc(unknown)
    }) {
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
    } : (tensor<4x8xf32>) -> tensor<4x8xf32> loc("all_reduce:"("inner/psum"("body"("body.py":21:1))))
    // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[AR]] : tensor<4x8xf32> loc("call:"("jit(run)/outer/shard_map"("run"("main.py":10:1))))
    %4 = stablehlo.tanh %2 : tensor<4x8xf32> loc(unknown)
    sdy.return %4 : tensor<4x8xf32>
  } : (tensor<16x8xf32>) -> tensor<4x8xf32> loc("call:"("jit(run)/outer/shard_map"("run"("main.py":10:1))))
  // CHECK-NEXT: return %[[TANH]] : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @nested_manual_computations_stack_locations
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>} loc({{[^)]*}}))
// CHECK-SAME: -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
func.func @nested_manual_computations_stack_locations(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y"}, {}]>]>} : tensor<2x8xf32> loc(callsite("mid/abs"("mid"("mid.py":20:1)) at "jit(run)/outer_shmap"("run"("main.py":10:1))))
    %1 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y"}, {}]>]>} : tensor<8x8xf32> loc("mid/abs"("mid"("mid.py":20:1)))
    %2 = sdy.manual_computation(%1)
      in_shardings=[<@mesh_2_4, [{"y"}, {}]>]
      out_shardings=[<@mesh_2_4, [{"y"}, {}]>]
      manual_axes={"y"} (%arg2: tensor<2x8xf32>) {
      // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ABS]], %[[ABS]] : tensor<2x8xf32> loc(callsite("leaf/add"("leaf"("leaf.py":30:1)) at callsite("mid/inner_shmap"("mid"("mid.py":25:1)) at "jit(run)/outer_shmap"("run"("main.py":10:1)))))
      %3 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32> loc("leaf/add"("leaf"("leaf.py":30:1)))
      sdy.return %3 : tensor<2x8xf32>
    } : (tensor<8x8xf32>) -> tensor<8x8xf32> loc("mid/inner_shmap"("mid"("mid.py":25:1)))
    sdy.return %2 : tensor<8x8xf32>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32> loc("jit(run)/outer_shmap"("run"("main.py":10:1)))
  // CHECK-NEXT: return %[[ADD]] : tensor<2x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @skips_redundant_prefix_location
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>} loc({{[^)]*}}))
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
func.func @skips_redundant_prefix_location(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{"x"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xf32>) {
    // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] : tensor<4x8xf32> loc("tanh:"("jit(run)/mod/shard_map/tanh"(callsite("shmap_body"("a.py":5:1) at "mod"("a.py":10:1)))))
    %1 = stablehlo.tanh %arg1 : tensor<4x8xf32> loc("tanh:"("jit(run)/mod/shard_map/tanh"(callsite("shmap_body"("a.py":5:1) at "mod"("a.py":10:1)))))
    sdy.return %1 : tensor<4x8xf32>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32> loc("shard_map:"("jit(run)/mod/shard_map"("mod"("a.py":10:1))))
  // CHECK-NEXT: return %[[TANH]] : tensor<4x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @compiler_generated_same_location_not_duplicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>} loc({{[^)]*}}))
// CHECK-SAME: -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
func.func @compiler_generated_same_location_not_duplicated(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<8x8xf32>) {
    %1 = sdy.manual_computation(%arg1)
      in_shardings=[<@mesh_2_4, [{"y"}, {}]>]
      out_shardings=[<@mesh_2_4, [{"y"}, {}]>]
      manual_axes={"y"} (%arg2: tensor<2x8xf32>) {
      // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] : tensor<2x8xf32> loc(callsite("inner/pad"("pad_fn"("b.py":20:1)) at "jit(run)/shard_map"("run"("a.py":10:1))))
      %2 = stablehlo.tanh %arg2 : tensor<2x8xf32> loc("inner/pad"("pad_fn"("b.py":20:1)))
      sdy.return %2 : tensor<2x8xf32>
    } : (tensor<8x8xf32>) -> tensor<8x8xf32> loc("inner/pad"("pad_fn"("b.py":20:1)))
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32> loc("jit(run)/shard_map"("run"("a.py":10:1)))
  // CHECK-NEXT: return %[[TANH]] : tensor<2x8xf32>
  return %0 : tensor<16x8xf32>
}
