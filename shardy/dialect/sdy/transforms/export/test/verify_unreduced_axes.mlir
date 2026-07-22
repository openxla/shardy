// RUN: sdy_opt %s -sdy-verify-unreduced-axes -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @dropped_by_add(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'stablehlo.add' op dropped unreduced axis 'y' without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @dropped_by_add_blessed(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func private @callee_without_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

func.func @call_argument_drops_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'func.call' op has unreduced axes mismatch for 'y' at call argument 0.}}
  %0 = func.call @callee_without_unreduced(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func private @callee_without_unreduced_blessed(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

func.func @call_argument_drops_unreduced_blessed(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  %1 = func.call @callee_without_unreduced_blessed(%0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func private @callee_with_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) {
  // expected-error@+1 {{'func.return' op has unreduced axes mismatch for 'y' at return value 0 without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
  return %arg0 : tensor<8x8xf32>
}

func.func @call_result_drops_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'func.call' op has unreduced axes mismatch for 'y' at call result 0.}}
  %0 = func.call @callee_with_unreduced(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func private @callee_returns_reduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

func.func @call_result_extra_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'func.call' op has unreduced axes mismatch for 'y' at call result 0.}}
  %0 = func.call @callee_returns_reduced(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func private @callee_expects_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

func.func @call_operand_missing_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'func.call' op has unreduced axes mismatch for 'y' at call argument 0.}}
  %0 = func.call @callee_expects_unreduced(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// This test verifies that standard operations are allowed to introduce new
// unreduced axes on results (e.g. contracting dims in dot_general or reduction).
func.func @dot_general_introduces_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// This test verifies that sdy.manual_computation ops are NOT allowed to drop unreduced axes.
func.func @manual_computation_drops_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'sdy.manual_computation' op dropped unreduced axis 'y' without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}], unreduced={"y"}>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg1: tensor<4x8xf32>) {
    %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    sdy.return %1 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// This test verifies that sdy.manual_computation ops are allowed to have unreduced axes, which are then dropped by a sharding_constraint.
func.func @manual_computation_drops_unreduced_proper(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}], unreduced={"y"}>] out_shardings=[<@mesh, [{"x"}, {}], unreduced={"y"}>] manual_axes={"x", "y"} (%arg1: tensor<4x8xf32>) {
    %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    sdy.return %1 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @reshard_mismatches_unreduced_kind(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced=max{"x", "y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'sdy.reshard' op cannot change the reduction operator of kept unreduced axes from max to sum.}}
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @constraint_mismatches_unreduced_kind(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced=max{"x", "y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'sdy.sharding_constraint' op cannot change the reduction operator of kept unreduced axes from max to sum.}}
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @dot_unreduced_max_error(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced=max{"y"}>}) {
  // expected-error@+1 {{cannot introduce 'max' unreduced axes. Expected 'sum'.}}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced=max{"y"}>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @reduce_sum_unreduced_max_error(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=max{"x"}>}) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // expected-error@+1 {{cannot introduce 'max' unreduced axes. Expected 'sum'.}}
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced=max{"x"}>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @reduce_max_unreduced_max_success(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=max{"x"}>}) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced=max{"x"}>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @stablehlo_all_reduce_mismatch(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced=max{"x"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'stablehlo.all_reduce' op cannot reduce 'max' unreduced axes. Expected 'sum'.}}
  %0 = "stablehlo.all_reduce"(%arg0) <{replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "x">]>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @reshard_mismatch_reduction_op(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=max{"x"}>}) -> tensor<8xf32> {
  // expected-error@+1 {{'sdy.reshard' op cannot change the reduction operator of kept unreduced axes from max to sum.}}
  %0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>

func.func @sdy_all_reduce_mismatch(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced=max{"y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'sdy.all_reduce' op cannot reduce 'max' unreduced axes. Expected 'sum'.}}
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>

func.func @sdy_all_reduce_unsharded(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{'sdy.all_reduce' op operand must be sharded.}}
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @dot_general_passthrough_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced=max{"y"}>}, %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced=max{"y"}>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced=max{"y"}>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @default_op_mismatches_unreduced_kind(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced=max{"y"}>}) -> tensor<8x8xf32> {
  // expected-error@+1 {{'stablehlo.add' op cannot change the reduction operator of kept unreduced axes from max to sum.}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @default_op_introduces_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @constant_adds_unreduced() -> tensor<8x8xf32> {
  %0 = "sdy.constant"() <{value = dense<1.000000e+00> : tensor<8x8xf32>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>} : () -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @default_op_drops_bypassed_reshard(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh_a = <["x"=2, "y"=2]>
sdy.mesh @mesh_b = <["z"=2, "w"=2]>

func.func @different_meshes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a, [{"y"}, {}], unreduced={"x"}>}) -> tensor<8x8xf32> {
  // If verifying, we would drop unreduced axis 'x' from mesh_a, and introduce 'w' from mesh_b.
  // Since they are on different meshes, verifyUnreducedAxesTransition returns success immediately.
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_b, [{"z"}, {}], unreduced={"w"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
