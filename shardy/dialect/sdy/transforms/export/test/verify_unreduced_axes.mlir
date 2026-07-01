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
  // expected-error@+1 {{'func.return' op has unreduced axes mismatch for 'y' without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
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

// This test verifies that sdy.manual_computation ops are allowed to drop unreduced axes.
func.func @manual_computation_drops_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
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

