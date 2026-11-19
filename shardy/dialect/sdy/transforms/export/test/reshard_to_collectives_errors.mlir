// RUN: sdy_opt %s -sdy-reshard-to-collectives -split-input-file -verify-diagnostics

sdy.mesh @mesh3d = <["x"=2, "y"=2, "z"=2]>

func.func @sharded_to_replicated_mismatched_unreduced_axes(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"z"}, {}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh3d, [{}, {}], unreduced={"y"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh2d = <["x"=2, "y"=2]>

func.func @sharded_to_replicated_introduced_unreduced_axes(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh2d, [{}, {}], unreduced=max{"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh2d_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>

func.func @cross_mesh_replicated_mismatched_unreduced_axes_and_reduction_op(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh2d_non_iota, [{}, {}], unreduced=max{"y"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh1d_6 = <["x"=6]>
sdy.mesh @mesh2d_2x3 = <["x"=2, "y"=3]>

func.func @replicated_dims_with_unreduced_incompatible_meshes(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{}, {}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh2d_2x3, [{}, {}], unreduced={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh2d_copy = <["x"=2, "y"=2]>

func.func @replicated_dims_with_unreduced_same_mesh_def(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh2d_copy, [{}, {}], unreduced={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh2d = <["x"=2, "y"=2]>

func.func @replicated_dims_with_unreduced_symbol_to_inlined_same_mesh_def(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <mesh<["x"=2, "y"=2]>, [{}, {}], unreduced={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh2d_copy = <["x"=2, "y"=2]>

func.func @sharded_with_unreduced_to_replicated_dims_same_mesh_def(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {"y"}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh2d_copy, [{}, {}], unreduced={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh1d_6 = <["x"=6]>

func.func @sharded_with_unreduced_to_replicated_no_unreduced_cross_mesh(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{}, {"y"}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh1d_6 = <["x"=6]>
sdy.mesh @mesh2d_2x3 = <["x"=2, "y"=3]>

func.func @replicated_with_unreduced_to_sharded_incompatible_meshes(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{}, {}], unreduced={"x"}>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'sdy.reshard' that was explicitly marked illegal}}
  %0 = sdy.reshard %arg0 <@mesh2d_2x3, [{"y"}, {}], unreduced={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
