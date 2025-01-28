// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_on_operand_without_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error @+1 {{collective on operand without sharding}}
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh2 = <["a"=2, "b"=2]>

// expected-note @+1 {{operand mesh: #sdy.mesh<["a"=2, "b"=2]>}}
func.func @all_gather_with_incompatible_meshes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"a"}, {"b"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result mesh does not match operand mesh}}
  %0 = sdy.all_gather [{}, {"b"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_invalid_out_sharding(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "x"}}
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x", "x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=4, "y"=2]>

func.func @all_gather_gathering_axes_can_be_merged(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{two consecutive sub-axes can be merged: "x":(1)2, "x":(2)2}}
  %0 = sdy.all_gather [{}, {"x":(1)2, "x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_duplicate_gathering_axes_across_dims(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "x"}}
  %0 = sdy.all_gather [{"y", "x"}, {"x"}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_gather_axes_rank_zero(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding has rank 2 but collective axes has rank 1}}
  %0 = sdy.all_gather [{}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_gather_axes_rank_one(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding has rank 2 but collective axes has rank 1}}
  %0 = sdy.all_gather [{"y", "x"}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_gathering_axis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{can't apply gathering axis "y" to operand sharding on dimension 1}}
  %0 = sdy.all_gather [{}, {"x", "y"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_too_many_gathering_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{can't apply gathering axis "x" to operand sharding on dimension 0}}
  %0 = sdy.all_gather [{"x", "y"}, {}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=4, "y"=2]>

func.func @all_gather_with_incomatible_operand_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{can't apply gathering axis "x" to operand sharding on dimension 1}}
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=8, "y"=2]>

func.func @all_gather_with_incomatible_gathering_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{can't apply gathering axis "x":(2)2 to operand sharding on dimension 1}}
  %0 = sdy.all_gather [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_result_sharding(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding [] on dimension 1}}
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=8, "y"=2]>

func.func @all_gather_with_incompatible_result_sharding_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x":(1)2] on dimension 1}}
  %0 = sdy.all_gather [{}, {"x":(2)4}] %arg0 out_sharding=<@mesh, [{"y"}, {"x":(1)4}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_on_operand_without_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error @+1 {{collective on operand without sharding}}
  %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh2 = <["a"=2, "b"=2]>

// expected-note @+1 {{operand mesh: #sdy.mesh<["a"=2, "b"=2]>}}
func.func @all_slice_with_incompatible_meshes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"a"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result mesh does not match operand mesh}}
  %0 = sdy.all_slice [{}, {"b"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_invalid_out_sharding(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "x"}}
  %0 = sdy.all_slice [{"y"}, {}] %arg0 out_sharding=<@mesh, [{"y"}, {"x", "x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=4, "y"=2]>

func.func @all_slice_gathering_axes_can_be_merged(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{two consecutive sub-axes can be merged: "x":(1)2, "x":(2)2}}
  %0 = sdy.all_slice [{}, {"x":(1)2, "x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_duplicate_gathering_axes_across_dims(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "x"}}
  %0 = sdy.all_slice [{"y", "x"}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_with_incompatible_gather_axes_rank_zero(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding has rank 2 but collective axes has rank 1}}
  %0 = sdy.all_slice [{}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_with_incompatible_gather_axes_rank_one(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding has rank 2 but collective axes has rank 1}}
  %0 = sdy.all_slice [{"y", "x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_with_incompatible_slicing_axis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x", "y"] on dimension 1}}
  %0 = sdy.all_slice [{}, {"x", "y"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_with_too_many_slicing_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x", "y"] on dimension 0}}
  %0 = sdy.all_slice [{"x", "y"}, {}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=4, "y"=2]>

func.func @all_slice_with_incomatible_result_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x"] on dimension 1}}
  %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x":(1)2}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=8, "y"=2]>

func.func @all_slice_with_incomatible_slicing_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x":(2)2] on dimension 1}}
  %0 = sdy.all_slice [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_slice_with_incompatible_result_sharding(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x", "x"] on dimension 1}}
  %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=8, "y"=2]>

func.func @all_slice_with_incompatible_result_sharding_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x":(1)2, "x":(4)2] on dimension 1}}
  %0 = sdy.all_slice [{}, {"x":(4)2}] %arg0 out_sharding=<@mesh, [{"y"}, {"x":(1)4}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh2 = <["a"=2, "b"=2]>

// expected-note @+1 {{operand mesh: #sdy.mesh<["a"=2, "b"=2]>}}
func.func @collective_permute_with_incompatible_meshes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"a"}, {"b"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{result mesh does not match operand mesh}}
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh1, [{"x"}, {"y"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @collective_permute_invalid_out_sharding(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "x"}}
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"y", "x"}, {"x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2, "w"=4]>

func.func @collective_permute_dim_sharded_size_mismatch(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x", "y"}, {"z"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{sharded size of result doesn't match operand on dimension 1: 4 != 2}}
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"y", "x"}, {"w"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4, "w"=4]>

func.func @collective_permute_dim_sharded_size_mismatch_sub_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z":(1)2, "w"}, {"x", "y"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{sharded size of result doesn't match operand on dimension 0: 4 != 8}}
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"w":(2)2, "z":(1)2}, {"y", "x"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=4, "z"=2]>

func.func @collective_permute_sharded_size_mismatch_across_dims(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x", "y"}, {"z"}]>}) -> tensor<16x8xf32> {
  // expected-error @+1 {{sharded size of result doesn't match operand on dimension 0: 2 != 8}}
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"x"}, {"y", "z"}]> :  tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
