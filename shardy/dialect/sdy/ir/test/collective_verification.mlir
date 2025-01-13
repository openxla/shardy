// RUN: sdy_opt %s -split-input-file -verify-diagnostics

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_on_operand_without_sharding(%arg0 : tensor<16x2xf32>) -> tensor<16x2xf32> {
  // expected-error @+1 {{gathering on operand without sharding}}
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh2 = <["a"=2, "b"=2]>

// expected-note @+1 {{operand mesh: #sdy.mesh<["a"=2, "b"=2]>}}
func.func @all_gather_with_incompatible_meshes(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"a"}, {"b"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{result mesh does not match operand mesh}}
  %0 = sdy.all_gather [{}, {"b"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_gather_axes_rank_zero(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{result sharding has rank 2 but gathering axes has rank 1}}
  %0 = sdy.all_gather [{}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_gather_axes_rank_one(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{result sharding has rank 2 but gathering axes has rank 1}}
  %0 = sdy.all_gather [{"y", "x"}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_gathering_axis(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{can't apply gathering axis "y" to operand sharding on dimension 1}}
  %0 = sdy.all_gather [{}, {"x", "y"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_too_many_gathering_axes(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{can't apply gathering axis "x" to operand sharding on dimension 0}}
  %0 = sdy.all_gather [{"x", "y"}, {}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=4, "y"=2]>

func.func @all_gather_with_incomatible_operand_subaxis(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{can't apply gathering axis "x" to operand sharding on dimension 1}}
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=8, "y"=2]>

func.func @all_gather_with_incomatible_gathering_subaxis(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{can't apply gathering axis "x":(2)2 to operand sharding on dimension 1}}
  %0 = sdy.all_gather [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @all_gather_with_incompatible_result_sharding(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding [] on dimension 1}}
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// -----

sdy.mesh @mesh = <["x"=8, "y"=2]>

func.func @all_gather_with_incompatible_result_sharding_subaxis(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // expected-error @+1 {{result sharding doesn't match expected sharding ["x":(1)2] on dimension 1}}
  %0 = sdy.all_gather [{}, {"x":(2)4}] %arg0 out_sharding=<@mesh, [{"y"}, {"x":(1)4}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}
