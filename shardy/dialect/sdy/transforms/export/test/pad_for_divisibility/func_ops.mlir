// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file -verify-diagnostics

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// expected-error @+2  {{argument #0 has a non-divisible sharding}}
// expected-error @+1  {{failed to legalize operation 'func.func'}}
func.func @indivisible_input(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> tensor<7x8xf32> {
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// expected-error @+2 {{result #0 has a non-divisible sharding}}
// expected-error @+1 {{failed to legalize operation 'func.func'}}
func.func @indivisible_output(
  %arg0: tensor<7x8xf32>)
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}
