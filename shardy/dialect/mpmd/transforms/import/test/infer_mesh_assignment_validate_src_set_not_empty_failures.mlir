// RUN: mpmd_opt %s -mpmd-infer-mesh-validate-src-set-not-empty='error-limit=-1' -verify-diagnostics -split-input-file

func.func @multiple_ops_have_empty_src_set(%arg0: tensor<4x8xf32> loc("x"))
  -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // expected-error @+1 {{Mesh assignment is not possible for op as its operands are on conflicting meshes}}
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<>} : tensor<4x8xf32>
  %1 = stablehlo.add %0, %0 {mpmd.src_set = #mpmd.meshes_with_origins<>} : tensor<4x8xf32> // No error since we already errored on the operands.

  // expected-error @+1 {{Mesh assignment is not possible for op as its operands are on conflicting meshes}}
  %2 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<>} : tensor<4x8xf32>
  func.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

func.func public @callee_args_have_empty_src_set(%arg0: tensor<3x5xf32> loc("x")
  ) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // expected-error-re @+1 {{Mesh assignment is not possible for op as its operands are on conflicting meshes{{.*}}Input "x"}}
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<>} : tensor<3x5xf32>

  %m1 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m1">} : tensor<3x5xf32>
  %m2 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2">} : tensor<3x5xf32>

  %2:2 = mpmd.call @shardy_mpmdf(%m1, %0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  %3:2 = mpmd.call @shardy_mpmdf(%m2, %0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  return %2#0, %3#0 : tensor<3x5xf32>, tensor<3x5xf32>
}

// No error on arg since the caller has empty src_set
// expected-error @+1 {{Mesh assignment is not possible for arg0 of mpmd.call "f" }}
func.func private @shardy_mpmdf(%arg0: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<>}, %arg1: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<>}) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
  return %arg0 : tensor<3x5xf32>
}

// -----

#loc1 = loc("_named_computation.<locals>.wrapped_fn")
module @test_printing_with_locs_through_call_ops {
  func.func public @main(%arg0: !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>> loc("x"), %arg1: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"mesh2"["inferred_in"]>, mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>}) -> (tensor<3x5xf32> {jax.result_info = "result"}) attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
    %0 = mpmd.unassign {origin = "user_in", mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["user_in"]>} %arg0 : (!mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>) -> tensor<3x5xf32> loc("x")
    %1 = mpmd.call @shardy_mpmdf(%0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }
  func.func private @shardy_mpmdf(%arg0: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["user_in"]>}, %arg1: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"mesh2"["inferred_in"]>, mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>}) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
    %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["user_in"]>} : tensor<3x5xf32>
    %1 = mpmd.assign {origin = "stage2", mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>} %arg1 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
    %2 = mpmd.fragment<mesh="mesh2", origin=["stage2"]> (%1) (%arg2: tensor<3x5xf32>) {
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x5xf32>
      %5 = stablehlo.add %arg2, %cst : tensor<3x5xf32>
      mpmd.return %5 : tensor<3x5xf32>
    } : (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
    %3 = mpmd.unassign {origin = "stage2", mpmd.src_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>} %2 : (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>) -> tensor<3x5xf32> loc(#loc1)
    // expected-error-re @+1 {{Mesh assignment is not possible for op as its operands{{.*}}Input "x": mesh1[user_in]{{.*}}named_computation "stage2": mesh2[stage2]}}
    %4 = stablehlo.divide %0, %3 {mpmd.src_set = #mpmd.meshes_with_origins<>} : tensor<3x5xf32>
    return %4 : tensor<3x5xf32>
  }
}
