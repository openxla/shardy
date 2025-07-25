// RUN: mpmd_opt %s -mpmd-infer-mesh-validate-no-additional-transfers-needed='error-limit=-1' -verify-diagnostics -split-input-file

func.func @use_set_not_contained_in_src_set_at_all(%arg0: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // expected-error @+1 {{Mesh assignment is not possible for op as it is used in {m1} but it can only be placed on {m2,m3}}}
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>

  func.return %0 : tensor<4x8xf32>
}

// -----

func.func @use_set_not_contained_in_src_set_partially(%arg0: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // expected-error @+1 {{Mesh assignment is not possible for op as it is used in {m1,m2} but it can only be placed on {m2,m3}}}
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>

  func.return %0 : tensor<4x8xf32>
}


// -----

func.func @use_set_not_contained_in_src_set_on_assign_op(%arg0: tensor<4x8xf32>)
  -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // expected-error @+1 {{Mesh assignment is not possible for op as it is used in {m1} but it can only be placed on {m2}}}
  %0 = mpmd.assign {mpmd.src_set = #mpmd.meshes_with_origins<"m2">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

  func.return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// -----

func.func public @call_op_failure(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
  %arg1: !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) -> tensor<3x5xf32> loc("x")
  %1 = mpmd.call @f(%0, %0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  %2 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<3x5xf32>>) -> tensor<3x5xf32> loc("y")
  %a2 = stablehlo.add %2, %2 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<3x5xf32>
  %3 = mpmd.call @f(%a2, %a2) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %1, %3 : tensor<3x5xf32>, tensor<3x5xf32>
}

// Only one error is reported on the signature, since the error for arg1 is emitted at %0.
// expected-error-re @+1 {{Mesh assignment is not possible for arg0 of mpmd.call "f" as it is used in {m1,m2} but it can only be placed on {m2,m3}{{.*}}Input "y": m2{{.*}}Input "x": m1}}
func.func private @f(
  %arg0: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
  %arg1: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {

  // expected-error-re @+1 {{Mesh assignment is not possible for op as it is used in {m1,m2} but it can only be placed on {m2,m3}{{.*}}Input "y": m2{{.*}}Input "x": m1}}
  %0 = stablehlo.add %arg1, %arg1 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<3x5xf32>

  return %arg0 : tensor<3x5xf32>
}

// -----

func.func public @chained_call_op_failure(
  %arg0: tensor<3x5xf32>,
  %arg1: tensor<3x5xf32>
) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  %c0:2 = mpmd.call @f(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  %c1:2 = mpmd.call @f(%c0#0, %c0#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  return %c1#0, %c1#1 : tensor<3x5xf32>, tensor<3x5xf32>
}

// Only one error is reported on the signature, since the error for arg1 is emitted at %0.
// expected-error @+2 {{Mesh assignment is not possible for mpmd.call "f" as it passes result 0 to arg 0 but they have mismatching mesh assignments: res: {m1[c]}, arg: {m2[a]}}}
// expected-error @+1 {{Mesh assignment is not possible for mpmd.call "f" as it passes result 1 to arg 1 but they have mismatching mesh assignments: res: {m2[d]}, arg: {m1[b]}}}
func.func private @f(
  %arg0: tensor<3x5xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2"["a"]>},
  %arg1: tensor<3x5xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1"["b"]>}
) -> (tensor<3x5xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1"["c"]>}, tensor<3x5xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2"["d"]>})
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {

  return %arg0, %arg1 : tensor<3x5xf32>, tensor<3x5xf32>
}

// -----

module @test_printing_with_locs_input_and_named_computation {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%arg0: !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>> loc("param[0]")) -> (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>)
    attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
    %2 = mpmd.unassign {origin = "user_in", mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["user_in"]>} %arg0 : (!mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>) -> tensor<3x5xf32> loc("param[0]")
    // expected-error-re @+1 {{Mesh assignment is not possible for op{{.*}}Input "param[0]": mesh1[user_in]{{.*}}named_computation "stage2": mesh2[stage2]}}
    %5 = stablehlo.sqrt %2 {mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["user_in"]>, mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>} : tensor<3x5xf32>
    %6 = mpmd.assign {origin = "stage2", mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>} %5 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>  loc("named_computation")
    %7 = mpmd.fragment<mesh="mesh2", origin=["stage2"]> (%6) (%arg1: tensor<3x5xf32>) {
      %11 = stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
      mpmd.return %11 : tensor<3x5xf32>
    } : (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
    return %7 : !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
  }
}

// -----

module @test_printing_with_locs_named_tensor_and_output {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%arg0: !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>) -> (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>)
    attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
    %2 = mpmd.unassign {origin = "stage1", mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["stage1"]>} %arg0 : (!mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>) -> tensor<3x5xf32> loc("named_tensor")
    // expected-error-re @+1 {{Mesh assignment is not possible for op{{.*}}named_tensor "stage1": mesh1[stage1]{{.*}}Output "result[0]": mesh2[user_out]}}
    %5 = stablehlo.sqrt %2 {mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["stage1"]>, mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["user_out"]>} : tensor<3x5xf32>
    %6 = mpmd.assign {origin = "user_out", mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["user_out"]>} %5 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>  loc("result[0]")
    return %6 : !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
  }
}


// -----

module @test_printing_with_locs {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%arg0: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["inferred_in"]>, mpmd.use_set = #mpmd.meshes_with_origins<"mesh1"["stage1"]>}) -> (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>> {jax.result_info = "result"}) attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
    %0 = mpmd.assign {origin = "stage1", mpmd.use_set = #mpmd.meshes_with_origins<"mesh1"["stage1"]>} %arg0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>
    %1 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%0) (%arg1: tensor<3x5xf32>) {
      %11 = stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
      mpmd.return %11 : tensor<3x5xf32>
    } : (!mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>
    %2 = mpmd.unassign {origin = "stage1", mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["stage1"]>} %1 : (!mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>) -> tensor<3x5xf32> loc("named_computation")
    // expected-error-re @+1 {{Mesh assignment is not possible for op{{.*}}named_computation "stage1": mesh1[stage1]{{.*}}named_computation "stage2": mesh2[stage2]}}
    %5 = stablehlo.sqrt %2 {mpmd.src_set = #mpmd.meshes_with_origins<"mesh1"["stage1"]>, mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>} : tensor<3x5xf32>
    %6 = mpmd.assign {origin = "stage2", mpmd.use_set = #mpmd.meshes_with_origins<"mesh2"["stage2"]>} %5 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>  loc("named_computation")
    %7 = mpmd.fragment<mesh="mesh2", origin=["stage2"]> (%6) (%arg1: tensor<3x5xf32>) {
      %11 = stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
      mpmd.return %11 : tensor<3x5xf32>
    } : (!mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>) -> !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
    return %7 : !mpmd.mesh_tensor<"mesh2", tensor<3x5xf32>>
  }
}
