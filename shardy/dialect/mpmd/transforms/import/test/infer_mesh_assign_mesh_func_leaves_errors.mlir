// RUN: mpmd_opt %s -mpmd-infer-mesh-assign-mesh-func-leaves -split-input-file -verify-diagnostics

// CHECK-LABEL: func @output_wo_sets
func.func @output_wo_sets() -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<>, mpmd.use_set = #mpmd.meshes_with_origins<>} dense<1.0> : tensor<4x8xf32>
  // expected-error @+1 {{Func output 0 has no mesh to be assigned to.}}
  func.return %0 : tensor<4x8xf32>
}

// -----

// expected-error @+1 {{Callee @unused_callee_input unused input 1 has no mesh to be assigned to.}}
func.func private @unused_callee_input(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32> {mpmd.src_set = #mpmd.meshes_with_origins<>}) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  func.return %arg0 : tensor<4x8xf32>
}

// -----

func.func private @unused_op_with_empty_src_set(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
 // expected-error @+1 {{src_set must not be empty for this op.}}
  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<>, mpmd.use_set = #mpmd.meshes_with_origins<>} dense<1.0> : tensor<4x8xf32>
  func.return %arg0 : tensor<4x8xf32>
}
