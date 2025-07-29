// RUN: mpmd_opt %s -mpmd-infer-mesh-validate-no-additional-transfers-needed='error-limit=1' -verify-diagnostics

// Although this function has two errors, the error limit was set to one.
// Therefore, we only emit one error per test.
// NOTE: If we add more tests, we need to split the file, each test resets the error count to zero.

func.func @only_one_conflict_reported(%arg0: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  // expected-error @+1 {{Mesh assignment is not possible for op as it is used in {m1} but it can only be placed on {m2,m3}}}
  %1 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}
