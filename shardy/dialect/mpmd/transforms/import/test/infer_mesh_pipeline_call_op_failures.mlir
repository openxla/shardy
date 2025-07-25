// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline -split-input-file -verify-diagnostics

module {

  // The sources of an edge are produced by different unassigns from different
  // meshes.
  func.func public @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
                         %arg1: !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
    attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
  {
    %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) -> tensor<3x5xf32>
    %1 = mpmd.call @f(%0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %2 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<3x5xf32>>) -> tensor<3x5xf32>
    %3 = mpmd.call @f(%2) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1, %3 : tensor<3x5xf32>, tensor<3x5xf32>
  }

  // expected-error @+1 {{Mesh assignment is not possible for arg0 of mpmd.call "f" }}
  func.func private @f(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
    return %arg0 : tensor<3x5xf32>
  }
}

// -----

#topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>, <"m3" : <["x"=1]>>>
module {

  // The sources of an edge are produced by different unassigns from different
  // meshes.
  func.func public @chained_calls_failure(
      %arg0: tensor<3x5xf32>,
      %arg1: tensor<3x5xf32>,
      %arg2: tensor<3x5xf32>
    ) -> (
    !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
    !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>,
    !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
  )
    attributes {topology = #topology}
  {
    // arg use: m3, m1, m2 - because of broadcast
    // res use: m1+m3, m1+m2, m2+m3
    %1:3 = mpmd.call @f(%arg0, %arg1, %arg2) : (
      tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
    ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)

    // arg use: m3, m1, m2 - because of broadcast
    // res use: m1, m2, m3
    %3:3 = mpmd.call @f(%1#0, %1#1, %1#2) : (
      tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
    ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
    %4 = mpmd.assign %3#0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
    %5 = mpmd.assign %3#1 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
    %6 = mpmd.assign %3#2 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
    return %4, %5, %6 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
  }

  // expected-error @+1 {{Mesh assignment is not possible for mpmd.call "f"}}
  func.func private @f(
    %arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>, %arg2: tensor<3x5xf32>
  ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>) attributes
  {topology = #topology} {
    %10 = mpmd.broadcast %arg0 : tensor<3x5xf32>
    %11 = mpmd.broadcast %arg1 : tensor<3x5xf32>
    %12 = mpmd.broadcast %arg2 : tensor<3x5xf32>
    return %11, %12, %10 : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
  }
}

// -----

#topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>, <"m3" : <["x"=1]>>>
module {

  // The sources of an edge are produced by different unassigns from different
  // meshes.
  func.func public @chained_calls_no_failure_because_different_callees(
      %arg0: tensor<3x5xf32>,
      %arg1: tensor<3x5xf32>,
      %arg2: tensor<3x5xf32>
    ) -> (
    !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
    !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>,
    !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
  )
    attributes {topology = #topology}
  {
    %1:3 = mpmd.call @f(%arg0, %arg1, %arg2) : (
      tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
    ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)

    %3:3 = mpmd.call @f_copy(%1#0, %1#1, %1#2) : (
      tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
    ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
    %4 = mpmd.assign %3#0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
    %5 = mpmd.assign %3#1 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
    %6 = mpmd.assign %3#2 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
    return %4, %5, %6 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
  }

  func.func private @f(
    %arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>, %arg2: tensor<3x5xf32>
  ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>) attributes
  {topology = #topology} {
    %10 = mpmd.broadcast %arg0 : tensor<3x5xf32>
    %11 = mpmd.broadcast %arg1 : tensor<3x5xf32>
    %12 = mpmd.broadcast %arg2 : tensor<3x5xf32>
    return %11, %12, %10 : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
  }

  func.func private @f_copy(
    %arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>, %arg2: tensor<3x5xf32>
  ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>) attributes
  {topology = #topology} {
    %10 = mpmd.broadcast %arg0 : tensor<3x5xf32>
    %11 = mpmd.broadcast %arg1 : tensor<3x5xf32>
    %12 = mpmd.broadcast %arg2 : tensor<3x5xf32>
    return %11, %12, %10 : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
  }
}

