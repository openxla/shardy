// RUN: mpmd_opt %s -mpmd-import-pipeline='name-to-mesh-assignment=f1@m1,f2@m2' -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: sdy.mesh @mesh = <["x"=2]>
#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// named_computation_duplicate_args_simplified
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#topology} {
// CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-NEXT: return %[[FRAG]]
  %1 = mpmd.named_computation<"f1"> (%arg0, %arg0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["x"=2]>
#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// named_computation_duplicate_and_noop_results_simplified
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) attributes {
    "topology"=#topology} {
// CHECK-NEXT: %[[SIMPLIFIED_FRAG:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg2
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK:     %[[HELPER_FRAG:.*]]:3 = mpmd.fragment<mesh="m2", origin=["f2"]>
// CHECK:     return %[[HELPER_FRAG]]#0, %[[HELPER_FRAG]]#1, %[[HELPER_FRAG]]#2
  %1:3 = mpmd.named_computation<"f1"> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %10, %10, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  // This named_computation f2 will not be simplified, but makes it easier to
  // see that f1 is simplified.
  %2:3 = mpmd.named_computation<"f2"> (%1#0, %1#1, %1#2) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    %11 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    %12 = stablehlo.add %arg4, %arg4 : tensor<4x8xf32>
    mpmd.return %10, %11, %12 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)

  func.return %2#0, %2#1, %2#2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["x"=2]>
#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// noop_named_computation_args_removed_before_mesh_assignment
// CHECK-LABEL: func @main
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>)  attributes {
    "topology"=#topology} {
// CHECK-NEXT: %[[FRAG1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD1:.*]] = stablehlo.add %arg2, %arg2
// CHECK-NEXT:   mpmd.return %[[ADD1]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[FRAG2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]> (%arg1) (%arg2: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add %arg2, %arg2
// CHECK-NEXT:   mpmd.return %[[ADD2]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAG1]], %[[FRAG2]]
  %0:2 = mpmd.named_computation<"f1"> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %2, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %1 = mpmd.named_computation<"f2"> (%0#1) (%arg2: tensor<4x8xf32>) {
    %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %2 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0#0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// mesh_inference_succeeds_through_returned_callee_arg
// CHECK-LABEL: func.func public @main
func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>) attributes {topology=#topology}
{
  // Callee arg in position 1 is returned and unused, and inference handles this
  // fine.
  %2:2 = mpmd.call @f(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)

  %1 = mpmd.named_computation<"f1"> (%2#0) (%arg3: tensor<3x5xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<3x5xf32>
    mpmd.return %10 : tensor<3x5xf32>
  } : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %1 : tensor<3x5xf32>
}

func.func private @f(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  return %0, %arg1 : tensor<3x5xf32>, tensor<3x5xf32>
}
// No error.

// -----
// CHECK-LABEL: sdy.mesh @mesh = <["x"=2]>
#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// Do not CSE on custom calls with no_cse attribute. It should also add side
// effect attribute to the custom call.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#topology} {
// CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1
// CHECK-NEXT:   %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @Sharding
// CHECK-SAME:  {has_side_effect = true, mhlo.no_cse}
// CHECK-NEXT:   %[[CUSTOM_CALL_2:.*]] = stablehlo.custom_call @Sharding
// CHECK-SAME:  {has_side_effect = true, mhlo.no_cse}
  %1:2 = mpmd.named_computation<"f1"> (%arg0, %arg0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %2 = stablehlo.custom_call @Sharding(%arg3) {mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %3 = stablehlo.custom_call @Sharding(%arg4) {mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %2, %3 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %1#0 : tensor<4x8xf32>
}
