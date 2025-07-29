// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline -split-input-file 2>&1 | FileCheck %s

!mesh1_3_5 = !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>
!mesh1_10_3 = !mpmd.mesh_tensor<"mesh1", tensor<10x3xf32>>
!mesh1_10_5 = !mpmd.mesh_tensor<"mesh1", tensor<10x5xf32>>
!mesh2_10_5 = !mpmd.mesh_tensor<"mesh2", tensor<10x5xf32>>
!mesh2_10_7 = !mpmd.mesh_tensor<"mesh2", tensor<10x7xf32>>
!mesh2_5_7 = !mpmd.mesh_tensor<"mesh2", tensor<5x7xf32>>


// Tests that assigns and unassigns in a non-main function will be propagated
// to its call sites.

// CHECK-LABEL: func.func public @main(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>,
// CHECK-SAME:   %arg1: !mpmd.mesh_tensor<"mesh2", tensor<5x7xf32>>,
// CHECK-SAME:   %arg2: !mpmd.mesh_tensor<"mesh1", tensor<10x3xf32>>) ->
// CHECK-SAME: (!mpmd.mesh_tensor<"mesh2", tensor<10x7xf32>>, !mpmd.mesh_tensor<"mesh2", tensor<10x7xf32>>)
func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<5x7xf32>, %arg2: tensor<10x3xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>) attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["y"=1]>>>} {
  // CHECK-NEXT: mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (!mpmd.mesh_tensor<"mesh1", {{.*}}>, !mpmd.mesh_tensor<"mesh2", {{.*}}>, !mpmd.mesh_tensor<"mesh1", {{.*}}>) -> !mpmd.mesh_tensor<"mesh2", {{.*}}>
  %0 = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (tensor<3x5xf32>, tensor<5x7xf32>, tensor<10x3xf32>) -> tensor<10x7xf32>
  // CHECK-NEXT: mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (!mpmd.mesh_tensor<"mesh1", {{.*}}>, !mpmd.mesh_tensor<"mesh2", {{.*}}>, !mpmd.mesh_tensor<"mesh1", {{.*}}>) -> !mpmd.mesh_tensor<"mesh2", {{.*}}>
  %1 = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (tensor<3x5xf32>, tensor<5x7xf32>, tensor<10x3xf32>) -> tensor<10x7xf32>
  // CHECK-NEXT: return
  return %0, %1 : tensor<10x7xf32>, tensor<10x7xf32>
}

// CHECK-LABEL: func.func private @shardy_mpmd_mlp(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>,
// CHECK-SAME:   %arg1: !mpmd.mesh_tensor<"mesh2", tensor<5x7xf32>>,
// CHECK-SAME:   %arg2: !mpmd.mesh_tensor<"mesh1", tensor<10x3xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"mesh2", tensor<10x7xf32>>
func.func private @shardy_mpmd_mlp(%arg0: tensor<3x5xf32>,
                                    %arg1: tensor<5x7xf32>,
                                    %arg2: tensor<10x3xf32>
) -> tensor<10x7xf32>
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["y"=1]>>>}
{
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg0, %arg2)
  // CHECK-NEXT:    dot_general
  // CHECK-NEXT:    mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"mesh1", {{.*}}>, !mpmd.mesh_tensor<"mesh1", {{.*}}>) -> !mpmd.mesh_tensor<"mesh1", {{.*}}>
  // CHECK-NEXT: %[[T:.*]] = mpmd.transfer %[[F1]] : (!mpmd.mesh_tensor<"mesh1", {{.*}}>) -> !mpmd.mesh_tensor<"mesh2", {{.*}}>
  // CHECK-NEXT: %[[F2:.*]] = mpmd.fragment<mesh="mesh2", origin=["stage2"]> (%arg1, %[[T]]) (%arg3: tensor<5x7xf32>, %arg4: tensor<10x5xf32>) {
  // CHECK-NEXT:    dot_general
  // CHECK-NEXT:    mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"mesh2", {{.*}}>, !mpmd.mesh_tensor<"mesh2", {{.*}}>) -> !mpmd.mesh_tensor<"mesh2", {{.*}}>
  // CHECK-NEXT: return %[[F2]] : !mpmd.mesh_tensor<"mesh2", {{.*}}>

  %0 = mpmd.assign %arg0 : (tensor<3x5xf32>) -> !mesh1_3_5
  %1 = mpmd.assign %arg2 : (tensor<10x3xf32>) -> !mesh1_10_3
  %2 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%0, %1) (%arg3: tensor<3x5xf32>, %arg4: tensor<10x3xf32>) {
    %7 = stablehlo.dot_general %arg4, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x3xf32>, tensor<3x5xf32>) -> tensor<10x5xf32>
    mpmd.return %7 : tensor<10x5xf32>
  } : (!mesh1_3_5, !mesh1_10_3) -> !mesh1_10_5
  %3 = mpmd.assign %arg1 : (tensor<5x7xf32>) -> !mesh2_5_7
  %4 = mpmd.transfer %2 : (!mesh1_10_5) -> !mesh2_10_5
  %5 = mpmd.fragment<mesh="mesh2", origin=["stage2"]> (%3, %4) (%arg3: tensor<5x7xf32>, %arg4: tensor<10x5xf32>) {
    %7 = stablehlo.dot_general %arg4, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
    mpmd.return %7 : tensor<10x7xf32>
  } : (!mesh2_5_7, !mesh2_10_5) -> !mesh2_10_7
  %6 = mpmd.unassign %5 : (!mesh2_10_7) -> tensor<10x7xf32>
  return %6 : tensor<10x7xf32>
}

// -----

// Tests that unassigns at a call-site are propagated to the body of the
// called function.

// CHECK-LABEL: func.func public @main(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
// CHECK-SAME:   %arg1: !mpmd.mesh_tensor<"m1", tensor<5x7xf32>>,
// CHECK-SAME:   %arg2: !mpmd.mesh_tensor<"m1", tensor<10x3xf32>>) ->
// CHECK-SAME:  (!mpmd.mesh_tensor<"m1", tensor<10x7xf32>>, !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>)
func.func public @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
                        %arg1: !mpmd.mesh_tensor<"m1", tensor<5x7xf32>>,
                        %arg2: !mpmd.mesh_tensor<"m1", tensor<10x3xf32>>
) -> (tensor<10x7xf32>, tensor<10x7xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[C1:.*]] = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (!mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: %[[C2:.*]] = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (!mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: return %[[C1]], %[[C2]] : !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>
  %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) -> tensor<3x5xf32>
  %1 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m1", tensor<5x7xf32>>) -> tensor<5x7xf32>
  %2 = mpmd.unassign %arg2 : (!mpmd.mesh_tensor<"m1", tensor<10x3xf32>>) -> tensor<10x3xf32>
  %3 = mpmd.call @shardy_mpmd_mlp(%0, %1, %2) : (tensor<3x5xf32>, tensor<5x7xf32>, tensor<10x3xf32>) -> tensor<10x7xf32>
  %4 = mpmd.call @shardy_mpmd_mlp(%0, %1, %2) : (tensor<3x5xf32>, tensor<5x7xf32>, tensor<10x3xf32>) -> tensor<10x7xf32>
  return %3, %4 : tensor<10x7xf32>, tensor<10x7xf32>
}

// CHECK-LABEL: func.func private @shardy_mpmd_mlp(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
// CHECK-SAME:   %arg1: !mpmd.mesh_tensor<"m1", tensor<5x7xf32>>,
// CHECK-SAME:   %arg2: !mpmd.mesh_tensor<"m1", tensor<10x3xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>

func.func private @shardy_mpmd_mlp(%arg0: tensor<3x5xf32>, %arg1: tensor<5x7xf32>, %arg2: tensor<10x3xf32>) -> tensor<10x7xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg1, %arg2, %arg0)
  // CHECK-NEXT:    dot_general
  // CHECK-NEXT:    dot_general
  // CHECK-NEXT:    mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: return %[[F1]] : !mpmd.mesh_tensor<"m1", {{.*}}>
  %0 = stablehlo.dot_general %arg2, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x3xf32>, tensor<3x5xf32>) -> tensor<10x5xf32>
  %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
  return %1 : tensor<10x7xf32>
}

// -----

// Tests that assigns at a call-site are propagated to the body of the
// called function.

// CHECK-LABEL: func.func public @main(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
// CHECK-SAME:   %arg1: !mpmd.mesh_tensor<"m1", tensor<5x7xf32>>,
// CHECK-SAME:   %arg2: !mpmd.mesh_tensor<"m1", tensor<10x3xf32>>) ->
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<10x7xf32>>, !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>)
func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<5x7xf32>, %arg2: tensor<10x3xf32>) -> (!mpmd.mesh_tensor<"m1", tensor<10x7xf32>>, !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>) attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK-NEXT: %[[C1:.*]] = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (!mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: %[[C2:.*]] = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (!mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: return %[[C1]], %[[C2]] : !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>
  %0 = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (tensor<3x5xf32>, tensor<5x7xf32>, tensor<10x3xf32>) -> tensor<10x7xf32>
  %1 = mpmd.call @shardy_mpmd_mlp(%arg0, %arg1, %arg2) : (tensor<3x5xf32>, tensor<5x7xf32>, tensor<10x3xf32>) -> tensor<10x7xf32>
  %2 = mpmd.assign %0 : (tensor<10x7xf32>) -> !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>
  %3 = mpmd.assign %1 : (tensor<10x7xf32>) -> !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>
  return %2, %3 : !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>, !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>
}

// CHECK-LABEL: func.func private @shardy_mpmd_mlp(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
// CHECK-SAME:   %arg1: !mpmd.mesh_tensor<"m1", tensor<5x7xf32>>,
// CHECK-SAME:   %arg2: !mpmd.mesh_tensor<"m1", tensor<10x3xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<10x7xf32>>
func.func private @shardy_mpmd_mlp(%arg0: tensor<3x5xf32>, %arg1: tensor<5x7xf32>, %arg2: tensor<10x3xf32>) -> tensor<10x7xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg1, %arg2, %arg0)
  // CHECK-NEXT:    dot_general
  // CHECK-NEXT:    dot_general
  // CHECK-NEXT:    mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: return %[[F1]] : !mpmd.mesh_tensor<"m1", {{.*}}>
  %0 = stablehlo.dot_general %arg2, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x3xf32>, tensor<3x5xf32>) -> tensor<10x5xf32>
  %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
  return %1 : tensor<10x7xf32>
}

// -----

// All targets are used by an assign_op, but one of them is assigned multiple
// times to the same mesh. This does not prevent inference though.

// CHECK-LABEL: func.func public @main(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) ->
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>)
func.func public @main(%arg0: tensor<3x5xf32>)
  ->
  (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%arg0) : (!mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m1", {{.*}}>
  // CHECK-NEXT: return %[[CALL]], %[[CALL]] : !mpmd.mesh_tensor<"m1", {{.*}}>, !mpmd.mesh_tensor<"m1", {{.*}}>
  %0 = mpmd.call @f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %1 = mpmd.assign %0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
  %2 = mpmd.assign %0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
  return %1, %2 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
}

// CHECK-LABEL: func.func private @f(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m1", {{.*}}>
func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK-NEXT: return %arg0 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
  return %arg0 : tensor<3x5xf32>
}

// -----

// Tests the case in which a transfer prevents an unassign to be pushed
// forward.

!mesh1_t = !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
!mesh2_t = !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>

// CHECK-LABEL: func.func public @main(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) ->
// CHECK-SAME: (!mpmd.mesh_tensor<"m2", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>)
func.func public @main(%arg0: !mesh1_t) -> (!mesh2_t, !mesh2_t)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", {{.*}}>) -> !mpmd.mesh_tensor<"m2", {{.*}}>
  // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%[[TRANSFER]]) : (!mpmd.mesh_tensor<"m2", {{.*}}>) -> !mpmd.mesh_tensor<"m2", {{.*}}>
  // CHECK-NEXT: return %[[CALL]], %[[TRANSFER]]
  %0 = mpmd.unassign %arg0 : (!mesh1_t) -> tensor<3x5xf32>
  %1 = mpmd.transfer %arg0 : (!mesh1_t) -> !mesh2_t
  %2 = mpmd.call @f(%0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %3 = mpmd.assign %2 : (tensor<3x5xf32>) -> !mesh2_t
  return %3, %1 : !mesh2_t, !mesh2_t
}

// CHECK-LABEL: func.func private @f(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>) ->
// CHECK-SAME: !mpmd.mesh_tensor<"m2", {{.*}}>
func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> attributes
  {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
  return %arg0 : tensor<3x5xf32>
}

// -----

!mesh2_t = !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>

// Not all targets are used by an assign_op. Even so, mesh inference will
// succeed and we propagate "sideways", from one target (`%2`) to the other
// (`%0`).
// CHECK-LABEL: func.func public @main(
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>) ->
// CHECK-SAME: (!mpmd.mesh_tensor<"m2", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>)
func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  ->
  (tensor<3x5xf32>, !mesh2_t)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[C0:.*]] = mpmd.call @f(%arg0) : (!mpmd.mesh_tensor<"m2", {{.*}}) -> !mpmd.mesh_tensor<"m2", {{.*}}>
  %0 = mpmd.call @f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: %[[C1:.*]] = mpmd.call @f(%arg1) : (!mpmd.mesh_tensor<"m2", {{.*}}>) -> !mpmd.mesh_tensor<"m2", {{.*}}>
  %2 = mpmd.call @f(%arg1) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %3 = mpmd.assign %2 : (tensor<3x5xf32>) -> !mesh2_t
  // CHECK-NEXT: return %[[C0]], %[[C1]]
  return %0, %3 : tensor<3x5xf32>, !mesh2_t
}

func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> attributes
  {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
  return %arg0 : tensor<3x5xf32>
}

// -----

!mesh2_t = !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>

// @f has an unused result and an unused op.
// CHECK-LABEL: func.func public @unused_in_callee(%arg0: !mpmd.mesh_tensor<"m2"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
// CHECK-SAME: -> ({{.*}}"m2"{{.*}}, {{.*}}"m2"
func.func public @unused_in_callee(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>, !mesh2_t)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[C0:.*]]:2 = mpmd.call @f(%arg0, %arg1, %arg1)
  %0:2 = mpmd.call @f(%arg0, %arg1, %arg1) : (tensor<3x5xf32>,
    tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  %3 = mpmd.assign %0#0 : (tensor<3x5xf32>) -> !mesh2_t
  // CHECK-NEXT: return %[[C0]]#0, %[[C0]]#0
  return %0#0, %3 : tensor<3x5xf32>, !mesh2_t
}

// CHECK-LABEL: func.func private @f(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m2"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1"
// CHECK-SAME: -> ({{.*}}"m2"{{.*}}, {{.*}}"m2"
func.func private @f(%used: tensor<3x5xf32>, %unused_arg: tensor<3x5xf32>,
    %unused_res_arg: tensor<3x5xf32>
  ) -> (tensor<3x5xf32>, tensor<3x5xf32>) attributes
  {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
  // CHECK-NEXT: %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2"
  // CHECK-NEXT:    stablehlo.add
  // CHECK-NEXT: mpmd.return

  // CHECK:      %[[ADD_FRAG2:.*]] = mpmd.fragment<mesh="m1"
  // CHECK-NEXT:    stablehlo.add
  // CHECK-NEXT: mpmd.return
  // CHECK: return %[[ADD_FRAG1]], %[[ADD_FRAG2]]
  %unused = stablehlo.add %unused_arg, %unused_arg : tensor<3x5xf32>
  %used_res = stablehlo.add %used, %used : tensor<3x5xf32>
  %unused_res = stablehlo.add %unused_res_arg, %unused_res_arg : tensor<3x5xf32>
  return %used_res, %unused_res : tensor<3x5xf32>, tensor<3x5xf32>
}

// -----

module {
  // Not all sources of an op are produced by an unassign but this is handled
  // by inference.
  // CHECK-LABEL: func.func public @multiple_calls_on_same_func(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME: -> ({{.*}}"m1"{{.*}}, {{.*}}"m1"
  func.func public @multiple_calls_on_same_func(%arg0: !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>,
                         %arg1: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
    attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
  {
    // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%arg0)
    // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%arg1)
    %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>) -> tensor<3x5xf32>
    %1 = mpmd.call @f(%0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %3 = mpmd.call @f(%arg1) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1, %3 : tensor<3x5xf32>, tensor<3x5xf32>
  }

  // CHECK-LABEL: func.func private @f(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME: -> {{.*}}"m1"{{.*}}
  func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["y"=1]>>>} {
    return %arg0 : tensor<3x5xf32>
  }
}

// -----

module {
  // CHECK-LABEL: func.func public @calls_assigned_to_different_meshes(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME: -> ({{.*}}"m1"{{.*}}, {{.*}}"m2"
  func.func public @calls_assigned_to_different_meshes(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
    ->
    (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>)
    attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
  {
    // CHECK-NEXT: %[[TRANSFER0:.*]] = mpmd.transfer %arg0
    // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%arg0, %[[TRANSFER0]])
    // CHECK-NEXT: %[[TRANSFER1:.*]] = mpmd.transfer %arg1
    // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%arg1, %[[TRANSFER1]])
    %0 = mpmd.call @f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %1 = mpmd.assign %0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
    %2 = mpmd.call @f(%arg1) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %3 = mpmd.assign %2 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
    return %1, %3 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
  }

  // `f` has args cloned to be assigned to multiple meshes.
  // CHECK-LABEL: func.func private @f(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2"
  // CHECK-SAME: -> ({{.*}}"m1"{{.*}}, {{.*}}"m2"
  func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
    return %arg0 : tensor<3x5xf32>
  }
}

// -----

module {

  // CHECK-LABEL: func.func public @call_assigned_to_different_meshes(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME: -> ({{.*}}"m1"{{.*}}, {{.*}}"m2"
  func.func public @call_assigned_to_different_meshes(%arg0: tensor<3x5xf32>)
    ->
    (!mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>)
    attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
  {
    // CHECK-NEXT: %[[TRANSFER0:.*]] = mpmd.transfer %arg0
    // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @f(%arg0, %[[TRANSFER0]])
    %0 = mpmd.call @f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %1 = mpmd.assign %0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
    %2 = mpmd.assign %0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
    return %1, %2 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
  }

  // `f` has args cloned to be assigned to multiple meshes.
  // CHECK-LABEL: func.func private @f(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2"
  // CHECK-SAME: -> ({{.*}}"m1"{{.*}}, {{.*}}"m2"
  func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>} {
    return %arg0 : tensor<3x5xf32>
  }
}

// -----

#topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>, <"m3" : <["x"=1]>>>
module {

  // CHECK-LABEL: func.func public @chained_calls_works_with_multiple_iterations(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1"
  func.func public @chained_calls_works_with_multiple_iterations(
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

    %3:3 = mpmd.call @f(%1#0, %1#1, %1#2) : (
      tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
    ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
    %4 = mpmd.assign %3#0 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
    %5 = mpmd.assign %3#1 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
    %6 = mpmd.assign %3#2 : (tensor<3x5xf32>) -> !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
    return %4, %5, %6 : !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>, !mpmd.mesh_tensor<"m3", tensor<3x5xf32>>
  }


  // args (and results) get cloned to each mesh.
  // CHECK-LABEL: func.func private @f(
  // CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:    %arg3: !mpmd.mesh_tensor<"m2"
  // CHECK-SAME:    %arg4: !mpmd.mesh_tensor<"m3"
  // CHECK-SAME:    %arg5: !mpmd.mesh_tensor<"m2"
  // CHECK-SAME:    %arg6: !mpmd.mesh_tensor<"m3"
  // CHECK-SAME:    %arg7: !mpmd.mesh_tensor<"m2"
  // CHECK-SAME:    %arg8: !mpmd.mesh_tensor<"m3"
  func.func private @f(
    %arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>, %arg2: tensor<3x5xf32>
  ) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>) attributes
  {topology = #topology} {
    return %arg1, %arg2, %arg0 : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
  }
}

