// RUN: mpmd_opt %s -mpmd-mark-fragment-reserved-memory 2>&1 | FileCheck %s

// NOTE:
// - mesh_1_tensor and mesh_2_tensor is 128 bytes.
// - mesh_1_tensor_dist_x is 32 bytes.
!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_dist_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @single_mesh
func.func @single_mesh(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>>} {

  // Fragment only takes inputs from the function, no intermediates, so 0.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // Fragment takes one input from the function and one intermediates, so 128 due to
  // %arg0.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // Fragment takes two intermediates, so 256 due to %arg0 and %arg1
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 256
  %2 = mpmd.fragment<mesh="m1", origin=["f3"]> (%0, %1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // Fragment takes an input and a intermediate %2. Note %0's and %1's last use
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %3 = mpmd.fragment<mesh="m1", origin=["f4"]> (%arg0, %2)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  func.return %3 : !mesh_1_tensor
}

// Make sure we don't subtract twice from live memory usage if a fragment takes
// two of the same inputs.
// CHECK-LABEL: func @duplicate_input
func.func @duplicate_input(%arg0: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>>} {

  // Fragment only takes inputs from the function, no intermediates, so 0.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @offloaded_value
func.func @offloaded_value(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>>} {

  // Fragment only takes inputs from the function, no intermediates, so 0.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    {res_attrs = [{mhlo.memory_kind = "pinned_host"}, {}]}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    mpmd.return %arg2, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)

  // Fragment takes one input from the function and one intermediates, so
  // 128 due to %arg0. The unused intermediate is on the host so it's ignored.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0#1, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // Fragment takes one input from the function and one intermediates, so
  // 128 due to %arg0. %0#1 had its last use removed so it's not tracked
  // anymore.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %2 = mpmd.fragment<mesh="m1", origin=["f3"]> (%0#0, %1, %arg1)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}, {}, {}]}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  func.return %2 : !mesh_1_tensor
}

// Same test as `@single_mesh` but now with some tensors existing on other
// meshes.
// CHECK-LABEL: func @multiple_meshes
func.func @multiple_meshes(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor)
    -> (!mesh_2_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>, <"m2": <["x"=2]>>>} {

  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %1 = mpmd.transfer %arg1 : (!mesh_2_tensor) -> !mesh_1_tensor

  // On m2, only %arg1 and %0 exist, which are inputs to f1, so 0 bytes.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %2 = mpmd.fragment<mesh="m2", origin=["f1"]> (%0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %8 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %8 : tensor<4x8xf32>
  } : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor

  %3 = mpmd.transfer %2 : (!mesh_2_tensor) -> !mesh_1_tensor

  // On m1, %3, %1 and %arg0 are still alive, so 128 bytes for %1.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %4 = mpmd.fragment<mesh="m1", origin=["f2"]> (%3, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %8 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %8 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // On m1, %4, %1 and %arg0 are still alive, %3 is dead now, so 128 bytes.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %5 = mpmd.fragment<mesh="m1", origin=["f3"]> (%4, %1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %8 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %8 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  %6 = mpmd.transfer %5 : (!mesh_1_tensor) -> !mesh_2_tensor

  // On m2, %6 and %arg1 are still alive, %0 is dead now, so 0 bytes.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %7 = mpmd.fragment<mesh="m2", origin=["f4"]> (%6, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %8 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %8 : tensor<4x8xf32>
  } : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor

  func.return %7 : !mesh_2_tensor
}

// Tests that the pass accounts for the per device local shape, not global
// shape, if the tensor is distributed.
// CHECK-LABEL: func @distributed_tensor
func.func @distributed_tensor(%arg0: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>>} {

  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_1_tensor_dist_x

  // Fragment takes all live values, so 0.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %arg3 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor_dist_x) -> !mesh_1_tensor

  // %1 and %0 are alive still, so 128+32=160 bytes.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 160
  %2 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // Only input is unused. The input is still considered alive because it
  // has not been donated to the program, so 128 bytes.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %3 = mpmd.fragment<mesh="m1", origin=["f3"]> (%2, %1, %0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg4 : tensor<4x8xf32>
    %5 = stablehlo.add %arg3, %4 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor_dist_x) -> !mesh_1_tensor

  func.return %3 : !mesh_1_tensor
}

// Test that verifies the unused output of a fragment is not accounted for in
// the live buffers.
// CHECK-LABEL: func.func @unused_fragment_result_is_not_counted
func.func @unused_fragment_result_is_not_counted(
    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
    %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
    %arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
    %arg3: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
      -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
          !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
      attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>} {

  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 256
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg4, %arg5 : tensor<4x8xf32>
    %11 = stablehlo.abs %arg5 : tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 256
  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg2, %arg3) (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    %14 = stablehlo.add %arg4, %arg5 : tensor<4x8xf32>
    %15 = stablehlo.abs %arg5 : tensor<4x8xf32>
    mpmd.return %15 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

  return %1, %arg3 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// Test that verifies a donated program argument is not accounted for after its
// last use. The test verifies both jax.buffer_donor and tf.aliasing_output
// attributes.
// CHECK-LABEL: func.func @donated_program_arg_is_not_counted_after_last_use
func.func @donated_program_arg_is_not_counted_after_last_use(
    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
    %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {jax.buffer_donor = true},
    %arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {tf.aliasing_output = 0 : i32},
    %arg3: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
      -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
          !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
      attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>} {

  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 256
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg1, %arg0) (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg4, %arg5 : tensor<4x8xf32>
    %11 = stablehlo.abs %arg5 : tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

  // %arg3 and %arg0 are still alive. %arg1 is donated and not used anymore so
  // it's not accounted for.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg2, %arg3) (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    %14 = stablehlo.add %arg4, %arg5 : tensor<4x8xf32>
    %15 = stablehlo.abs %arg5 : tensor<4x8xf32>
    mpmd.return %15 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

  // %arg3, %arg0, and %1 are still alive.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 384
  %2 = mpmd.fragment<mesh="m1", origin=["f3"]> (%arg3, %arg3) (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    %14 = stablehlo.add %arg4, %arg5 : tensor<4x8xf32>
    %15 = stablehlo.abs %arg5 : tensor<4x8xf32>
    mpmd.return %15 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

  return %1, %2 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// Test that verifies args on hosts or donated args are not accounted for.
// CHECK-LABEL: func @offloaded_or_unused_donated_args_are_not_counted
func.func @offloaded_or_unused_donated_args_are_not_counted(
    %arg0: !mesh_1_tensor {mhlo.memory_kind = "pinned_host"},
    %arg1: !mesh_1_tensor,
    %arg2: !mesh_1_tensor,
    %arg3: !mesh_1_tensor {jax.buffer_donor = true})
      -> (!mesh_1_tensor, !mesh_1_tensor)
        attributes {"topology"=#mpmd.topology<<"m1": <["x"=4]>>>} {

  // The program arguments that are on the host or donated and not be accounted
  // for.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 0
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg1, %arg2)
    (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    mpmd.return %arg4, %arg5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)

  // Account for %0#0 and %arg1 which are alive until the end of the program.
  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 256
  %1:2 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0, %arg2)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}, {}]}
    (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    mpmd.return %arg4, %arg5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)

  func.return %0#0, %1#0 : !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @unused_input_not_donated
func.func @unused_input_not_donated(%arg0: !mesh_1_tensor, %unused_arg1: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>>} {

  // CHECK: mpmd.fragment
  // CHECK-SAME: xla_tpu_user_reserved_hbm_bytes = 128
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  func.return %0 : !mesh_1_tensor
}

