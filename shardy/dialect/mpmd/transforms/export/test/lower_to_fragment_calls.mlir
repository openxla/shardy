// RUN: mpmd_opt %s -mpmd-lower-to-fragment-calls='group-across-meshes=false' -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT_CALL_0:.*]] = mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0, %arg1) {remat}
  // NB: just setting the remat flag on to see it preserved in the first (but only the first!) fragment call.
  %f0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64, remat}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor


  // CHECK-NEXT: %[[FRAGMENT_CALL_1:.*]] = mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT1:.*]](%arg0, %arg1)
  %f1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) {xla_tpu_user_reserved_hbm_bytes = 128 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  func.return %f1: !mesh_1_tensor
}

// CHECK:       func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-NEXT:    return %[[ADD]]
// CHECK-NEXT:  }


// CHECK:       func @[[FRAGMENT1]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %arg0, %arg1
// CHECK-NEXT:    return %[[MUL]]
// CHECK-NEXT:  }

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
    -> (!mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=2, "y"=2]>>,
      <"m3": <["x"=4]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT_CALL_0:.*]] = mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0, %arg1)
  %f0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  // This fragment has the same body and mesh as the fragment `%f0`,
  // therefore the two fragments will call the same function.
  // CHECK-NEXT: %[[FRAGMENT_CALL_2:.*]] = mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0, %arg1)
  %f1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  func.return %f1 : !mesh_1_tensor
}

// CHECK:       func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-NEXT:    return %[[ADD]]
// CHECK-NEXT:  }

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
    -> (!mesh_2_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=2, "y"=2]>>,
      <"m3": <["x"=4]>>
    >} {

  // CHECK-NEXT: %[[FRAGMENT_CALL_0:.*]] = mpmd.fragment_call<mesh="m1", origin=["f0"]> @[[FRAGMENT0:.*]](%arg0, %arg1)
  %f0 = mpmd.fragment<mesh="m1", origin=["f0"]> (%arg0, %arg1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // CHECK-NEXT: %[[TRANSFER_1:.*]] = mpmd.transfer %[[FRAGMENT_CALL_0:.*]]
  %transfer1 = mpmd.transfer %f0 : (!mesh_1_tensor) -> !mesh_2_tensor

  // This fragment has the same body as the fragment `%f0` fragment but a
  // different mesh shape, therefore the two fragments won't call the same
  // function.
  // CHECK-NEXT: %[[FRAGMENT_CALL_3:.*]] = mpmd.fragment_call<mesh="m2", origin=["f3"]> @[[FRAGMENT2:.*]](%[[TRANSFER_1]], %[[TRANSFER_1]])
  %f4 = mpmd.fragment<mesh="m2", origin=["f3"]> (%transfer1, %transfer1) {xla_tpu_user_reserved_hbm_bytes = 384 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor

  func.return %f4: !mesh_2_tensor
}

// CHECK:       func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-NEXT:    return %[[ADD]]
// CHECK-NEXT:  }

// CHECK:       func @[[FRAGMENT2]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=2]>, xla_tpu_user_reserved_hbm_bytes = 384 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-NEXT:    return %[[ADD]]
// CHECK-NEXT:  }

// -----


!mesh_2_tensor_dist_x = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_2_tensor_dist_x)
    -> (!mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x) attributes {
    "topology"=#mpmd.topology<
      <"m2": <["x"=2, "y"=2]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT_CALL_4:.*]]:2 = mpmd.fragment_call<mesh="m2", origin=["f4"]> @[[FRAGMENT3:.*]](%arg0)
  %f5:2 = mpmd.fragment<mesh="m2", origin=["f4"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg2: tensor<4x8xf32>) {
    %13 = stablehlo.subtract %arg2, %arg2 : tensor<4x8xf32>
    %14 = stablehlo.divide %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %14, %13 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_2_tensor_dist_x) -> (!mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x)

  // This fragment has the same body as the fragment `%f5` above up to the
  // terminator, which has a different order of operands, therefore the two
  // fragments won't call the same function.
  // CHECK-NEXT: %[[FRAGMENT_CALL_5:.*]]:2 = mpmd.fragment_call<mesh="m2", origin=["f4"]> @[[FRAGMENT4:.*]](%arg0)
  %f6:2 = mpmd.fragment<mesh="m2", origin=["f4"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 128 : i64}
    (%arg2: tensor<4x8xf32>) {
    %13 = stablehlo.subtract %arg2, %arg2 : tensor<4x8xf32>
    %14 = stablehlo.divide %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %13, %14 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_2_tensor_dist_x) -> (!mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x)

  func.return %f6#0, %f6#1 : !mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x

}

// CHECK:       func @[[FRAGMENT3]](%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=2]>, xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT:    %[[SUBTRACT:.*]] = stablehlo.subtract %arg0, %arg0
// CHECK-NEXT:    %[[DIVIDE:.*]] = stablehlo.divide %arg0, %arg0
// CHECK-NEXT:    return %[[DIVIDE]], %[[SUBTRACT]]
// CHECK-NEXT:  }

// CHECK:       func @[[FRAGMENT4]](%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:    %[[SUBTRACT:.*]] = stablehlo.subtract %arg0, %arg0
// CHECK-NEXT:    %[[DIVIDE:.*]] = stablehlo.divide %arg0, %arg0
// CHECK-NEXT:    return %[[SUBTRACT]], %[[DIVIDE]]
// CHECK-NEXT:  }

// -----

!mesh_2_tensor_dist_x = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_2_tensor_dist_x)
    -> (!mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x) attributes {
    "topology"=#mpmd.topology<
      <"m2": <["x"=2, "y"=2]>>
    >} {

  // CHECK-NEXT: %[[FRAGMENT_CALL_5:.*]]:2 = mpmd.fragment_call<mesh="m2", origin=["f4"]> @[[FRAGMENT4:.*]](%arg0)
  %f6:2 = mpmd.fragment<mesh="m2", origin=["f4"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 128 : i64}
    (%arg2: tensor<4x8xf32>) {
    %13 = stablehlo.subtract %arg2, %arg2 : tensor<4x8xf32>
    %14 = stablehlo.divide %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %13, %14 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_2_tensor_dist_x) -> (!mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x)

  // CHECK-NEXT: %[[TRANSFER_2:.*]] = mpmd.transfer %[[FRAGMENT_CALL_5]]#0
  %transfer3 = mpmd.transfer %f6#0 : (!mesh_2_tensor_dist_x) -> !mesh_2_tensor_dist_x

  // This fragment has the same body as the second fragment `%f6` above but
  // different operand/result mesh types. Since the outer types aren't relevant
  // for the called function, the two fragments will call the same function.
  // CHECK-NEXT: %[[FRAGMENT_CALL_6:.*]]:2 = mpmd.fragment_call<mesh="m2", origin=["f4"]> @[[FRAGMENT4:.*]](%[[TRANSFER_2]])
  %f7:2 = mpmd.fragment<mesh="m2", origin=["f4"]> (%transfer3) {xla_tpu_user_reserved_hbm_bytes = 0 : i64}
    (%arg2: tensor<4x8xf32>) {
    %13 = stablehlo.subtract %arg2, %arg2 : tensor<4x8xf32>
    %14 = stablehlo.divide %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %13, %14 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_2_tensor_dist_x) -> (!mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x)

   func.return %f7#0, %f7#1 : !mesh_2_tensor_dist_x, !mesh_2_tensor_dist_x
}

// CHECK:       func @[[FRAGMENT4]](%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:    %[[SUBTRACT:.*]] = stablehlo.subtract %arg0, %arg0
// CHECK-NEXT:    %[[DIVIDE:.*]] = stablehlo.divide %arg0, %arg0
// CHECK-NEXT:    return %[[SUBTRACT]], %[[DIVIDE]]
// CHECK-NEXT:  }

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_2_tensor_dist_x = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_2_tensor_dist_y = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>
!mesh_3_tensor = !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
    -> (!mesh_3_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=2, "y"=2]>>,
      <"m3": <["x"=4]>>
    >} {

  // CHECK-NEXT: %[[FRAGMENT_CALL_0:.*]] = mpmd.fragment_call<mesh="m1", origin=["f0"]> @[[FRAGMENT0:.*]](%arg0, %arg1)
  %f0 = mpmd.fragment<mesh="m1", origin=["f0"]> (%arg0, %arg1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor

  // CHECK-NEXT: %[[TRANSFER_3:.*]] = mpmd.transfer %arg0
  %transfer4 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_3_tensor

  // This fragment has the same body as the fragment `%f0` above but is
  // assigned to a different mesh, therefore the two fragments won't call the
  // same function.
  // CHECK-NEXT: %[[FRAGMENT_CALL_7:.*]] = mpmd.fragment_call<mesh="m3", origin=["f5"]> @[[FRAGMENT5:.*]](%[[TRANSFER_3]], %[[TRANSFER_3]])
  %f8 = mpmd.fragment<mesh="m3", origin=["f5"]> (%transfer4, %transfer4) {xla_tpu_user_reserved_hbm_bytes = 0 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_3_tensor, !mesh_3_tensor) -> !mesh_3_tensor

  // This fragment has the same body as the fragment `%f8` but an additional
  // unused block argument, therefore the two fragments won't call the same
  // function.
  // CHECK-NEXT: %[[FRAGMENT_CALL_8:.*]] = mpmd.fragment_call<mesh="m3", origin=["f5"]> @[[FRAGMENT6:.*]](%[[FRAGMENT_CALL_7]], %[[FRAGMENT_CALL_7]], %[[FRAGMENT_CALL_7]])
  %f9 = mpmd.fragment<mesh="m3", origin=["f5"]> (%f8, %f8, %f8) {xla_tpu_user_reserved_hbm_bytes = 128 : i64}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mesh_3_tensor, !mesh_3_tensor, !mesh_3_tensor) -> !mesh_3_tensor

  // CHECK-NEXT: return %[[FRAGMENT_CALL_8]]
  func.return %f9 : !mesh_3_tensor
}

// CHECK:       func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-NEXT:    return %[[ADD]]
// CHECK-NEXT:  }

// CHECK:       func @[[FRAGMENT5]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:    return %[[ADD]] : tensor<4x8xf32>
// CHECK-NEXT:  }

// CHECK:       func @[[FRAGMENT6]](%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:      attributes {mesh_shape = #sdy.mesh<["x"=4]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-NEXT:    return %[[ADD]]
// CHECK-NEXT:  }

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
  // This fragment and the next fragment are the same except for the arg_attr attributes. So there should be two fragment calls to different functions.
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {arg_attrs = [{tf.aliasing_output = 0 : i32}], xla_tpu_user_reserved_hbm_bytes = 128 : i64} (%arg2: tensor<4x8xf32>) {
    %1 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT1:.*]](%arg0)
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 128 : i64} (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// This fragment call function has the tf.aliasing_output arg attribute.
// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32> {tf.aliasing_output = 0 : i32})
// CHECK-SAME: attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 : tensor<4x8xf32>
// CHECK-NEXT:  return %0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// This fragment call function does not have the tf.aliasing_output arg attribute.
// CHECK: func @[[FRAGMENT1]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME: attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 : tensor<4x8xf32>
// CHECK-NEXT:  return %0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
  // This fragment and the next fragment are the same except for the res_attr attributes.
  // So there should be two fragment calls to different functions.
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {res_attrs = [{mhlo.memory_kind = "pinned_host"}], xla_tpu_user_reserved_hbm_bytes = 0 : i64} (%arg2: tensor<4x8xf32>) {
    %1 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT1:.*]](%arg0)
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 0 : i64} (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// This fragment call function has the mhlo.memory_kind res attribute.
// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32> {mhlo.memory_kind = "pinned_host"})
// This fragment call function does not have the tf.aliasing_output arg attribute.
// CHECK: func @[[FRAGMENT1]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
  // This fragment and the next fragment are exactly the same. So there should be two fragment calls to the same function.
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {arg_attrs = [{tf.aliasing_output = 0 : i32}], xla_tpu_user_reserved_hbm_bytes = 128 : i64} (%arg2: tensor<4x8xf32>) {
    %1 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0)
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {arg_attrs = [{tf.aliasing_output = 0 : i32}], xla_tpu_user_reserved_hbm_bytes = 128 : i64} (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// This fragment call has the tf.aliasing_output arg attribute.
// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 : tensor<4x8xf32>
// CHECK-NEXT:   return %0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
  // This fragment and the next fragment are exactly the same. So there should
  // be two fragment calls to the same function. One of the calls is annotated
  // with `mpmd.is_gspmd_partitioned` and the other isn't.
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) {mpmd.is_gspmd_partitioned
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {mpmd.is_gspmd_partitioned, xla_tpu_user_reserved_hbm_bytes = 0 : i64} (%arg2: tensor<4x8xf32>) {
    %1 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) :
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 0 : i64} (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 : tensor<4x8xf32>
// CHECK-NEXT:   return %0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @no_reserved_memory
func.func @no_reserved_memory(%arg0: !mesh_tensor)
  -> (!mesh_tensor) attributes {"topology"=#mpmd.topology< <"m1": <["x"=2]>>>} {

  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) :
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> (!mesh_tensor)

  func.return %0 : !mesh_tensor
}

// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
// CHECK-NEXT:   return %arg0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_tensor)
  -> (!mesh_tensor) attributes {"topology"=#mpmd.topology< <"m1": <["x"=2]>>>} {

  // Only the FIRST fragment has a xla_tpu_user_reserved_hbm_bytes annotation.
  // hbm_bytes gets attached to fragment function.
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) :
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 128 : i64} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> (!mesh_tensor)
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) :
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> (!mesh_tensor)

  func.return %2 : !mesh_tensor
}

// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:   return %arg0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_tensor)
  -> (!mesh_tensor) attributes {"topology"=#mpmd.topology< <"m1": <["x"=2]>>>} {

  // Only the SECOND fragment has a xla_tpu_user_reserved_hbm_bytes annotation.
  // hbm_bytes gets attached to fragment function.
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) :
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)  (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> (!mesh_tensor)
  // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0) :
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 128 : i64} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> (!mesh_tensor)

  func.return %2 : !mesh_tensor
}

// CHECK: func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>, xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
// CHECK-NEXT:   return %arg0 : tensor<4x8xf32>
// CHECK-NEXT:  }

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>

module attributes {mpmd.sdy_lowered} {
  sdy.mesh @mesh = <["x"=2]>

  // CHECK-LABEL: func @sdy_partitioned
  func.func @sdy_partitioned(%arg0: !mesh_tensor) -> (!mesh_tensor)
      attributes {"topology"=#mpmd.topology< <"m1": <["x"=2]>>>} {
    // CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]>
    %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
      mpmd.return %arg2 : tensor<4x8xf32>
    } : (!mesh_tensor) -> (!mesh_tensor)

    func.return %0 : !mesh_tensor
  }
}

// CHECK: func.func @[[FRAGMENT0:.*]](%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>}) ->
// CHECK-SAME: (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>}) attributes {mesh_shape = #sdy.mesh<["x"=2]>}
