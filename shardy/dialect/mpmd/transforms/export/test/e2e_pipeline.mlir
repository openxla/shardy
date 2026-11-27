// RUN: mpmd_opt %s -mpmd-import-pipeline='name-to-mesh-assignment=f1@m1,f2@m2' -mpmd-optimize-pipeline -mpmd-sharding-propagation-pipeline -mpmd-export-pipeline 2>&1 | FileCheck %s

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#topology} {
  // CHECK: %[[FRAGMENT_CALL:.*]] = mpmd.fragment_call<mesh="m1", origin=["f1"]> @p0_f1_fwd.main(%arg0)
  %1:2 = mpmd.named_computation<"f1"> (%arg0, %arg0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %2 = stablehlo.custom_call @sdy_testonly(%arg3) {mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %3 = stablehlo.custom_call @sdy_testonly(%arg4) {mhlo.no_cse} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %2, %3 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %1#0 : tensor<4x8xf32>
}
// CHECK-LABEL: func.func @p0_f1_fwd.main
// CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @sdy_testonly
// CHECK-NOT: has_side_effect
// CHECK-SAME: {mhlo.no_cse}
// CHECK-NEXT: %[[CUSTOM_CALL_2:.*]] = stablehlo.custom_call @sdy_testonly
// CHECK-NOT: has_side_effect
// CHECK-SAME: {mhlo.no_cse}
