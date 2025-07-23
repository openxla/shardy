// RUN: mpmd_opt %s -mpmd-import-pipeline='name-to-mesh-assignment=f1@m1/1,f2@m1/2 merge-after-scheduling=false' -mpmd-optimize-pipeline='pipeline-schedule=Circular merge-after-scheduling=false' -split-input-file 2>&1 | FileCheck %s

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// Can't get Circular pipeline because of merging.
// CHECK-LABEL: func.func public @main
func.func public @main(%arg0: tensor<3x5xf32>)
  -> (tensor<3x5xf32>, tensor<3x5xf32>) attributes {topology=#topology}
{
  // Note: not circular. Circular would be stage 1, 1, 2, 2.
  // CHECK: stage=1{{.*}}call_counter = 0
  // CHECK: stage=2{{.*}}call_counter = 0
  // CHECK: stage=1{{.*}}call_counter = 1
  // CHECK: stage=2{{.*}}call_counter = 1
  %2:2 = mpmd.call @f(%arg0) {call_counter = 0 : ui32} : (tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  %3:2 = mpmd.call @f(%2#0) {call_counter = 1 : ui32} : (tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)

  return %2#1, %3#1 : tensor<3x5xf32>, tensor<3x5xf32>
}

func.func private @f(%arg0: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>) {
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg3: tensor<3x5xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<3x5xf32>
    mpmd.return %10 : tensor<3x5xf32>
  } : (tensor<3x5xf32>) -> tensor<3x5xf32>

  %2 = mpmd.named_computation<"f2"> (%1) (%arg3: tensor<3x5xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<3x5xf32>
    mpmd.return %10 : tensor<3x5xf32>
  } : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %3 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  %4 = stablehlo.add %2, %3 : tensor<3x5xf32>

  return %3, %4 : tensor<3x5xf32>, tensor<3x5xf32>
}
