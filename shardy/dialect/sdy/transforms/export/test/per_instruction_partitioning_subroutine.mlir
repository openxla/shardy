// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter=func=subroutine,add" | FileCheck %s

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func private @subroutine
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @subroutine(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // The add inside @subroutine is selected and wrapped in sdy.manual_computation.
  // CHECK:      %[[MANUAL_ADD:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_ARG0:.*]]: tensor<4x16xf32>, %[[LOCAL_ARG1:.*]]: tensor<4x16xf32>) {
  // CHECK-NEXT:   %[[LOCAL_ADD:.*]] = stablehlo.add %[[LOCAL_ARG0]], %[[LOCAL_ARG1]] : tensor<4x16xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL_ADD]] : tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[MANUAL_ADD]] : tensor<8x16xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @main
func.func @main(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // The add in @main is not partitioned because filter restricts to func=subroutine.
  // CHECK-NOT: sdy.manual_computation
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK: %[[CALL:.*]] = call @subroutine(%[[ADD]], %arg1)
  // CHECK: return %[[CALL]] : tensor<8x16xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  %1 = func.call @subroutine(%0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
