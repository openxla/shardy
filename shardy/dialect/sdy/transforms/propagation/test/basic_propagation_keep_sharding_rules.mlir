// RUN: sdy_opt %s -sdy-basic-propagate='keep-sharding-rules=true' 2>&1 | FileCheck %s

sdy.mesh @mesh = <"a"=2, "b"=2>

// CHECK-LABEL: func @existing_and_created_rules_remain
func.func @existing_and_created_rules_remain(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                  %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-SAME:     {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
// CHECK-SAME:      sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=8}>}
// CHECK-NEXT: stablehlo.add %[[ADD]], %[[ADD]]
// CHECK-SAME:     {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
// CHECK-SAME:      sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k], [ij, k])->([ij, k]) {i=4, j=2, k=8}>} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
