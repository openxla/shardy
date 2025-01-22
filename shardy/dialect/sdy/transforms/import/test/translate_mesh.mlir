// RUN: sdy_opt %s -sdy-translate-mesh="old-mesh-name=my_mesh axis-names='data,model'" 2>&1 | FileCheck %s

// CHECK-LABEL: @my_mesh
// CHECK-SAME{LITERAL}: <["data"=2, "model"=4]>
sdy.mesh @my_mesh = <["a"=2, "b"=4]>

// CHECK-NOT: <["a"=2, "b"=4]>

// CHECK-LABEL: @foo
func.func @foo(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@my_mesh, [{"data", ?}p1, {}], replicated={"model"}>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@my_mesh, [{"a", ?}p1, {}], replicated={"b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
