// RUN: mpmd_opt %s -verify-diagnostics -split-input-file


func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
      -> (!mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=4]>>
    >} {
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    func.return %0 : !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
}

// -----

func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<1x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
      -> (!mpmd.mesh_tensor<"m2", tensor<1x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=4]>>
    >} {
    // expected-error @+1 {{'mpmd.transfer' op dim 0 with size 1 is not divisible by its sharded size 4}}
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<1x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<1x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    func.return %0 : !mpmd.mesh_tensor<"m2", tensor<1x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
}

// -----

func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"y"}, {?}]>>)
      -> (!mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=4]>>
    >} {
    // expected-error @+1 {{'mpmd.transfer' op unknown axis name: "y"}}
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"y"}, {?}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    func.return %0 : !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
}

// -----

func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<32xf32>, sharding=<@mesh, [{"x"}, {"x"}]>>)
      -> (!mpmd.mesh_tensor<"m2", tensor<32xf32>, sharding=<@mesh, [{"x"}, {"x"}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=4]>>
    >} {
    // expected-error @+1 {{'mpmd.transfer' op sharding doesn't match tensor rank: 2 != 1}}
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<32xf32>, sharding=<@mesh, [{"x"}, {"x"}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<32xf32>, sharding=<@mesh, [{"x"}, {"x"}]>>
    func.return %0 : !mpmd.mesh_tensor<"m2", tensor<32xf32>, sharding=<@mesh, [{"x"}, {"x"}]>>
}


!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @subaxes_are_allowed_when_they_divide_sharding_dim(
   %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x":(1)4}, {}]>>)
  -> !mesh_2_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=16]>>,
      <"m2": <["x"=16]>>
    >} {
  %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x":(1)4}, {}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  func.return %0 : !mesh_2_tensor_4_8_f32
}
