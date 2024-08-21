// RUN: sdy_opt %s -split-input-file -verify-diagnostics

// expected-error @below {{axis size must be at least 1, got: 0}}
// expected-error @below {{custom op 'sdy.mesh' failed to parse Sdy_Mesh parameter 'axes' which is to be a `::llvm::ArrayRef<MeshAxisAttr>`}}
sdy.mesh @mesh = <["a"=2, "b"=0]>

// -----

// expected-error @+1 {{duplicate axis name: "a"}}
sdy.mesh @mesh = <["a"=2, "b"=2, "a"=4]>

// -----
// expected-error @+1 {{device id must be non-negative, got: -1}}
sdy.mesh @mesh = <device_ids=[-1]>

// -----
// expected-error @+1 {{total product of axis sizes must match total number of device ids, got: 1 != 2}}
sdy.mesh @mesh = <device_ids=[1, 0]>

// -----
// expected-error @+1 {{total product of axis sizes must match total number of device ids, got: 4 != 6}}
sdy.mesh @mesh = <["a"=2, "b"=2] device_ids=[0, 2, 1, 3, 4, 5]>

// -----
// expected-error @+1 {{custom op 'sdy.mesh' total product of axis sizes must match total number of device ids, got: 2 != 1}}
sdy.mesh @mesh = <["a"=2] device_ids=[0]>

// -----
// expected-error @+1 {{custom op 'sdy.mesh' if the ordered device ids are the same as iota(product(axes)), no need to specify them for simplicity}}
sdy.mesh @mesh_iota_device_ids = <["a"=2] device_ids=[0, 1]>

// -----
// expected-error @+1 {{custom op 'sdy.mesh' sorted device ids must be iota(product(axes)), got: 1, 1}}
sdy.mesh @mesh_duplicated_device_ids = <["a"=2] device_ids=[1, 1]>

// -----
// expected-error @+1 {{custom op 'sdy.mesh' sorted device ids must be iota(product(axes)), got: 1, 2}}
sdy.mesh @mesh_out_if_bound_device_ids = <["a"=2] device_ids=[2, 1]>
