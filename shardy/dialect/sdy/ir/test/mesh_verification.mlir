// RUN: sdy_opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{axis size must be at least 1, got: 0}}
sdy.mesh @mesh = <"a"=2, "b"=0>

// -----

// expected-error @+1 {{duplicate axis name: "a"}}
sdy.mesh @mesh = <"a"=2, "b"=2, "a"=4>

// -----
// expected-error @+1 {{device id must be non-negative, got: -1}}
sdy.mesh @mesh = <device_id=-1>

// -----
// expected-error @below {{custom op 'sdy.mesh' expected string}}
// expected-error @below {{custom op 'sdy.mesh' failed to parse Sdy_MeshAxis parameter 'name' which is to be a `::llvm::StringRef`}}
// expected-error @below {{custom op 'sdy.mesh' failed to parse Sdy_Mesh parameter 'axes' which is to be a `::llvm::ArrayRef<MeshAxisAttr>`}}
sdy.mesh @mesh = <"a"=2, "b"=2, device_id=2>
