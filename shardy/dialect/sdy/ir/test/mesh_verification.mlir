// RUN: sdy_opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{axis size must be at least 1, got: 0}}
sdy.mesh @mesh = <["a"=2, "b"=0]>

// -----

// expected-error @+1 {{duplicate axis name: "a"}}
sdy.mesh @mesh = <["a"=2, "b"=2, "a"=4]>

// -----
// expected-error @+1 {{device id must be non-negative, got: -1}}
sdy.mesh @mesh = <device_ids=[-1]>

// -----
// expected-error @+1 {{expected '>'}}
sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[2]>
