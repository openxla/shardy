# MPMD C APIs.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

MPMD_CAPI_SOURCES = [
    "attributes.cc",
    "dialect.cc",
    "passes.cc",
]

MPMD_CAPI_HEADERS = [
    "attributes.h",
    "dialect.h",
    "passes.h",
]

cc_library(
    name = "mpmd_capi",
    srcs = MPMD_CAPI_SOURCES,
    hdrs = MPMD_CAPI_HEADERS,
    deps = [
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/transforms:passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "mpmd_capi_headers",
    hdrs = MPMD_CAPI_HEADERS,
    includes = ["."],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "mpmd_capi_objects",
    srcs = MPMD_CAPI_SOURCES,
    hdrs = MPMD_CAPI_HEADERS,
    deps = [
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/transforms:passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = True,
)
