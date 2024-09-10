# SDY C APIs.

package(default_visibility = ["//visibility:public"])

SDY_CAPI_SOURCES = [
    "attributes.cc",
    "dialect.cc",
]

SDY_CAPI_HEADERS = [
    "attributes.h",
    "dialect.h",
]

cc_library(
    name = "sdy_capi",
    srcs = SDY_CAPI_SOURCES,
    hdrs = SDY_CAPI_HEADERS,
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "sdy_capi_headers",
    hdrs = SDY_CAPI_HEADERS,
    includes = ["."],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "sdy_capi_objects",
    srcs = SDY_CAPI_SOURCES,
    hdrs = SDY_CAPI_HEADERS,
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = True,
)
