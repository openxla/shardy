package(
    default_applicable_licenses = ["//third_party/openxla:package_license"],
    default_restricted_to = [
        "//buildenv/target:prod",
        "//buildenv/target:libtpu",
    ],
    default_visibility = ["//third_party/openxla/shardy:__subpackages__"],
)

# Allow visibility from the mlir language server.
package_group(
    name = "friends",
    packages = ["//third_party/py/openxla/sdy/..."],
)
