"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    #
    STABLEHLO_COMMIT = "8d9a84b5efbd1fe57cfcb84c6fa38f751bdbabe8"
    STABLEHLO_SHA256 = "6e4a05f016d428778b9a95e15da1c2126c4376c32105734343a86cc1b7adfbf4"
    #

    tf_http_archive(
        name = "stablehlo",
        sha256 = STABLEHLO_SHA256,
        strip_prefix = "stablehlo-{commit}".format(commit = STABLEHLO_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/stablehlo/archive/{commit}.zip".format(commit = STABLEHLO_COMMIT)),
        patch_file = [
            "//third_party/stablehlo:temporary.patch",  # Autogenerated, don't remove.
        ],
    )
