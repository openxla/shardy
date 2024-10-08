"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    #
    STABLEHLO_COMMIT = "44a9acf2de746ac1c6c885ba11b414f1d38637d5"
    STABLEHLO_SHA256 = "1918d011b782868a30de5b264fe99d5f8a7139d04bfbb44e5588bace04abb812"
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
