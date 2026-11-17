# Compatibility test workflow

For each new change in Shardy attributes we follow the process:

1. Bump the version of SdyDialect in [SdyDialectVersion::getCurrentVersion()](../dialect.h) (refer to [Version changes](#versioning)).
2. Copy the latest `compatibility_test.{Major}_{Minor}_{Patch}.mlir` file and
 create a new one with new version number in the filename
3. Inside the `mlir` test change the mentions of previous version to new version
4. Append tests for new functionality to the file
5. Generate `bc` file for this test:

```sh
bazel build //shardy/tools:sdy_translate

# change the version to the correct one
./bazel-bin/shardy/tools/sdy_translate --strip-debuginfo --serialize shardy/dialect/sdy/ir/compatibility_test/compatibility_test.0_0_0.mlir > shardy/dialect/sdy/ir/compatibility_test/compatibility_test.0_0_0.mlir.bc
```

Note: **forward compatibility** is supported by keeping old .bc files and
running checks that compiler can still generate old .bc files (when asked for
certain target_version) and the test works as expected

Note: **backward compatibility** is supported by running old tests and being
able to deserialize with new compiler and test works as expected

## Version changes {#versioning}

When the module is serialized, extra passes can be added to downgrade
the module to older version in `compatibility.cc` `downgradeModule`.

When the module is deserialized, extra passes can be added to
upgrade the module from older (dumped) version to current (latest) in
`SdyDialectBytecodeInterface::upgradeFromVersion`.
