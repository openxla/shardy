#!/bin/bash
# Copyright 2026 The Shardy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Wrapper to run sdy execution tests.

set -e

SRC=$1
TMP=$2

SPLIT_FILE=${SPLIT_FILE:-split-file}
SDY_OPT=${SDY_OPT:-sdy_opt}
STABLEHLO_TRANSLATE=${STABLEHLO_TRANSLATE:-stablehlo-translate}

"$SPLIT_FILE" "$SRC" "$TMP"
"$SDY_OPT" "$TMP/part1.mlir" --sdy-convert-global-to-local --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > "$TMP/part1_processed.mlir"
sed '1d; /^}/,$d' "$TMP/part1_processed.mlir" > "$TMP/combined.mlir"

# If part1.mlir contains @parallel_x but not @sequential_x, then remove sharding
# from @parallel_x and rename it to @sequential_x.
if (grep -q "@parallel_" "$TMP/part1.mlir") && (! grep -q "@sequential_" "$TMP/part1.mlir"); then
  "$SDY_OPT" "$TMP/part1.mlir" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
  sed 's/parallel_/sequential_/g' > "$TMP/part1_sequential.mlir"
  sed '1d; /^}/,$d' "$TMP/part1_sequential.mlir" >> "$TMP/combined.mlir"
fi

cat "$TMP/part2.mlir" >> "$TMP/combined.mlir"
cat "$TMP/combined.mlir"
"$STABLEHLO_TRANSLATE" --interpret "$TMP/combined.mlir"
