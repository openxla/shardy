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
# Wrapper to run sdy execution tests with the partitioner pipeline.

set -e

SRC=""
TMP=""
ENABLE_HALO_EXCHANGE=true
# When replica_count=partition_count=1, the test use partition id.
REPLICA_COUNT=1
PARTITION_COUNT=1

pos_idx=0

for arg in "$@"; do
  case "$arg" in
    --src=*)                  SRC="${arg#*=}" ;;
    --temp_dir=*)             TMP="${arg#*=}" ;;
    --enable_halo_exchange=*) ENABLE_HALO_EXCHANGE="${arg#*=}" ;;
    --replica_count=*)        REPLICA_COUNT="${arg#*=}" ;;
    --partition_count=*)      PARTITION_COUNT="${arg#*=}" ;;
    --*)                      echo "Warning: Unknown flag '$arg'" >&2 ;;
    *)
      pos_idx=$((pos_idx + 1))
      case "$pos_idx" in
        1) SRC="$arg" ;;
        2) TMP="$arg" ;;
        3) ENABLE_HALO_EXCHANGE="$arg" ;;
        4) REPLICA_COUNT="$arg" ;;
        5) PARTITION_COUNT="$arg" ;;
      esac
      ;;
  esac
done

if [ -z "$SRC" ] || [ -z "$TMP" ]; then
  echo "Error: Missing required arguments. Usage: $0 <src.mlir> <temp_dir> [flags...]" >&2
  exit 1
fi

SPLIT_FILE=${SPLIT_FILE:-split-file}
SDY_OPT=${SDY_OPT:-sdy_opt}
STABLEHLO_TRANSLATE=${STABLEHLO_TRANSLATE:-stablehlo-translate}

"$SPLIT_FILE" "$SRC" "$TMP"
# Run the partitioner pipeline passes.
"$SDY_OPT" "$TMP/part1.mlir" \
  --sdy-insert-explicit-reshards="enable-full-version=true" \
  --sdy-resolve-permutation-factors="enable-halo-exchange=$ENABLE_HALO_EXCHANGE replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \
  --sdy-reshard-to-collectives \
  --sdy-optimize-collectives \
  --sdy-pad-for-divisibility \
  --sdy-resolve-single-device-sharding="replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \
  --sdy-convert-global-to-local="replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \
  --sdy-inline-meshes \
  --sdy-drop-sharding-and-mesh \
  --allow-unregistered-dialect > "$TMP/part1_processed.mlir"

sed '1d; /^}/,$d' "$TMP/part1_processed.mlir" > "$TMP/combined.mlir"

# If part1.mlir contains @parallel_x but not @sequential_x, then remove sharding
# from @parallel_x and rename it to @sequential_x.
if (grep -q "@parallel_" "$TMP/part1.mlir") && (! grep -q "@sequential_" "$TMP/part1.mlir"); then
  "$SDY_OPT" "$TMP/part1.mlir" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
  sed 's/parallel_/sequential_/g' > "$TMP/part1_sequential.mlir"
  sed '1d; /^}/,$d' "$TMP/part1_sequential.mlir" >> "$TMP/combined.mlir"
fi

cat "$TMP/part2.mlir" >> "$TMP/combined.mlir"
"$STABLEHLO_TRANSLATE" --interpret "$TMP/combined.mlir"

