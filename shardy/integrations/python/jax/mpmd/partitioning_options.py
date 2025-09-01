# Copyright 2025 The MPMD Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines options for MPMD partitioning."""

PARTIR_MPMD_BOOLEAN_OPTIONS = frozenset({
    'partir_mpmd_infer_transfers',
    'partir_mpmd_infer_cross_mesh_reductions',
    'partir_mpmd_merge_inferred_with_cloning_during_import',
    'partir_mpmd_gspmd_propagate_sharding_across_meshes',
    'partir_mpmd_allow_intra_mesh_transfer',
    'partir_mpmd_fragment_remat',
    'partir_mpmd_merge_remat_fragments',
    'partir_mpmd_split_bwd_fragments',
    'partir_mpmd_assume_homogeneous_devices',
    'partir_mpmd_absorb_inferred_fragments_on_entry_point_function',
    'partir_mpmd_copy_constant_creation_from_producer_to_consumer',
    'partir_mpmd_apply_merge_transfers_pass',
    'partir_mpmd_merge_after_scheduling',
})

PARTIR_MPMD_PIPELINE_SCHEDULE_OPTION = 'partir_mpmd_pipeline_schedule'

PARTIR_MPMD_PIPELINE_SCHEDULE_VALUES = frozenset({
    'NONE',
    'ONE_FWD_ONE_BWD',
    'GPIPE',
    'GPIPE_BUT_1F1B_FOR_LAST_MESH',
    'ZERO_BUBBLE_H1',
    'ZERO_BUBBLE_H2_ZERO_TX_LATENCY',
    'ZERO_BUBBLE_H2_HALF_TX_LATENCY',
    'ZERO_BUBBLE_H2_FULL_TX_LATENCY',
    'PARALLEL_PIPELINES_WITH_WRAP_AROUND',
    'CIRCULAR',
    'CIRCULAR_WITH_REVERSED_BACKWARD',
})

PARTIR_MPMD_OPTIONS = PARTIR_MPMD_BOOLEAN_OPTIONS | frozenset(
    {PARTIR_MPMD_PIPELINE_SCHEDULE_OPTION}
)
