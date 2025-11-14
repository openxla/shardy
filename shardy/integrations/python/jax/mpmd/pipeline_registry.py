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

"""Pipeline schedule registry.

Central registry mapping schedule names to PipelineSchedule objects. Each
schedule defines fragment merging and ordering using binary predicate functions.

Usage:
    schedule = get_pipeline_schedule('1F1B')
    config = make_config(
        topology=topology,
        name_to_mesh_assignment=mesh_assignment,
        pipeline_schedule=schedule,
    )
"""

import functools

import immutabledict

from shardy.integrations.python.jax.mpmd import pipeline
from shardy.integrations.python.jax.mpmd import schedule_impl

ImmutableDict = immutabledict.immutabledict

PIPELINE_SCHEDULES: ImmutableDict[str, pipeline.PipelineSchedule] = (
    ImmutableDict({
        'ONE_FWD_ONE_BWD': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=schedule_impl.one_fwd_one_bwd_schedule_predicate,
                )
            ],
            required_mpmd_options={'mpmd_pipeline_schedule': '1F1B'},
        ),
        'GPIPE': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=schedule_impl.gpipe_schedule_predicate,
                )
            ],
            required_mpmd_options={'mpmd_pipeline_schedule': 'GPipe'},
        ),
        'GPIPE_BUT_1F1B_FOR_LAST_MESH': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=schedule_impl.gpipe_with_1f1b_on_last_mesh_schedule_predicate,
                )
            ],
            required_mpmd_options={
                'mpmd_pipeline_schedule': 'GPipeBut1F1BForLastMesh'
            },
        ),
        'ZERO_BUBBLE_H1': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=schedule_impl.zero_bubble_h1_schedule_predicate,
                )
            ],
            required_mpmd_options={
                'mpmd_pipeline_schedule': 'ZeroBubbleH1',
                'mpmd_split_bwd_fragments': True,
            },
        ),
        'ZERO_BUBBLE_H2_ZERO_TX_LATENCY': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=functools.partial(
                        schedule_impl.latency_hiding_zero_bubble_h2_schedule_predicate,
                        latency_stage_fraction=0.0,
                    ),
                )
            ],
            required_mpmd_options={
                'mpmd_split_bwd_fragments': True,
                'mpmd_pipeline_schedule': 'ZeroBubbleH2ZeroTxLatency',
            },
        ),
        'ZERO_BUBBLE_H2_HALF_TX_LATENCY': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=functools.partial(
                        schedule_impl.latency_hiding_zero_bubble_h2_schedule_predicate,
                        latency_stage_fraction=0.5,
                    ),
                )
            ],
            required_mpmd_options={
                'mpmd_split_bwd_fragments': True,
                'mpmd_pipeline_schedule': 'ZeroBubbleH2HalfTxLatency',
            },
        ),
        'ZERO_BUBBLE_H2_FULL_TX_LATENCY': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=functools.partial(
                        schedule_impl.latency_hiding_zero_bubble_h2_schedule_predicate,
                        latency_stage_fraction=1.0,
                    ),
                )
            ],
            required_mpmd_options={
                'mpmd_split_bwd_fragments': True,
                'mpmd_pipeline_schedule': 'ZeroBubbleH2FullTxLatency',
            },
        ),
        'PARALLEL_PIPELINES_WITH_WRAP_AROUND': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=schedule_impl.parallel_pipelines_with_wraparound_schedule_predicate,
                )
            ],
            required_mpmd_options={
                'mpmd_pipeline_schedule': 'ParallelPipelinesWithWrapAround',
            },
        ),
        'CIRCULAR': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=functools.partial(
                        schedule_impl.circular_schedule_predicate_base,
                        reverse_backward=False,
                    ),
                )
            ],
            required_mpmd_options={
                'mpmd_pipeline_schedule': 'Circular',
            },
        ),
        'CIRCULAR_WITH_REVERSED_BACKWARD': pipeline.PipelineSchedule(
            schedule_merge_rule_builders=[
                functools.partial(
                    pipeline.build_schedule_rules_from_predicate,
                    before_pred=functools.partial(
                        schedule_impl.circular_schedule_predicate_base,
                        reverse_backward=True,
                    ),
                )
            ],
            required_mpmd_options={
                'mpmd_pipeline_schedule': 'CircularWithReversedBackward',
            },
        ),
    })
)


def get_pipeline_schedule(schedule_name: str) -> pipeline.PipelineSchedule:
  """Get a PipelineSchedule object for the given schedule name."""
  if schedule_name not in PIPELINE_SCHEDULES:
    valid_schedules = sorted(PIPELINE_SCHEDULES.keys())
    raise KeyError(
        f"Unknown pipeline schedule '{schedule_name}'. "
        f'Valid schedules are: {valid_schedules!r}'
    )
  return PIPELINE_SCHEDULES[schedule_name]
