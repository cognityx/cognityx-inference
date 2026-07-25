from __future__ import annotations

from cognityx_inference.evaluation import (
    BoundaryConfig,
    BoundaryEvaluator,
    BoundaryPlanner,
    TrialOutcome,
)


def test_plan_is_finite_and_groups_compatible_loads() -> None:
    config = BoundaryConfig(
        axes={
            "model": ("model-a", "model-b"),
            "context_length": (512, 1024),
            "generation_length": (16, 32),
        },
        monotonic_axes=("context_length", "generation_length"),
    )

    trials = BoundaryPlanner(config).trials()

    assert len(trials) == 8
    assert len({trial.load_identity for trial in trials}) == 4
    assert trials[0].load_identity == trials[1].load_identity


def test_timeout_prunes_higher_value_only_for_matching_context() -> None:
    config = BoundaryConfig(
        axes={
            "model": ("model-a", "model-b"),
            "context_length": (512,),
            "batch_size": (1, 2, 4, 8),
        },
        ordered_axes=("model", "context_length", "batch_size"),
        monotonic_axes=("batch_size",),
    )

    def execute(trial):
        timed_out = (
            trial.configuration["model"] == "model-a"
            and trial.configuration["batch_size"] == 2
        )
        return TrialOutcome(
            trial_id=trial.trial_id,
            status="timeout" if timed_out else "completed",
            configuration=trial.configuration,
            runtime_seconds=2,
        )

    summary = BoundaryEvaluator(config, execute).run("job-1")
    executed = [item["configuration"] for item in summary["outcomes"]]
    pruned = [item["configuration"] for item in summary["pruned_candidates"]]

    assert {"model": "model-a", "context_length": 512, "batch_size": 4} in pruned
    assert {"model": "model-a", "context_length": 512, "batch_size": 8} in pruned
    assert {"model": "model-b", "context_length": 512, "batch_size": 8} in executed


def test_slow_threshold_becomes_a_recorded_boundary() -> None:
    config = BoundaryConfig(
        axes={"generation_length": (16, 32, 64)},
        monotonic_axes=("generation_length",),
        minimum_tokens_per_second=5,
        slow_window_seconds=1,
    )

    def execute(trial):
        return TrialOutcome(
            trial_id=trial.trial_id,
            status="completed",
            configuration=trial.configuration,
            runtime_seconds=2,
            metrics={"tokens_per_second": 4},
        )

    summary = BoundaryEvaluator(config, execute).run("job-2")

    assert summary["outcomes"][0]["status"] == "slow"
    assert len(summary["pruned_candidates"]) == 2
