"""Metric and bootstrap behaviour."""

from verifier_pilot.evaluation import bootstrap, metrics


def _rows(spec):
    """spec: list of (submission_id, negative_type, true, pred)."""
    out = []
    for index, (sid, kind, true, pred) in enumerate(spec):
        out.append({
            "pair_id": f"p{index}", "submission_id": sid, "problem_id": "q1",
            "negative_type": kind, "label": true, "prediction": pred,
            "target_label": "IndexError", "gold_labels": ["IndexError"],
            "error_kind": None if pred in ("aligned", "not_aligned") else "RateLimitError",
        })
    return out


def test_perfect_predictions():
    rows = _rows([("s1", "positive", "aligned", "aligned"),
                  ("s1", "random", "not_aligned", "not_aligned")])
    result = metrics.core_metrics(rows)
    assert result["macro_f1"] == 1.0
    assert result["aligned_precision"] == 1.0
    assert result["n_failed"] == 0


def test_api_failures_are_excluded_not_scored_wrong():
    rows = _rows([("s1", "positive", "aligned", "aligned"),
                  ("s1", "random", "not_aligned", "not_aligned"),
                  ("s2", "positive", "aligned", "error")])
    result = metrics.core_metrics(rows)
    assert result["n_total"] == 3
    assert result["n_scored"] == 2
    assert result["n_failed"] == 1
    assert result["macro_f1"] == 1.0     # the failure did not count as wrong
    assert metrics.failure_breakdown(rows) == {"RateLimitError": 1}


def test_false_aligned_is_tracked_separately():
    """The reward-hacking cell: truth not_aligned, prediction aligned."""
    rows = _rows([("s1", "positive", "aligned", "aligned"),
                  ("s2", "random", "not_aligned", "aligned"),
                  ("s3", "random", "not_aligned", "not_aligned")])
    result = metrics.core_metrics(rows)
    assert result["confusion"]["true_not_aligned_pred_aligned"] == 1
    assert result["false_aligned_rate"] == 0.5
    assert result["aligned_precision"] == 0.5


def test_by_negative_type_shares_positives_and_reports_gap():
    rows = _rows([
        ("s1", "positive", "aligned", "aligned"),
        ("s1", "random", "not_aligned", "not_aligned"),
        ("s1", "hard", "not_aligned", "aligned"),
    ])
    result = metrics.by_negative_type(rows)
    assert result["random"]["n_scored"] == 2
    assert result["hard"]["n_scored"] == 2
    assert result["random"]["macro_f1"] > result["hard"]["macro_f1"]
    assert result["hard_minus_random_f1"] < 0


def test_single_class_slice_is_flagged():
    """macro F1 is not meaningful when only one true class is present."""
    rows = _rows([("s1", "positive", "aligned", "aligned"),
                  ("s2", "positive", "aligned", "aligned")])
    result = metrics.core_metrics(rows)
    assert result["n_true_classes"] == 1
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 0.5     # the absent class scores 0 -- hence the flag


def test_all_failed_returns_none_not_a_crash():
    result = metrics.core_metrics(_rows([("s1", "positive", "aligned", "error")]))
    assert result["n_scored"] == 0
    assert result["macro_f1"] is None


def test_bootstrap_clusters_by_submission():
    rows = _rows([("s1", "positive", "aligned", "aligned"),
                  ("s1", "random", "not_aligned", "not_aligned"),
                  ("s2", "positive", "aligned", "not_aligned"),
                  ("s2", "random", "not_aligned", "not_aligned")])
    ci = bootstrap.bootstrap_ci(rows, resamples=50, seed=1)
    assert ci["n_clusters"] == 2          # 2 submissions, not 4 rows
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_paired_comparison_detects_a_consistent_gap():
    good = _rows([(f"s{i}", "positive", "aligned", "aligned") for i in range(30)] +
                 [(f"s{i}", "random", "not_aligned", "not_aligned") for i in range(30)])
    bad = _rows([(f"s{i}", "positive", "aligned", "not_aligned") for i in range(30)] +
                [(f"s{i}", "random", "not_aligned", "not_aligned") for i in range(30)])
    result = bootstrap.paired_comparison(good, bad, resamples=200, seed=1)
    assert result["diff"] > 0
    assert result["significant"]


def test_paired_comparison_uses_only_shared_submissions():
    a = _rows([("s1", "positive", "aligned", "aligned"),
               ("s2", "positive", "aligned", "aligned")])
    b = _rows([("s2", "positive", "aligned", "aligned"),
               ("s3", "positive", "aligned", "aligned")])
    assert bootstrap.paired_comparison(a, b, resamples=20)["n_clusters"] == 1


def test_identical_models_show_no_difference():
    rows = _rows([(f"s{i}", "positive", "aligned", "aligned") for i in range(20)] +
                 [(f"s{i}", "random", "not_aligned", "aligned") for i in range(20)])
    result = bootstrap.paired_comparison(rows, list(rows), resamples=100, seed=1)
    assert result["diff"] == 0
    assert not result["significant"]


# --- sampling-parameter provenance -------------------------------------------

def test_param_manifest_records_requested_and_effective():
    from verifier_pilot.clients.base import VerifierClient

    class Dummy(VerifierClient):
        name, provider = "dummy", "test"
        def predict(self, system, user): ...

    client = Dummy()
    client._init_params({"temperature": 0.0, "seed": 42}, strict_params=False)
    manifest = client.params_manifest()
    assert manifest["requested"] == {"temperature": 0.0, "seed": 42}
    assert manifest["effective"] == {"temperature": 0.0, "seed": 42}
    assert manifest["param_events"] == []


def test_dropped_parameter_is_recorded_with_the_raw_error():
    from verifier_pilot.clients.base import VerifierClient

    class Dummy(VerifierClient):
        name, provider = "dummy", "test"
        def predict(self, system, user): ...

    client = Dummy()
    client._init_params({"temperature": 0.0}, strict_params=False)
    client.record_param_event("temperature", "dropped", "400 unsupported_value", "test")

    manifest = client.params_manifest()
    assert manifest["requested"]["temperature"] == 0.0
    assert "dropped" in str(manifest["effective"]["temperature"])
    assert manifest["param_events"][0]["raw_error"] == "400 unsupported_value"


def test_parameter_rejected_error_carries_the_raw_provider_error():
    from verifier_pilot.clients.base import ParameterRejectedError

    exc = ParameterRejectedError("gpt-5.1", "temperature", "400 unsupported_value", {"temperature": 0.0})
    assert exc.parameter == "temperature"
    assert exc.raw_error == "400 unsupported_value"
    assert "temperature" in str(exc)
    assert "--allow-param-fallback" in str(exc)


def test_params_manifest_is_json_serialisable_for_local_clients():
    """Local clients hold an nn.Module in .model; the manifest must not carry it."""
    import json
    from verifier_pilot.clients.base import VerifierClient

    class FakeModule:            # stands in for a loaded transformers model
        pass

    class Local(VerifierClient):
        name, provider = "qwen2.5-coder-7b", "local"
        model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
        model = FakeModule()
        def predict(self, system, user): ...

    client = Local()
    client._init_params({"decoding": "greedy", "seed": 42})
    manifest = client.params_manifest()
    assert manifest["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    json.dumps(manifest)         # must not raise
