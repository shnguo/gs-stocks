from pathlib import Path

from collect_earnings_inputs import collect_jobs, market_input_ready


def test_failed_api_is_not_retried_on_resume_and_other_sources_continue(tmp_path, monkeypatch):
    calls = []

    def capture(job, *args):
        calls.append(job["name"])
        return {**job, "status": "provider_rejected" if job["api"] == "adj_factor" else "captured"}

    monkeypatch.setattr("collect_earnings_inputs.capture_job", capture)
    monkeypatch.setattr("collect_earnings_inputs.time.sleep", lambda _: None)
    jobs = [
        {"name": "factor-1", "api": "adj_factor", "params": {}},
        {"name": "daily-1", "api": "daily", "params": {}},
        {"name": "factor-2", "api": "adj_factor", "params": {}},
    ]
    args = (tmp_path, Path("rust-cli"), Path("repo"), Path("env"), 10, 2)
    first = collect_jobs(jobs, *args)
    assert calls == ["factor-1", "daily-1"]
    assert [row["status"] for row in first] == [
        "provider_rejected", "captured", "skipped_api_failure",
    ]
    calls.clear()
    second = collect_jobs(jobs, *args)
    assert calls == ["daily-1"]  # Revalidate the successful archive only.
    assert second == first


def test_request_budget_preserves_pending_jobs_and_allows_explicit_resume(tmp_path, monkeypatch):
    calls = []

    def capture(job, *args):
        calls.append(job["name"])
        return {**job, "status": "captured"}

    monkeypatch.setattr("collect_earnings_inputs.capture_job", capture)
    monkeypatch.setattr("collect_earnings_inputs.time.sleep", lambda _: None)
    jobs = [{"name": str(i), "api": "daily", "params": {}} for i in range(3)]
    args = (tmp_path, Path("rust-cli"), Path("repo"), Path("env"), 1, 2)
    first = collect_jobs(jobs, *args)
    assert [r["status"] for r in first] == ["captured", "not_requested_budget", "not_requested_budget"]
    second = collect_jobs(jobs, *args)
    assert [r["status"] for r in second] == ["captured", "captured", "not_requested_budget"]
    assert calls.count("2") == 0


def test_missing_reference_blocks_input_but_optional_factor_failure_does_not_certify_execution():
    rows = [
        {"api": "trade_cal", "status": "captured"},
        {"api": "stock_basic", "status": "captured"},
        {"api": "daily", "status": "captured"},
        {"api": "adj_factor", "status": "provider_rejected"},
    ]
    assert market_input_ready(rows)
    rows[2]["status"] = "transport_failed"
    assert not market_input_ready(rows)
    assert not market_input_ready([])
