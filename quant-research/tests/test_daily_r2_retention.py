from daily_token_cycle import daily_retention_sources


def test_daily_retention_includes_run_source_reports_and_code(tmp_path):
    run = tmp_path / "artifacts/daily-token-live-v1/runs/2026-09-22"
    source = tmp_path / "artifacts/daily-token-live-v1/data/2026-09-22/input"
    report = tmp_path / "artifacts/daily-token-live-v1/reports/2026-09-22/report.md"
    config = tmp_path / "configs/daily-token-v1.json"
    for path in (run, source, report.parent, config.parent):
        path.mkdir(parents=True, exist_ok=True)
    (run / "binding.json").write_text('{"source": "' + str(source) + '"}')
    report.write_text("report")
    config.write_text("{}")
    selected = daily_retention_sources(
        {"run": str(run), "signal": "2026-09-22", "report": str(report)}, config
    )
    assert run.resolve() in selected
    assert source.parent.resolve() in selected
    assert report.parent.resolve() in selected
    assert config.resolve() in selected
    assert any(path.name == "src" for path in selected)
