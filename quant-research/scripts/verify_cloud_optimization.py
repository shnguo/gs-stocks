"""Read-only performance and preservation checks for the research cloud pipeline."""
import argparse
import collections
import datetime
import json
from pathlib import Path

from verify_cloud_migration import query, settings


def timestamp(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = settings(args.config)
    campaign_id = config["RESEARCH_CAMPAIGN_ID"]
    baseline = json.loads(args.baseline.read_text())
    campaign = query(config, "SELECT * FROM research_campaigns WHERE campaign_id=?", [campaign_id])[0]
    progress = query(config, "SELECT status,warehouse_status,count(*) AS jobs,sum(rows) AS rows FROM research_jobs WHERE campaign_id=? GROUP BY status,warehouse_status", [campaign_id])
    runs = query(config, "SELECT * FROM research_pipeline_runs WHERE campaign_id=? ORDER BY started_at", [campaign_id])
    metrics = query(config, "SELECT * FROM research_capture_metrics WHERE campaign_id=? ORDER BY started_at", [campaign_id])
    summaries = []
    for run in runs:
        jobs = [m for m in metrics if m["run_id"] == run["run_id"]]
        summary = dict(run, completed=len(jobs))
        if jobs:
            duration = (max(timestamp(m["finished_at"]) for m in jobs) - min(timestamp(m["started_at"]) for m in jobs)).total_seconds()
            gaps = [(timestamp(b["started_at"]) - timestamp(a["started_at"])).total_seconds() for a, b in zip(jobs, jobs[1:])]
            summary.update(capture_window_seconds=duration, jobs_per_minute=len(jobs) * 60 / duration, rows=sum(m["rows"] for m in jobs), mean_fetch_ms=sum(m["fetch_ms"] for m in jobs)/len(jobs), mean_archive_ms=sum(m["archive_ms"] for m in jobs)/len(jobs), reused_sessions=sum(m["session_reused"] for m in jobs), status_counts=dict(collections.Counter(m["status"] for m in jobs)), source_mix=dict(collections.Counter(m["provider"] + "/" + m["api"] for m in jobs)), minimum_query_start_gap_seconds=min(gaps) if gaps else None)
            events = sorted([(timestamp(m["started_at"]), 1) for m in jobs] + [(timestamp(m["finished_at"]), -1) for m in jobs])
            active = peak = 0
            for _, delta in events:
                active += delta
                peak = max(peak, active)
            summary["maximum_overlapping_captures"] = peak
            assert peak <= run["concurrency"]
            for worker in {m["worker"] for m in jobs}:
                lane = [m for m in jobs if m["worker"] == worker]
                assert all(timestamp(a["finished_at"]) <= timestamp(b["started_at"]) for a, b in zip(lane, lane[1:]))
            assert not gaps or min(gaps) >= 1.95, "source start pacing was violated"
            if run["capture_budget"]:
                assert len(jobs) <= run["capture_budget"], "benchmark exceeded budget"
                if run["status"] == "drained":
                    assert len(jobs) == run["capture_budget"], "bounded benchmark ended short"
        summaries.append(summary)
    result = {"verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "campaign": campaign, "progress": progress, "runs": summaries, "metrics": metrics, "data_ready": False}
    if args.full:
        rows = []
        for offset in range(0, campaign["expected_jobs"], 1000):
            rows += query(config, "SELECT * FROM research_jobs WHERE campaign_id=? ORDER BY ordinal LIMIT 1000 OFFSET ?", [campaign_id, offset])
        assert len(rows) == len(baseline["jobs"]) == campaign["expected_jobs"]
        actual = {r["job_id"]: r for r in rows}
        assert len(actual) == len(rows)
        protected = collections.Counter()
        for old in baseline["jobs"]:
            current = actual[old["job_id"]]
            assert old["ordinal"] == current["ordinal"]
            assert json.loads(old["identity_json"]) == json.loads(current["identity_json"])
            if old["status"] != "pending":
                for field in ["status", "rows", "raw_key", "raw_sha256", "capture_path", "last_error"]:
                    assert old[field] == current[field], (old["job_id"], field)
                protected[old["status"]] += 1
            if old["warehouse_status"] == "published":
                assert current["warehouse_status"] == "published" and current["commit_id"] == old["commit_id"]
        assert len({m["job_id"] for m in metrics}) == len(metrics)
        result.update(all_jobs_preserved=len(rows), protected_states_preserved=dict(protected), published_jobs=sum(r["warehouse_status"] == "published" for r in rows), warehouse_rows=sum(r["rows"] for r in rows if r["warehouse_status"] == "published"))
        result["published_nonempty_jobs"] = sum(r["warehouse_status"] == "published" and r["rows"] > 0 for r in rows)
        result["new_captures"] = [{k:r[k] for k in ["job_id","identity_json","status","raw_key","raw_sha256","capture_path","rows"]} for r in rows if r["job_id"] in {m["job_id"] for m in metrics}]
    with args.output.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ["metrics", "new_captures"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
