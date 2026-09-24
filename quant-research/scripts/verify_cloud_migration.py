"""Independent, read-only verification of a Cloudflare research migration."""

import argparse
import collections
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path


def settings(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    if "RESEARCH_SECRETS_FILE" in values:
        values.update(settings(Path(values.pop("RESEARCH_SECRETS_FILE"))))
    return values


def query(config, sql, params):
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + config["CLOUDFLARE_ACCOUNT_ID"]
        + "/d1/database/"
        + config["D1_DATABASE_ID"]
        + "/query"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps({"sql": sql, "params": params}).encode(),
        headers={
            "Authorization": "Bearer " + config["CLOUDFLARE_D1_API_TOKEN"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert result["success"] and result["result"][0]["success"]
    return result["result"][0]["results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = settings(args.config)
    plan = json.loads((args.campaign / "plan.json").read_text())
    receipt = json.loads(args.receipt.read_text())
    restored = Path(receipt["restore_directory"])
    manifest = json.loads((restored / "ARCHIVE-MANIFEST.json").read_text())
    total_bytes = 0
    for name, expected in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        data = (restored / relative).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
        total_bytes += len(data)
    campaign_id = config["RESEARCH_CAMPAIGN_ID"]
    cloud = query(config, "SELECT * FROM research_campaigns WHERE campaign_id=?", [campaign_id])[0]
    assert cloud["archive_key"] == receipt["key"]
    assert cloud["archive_sha256"] == receipt["sha256"]
    assert cloud["expected_jobs"] == len(plan["jobs"])
    rows = []
    for offset in range(0, len(plan["jobs"]), 1000):
        rows.extend(
            query(
                config,
                "SELECT * FROM research_jobs WHERE campaign_id=? ORDER BY ordinal LIMIT 1000 OFFSET ?",
                [campaign_id, offset],
            )
        )
    assert len(rows) == len(plan["jobs"])
    assert len({row["job_id"] for row in rows}) == len(rows)
    by_id = {row["job_id"]: row for row in rows}
    for index, job in enumerate(plan["jobs"]):
        actual = by_id[job["job_id"]]
        assert actual["ordinal"] == index
        assert json.loads(actual["identity_json"]) == job["identity"]
    original_report = max(
        (restored / "data/backfills" / args.campaign.name / "reports").glob("*.json")
    )
    original = json.loads(original_report.read_text())
    preserved = collections.Counter()
    for job_id, state in original["jobs"].items():
        if state["status"] != "pending":
            assert by_id[job_id]["status"] == state["status"], job_id
            assert by_id[job_id]["raw_key"] == receipt["key"]
            if state["status"] == "captured_raw_verified":
                assert by_id[job_id]["rows"] == state["rows"]
            preserved[state["status"]] += 1
    result = {
        "verified_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "campaign_id": campaign_id,
        "all_full_market_jobs_retained": len(rows),
        "restored_files_verified": len(manifest["files"]),
        "restored_bytes_verified": total_bytes,
        "original_states_preserved": dict(preserved),
        "cloud_status": cloud["status"],
        "job_counts": dict(collections.Counter(row["status"] for row in rows)),
        "warehouse_counts": dict(collections.Counter(row["warehouse_status"] for row in rows)),
        "warehouse_rows": sum(
            row["rows"] for row in rows if row["warehouse_status"] == "published"
        ),
        "data_ready": False,
    }
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
