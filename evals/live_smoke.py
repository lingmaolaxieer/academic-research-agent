"""Small, budgeted live acceptance run. Never logs provider credentials."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv


PARAMETERS = {
    "case_parallelism": 1,
    "max_workers": 2,
    "max_research_subtasks": 2,
    "max_research_iterations": 1,
    "max_researcher_model_calls": 3,
    "max_researcher_tool_calls": 3,
    "max_model_calls": 16,
    "model_timeout_seconds": 45,
    "task_timeout_seconds": 180,
    "retry_max_attempts": 2,
    "max_output_repairs": 1,
    "max_context_chars": 10000,
    "poll_interval_seconds": 3,
    "poll_deadline_seconds": 210,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[], help="Run only these case IDs")
    parser.add_argument("--embedding-probe", action="store_true")
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error("The env file does not exist")
    load_dotenv(args.env_file, override=True)

    from fastapi.testclient import TestClient

    from research_agent.config import settings

    # Keep the real chat/search providers while avoiding production data and
    # Docker dependencies. These changes affect this process only.
    settings.runtime_profile = "local"
    settings.task_backend = "thread"
    settings.rate_limit_backend = "memory"
    settings.knowledge_transport = "local"
    settings.retrieval_mode = "bm25"
    settings.reranker_model = ""
    settings.max_workers = PARAMETERS["max_workers"]
    settings.max_research_subtasks = PARAMETERS["max_research_subtasks"]
    settings.max_research_iterations = PARAMETERS["max_research_iterations"]
    settings.max_researcher_model_calls = PARAMETERS["max_researcher_model_calls"]
    settings.max_researcher_tool_calls = PARAMETERS["max_researcher_tool_calls"]
    settings.max_model_calls = PARAMETERS["max_model_calls"]
    settings.model_timeout_seconds = PARAMETERS["model_timeout_seconds"]
    settings.task_timeout_seconds = PARAMETERS["task_timeout_seconds"]
    settings.retry_max_attempts = PARAMETERS["retry_max_attempts"]
    settings.max_output_repairs = PARAMETERS["max_output_repairs"]
    settings.max_context_chars = PARAMETERS["max_context_chars"]

    from research_agent.api import create_api

    dataset = json.loads(Path(__file__).with_name("live_smoke_cases.json").read_text(
        encoding="utf-8"
    ))
    selected = set(args.case)
    cases = [case for case in dataset["cases"] if not selected or case["id"] in selected]
    if len(cases) != len(selected) and selected:
        parser.error("At least one requested case ID was not found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"parameters": PARAMETERS, "cases": [], "embedding_probe": None}
    headers = {}
    if settings.api_keys:
        headers["X-API-Key"] = next(iter(settings.api_keys.values()))

    with tempfile.TemporaryDirectory(prefix="research-live-", dir=args.output.parent) as directory:
        root = Path(directory)
        settings.knowledge_db_path = str(root / "knowledge.db")
        settings.job_db_path = str(root / "jobs.db")
        settings.checkpoint_db_path = str(root / "checkpoints.db")

        if args.embedding_probe:
            try:
                from research_agent.knowledge import embedding_client

                started = time.monotonic()
                vectors = embedding_client().embed_documents([
                    "可见光目标检测", "多模态目标检测"
                ])
                output["embedding_probe"] = {
                    "status": "passed" if len(vectors) == 2 else "failed",
                    "dimensions": len(vectors[0]) if vectors else None,
                    "elapsed_s": round(time.monotonic() - started, 2),
                }
            except Exception as exc:
                output["embedding_probe"] = {
                    "status": "failed", "error_type": type(exc).__name__
                }
            print("embedding:", output["embedding_probe"]["status"], flush=True)

        with TestClient(create_api()) as client:
            doc_ids = []
            for document in dataset["documents"]:
                response = client.post(
                    "/documents", headers=headers,
                    files={"file": (document["name"],
                                    document["content"].encode("utf-8"), "text/markdown")},
                )
                response.raise_for_status()
                doc_ids.append(response.json()["document_id"])

            for case in cases:
                started = time.monotonic()
                body = {"query": case["query"], "mode": case["mode"],
                        "depth": case["depth"], "require_approval": False,
                        "document_ids": doc_ids if case["mode"] != "web" else []}
                response = client.post("/research", headers=headers, json=body)
                if response.status_code != 202:
                    entry = {"id": case["id"], "status": "submit_failed",
                             "http_status": response.status_code}
                    output["cases"].append(entry)
                    print(case["id"], entry["status"], flush=True)
                    continue
                thread_id = response.json()["thread_id"]
                final = None
                deadline = time.monotonic() + PARAMETERS["poll_deadline_seconds"]
                while time.monotonic() < deadline:
                    status_response = client.get(f"/research/{thread_id}", headers=headers)
                    if status_response.status_code == 429:
                        time.sleep(min(60, max(3, int(status_response.headers.get(
                            "Retry-After", "3"
                        )))))
                        continue
                    if status_response.status_code != 200:
                        final = {"status": "status_lookup_failed",
                                 "error": f"HTTP {status_response.status_code}"}
                        break
                    final = status_response.json()
                    if final["status"] in {"completed", "error", "cancelled", "interrupted"}:
                        break
                    time.sleep(PARAMETERS["poll_interval_seconds"])

                if final and final["status"] in {"queued", "running"}:
                    client.post(f"/research/{thread_id}/cancel", headers=headers)
                    final["status"] = "poll_deadline_exceeded"

                report = (final or {}).get("report") or {}
                evidence = (final or {}).get("evidence") or []
                text = "\n".join([report.get("summary", ""),
                                   *(report.get("sections") or [])])
                all_report_text = "\n".join([text, *(report.get("limitations") or [])])
                listed = set(report.get("citations") or [])
                inline = set(re.findall(r"\[(ev_[a-zA-Z0-9_]+)\]", all_report_text))
                available = {item["evidence_id"] for item in evidence}
                source_types = sorted({item["source_type"] for item in evidence})
                entry = {
                    "id": case["id"], "thread_id": thread_id,
                    "status": (final or {}).get("status"),
                    "elapsed_s": round(time.monotonic() - started, 2),
                    "model_calls": ((final or {}).get("metrics") or {}).get("model_calls"),
                    "source_types": source_types,
                    "expected_source_found": case["expected_source"] in source_types,
                    "citation_ids_valid": bool(listed) and listed == inline
                    and listed.issubset(available),
                    "summary_cited": bool(re.search(
                        r"\[ev_[a-zA-Z0-9_]+\]", report.get("summary", "")
                    )),
                    "sections_cited": all(
                        len(section.strip()) < 50 or bool(re.search(
                            r"\[ev_[a-zA-Z0-9_]+\]", section
                        )) for section in (report.get("sections") or [])
                    ) and (not report.get("sections") or any(
                        re.search(r"\[ev_[a-zA-Z0-9_]+\]", section)
                        for section in report["sections"]
                    )),
                    "expected_terms_found": {
                        pattern: bool(re.search(pattern, text, re.IGNORECASE))
                        for pattern in case["expected_terms"]
                    },
                    "guardrail_flags": (final or {}).get("guardrail_flags", []),
                    "source_errors": (final or {}).get("source_errors", []),
                    "error": (final or {}).get("error"),
                    "report": report,
                }
                output["cases"].append(entry)
                args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
                print(case["id"], entry["status"], entry["elapsed_s"],
                      "s", flush=True)

    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return 0 if all(case["status"] == "completed" for case in output["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
