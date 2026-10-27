import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from Agent.agent import answer_question
from Pipeline.extraction import norm
from Pipeline.process_document import process_document
from db.database import (
    create_workspace, delete_workspace, get_workspace, list_documents,
    list_workspaces, register_document,
)

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
DATASET = ROOT / "dataset.json"
REPORT = ROOT / "last_report.json"
SESSION = "eval-cli"

NOT_FOUND_HINTS = [
    "does not", "do not", "doesn't", "not mention", "no information", "not specified",
    "not stated", "not provided", "not found", "cannot find", "can't find", "could not find",
    "couldn't find", "no mention", "not included", "isn't", "is not",
]


def setup_workspace(reuse: bool) -> str:
    existing = list_workspaces(SESSION)
    if reuse and existing:
        return str(existing[0]["workspace_id"])
    for w in existing:
        delete_workspace(str(w["workspace_id"]), SESSION)

    ws_id = str(create_workspace(SESSION, "Eval workspace")["workspace_id"])
    for path in sorted(SAMPLES.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        reg = register_document(ws_id, path.name, digest, force=True)
        print(f"processing {path.name} ...", flush=True)
        process_document(str(path), reg["doc_id"])

    for d in list_documents(ws_id):
        not_done = [n for n, b in (d["branches"] or {}).items() if b["status"] != "done"]
        note = "" if not not_done else "  (steps not done: " + ", ".join(not_done) + ")"
        print(f"  {d['filename']}: {d['status']}{note}")
    return ws_id


def check_case(case: dict, out: dict) -> list[str]:
    failures = []
    answer = out["answer"]
    n_answer = norm(answer)

    for group in case.get("expect", []):
        if not any(norm(alt) in n_answer for alt in group):
            failures.append("missing: " + " / ".join(group))
    for bad in case.get("must_not_include", []):
        if norm(bad) in n_answer:
            failures.append("must not include: " + bad)
    for bad in case.get("forbid_exact", []):
        if n_answer.strip(" .\"'") == norm(bad):
            failures.append("answer is only: " + bad)
    if case.get("expect_not_found") and not any(h in n_answer for h in NOT_FOUND_HINTS):
        failures.append("should say the information is not in the documents")
    if case.get("tools_any"):
        used = {s["tool"] for s in out["steps"]}
        if not used & set(case["tools_any"]):
            failures.append("expected one of the tools: " + ", ".join(case["tools_any"]))

    status = (out.get("verification") or {}).get("status")
    if status not in ("grounded", "no_claims"):
        failures.append(f"verification status: {status}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the answer-quality evals")
    ap.add_argument("--reuse", action="store_true", help="reuse the eval workspace (skips re-processing the samples)")
    ap.add_argument("--clean", action="store_true", help="delete the eval workspace at the end")
    ap.add_argument("--only", default="", help="comma-separated case ids to run")
    ap.add_argument("--min-pass", type=float, default=0.8)
    ap.add_argument("--min-faithfulness", type=float, default=0.9)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cases = json.loads(DATASET.read_text())
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    ws_id = setup_workspace(args.reuse)
    isolated = (get_workspace(ws_id, SESSION) is not None
                and get_workspace(ws_id, "another-visitor") is None)
    print(f"\nisolation check (another visitor cannot open this workspace): {'PASS' if isolated else 'FAIL'}\n")

    results = []
    for case in cases:
        t0 = time.time()
        out = answer_question(ws_id, case["question"], [])
        seconds = time.time() - t0
        failures = check_case(case, out)
        v = out.get("verification") or {}
        results.append({
            "id": case["id"], "question": case["question"], "passed": not failures, "failures": failures,
            "answer": out["answer"], "status": v.get("status"), "faithfulness": v.get("faithfulness"),
            "sources": v.get("sources", []), "repaired": v.get("repaired", False),
            "tools": [s["tool"] for s in out["steps"]], "seconds": round(seconds, 1),
        })
        mark = "PASS" if not failures else "FAIL"
        faith = "-" if v.get("faithfulness") is None else f"{v['faithfulness']:.2f}"
        print(f"{mark}  {case['id']:<22} {str(v.get('status')):<10} faith={faith:<5} "
              f"{seconds:5.1f}s  tools={','.join(results[-1]['tools']) or '-'}")
        for f in failures:
            print(f"        - {f}")

    scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    sources = [s for r in results for s in r["sources"]]
    metrics = {
        "cases": len(results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3) if results else 0,
        "avg_faithfulness": round(sum(scores) / len(scores), 3) if scores else None,
        "citation_validity": round(sum(s["verified"] for s in sources) / len(sources), 3) if sources else None,
        "answers_repaired": sum(r["repaired"] for r in results),
        "avg_tool_calls": round(sum(len(r["tools"]) for r in results) / len(results), 2) if results else 0,
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 1) if results else 0,
        "isolation": isolated,
    }
    print("\nMETRICS:", json.dumps(metrics, indent=2))

    if REPORT.exists():
        prev = json.loads(REPORT.read_text()).get("metrics", {})
        for key in ("pass_rate", "avg_faithfulness", "citation_validity"):
            if prev.get(key) is not None and metrics.get(key) is not None:
                print(f"{key}: {prev[key]} -> {metrics[key]} ({metrics[key] - prev[key]:+.3f})")
    REPORT.write_text(json.dumps(
        {"time": datetime.now().isoformat(timespec="seconds"), "metrics": metrics, "results": results},
        indent=2, ensure_ascii=False))

    if args.clean:
        delete_workspace(ws_id, SESSION)

    problems = []
    if not isolated:
        problems.append("isolation check failed")
    if metrics["pass_rate"] < args.min_pass:
        problems.append(f"pass rate {metrics['pass_rate']} is below {args.min_pass}")
    if metrics["avg_faithfulness"] is not None and metrics["avg_faithfulness"] < args.min_faithfulness:
        problems.append(f"faithfulness {metrics['avg_faithfulness']} is below {args.min_faithfulness}")
    if problems:
        print("\nFAILED: " + "; ".join(problems))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())