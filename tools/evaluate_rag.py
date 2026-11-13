"""Reproducible, offline retrieval and grounding check for the bundled RAG set."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from medvision_ai.rag import RAGService  # noqa: E402


def main() -> None:
    cases = json.loads((ROOT / "tests/fixtures/rag-evaluation.json").read_text())
    service = RAGService()
    recall = citation_precision = grounded = 0.0
    latencies = []
    for case in cases:
        started = time.perf_counter_ns()
        ranked = service.embedder.search(case["question"], 3)
        retrieved = {chunk.source_id for chunk, _score in ranked}
        recall += len(retrieved & set(case["supporting_sources"])) / len(case["supporting_sources"])
        result = service.answer(case["question"])
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        citations = {item["sourceId"] for item in result["citations"]}
        citation_precision += (len(citations & retrieved) / len(citations)) if citations else 0.0
        grounded += float(case["expected_fact"].lower() in result["answer"].lower())
    count = len(cases)
    sorted_latencies = sorted(latencies)
    print(json.dumps({
        "cases": count,
        "supporting_source_recall_at_3": round(recall / count, 4),
        "citation_precision": round(citation_precision / count, 4),
        "expected_fact_grounding_rate": round(grounded / count, 4),
        "latency_ms_mean": round(sum(latencies) / count, 3),
        "latency_ms_p95": round(sorted_latencies[min(count - 1, int(count * .95))], 3),
        "note": "Small deterministic curated-set smoke evaluation, not a clinical or external benchmark."
    }, indent=2))


if __name__ == "__main__":
    main()
