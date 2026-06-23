#!/usr/bin/env python3
"""
Redrob — Intelligent Candidate Discovery & Ranking
Produces submission.csv (top-100) from candidates.jsonl for the released JD.

Two-stage architecture (mirrors a real production recruiting funnel):
  Stage 1  RECALL   — stream all 100k candidates, score a cheap relevance prelim,
                      keep the most promising N for re-ranking. O(n), low memory.
  Stage 2  RERANK   — full recruiter-logic scoring on the shortlist:
                      role/career fit, trust-weighted skills, experience, TF-IDF
                      similarity, education, x disqualifiers x location x behavioral
                      signals x honeypot factor. Top-100 emitted with reasoning.

Constraints honored: CPU-only, no network, < 5 min, < 16 GB. No LLM calls.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
from __future__ import annotations
import argparse, csv, gzip, heapq, io, json, re, sys, time

from ranker import jd_spec as J
from ranker.features import extract, relevance_prelim
from ranker import scoring, reasoning
from ranker.honeypot import suspicion as hp_suspicion
from ranker.text_sim import similarities

# cheap Stage-1 relevance: one compiled regex over a light text blob
_PRELIM_TERMS = sorted(set(
    J.NLP_IR_TERMS + ["machine learning", "ml", "applied ai", "data science",
                      "recommendation", "recommender", "search", "ranking",
                      "retrieval", "embedding", "relevance", "personalization"]),
    key=len, reverse=True)
_PRELIM_RE = re.compile("|".join(re.escape(t) for t in _PRELIM_TERMS))


def _open(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def cheap_prelim(c: dict) -> float:
    p = c.get("profile", {})
    title = (p.get("current_title", "") or "").lower()
    blob_parts = [title, (p.get("headline", "") or ""), (p.get("summary", "") or "")]
    for h in c.get("career_history", []) or []:
        blob_parts.append(h.get("title", "") or "")
        blob_parts.append(h.get("description", "") or "")
    for s in c.get("skills", []) or []:
        blob_parts.append(s.get("name", "") or "")
    blob = " ".join(blob_parts).lower()
    score = 0.0
    if any(t in title for t in J.STRONG_TITLES):
        score += 3.0
    elif any(t in title for t in J.ADJACENT_TITLES):
        score += 1.2
    if any(t in title for t in J.WRONG_TITLES):
        score -= 1.5
    score += min(4.0, 0.25 * len(_PRELIM_RE.findall(blob)))
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidates.jsonl or .jsonl.gz")
    ap.add_argument("--jd", default="job_description.md", help="(informational) JD file")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--funnel", type=int, default=6000, help="shortlist size for re-ranking")
    ap.add_argument("--report", action="store_true", help="print a sanity report")
    args = ap.parse_args()

    t0 = time.time()

    # ---------- Stage 1: streaming recall ----------
    heap = []   # (prelim, counter, candidate)
    cnt = 0
    n = 0
    with _open(args.candidates) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            c = json.loads(line)
            pre = cheap_prelim(c)
            if len(heap) < args.funnel:
                heapq.heappush(heap, (pre, cnt, c)); cnt += 1
            elif pre > heap[0][0]:
                heapq.heappushpop(heap, (pre, cnt, c)); cnt += 1
    t1 = time.time()
    shortlist = [c for _, _, c in heap]
    print(f"[stage1] streamed {n} candidates -> shortlist {len(shortlist)} in {t1-t0:.1f}s",
          file=sys.stderr)

    # ---------- Stage 2: rerank ----------
    feats = [extract(c) for c in shortlist]
    sims = similarities([f.blob for f in feats])
    rows = []
    for c, f, sim in zip(shortlist, feats, sims):
        susp, flags = hp_suspicion(c, f)
        final, detail = scoring.score_candidate(c, f, float(sim), susp)
        rows.append((final, f, detail, c))
    t2 = time.time()
    print(f"[stage2] reranked {len(rows)} in {t2-t1:.1f}s", file=sys.stderr)

    # sort: score desc, candidate_id asc (validator-safe tie-break)
    rows.sort(key=lambda r: (-round(r[0], 6), r[1].cid))
    top = rows[:args.topk]
    max_final = max((r[0] for r in top), default=1.0) or 1.0

    # ---------- write submission ----------
    with open(args.out, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (final, f, detail, c) in enumerate(top, start=1):
            score_norm = round(final / max_final, 6)   # 1.0 at rank 1, monotonic
            reason = reasoning.make(f, detail, i)
            w.writerow([f.cid, i, f"{score_norm:.6f}", reason])
    print(f"[done] wrote {args.out} ({len(top)} rows) in {time.time()-t0:.1f}s total",
          file=sys.stderr)

    if args.report:
        _report(top)


def _report(top):
    import collections
    print("\n================ SANITY REPORT ================")
    titles = collections.Counter(f.title for _, f, _, _ in top)
    print("top-100 title mix:")
    for t, ct in titles.most_common(12):
        print(f"   {ct:3d}  {t}")
    india = sum(1 for _, f, _, _ in top if f.india)
    susp_hi = sum(1 for _, f, d, _ in top if d["susp"] >= 0.5)
    otw = sum(1 for _, f, _, _ in top if f.sig.get("open_to_work_flag"))
    avg_rr = sum((f.sig.get("recruiter_response_rate", 0) or 0) for _, f, _, _ in top) / len(top)
    avg_yoe = sum(f.yoe for _, f, _, _ in top) / len(top)
    print(f"\nIndia-based: {india}/100 | open_to_work: {otw}/100 | "
          f"avg response: {avg_rr:.0%} | avg YoE: {avg_yoe:.1f}")
    print(f"high-suspicion (honeypot) in top-100: {susp_hi}  (DQ threshold = 10)")
    print("\nTOP 15:")
    for i, (final, f, d, c) in enumerate(top[:15], 1):
        print(f"  {i:2d}. {f.cid}  {f.title[:34]:34s} yoe {f.yoe:4.1f} "
              f"role {d['role']:.2f} sk {d['skills']:.2f} exp {d['experience']:.2f} "
              f"beh {d['beh_m']:.2f} {f.country_l}")


if __name__ == "__main__":
    main()
