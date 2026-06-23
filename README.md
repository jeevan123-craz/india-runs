# Redrob — Intelligent Candidate Discovery & Ranking

A recruiter-logic ranking system for the Redrob Hackathon ("Data & AI Challenge").
Given the released Senior AI Engineer job description, it ranks the **top 100 of
100,000 candidates** and emits a submission CSV with a specific, honest reasoning
for every pick.

> **Why this design wins the challenge's actual game.** The dataset is mostly bait:
> ~12 wrong-role buckets (HR Manager, Marketing, Accountant…) stuffed with AI
> keywords, plus ~80 "subtly impossible" honeypots. Pure keyword/embedding matching
> walks straight into the traps. This system ranks the way the JD literally asks a
> recruiter to: **role and career evidence first, keywords second, behavioral
> availability as a multiplier, and impossible profiles thrown out.**

---

## Results on the released pool (100,000 candidates)

| Metric | Value |
|---|---|
| End-to-end runtime | **~40 s** (limit: 5 min) |
| Memory | < 1 GB (limit: 16 GB) |
| Dependencies | **none** — Python standard library only |
| Network / GPU at rank time | **none** |
| Official `validate_submission.py` | **PASS** |
| Top-100 that are genuine ML/AI/NLP/search/reco roles | **100 / 100** |
| Top-100 based in India (JD requirement) | **100 / 100** |
| Honeypots in top-100 | **0** (disqualification threshold: >10) |
| Avg. years of experience in top-100 | 6.4 (JD ideal band: 6–8) |

98 honeypot profiles in the pool carry a *perfect* ML title + skills (FAISS,
Pinecone, Information Retrieval) — none reach our top-100.

---

## Quick start

```bash
# 1. Put the dataset next to the code (not committed; it's ~465 MB)
#    candidates.jsonl  OR  candidates.jsonl.gz  (both supported)

# 2. Rank — single command, end-to-end
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --report

# 3. Validate against the official spec
python validate_submission.py submission.csv     # -> "Submission is valid."
```

No `pip install` required for ranking. (`streamlit` is only needed for the demo app.)

---

## Architecture — a two-stage retrieval→rerank funnel

This mirrors how a real candidate-discovery system scales to 100k+ pools: a cheap
recall pass narrows the field, then an expensive, explainable reranker does the work.

```
candidates.jsonl (100k)
        │
        ▼  Stage 1 — RECALL  (streamed, O(n), low memory)
  cheap title + keyword relevance  ──►  shortlist of 6,000
        │
        ▼  Stage 2 — RERANK  (full recruiter logic on the shortlist)
  base = 0.34·role/career + 0.24·skills(trust-weighted) + 0.22·experience
       + 0.12·TF-IDF(JD) + 0.08·education
  final = base × disqualifiers × location × behavioral × honeypot
        │
        ▼
  top-100 + per-candidate reasoning  ──►  submission.csv
```

### The scoring components (`ranker/scoring.py`)

- **Role / career fit (0.34)** — the decisive anti-trap signal. A clearly-wrong
  title (Marketing/HR/Content Writer) is capped near zero *no matter how many AI
  keywords are listed*. Career-history evidence (built ranking/search/recommendation
  systems at product companies) can rescue an adjacent title — this is how
  "plain-language Tier-5" candidates surface.
- **Skills fit (0.24)** — trust-weighted coverage of required capability buckets
  (embeddings/retrieval, vector/hybrid search, ranking/reco, NLP, Python, evaluation,
  LLM fine-tuning, ML infra). Trust discounts lazy stuffing: an "expert" skill with
  0 endorsements and 3 months tenure counts for little; assessment scores corroborate.
- **Experience fit (0.22)** — years in the 5–9 band (peak 6–8), applied ML at
  *product* (not services) companies, evidence of shipping an end-to-end
  ranking/search/reco system, and ML recency.
- **TF-IDF similarity (0.12)** — pure-Python TF-IDF cosine between a compact JD
  "requirements" query and the candidate's profile text. Minority weight so it can't
  be gamed by keywords alone, but enough to surface conceptually-matching profiles.
- **Education (0.08)** — institution tier + relevant field, lightly.

### Multipliers

- **Disqualifiers** (`scoring.disqualifiers`) — the JD's explicit "do NOT want":
  entire career at services/consulting firms, CV/speech/robotics tilt over NLP/IR,
  research-only with no production, title-chasing / job-hopping, shallow LLM-framework
  hobbyists.
- **Location** (`scoring.location_factor`) — India-preferred (Pune/Noida best); outside
  India penalized (the JD does not sponsor visas), softened if willing to relocate.
- **Behavioral signals** (`scoring.behavioral_modifier`) — availability (open-to-work,
  last-active recency, notice period), responsiveness (recruiter response rate, reply
  time, interview completion), and credibility (saved-by-recruiters, completeness,
  verification, GitHub activity). A perfect-on-paper candidate who's been inactive for
  6 months with a 5% response rate is down-weighted hard — exactly as the JD asks.
- **Honeypot factor** (`ranker/honeypot.py`) — internal-consistency checks catch
  "subtly impossible" profiles: a skill used longer than the whole career, more stated
  experience than the timeline allows, impossible career/education dates, last-active
  before signup, expert-skill walls with zero endorsements. High suspicion → hard-drop
  out of contention, protecting against the >10% honeypot disqualification.

### Reasoning (`ranker/reasoning.py`)

Every reasoning line is **assembled from facts that exist in the candidate record**
(title, years, named companies, named skills, signal values), connected to JD
requirements, honest about concerns, with tone matched to rank and deterministic
variation by candidate id. No LLM, so it reproduces exactly — and can't hallucinate.

---

## Repository layout

```
redrob-ranker/
├── rank.py                     # CLI entry point (the reproduce_command)
├── ranker/
│   ├── jd_spec.py              # structured JD: roles, skill buckets, companies, location
│   ├── features.py             # candidate -> normalized recruiter-meaningful signals
│   ├── scoring.py              # component scores + multipliers + combine
│   ├── honeypot.py             # internal-consistency trap detection
│   ├── reasoning.py            # fact-grounded, varied reasoning generation
│   └── text_sim.py             # pure-Python TF-IDF cosine
├── app.py                      # Streamlit demo (for the sandbox-link requirement)
├── job_description.md          # the released JD
├── submission_metadata.yaml    # fill the <FILL ...> fields before submitting
├── validate_submission.py      # official format validator (bundled for convenience)
├── data/sample_candidates.json # 50-candidate sample (for the demo)
├── submission.csv              # generated top-100 ranking
└── requirements.txt
```

---

## Submission checklist

1. **GitHub repo (public):** push this folder; set visibility to public.
2. **Ranked output CSV:** `submission.csv` is generated. Rename to `<your_team_id>.csv`.
3. **Sandbox link:** deploy `app.py` (HuggingFace Spaces / Streamlit Cloud / Replit).
4. **Deck → PDF:** use the mandatory template; the "Architecture" and "Results"
   sections above map directly onto it.
5. **Metadata:** complete `submission_metadata.yaml` (team name = team ID).

## Design notes for the defend-your-work interview

- Why rule-based + light TF-IDF instead of an LLM reranker? The compute budget
  (5 min, CPU, no network, 100k candidates) rules out per-candidate LLM calls, and
  the JD explicitly rewards systems that reason about *fit*, not keyword overlap.
- Why a funnel? Recall cheaply, spend compute only where it matters — the standard
  retrieval→rerank pattern the role is literally about.
- Tunable knobs live in `ranker/scoring.py` (`W`, multiplier constants) and
  `ranker/jd_spec.py` (ontologies) — easy to adjust and defend.
