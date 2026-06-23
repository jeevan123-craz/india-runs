"""
Hosted demo for the sandbox-link submission requirement.

Run locally:   streamlit run app.py
Deploy:        push this repo to HuggingFace Spaces / Streamlit Cloud / Replit.

Lets a reviewer rank a small candidate sample (bundled, or their own uploaded
.jsonl) and inspect the per-candidate reasoning. Uses the exact same ranker
package as the CLI — no separate logic.
"""
import json
import streamlit as st

from ranker.features import extract
from ranker.text_sim import similarities
from ranker import scoring, reasoning
from ranker.honeypot import suspicion

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Discovery & Ranking")
st.caption("Recruiter-logic ranker · CPU-only · no network · no LLM calls at rank time")


@st.cache_data(show_spinner=False)
def load_default():
    with open("data/sample_candidates.json") as f:
        return json.load(f)


def rank(cands, topk):
    feats = [extract(c) for c in cands]
    sims = similarities([f.blob for f in feats])
    scored = []
    for c, f, sim in zip(cands, feats, sims):
        susp, _ = suspicion(c, f)
        final, detail = scoring.score_candidate(c, f, float(sim), susp)
        scored.append((final, f, detail))
    scored.sort(key=lambda r: (-round(r[0], 6), r[1].cid))
    top = scored[:topk]
    mx = max((r[0] for r in top), default=1.0) or 1.0
    out = []
    for i, (final, f, detail) in enumerate(top, 1):
        out.append({"rank": i, "candidate_id": f.cid, "title": f.title,
                    "yoe": f.yoe, "score": round(final / mx, 4),
                    "reasoning": reasoning.make(f, detail, i)})
    return out


up = st.file_uploader("Upload candidates .jsonl (or use the bundled 50-candidate sample)", type=["jsonl"])
if up:
    cands = [json.loads(l) for l in up.read().decode("utf-8").splitlines() if l.strip()]
else:
    cands = load_default()

topk = st.slider("Show top-K", 5, min(100, len(cands)), min(20, len(cands)))
st.write(f"Ranking **{len(cands)}** candidates …")
rows = rank(cands, topk)
st.dataframe(rows, use_container_width=True, hide_index=True)
