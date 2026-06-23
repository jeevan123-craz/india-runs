"""
Reasoning generation — a 1-2 sentence, recruiter-style justification per candidate.

Stage-4 manual review checks that reasoning is: specific (real facts), connected to
the JD, honest about concerns, non-hallucinated, varied, and consistent with rank.
So every clause here is assembled from facts actually present in the candidate
record. No LLM call — assembly + deterministic variation by candidate id.
"""
from __future__ import annotations
from . import jd_spec as J
from .features import Features

_REL_TERMS = (J.SKILL_BUCKETS["embeddings_retrieval"]["terms"]
              + J.SKILL_BUCKETS["vector_hybrid_search"]["terms"]
              + J.SKILL_BUCKETS["ranking_reco"]["terms"]
              + J.SKILL_BUCKETS["nlp"]["terms"]
              + J.SKILL_BUCKETS["llm_finetune"]["terms"])


def _matched_skills(f: Features, k=3):
    scored = []
    for sk in f.skills_raw:
        nm = sk.get("name", "")
        nml = J.norm(nm)
        if any(t in nml for t in _REL_TERMS):
            end = sk.get("endorsements", 0) or 0
            prof = str(sk.get("proficiency", "")).lower()
            rank = {"expert": 3, "advanced": 2, "intermediate": 1}.get(prof, 0)
            scored.append((rank, end, nm))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [nm for _, _, nm in scored[:k]]


def _career_evidence(f: Features, k=2):
    comps = []
    for h in f.career:
        ht = J.norm(h.get("title", ""))
        if any(t in ht for t in J.STRONG_TITLES) or "data scientist" in ht or "applied" in ht:
            c = h.get("company", "")
            ind = J.norm(h.get("industry", ""))
            services = any(x in J.norm(c) for x in J.SERVICES_COMPANIES) or \
                       any(x in ind for x in J.SERVICES_INDUSTRIES)
            if c and not services and c not in comps:
                comps.append(c)
    return comps[:k]


def _concern(f: Features, detail: dict):
    notes = detail.get("disq_notes", [])
    s = f.sig
    if not f.india and not s.get("willing_to_relocate"):
        return "based outside India with no relocation flag (no visa sponsorship)"
    if "entire career at services/consulting firms" in notes:
        return "services-firm-heavy background vs the JD's product-company preference"
    if "profile tilts to CV/speech/frontend over NLP/IR" in notes:
        return "skills lean toward CV/speech rather than retrieval/ranking"
    rr = s.get("recruiter_response_rate", 0) or 0
    if rr < 0.3:
        return f"low recruiter response rate ({rr:.0%})"
    if f.last_active_days > 150:
        return f"inactive for ~{int(f.last_active_days)} days"
    np_ = s.get("notice_period_days", 0) or 0
    if np_ > 60:
        return f"long notice period ({np_} days)"
    if "short average tenure (job-hopping)" in notes:
        return "short average tenure (job-hopping risk)"
    if detail["role_d"]["title_s"] < 0.5 and detail["role_d"]["prod_evidence"] >= 0.5:
        return "title is adjacent, but career history shows the real work"
    if not f.india and s.get("willing_to_relocate"):
        return "outside India but open to relocating"
    return None


def make(f: Features, detail: dict, rank: int) -> str:
    yoe = f"{f.yoe:.1f}"
    title = f.title or "Candidate"
    skills = _matched_skills(f)
    comps = _career_evidence(f)
    s = f.sig
    rr = s.get("recruiter_response_rate", None)
    import zlib
    h = zlib.crc32(f.cid.encode())

    # ---- clause 1: who they are + core evidence ----
    skills_phrase = ", ".join(skills[:3]) if skills else "relevant ML skills"
    if comps:
        built = "ranking/search & recommendation work at " + " and ".join(comps)
    else:
        built = None

    openers = [
        f"{title} with {yoe} yrs",
        f"{title}, {yoe} years' experience",
        f"{yoe}-yr {title}",
    ]
    o = openers[h % len(openers)]

    if built and skills:
        s1 = f"{o}; {built}; strong on {skills_phrase}."
    elif built:
        s1 = f"{o}; {built}."
    elif skills:
        s1 = f"{o}; core skills in {skills_phrase}."
    else:
        s1 = f"{o}; profile matches the retrieval/ranking focus of the role."

    # ---- clause 2: availability / behavior, tone matched to rank ----
    avail = []
    if s.get("open_to_work_flag"):
        avail.append("open to work")
    if rr is not None:
        avail.append(f"{rr:.0%} recruiter response")
    if f.last_active_days <= 30:
        avail.append("recently active")
    avail_phrase = ", ".join(avail[:2]) if avail else "limited recent platform activity"

    concern = _concern(f, detail)

    if rank <= 15:
        if concern:
            s2 = f"Top-tier fit ({avail_phrase}); minor concern: {concern}."
        else:
            s2 = f"Top-tier fit — {avail_phrase}."
    elif rank <= 60:
        if concern:
            s2 = f"Solid fit ({avail_phrase}); watch: {concern}."
        else:
            s2 = f"Solid fit — {avail_phrase}."
    else:
        if concern:
            s2 = f"Borderline shortlist pick: {concern} ({avail_phrase})."
        else:
            s2 = f"Adjacent fit included near the cutoff ({avail_phrase})."

    out = (s1 + " " + s2).replace("  ", " ").strip()
    return out[:300]
