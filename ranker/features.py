"""
Feature extraction: turn a raw candidate dict into normalized, recruiter-meaningful
signals. Pure functions, no side effects, no network.

The output `Features` object is consumed by scoring.py, honeypot.py and reasoning.py.
"""
from __future__ import annotations
import datetime as _dt
import math
import re
from dataclasses import dataclass, field

from . import jd_spec as J

NOW = _dt.date(2026, 6, 23)  # dataset reference "today"

PROF_BASE = {"beginner": 0.30, "intermediate": 0.55, "advanced": 0.80, "expert": 1.00}


def _date(s):
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _contains_any(text, terms):
    return any(t in text for t in terms)


def _count_terms(text, terms):
    return sum(1 for t in terms if t in text)


def skill_trust(sk: dict) -> float:
    """How much do we believe this claimed skill? Discounts lazy keyword stuffing.

    expert+many endorsements+long duration -> ~1.0
    'expert' with 0 endorsements and 3 months -> heavily discounted
    """
    base = PROF_BASE.get(str(sk.get("proficiency", "")).lower(), 0.4)
    end = sk.get("endorsements", 0) or 0
    dur = sk.get("duration_months", 0) or 0
    end_factor = min(1.4, 1.0 + math.log1p(end) / 5.0)      # 0 -> 1.0, 50 -> ~1.38
    dur_factor = min(1.0, 0.4 + dur / 30.0)                  # short duration discounts
    trust = base * end_factor * dur_factor
    # stuffing signature: high claimed proficiency, zero social proof, short tenure
    if base >= 0.8 and end == 0 and dur < 12:
        trust *= 0.5
    return max(0.0, min(1.4, trust))


@dataclass
class Features:
    cid: str
    yoe: float
    title: str
    title_l: str
    headline_l: str
    summary_l: str
    blob: str                      # full searchable text (lower)
    skills_raw: list
    skill_trust: dict              # name_l -> trust
    assess: dict                   # skill_l -> 0..100
    career: list
    companies_l: list
    industries_l: list
    tenures: list
    n_jobs: int
    avg_tenure: float
    months_since_first_job: float
    location_l: str
    country_l: str
    india: bool
    tier1: bool
    preferred_city: bool
    sig: dict                      # redrob_signals
    last_active_days: float
    nlp_ir_hits: int
    off_domain_hits: int
    bucket_scores: dict = field(default_factory=dict)


def extract(c: dict) -> Features:
    p = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []

    title = p.get("current_title", "") or ""
    headline = p.get("headline", "") or ""
    summary = p.get("summary", "") or ""

    career_text = " ".join(
        f"{h.get('title','')} {h.get('company','')} {h.get('industry','')} {h.get('description','')}"
        for h in career
    )
    skills_text = " ".join(s.get("name", "") for s in skills)
    blob = J.norm(" ".join([title, headline, summary, career_text, skills_text]))

    skill_trust_map = {}
    for s in skills:
        nm = J.norm(s.get("name", ""))
        if nm:
            skill_trust_map[nm] = max(skill_trust_map.get(nm, 0.0), skill_trust(s))
    assess = {J.norm(k): float(v) for k, v in (sig.get("skill_assessment_scores", {}) or {}).items()}

    companies_l = [J.norm(h.get("company", "")) for h in career]
    industries_l = [J.norm(h.get("industry", "")) for h in career]
    tenures = [int(h.get("duration_months", 0) or 0) for h in career]
    n_jobs = len(career)
    avg_tenure = (sum(tenures) / n_jobs) if n_jobs else 0.0

    starts = [d for d in (_date(h.get("start_date")) for h in career) if d]
    first = min(starts) if starts else None
    months_since_first = ((NOW - first).days / 30.44) if first else 0.0

    loc = J.norm(p.get("location", ""))
    country = J.norm(p.get("country", ""))
    india = ("india" in country) or any(ci in loc for ci in J.TIER1_CITIES)
    tier1 = any(ci in loc for ci in J.TIER1_CITIES) and india
    preferred = any(ci in loc for ci in J.PREFERRED_CITIES)

    la = _date(sig.get("last_active_date"))
    last_active_days = ((NOW - la).days) if la else 9999.0

    f = Features(
        cid=c.get("candidate_id", ""),
        yoe=float(p.get("years_of_experience", 0) or 0),
        title=title, title_l=J.norm(title), headline_l=J.norm(headline),
        summary_l=J.norm(summary), blob=blob,
        skills_raw=skills, skill_trust=skill_trust_map, assess=assess,
        career=career, companies_l=companies_l, industries_l=industries_l,
        tenures=tenures, n_jobs=n_jobs, avg_tenure=avg_tenure,
        months_since_first_job=months_since_first,
        location_l=loc, country_l=country, india=india, tier1=tier1, preferred_city=preferred,
        sig=sig, last_active_days=last_active_days,
        nlp_ir_hits=_count_terms(blob, J.NLP_IR_TERMS),
        off_domain_hits=_count_terms(blob, J.OFF_DOMAIN_TERMS),
    )
    f.bucket_scores = _bucket_scores(f)
    return f


def _bucket_scores(f: Features) -> dict:
    """Trust-weighted coverage per capability bucket, corroborated by assessments + text."""
    out = {}
    for bucket, spec in J.SKILL_BUCKETS.items():
        terms = spec["terms"]
        # evidence from listed skills (trust-weighted)
        skill_mass = 0.0
        for nm, tr in f.skill_trust.items():
            if any(t in nm for t in terms):
                skill_mass += tr
        # objective assessment evidence
        assess_mass = 0.0
        for nm, sc in f.assess.items():
            if any(t in nm for t in terms):
                assess_mass = max(assess_mass, sc / 100.0)
        # plain-language evidence from career text (capped, lighter)
        text_hits = _count_terms(f.blob, terms)
        text_mass = min(0.6, 0.2 * text_hits)
        score = 1 - math.exp(-(skill_mass + 0.8 * assess_mass + text_mass))  # saturating
        out[bucket] = max(0.0, min(1.0, score))
    return out


def relevance_prelim(f: Features) -> float:
    """Cheap recall score for the funnel stage (higher = more worth re-ranking)."""
    t = f.title_l
    score = 0.0
    if _contains_any(t, J.STRONG_TITLES):
        score += 3.0
    elif _contains_any(t, J.ADJACENT_TITLES):
        score += 1.2
    if _contains_any(t, J.WRONG_TITLES):
        score -= 1.5
    # any genuine ML/IR signal in skills/career
    score += min(3.0, 0.6 * sum(f.bucket_scores[b] for b in
                                ["embeddings_retrieval", "vector_hybrid_search",
                                 "ranking_reco", "nlp"]))
    score += min(1.5, 0.02 * f.nlp_ir_hits)
    # career history with ML/AI roles
    for h in f.career:
        ht = J.norm(h.get("title", ""))
        if _contains_any(ht, J.STRONG_TITLES):
            score += 0.6
    return score
