"""
Scoring: combine recruiter-meaningful components into a single fit score.

Philosophy (straight from the JD's hackathon note):
  * The decisive signal against keyword-stuffer traps is ROLE / CAREER fit, not
    the number of AI keywords in the skills list.
  * Plain-language candidates who actually built ranking/search/recommendation
    systems at product companies should rank high even without the buzzwords.
  * Behavioral signals are a multiplier: a perfect-on-paper candidate who is
    inactive and unresponsive is, for hiring, not available.

final = base(content fit) x disqualifiers x location x behavioral x honeypot
"""
from __future__ import annotations
from .features import Features
from . import jd_spec as J

# weights for the content "base" (sum to 1.0)
W = {"role": 0.34, "skills": 0.24, "experience": 0.22, "text": 0.12, "education": 0.08}


def _any(t, terms):
    return any(x in t for x in terms)


def role_career(f: Features) -> tuple[float, dict]:
    """Is this person actually (close to) the role? Title + career evidence."""
    t = f.title_l
    if _any(t, J.STRONG_TITLES):
        title_s = 0.75 if t.startswith("junior") or "junior" in t else 1.0
    elif _any(t, J.ADJACENT_TITLES):
        title_s = 0.45
    elif _any(t, J.WRONG_TITLES):
        title_s = 0.04
    else:
        title_s = 0.18

    # career history: fraction of roles that are real ML/AI roles
    ml_roles = sum(1 for h in f.career if _any(J.norm(h.get("title", "")), J.STRONG_TITLES))
    career_role_s = min(1.0, ml_roles / max(1, min(3, f.n_jobs))) if f.n_jobs else 0.0

    # product-company ML evidence in descriptions (catches plain-language Tier-5s)
    prod_evidence = 0.0
    for h in f.career:
        desc = J.norm(h.get("title", "") + " " + h.get("description", ""))
        ind = J.norm(h.get("industry", ""))
        comp = J.norm(h.get("company", ""))
        is_services = _any(comp, J.SERVICES_COMPANIES) or _any(ind, J.SERVICES_INDUSTRIES)
        if _any(desc, J.SKILL_BUCKETS["ranking_reco"]["terms"] +
                       J.SKILL_BUCKETS["embeddings_retrieval"]["terms"] +
                       ["production", "in production", "at scale", "deployed", "shipped"]):
            prod_evidence = max(prod_evidence, 0.6 if is_services else 1.0)

    score = 0.55 * title_s + 0.27 * career_role_s + 0.18 * prod_evidence
    # career evidence can rescue an adjacent title, but a clearly-wrong title stays capped low
    if title_s <= 0.05 and prod_evidence < 0.5:
        score = min(score, 0.10)
    return min(1.0, score), {"title_s": title_s, "career_role_s": career_role_s,
                              "prod_evidence": prod_evidence, "ml_roles": ml_roles}


def skills_fit(f: Features) -> float:
    return min(1.0, sum(J.SKILL_BUCKETS[b]["weight"] * sc for b, sc in f.bucket_scores.items()))


def experience_fit(f: Features) -> tuple[float, dict]:
    y = f.yoe
    if J.EXP_IDEAL_LOW <= y <= J.EXP_IDEAL_HIGH:
        band = 1.0
    elif J.EXP_BAND_LOW <= y <= J.EXP_BAND_HIGH:
        band = 0.9
    elif 4 <= y <= 10:
        band = 0.7
    elif 3 <= y <= 11:
        band = 0.5
    elif y < 3:
        band = 0.3
    else:
        band = 0.5

    # applied ML at product companies
    prod_ml = 0.0
    n = 0
    for h in f.career:
        ht = J.norm(h.get("title", ""))
        ind = J.norm(h.get("industry", "")); comp = J.norm(h.get("company", ""))
        if _any(ht, J.STRONG_TITLES) or _any(ht, ["data scientist", "applied"]):
            n += 1
            services = _any(comp, J.SERVICES_COMPANIES) or _any(ind, J.SERVICES_INDUSTRIES)
            prod_ml += 0.5 if services else 1.0
    applied = min(1.0, prod_ml / 2.0)

    shipped = 1.0 if _any(f.blob, [
        "recommendation system", "ranking system", "search system", "search relevance",
        "recommender", "retrieval system", "matching system", "ranking pipeline",
        "built and deployed", "shipped", "in production at scale", "to production",
    ]) else 0.0

    # recency: is current / most-recent role an ML/AI role?
    recency = 0.0
    if f.career:
        cur = J.norm(f.career[0].get("title", ""))
        if _any(cur, J.STRONG_TITLES) or _any(cur, ["data scientist", "applied"]):
            recency = 1.0
        elif _any(f.title_l, J.STRONG_TITLES):
            recency = 0.8

    score = 0.40 * band + 0.30 * applied + 0.18 * shipped + 0.12 * recency
    return min(1.0, score), {"band": band, "applied": applied, "shipped": shipped, "recency": recency}


def education_fit(c: dict) -> float:
    best = 0.0
    for e in c.get("education", []) or []:
        tier = str(e.get("tier", "unknown"))
        t = {"tier_1": 0.6, "tier_2": 0.4, "tier_3": 0.2, "tier_4": 0.1}.get(tier, 0.1)
        field = J.norm(e.get("field_of_study", "") + " " + e.get("degree", ""))
        if _any(field, ["computer", "artificial intelligence", "machine learning",
                         "data science", "statistics", "mathematics", "electrical",
                         "information", "software"]):
            t += 0.4
        best = max(best, min(1.0, t))
    return best


def disqualifiers(f: Features) -> tuple[float, list]:
    """Multiplicative penalties for the JD's explicit 'do NOT want' list."""
    mult = 1.0
    notes = []
    comps, inds = f.companies_l, f.industries_l

    services_all = bool(comps) and all(
        _any(c, J.SERVICES_COMPANIES) or _any(i, J.SERVICES_INDUSTRIES)
        for c, i in zip(comps, inds))
    has_product = any(not (_any(c, J.SERVICES_COMPANIES) or _any(i, J.SERVICES_INDUSTRIES))
                      for c, i in zip(comps, inds))
    if services_all:
        mult *= 0.55; notes.append("entire career at services/consulting firms")
    elif (_any(comps[0], J.SERVICES_COMPANIES) or _any(inds[0], J.SERVICES_INDUSTRIES)) if comps else False:
        if not has_product:
            mult *= 0.9

    # off-target domain (CV / speech / robotics / frontend) dominating, weak NLP/IR
    if f.off_domain_hits >= 3 and f.off_domain_hits > f.nlp_ir_hits:
        mult *= 0.6; notes.append("profile tilts to CV/speech/frontend over NLP/IR")

    # pure research / academia without production
    if _any(f.blob, ["phd", "postdoc", "research lab", "publications", "thesis"]) and \
       not _any(f.blob, ["production", "deployed", "shipped", "product"]) and \
       any(_any(i, ["education", "research", "university", "academia"]) for i in inds):
        mult *= 0.6; notes.append("research-only background, no production deployment")

    # title-chasing / job-hopping
    if f.n_jobs >= 4 and f.avg_tenure and f.avg_tenure < 18:
        mult *= 0.82; notes.append("short average tenure (job-hopping)")

    # framework enthusiast: shallow recent LLM-API work, no retrieval/eval depth
    shallow = _any(f.blob, ["langchain", "openai api", "gpt wrapper", "prompt engineering"])
    depth = (f.bucket_scores["embeddings_retrieval"] + f.bucket_scores["vector_hybrid_search"]
             + f.bucket_scores["evaluation"]) > 0.6
    if shallow and not depth and f.yoe < 4:
        mult *= 0.8; notes.append("LLM-framework hobbyist without retrieval/eval depth")

    return mult, notes


def location_factor(f: Features) -> tuple[float, str]:
    if f.india:
        if f.preferred_city:
            return 1.05, "based in Pune/Noida (preferred)"
        if f.tier1:
            return 1.0, "based in a Tier-1 Indian city"
        return 0.92, "based in India"
    # outside India: no visa sponsorship per JD
    if f.sig.get("willing_to_relocate"):
        return 0.72, "outside India but willing to relocate"
    return 0.55, "outside India, not open to relocation"


def behavioral_modifier(f: Features) -> tuple[float, dict]:
    s = f.sig
    adj = 0.0
    if s.get("open_to_work_flag"): adj += 0.08
    else: adj -= 0.05
    d = f.last_active_days
    adj += 0.05 if d <= 30 else (0.0 if d <= 90 else (-0.05 if d <= 180 else -0.12))
    np_ = s.get("notice_period_days", 90) or 0
    adj += 0.04 if np_ <= 30 else (0.0 if np_ <= 60 else (-0.04 if np_ <= 90 else -0.08))
    rr = s.get("recruiter_response_rate", 0.0) or 0.0
    adj += (rr - 0.5) * 0.20           # 1.0 -> +0.10, 0.0 -> -0.10
    art = s.get("avg_response_time_hours", 72) or 72
    adj += 0.03 if art <= 24 else (-0.04 if art > 120 else 0.0)
    icr = s.get("interview_completion_rate", 0.0) or 0.0
    adj += 0.03 if icr >= 0.7 else (-0.04 if icr < 0.3 else 0.0)
    if (s.get("saved_by_recruiters_30d", 0) or 0) > 0: adj += 0.03
    pc = s.get("profile_completeness_score", 0) or 0
    adj += 0.03 if pc >= 80 else (-0.04 if pc < 40 else 0.0)
    if s.get("verified_email") and s.get("verified_phone"): adj += 0.02
    if s.get("linkedin_connected"): adj += 0.01
    gh = s.get("github_activity_score", -1)
    adj += 0.04 if gh >= 30 else (0.01 if gh >= 0 else -0.02)
    oar = s.get("offer_acceptance_rate", -1)
    if isinstance(oar, (int, float)) and oar > 0: adj += 0.02
    mod = max(0.55, min(1.15, 1.0 + adj))
    return mod, {"adj": round(adj, 3)}


def score_candidate(c: dict, f: Features, text_sim: float, susp: float):
    role_s, role_d = role_career(f)
    sk_s = skills_fit(f)
    exp_s, exp_d = experience_fit(f)
    edu_s = education_fit(c)
    base = (W["role"] * role_s + W["skills"] * sk_s + W["experience"] * exp_s
            + W["text"] * text_sim + W["education"] * edu_s)
    disq_m, disq_notes = disqualifiers(f)
    loc_m, loc_note = location_factor(f)
    beh_m, beh_d = behavioral_modifier(f)
    from .honeypot import factor as hp_factor
    hp_m = hp_factor(susp)
    final = base * disq_m * loc_m * beh_m * hp_m
    detail = {
        "role": role_s, "skills": sk_s, "experience": exp_s, "text": text_sim,
        "education": edu_s, "base": base, "disq_m": disq_m, "loc_m": loc_m,
        "beh_m": beh_m, "hp_m": hp_m, "susp": susp,
        "role_d": role_d, "exp_d": exp_d, "disq_notes": disq_notes,
        "loc_note": loc_note, "beh_d": beh_d,
    }
    return final, detail
