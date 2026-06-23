"""
Honeypot / trap detection.

The dataset embeds ~80 "subtly impossible" profiles plus keyword-stuffers and
behavioral twins. Letting >10% honeypots into the top-100 is an automatic
disqualification, so we score an internal-consistency "suspicion" and hard-drop
the worst offenders out of contention.

We rely on *logical impossibilities* a careful recruiter would catch:
  - using a skill for longer than you've been working
  - claiming more total experience than your timeline allows
  - career/education dates that don't make physical sense
  - "perfect on paper" skill walls with zero social proof
"""
from __future__ import annotations
import datetime as _dt
from .features import Features, _date, NOW


def suspicion(c: dict, f: Features) -> tuple[float, list]:
    s = 0.0
    flags = []

    yoe_months = f.yoe * 12.0

    # 1) Skill used longer than the whole career (+ generous buffer) -> impossible
    over = [sk for sk in f.skills_raw
            if (sk.get("duration_months", 0) or 0) > yoe_months + 18]
    if over:
        s += 0.5
        flags.append(f"claims skill tenure exceeding total experience ({over[0].get('name')})")

    # 2) Skill used longer than time since first job
    if f.months_since_first_job > 0:
        over2 = [sk for sk in f.skills_raw
                 if (sk.get("duration_months", 0) or 0) > f.months_since_first_job + 12]
        if over2 and not over:
            s += 0.3
            flags.append("skill tenure exceeds career timeline")

    # 3) Career dates physically impossible
    for h in f.career:
        sd, ed = _date(h.get("start_date")), _date(h.get("end_date"))
        if sd and ed and ed < sd:
            s += 0.6; flags.append("career end-date precedes start-date"); break
        if sd and sd > NOW:
            s += 0.6; flags.append("career start-date in the future"); break

    # 4) More claimed experience than the timeline supports
    if f.months_since_first_job > 0 and yoe_months > f.months_since_first_job + 42:
        s += 0.3
        flags.append("stated experience exceeds career span")

    # 5) Sum of tenures wildly exceeds career span (impossible overlap)
    if f.months_since_first_job > 0 and sum(f.tenures) > f.months_since_first_job * 1.9 + 12:
        s += 0.25
        flags.append("overlapping roles exceed plausible timeline")

    # 6) last active before signup (logically impossible)
    sg, la = _date(f.sig.get("signup_date")), _date(f.sig.get("last_active_date"))
    if sg and la and la < sg:
        s += 0.2
        flags.append("last-active precedes signup date")

    # 7) Education dates impossible
    for e in c.get("education", []) or []:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            s += 0.4; flags.append("education end-year precedes start-year"); break

    # 8) "Perfect on paper" wall: many expert skills, zero endorsements anywhere
    experts = [sk for sk in f.skills_raw if str(sk.get("proficiency", "")).lower() == "expert"]
    total_end = sum((sk.get("endorsements", 0) or 0) for sk in f.skills_raw)
    if len(experts) >= 10 and total_end == 0:
        s += 0.3
        flags.append("many 'expert' skills with zero endorsements")

    return min(1.0, s), flags


def factor(susp: float) -> float:
    """Multiplier applied to final score based on suspicion."""
    if susp >= 0.5:
        return 0.04          # effectively removed from top-100
    if susp >= 0.3:
        return 0.55
    return 1.0 - 0.4 * susp
