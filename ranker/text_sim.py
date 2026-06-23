"""
TF-IDF cosine similarity between the JD requirements and candidate profiles.

Pure standard library (no numpy / scikit-learn) so the ranker has zero third-party
dependencies and reproduces exactly inside the offline, CPU-only evaluation sandbox.
Over a ~6k shortlist this runs in well under a second.

This is the semantic-ish signal that lets plain-language candidates surface; it is
deliberately a minority of the final score so keyword stuffing can't game it.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from . import jd_spec as J

_TOK = re.compile(r"[a-z0-9][a-z0-9+#.]*")
_STOP = set("the a an and or of to in for with on at by from is are be as we you i "
            "our your they it this that these those will would can could role "
            "have has had not but if then so do does".split())


def _tokens(text):
    toks = [t for t in _TOK.findall(text.lower()) if len(t) > 1 and t not in _STOP]
    bigrams = [toks[i] + "_" + toks[i + 1] for i in range(len(toks) - 1)]
    return toks + bigrams


def similarities(blobs):
    query_tokens = _tokens(J.build_jd_query())
    docs = [Counter(_tokens(b)) for b in blobs]

    N = len(docs) + 1
    df = Counter()
    for d in docs:
        df.update(d.keys())
    df.update(set(query_tokens))
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}

    def tfidf_vec(counter):
        v = {}
        for t, c in counter.items():
            if t in idf:
                v[t] = (1.0 + math.log(c)) * idf[t]
        return v

    qv = tfidf_vec(Counter(query_tokens))
    qnorm = math.sqrt(sum(w * w for w in qv.values())) or 1.0

    sims = []
    for d in docs:
        dv = tfidf_vec(d)
        dnorm = math.sqrt(sum(w * w for w in dv.values())) or 1.0
        if len(qv) < len(dv):
            dot = sum(w * dv.get(t, 0.0) for t, w in qv.items())
        else:
            dot = sum(w * qv.get(t, 0.0) for t, w in dv.items())
        sims.append(dot / (qnorm * dnorm))

    lo, hi = min(sims), max(sims)
    if hi - lo < 1e-12:
        return [0.0] * len(sims)
    return [(s - lo) / (hi - lo) for s in sims]
