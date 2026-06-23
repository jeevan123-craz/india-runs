"""
Structured representation of the job description.

The challenge JD is rambling prose on purpose. A great recruiter reads it and
extracts: the role, the *must-have* capabilities, the *nice-to-haves*, the hard
disqualifiers, and the soft preferences (location, notice period, availability).

We encode that here as an explicit spec + keyword ontologies. `build_jd_query`
produces a compact "requirements" text used for TF-IDF similarity so that
plain-language candidates (who describe building a recommender without using the
word "RAG") still surface.

Nothing here calls a network or an LLM — it is a static, auditable spec.
"""
from __future__ import annotations
import re

# ---- Target role vocabulary -------------------------------------------------
# Titles that ARE the role (or one rung away from it).
STRONG_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied ml engineer", "applied scientist", "applied ai",
    "recommendation systems engineer", "recommender", "search engineer",
    "ranking engineer", "relevance engineer", "nlp engineer",
    "research engineer", "deep learning engineer", "ml scientist",
    "machine learning scientist",
]
# Adjacent titles — credible IF career history shows real ML/IR work.
ADJACENT_TITLES = [
    "data scientist", "data science", "applied research",
    "software engineer", "backend engineer", "full stack", "fullstack",
    "data engineer", "analytics engineer", "mlops", "platform engineer",
    "research scientist",
]
# Titles that are NOT the role. Keyword-stuffer bait lives here.
WRONG_TITLES = [
    "hr ", "human resource", "recruiter", "talent acquisition",
    "marketing", "sales", "content writer", "copywriter", "seo",
    "accountant", "finance", "audit", "operations manager", "ops manager",
    "project manager", "program manager", "scrum master",
    "business analyst", "customer support", "customer success",
    "graphic designer", "ui/ux designer", "ux designer", "product designer",
    "mechanical engineer", "civil engineer", "electrical engineer",
    "teacher", "lecturer", "professor", "qa engineer", "quality assurance",
    "test engineer", "support engineer", "administrator", "consultant",
]

# ---- Skill / capability buckets (with weights) ------------------------------
# Each bucket: list of surface forms to match against skills + free text.
SKILL_BUCKETS = {
    "embeddings_retrieval": {
        "weight": 0.22,
        "must": True,
        "terms": ["embedding", "embeddings", "sentence-transformer", "sentence transformer",
                   "dense retrieval", "semantic search", "retrieval", "rag",
                   "information retrieval", "bge", "e5", "bi-encoder", "cross-encoder",
                   "neural search", "vector embedding"],
    },
    "vector_hybrid_search": {
        "weight": 0.18,
        "must": True,
        "terms": ["faiss", "pinecone", "weaviate", "qdrant", "milvus",
                   "elasticsearch", "opensearch", "vector database", "vector db",
                   "vector search", "hybrid search", "bm25", "lucene", "solr",
                   "approximate nearest neighbor", "ann", "hnsw"],
    },
    "ranking_reco": {
        "weight": 0.15,
        "must": False,
        "terms": ["learning to rank", "learning-to-rank", "ltr", "ranking",
                   "recommendation", "recommender", "recommendation system",
                   "lambdamart", "xgboost ranker", "search ranking", "relevance",
                   "candidate ranking", "matching"],
    },
    "nlp": {
        "weight": 0.12,
        "must": False,
        "terms": ["nlp", "natural language processing", "transformer", "transformers",
                   "bert", "llm", "large language model", "language model",
                   "text classification", "named entity", "ner", "question answering"],
    },
    "python": {
        "weight": 0.10,
        "must": True,
        "terms": ["python"],
    },
    "evaluation": {
        "weight": 0.10,
        "must": True,
        "terms": ["ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
                   "precision@k", "recall@k", "a/b test", "ab test", "ab testing",
                   "offline evaluation", "ranking metrics", "evaluation framework",
                   "offline-to-online"],
    },
    "llm_finetune": {
        "weight": 0.07,
        "must": False,
        "terms": ["fine-tuning", "fine tuning", "finetune", "lora", "qlora", "peft",
                   "instruction tuning", "rlhf", "distillation"],
    },
    "ml_infra": {
        "weight": 0.06,
        "must": False,
        "terms": ["pytorch", "tensorflow", "huggingface", "hugging face", "onnx",
                   "triton", "model serving", "bentoml", "vllm", "mlflow", "ray",
                   "kubernetes", "docker", "feature store"],
    },
}
MUST_BUCKETS = [k for k, v in SKILL_BUCKETS.items() if v["must"]]

# ---- Domain relevance: NLP/IR (good) vs CV/Speech/Robotics (off-target) -----
NLP_IR_TERMS = SKILL_BUCKETS["embeddings_retrieval"]["terms"] + \
    SKILL_BUCKETS["vector_hybrid_search"]["terms"] + \
    SKILL_BUCKETS["ranking_reco"]["terms"] + SKILL_BUCKETS["nlp"]["terms"]
OFF_DOMAIN_TERMS = [
    "image classification", "object detection", "segmentation", "opencv",
    "computer vision", "cv ", "image processing", "ocr",
    "speech recognition", "asr", "text-to-speech", "tts", "audio", "wav2vec",
    "robotics", "ros ", "slam", "autonomous", "lidar",
    "frontend", "react", "angular", "tailwind", "photoshop", "figma", "css",
]

# ---- Company classification -------------------------------------------------
SERVICES_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "ltl",
    "mphasis", "igate", "syntel", "hexaware", "dxc", "genpact", "birlasoft",
    "ibm services", "deloitte", "pwc", "kpmg", "ernst", "zensar", "coforge",
    "persistent systems", "nttdata", "ntt data", "atos",
]
SERVICES_INDUSTRIES = ["it services", "consulting", "outsourcing", "bpo", "professional services"]

# ---- Location preferences ---------------------------------------------------
TIER1_CITIES = ["bangalore", "bengaluru", "pune", "noida", "gurgaon", "gurugram",
                "delhi", "new delhi", "ncr", "mumbai", "hyderabad", "chennai", "kolkata"]
PREFERRED_CITIES = ["pune", "noida"]

# ---- Experience band --------------------------------------------------------
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6, 8
EXP_BAND_LOW, EXP_BAND_HIGH = 5, 9

JD_TITLE = "Senior AI Engineer — Founding Team (Redrob AI)"


def build_jd_query() -> str:
    """Compact requirements text for TF-IDF similarity against candidate profiles."""
    parts = [
        "senior ai machine learning engineer ranking retrieval matching recommendation search",
        "production embeddings based retrieval sentence-transformers bge e5 vector database",
        "faiss pinecone weaviate qdrant milvus elasticsearch opensearch hybrid search bm25",
        "learning to rank recommendation system search relevance candidate ranking",
        "nlp transformers llm reranking dense retrieval semantic search information retrieval",
        "evaluation framework ndcg mrr map offline online a/b testing ranking metrics",
        "strong python production deployment shipped end to end ranking search recommendation system at scale",
        "applied ml at product company hybrid retrieval llm fine-tuning lora",
    ]
    return " ".join(parts)


_word_re = re.compile(r"[a-z0-9+#.\-/&]+")

def norm(text: str) -> str:
    return (text or "").lower()
