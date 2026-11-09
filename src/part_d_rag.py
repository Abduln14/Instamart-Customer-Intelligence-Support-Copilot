"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering and evaluation (mock LLM only)."""
import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
import mock_llm

CHUNK_WORDS = 40
DEFAULT_OVERLAP = 10
RAG_SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
              "If the answer is not in the context, say you could not find the information.")


def chunk_words(words, size=CHUNK_WORDS, overlap=DEFAULT_OVERLAP):
    """
    Chunk words with a sliding window.
    Chunk 0 covers words 0..size-1, the next starts size-overlap later;
    stops after the chunk that reaches the end of the text.
    """
    step, out, start = max(1, size - overlap), [], 0
    while True:
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return out


def load_chunks(policy_dir="data/policies", size=CHUNK_WORDS, overlap=DEFAULT_OVERLAP):
    chunks = []
    for path in sorted(glob.glob(os.path.join(policy_dir, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for i, ch in enumerate(chunk_words(words, size, overlap)):
            chunks.append({"source": os.path.basename(path), "chunk_id": i, "text": ch})
    return chunks


class PolicyIndex:
    """
    TF-IDF Policy Retrieval Index with word and bigram features,
    cosine similarity ranking, and relevance score thresholding.
    """
    def __init__(self, overlap=DEFAULT_OVERLAP, size=CHUNK_WORDS, ngram_range=(1, 2), min_similarity=0.05):
        self.overlap = overlap
        self.size = size
        self.ngram_range = ngram_range
        self.min_similarity = min_similarity
        self.chunks = load_chunks(size=size, overlap=overlap)
        # Word + bigram TF-IDF vectorizer with sublinear scaling
        self.vec = TfidfVectorizer(ngram_range=self.ngram_range, sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        """Retrieve top-k chunks ranked by cosine similarity."""
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [
            {"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]}
            for i in order
        ]

    def answer(self, question):
        """
        Retrieve relevant policy chunks and generate a grounded answer.
        Filters out low-similarity chunks (< min_similarity) to avoid treating
        irrelevant context as valid evidence.
        """
        hits = self.retrieve(question, 3)
        # Relevance filtering: do not treat zero/low similarity chunks as valid evidence
        relevant_hits = [h for h in hits if h["score"] >= self.min_similarity]
        if not relevant_hits:
            return "I could not find that information in the policy documents.", []

        ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in relevant_hits)
        user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
        resp = mock_llm.chat([
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user}
        ])
        return resp.choices[0].message.content, [h["source"] for h in relevant_hits]


_DEFAULT = None
def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex(overlap=DEFAULT_OVERLAP, ngram_range=(1, 2))
    return _DEFAULT

def retrieve(query, k=3):
    return default_index().retrieve(query, k)

def answer(question):
    return default_index().answer(question)


def evaluate(overlap=DEFAULT_OVERLAP, ngram_range=(1, 2), verbose=True):
    """Evaluate retrieval and answering against policy_eval.csv."""
    idx = PolicyIndex(overlap=overlap, ngram_range=ngram_range)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = n_ans = n_unans = 0
    fails = []
    for q in ev.itertuples(index=False):
        text, sources = idx.answer(q.question)
        if q.expected_doc == "NONE":
            n_unans += 1
            is_fb = text.startswith("I could not find")
            fb_ok += is_fb
            if not is_fb:
                fails.append((q.question, "should have fallen back", text))
        else:
            n_ans += 1
            d = q.expected_doc in sources
            f = q.expected_fact in text
            doc_ok += d
            fact_ok += f
            if not (d and f):
                fails.append((q.question, f"doc_in_top3={d} fact_in_answer={f}", text))

    counts = pd.Series([c["source"] for c in idx.chunks]).value_counts().to_dict()
    if verbose:
        print(f"overlap={overlap} ngram={ngram_range}: chunks={len(idx.chunks)} {counts}")
        print(f"  answerable: expected_doc in top-3 {doc_ok}/{n_ans} | expected_fact in answer {fact_ok}/{n_ans} | "
              f"unanswerable fallback {fb_ok}/{n_unans}")
        for q, why, text in fails:
            print(f"  FAIL: {q} -> {why}\n        answer: {text}")
    return idx, (doc_ok, fact_ok, fb_ok, fails)


if __name__ == "__main__":
    print("=" * 70)
    print(" PART D - RAG RETRIEVAL & ANSWERING EVALUATION")
    print("=" * 70)

    # 1. Main evaluation with default (word + bigram, overlap 10)
    print("\n--- [1] Default Configuration (overlap=10, ngram=(1,2)) ---")
    idx, _ = evaluate(10, ngram_range=(1, 2))

    print("\nQuery 'refund policy late delivery' top-3:")
    for h in idx.retrieve("refund policy late delivery"):
        print(f"  {h['source']} score={h['score']:.3f}")
    ans_sample, src_sample = idx.answer("Who must approve a refund above INR 1000?")
    print(f"\nExample answer: ('{ans_sample}', {src_sample})")

    # 2. Overlap sensitivity analysis: compare 0, 5, 10, 15
    print("\n--- [2] Overlap Sensitivity Grid (5, 10, 15 and 0 baseline) ---")
    for ov in [0, 5, 10, 15]:
        evaluate(ov, ngram_range=(1, 2))

    # 3. N-gram feature comparison: unigram (1,1) vs unigram+bigram (1,2)
    print("\n--- [3] Feature Representation: Unigram vs Unigram + Bigram (overlap=15) ---")
    print("  Evaluating unigram only (1,1):")
    evaluate(15, ngram_range=(1, 1))
    print("  Evaluating unigram + bigram (1,2):")
    evaluate(15, ngram_range=(1, 2))

