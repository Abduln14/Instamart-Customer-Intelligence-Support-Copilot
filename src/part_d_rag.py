"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering and evaluation (mock LLM only)."""
import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
import mock_llm

CHUNK_WORDS = 40
RAG_SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
              "If the answer is not in the context, say you could not find the information.")


def chunk_words(words, size=CHUNK_WORDS, overlap=10):
    """Chunk 0 covers words 0..size-1, the next starts size-overlap later; stop after the chunk that reaches the end."""
    step, out, start = size - overlap, [], 0
    while True:
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return out


def load_chunks(policy_dir="data/policies", size=CHUNK_WORDS, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(policy_dir, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for i, ch in enumerate(chunk_words(words, size, overlap)):
            chunks.append({"source": os.path.basename(path), "chunk_id": i, "text": ch})
    return chunks


class PolicyIndex:
    def __init__(self, overlap=10, size=CHUNK_WORDS):
        self.chunks = load_chunks(size=size, overlap=overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question):
        hits = self.retrieve(question, 3)
        ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
        resp = mock_llm.chat([{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}])
        return resp.choices[0].message.content, [h["source"] for h in hits]


_DEFAULT = None
def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex(overlap=10)
    return _DEFAULT

def retrieve(query, k=3):
    return default_index().retrieve(query, k)

def answer(question):
    return default_index().answer(question)


def evaluate(overlap, verbose=True):
    idx = PolicyIndex(overlap=overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = n_ans = n_unans = 0
    fails = []
    for q in ev.itertuples(index=False):
        text, sources = idx.answer(q.question)
        if q.expected_doc == "NONE":
            n_unans += 1
            fb_ok += text.startswith("I could not find")
            if not text.startswith("I could not find"):
                fails.append((q.question, "should have fallen back", text))
        else:
            n_ans += 1
            d, f = q.expected_doc in sources, q.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f):
                fails.append((q.question, f"doc_in_top3={d} fact_in_answer={f}", text))
    counts = pd.Series([c["source"] for c in idx.chunks]).value_counts().to_dict()
    print(f"overlap={overlap}: chunks={len(idx.chunks)} {counts}")
    print(f"  answerable: expected_doc in top-3 {doc_ok}/{n_ans} | expected_fact in answer {fact_ok}/{n_ans} | "
          f"unanswerable fallback {fb_ok}/{n_unans}")
    for q, why, text in fails:
        print(f"  FAIL: {q} -> {why}\n        answer: {text}")
    return idx, (doc_ok, fact_ok, fb_ok, fails)


if __name__ == "__main__":
    idx, _ = evaluate(10)
    print("\nquery 'refund policy late delivery' top-3:")
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']} score={h['score']:.3f}")
    print("\nexample answer:", answer("Who must approve a refund above INR 1000?"))
    print()
    evaluate(0)
