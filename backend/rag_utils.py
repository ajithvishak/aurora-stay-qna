
from langchain_openai import ChatOpenAI
import os

# Hybrid retrieval: combine keyword search (BM25) with the FAISS vector search
#
# CHANGE: the template used vector search only (vectorstore.similarity_search). The small
# 128-token embedding model often missed exact terms such as room types ("Double Queen") and
# property codes ("ASH-SEA Pike"), and returned generic chunks instead. BM25 matches those exact
# words, while vector search still finds paraphrased questions. The two rankings are merged with
# Reciprocal Rank Fusion (RRF), which needs no score calibration between the two methods.
# Same hybrid retrieve() as in the notebook (keep the two in sync), so the deployed API retrieves
# exactly the chunks that were evaluated.

import re
from rank_bm25 import BM25Okapi

CANDIDATES = 20   # Chunks taken from each method before the rankings are merged
RRF_K = 60        # Standard RRF constant: limits how much a single top-ranked hit can dominate

_bm25_cache = {}  # One BM25 index per vector store, built on first use


def _tokenize(text: str) -> list[str]:
    # Lower-case words, numbers and dollar amounts ("$100" stays one token)
    return re.findall(r"[a-z0-9$]+", text.lower())


def _bm25_for(vectorstore):
    # Build the keyword index from the same chunks that are stored in the FAISS index,
    # so both methods always search the same collection
    key = id(vectorstore)
    if key not in _bm25_cache:
        docs = [vectorstore.docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]
        _bm25_cache[key] = (BM25Okapi([_tokenize(d.page_content) for d in docs]), docs)
    return _bm25_cache[key]


# CHANGE: re-ranking stage added after the hybrid search. A cross-encoder reads the question and
# each candidate chunk together and scores how well the chunk answers the question. This ranks
# relevant chunks above chunks that are only on a related topic, which the hybrid search cannot
# separate (it scores the question and the chunks independently).
from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-base"
_reranker = None   # loaded on first use


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    return _reranker


def _content(formatted_chunk: str) -> str:
    # Extract the chunk text from the "Content: ... Metadata: ..." format
    return formatted_chunk.split("Content:\n", 1)[1].split("\n\nMetadata:", 1)[0]


# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    # 1. Vector (semantic) ranking
    dense = vectorstore.similarity_search(query, k=CANDIDATES)

    # 2. Keyword (BM25) ranking
    bm25, docs = _bm25_for(vectorstore)
    scores = bm25.get_scores(_tokenize(query))
    top_ids = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    sparse = [docs[i] for i in top_ids]

    # 3. Reciprocal Rank Fusion: each method adds 1 / (RRF_K + rank) to a chunk's score,
    #    so chunks ranked highly by both methods come first
    fused = {}
    for ranking in (dense, sparse):
        for rank, doc in enumerate(ranking, start=1):
            score, _ = fused.get(doc.page_content, (0.0, doc))
            fused[doc.page_content] = (score + 1 / (RRF_K + rank), doc)
    candidates = [doc for _, doc in sorted(fused.values(), key=lambda x: x[0], reverse=True)[:CANDIDATES]]

    # 4. Re-rank the fused candidates with the cross-encoder and keep the best k
    rerank_scores = _get_reranker().predict([(query, d.page_content) for d in candidates])
    results = [d for _, d in sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)[:k]]

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]


# LLM relevance filter: keep only the retrieved chunks that a complete answer would use
#
# CHANGE: added to raise Contextual Precision. After re-ranking, a short call to the LLM receives
# the question and the candidate chunks and returns the numbers of the chunks that contain a fact,
# rule, condition, fee or exception needed for the answer. Chunks that are only on a related topic
# are dropped, so the answer (and the evaluation) only sees the evidence that is actually used.
import json

RERANK_POOL = 8   # re-ranked candidates shown to the filter
MIN_CHUNKS = 2    # never return fewer chunks than this (the best-ranked ones are added back)

FILTER_PROMPT = """You select the policy passages needed to answer a hotel staff question.

Question:
{query}

Passages:
{passages}

Return the numbers of every passage that contains a fact, rule, condition, fee or exception that a complete and correct answer to this question would use.
Leave out passages that are only on a related topic but would not be used in the answer.
Reply with a JSON list of numbers only, for example [1, 3]. Reply [] if no passage is needed."""


def filter_chunks(query: str, ranked_chunks: list[str], filter_model, max_chunks: int) -> list[str]:
    passages = "\n\n".join(f"[{i}] {_content(c)}" for i, c in enumerate(ranked_chunks, start=1))
    reply = filter_model.invoke(FILTER_PROMPT.format(query=query, passages=passages)).content

    # Parse the JSON list of chunk numbers; ignore anything that is not a valid number
    match = re.search(r"\[[\d,\s]*\]", reply)
    picked = [n for n in (json.loads(match.group(0)) if match else []) if 1 <= n <= len(ranked_chunks)]
    keep = [ranked_chunks[n - 1] for n in sorted(set(picked))]

    # Keep at least MIN_CHUNKS chunks so the answer always has some context
    for chunk in ranked_chunks:
        if len(keep) >= MIN_CHUNKS:
            break
        if chunk not in keep:
            keep.append(chunk)

    # Return the kept chunks in re-ranker order, at most max_chunks of them
    keep.sort(key=ranked_chunks.index)
    return keep[:max_chunks]


# Define the generation function that prompts the LLM with retrieved context
#
# FIX: the template prompt described the assistant as answering questions about
# "medical insurance policies" (left over from another project) and used insurance rules
# (benefits, waiting periods, coverage, exclusions). It now describes hotel policies and uses
# exactly the same rules as generate() in the notebook. The deployed backend must behave like
# the pipeline evaluated with DeepEval; otherwise the evaluation scores would not describe the
# answers hotel staff actually receive from the app.
def generate(query, retrieved_chunks,model) -> str:

    # Same rules as generate() in the notebook (keep the two in sync)
    rules = """
    1. Use only information explicitly stated in the Retrieved Policy Information. Do not use outside knowledge or assumptions.
    2. Read every retrieved chunk before answering. Table rows are written as "<subject>. <column>: <value>; ..." (for example, "ASH-ATL Midtown, Atlanta GA. Parking: Self parking, $28 per night"), and every value in a row belongs to the subject at the start of that row.
    3. A property may be named by its code, its name or its city (for example, "Atlanta" means ASH-ATL Midtown, Atlanta GA). Use the details of the property the user means, and do not mix in details from other properties.
    4. Answer the question directly in the first sentence, then add the supporting details.
    5. Include every condition that changes the answer: deadlines, fees and amounts, taxes, limits, exceptions, and any approval that is required (for example, manager approval).
    6. When a policy is described as a baseline, "typical", "subject to availability", or "subject to approval", say so. Do not promise refunds, waivers, changes, or exceptions that the policy does not guarantee.
    7. Do not invent fees, times, property names, amenities, or procedures.
    8. Include only information relevant to the query; leave out unrelated retrieved content.
    9. Keep the answer concise and clear for hotel staff; use short bullet points when listing several conditions.
    10. If the retrieved information only partly answers the query, answer the supported part and state clearly which part is not covered.
    11. Only if none of the retrieved information addresses the query, say: "The available policy documents do not contain enough information to answer this question." and suggest checking the live property profile or escalating to the manager on duty.
    12. If the retrieved information conflicts, point out the conflict instead of choosing one version.
    """

    # FIX: was "medical insurance policies"
    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    # Same two-stage selection as in the notebook: re-ranked candidates, then the LLM relevance filter
    candidates = retrieve(query=query, k=max(k, RERANK_POOL), vectorstore=vectorstore)
    filter_model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=50,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )
    retrieved_chunks = filter_chunks(query, candidates, filter_model, max_chunks=k)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
