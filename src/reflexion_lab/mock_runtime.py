import os
import time
import json
from openai import OpenAI
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

# Global telemetry store for real API / GGUF calls
LAST_CALL_METRICS = {"tokens": 0, "latency_ms": 0}

def clear_metrics():
    LAST_CALL_METRICS["tokens"] = 0
    LAST_CALL_METRICS["latency_ms"] = 0

def add_metrics(tokens: int, latency_ms: int):
    LAST_CALL_METRICS["tokens"] += tokens
    LAST_CALL_METRICS["latency_ms"] += latency_ms


# ---------------------------------------------------------------------------
# GGUF backend via llama-cpp-python
# ---------------------------------------------------------------------------

_llama_instance = None

def _get_llama():
    """Lazy-load the GGUF model. Cached after first load."""
    global _llama_instance
    if _llama_instance is not None:
        return _llama_instance

    from llama_cpp import Llama

    model_path = os.environ.get(
        "GGUF_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models",
                     "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    )
    model_path = os.path.normpath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GGUF model not found at: {model_path}\n"
            "Run: python download_model.py  (or set GGUF_MODEL_PATH env var)"
        )

    n_gpu_layers = int(os.environ.get("GGUF_GPU_LAYERS", "0"))  # 0 = CPU-only
    n_ctx = int(os.environ.get("GGUF_CTX", "2048"))
    n_threads = int(os.environ.get("GGUF_THREADS", str(os.cpu_count() or 4)))

    print(f"[gguf] Loading model from {model_path}  (gpu_layers={n_gpu_layers}, ctx={n_ctx}, threads={n_threads})")
    _llama_instance = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )
    print("[gguf] Model loaded.")
    return _llama_instance


def _gguf_chat(messages: list[dict], max_tokens: int = 256, temperature: float = 0.0) -> tuple[str, int, int]:
    """
    Call llama-cpp-python with a list of chat messages.
    Returns (text, total_tokens, latency_ms).
    """
    llm = _get_llama()
    start = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency = int((time.time() - start) * 1000)
    text = response["choices"][0]["message"]["content"].strip()
    usage = response.get("usage", {})
    tokens = usage.get("total_tokens", 0)
    return text, tokens, latency


# ---------------------------------------------------------------------------
# OpenAI / Ollama backend
# ---------------------------------------------------------------------------

def get_llm_client_and_model():
    mode = os.environ.get("REFLEXION_MODE", "mock")
    if mode == "real":
        return OpenAI(), "gpt-4o-mini"
    elif mode == "ollama":
        base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        # Default model changed to qwen2.5:1.5b (Qwen/Qwen2.5-1.5B-Instruct-GGUF)
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
        return OpenAI(base_url=base_url, api_key="ollama"), model
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    mode = os.environ.get("REFLEXION_MODE", "mock")

    # ── GGUF mode ──────────────────────────────────────────────────────────
    if mode == "gguf":
        clear_metrics()
        # Truncate context to avoid overflowing ctx window (keep first 1800 chars per chunk, max 3 chunks)
        chunks = example.context[:5]
        context_str = "\n\n".join(
            f"Title: {c.title}\nText: {c.text[:800]}" for c in chunks
        )
        messages: list[dict] = [{"role": "system", "content": ACTOR_SYSTEM}]
        if reflection_memory:
            memory_str = "\n".join(f"- Lesson/Strategy: {s}" for s in reflection_memory)
            messages.append({
                "role": "system",
                "content": f"Here are the lessons and strategies from your previous failed attempts:\n{memory_str}"
            })
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {example.question}\nAnswer:"
        })
        text, tokens, latency = _gguf_chat(messages, max_tokens=128)
        add_metrics(tokens, latency)
        return text

    # ── OpenAI / Ollama mode ───────────────────────────────────────────────
    base_qid = example.qid.split("_")[0]
    client, model = get_llm_client_and_model()
    if not client:
        # Mock fallback
        if base_qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[base_qid]
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[base_qid]
        return example.gold_answer

    clear_metrics()
    context_str = "\n\n".join(f"Title: {c.title}\nText: {c.text}" for c in example.context)
    messages = [{"role": "system", "content": ACTOR_SYSTEM}]
    if reflection_memory:
        memory_str = "\n".join(f"- Lesson/Strategy: {s}" for s in reflection_memory)
        messages.append({
            "role": "system",
            "content": f"Here are the lessons and strategies from your previous failed attempts:\n{memory_str}"
        })
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_str}\n\nQuestion: {example.question}\nAnswer:"
    })
    start_time = time.time()
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    add_metrics(tokens, latency)
    return response.choices[0].message.content.strip()


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    mode = os.environ.get("REFLEXION_MODE", "mock")

    # ── GGUF mode ──────────────────────────────────────────────────────────
    if mode == "gguf":
        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"}
        ]
        text, tokens, latency = _gguf_chat(messages, max_tokens=256)
        add_metrics(tokens, latency)
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        try:
            data = json.loads(text)
            return JudgeResult(
                score=int(data.get("score", 0)),
                reason=str(data.get("reason", "")),
                missing_evidence=list(data.get("missing_evidence", [])),
                spurious_claims=list(data.get("spurious_claims", []))
            )
        except Exception:
            # Fallback: exact-match
            score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
            return JudgeResult(score=score, reason=f"Parse failed; fallback to exact-match. Raw: {text[:120]}")

    # ── OpenAI / Ollama mode ───────────────────────────────────────────────
    client, model = get_llm_client_and_model()
    if not client:
        # Mock fallback
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"}
    ]
    start_time = time.time()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0,
        response_format={"type": "json_object"}
    )
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    add_metrics(tokens, latency)
    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", "")),
            missing_evidence=list(data.get("missing_evidence", [])),
            spurious_claims=list(data.get("spurious_claims", []))
        )
    except Exception as e:
        return JudgeResult(
            score=1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0,
            reason=f"Failed to parse evaluator response: {e}. Raw: {content}"
        )


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    mode = os.environ.get("REFLEXION_MODE", "mock")

    # ── GGUF mode ──────────────────────────────────────────────────────────
    if mode == "gguf":
        prompt_content = (
            f"Question: {example.question}\n"
            f"Gold Answer: {example.gold_answer}\n"
            f"Wrong Answer Reason: {judge.reason}\n"
            f"Missing Evidence: {judge.missing_evidence}\n"
            f"Spurious Claims: {judge.spurious_claims}"
        )
        messages = [
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": prompt_content}
        ]
        text, tokens, latency = _gguf_chat(messages, max_tokens=256)
        add_metrics(tokens, latency)
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        try:
            data = json.loads(text)
            return ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=str(data.get("failure_reason", judge.reason)),
                lesson=str(data.get("lesson", "")),
                next_strategy=str(data.get("next_strategy", ""))
            )
        except Exception:
            return ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Failed to parse reflection.",
                next_strategy="Re-read the context carefully and focus on all required hops."
            )

    # ── OpenAI / Ollama mode ───────────────────────────────────────────────
    base_qid = example.qid.split("_")[0]
    client, model = get_llm_client_and_model()
    if not client:
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if base_qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

    prompt_content = (
        f"Question: {example.question}\n"
        f"Gold Answer: {example.gold_answer}\n"
        f"Wrong Answer Reason: {judge.reason}\n"
        f"Missing Evidence: {judge.missing_evidence}\n"
        f"Spurious Claims: {judge.spurious_claims}"
    )
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": prompt_content}
    ]
    start_time = time.time()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0,
        response_format={"type": "json_object"}
    )
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    add_metrics(tokens, latency)
    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "")),
            next_strategy=str(data.get("next_strategy", ""))
        )
    except Exception as e:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Failed to parse reflection.",
            next_strategy="Try a direct approach."
        )
