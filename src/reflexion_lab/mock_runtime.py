import os
import time
import json
from openai import OpenAI
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

# Global telemetry store for real API calls
LAST_CALL_METRICS = {"tokens": 0, "latency_ms": 0}

def clear_metrics():
    LAST_CALL_METRICS["tokens"] = 0
    LAST_CALL_METRICS["latency_ms"] = 0

def add_metrics(tokens: int, latency_ms: int):
    LAST_CALL_METRICS["tokens"] += tokens
    LAST_CALL_METRICS["latency_ms"] += latency_ms

def get_llm_client_and_model():
    mode = os.environ.get("REFLEXION_MODE", "mock")
    if mode == "real":
        return OpenAI(), "gpt-4o-mini"
    elif mode == "ollama":
        base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        return OpenAI(base_url=base_url, api_key="ollama"), model
    return None, None

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    base_qid = example.qid.split("_")[0]
    client, model = get_llm_client_and_model()
    if not client:
        if base_qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[base_qid]
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[base_qid]
        return example.gold_answer

    # Real/Ollama LLM Implementation
    clear_metrics()
    context_str = "\n\n".join(f"Title: {c.title}\nText: {c.text}" for c in example.context)
    
    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
    ]
    if reflection_memory:
        memory_str = "\n".join(f"- Lesson/Strategy: {strategy}" for strategy in reflection_memory)
        messages.append({
            "role": "system", 
            "content": f"Here are the lessons and strategies from your previous failed attempts:\n{memory_str}"
        })
    
    messages.append({
        "role": "user", 
        "content": f"Context:\n{context_str}\n\nQuestion: {example.question}\nAnswer:"
    })
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    add_metrics(tokens, latency)
    
    return response.choices[0].message.content.strip()

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    client, model = get_llm_client_and_model()
    if not client:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

    # Real/Ollama LLM Implementation
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"}
    ]
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
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
            reason=f"Failed to parse evaluator response: {e}. Raw content: {content}"
        )

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    base_qid = example.qid.split("_")[0]
    client, model = get_llm_client_and_model()
    if not client:
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if base_qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

    # Real/Ollama LLM Implementation
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
        model=model,
        messages=messages,
        temperature=0.0,
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
