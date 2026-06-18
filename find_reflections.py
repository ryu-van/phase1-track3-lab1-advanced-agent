import json
from pathlib import Path

def main():
    path = Path("outputs/hotpot_dev_distractor_100/reflexion_runs.jsonl")
    if not path.exists():
        print(f"Error: {path} not found.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    reflexion_runs = [r for r in records if r.get("attempts", 1) > 1]
    print(f"Total reflexion runs with attempts > 1: {len(reflexion_runs)}")
    
    for i, r in enumerate(reflexion_runs):
        print(f"--- Example {i+1} ---")
        print(f"QID: {r.get('qid')}")
        print(f"Question: {r.get('question')}")
        print(f"Gold Answer: {r.get('gold_answer')}")
        print(f"Predicted Answer: {r.get('predicted_answer')}")
        print(f"Is Correct: {r.get('is_correct')}")
        print(f"Attempts: {r.get('attempts')}")
        print("Traces:")
        for trace in r.get("traces", []):
            print(f"  Attempt {trace.get('attempt_id')}:")
            print(f"    Answer: {trace.get('answer')}")
            print(f"    Score: {trace.get('score')}")
            print(f"    Reason: {trace.get('reason')}")
            if trace.get("reflection"):
                ref = trace["reflection"]
                print(f"    Reflection Failure Reason: {ref.get('failure_reason')}")
                print(f"    Reflection Lesson: {ref.get('lesson')}")
                print(f"    Reflection Strategy: {ref.get('next_strategy')}")
        print()

if __name__ == "__main__":
    main()
