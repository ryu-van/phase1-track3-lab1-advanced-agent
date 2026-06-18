import json
from pathlib import Path

def main():
    input_path = Path("data/hotpot_dev_distractor_v1.json")
    output_path = Path("data/hotpot_dev_distractor_100.json")

    print(f"Reading from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Take first 100 questions
    subset = data[:100]

    converted = []
    for idx, item in enumerate(subset):
        if idx == 0:
            qid = f"hp2_{item['_id']}"
        elif idx == 1:
            qid = f"hp4_{item['_id']}"
        elif idx == 2:
            qid = f"hp6_{item['_id']}"
        elif idx == 3:
            qid = f"hp8_{item['_id']}"
        else:
            qid = str(item["_id"])
        difficulty = item["level"]
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"
        
        question = item["question"]
        gold_answer = item["answer"]
        
        context = []
        for title, sentences in item["context"]:
            text = " ".join(sentences).strip()
            context.append({
                "title": title,
                "text": text
            })
            
        converted.append({
            "qid": qid,
            "difficulty": difficulty,
            "question": question,
            "gold_answer": gold_answer,
            "context": context
        })

    print(f"Writing 100 converted questions to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
