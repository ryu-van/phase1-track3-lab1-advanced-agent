from pathlib import Path
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.reporting import build_report, save_report

def main():
    out_dir = Path("outputs/hotpot_dev_distractor_100")
    
    # Load react runs
    react_records = []
    with open(out_dir / "react_runs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                react_records.append(RunRecord.model_validate_json(line))
                
    # Load reflexion runs
    reflexion_records = []
    with open(out_dir / "reflexion_runs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reflexion_records.append(RunRecord.model_validate_json(line))
                
    all_records = react_records + reflexion_records
    
    # Rebuild report
    report = build_report(all_records, dataset_name="hotpot_dev_distractor_100.json", mode="real")
    json_path, md_path = save_report(report, out_dir)
    print("Report rebuilt successfully!")

if __name__ == "__main__":
    main()
