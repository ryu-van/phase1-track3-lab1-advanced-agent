import json
from pathlib import Path

def estimate_costs(report_path: str = "outputs/hotpot_dev_distractor_100/report.json", output_md_path: str = "outputs/hotpot_dev_distractor_100/token_cost_estimation.md"):
    report_file = Path(report_path)
    if not report_file.exists():
        print(f"Error: {report_path} does not exist. Run benchmark first.")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})

    # Define API pricing (e.g., GPT-4o-mini)
    # Input: $0.15 / 1M tokens ($0.00015 / 1k tokens)
    # Output: $0.60 / 1M tokens ($0.00060 / 1k tokens)
    # Blended average (assuming 80% input, 20% output): $0.24 / 1M tokens ($0.00024 / 1k tokens)
    price_per_1k_tokens = 0.00024  # USD

    react_count = react.get("count", 100)
    react_avg_tokens = react.get("avg_token_estimate", 0)
    react_total_tokens = int(react_avg_tokens * react_count)
    react_total_cost = (react_total_tokens / 1000) * price_per_1k_tokens

    reflexion_count = reflexion.get("count", 100)
    reflexion_avg_tokens = reflexion.get("avg_token_estimate", 0)
    reflexion_total_tokens = int(reflexion_avg_tokens * reflexion_count)
    reflexion_total_cost = (reflexion_total_tokens / 1000) * price_per_1k_tokens

    cost_delta = reflexion_total_cost - react_total_cost
    token_delta = reflexion_total_tokens - react_total_tokens

    md_content = f"""# Token Cost Estimation Report

This report estimates the LLM API cost of running the benchmark on **{react_count}** questions from the HotpotQA dev dataset.
Pricing is calculated based on **GPT-4o-mini** rates with a blended price of **${price_per_1k_tokens:.5f}** per 1,000 tokens (assuming an 80% input and 20% output token split).

## Cost Summary Table

| Agent Type | Total Questions | Avg Tokens/Q | Total Tokens | Estimated Cost (USD) | Cost / 1k Questions |
|---|---|---|---|---|---|
| **ReAct** | {react_count} | {react_avg_tokens:,.1f} | {react_total_tokens:,} | ${react_total_cost:.5f} | ${react_total_cost * (1000/react_count):.2f} |
| **Reflexion** | {reflexion_count} | {reflexion_avg_tokens:,.1f} | {reflexion_total_tokens:,} | ${reflexion_total_cost:.5f} | ${reflexion_total_cost * (1000/reflexion_count):.2f} |
| **Difference (Reflexion - ReAct)** | - | {reflexion_avg_tokens - react_avg_tokens:+,.1f} | {token_delta:+,} | ${cost_delta:+.5f} | ${cost_delta * (1000/react_count):+.2f} |

## Cost Analysis & Key Takeaways

1. **Reflexion Overhead**: Running the Reflexion agent increases the total token usage by **{token_delta:,} tokens** (+{(token_delta/react_total_tokens)*100:.1f}%) due to:
   - Multiple attempts for wrong answers.
   - Reflector prompts analyzing failures.
   - Reflection history appended to subsequent prompts.
2. **Cost-Effectiveness**: 
   - ReAct costs **${react_total_cost * (1000/react_count):.2f} per 1,000 queries**.
   - Reflexion costs **${reflexion_total_cost * (1000/reflexion_count):.2f} per 1,000 queries**.
   - The accuracy improvement (Delta EM) must be weighed against this cost increase of **{ (reflexion_total_cost - react_total_cost)/react_total_cost * 100:.1f}%**.
"""

    out_path = Path(output_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"Cost estimation report saved to {out_path}")

if __name__ == "__main__":
    estimate_costs()
