# 📊 ChartAlignBench
Code for "ChartAB: A Benchmark for Chart Grounding &amp; Dense Alignment"

[**📖 Paper**]() | [**📚HuggingFace Dataset**](https://huggingface.co/datasets/umd-zhou-lab/ChartAlignBench)

This repo contains the official evaluation code and dataset for the paper "ChartAlignBench: A Benchmark for Chart Grounding &amp; Dense Alignment"<br>

## Highlights
- 🔥 **9,000+** instances for VLM evaluation on **fine grained chart understanding**.
- 🔥 Comprehensive benchmark covering VLM evaluation of **Grounding**, **Multi-Chart Alignment**, **Robustness**, and **Downstream QA** capabilities.
- 🔥 Evaluation using novel **two stage pipeline** that decomposes task into **intermediate grounding followed by reasoning** resulting in significant accuracy improvement.
- 🔥 Evaluates both **data** and **attribute** understanding across **diverse chart types and complexities**.

## Findings
- 🔎 **Performance degradation on complex charts**: VLMs demonstrate strong data understanding on simple charts (e.g., bar, line, or numbered bar/line), but their performance drops substantially on complex types (e.g., 3D, box, radar, rose, or multi-axis charts) due to intricate layouts and component interactions.
- 🔎 **Weak attribute understanding**: VLMs exhibit poor recognition of text styles (<20% accuracy for size/font), limited color perception (median RGB error >50), and strong spatial biases in legend positioning.
- 🔎 **Two-stage pipeline proves superior**: The ground-then-compare approach consistently outperforms single-stage methods (stitched-chart, multi-image), reducing hallucinations through intermediate grounding steps.
- 🔎 **Poor grounding/alignment degrade downstream QA**: Precise data grounding and alignment correlate positively with QA accuracy, establishing dense chart understanding as essential for reliable reasoning performance.
- 🔎 **Scaling law holds for most alignment tasks**: Larger models consistently outperform smaller ones across data, color, and legend alignment, though text-style alignment deviates due to JSON generation complexity.
