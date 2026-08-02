<div align="center">

# 🎯 ScreenSight

### Zero-Shot GUI Grounding with Vision-Language Models

*Point at a screenshot in plain English. ScreenSight finds the exact pixel to click.*

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.49+-yellow.svg)
![Model](https://img.shields.io/badge/model-Qwen2.5--VL--3B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

---

## 📸 Demo

<!-- Optional: drop a screenshot or GIF of the Streamlit app here, e.g. ![ScreenSight demo](demo.png), and add the image file to the repo root -->

Upload a screenshot, type an instruction like "click the submit button," and watch the model mark exactly where it would click.

---

## 🧠 What This Is

GUI grounding — turning a natural-language instruction into a precise on-screen coordinate — is the foundational perception skill behind **computer-use agents**: the emerging class of AI systems (Claude's computer use, OpenAI's Operator, and similar) that operate software directly through the screen instead of through APIs. Before an agent can click, type, or navigate, it has to *find* the right element. That's the problem this project tackles in isolation, cleanly and measurably.

**ScreenSight evaluates and demos [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) as a zero-shot GUI grounding model** — no fine-tuning, no task-specific training data. Just a carefully engineered prompt, correct handling of the model's internal image resizing, and a full, honest evaluation against a public academic benchmark.

> ### 📊 Key Result
> **80.7% accuracy** (1,027 / 1,272 correct) on the full **ScreenSpot** benchmark — a public GUI grounding benchmark spanning 8 platforms (Windows, macOS, iOS, Android, GitLab, Shop, Forum, Tool) — using a 3B-parameter model, 4-bit quantized, running entirely on a free-tier Colab GPU.

For context: published zero-shot numbers for Qwen2.5-VL-3B on this task vary widely across papers depending on prompting and evaluation setup, and specialist models *fine-tuned specifically* for GUI grounding typically land in the 80–90% range. Reaching 80.7% with **no fine-tuning at all** is the headline finding of this project.

---

## 🏗️ How It Works

```
Screenshot + Instruction
   ("click the submit button")
              │
              ▼
   Qwen2.5-VL-3B-Instruct (4-bit, NF4 quantized)
              │
              ▼
   Structured JSON output: {"point": [x, y]}
              │
              ▼
   Coordinate rescaling (model's internal
   "smart resize" space → original image space)
              │
              ▼
   Predicted click point, overlaid on the screenshot
              │
        ┌─────┴─────┐
        │           │
   Streamlit demo   Benchmark scoring
   (interactive)    (point-in-bbox against ScreenSpot ground truth)
```

---

## 📈 Full Results

**Overall: 80.7%** (1,027 / 1,272) — zero-shot, no fine-tuning, evaluated on the complete ScreenSpot test set.

### By platform / source

| Source | Accuracy | Correct / Total |
|---|---|---|
| Shop | **86.6%** | 103 / 119 |
| iOS | **85.1%** | 217 / 255 |
| Android | **84.2%** | 208 / 247 |
| Forum | **83.1%** | 74 / 89 |
| macOS | **80.2%** | 138 / 172 |
| Windows | **79.6%** | 129 / 162 |
| GitLab | **71.9%** | 64 / 89 |
| Tool | **67.6%** | 94 / 139 |

### By element type

| Type | Accuracy | Correct / Total |
|---|---|---|
| Text elements | **89.8%** | 626 / 697 |
| Icon elements | **69.7%** | 401 / 575 |

**Interpretation:** the text/icon gap (89.8% vs. 69.7%) is a well-known pattern in GUI grounding research — instructions referencing visible text can be matched almost like OCR + search, while icons require genuine visual-semantic reasoning ("the gear icon means settings") with no literal string to anchor on. This model's biggest headroom is icon-only grounding, which is the natural next benchmark to target.

---

## 🔧 Technical Notes (the parts that weren't trivial)

**1. Qwen2.5-VL's "smart resize" coordinate trap.**
Qwen2.5-VL doesn't see your image at its original resolution — it internally resizes to a dynamic resolution within a min/max pixel budget before generating coordinates. A model's output of `[x, y]` is in *that* resized space, not your screenshot's. Naively comparing raw model output against ground-truth bounding boxes silently produces wrong accuracy numbers. `model.py` recovers the actual resized dimensions from `qwen_vl_utils.process_vision_info` and rescales every prediction back to the original image before scoring.

**2. Defensive output parsing.**
The model is prompted to return strict JSON (`{"point": [x, y]}`), but VLMs occasionally wrap that in markdown fences or trailing commentary. Parsing falls back to a regex sweep for the first two numbers in the output rather than failing outright — this kept the eval loop from silently losing samples to formatting noise.

**3. Built for free-tier reality, not a lab GPU.**
The full ScreenSpot evaluation (1,272 samples) takes hours end-to-end. Free Colab sessions disconnect. The eval loop checkpoints progress and commits results to git every 20 samples, so a dropped session costs minutes, not hours. The model runs 4-bit quantized (NF4, bfloat16 compute) specifically to fit comfortably on a T4.

---

## 🛠️ Tech Stack

- **Model:** Qwen2.5-VL-3B-Instruct ([Qwen team](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct))
- **Inference:** 🤗 Transformers, `qwen-vl-utils`, `bitsandbytes` (4-bit NF4 quantization)
- **Benchmark:** [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) (Cheng et al., 2024) via 🤗 Datasets
- **Demo:** Streamlit
- **Compute:** Free-tier Google Colab (T4 GPU)

---

## 📂 Project Structure

```
ScreenSight/
│
├── notebooks/
│   └── screensight_colab.ipynb   # Full ScreenSpot benchmark run (checkpointed)
│
├── src/
│   ├── __init__.py
│   ├── evaluate.py               # Benchmark evaluation logic
│   └── model.py                  # ScreenSightModel — grounding + coordinate rescaling
│
├── app.py                        # Streamlit demo
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/ScreenSight.git
cd ScreenSight
pip install -r requirements.txt
streamlit run app.py
```

Upload a screenshot, type an instruction (e.g. *"click the settings icon"*), and ScreenSight will mark its predicted click location in red.

To reproduce the benchmark yourself, open `notebooks/screensight_colab.ipynb` in Colab (T4 GPU runtime) — it downloads ScreenSpot automatically and checkpoints as it runs. `src/evaluate.py` holds the same evaluation logic as a standalone module.

---

## ⚠️ Honest Limitations

- Evaluated zero-shot only — no fine-tuning has been applied yet (see Roadmap).
- Evaluated on **ScreenSpot**, the original and comparatively easier grounding benchmark. Harder benchmarks like **ScreenSpot-Pro** (high-resolution professional software with small, dense targets) would very likely show a lower score — that gap is expected and is the honest next test.
- Single-model evaluation. No comparison against other open VLMs (Moondream2, SeeClick, UI-TARS) is included yet — see Roadmap.
- Icon grounding (69.7%) is meaningfully weaker than text grounding (89.8%) and is the clearest concrete weakness of the current setup.

---

## 🗺️ Roadmap

- [ ] LoRA fine-tuning on a self-generated dataset (Playwright-scraped screenshots + DOM element coordinates)
- [ ] Evaluate on ScreenSpot-Pro and OSWorld-G for a harder, more current benchmark comparison
- [ ] Side-by-side comparison against Moondream2 and other small open VLMs
- [ ] Live execution loop — Playwright actually performs the predicted click on a handful of real browser tasks
- [ ] Icon-specific grounding improvements (the current weakest category)

---

## 🙏 Acknowledgments

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — Qwen Team, Alibaba
- [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) benchmark — Cheng et al., 2024

---

# 🧑‍💻 Author

**Hoor Shumail**

AI | Machine Learning | Computer Vision | Vision-Language Models | Agentic AI

- **GitHub:** https://github.com/HoorShumail
- **LinkedIn:** https://www.linkedin.com/in/hoor-shumail-a3a076326/

---

# 📜 License

This project is developed for educational, research, and portfolio purposes.

The project builds upon publicly available open-source models and benchmark datasets. Please refer to the respective licenses of **Qwen2.5-VL**, **ScreenSpot**, and other third-party libraries used in this repository.
