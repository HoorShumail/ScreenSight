# 🎯 ScreenSight
### Zero-Shot & Fine-Tuned GUI Grounding with Vision-Language Models

Point at a screenshot in plain English. ScreenSight finds the exact pixel to click.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-red) ![Transformers](https://img.shields.io/badge/🤗%20Transformers-yellow) ![License](https://img.shields.io/badge/license-research-lightgrey)

---

## 📸 Demo

Two modes, one app:

1. **📷 Single Screenshot** — upload an image, get a predicted click point. If the fine-tuned checkpoint is available, toggle a side-by-side comparison of zero-shot vs. fine-tuned predictions on the same image.
2. **🌐 Live Agent** — give a real URL and a plain-English task, and watch the model drive an actual headless browser through a multi-step click sequence, screenshot by screenshot, with a loop-detector that stops the agent if a click doesn't change the page (rather than clicking the same spot forever).

The app auto-detects whether a fine-tuned checkpoint is present at `CHECKPOINT_PATH` and falls back to zero-shot-only if not — see [Getting Started](#-getting-started) for how to point it at your own checkpoint.

---

## 🧠 What This Is

GUI grounding — turning a natural-language instruction into a precise on-screen coordinate — is the foundational perception skill behind computer-use agents: the emerging class of AI systems (Claude's computer use, OpenAI's Operator, and similar) that operate software directly through the screen instead of through APIs. Before an agent can click, type, or navigate, it has to find the right element. That's the problem this project tackles, in two stages:

1. **Zero-shot benchmarking** — how well does an off-the-shelf 3B-parameter VLM do at GUI grounding with no task-specific training, evaluated rigorously against a public academic benchmark?
2. **Targeted fine-tuning** — starting from that zero-shot baseline, can a small self-collected dataset meaningfully improve grounding on a specific, underrepresented domain (Pakistani fintech UIs) that public benchmarks don't cover?

Both are evaluated honestly, including where each one still falls short.

---

## 📊 Key Results

### Zero-shot: full ScreenSpot benchmark
**80.7% accuracy (1,027 / 1,272 correct)** on the full [ScreenSpot](https://arxiv.org/abs/2401.10935) benchmark — a public GUI grounding benchmark spanning 8 platforms (Windows, macOS, iOS, Android, GitLab, Shop, Forum, Tool) — using Qwen2.5-VL-3B-Instruct, 4-bit quantized, running entirely on a free-tier Colab GPU, **with no fine-tuning at all**.

For context: published zero-shot numbers for Qwen2.5-VL-3B on this task vary widely across papers depending on prompting and evaluation setup, and specialist models fine-tuned specifically for GUI grounding typically land in the 80–90% range. Reaching 80.7% zero-shot is the headline finding of the base project.

### Fine-tuned: Pakistani fintech GUI grounding
Starting from that same zero-shot model, a small LoRA adapter was trained on a self-collected dataset of Pakistani fintech and general-web UI elements (JazzCash, Easypaisa, SadaPay, NayaPay, UBL Digital, HBL, Meezan Bank, plus GitHub/HN/Wikipedia for diversity), scraped and labeled automatically with Playwright.

On a held-out validation split:

| | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| **Overall accuracy** | 22.5% | 47.5% | **+25.0 pts** |
| **Fintech-only accuracy** | 31.6% | 63.2% | **+31.6 pts** |
| Failed-to-parse predictions | 3/40 | 0/40 | — |

*(Note: these fine-tuning numbers are not directly comparable to the 80.7% ScreenSpot number above — they're measured on a much smaller, self-collected, harder-skewed validation set, not ScreenSpot. See [Honest Limitations](#-honest-limitations).)*

---

## 🏗️ How It Works

```
Screenshot + Instruction
   ("click the submit button")
              │
              ▼
   Qwen2.5-VL-3B-Instruct (4-bit, NF4 quantized)
   [base model, or +LoRA adapter for fintech fine-tuning]
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
   (interactive)    (point-in-bbox against ground truth)
```

---

## 📈 Zero-Shot: Full ScreenSpot Results

**Overall: 80.7% (1,027 / 1,272)** — zero-shot, no fine-tuning, evaluated on the complete ScreenSpot test set.

**By platform / source**

| Source | Accuracy | Correct / Total |
|---|---|---|
| Shop | 86.6% | 103 / 119 |
| iOS | 85.1% | 217 / 255 |
| Android | 84.2% | 208 / 247 |
| Forum | 83.1% | 74 / 89 |
| macOS | 80.2% | 138 / 172 |
| Windows | 79.6% | 129 / 162 |
| GitLab | 71.9% | 64 / 89 |
| Tool | 67.6% | 94 / 139 |

**By element type**

| Type | Accuracy | Correct / Total |
|---|---|---|
| Text elements | 89.8% | 626 / 697 |
| Icon elements | 69.7% | 401 / 575 |

**Interpretation:** the text/icon gap (89.8% vs. 69.7%) is a well-known pattern in GUI grounding research — instructions referencing visible text can be matched almost like OCR + search, while icons require genuine visual-semantic reasoning ("the gear icon means settings") with no literal string to anchor on. This is the model's biggest headroom.

---

## 🇵🇰 Fine-Tuning: Pakistani Fintech GUI Grounding

**Novel contribution.** Public GUI grounding benchmarks (ScreenSpot included) are built almost entirely from Western/global consumer software and don't cover regional fintech products — apps and web portals that are visually and structurally distinct (dense Urdu/English bilingual layouts, region-specific iconography, different information density) from what these models see in pretraining or in standard benchmarks. This stage asks: does a lightweight, cheaply-collected fine-tune close some of that gap?

### Dataset
- Collected via Playwright: for each target site, every clickable element (`button`, `a`, `input`, `[role=button]`, `[role=link]`, `select`, `textarea`) with a usable label (visible text, `aria-label`, `placeholder`, `title`, or `alt`) was screenshotted and paired with its center-point click coordinate and bounding box.
- Capped at 80 samples/site to prevent generic high-element-count sites (GitHub, Hacker News) from drowning out the fintech samples.
- Sites: JazzCash, Easypaisa, SadaPay, NayaPay, UBL Digital, HBL, Meezan Bank (fintech) + Wikipedia, GitHub, Hacker News (general, for diversity and to avoid overfitting purely to fintech UI patterns).
- 90/10 train/val split, seeded shuffle.

### Method
- **LoRA** (`r=16`, `alpha=32`, targeting all attention + MLP projection layers) on top of the same 4-bit quantized Qwen2.5-VL-3B-Instruct.
- Trained 3 epochs, checkpointed and verified after each epoch (weight-file existence + size check) to Google Drive, with a resumable epoch-tracking state file to survive Colab session drops.
- Prompt format and coordinate convention kept identical to the zero-shot setup — training targets are the **original, unresized** image dimensions and click coordinates, so the fine-tuned model learns the same coordinate space the base model was prompted in.

### Results
See the [Key Results](#-key-results) table above: **+25.0 pts overall, +31.6 pts on fintech sites specifically**, with the fine-tuned model also eliminating all unparseable/failed predictions on the validation set (down from 3/40 to 0/40).

### Known failure modes (found via manual inspection of predictions, not just aggregate accuracy)
- **Ambiguous or repeated instruction text.** When a page has multiple elements with identical or near-identical labels (e.g. several "hide" links on a Hacker News listing at different y-positions), the model has no way to disambiguate from the instruction text alone and tends to collapse to a single "typical" learned location rather than the specific instance meant.
- **Non-Latin-script instructions.** The training set skewed English/fintech-heavy; instructions in scripts like Chinese, Punjabi (Gurmukhi), or Khmer (e.g. Wikipedia's language switcher links) were underrepresented, and the fine-tuned model sometimes falls back to a generic point rather than genuinely reading the label.

Both patterns are explainable given the training data's composition, not signs of a broken pipeline — and both are concrete targets for future data-collection work.

---

## 🔧 Technical Notes (the parts that weren't trivial)

1. **Qwen2.5-VL's "smart resize" coordinate trap.** Qwen2.5-VL doesn't see your image at its original resolution — it internally resizes to a dynamic resolution within a min/max pixel budget before generating coordinates. A model's output of `[x, y]` is in that resized space, not your screenshot's. Naively comparing raw model output against ground-truth bounding boxes silently produces wrong accuracy numbers. `model.py` recovers the actual resized dimensions from `qwen_vl_utils.process_vision_info` and rescales every prediction back to the original image before scoring. During the fine-tuning evaluation pass, this rescale had to be re-verified separately, since training targets were already in original-image space — applying the resize-correction unconditionally to a fine-tuned model's output (which was never trained to need it) would have silently corrupted otherwise-correct predictions.

2. **Defensive output parsing.** The model is prompted to return strict JSON (`{"point": [x, y]}`), but VLMs occasionally wrap that in markdown fences or trailing commentary. Parsing falls back to a regex sweep for the first two numbers in the output rather than failing outright — this kept the eval loop from silently losing samples to formatting noise.

3. **Built for free-tier reality, not a lab GPU.** The full ScreenSpot evaluation (1,272 samples) takes hours end-to-end. Free Colab sessions disconnect. The eval loop checkpoints progress and commits results to git every 20 samples, so a dropped session costs minutes, not hours. The model runs 4-bit quantized (NF4, bfloat16 compute) specifically to fit comfortably on a T4. LoRA training used the same quantization plus gradient checkpointing and 8-step gradient accumulation to fit fine-tuning on the same free-tier hardware.

---

## 🛠️ Tech Stack

- **Model:** Qwen2.5-VL-3B-Instruct (Qwen team)
- **Inference:** 🤗 Transformers, `qwen-vl-utils`, `bitsandbytes` (4-bit NF4 quantization)
- **Fine-tuning:** `peft` (LoRA)
- **Benchmark:** ScreenSpot (Cheng et al., 2024) via 🤗 Datasets
- **Custom dataset collection:** Playwright (headless Chromium)
- **Demo:** Streamlit (single-screenshot comparison mode + live multi-step browser agent mode)
- **Compute:** Free-tier Google Colab (T4 GPU)

---

## 📂 Project Structure

```
ScreenSight/
│
├── notebooks/
│   └── screensight_colab.ipynb   # Full ScreenSpot benchmark run (checkpointed)
│                                  # + custom dataset collection, LoRA fine-tuning,
│                                  #   and fine-tuned vs. zero-shot evaluation
│
├── src/
│   ├── __init__.py
│   ├── evaluate.py               # Benchmark evaluation logic
│   └── model.py                  # ScreenSightModel — grounding + coordinate rescaling
│
├── app.py                        # Unified Streamlit demo: single-screenshot mode
│                                  # (zero-shot vs. fine-tuned comparison) +
│                                  # live multi-step browser agent mode
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/HoorShumail/ScreenSight.git
cd ScreenSight
pip install -r requirements.txt
playwright install chromium   # needed for Live Agent mode
streamlit run app.py
```

By default, `app.py` looks for a fine-tuned LoRA checkpoint at `CHECKPOINT_PATH` (set in `app.py`, currently a Google Drive path from the Colab training run). If it's not found, the app runs zero-shot only — no crash, just a sidebar warning. To use your own fine-tuned checkpoint, update `CHECKPOINT_PATH` to point at your `adapter_model.safetensors` directory.

Upload a screenshot, type an instruction (e.g. "click the settings icon"), and ScreenSight will mark its predicted click location in red — or switch to Live Agent mode to point it at a real URL and task.

To reproduce the full benchmark, or the fintech fine-tuning + evaluation, open `notebooks/screensight_colab.ipynb` in Colab (T4 GPU runtime) — it downloads ScreenSpot automatically, checkpoints as it runs, and contains the full custom-dataset collection, LoRA training, and before/after evaluation cells.

---

## ⚠️ Honest Limitations

- **Zero-shot benchmark (ScreenSpot):** evaluated on the original and comparatively easier grounding benchmark. Harder benchmarks like ScreenSpot-Pro (high-resolution professional software with small, dense targets) would very likely show a lower score — that gap is expected and is the honest next test. Icon grounding (69.7%) is meaningfully weaker than text grounding (89.8%) and is the clearest concrete weakness of the current zero-shot setup.
- **Fine-tuning validation set is small and self-collected** (40 held-out samples), not a standardized public benchmark — the +25.0/+31.6 point improvements are real and internally consistent, but shouldn't be read as directly comparable to the 80.7% ScreenSpot number, which is a much larger and independently-curated test set.
- **The fine-tuned checkpoint path is currently hardcoded to a Google Drive location** from the Colab training run — anyone else running the app locally needs to update `CHECKPOINT_PATH` in `app.py` to their own checkpoint, or the app silently falls back to zero-shot-only (with a sidebar warning, not a crash).
- **The Live Agent mode is a working demo, not a robust production agent** — it's a straightforward loop of screenshot → next-instruction → click, with a same-instruction/same-screenshot loop-detector as its only real safety net. It inherits every limitation of the underlying grounding model, including the two below.
- **Non-Latin-script and ambiguous/repeated-label instructions** are known weak points of the fine-tuned model, likely a direct consequence of the training set's English/fintech skew — see [Known failure modes](#known-failure-modes-found-via-manual-inspection-of-predictions-not-just-aggregate-accuracy) above.
- **Single-model evaluation.** No comparison against other open VLMs (Moondream2, SeeClick, UI-TARS) is included yet.

---

## 🙏 Acknowledgments

- Qwen2.5-VL — Qwen Team, Alibaba
- ScreenSpot benchmark — Cheng et al., 2024

## 🧑‍💻 Author

**Hoor Shumail**
AI | Machine Learning | Computer Vision | Vision-Language Models | Agentic AI

- GitHub: [https://github.com/HoorShumail](https://github.com/HoorShumail)
- LinkedIn: [https://www.linkedin.com/in/hoor-shumail-a3a076326/](https://www.linkedin.com/in/hoor-shumail-a3a076326/)

## 📜 License

This project is developed for educational, research, and portfolio purposes.

The project builds upon publicly available open-source models and benchmark datasets. Please refer to the respective licenses of Qwen2.5-VL, ScreenSpot, and other third-party libraries used in this repository.