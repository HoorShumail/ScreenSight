# ScreenSight

Zero-shot GUI click-target grounding with Qwen2.5-VL-3B, benchmarked on ScreenSpot.

Given a screenshot and an instruction like *"click the submit button"*, ScreenSight
predicts the pixel coordinate a human would click — the core perception skill
behind computer-use agents (Claude's computer use, OpenAI's Operator, and most
"agentic AI" products are built on this exact task).

## What's here

- `src/model.py` — wraps Qwen2.5-VL-3B-Instruct. Handles the coordinate-rescaling
  gotcha: the model reasons over an internally-resized copy of your screenshot,
  so predictions get mapped back to your image's real resolution before scoring.
- `src/evaluate.py` — runs the model on the ScreenSpot benchmark and reports
  accuracy overall, by element type (text vs. icon), and by platform (iOS,
  Android, macOS, Windows, Web).
- `app.py` — Streamlit demo: upload a screenshot, type an instruction, see the
  predicted click point drawn on the image.
- `notebooks/screensight_colab.ipynb` — self-contained notebook to run the
  whole eval on a free Colab GPU. Start here.

## Quickstart (Colab — recommended, no local GPU needed)

1. Upload `notebooks/screensight_colab.ipynb` to https://colab.research.google.com
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells top to bottom. The first run downloads the model (~6GB) and
   the ScreenSpot dataset — give it a few minutes.
4. Section 5 runs on `LIMIT = 100` samples by default so you get a number fast.
   Set `LIMIT = None` once everything works to run the full ~1,200-sample set.

## Quickstart (local, with an NVIDIA GPU)

```bash
pip install -r requirements.txt
python -m src.evaluate --limit 100    # quick smoke test
python -m src.evaluate                # full ScreenSpot test set
streamlit run app.py
```

> On non-Linux systems, `qwen-vl-utils` falls back to `torchvision` for image
> processing automatically — no action needed, just slightly slower.

## Reading the numbers

Qwen2.5-VL-3B zero-shot typically lands somewhere in the **50–60% range** on
ScreenSpot, well below larger or fine-tuned models (80–90%+). That gap isn't a
bug — it's the actual point of the project. Frame it as a parameter-efficiency
story ("what does a 3B model get you for free, and what does closing the gap
cost?"), not a leaderboard entry where the small model "loses."

Plain ScreenSpot is also getting fairly saturated at the top end (top models
now score 90%+), so for a more current, harder comparison point, consider also
running against **ScreenSpot-Pro** (`Voxel51/ScreenSpot-Pro` on Hugging Face) —
high-resolution professional software (CAD, IDEs, scientific tools) where even
frontier models stay under 90% and the average is closer to 60%. Swap
`DATASET_ID` in `src/evaluate.py` to try it (note: ScreenSpot-Pro's schema is
similar but images are much larger, so expect slower inference).

## Next steps (what turns this from "core" into "best project")

- [ ] LoRA fine-tune on a small labeled set (a few thousand examples is enough
      per published results) and re-run this same eval to show the before/after
      jump — this is the single highest-leverage addition.
- [ ] Add ScreenSpot-Pro as a second, harder benchmark.
- [ ] Wire up 3–5 live browser tasks with Playwright so the model actually
      clicks in real time — the best demo material of anything here.

## Resume line

"Built a GUI grounding system evaluating vision-language models on the
ScreenSpot benchmark for click-target localization, the core perception task
behind computer-use AI agents."
