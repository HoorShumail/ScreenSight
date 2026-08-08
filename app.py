"""
ScreenSight — unified demo.

Two modes in one app:
  1. Single Screenshot — upload an image, get a predicted click point.
     Optionally compares zero-shot vs the fine-tuned LoRA checkpoint side by side.
  2. Live Agent — give a real URL + task, watch the model drive a live
     browser through a multi-step click sequence.

Run with:
    streamlit run app.py
"""

import streamlit as st
import asyncio
import hashlib
import io
import json
import os
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from playwright.async_api import async_playwright

st.set_page_config(page_title="ScreenSight", layout="wide")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# --- Adjust this to wherever your final checkpoint actually lives ---
CHECKPOINT_PATH = "/content/drive/MyDrive/screensight_lora_epoch3"

GROUNDING_PROMPT = (
    'You are looking at a screenshot that is {width}x{height} pixels.\n'
    'Find the UI element for this instruction: "{instruction}"\n'
    'Respond with ONLY a JSON object giving the pixel coordinate to click, '
    'in this exact format: {{"point": [x, y]}}\n'
    "x is measured from the left edge, y from the top edge. No other text."
)

NEXT_STEP_PROMPT = (
    'You are controlling a web browser to complete this task: "{task}"\n'
    'Steps already completed: {history}\n'
    'Look at the current screenshot. If the task is already complete, respond with exactly: DONE\n'
    "Otherwise, respond with ONLY a short instruction for the single next click needed "
    '(example: "click the login button"). No other text.'
)


@st.cache_resource
def load_model():
    """Loads the base model once. If a fine-tuned checkpoint exists, attaches it as
    a LoRA adapter — this lets the SAME model object serve both zero-shot
    (adapter disabled) and fine-tuned (adapter enabled) predictions."""
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}
    )

    has_finetuned = os.path.exists(CHECKPOINT_PATH)
    if has_finetuned:
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH, is_trainable=False)
    else:
        model = base_model

    model.eval()
    return model, processor, has_finetuned


def parse_point(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            pt = obj.get("point")
            if pt and len(pt) == 2:
                return float(pt[0]), float(pt[1])
    except Exception:
        pass
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


@torch.inference_mode()
def predict_click(model, processor, image, instruction, use_finetuned, has_finetuned, max_new_tokens=128):
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": GROUNDING_PROMPT.format(width=orig_w, height=orig_h, instruction=instruction)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    resized_w, resized_h = image_inputs[0].size
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    def _generate():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if has_finetuned and not use_finetuned:
        with model.disable_adapter():
            output_text = _generate()
    else:
        output_text = _generate()

    point = parse_point(output_text)
    if point is None:
        return None
    x, y = point
    return x * (orig_w / resized_w), y * (orig_h / resized_h)


@torch.inference_mode()
def decide_next_step(model, processor, image, task, history):
    image = image.convert("RGB")
    hist_text = "; ".join(history) if history else "none yet"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": NEXT_STEP_PROMPT.format(task=task, history=hist_text)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def mark_click(image, point, color="red"):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x, y = point
    r = max(6, min(image.size) // 100)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4)
    draw.line([x - r * 2, y, x + r * 2, y], fill=color, width=2)
    draw.line([x, y - r * 2, x, y + r * 2], fill=color, width=2)
    return marked


async def run_agent(model, processor, url, task, max_steps, ui, has_finetuned):
    history = []
    previous_instruction = None
    previous_hash = None

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        for step in range(max_steps):
            step_box = ui.container()
            step_box.markdown(f"**Step {step + 1}**")

            screenshot_bytes = await page.screenshot()
            current_hash = hashlib.md5(screenshot_bytes).hexdigest()
            image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

            next_instruction = decide_next_step(model, processor, image, task, history).strip()
            step_box.write(f"Model says: `{next_instruction}`")

            if next_instruction.upper() == "DONE":
                step_box.success("Task marked complete.")
                break

            if not next_instruction.lower().startswith("click") or next_instruction.lower() == "click":
                step_box.warning(f"Invalid instruction received — stopping: {next_instruction!r}")
                break

            if previous_instruction == next_instruction and previous_hash == current_hash:
                step_box.warning(
                    "Page did not change after the previous click — stopping to avoid a loop. "
                    "(Known limitation: the fine-tuned checkpoint currently over-relies on a few "
                    "common coordinates for ambiguous instructions — see project README.)"
                )
                break

            target = next_instruction.replace("click", "", 1).strip().strip('"').strip("'")
            pred = predict_click(model, processor, image, target, use_finetuned=True, has_finetuned=has_finetuned)

            if pred is None:
                step_box.error(f"Could not locate: {target}")
                break

            marked = mark_click(image, pred)
            step_box.image(marked, caption=f"Predicted click for: '{target}'", use_container_width=True)

            x, y = pred
            await page.mouse.click(x, y)
            await page.wait_for_timeout(2000)

            history.append(next_instruction)
            previous_instruction = next_instruction
            previous_hash = current_hash

        await browser.close()
    return history


# ==================== UI ====================

st.title("🎯 ScreenSight")
st.caption("Zero-shot & fine-tuned GUI grounding — single screenshots, or a live multi-step browser agent.")

model, processor, has_finetuned = load_model()
if not has_finetuned:
    st.sidebar.warning("Fine-tuned checkpoint not found at CHECKPOINT_PATH — running zero-shot only.")

mode = st.sidebar.radio("Mode", ["📷 Single Screenshot", "🌐 Live Agent"])

# ---------- Mode 1: Single Screenshot ----------
if mode == "📷 Single Screenshot":
    st.subheader("Single Screenshot — Click Prediction")

    uploaded = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])
    instruction = st.text_input("Instruction", placeholder='e.g. "click the submit button"')
    compare = st.checkbox(
        "Compare zero-shot vs fine-tuned", value=has_finetuned, disabled=not has_finetuned
    )
    run = st.button("Predict click location", disabled=not (uploaded and instruction))

    if run:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Running inference..."):
            ft_point = predict_click(model, processor, image, instruction, use_finetuned=True, has_finetuned=has_finetuned)
            zs_point = None
            if compare and has_finetuned:
                zs_point = predict_click(model, processor, image, instruction, use_finetuned=False, has_finetuned=has_finetuned)

        if ft_point is None:
            st.error("Couldn't parse a coordinate from the model's output. Try rephrasing the instruction.")
        elif compare and has_finetuned:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Zero-shot (base model)**")
                if zs_point:
                    st.image(mark_click(image, zs_point, color="blue"), use_container_width=True)
                    st.metric("x, y", f"{zs_point[0]:.0f}, {zs_point[1]:.0f}")
                else:
                    st.error("No prediction parsed.")
            with col2:
                st.markdown("**Fine-tuned**")
                st.image(mark_click(image, ft_point, color="red"), use_container_width=True)
                st.metric("x, y", f"{ft_point[0]:.0f}, {ft_point[1]:.0f}")
        else:
            st.image(mark_click(image, ft_point, color="red"), use_container_width=True)
            st.metric("x, y", f"{ft_point[0]:.0f}, {ft_point[1]:.0f}")
    else:
        st.info("Upload a screenshot and enter an instruction to get started.")

# ---------- Mode 2: Live Agent ----------
else:
    st.subheader("Live Agent — Multi-Step Browser Task")
    if not has_finetuned:
        st.warning("Live agent mode works best with the fine-tuned checkpoint attached.")

    url = st.text_input("Website URL", value="https://www.jazzcash.com.pk")
    task = st.text_input("Task", value="Find and click the login option")
    max_steps = st.slider("Max steps", 1, 10, 5)
    run_agent_btn = st.button("Run agent", type="primary")

    if run_agent_btn:
        results_container = st.container()
        with st.spinner("Agent running..."):
            history = asyncio.run(
                run_agent(model, processor, url, task, max_steps, results_container, has_finetuned)
            )
        st.markdown("---")
        st.write("**Completed steps:**", history if history else "(none — see stop reason above)")
    else:
        st.info("Enter a URL and task above, then click **Run agent**.")