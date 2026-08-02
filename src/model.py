"""
Wrapper around Qwen2.5-VL-3B-Instruct for GUI click-target grounding.

Given a screenshot and a natural-language instruction describing a UI
element, predicts the (x, y) pixel coordinate to click, in the
ORIGINAL image's coordinate space.

Handles the Qwen2.5-VL "smart resize" gotcha: the model actually sees a
resized copy of your screenshot internally, and its coordinate outputs
are in THAT resized space, not your original image. We recover the
resize ratio from the processed image and rescale predictions back to
the original screenshot before comparing against ground truth.
"""

import json
import re
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

GROUNDING_PROMPT = (
    'You are looking at a screenshot that is {width}x{height} pixels.\n'
    'Find the UI element for this instruction: "{instruction}"\n'
    'Respond with ONLY a JSON object giving the pixel coordinate to click, '
    'in this exact format: {{"point": [x, y]}}\n'
    "x is measured from the left edge, y from the top edge. No other text."
)


class ScreenSightModel:
    def __init__(self, model_id: str = MODEL_ID, device_map: str = "auto"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_click(
        self, image: Image.Image, instruction: str, max_new_tokens: int = 128
    ) -> Optional[Tuple[float, float]]:
        """Return (x, y) in the ORIGINAL image's pixel coordinates, or
        None if the model's output couldn't be parsed into a point."""
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": GROUNDING_PROMPT.format(
                            width=orig_w, height=orig_h, instruction=instruction
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # process_vision_info applies Qwen's internal "smart resize" and
        # returns the ACTUAL resized PIL image(s) that get fed to the
        # model. Reading .size off that tells us the coordinate space
        # the model is reasoning in, which is generally NOT the same
        # as your original screenshot's resolution.
        image_inputs, video_inputs = process_vision_info(messages)
        resized_w, resized_h = image_inputs[0].size

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        point = self._parse_point(output_text)
        if point is None:
            return None

        x, y = point
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h
        return x * scale_x, y * scale_y

    @staticmethod
    def _parse_point(text: str) -> Optional[Tuple[float, float]]:
        """Model output should be {"point": [x, y]}, but VLMs sometimes
        wrap that in extra text or markdown fences -- parse defensively."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                obj = json.loads(match.group(0))
                pt = obj.get("point")
                if pt and len(pt) == 2:
                    return float(pt[0]), float(pt[1])
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass

        # Fallback: grab the first two numbers anywhere in the string.
        nums = re.findall(r"-?\d+\.?\d*", text)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        return None
