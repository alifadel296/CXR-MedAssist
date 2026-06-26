
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

class QwenCaptioner:
    def __init__(self):
        model_id = "AliFadel/Qwen_2.5_VL_7B_Instruct_MIMIC-CXR"
        
        # 4-bit configuration to prevent Colab GPU crash
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def generate(self, pil_image):
        # Enforcing Note 1: Exclude previous studies completely
        prompt = (
            "Analyze this chest X-ray image and provide a concise description of the findings. "
            "Do NOT reference or mention any previous studies, historical comparisons, or prior reports. "
            "Provide only the absolute current visual findings present in the image."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=pil_image,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        
        # Strip prompt tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text