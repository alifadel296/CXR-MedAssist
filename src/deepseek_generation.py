import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class DeepSeekChat:
    def __init__(self):
        model_id = "AliFadel/DeepSeek-R1-Medical-O1-QA"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def clean_output(self, raw_text):
        # Replace special tokens with their intended representations (it happens due to modules conflict)
        clean_text = raw_text.replace('Ġ', ' ').replace('Ċ', '\n')
        
        # Strip out reasoning blocks cleanly
        clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<think>.*', '', clean_text, flags=re.DOTALL) # Catch open-ended reasoning
        
        # Dynamic Truncation: Hard-stop processing if the model hallucinates a new loop
        stop_patterns = [
            r'<\|eot_id\|>', 
            r'<\|im_end\|>', 
            r'<｜begin of sentence｜>',
            r'(?i):\/\/\s*user', 
            r'(?i):\/\/\s*assistant',
            r'(?i)\nuser:', 
            r'(?i)\nassistant:'
        ]
        
        for pattern in stop_patterns:
            # re.split keeps everything BEFORE the match in index [0] and discards the rest
            clean_text = re.split(pattern, clean_text)[0]
        
        # Clean up any remaining raw structural brackets
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        
        return clean_text.strip()

    def answer_question(self, caption, history, user_message):
        messages = [
            {
                "role": "system", 
                "content": f"You are a helpful clinical assistant. Answer questions based only on these Chest X-ray findings:\n{caption}"
            }
        ]
        
        for user_h, assistant_h in history:
            if user_h: messages.append({"role": "user", "content": user_h})
            if assistant_h: messages.append({"role": "assistant", "content": assistant_h})
        
        messages.append({"role": "user", "content": user_message})
        
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(tokenized_chat, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.25,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the newly generated tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        return self.clean_output(decoded)