
import os
import sys
import gradio as gr

# Ensure src module pathing works natively
sys.path.append(os.path.abspath('.'))

from src.qwen_generation import QwenCaptioner
from src.deepseek_generation import DeepSeekChat

# Initialising Vision pipeline
qwen_model = QwenCaptioner()
# Initialising Reasoning pipeline
deepseek_model = DeepSeekChat()

def process_image(image):
    if image is None:
        return "Please upload an image."
    return qwen_model.generate(image)

def chat_response(user_message, history, caption):
    # To exit type q, quit, or exit (no matter the character case)
    if user_message.lower().strip() in ['q', 'quit', 'exit']:
        return history + [[user_message, "Session closed. Please refresh the app to start over."]], ""
    # Force the user to extract the findings first (Medical Caption).
    if not caption or "Please upload" in caption:
        return history + [[user_message, "Please extract CXR findings using Step 1 before chatting."]], ""
    
    clean_reply = deepseek_model.answer_question(caption, history, user_message)
    history.append((user_message, clean_reply))
    return history, ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multimodal Chest X-Ray Diagnosis Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="1. Upload Chest X-Ray")
            btn_caption = gr.Button("Extract Findings", variant="primary")
            caption_output = gr.Textbox(label="Extracted Clinical Findings (Context)", lines=6, interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="2. Consult DeepSeek-R1 Engine")
            msg_input = gr.Textbox(placeholder="Ask a diagnostic query... (type 'q' or 'exit' to quit)", label="User Inquiry")
            btn_clear = gr.Button("Clear History")

    btn_caption.click(fn=process_image, inputs=input_image, outputs=caption_output)
    msg_input.submit(fn=chat_response, inputs=[msg_input, chatbot, caption_output], outputs=[chatbot, msg_input])
    btn_clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # share=True creates a public live URL accessible outside your runtime.
    demo.launch(share=False, debug=True)