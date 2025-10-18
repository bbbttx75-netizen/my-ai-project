# ----------------------------------------------------
# app.py - منصة الذكاء الاصطناعي الشاملة (Gradio/Hugging Face)
# ----------------------------------------------------
import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import os

# 1. إعداد نموذج المحادثة (Chatbot)
# استخدام GPT-2 كنموذج خفيف ومجاني للتشغيل على CPU
CHAT_MODEL_NAME = "gpt2"
chatbot_pipeline = pipeline("text-generation", model=CHAT_MODEL_NAME)

def respond_to_user(message, history):
    # بناء الموجه (Prompt) من سجل المحادثة
    full_prompt = "System: أنت ذكاء اصطناعي محترف وأسطوري. أجب بأسلوب واضح ومباشر.\n"
    for human_msg, ai_msg in history:
        full_prompt += f"User: {human_msg}\nAssistant: {ai_msg}\n"
    full_prompt += f"User: {message}\nAssistant:"

    # توليد الرد
    output = chatbot_pipeline(
        full_prompt, 
        max_new_tokens=100, 
        temperature=0.7,
        pad_token_id=chatbot_pipeline.tokenizer.eos_token_id
    )
    
    # تنظيف الرد
    response = output[0]['generated_text'].split('Assistant:')[-1].strip()
    return response

# 2. إعداد نموذج توليد الصور (Image Generation)
# استخدام نموذج Stable Diffusion خفيف جداً لضمان التشغيل
IMAGE_MODEL_NAME = "hf-internal-testing/tiny-random-stable-diffusion"
try:
    # تحميل النموذج ونقله إلى CPU (لتجنب متطلبات GPU)
    pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL_NAME)
    pipe.to("cpu") 
    def generate_image(prompt, negative_prompt="Low quality, blurry"):
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25).images[0]
        return image
except Exception as e:
    print(f"فشل تحميل نموذج الصورة: {e}")
    # دالة وهمية في حالة فشل التحميل
    def generate_image(prompt, negative_prompt=""):
        raise gr.Error("نموذج توليد الصور غير متوفر حاليًا. يتطلب جهاز أقوى.")

# 3. بناء واجهة Gradio الشاملة (Tabbed Interface)
chat_tab = gr.ChatInterface(
    respond_to_user,
    title="المحرك النصي الأسطوري (محادثة)"
)

image_tab = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="وصف الصورة (Prompt) - بالإنجليزية"),
        gr.Textbox(label="الأشياء التي لا تريدها (Negative Prompt)", value="Low quality, blurry")
    ],
    outputs="image",
    title="مولد الصور الأسطوري (خفيف)",
)

# تشغيل الواجهة
gr.TabbedInterface(
    [chat_tab, image_tab], 
    ["محادثة (Chatbot)", "توليد صور (Image Gen)"]
).launch() 
