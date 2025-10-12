# تثبيت المكتبات الأساسية والضرورية
!pip install transformers torch accelerate Pillow gradio
!pip install diffusers

import gradio as gr
from transformers import pipeline
from diffusers import DiffusionPipeline # نستخدم البايبلاين الأساسي
import torch

# 1. موديل الأسئلة والأجوبة (النص)
qa_pipeline = pipeline("text-generation", model="gpt2")
def ask_ai(question):
    if not question:
        return "من فضلك أدخل سؤالك."
    result = qa_pipeline(question, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']

# 2. موديل توليد الصور السريع جداً (للتأكد من عدم الفصل)
# نستخدم موديل صغير وخفيف (SD-Tiny)
image_pipeline = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-random-stable-diffusion-pipe")

def generate_image(prompt):
    if not prompt:
        return None
    
    print(f"بدء توليد صورة لـ: {prompt}")
    
    # توليد الصورة بسرعة عالية
    image = image_pipeline(prompt).images[0]
    
    print("تم توليد الصورة بنجاح (بموديل خفيف).")
    return image

# 3. بناء الواجهة الجديدة الشاملة
with gr.Blocks() as demo:
    gr.Markdown("# تطبيق الذكاء الاصطناعي الخاص بك (النسخة الشاملة والأكثر استقرارًا)")
    
    with gr.Tab("الأسئلة والأجوبة"):
        gr.Interface(
            fn=ask_ai,
            inputs=gr.Textbox(label="أداة أسئلة وأجوبة"),
            outputs="text",
            title="أداة الأسئلة والأجوبة"
        )
        
    with gr.Tab("توليد الصور"):
        gr.Interface(
            fn=generate_image,
            inputs=gr.Textbox(label="صف الصورة التي تريدها (باللغة الإنجليزية)"),
            outputs="image",
            title="توليد الصور (موديل خفيف وسريع)"
        )

# 4. تشغيل التطبيق
demo.launch(share=True)
