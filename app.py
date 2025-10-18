# ----------------------------------------------------
# app.py - كود Streamlit النظيف (المحادثة فقط)
# ----------------------------------------------------
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    # استخدام نموذج GPT-2 الخفيف لضمان التشغيل المجاني
    return pipeline("text-generation", model="gpt2")

chatbot_pipeline = load_model()

# واجهة المستخدم والتصميم
st.set_page_config(page_title="منصة الذكاء الاصطناعي الأسطورية", layout="wide")
st.title("🤖 جميني ماستر جراند - محرك المحادثات")

# تهيئة سجل المحادثة
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "أهلاً بك! كيف يمكنني خدمتك اليوم؟"}]

# عرض سجل المحادثة
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# دالة توليد الرد
def generate_response(prompt_input):
    full_prompt = "System: أنت ذكاء اصطناعي محترف وأسطوري.\n"
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            full_prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            full_prompt += f"Assistant: {msg['content']}\n"

    full_prompt += f"User: {prompt_input}\nAssistant:"

    output = chatbot_pipeline(
        full_prompt, 
        max_new_tokens=100, 
        temperature=0.8,
        pad_token_id=chatbot_pipeline.tokenizer.eos_token_id
    )
    response = output[0]['generated_text'].split('Assistant:')[-1].strip()
    return response

# مربع إدخال المستخدم
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("الماستر يفكر..."):
        response = generate_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
