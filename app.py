import streamlit as st 
from transformers import pipeline

# الدالة لتحميل النموذج (تم اختيار DistilGPT2 لضمان التشغيل)
def load_model(): 
    return pipeline("text-generation", model="distilgpt2") 

# يجب تحميل النموذج أولاً لتعريف المتغير "chatbot_pipeline"
chatbot_pipeline = load_model() 

# إعداد واجهة المستخدم
st.set_page_config(page_title="الأسطورة", layout="wide") 
st.title("🧙‍♂️ نموذج الذكاء الاصطناعي لتوليد النصوص") 

# تعريف المتغير "prompt"
prompt = st.text_area("أدخل النص الأولي (Prompt) هنا:", "رحلة الماستر بدأت بـ")

if st.button("توليد النص"):
    if prompt:
        with st.spinner("جاري توليد النص... قد يستغرق الأمر بعض الوقت..."):
            # توليد النص (بدون إعدادات متقدمة لتجنب NotImplementedError)
            result = chatbot_pipeline(prompt, max_length=150, num_return_sequences=1) 
            
            # عرض النتيجة
            st.subheader("النتيجة:")
            st.code(result[0]['generated_text'])
    else:
        st.error("الرجاء إدخال نص أولي لبدء التوليد.")

 

