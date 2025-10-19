import streamlit as st 
from transformers import pipeline #

# @st.cache_resource #
# تم إزالة @st.cache_resource مؤقتاً لتجنب المشاكل، يمكنك إضافته لاحقاً
def load_model(): #
    # لتحقيق Master Grand - استخدام نموذج GPT-2 لتوليد النصوص
    return pipeline("text-generation", model="gpt2") 

# يجب أن يتأكد Streamlit من تثبيت المكتبات (transformers, torch) في requirements.txt

chatbot_pipeline = load_model() #

# واجهة المستخدم والتصميم
st.set_page_config(page_title="الأسطورة", layout="wide") #
st.title("🧙‍♂️ نموذج الذكاء الاصطناعي لتوليد النصوص") #

# صندوق إدخال النص
prompt = st.text_area("أدخل النص الأولي (Prompt) هنا:", "رحلة الماستر بدأت بـ")

if st.button("توليد النص"):
    if prompt:
        with st.spinner("جاري توليد النص... قد يستغرق الأمر بعض الوقت..."):
            # توليد النص باستخدام النموذج
            result = chatbot_pipeline(prompt, max_length=150, num_return_sequences=1)
            
            # عرض النتيجة
            st.subheader("النتيجة:")
            st.code(result[0]['generated_text'])
    else:
        st.error("الرجاء إدخال نص أولي لبدء التوليد.")
 

