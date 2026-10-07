import gradio as gr
import joblib
import re
import os

# ============================================
# تحميل الموديل ومحوّل TF-IDF
# ============================================
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ============================================
# نفس دالة تنظيف النص المستخدمة أثناء التدريب بالضبط
# ============================================
def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================
# دالة التنبؤ - Gradio يستدعيها تلقائيًا عند أي إدخال جديد
# ============================================
def predict_sentiment(review):
    if not review or review.strip() == "":
        return "⚠️ اكتبي مراجعة أول"
    cleaned = clean_arabic_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    if prediction == 1:
        return f"✅ مراجعة إيجابية (بثقة {probabilities[1]*100:.1f}%)"
    else:
        return f"❌ مراجعة سلبية (بثقة {probabilities[0]*100:.1f}%)"

# ============================================
# بناء الواجهة
# ============================================
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="نص المراجعة",
        placeholder="اكتب مراجعتك هنا، مثال: الفندق نظيف والموظفون متعاونون",
        rtl=True,
        lines=4
    ),
    outputs=gr.Textbox(label="نتيجة التحليل", rtl=True),
    title="محلل المشاعر للمراجعات الفندقية",
    description=(
        "أداة لتحليل مراجعات الفنادق باللغة العربية وتصنيفها آليًا إلى إيجابية أو سلبية، "
        "مبنية على نموذج تعلّم آلة مدرّب على أكثر من 100 ألف مراجعة حقيقية."
    ),
    theme=gr.themes.Soft(primary_hue="teal"),
    examples=[
        ["الفندق نظيف جدا والموظفين متعاونين، تجربة رائعة"],
        ["الغرفة وسخة والخدمة سيئة جدا، لن اكرر التجربة"],
    ],
    flagging_mode="never",
    article="تم تطوير هذه الأداة كمشروع تعليمي في مجال علم البيانات ومعالجة اللغة الطبيعية."
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))