# 🏨 Arabic Hotel Review Sentiment Analyzer

A machine learning tool that classifies Arabic hotel reviews as positive or negative, deployed as an interactive web app.

## 🖥️ Live Demo

Try it here: [Arabic Hotel Sentiment Analyzer](https://arabic-hotel-sentiment-analyzer.onrender.com/
)

> Hosted on a free tier, so the first load after a period of inactivity may take about a minute.

## 🎯 Project Goal

Hotels and booking platforms receive thousands of reviews. This project automates the first step of understanding them: telling whether a review is positive or negative.

## 📊 Dataset

105,698 real hotel reviews in Arabic (Modern Standard Arabic and dialects) from Booking.com, from the HARD dataset (Elnagar et al., 2018). Neutral ratings were already excluded at the source, so the task is binary: ratings of 1-2 stars are negative, 4-5 stars are positive. The classes are perfectly balanced (52,849 each).

## 🔍 Approach

1. **Exploration:** checked class balance and review length. Negative reviews are noticeably longer on average (about 28 words vs 20), which fits the common pattern that dissatisfied guests write more.
2. **Arabic text cleaning:** removed diacritics, normalized alef forms and taa marbouta, and stripped non-Arabic characters.
3. **Feature extraction:** TF-IDF on the 5,000 most informative words.
4. **Model:** Logistic Regression, trained on 80% of the data and tested on the remaining 20%.

## 📈 Results

| Model | Test set | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | Full test set (21,140 reviews) | **93.1%** |

Precision and recall are balanced across both classes (0.92-0.94), so the model is not biased toward one class.

### Comparison with a pre-trained Arabic model

I compared the model with CAMelBERT, a pre-trained Arabic sentiment model, on a random sample of 500 test reviews. CAMelBERT also predicts a third "neutral" class, which does not exist in this dataset, so the 76 neutral predictions were excluded from both models to keep the comparison fair.

| Model | Accuracy (424 reviews) |
|---|---|
| TF-IDF + Logistic Regression | **94.8%** |
| CAMelBERT (pre-trained) | 92.5% |

**Takeaway:** for a narrow, well-defined task with clean and balanced data, a simple model trained on in-domain data can match or beat a larger general-purpose model, at a much lower computational cost. The comparison used a small sample, so the difference should be read as indicative, not definitive.

## ⚠️ Limitations

- Trained on hotel reviews only; accuracy on other text (products, tweets, restaurants) is not guaranteed.
- Binary classification: neutral or mixed reviews are forced into positive or negative.
- Dialect coverage depends on what appears in the dataset.
- Sarcasm and negation are not handled explicitly.

## 🛠️ Tech Stack

Python, pandas, scikit-learn, Hugging Face Transformers (for the comparison), Gradio, deployed on Render.

## 🚀 Run Locally

```bash
git clone https://github.com/asmaaakbari/arabic-hotel-sentiment-analyzer
cd arabic-hotel-sentiment-analyzer
pip install -r requirements.txt
python app.py
```

## 📚 Sources

- Elnagar, A., Khalifa, Y. S., & Einea, A. (2018). *Hotel Arabic-Reviews Dataset Construction for Sentiment Analysis Applications.*
- Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., & Habash, N. (2021). *The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models.* WANLP 2021.

## 👩‍💻 Author

**Asmaa Alakbari**, Data Science Student
[LinkedIn](https://www.linkedin.com/in/asmaa-alakbari-061b34361/)