import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1) Завантажуємо класичні пайплайни
lr_base    = joblib.load('models/original/lr_baseline_countv.pkl')
lr_tuned   = joblib.load('models/original/lr_tfidf_tuned.pkl')
rf_pipeline= joblib.load('models/original/rf_tfidf.pkl')
xgb_pipeline=joblib.load('models/original/xgb_tfidf.pkl')

# 2) Завантажуємо BERT
tokenizer  = BertTokenizer.from_pretrained('models/original/bert_finetuned')
model      = BertForSequenceClassification.from_pretrained('models/original/bert_finetuned')
model.eval()

# 3) Функція inference
def predict_all(texts):
    # LR baseline
    print("LR base:", lr_base.predict(texts))
    # LR tuned
    print("LR tuned:", lr_tuned.predict(texts))
    # RF
    print("RF:", rf_pipeline.predict(texts))
    # XGB
    print("XGB:", xgb_pipeline.predict(texts))
    # BERT
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    print("BERT:", logits.argmax(dim=-1).tolist())

if __name__ == "__main__":
    sample = ["Some breaking news headline.", "Another suspicious story…"]
    predict_all(sample)