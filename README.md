# ML_Project_FakeNews
# Fake News Classifier

**Класифікація новин на правдиві (is_fake = 0) та фейкові (is_fake = 1) за допомогою класичних ML-моделей та BERT-файнтюнінгу.**

##  Структура репозиторію

```plaintext
fake-news-classifier/
├── data/
│ └── fake_news_full_data.csv # оригінальний датасет
├── models/
│ ├── original/ # моделі на чистому тексті
│ │ ├── lr_baseline_countv.pkl
│ │ ├── lr_tfidf_tuned.pkl
│ │ ├── rf_tfidf.pkl
│ │ ├── xgb_tfidf.pkl
│ │ └── bert_finetuned/ # чекпоінти й токенізатор BERT
│ ├── lr_tfidf_masked.pkl # моделі на masked-тексті
│ ├── rf_tfidf_masked.pkl # моделі на masked-тексті
│ └── xgb_tfidf_masked.pkl # моделі на masked-тексті
├── notebooks/
│ ├── FinalProject_main.ipynb # основний ноутбук
│ └── FinalProject_masked.ipynb # ноутбук із masked-експериментом
├── src/
│ ├── inference.py # скрипт для швидкого inference
│ └── api.py # (опційно) REST API на FastAPI
├── results/
│ └── final_model_comparison.csv # зведена таблиця метрик
├── README.md # цей файл
└── requirements.txt # залежності для pip install
```
