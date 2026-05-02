# PhishGuard – Phishing URL Detection System

An end-to-end Machine Learning system to classify URLs as **Legitimate (0)** or **Phishing (1)**.

---

## 📁 Project Structure

```
phishing_detector/
├── app.py                  # Flask web application
├── train.py                # Model training & evaluation
├── features.py             # URL feature extraction (39 features)
├── predictor.py            # Prediction module (loads saved model)
├── requirements.txt        # Python dependencies
├── data/
│   ├── generate_dataset.py # Dataset generator
│   └── phishing_dataset.csv
├── models/
│   ├── phishing_model.pkl  # Primary: Random Forest
│   ├── logistic_model.pkl  # Secondary: Logistic Regression
│   ├── scaler.pkl          # StandardScaler
│   └── metadata.json       # Feature names + metrics
└── templates/
    └── index.html          # Web interface
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset (already done)
python data/generate_dataset.py

# 3. Train models (already done)
python train.py

# 4. Run the web app
python app.py
# → Open http://localhost:5000
```

---

## 🔍 Features Extracted (39 total)

| Category | Features |
|----------|----------|
| Length | url_length, hostname_length, path_length, query_length |
| Characters | num_dots, num_hyphens, num_digits, num_slashes, num_special |
| Symbols | has_at, num_at |
| Protocol | has_https, has_http |
| Structure | num_subdomains, has_ip_address, has_punycode |
| Keywords | has_suspicious_keywords, suspicious_keyword_count, kw_login, kw_verify, … |
| Services | is_shortened, has_suspicious_tld |
| Patterns | has_repeated_chars, has_double_extension, has_encoded_chars, has_redirect |
| Ratios | digit_ratio, special_ratio, hyphen_ratio |

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 99.17% | 100% | 98.33% | 99.16% |
| **Random Forest** ⭐ | **100%** | **100%** | **100%** | **100%** |
| Decision Tree | 100% | 100% | 100% | 100% |

---

## 🌐 API

### POST /predict
```json
{ "url": "http://example.com" }
```
Response:
```json
{
  "label": 1,
  "prediction": "Phishing",
  "confidence": 0.98,
  "confidence_pct": 98.0,
  "phishing_prob": 98.0,
  "legit_prob": 2.0,
  "risk_indicators": ["No HTTPS", "Suspicious TLD"],
  "features": { ... }
}
```
