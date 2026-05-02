"""
Prediction module - loads saved models and predicts on new URLs.
"""
import os
import json
import numpy as np
import joblib

from features import extract_features

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class PhishingPredictor:
    def __init__(self):
        self.model   = joblib.load(os.path.join(MODEL_DIR, "phishing_model.pkl"))
        self.scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]

    def predict(self, url: str) -> dict:
        feats   = extract_features(url)
        X       = np.array([[feats.get(fn, 0) for fn in self.feature_names]])
        label   = int(self.model.predict(X)[0])
        proba   = self.model.predict_proba(X)[0]
        conf    = float(proba[label])

        # Risk-score breakdown
        risk_indicators = []
        if feats.get("has_ip_address"):
            risk_indicators.append("IP address used as hostname")
        if feats.get("has_at"):
            risk_indicators.append("'@' symbol in URL")
        if not feats.get("has_https"):
            risk_indicators.append("No HTTPS")
        if feats.get("has_suspicious_tld"):
            risk_indicators.append("Suspicious top-level domain")
        if feats.get("is_shortened"):
            risk_indicators.append("URL shortening service detected")
        if feats.get("has_punycode"):
            risk_indicators.append("Punycode / homograph characters")
        if feats.get("suspicious_keyword_count", 0) > 2:
            risk_indicators.append(f"{feats['suspicious_keyword_count']} suspicious keywords found")
        elif feats.get("has_suspicious_keywords"):
            risk_indicators.append("Suspicious keyword in URL")
        if feats.get("url_length_long"):
            risk_indicators.append(f"Unusually long URL ({feats['url_length']} chars)")
        if feats.get("num_hyphens", 0) > 3:
            risk_indicators.append(f"Many hyphens ({feats['num_hyphens']})")
        if feats.get("num_dots", 0) > 4:
            risk_indicators.append(f"Excessive dots ({feats['num_dots']}) — possible subdomain abuse")
        if feats.get("has_redirect"):
            risk_indicators.append("Redirect parameter detected")

        return {
            "url"        : url,
            "label"      : label,
            "prediction" : "Phishing" if label == 1 else "Legitimate",
            "confidence" : conf,
            "confidence_pct": round(conf * 100, 2),
            "phishing_prob" : round(float(proba[1]) * 100, 2),
            "legit_prob"    : round(float(proba[0]) * 100, 2),
            "risk_indicators": risk_indicators,
            "features"   : feats,
        }


# Singleton
_predictor = None

def get_predictor() -> PhishingPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PhishingPredictor()
    return _predictor


if __name__ == "__main__":
    p = get_predictor()
    test = [
        "https://www.google.com/search?q=python",
        "http://paypa1-verify-account.ml/login",
        "http://192.168.1.1/bank/verify.php",
        "https://github.com/openai/gpt-4",
        "http://secure-bank-login-update.tk/signin",
    ]
    for url in test:
        r = p.predict(url)
        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"→ {r['prediction']} | Confidence: {r['confidence_pct']}%")
        print(f"  Phishing prob: {r['phishing_prob']}% | Legit prob: {r['legit_prob']}%")
        if r["risk_indicators"]:
            print(f"  ⚠ Risks: {', '.join(r['risk_indicators'])}")
