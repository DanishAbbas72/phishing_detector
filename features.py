"""
Feature extraction from URLs for phishing detection.
"""
import re
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = [
    "login", "verify", "bank", "secure", "update", "account",
    "signin", "confirm", "billing", "payment", "password",
    "credential", "alert", "suspended", "urgent", "limited",
    "reactivate", "validate", "authenticate", "checkpoint",
    "recover", "unlock", "restore", "claim", "refund"
]

SHORT_URL_SERVICES = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "short.link", "adf.ly", "bc.vc", "mcaf.ee", "x.co"
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".online", ".site", ".website", ".space"
]


def extract_features(url: str) -> dict:
    """Extract a comprehensive feature vector from a URL."""
    url = str(url).strip()
    features = {}

    # ── Basic length features ──────────────────────────────────────
    features["url_length"] = len(url)
    features["url_length_long"] = int(len(url) > 75)

    # ── Character-count features ───────────────────────────────────
    features["num_dots"]       = url.count(".")
    features["num_hyphens"]    = url.count("-")
    features["num_underscores"]= url.count("_")
    features["num_slashes"]    = url.count("/")
    features["num_digits"]     = sum(c.isdigit() for c in url)
    features["num_special"]    = sum(1 for c in url if c in "@#%=&?~+!")
    features["num_at"]         = url.count("@")
    features["has_at"]         = int("@" in url)

    # ── Protocol / scheme ──────────────────────────────────────────
    features["has_https"]      = int(url.lower().startswith("https://"))
    features["has_http"]       = int(url.lower().startswith("http://"))

    # ── Parsed-URL components ──────────────────────────────────────
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        query    = parsed.query or ""
    except Exception:
        hostname = ""
        path     = ""
        query    = ""

    features["hostname_length"] = len(hostname)
    features["path_length"]     = len(path)
    features["query_length"]    = len(query)
    features["num_subdomains"]  = len(hostname.split(".")) - 2 if hostname else 0
    features["num_subdomains"]  = max(0, features["num_subdomains"])

    # ── IP address as hostname ─────────────────────────────────────
    ip_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$"
    )
    features["has_ip_address"] = int(bool(ip_pattern.match(hostname)))

    # ── Suspicious keywords ────────────────────────────────────────
    url_lower = url.lower()
    features["has_suspicious_keywords"] = int(
        any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)
    )
    features["suspicious_keyword_count"] = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower
    )

    # ── Specific suspicious keywords (individual flags) ────────────
    for kw in ["login", "verify", "bank", "secure", "update",
               "account", "billing", "payment", "signin", "confirm"]:
        features[f"kw_{kw}"] = int(kw in url_lower)

    # ── URL shortening services ────────────────────────────────────
    features["is_shortened"] = int(
        any(svc in url_lower for svc in SHORT_URL_SERVICES)
    )

    # ── Suspicious TLD ─────────────────────────────────────────────
    features["has_suspicious_tld"] = int(
        any(url_lower.endswith(tld) or f"{tld}/" in url_lower
            for tld in SUSPICIOUS_TLDS)
    )

    # ── Punycode / homograph indicators ───────────────────────────
    features["has_punycode"] = int("xn--" in url_lower)

    # ── Ratio features ─────────────────────────────────────────────
    features["digit_ratio"]   = features["num_digits"] / max(len(url), 1)
    features["special_ratio"] = features["num_special"] / max(len(url), 1)
    features["hyphen_ratio"]  = features["num_hyphens"] / max(len(hostname), 1)

    # ── Repeated characters ────────────────────────────────────────
    features["has_repeated_chars"] = int(
        bool(re.search(r"(.)\1{3,}", url))
    )

    # ── Double extension ───────────────────────────────────────────
    features["has_double_extension"] = int(
        bool(re.search(r"\.(exe|php|html|js|pdf)\.(php|html|asp|jsp)", url_lower))
    )

    # ── Encoded characters ─────────────────────────────────────────
    features["has_encoded_chars"] = int("%" in url)

    # ── Redirects via query ────────────────────────────────────────
    features["has_redirect"] = int(
        "redirect" in url_lower or "url=" in url_lower or "next=" in url_lower
    )

    return features


def get_feature_names() -> list:
    """Return the ordered list of feature names."""
    sample = extract_features("https://www.example.com/path?q=test")
    return list(sample.keys())


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/search?q=python",
        "http://paypa1-verify-account.ml/login?user=abc",
        "http://192.168.1.1/bank/verify.php",
        "http://secure-bank-login-verify-account.tk/update",
    ]
    for u in test_urls:
        feats = extract_features(u)
        print(f"\nURL: {u}")
        for k, v in feats.items():
            print(f"  {k}: {v}")
