#!/usr/bin/env python3
"""
URL Phishing Detector

Usage:
    python url_predict.py <url>

This script loads a trained scaler and model, extracts basic features from the URL,
scales them, and predicts whether the URL is phishing or legitimate.
"""
import argparse
import pickle
import pandas as pd
from urllib.parse import urlparse
import ipaddress
import requests

# Paths to saved artifacts
SCALER_PATH = 'url_scaler.pkl'
MODEL_PATH = 'url.pkl'


def load_artifacts(scaler_path=SCALER_PATH, model_path=MODEL_PATH):
    """Load the scaler and model from disk."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return scaler, model


def extract_features(url: str, feature_names) -> pd.DataFrame:
    """Extract features matching training names dynamically."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ''

    # Compute raw metrics
    metrics = {
        'UsingIP': int(_is_ip := True) if (lambda h: not (_is_ip := False)) else None,
    }
    try:
        ipaddress.ip_address(hostname)
        metrics['UsingIP'] = 1
    except ValueError:
        metrics['UsingIP'] = 0
    metrics['LongURL'] = len(url)
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co']
    metrics['ShortURL'] = int(any(s in hostname for s in shorteners))
    metrics['Symbol@'] = int('@' in url)
    metrics['Redirecting//'] = int('//' in (parsed.path or ''))
    metrics['PrefixSuffix-'] = hostname.count('-')
    metrics['SubDomains'] = hostname.count('.')
    metrics['HTTPS'] = int(parsed.scheme == 'https')

    # Placeholder defaults for other features
    df = pd.DataFrame([{name: 0 for name in feature_names}])

    # Map computed metrics into df
    for key, val in metrics.items():
        if key in feature_names:
            df.at[0, key] = val
    return df


def predict_url(url: str) -> str:
    """Load artifacts, extract features, scale, and predict."""
    scaler, model = load_artifacts()
    feature_names = list(scaler.feature_names_in_)
    feats = extract_features(url, feature_names)
    print("Extracted Features:")
    print(feats.to_string(index=False))
    # Align columns
    feats = feats[feature_names]
    scaled = scaler.transform(feats)
    pred = model.predict(scaled)
    return 'Phishing' if pred[0] == 1 else 'Legitimate'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Phishing Detector')
    parser.add_argument('url', help='URL to analyze')
    args = parser.parse_args()

    result = predict_url(args.url)
    print(f"\nURL: {args.url}\nPrediction: {result}")
