"""
utils.py – Wrappers that call the trained models from dummy_project.

Pipeline (as per pipeline_design.jpg):

LEFT PATH  (Saving Account, Guarantees, Junior Account, Loan, Pension)
  Customer Profile + Transaction Data
    → Data Processing & Feature Engineering
      → Banking Products Category Prediction Model
        → Loans Products Prediction Model

RIGHT PATH (Direct Debit, Credit Card)
  Customer Profile + Transaction Data
    → Data Processing & Feature Engineering
      → RFM Features Extraction
        → Customer Segmentation Model
          → Credit Card Products Prediction Model
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_BANKING   = os.path.join(
    BASE_DIR,
    "models",
    "best_multilabel_model.pkl",
)
MODEL_CC        = os.path.join(
    BASE_DIR,
    "models",
    "kmeans_model_creditcardsubtypeunsupervised.pkl",
)
MODEL_SEG       = os.path.join(
    BASE_DIR,
    "models",
    "kmeans_customer_segments.pkl",
)
MCC_CODES_PATH  = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "mcc_codes.json",
)
CC_CATS_PATH    = os.path.join(
    BASE_DIR,
    "models",
    "CreditCardSpendingCategories.txt",
)

# Product label order used when the banking model was trained
BANKING_LABELS = [
    "saving_account",
    "guarantees",
    "junior_account",
    "loans",
    "pension",
]

# Credit Card product mapping (cluster → card name)
CC_PRODUCT_MAP = {
    "FUEL":          "Fuel Rewards Card",
    "RETAIL":        "Shopping Rewards Card",
    "TRAVEL":        "Travel Miles Card",
    "ENTERTAINMENT": "Entertainment Plus Card",
    "GROCERIES":     "Everyday Cashback Card",
    "OTHER":         "Standard Cashback Card",
}

CC_PRODUCT_DESC = {
    "Fuel Rewards Card":        "Earn cashback on every fuel purchase and toll payment.",
    "Shopping Rewards Card":    "Get rewards on retail, clothing and electronics.",
    "Travel Miles Card":        "Accumulate miles on flights, hotels and travel agencies.",
    "Entertainment Plus Card":  "Enjoy cashback on dining, streaming and entertainment.",
    "Everyday Cashback Card":   "Straightforward cashback on groceries and daily shopping.",
    "Standard Cashback Card":   "Flexible cashback across all eligible spending categories.",
}

AFFINITY_THRESHOLD = 0.30   # 30 % minimum spending affinity to qualify

# ─────────────────────────────────────────────
# Lazy-loaded globals
# ─────────────────────────────────────────────
_banking_model = None
_seg_model     = None
_cc_model      = None


def _load_banking_model():
    global _banking_model
    if _banking_model is None:
        _banking_model = joblib.load(MODEL_BANKING)
    return _banking_model


def _load_seg_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = joblib.load(MODEL_SEG)
    return _seg_model


def _load_cc_model():
    global _cc_model
    if _cc_model is None:
        _cc_model = joblib.load(MODEL_CC)
    return _cc_model


# ─────────────────────────────────────────────
# Feature Engineering helpers
# ─────────────────────────────────────────────

def _tenure_months(first_join_date: str) -> float:
    """Return months since customer joined."""
    try:
        joined = pd.to_datetime(first_join_date)
        now    = pd.Timestamp.today()
        return max((now.year - joined.year) * 12 + (now.month - joined.month), 1)
    except Exception:
        return 24.0


def _segment_to_code(segment_str: str) -> int:
    mapping = {
        "01 - TOP":          0,
        "02 - PARTICULARES": 1,
        "03 - UNIVERSITARIO": 2,
    }
    return mapping.get(segment_str, 1)


def _employment_to_code(status: int) -> int:
    # Already numeric from sidebar: 1=Employed, 0=Unemployed, 2=Self-Employed
    return int(status)


def _channel_to_code(channel: str) -> int:
    channels = ["KHD", "KHE", "KHF", "KHL", "KFC", "KDB", "KAT", "KAZ"]
    try:
        return channels.index(channel)
    except ValueError:
        return 0


def _country_to_code(country: str) -> int:
    countries = ["ES", "FR", "DE", "UK", "IT"]
    try:
        return countries.index(country)
    except ValueError:
        return 0


def _compute_sps(customer: dict) -> float:
    """Spending Pattern Score – simplified version."""
    avg_tx  = max(customer.get("avg_transactions_per_month", 1.0), 0.001)
    avg_inc = max(customer.get("avg_income_days_per_month", 1.0), 0.001)
    return min(avg_tx / (avg_inc + avg_tx), 1.0)


def _compute_tsi(customer: dict) -> float:
    """Transaction Stability Index – simplified version."""
    std = customer.get("monthly_transaction_std", 0.5)
    avg = max(customer.get("avg_monthly_transaction_count", 1.0), 0.001)
    cv  = std / avg
    return max(1.0 - cv, 0.0)


def _compute_demographic_score(customer: dict) -> float:
    age        = customer.get("age", 35) / 80.0
    income     = min(customer.get("household_gross_income", 30_000) / 200_000, 1.0)
    credit     = (customer.get("credit_score", 650) - 300) / 550.0
    return (age + income + credit) / 3.0


def preprocess_customer_input(customer: dict) -> pd.DataFrame:
    """
    Build the feature vector used by the banking category model.
    The order must match the training data column order (exactly 22 features).
    """
    from pandas.api.types import CategoricalDtype

    # Exact categorical definitions from training set
    residence_country_categories = ['AD', 'AE', 'AL', 'AO', 'AR', 'AT', 'AU', 'BA', 'BE', 'BG', 'BO', 'BR', 'BY', 'BZ', 'CA', 'CD', 'CF', 'CG', 'CH', 'CI', 'CL', 'CM', 'CN', 'CO', 'CR', 'CU', 'CZ', 'DE', 'DJ', 'DK', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ES', 'ET', 'FI', 'FR', 'GA', 'GB', 'GE', 'GH', 'GI', 'GM', 'GN', 'GQ', 'GR', 'GT', 'GW', 'HK', 'HN', 'HR', 'HU', 'IE', 'IL', 'IN', 'IS', 'IT', 'JM', 'JP', 'KE', 'KH', 'KR', 'KW', 'KZ', 'LB', 'LT', 'LU', 'LV', 'LY', 'MA', 'MD', 'MK', 'ML', 'MM', 'MR', 'MX', 'MZ', 'NG', 'NI', 'NL', 'NO', 'NZ', 'OM', 'PA', 'PE', 'PH', 'PK', 'PL', 'PR', 'PT', 'PY', 'QA', 'RO', 'RS', 'RU', 'SA', 'SE', 'SG', 'SK', 'SL', 'SN', 'SV', 'TG', 'TH', 'TN', 'TR', 'TW', 'UA', 'US', 'UY', 'VE', 'VN', 'ZA', 'ZW']
    residence_index_categories = ['N', 'Y']
    channel_entrace_categories = ['004', '007', '013', '025', 'K00', 'KAA', 'KAB', 'KAC', 'KAD', 'KAE', 'KAF', 'KAG', 'KAH', 'KAI', 'KAJ', 'KAK', 'KAL', 'KAM', 'KAN', 'KAO', 'KAP', 'KAQ', 'KAR', 'KAS', 'KAT', 'KAU', 'KAV', 'KAW', 'KAY', 'KAZ', 'KBB', 'KBD', 'KBE', 'KBF', 'KBG', 'KBH', 'KBJ', 'KBL', 'KBM', 'KBN', 'KBO', 'KBP', 'KBQ', 'KBR', 'KBS', 'KBU', 'KBV', 'KBW', 'KBX', 'KBY', 'KBZ', 'KCA', 'KCB', 'KCC', 'KCD', 'KCE', 'KCF', 'KCG', 'KCH', 'KCI', 'KCJ', 'KCK', 'KCL', 'KCM', 'KCN', 'KCO', 'KCP', 'KCQ', 'KCR', 'KCS', 'KCT', 'KCU', 'KCV', 'KCX', 'KDA', 'KDB', 'KDC', 'KDD', 'KDE', 'KDF', 'KDG', 'KDH', 'KDI', 'KDL', 'KDM', 'KDN', 'KDO', 'KDP', 'KDQ', 'KDR', 'KDS', 'KDT', 'KDU', 'KDV', 'KDW', 'KDX', 'KDY', 'KDZ', 'KEA', 'KEB', 'KEC', 'KED', 'KEE', 'KEF', 'KEG', 'KEH', 'KEI', 'KEJ', 'KEK', 'KEL', 'KEM', 'KEN', 'KEO', 'KEQ', 'KES', 'KEU', 'KEV', 'KEW', 'KEY', 'KEZ', 'KFA', 'KFB', 'KFC', 'KFD', 'KFE', 'KFF', 'KFG', 'KFH', 'KFI', 'KFJ', 'KFK', 'KFL', 'KFM', 'KFN', 'KFP', 'KFR', 'KFS', 'KFT', 'KFU', 'KFV', 'KGC', 'KGN', 'KGU', 'KGV', 'KGW', 'KGX', 'KGY', 'KHA', 'KHC', 'KHD', 'KHE', 'KHF', 'KHK', 'KHL', 'KHM', 'KHN', 'KHO', 'KHP', 'KHQ', 'RED']
    customer_segment_model_categories = ['0-1 year', '2-4 years', 'More than 5 years']
    tax_rate_categories = ['0%', '20%', '40%', '45%']

    sps    = _compute_sps(customer)
    tsi    = _compute_tsi(customer)
    
    try:
        joined = pd.to_datetime(customer.get("first_join_date", "2018-01-01"))
        membership_days = float(max((pd.Timestamp.now() - joined).days, 30))
    except Exception:
        membership_days = 720.0

    # Clean & Map residence_country
    country = str(customer.get("residence_country", "ES")).upper()
    if country == "UK":
        country = "GB"
    if country not in residence_country_categories:
        country = "ES"

    # Clean & Map residence_index
    res_idx = str(customer.get("residence_index", "Y")).upper()
    if res_idx not in residence_index_categories:
        res_idx = "Y"

    # Clean & Map channel_entrace
    channel = str(customer.get("channel_entrace", "KHD")).upper()
    if channel not in channel_entrace_categories:
        channel = "KHD"

    # Clean & Map customer_segment_model
    segment = customer.get("customer_segment_model", customer.get("customer_segment", "02 - PARTICULARES"))
    segment_map = {
        "01 - TOP": "More than 5 years",
        "02 - PARTICULARES": "2-4 years",
        "03 - UNIVERSITARIO": "0-1 year"
    }
    if segment in segment_map:
        segment = segment_map[segment]
    if segment not in customer_segment_model_categories:
        segment = "0-1 year"

    # Clean & Map tax_rate
    tax = str(customer.get("tax_rate", "0%"))
    if tax == "0":
        tax = "0%"
    elif not tax.endswith("%") and tax.isdigit():
        tax = tax + "%"
    if tax not in tax_rate_categories:
        tax = "0%"

    data = {
        "residence_country": country,
        "gender": int(customer.get("gender", 0)),
        "age": int(customer.get("age", 35)),
        "residence_index": res_idx,
        "channel_entrace": channel,
        "activity_status": int(customer.get("activity_status", 1)),
        "household_gross_income": float(customer.get("household_gross_income", 30000.0)),
        "personal_income": float(customer.get("personal_income", 20000.0)),
        "number_of_children": int(customer.get("number_of_children", 0)),
        "employment_status": int(customer.get("employment_status", 1)),
        "customer_segment_model": segment,
        "tax_rate": tax,
        "avg_income_days_per_month": float(customer.get("avg_income_days_per_month", 2.0)),
        "income_amount_cv": float(customer.get("income_amount_cv", 0.8)),
        "avg_expense_days_per_month": float(customer.get("avg_expense_days_per_month", 3.0)),
        "expense_amount_cv": float(customer.get("expense_amount_cv", 0.8)),
        "avg_transactions_per_month": float(customer.get("avg_transactions_per_month", 1.5)),
        "monthly_transaction_std": float(customer.get("monthly_transaction_std", 0.5)),
        "active_months": float(customer.get("active_months", 12.0)),
        "SPS": float(sps),
        "TSI": float(tsi),
        "membership_days": float(membership_days)
    }

    df = pd.DataFrame([data])
    
    # Cast using defined CategoricalDtypes to guarantee matching categories with training set
    df['residence_country'] = df['residence_country'].astype(CategoricalDtype(categories=residence_country_categories, ordered=False))
    df['residence_index'] = df['residence_index'].astype(CategoricalDtype(categories=residence_index_categories, ordered=False))
    df['channel_entrace'] = df['channel_entrace'].astype(CategoricalDtype(categories=channel_entrace_categories, ordered=False))
    df['customer_segment_model'] = df['customer_segment_model'].astype(CategoricalDtype(categories=customer_segment_model_categories, ordered=False))
    df['tax_rate'] = df['tax_rate'].astype(CategoricalDtype(categories=tax_rate_categories, ordered=False))

    return df


# ─────────────────────────────────────────────
# LEFT PATH functions
# ─────────────────────────────────────────────

def get_banking_product_recommendations(customer: dict) -> list[dict]:
    """
    Run the multi-label banking category model and return a ranked
    list of recommended products with confidence scores.

    LEFT PATH products: saving_account, guarantees, junior_account,
                        loans, credit_card, pension, direct_debit
    """
    model  = _load_banking_model()
    X      = preprocess_customer_input(customer)

    try:
        proba = model.predict_proba(X)
    except AttributeError:
        # Some sklearn pipelines wrap predict_proba differently
        proba = [est.predict_proba(X) for est in model.estimators_]

    # Collect (label, confidence) pairs
    results = []
    for i, label in enumerate(BANKING_LABELS):
        # Support both MultiOutputClassifier and single-output list
        if isinstance(proba, list):
            conf = float(proba[i][0, 1])
        else:
            conf = float(proba[0, i]) if proba.ndim == 2 else float(proba[i][0, 1])
        results.append({"name": label, "confidence": conf})

    # Sort descending and keep only products with conf > 0.3
    results.sort(key=lambda x: x["confidence"], reverse=True)
    recommended = [r for r in results if r["confidence"] > 0.3]
    return recommended if recommended else results[:3]


def get_loan_recommendation(customer: dict) -> dict | None:
    """
    Rule-based loan recommendation inspired by the loan model notebook.

    Strategy (multi-level):
      1. Spending-based: dominant spending category → matching loan product
      2. Debt-consolidation: DTI > 0.25
      3. Safe fallback: credit_score >= 540 → Personal Loan
    """
    credit_score  = customer.get("credit_score", 650)
    income        = max(customer.get("personal_income", 1), 1)
    loan_amount   = customer.get("current_loan_amount", 0)
    activity      = customer.get("activity_status", 1)
    balance       = customer.get("avg_balance", 0)

    # DTI
    monthly_payment = loan_amount / max(36, 1)   # assume 3-year term
    monthly_income  = income / 12
    dti = monthly_payment / monthly_income if monthly_income > 0 else 1.0

    # Product catalog
    PRODUCTS = {
        "personal_loan":          {"rate_low": 6.5,  "rate_high": 15.0, "term_low": 12, "term_high": 60,  "min_score": 540},
        "auto_loan":              {"rate_low": 4.5,  "rate_high": 10.0, "term_low": 24, "term_high": 72,  "min_score": 580},
        "home_improvement_loan":  {"rate_low": 5.0,  "rate_high": 12.0, "term_low": 12, "term_high": 120, "min_score": 600},
        "education_loan":         {"rate_low": 3.5,  "rate_high": 8.0,  "term_low": 12, "term_high": 120, "min_score": 560},
        "travel_loan":            {"rate_low": 7.0,  "rate_high": 16.0, "term_low": 12, "term_high": 36,  "min_score": 550},
        "medical_loan":           {"rate_low": 6.0,  "rate_high": 14.0, "term_low": 12, "term_high": 60,  "min_score": 530},
        "business_loan":          {"rate_low": 5.5,  "rate_high": 13.0, "term_low": 24, "term_high": 84,  "min_score": 620},
        "debt_consolidation_loan":{"rate_low": 7.0,  "rate_high": 18.0, "term_low": 24, "term_high": 60,  "min_score": 520},
    }

    def _build_result(product_key: str, reason: str) -> dict:
        p = PRODUCTS.get(product_key, PRODUCTS["personal_loan"])
        # Personalise rate via credit score
        rate_range = p["rate_high"] - p["rate_low"]
        score_factor = max(0, (credit_score - 300) / 550)  # 0..1
        rate = p["rate_high"] - score_factor * rate_range
        return {
            "product":   product_key,
            "reason":    reason,
            "rate_low":  round(p["rate_low"], 2),
            "rate_high": round(p["rate_high"], 2),
            "rate":      round(rate, 2),
            "term_low":  p["term_low"],
            "term_high": p["term_high"],
        }

    # Level 1 – spending-based (simple rule on avg_balance / income)
    if balance > income * 0.5 and credit_score >= 600:
        return _build_result("home_improvement_loan", "High balance relative to income – home improvement loan matches profile.")

    # Level 2 – debt consolidation
    if dti > 0.25 and credit_score >= PRODUCTS["debt_consolidation_loan"]["min_score"] and activity:
        return _build_result("debt_consolidation_loan", f"Debt-to-income ratio {dti:.1%} > 25% – debt consolidation recommended.")

    # Level 3 – safe fallback
    if credit_score >= PRODUCTS["personal_loan"]["min_score"] and activity:
        return _build_result("personal_loan", "Personal loan offered as a standard product for eligible customers.")

    return None  # Not eligible


# ─────────────────────────────────────────────
# RIGHT PATH functions
# ─────────────────────────────────────────────

def get_rfm_features(customer: dict) -> dict:
    """Return RFM dict from pre-aggregated inputs provided in the sidebar."""
    return {
        "recency":   int(customer.get("recency_days", 30)),
        "frequency": int(customer.get("frequency_per_month", 5)),
        "monetary":  float(customer.get("monetary_per_month", 1_250)),
    }


# Segment profile reference (cluster descriptions based on notebook)
_SEGMENT_PROFILES = [
    {
        "name":        "Cluster 1 - High-Value Loyalists",
        "description": "Long-tenured, frequent, high-monetary customers with premium engagement.",
        "badge":       "high",
    },
    {
        "name":        "Cluster 2 - Occasional Big-Spenders",
        "description": "Infrequent visits but large transaction amounts when active.",
        "badge":       "medium",
    },
    {
        "name":        "Cluster 3 - Regular Modest Customers",
        "description": "Consistent but lower-value spending patterns.",
        "badge":       "low",
    },
]


def get_customer_segment(rfm: dict) -> dict:
    """
    Use KMeans customer-segmentation model to assign a segment.
    Falls back to a simple rule-based heuristic if the model
    feature space doesn't match.
    """
    try:
        model = _load_seg_model()
        X = np.array([[rfm["recency"], rfm["frequency"], rfm["monetary"]]], dtype=float)

        # Standardise (the saved model may contain a scaler or raw KMeans)
        # Try predict directly; if it fails, fall back to heuristic
        label = int(model.predict(X)[0])
        n_clusters = len(_SEGMENT_PROFILES)
        label = label % n_clusters  # guard against unexpected cluster count
        return _SEGMENT_PROFILES[label]
    except Exception:
        pass  # fall through to heuristic

    # Heuristic fallback
    r, f, m = rfm["recency"], rfm["frequency"], rfm["monetary"]
    if r <= 30 and f >= 5 and m >= 1_000:
        return _SEGMENT_PROFILES[0]
    elif m >= 1_000:
        return _SEGMENT_PROFILES[1]
    else:
        return _SEGMENT_PROFILES[2]


def get_credit_card_recommendation(customer: dict, segment: dict) -> dict | None:
    """
    Attempt to use the KMeans CC-subtype model.
    Falls back to a rule-based spending-category approach.

    Returns a dict with: product, description, spending_category,
                         affinity_score, qualified.
    """
    # Spending percentages (sidebar inputs used as proxy)
    # In a real pipeline these would come from actual transaction data
    total_spend = max(customer.get("monetary_per_month", 1_000), 1)

    # Simulate spending distribution from sidebar inputs (simplified)
    # In production: read real transaction data and compute per category
    spend_retail        = total_spend * 0.30
    spend_groceries     = total_spend * 0.20
    spend_entertainment = total_spend * 0.15
    spend_fuel          = total_spend * 0.10
    spend_travel        = total_spend * 0.10
    spend_other         = total_spend * 0.15

    categories = {
        "FUEL":          spend_fuel / total_spend,
        "RETAIL":        spend_retail / total_spend,
        "TRAVEL":        spend_travel / total_spend,
        "ENTERTAINMENT": spend_entertainment / total_spend,
        "GROCERIES":     spend_groceries / total_spend,
        "OTHER":         spend_other / total_spend,
    }

    # Try to use the CC KMeans model for spending cluster
    try:
        model = _load_cc_model()
        pct_features = np.array(
            [[categories[c] for c in ["FUEL", "RETAIL", "TRAVEL", "ENTERTAINMENT", "GROCERIES", "OTHER"]]],
            dtype=float,
        )
        cluster = int(model.predict(pct_features)[0])
        # Map cluster → category (best effort; real mapping derived from notebook)
        cat_list = list(categories.keys())
        dominant_cat = cat_list[cluster % len(cat_list)]
    except Exception:
        # Fallback: pick category with highest pct
        dominant_cat = max(categories, key=categories.get)

    affinity_score = categories[dominant_cat]
    product        = CC_PRODUCT_MAP.get(dominant_cat, "Standard Cashback Card")
    description    = CC_PRODUCT_DESC.get(product, "")
    qualified      = affinity_score >= AFFINITY_THRESHOLD

    return {
        "product":          product,
        "description":      description,
        "spending_category": dominant_cat,
        "affinity_score":   round(affinity_score, 4),
        "qualified":        qualified,
    }
