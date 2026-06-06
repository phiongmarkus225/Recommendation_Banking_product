import os
import sys
import json
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    get_banking_product_recommendations,
    get_loan_recommendation,
    get_rfm_features,
    get_customer_segment,
    get_credit_card_recommendation
)

app = FastAPI(
    title="API Sistem Rekomendasi Produk Perbankan",
    description="API Gateway untuk merekomendasikan produk bank dan menganalisis segmentasi nasabah.",
    version="1.0.0"
)

# -------------------------------------------------------------
# Database Setup for Logging
# -------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            endpoint TEXT,
            request_payload TEXT,
            response_payload TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_prediction(endpoint: str, request_payload: dict, response_payload: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prediction_logs (endpoint, request_payload, response_payload) VALUES (?, ?, ?)",
            (endpoint, json.dumps(request_payload), json.dumps(response_payload))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to SQLite database: {e}")

# -------------------------------------------------------------
# Global Customer Data Loading (for existing customers)
# -------------------------------------------------------------
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "clean_customer_dataNEW.csv"
)

try:
    df_customers = pd.read_csv(CSV_PATH)
    print(f"Database customer loaded successfully: {len(df_customers)} rows.")
except Exception as e:
    df_customers = pd.DataFrame()
    print(f"Warning: Failed to load customer database: {e}")

def get_full_customer_data(row, cust_id):
    np.random.seed(int(cust_id))
    inc = row.get("personal_income", 30000)
    return {
        "age": int(row.get("age", 35)),
        "gender": 1 if str(row.get("gender")) == "1" else 0,
        "residence_country": row.get("residence_country", "ES"),
        "residence_index": row.get("residence_index", "Y"),
        "channel_entrace": row.get("channel_entrace", "KHD"),
        "activity_status": int(row.get("activity_status", 1)),
        "household_gross_income": float(row.get("household_gross_income", inc * 1.2)),
        "personal_income": float(inc),
        "credit_score": int(row.get("credit_score", int(np.random.randint(500, 800)))),
        "current_loan_amount": float(row.get("current_loan_amount", 0.0)),
        "number_of_children": int(row.get("number_of_children", 0)),
        "employment_status": int(row.get("employment_status", 1)),
        "customer_segment": row.get("customer_segment_model", "02 - PARTICULARES"),
        "first_join_date": str(row.get("first_join_date", "2019-01-01")),
        "tax_rate": "0",
        "avg_balance": float(np.random.uniform(inc * 0.05, inc * 0.2)),
        "min_balance": float(np.random.uniform(100, inc * 0.05)),
        "max_balance": float(np.random.uniform(inc * 0.2, inc * 0.5)),
        "total_transactions": int(np.random.randint(10, 100)),
        "active_months": int(np.random.randint(6, 60)),
        "avg_monthly_transaction_count": float(np.random.uniform(1.0, 10.0)),
        "avg_income_days_per_month": float(np.random.uniform(1.0, 4.0)),
        "avg_expense_days_per_month": float(np.random.uniform(2.0, 15.0)),
        "avg_transactions_per_month": float(np.random.uniform(1.0, 10.0)),
        "monthly_transaction_std": float(np.random.uniform(0.1, 2.0)),
        "expense_amount_cv": float(np.random.uniform(0.1, 1.5)),
        "recency_days": int(np.random.randint(1, 90)),
        "frequency_per_month": int(np.random.randint(1, 15)),
        "monetary_per_month": float(np.random.uniform(100, 5000)),
        "spend_fuel": float(np.random.uniform(0, 0.3)),
        "spend_retail": float(np.random.uniform(0.1, 0.5)),
        "spend_travel": float(np.random.uniform(0, 0.2)),
        "spend_entertain": float(np.random.uniform(0.05, 0.3)),
        "spend_grocery": float(np.random.uniform(0.1, 0.4)),
        "spend_other": float(np.random.uniform(0.05, 0.2))
    }

# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class NewCustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Usia nasabah")
    gender: str = Field(..., description="Gender: Male atau Female")
    residence_country: str = Field("ES", description="Negara tempat tinggal")
    household_gross_income: float = Field(..., gt=0, description="Pendapatan Rumah Tangga per tahun (€)")
    personal_income: float = Field(..., gt=0, description="Pendapatan Pribadi per tahun (€)")
    credit_score: int = Field(..., ge=300, le=850, description="Skor kredit (300-850)")
    number_of_children: int = Field(0, ge=0, description="Jumlah anak")
    employment_status: int = Field(1, description="Status kerja: 1=Employed, 0=Unemployed, 2=Self-Employed")
    customer_segment: str = Field("02 - PARTICULARES", description="Segmen nasabah")
    channel_entrace: str = Field("KHD", description="Channel pendaftaran")

class ExistingCustomerRequest(BaseModel):
    customer_id: int = Field(..., description="ID Nasabah yang ada di database")

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "Banking Recommendation API Gateway",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/recommend/new")
def recommend_new(input_data: NewCustomerInput):
    cust = {
        "age": input_data.age,
        "gender": 1 if input_data.gender == "Female" else 0,
        "residence_country": input_data.residence_country,
        "residence_index": "Y",
        "channel_entrace": input_data.channel_entrace,
        "activity_status": 1,
        "household_gross_income": input_data.household_gross_income,
        "personal_income": input_data.personal_income,
        "credit_score": input_data.credit_score,
        "current_loan_amount": 0.0,
        "number_of_children": input_data.number_of_children,
        "employment_status": input_data.employment_status,
        "customer_segment": input_data.customer_segment,
        "first_join_date": str(pd.Timestamp.today().date()),
        "tax_rate": "0",
        "avg_balance": input_data.personal_income * 0.1,
        "min_balance": 1000.0,
        "max_balance": input_data.personal_income * 0.2,
        "total_transactions": 5,
        "active_months": 1,
        "avg_monthly_transaction_count": 1.0,
        "avg_income_days_per_month": 1.0,
        "avg_expense_days_per_month": 2.0,
        "avg_transactions_per_month": 1.0,
        "monthly_transaction_std": 0.3,
        "expense_amount_cv": 0.5,
        "recency_days": 1,
        "frequency_per_month": 1,
        "monetary_per_month": input_data.personal_income / 12.0
    }
    
    try:
        prods = get_banking_product_recommendations(cust)
        loan = get_loan_recommendation(cust)
        
        response = {
            "products": prods,
            "loan": loan
        }
        
        log_prediction("/api/recommend/new", input_data.model_dump(), response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

@app.post("/api/recommend/existing")
def recommend_existing(req: ExistingCustomerRequest):
    if df_customers.empty:
        raise HTTPException(status_code=503, detail="Database nasabah kosong atau tidak termuat.")
    
    customer_row = df_customers[df_customers["customer_id"] == req.customer_id]
    if customer_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {req.customer_id} tidak ditemukan di database.")
        
    cust = get_full_customer_data(customer_row.iloc[0], req.customer_id)
    
    try:
        prods = get_banking_product_recommendations(cust)
        loan = get_loan_recommendation(cust)
        rfm = get_rfm_features(cust)
        seg = get_customer_segment(rfm)
        cc = get_credit_card_recommendation(cust, seg)
        
        response = {
            "customer_profile": {
                "age": cust["age"],
                "gender": "Female" if cust["gender"] == 1 else "Male",
                "personal_income": cust["personal_income"],
                "credit_score": cust["credit_score"],
                "activity_status": "Aktif" if cust["activity_status"] == 1 else "Tidak Aktif"
            },
            "products": prods,
            "loan": loan,
            "rfm": rfm,
            "segment": seg,
            "credit_card": cc
        }
        
        log_prediction("/api/recommend/existing", req.model_dump(), response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

@app.post("/api/segment")
def segment_customer(req: ExistingCustomerRequest):
    if df_customers.empty:
        raise HTTPException(status_code=503, detail="Database nasabah kosong atau tidak termuat.")
        
    customer_row = df_customers[df_customers["customer_id"] == req.customer_id]
    if customer_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {req.customer_id} tidak ditemukan di database.")
        
    cust = get_full_customer_data(customer_row.iloc[0], req.customer_id)
    
    try:
        rfm_data = {
            "recency": cust["recency_days"],
            "frequency": cust["frequency_per_month"],
            "monetary": float(cust["monetary_per_month"])
        }
        
        seg = get_customer_segment(rfm_data)
        
        # Spend percentages
        s_fuel = cust["spend_fuel"]
        s_retail = cust["spend_retail"]
        s_travel = cust["spend_travel"]
        s_entertain = cust["spend_entertain"]
        s_grocery = cust["spend_grocery"]
        s_other = cust["spend_other"]
        total_pct = s_fuel + s_retail + s_travel + s_entertain + s_grocery + s_other
        
        spending_patterns = {
            "Bahan Bakar": round((s_fuel / total_pct) * 100, 2),
            "Retail": round((s_retail / total_pct) * 100, 2),
            "Travel": round((s_travel / total_pct) * 100, 2),
            "Hiburan": round((s_entertain / total_pct) * 100, 2),
            "Groceries": round((s_grocery / total_pct) * 100, 2),
            "Lainnya": round((s_other / total_pct) * 100, 2)
        }
        
        # Credit Card Recommendation
        spend_map = {
            "FUEL": s_fuel, "RETAIL": s_retail, "TRAVEL": s_travel,
            "ENTERTAINMENT": s_entertain, "GROCERIES": s_grocery, "OTHER": s_other
        }
        dom = max(spend_map, key=spend_map.get)
        cc_map = {
            "FUEL": "Fuel Rewards Card", "RETAIL": "Shopping Rewards Card",
            "TRAVEL": "Travel Miles Card", "ENTERTAINMENT": "Entertainment Plus Card",
            "GROCERIES": "Everyday Cashback Card", "OTHER": "Standard Cashback Card"
        }
        cc_desc = {
            "Fuel Rewards Card": "Cashback untuk BBM dan tol.",
            "Shopping Rewards Card": "Reward untuk belanja retail dan fashion.",
            "Travel Miles Card": "Miles untuk penerbangan dan hotel.",
            "Entertainment Plus Card": "Cashback untuk makan, streaming, hiburan.",
            "Everyday Cashback Card": "Cashback untuk belanja harian dan groceries.",
            "Standard Cashback Card": "Cashback fleksibel untuk semua kategori."
        }
        cc_seg_bonus = {
            "Cluster 1 - High-Value Loyalists": "Bonus limit tinggi dan reward premium.",
            "Cluster 2 - Occasional Big-Spenders": "Cashback lebih tinggi untuk transaksi besar.",
            "Cluster 3 - Regular Modest Customers": "Cicilan 0% dan reward poin harian."
        }
        
        card_name = cc_map.get(dom, "Standard Cashback Card")
        card_desc = cc_desc.get(card_name, "")
        bonus = cc_seg_bonus.get(seg["name"], "")
        affinity = spend_map[dom] / total_pct
        
        cc_rec = None
        if cust["credit_score"] >= 540:
            cc_rec = {
                "card_name": card_name,
                "description": card_desc,
                "bonus": bonus,
                "spending_affinity": round(affinity * 100, 2),
                "dominant_category": dom.title()
            }
            
        response = {
            "segment_name": seg["name"],
            "segment_description": seg["description"],
            "segment_badge": seg["badge"],
            "rfm": rfm_data,
            "spending_patterns": spending_patterns,
            "credit_card_recommendation": cc_rec,
            "raw_spends": {
                "fuel": float(s_fuel),
                "retail": float(s_retail),
                "travel": float(s_travel),
                "entertain": float(s_entertain),
                "grocery": float(s_grocery),
                "other": float(s_other)
            },
            "credit_score": int(cust["credit_score"])
        }
        
        log_prediction("/api/segment", req.model_dump(), response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation Pipeline Error: {str(e)}")

@app.get("/api/logs")
def get_logs(limit: int = 10):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, endpoint, request_payload, response_payload FROM prediction_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for r in rows:
            logs.append({
                "id": r[0],
                "timestamp": r[1],
                "endpoint": r[2],
                "request": json.loads(r[3]),
                "response": json.loads(r[4])
            })
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Read Error: {str(e)}")
