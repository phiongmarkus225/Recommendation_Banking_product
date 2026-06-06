import streamlit as st
import numpy as np
import pandas as pd
import os, sys
import requests

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Banking Recommendation", page_icon="🏦", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.hero{background:linear-gradient(135deg,#1a1a2e,#0f3460);border-radius:16px;padding:32px 40px;margin-bottom:24px;}
.hero h1{color:#e2e8f0;font-size:2rem;margin:0;}
.hero h1 span{color:#00d4ff;}
.hero p{color:#a0aec0;margin:4px 0 0;}
.prod{background:linear-gradient(135deg,#0f2744,#1a3a5c);border:1px solid #2a5080;border-radius:12px;padding:16px 20px;margin:8px 0;}
.pname{font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:4px;}
.pdesc{font-size:.83rem;color:#a0aec0;margin-bottom:8px;}
.bar-bg{background:#0a1520;border-radius:999px;height:8px;overflow:hidden;}
.bar-fill{height:8px;border-radius:999px;background:linear-gradient(90deg,#00d4ff,#0066cc);}
.chip{display:inline-flex;flex-direction:column;align-items:center;background:#0f1d2e;border:1px solid #2d3e50;border-radius:10px;padding:10px 16px;margin:4px;min-width:100px;}
.chip .l{font-size:.7rem;color:#718096;text-transform:uppercase;letter-spacing:.05em;}
.chip .v{font-size:1.1rem;font-weight:700;color:#00d4ff;}
.badge-high{background:rgba(0,200,100,.15);color:#00c864;border:1px solid #00c864;border-radius:999px;padding:3px 12px;font-size:.78rem;font-weight:600;}
.badge-medium{background:rgba(255,180,0,.15);color:#ffb400;border:1px solid #ffb400;border-radius:999px;padding:3px 12px;font-size:.78rem;font-weight:600;}
.badge-low{background:rgba(255,80,80,.15);color:#ff5050;border:1px solid #ff5050;border-radius:999px;padding:3px 12px;font-size:.78rem;font-weight:600;}
.stButton>button{background:linear-gradient(135deg,#0f3460,#1a6fb8)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "clean_customer_dataNEW.csv")
    try:
        return pd.read_csv(csv_path)
    except:
        return pd.DataFrame()

df = load_data()

# Helper to generate deterministic missing fields based on customer_id
def get_full_customer_data(row, cust_id):
    np.random.seed(int(cust_id))
    inc = row.get("personal_income", 30000)
    return {
        "age": row.get("age", 35),
        "gender": 1 if str(row.get("gender"))=="1" else 0,
        "residence_country": row.get("residence_country", "ES"),
        "residence_index": row.get("residence_index", "Y"),
        "channel_entrace": row.get("channel_entrace", "KHD"),
        "activity_status": row.get("activity_status", 1),
        "household_gross_income": row.get("household_gross_income", inc*1.2),
        "personal_income": inc,
        "credit_score": row.get("credit_score", int(np.random.randint(500, 800))),
        "current_loan_amount": row.get("current_loan_amount", 0.0),
        "number_of_children": row.get("number_of_children", 0),
        "employment_status": row.get("employment_status", 1),
        "customer_segment": row.get("customer_segment_model", "02 - PARTICULARES"),
        "first_join_date": str(row.get("first_join_date", "2019-01-01")),
        "tax_rate": "0",
        "avg_balance": float(np.random.uniform(inc*0.05, inc*0.2)),
        "min_balance": float(np.random.uniform(100, inc*0.05)),
        "max_balance": float(np.random.uniform(inc*0.2, inc*0.5)),
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
        "spend_fuel": np.random.uniform(0, 0.3),
        "spend_retail": np.random.uniform(0.1, 0.5),
        "spend_travel": np.random.uniform(0, 0.2),
        "spend_entertain": np.random.uniform(0.05, 0.3),
        "spend_grocery": np.random.uniform(0.1, 0.4),
        "spend_other": np.random.uniform(0.05, 0.2)
    }

st.markdown("""<div class="hero"><h1>🏦 Banking Product <span>Recommendation</span> System</h1>
<p>AI-powered pipeline: Profile → Feature Engineering → Model → Recommendation</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🆕 Nasabah Baru", "👤 Nasabah yang Ada", "💳 Segmentasi Credit Card"])

# ── TAB 1: NEW CUSTOMER ────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🆕 Rekomendasi Produk untuk Nasabah Baru")
    st.info("Masukkan profil nasabah baru untuk mendapatkan rekomendasi produk perbankan awal.")
    with st.form("form_new"):
        c1,c2,c3 = st.columns(3)
        with c1:
            n_age = st.slider("Usia",18,80,30)
            n_gender = st.selectbox("Gender",["Male","Female"],key="ng")
            n_country = st.selectbox("Negara",["ES","FR","DE","UK","IT"],key="nc")
        with c2:
            n_income = st.number_input("Pendapatan Rumah Tangga (€)",5000,500000,35000,step=1000,key="ni")
            n_personal_income = st.number_input("Pendapatan Pribadi (€)",5000,300000,25000,step=1000,key="npi")
            n_credit = st.slider("Credit Score",300,850,600,key="ncs")
        with c3:
            n_children = st.slider("Jml Anak",0,6,0,key="nch")
            n_emp = st.selectbox("Status Kerja",[1,0,2],format_func=lambda x:{1:"Employed",0:"Unemployed",2:"Self-Employed"}[x],key="nem")
            n_seg = st.selectbox("Segmen",["01 - TOP","02 - PARTICULARES","03 - UNIVERSITARIO"],key="nseg")
        n_channel = st.selectbox("Channel",["KHD","KHE","KHF","KHL","KFC","KDB","KAT","KAZ"],key="nch2")
        n_submit = st.form_submit_button("🔍 Rekomendasikan", use_container_width=True)

    if n_submit:
        payload = {
            "age": int(n_age),
            "gender": n_gender,
            "residence_country": n_country,
            "household_gross_income": float(n_income),
            "personal_income": float(n_personal_income),
            "credit_score": int(n_credit),
            "number_of_children": int(n_children),
            "employment_status": int(n_emp),
            "customer_segment": n_seg,
            "channel_entrace": n_channel
        }
        with st.spinner("Menganalisis profil nasabah baru..."):
            try:
                res = requests.post(f"{API_URL}/api/recommend/new", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    prods = data["products"]
                    loan = data["loan"]
                    ok = True
                else:
                    ok = False; err = f"API Error: {res.text}"
            except Exception as e:
                ok = False; err = str(e)
                
        if not ok:
            st.error(f"Error: {err}")
        else:
            st.success("✅ Rekomendasi berhasil digenerate!")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("#### 🎯 Produk Perbankan yang Direkomendasikan")
                for p in prods:
                    nm = p["name"].replace("_"," ").title(); cf = p["confidence"]
                    st.markdown(f"""<div class="prod"><div class="pname">{nm}</div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{cf*100:.0f}%"></div></div>
                        <div class="pdesc">Confidence: {cf*100:.1f}%</div></div>""",unsafe_allow_html=True)
            with c2:
                st.markdown("#### 💰 Rekomendasi Pinjaman")
                if loan:
                    ln = loan["product"].replace("_"," ").title()
                    st.markdown(f"""<div class="prod"><div class="pname">{ln}</div>
                        <div class="pdesc">{loan["reason"]}</div>
                        <div><span class="chip"><span class="l">Rate</span><span class="v">{loan["rate_low"]}%-{loan["rate_high"]}%</span></span>
                        <span class="chip"><span class="l">Tenor</span><span class="v">{loan["term_low"]}-{loan["term_high"]} bln</span></span></div>
                        </div>""",unsafe_allow_html=True)
                else:
                    st.info("Belum memenuhi syarat pinjaman saat ini.")

# ── TAB 2: EXISTING CUSTOMER ───────────────────────────────────────────────
with tab2:
    st.markdown("### 👤 Rekomendasi untuk Nasabah yang Ada")
    st.info("Pilih Customer ID dari database untuk menampilkan rekomendasi produk yang disesuaikan secara otomatis.")
    
    if df.empty:
        st.error("Gagal memuat dataset nasabah. Harap cek clean_customer_dataNEW.csv")
    else:
        with st.form("form_exist"):
            e_cust_id = st.selectbox("Pilih Customer ID", df["customer_id"].unique()[:1000], index=0)
            e_submit = st.form_submit_button("🔍 Rekomendasikan", use_container_width=True)

        if e_submit:
            with st.spinner("Menjalankan pipeline rekomendasi..."):
                try:
                    payload = {"customer_id": int(e_cust_id)}
                    res = requests.post(f"{API_URL}/api/recommend/existing", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        cust_profile = data["customer_profile"]
                        prods = data["products"]
                        loan = data["loan"]
                        rfm = data["rfm"]
                        seg = data["segment"]
                        cc = data["credit_card"]
                        
                        cust = {
                            "age": cust_profile["age"],
                            "personal_income": cust_profile["personal_income"],
                            "credit_score": cust_profile["credit_score"],
                            "activity_status": 1 if cust_profile["activity_status"] == "Aktif" else 0,
                            "gender": 1 if cust_profile["gender"] == "Female" else 0
                        }
                        ok = True
                    else:
                        ok = False; err = f"API Error: {res.text}"
                except Exception as e:
                    ok = False; err = str(e)
                    
            if not ok:
                st.error(f"Error: {err}")
            else:
                st.success(f"✅ Analisis berhasil untuk Customer ID: {e_cust_id}")
                
                st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px;background:#1a2434;padding:16px;border-radius:12px;border:1px solid #253347;">
                    <div style="margin-right:20px"><strong style="color:#a0aec0">Usia:</strong> {cust["age"]} thn</div>
                    <div style="margin-right:20px"><strong style="color:#a0aec0">Pendapatan:</strong> €{cust["personal_income"]:,.0f}</div>
                    <div style="margin-right:20px"><strong style="color:#a0aec0">Credit Score:</strong> {cust["credit_score"]}</div>
                    <div><strong style="color:#a0aec0">Status:</strong> {'Aktif' if cust['activity_status']==1 else 'Tidak Aktif'}</div>
                </div>""", unsafe_allow_html=True)
                
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🎯 Produk Perbankan")
                    for p in prods:
                        nm = p["name"].replace("_"," ").title(); cf = p["confidence"]
                        st.markdown(f"""<div class="prod"><div class="pname">{nm}</div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{cf*100:.0f}%"></div></div>
                            <div class="pdesc">Confidence: {cf*100:.1f}%</div></div>""",unsafe_allow_html=True)
                    st.markdown("#### 💰 Pinjaman")
                    if loan:
                        ln = loan["product"].replace("_"," ").title()
                        st.markdown(f"""<div class="prod"><div class="pname">{ln}</div>
                            <div class="pdesc">{loan["reason"]}</div>
                            <span class="chip"><span class="l">Rate</span><span class="v">{loan["rate_low"]}%-{loan["rate_high"]}%</span></span>
                            <span class="chip"><span class="l">Tenor</span><span class="v">{loan["term_low"]}-{loan["term_high"]} bln</span></span>
                            </div>""",unsafe_allow_html=True)
                    else:
                        st.info("Tidak memenuhi syarat pinjaman.")
                with c2:
                    st.markdown("#### 📊 Segmen & Credit Card")
                    sn = seg["name"]; sd = seg["description"]; sb = seg["badge"]
                    st.markdown(f"""<div class="prod"><span class="badge-{sb}">{sn}</span>
                        <div class="pdesc" style="margin-top:8px">{sd}</div></div>""",unsafe_allow_html=True)
                    st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:6px;margin:12px 0">
                        <div class="chip"><span class="l">Recency</span><span class="v">{rfm["recency"]}d</span></div>
                        <div class="chip"><span class="l">Frequency</span><span class="v">{rfm["frequency"]}/bln</span></div>
                        <div class="chip"><span class="l">Monetary</span><span class="v">€{rfm["monetary"]:,.0f}</span></div>
                        </div>""",unsafe_allow_html=True)
                    if cc and cc.get("qualified"):
                        cn = cc["product"]; cd = cc["description"]; ca = cc["affinity_score"]; ccat = cc["spending_category"]
                        st.markdown(f"""<div class="prod"><div class="pname">💳 {cn}</div>
                            <div class="pdesc">{cd}</div>
                            <div class="pdesc">🛒 Kategori: {ccat.title()}</div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{min(ca*100,100):.0f}%"></div></div>
                            <div class="pdesc">Affinity: {ca*100:.1f}%</div></div>""",unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada rekomendasi kartu kredit.")

# ── TAB 3: CREDIT CARD SEGMENTATION ───────────────────────────────────────
with tab3:
    st.markdown("### 💳 Segmentasi Nasabah Pengguna Credit Card")
    st.info("Pilih Customer ID untuk langsung melihat segmentasi dan rekomendasi kartu kredit.")
    
    if not df.empty:
        with st.form("form_cc"):
            s_cust_id = st.selectbox("Pilih Customer ID", df["customer_id"].unique()[:1000], index=0, key="cc_cust_id")
            s_submit = st.form_submit_button("🔍 Segmentasi & Rekomendasikan", use_container_width=True)

        if s_submit:
            with st.spinner("Menjalankan segmentasi..."):
                try:
                    payload = {"customer_id": int(s_cust_id)}
                    res = requests.post(f"{API_URL}/api/segment", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        rfm_data = data["rfm"]
                        s_rec = rfm_data["recency"]
                        s_freq = rfm_data["frequency"]
                        s_mon = rfm_data["monetary"]
                        
                        spending_patterns = data["spending_patterns"]
                        s_fuel = data["raw_spends"]["fuel"]
                        s_retail = data["raw_spends"]["retail"]
                        s_travel = data["raw_spends"]["travel"]
                        s_entertain = data["raw_spends"]["entertain"]
                        s_grocery = data["raw_spends"]["grocery"]
                        s_other = data["raw_spends"]["other"]
                        total_pct = s_fuel + s_retail + s_travel + s_entertain + s_grocery + s_other
                        
                        seg_name = data["segment_name"]
                        seg_desc = data["segment_description"]
                        seg_badge = data["segment_badge"]
                        
                        cc_rec = data["credit_card_recommendation"]
                        credit_score = data["credit_score"]
                        ok = True
                    else:
                        ok = False; err = f"API Error: {res.text}"
                except Exception as e:
                    ok = False; err = str(e)
                    
            if not ok:
                st.error(f"Error: {err}")
            else:
                st.success(f"✅ Segmentasi berhasil untuk Customer ID: {s_cust_id}")
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown("#### 🔍 Segmen Nasabah")
                    st.markdown(f"""<div class="prod"><span class="badge-{seg_badge}">{seg_name}</span>
                        <div class="pdesc" style="margin-top:8px">{seg_desc}</div>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">
                        <div class="chip"><span class="l">Recency</span><span class="v">{s_rec}d</span></div>
                        <div class="chip"><span class="l">Frequency</span><span class="v">{s_freq}/bln</span></div>
                        <div class="chip"><span class="l">Monetary</span><span class="v">€{s_mon:,.0f}</span></div>
                        </div></div>""",unsafe_allow_html=True)
                with c2:
                    st.markdown("#### 📊 Pola Pengeluaran")
                    cats = {"Bahan Bakar":s_fuel,"Retail":s_retail,"Travel":s_travel,
                        "Hiburan":s_entertain,"Groceries":s_grocery,"Lainnya":s_other}
                    for cat,pct in sorted(cats.items(),key=lambda x:-x[1]):
                        norm = pct/total_pct
                        st.markdown(f"""<div style="margin-bottom:10px">
                            <div style="display:flex;justify-content:space-between;color:#a0aec0;font-size:.82rem">
                            <span>{cat}</span><span>{norm*100:.1f}%</span></div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{norm*100:.0f}%"></div></div>
                            </div>""",unsafe_allow_html=True)
                with c3:
                    st.markdown("#### 💳 Rekomendasi Kartu")
                    if credit_score >= 540 and cc_rec:
                        cn = cc_rec["card_name"]
                        cd = cc_rec["description"]
                        bonus = cc_rec["bonus"]
                        affinity = cc_rec["spending_affinity"]
                        dom = cc_rec["dominant_category"]
                        st.markdown(f"""<div class="prod"><div class="pname">💳 {cn}</div>
                            <div class="pdesc">{cd}</div>
                            <div class="pdesc" style="color:#00d4ff">✨ {bonus}</div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{min(affinity,100):.0f}%"></div></div>
                            <div class="pdesc">Spending Affinity: {affinity:.1f}%</div>
                            <div class="pdesc">Kategori Dominan: {dom}</div></div>""",unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Credit score {credit_score} terlalu rendah untuk kartu kredit (min: 540).")

st.markdown("<p style='text-align:center;color:#4a6080;font-size:.8rem;margin-top:32px'>"
    "Banking Product Recommendation System • Powered by Machine Learning</p>",unsafe_allow_html=True)
