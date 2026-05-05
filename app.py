import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor – E-Commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .hero-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem;
        color: white; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .hero-card h1 { font-size: 1.9rem; margin-bottom: 0.3rem; }
    .hero-card p  { opacity: 0.75; margin: 0; font-size: 0.95rem; }

    .badge-churn {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white; border-radius: 50px;
        padding: 0.5rem 1.6rem;
        font-size: 1.05rem; font-weight: 700;
    }
    .badge-safe {
        display: inline-block;
        background: linear-gradient(135deg, #27ae60, #1e8449);
        color: white; border-radius: 50px;
        padding: 0.5rem 1.6rem;
        font-size: 1.05rem; font-weight: 700;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 700;
        color: #e0e0e0; border-bottom: 2px solid #e94560;
        padding-bottom: 0.35rem; margin-bottom: 1rem;
    }
    .info-box {
        background: #1a3a52; border-left: 4px solid #4aa3df;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.88rem; margin-bottom: 1rem;
        color: #d6eaf8 !important;
    }
    .info-box a { color: #7dc9f5 !important; }
    .info-box b { color: #ffffff !important; }
    .info-box code {
        background: rgba(255,255,255,0.12);
        color: #a8d8ff !important;
        padding: 0.1rem 0.35rem; border-radius: 4px;
    }
    .warn-box {
        background: #3d2e00; border-left: 4px solid #f39c12;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.88rem; margin-bottom: 1rem;
        color: #fdebd0 !important;
    }
    .warn-box b { color: #ffffff !important; }
    .warn-box code {
        background: rgba(255,255,255,0.12);
        color: #ffd580 !important;
        padding: 0.1rem 0.35rem; border-radius: 4px;
    }
    .err-box {
        background: #3d0a14; border-left: 4px solid #e94560;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.88rem; margin-bottom: 1rem;
        color: #f5b7b1 !important;
    }
    .err-box b { color: #ffffff !important; }
    .err-box code {
        background: rgba(255,255,255,0.12);
        color: #ff9aab !important;
        padding: 0.1rem 0.35rem; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Fitur TEPAT yang dipakai saat training
# (sesuai header CSV training)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "DaySinceLastOrder",
    "CashbackAmount",
]

PREF_ORDER_CATS  = ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
MARITAL_STATUSES = ["Single", "Divorced", "Married"]

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    for path in ["models/best_churn_model.pkl", "best_churn_model.pkl"]:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

# ─────────────────────────────────────────────
# Predict helper
# ─────────────────────────────────────────────
def predict(df: pd.DataFrame):
    """Return (labels ndarray, probabilities ndarray)."""
    preds = model.predict(df)
    try:
        probas = model.predict_proba(df)[:, 1]
    except AttributeError:
        # SVM tanpa probability=True → pakai decision_function + sigmoid
        scores = model.decision_function(df)
        probas = 1 / (1 + np.exp(-scores))
    return preds, probas

def risk_label(p: float) -> str:
    if p >= 0.7: return "🔴 Tinggi"
    if p >= 0.4: return "🟡 Sedang"
    return "🟢 Rendah"

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1>🛒 Customer Churn Predictor</h1>
    <p>E-Commerce · Capstone ML Project · Model: SVM | Pipeline: OHE · BinaryEncoder · Imputer · RobustScaler · SelectKBest · SMOTE</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Info Aplikasi")
    st.markdown("""
    Prediksi **customer churn** menggunakan model SVM yang dilatih dengan 10 fitur:

    | # | Fitur |
    |---|-------|
    | 1 | Tenure |
    | 2 | WarehouseToHome |
    | 3 | NumberOfDeviceRegistered |
    | 4 | PreferedOrderCat |
    | 5 | SatisfactionScore |
    | 6 | MaritalStatus |
    | 7 | NumberOfAddress |
    | 8 | Complain |
    | 9 | DaySinceLastOrder |
    | 10 | CashbackAmount |
    """)
    st.divider()
    if model is None:
        st.error("⚠️ Model tidak ditemukan!\n\nLetakkan `best_churn_model.pkl` di folder `models/`")
    else:
        st.success("✅ Model berhasil dimuat")
    st.divider()
    st.markdown("**Performa Model (Test Set)**")
    c1, c2 = st.columns(2)
    c1.metric("Recall",   "1.00")
    c2.metric("Presisi",  "0.15")
    c1.metric("F1-Score", "0.264")
    c2.metric("Fokus",    "Recall ↑")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Prediksi Tunggal",
    "📊 Prediksi Batch (CSV)",
    "📋 Format & Template CSV",
])

# ══════════════════════════════════════════════
# TAB 1 — PREDIKSI TUNGGAL
# ══════════════════════════════════════════════
with tab1:
    if model is None:
        st.markdown('<div class="err-box">❌ Model belum dimuat. Letakkan <code>best_churn_model.pkl</code> di folder <code>models/</code>.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<p class="section-title">Input Data Satu Pelanggan</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📦 Transaksi & Produk**")

        tenure = st.number_input(
            "Tenure — Lama berlangganan (bulan)",
            min_value=0.0, max_value=72.0, value=12.0, step=1.0,
            help="Berapa bulan pelanggan sudah bergabung di platform"
        )
        warehouse_to_home = st.number_input(
            "WarehouseToHome — Jarak gudang ke rumah (km)",
            min_value=0.0, max_value=200.0, value=20.0, step=1.0,
            help="Estimasi jarak pengiriman"
        )
        day_since_last = st.number_input(
            "DaySinceLastOrder — Hari sejak order terakhir",
            min_value=0.0, max_value=90.0, value=7.0, step=1.0,
            help="Semakin besar = semakin lama tidak order"
        )
        cashback = st.number_input(
            "CashbackAmount — Rata-rata cashback diterima",
            min_value=0.0, max_value=500.0, value=150.0, step=10.0,
        )
        pref_order_cat = st.selectbox(
            "PreferedOrderCat — Kategori produk favorit",
            PREF_ORDER_CATS
        )

    with col2:
        st.markdown("**👤 Profil & Kepuasan**")

        num_device = st.number_input(
            "NumberOfDeviceRegistered — Jumlah device terdaftar",
            min_value=1, max_value=10, value=3, step=1
        )
        satisfaction = st.slider(
            "SatisfactionScore — Skor kepuasan (1–5)",
            min_value=1, max_value=5, value=3,
            help="1 = sangat tidak puas · 5 = sangat puas"
        )
        marital_status = st.selectbox(
            "MaritalStatus — Status pernikahan",
            MARITAL_STATUSES
        )
        num_address = st.number_input(
            "NumberOfAddress — Jumlah alamat pengiriman",
            min_value=1, max_value=25, value=3, step=1
        )
        complain = st.radio(
            "Complain — Pernah komplain bulan ini?",
            options=[0, 1],
            format_func=lambda x: "✅ Tidak (0)" if x == 0 else "⚠️ Ya (1)",
            horizontal=True
        )

    st.divider()
    btn = st.button("Prediksi Sekarang", use_container_width=True, type="primary")

    if btn:
        row = {
            "Tenure":                   tenure,
            "WarehouseToHome":          warehouse_to_home,
            "NumberOfDeviceRegistered": float(num_device),
            "PreferedOrderCat":         pref_order_cat,
            "SatisfactionScore":        float(satisfaction),
            "MaritalStatus":            marital_status,
            "NumberOfAddress":          float(num_address),
            "Complain":                 float(complain),
            "DaySinceLastOrder":        day_since_last,
            "CashbackAmount":           cashback,
        }
        df_single = pd.DataFrame([row])

        preds, probas = predict(df_single)
        pred_label = int(preds[0])
        churn_prob = float(probas[0])

        st.markdown("### 🎯 Hasil Prediksi")
        r1, r2, r3 = st.columns(3)

        with r1:
            if pred_label == 1:
                st.markdown('<span class="badge-churn">🚨 CHURN</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-safe">✅ TIDAK CHURN</span>', unsafe_allow_html=True)
        with r2:
            st.metric("Probabilitas Churn", f"{churn_prob:.1%}")
        with r3:
            st.metric("Tingkat Risiko", risk_label(churn_prob))

        st.progress(min(churn_prob, 1.0), text=f"Churn Score: {churn_prob:.1%}")

        if pred_label == 1:
            st.markdown("""
            <div class="warn-box">
            ⚠️ <b>Rekomendasi:</b> Pelanggan ini berisiko tinggi churn.
            Pertimbangkan: kirim voucher/diskon personal, loyalty reward,
            atau lakukan follow-up langsung dari tim CRM.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            ℹ️ <b>Status Aman:</b> Pelanggan diprediksi akan tetap aktif.
            Tetap pantau secara berkala dan pertahankan kualitas layanan.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔎 Lihat Detail Input"):
            st.dataframe(
                df_single.T.rename(columns={0: "Nilai yang Diinput"}),
                use_container_width=True
            )


# ══════════════════════════════════════════════
# TAB 2 — PREDIKSI BATCH
# ══════════════════════════════════════════════
with tab2:
    if model is None:
        st.markdown('<div class="err-box">❌ Model belum dimuat.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<p class="section-title">Upload CSV untuk Prediksi Massal</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📁 Upload file CSV dengan <b>10 kolom fitur</b> sesuai format training.S
    Urutan kolom bebas — aplikasi mencocokkan berdasarkan <b>nama kolom</b>.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ File dimuat: **{len(df_raw):,} baris · {df_raw.shape[1]} kolom**")

            # Validasi kolom
            missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
            if missing:
                st.markdown(
                    '<div class="err-box">❌ Kolom tidak ditemukan: '
                    + ", ".join(f"<code>{c}</code>" for c in missing)
                    + "</div>",
                    unsafe_allow_html=True
                )
                st.stop()

            # Kolom extra (termasuk Churn) → abaikan
            extra = [c for c in df_raw.columns if c not in FEATURE_COLS]
            if extra:
                st.markdown(
                    '<div class="info-box">ℹ️ Kolom berikut diabaikan: '
                    + ", ".join(f"<code>{c}</code>" for c in extra)
                    + "</div>",
                    unsafe_allow_html=True
                )

            with st.expander("👀 Preview 5 baris pertama", expanded=True):
                st.dataframe(df_raw[FEATURE_COLS].head(), use_container_width=True)

            if st.button("Prediksi Batch", use_container_width=True, type="primary"):
                with st.spinner("Memproses prediksi untuk semua baris..."):
                    X_batch = df_raw[FEATURE_COLS].copy()
                    preds_b, probas_b = predict(X_batch)

                df_result = df_raw.copy()
                df_result["Churn_Prediction"]  = preds_b
                df_result["Churn_Probability"] = probas_b.round(4)
                df_result["Churn_Label"]       = np.where(preds_b == 1, "CHURN", "TIDAK CHURN")
                df_result["Risk_Level"]        = [risk_label(p) for p in probas_b]

                # ── Summary metrics ──
                n_total   = len(preds_b)
                n_churn   = int((preds_b == 1).sum())
                n_safe    = n_total - n_churn
                pct_churn = n_churn / n_total * 100

                st.markdown("---")
                st.markdown("### 📊 Ringkasan Hasil Prediksi")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Pelanggan",  f"{n_total:,}")
                m2.metric("Diprediksi Churn", f"{n_churn:,}",
                          delta=f"{pct_churn:.1f}%", delta_color="inverse")
                m3.metric("Tidak Churn",      f"{n_safe:,}")
                m4.metric("Avg Churn Score",  f"{probas_b.mean():.1%}")

                # Distribusi risiko
                st.markdown("**Distribusi Tingkat Risiko:**")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("🔴 Risiko Tinggi", int((probas_b >= 0.7).sum()))
                rc2.metric("🟡 Risiko Sedang", int(((probas_b >= 0.4) & (probas_b < 0.7)).sum()))
                rc3.metric("🟢 Risiko Rendah", int((probas_b < 0.4).sum()))

                # ── Top 10 churners ──
                st.markdown("### 🚨 Top 10 Pelanggan Risiko Churn Tertinggi")
                top10 = (
                    df_result[df_result["Churn_Prediction"] == 1]
                    .sort_values("Churn_Probability", ascending=False)
                    .head(10)
                )
                if len(top10):
                    show_cols = ["Churn_Probability", "Risk_Level"] + FEATURE_COLS
                    st.dataframe(
                        top10[show_cols]
                        .reset_index(drop=True)
                        .style.background_gradient(subset=["Churn_Probability"], cmap="Reds"),
                        use_container_width=True
                    )
                else:
                    st.info("Tidak ada pelanggan yang diprediksi churn.")

                # ── Full table ──
                with st.expander("📋 Lihat Semua Hasil Prediksi"):
                    st.dataframe(df_result.reset_index(drop=True), use_container_width=True)

                # ── Download ──
                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_churn.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.markdown(f'<div class="err-box">❌ Error saat memproses file: {e}</div>',
                        unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="warn-box">
        📌 Belum ada file yang diupload.
        Gunakan tab <b>Format & Template CSV</b> untuk mendownload template siap pakai.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — FORMAT GUIDE + TEMPLATE DOWNLOAD
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Format Kolom CSV (10 Fitur Wajib)</p>', unsafe_allow_html=True)

    guide = pd.DataFrame({
        "No": range(1, 11),
        "Nama Kolom": FEATURE_COLS,
        "Tipe": [
            "float", "float", "int",
            "object", "int (1–5)",
            "object", "int",
            "int (0/1)", "float", "float"
        ],
        "Contoh": [
            "12.0", "20.0", "3",
            "Laptop & Accessory", "3",
            "Single", "3",
            "0 atau 1", "7.0", "150.0"
        ],
        "Keterangan / Nilai yang Valid": [
            "Lama berlangganan dalam bulan",
            "Jarak gudang ke rumah (km)",
            "Jumlah perangkat terdaftar di akun",
            "Laptop & Accessory | Mobile Phone | Fashion | Grocery | Others",
            "Skala 1 (sangat tidak puas) sampai 5 (sangat puas)",
            "Single | Divorced | Married",
            "Jumlah alamat pengiriman terdaftar",
            "0 = tidak pernah komplain · 1 = pernah komplain",
            "Hari sejak terakhir kali melakukan order",
            "Rata-rata nominal cashback yang diterima",
        ]
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📥 Download Template CSV Siap Pakai")
    st.markdown("Berisi **5 baris contoh data**. Hapus atau ganti dengan data pelanggan asli.")

    template = pd.DataFrame({
        "Tenure":                   [12.0, 3.0, 24.0, 1.0, 18.0],
        "WarehouseToHome":          [20.0, 45.0, 10.0, 70.0, 30.0],
        "NumberOfDeviceRegistered": [3, 2, 4, 1, 3],
        "PreferedOrderCat":         ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
        "SatisfactionScore":        [4, 1, 5, 2, 3],
        "MaritalStatus":            ["Single", "Married", "Divorced", "Single", "Married"],
        "NumberOfAddress":          [3, 5, 2, 8, 3],
        "Complain":                 [0, 1, 0, 1, 0],
        "DaySinceLastOrder":        [7.0, 30.0, 3.0, 45.0, 5.0],
        "CashbackAmount":           [150.0, 80.0, 220.0, 50.0, 175.0],
    })
    st.dataframe(template, use_container_width=True, hide_index=True)

    csv_tmpl = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download template_churn_prediction.csv",
        data=csv_tmpl,
        file_name="template_churn_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )