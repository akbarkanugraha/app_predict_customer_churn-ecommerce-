import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

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
# Load Model + Metadata
# ─────────────────────────────────────────────
MODEL_PATHS    = ["models/best_churn_model.pkl", "best_churn_model.pkl"]
METADATA_PATHS = ["models/model_metadata.json",  "model_metadata.json"]

@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

@st.cache_data
def load_metadata():
    for path in METADATA_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

model, model_path = load_model()
metadata = load_metadata()

# ─────────────────────────────────────────────
# Ekstrak Info Model Secara Dinamis
# ─────────────────────────────────────────────
def get_model_info(mdl, meta: dict):
    """
    Ekstrak info model dari metadata JSON (diutamakan) atau
    dari introspeksi pipeline sebagai fallback.

    Metadata JSON (disimpan notebook):
        model_name      : nama kelas classifier
        pipeline_steps  : list nama step
        feature_cols    : list nama fitur input
        metrics         : dict {recall, precision, f1, fbeta, accuracy}
        categorical_values : dict {kolom: [nilai_valid]}
    """
    info = {
        "model_name":          "Unknown",
        "pipeline_desc":       "",
        "feature_cols":        None,
        "metrics":             {},
        "categorical_values":  {},
    }

    # ── Utama: baca dari metadata JSON ──
    if meta:
        info["model_name"]         = meta.get("model_name", info["model_name"])
        info["pipeline_desc"]      = meta.get("pipeline_desc", "")
        info["feature_cols"]       = meta.get("feature_cols")
        info["metrics"]            = meta.get("metrics", {})
        info["categorical_values"] = meta.get("categorical_values", {})
        return info

    # ── Fallback: introspeksi pipeline ──
    if mdl is None:
        return info

    # Nama classifier
    clf = mdl.named_steps.get("classifier")
    if clf is not None:
        info["model_name"] = type(clf).__name__

    # Deskripsi pipeline
    info["pipeline_desc"] = " · ".join(type(s).__name__ for _, s in mdl.steps)

    # Kolom fitur dari ColumnTransformer
    pre = mdl.named_steps.get("transform") or mdl.named_steps.get("preprocessor")
    if pre is not None:
        cols = []
        for _, _, c in pre.transformers_:
            if isinstance(c, list):
                cols.extend(c)
        if cols:
            info["feature_cols"] = cols

    return info


_info         = get_model_info(model, metadata)
MODEL_NAME    = _info["model_name"]
PIPELINE_DESC = _info["pipeline_desc"]
METRICS       = _info["metrics"]

# ── Kolom fitur (dari metadata atau fallback hardcode sesuai notebook) ──
# Urutan harus sesuai dengan X = df_clean.drop('Churn')
_DEFAULT_FEATURE_COLS = [
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
FEATURE_COLS = _info["feature_cols"] or _DEFAULT_FEATURE_COLS

# ── Nilai kategorikal valid (dari metadata atau fallback) ──
_DEFAULT_CAT_VALUES = {
    "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
    "MaritalStatus":    ["Single", "Divorced", "Married"],
}
_cat_from_meta = _info["categorical_values"]
PREF_ORDER_CATS  = _cat_from_meta.get("PreferedOrderCat", _DEFAULT_CAT_VALUES["PreferedOrderCat"])
MARITAL_STATUSES = _cat_from_meta.get("MaritalStatus",    _DEFAULT_CAT_VALUES["MaritalStatus"])

# ─────────────────────────────────────────────
# Predict Helper
# ─────────────────────────────────────────────
def predict(df: pd.DataFrame):
    """
    Return (labels ndarray, probabilities ndarray).
    Mendukung semua classifier dalam pipeline:
      - predict_proba  → dipakai langsung
      - decision_function → sigmoid transform
      - fallback         → label sebagai probabilitas
    """
    preds = model.predict(df)
    clf   = model.named_steps.get("classifier")

    if clf is not None and hasattr(clf, "predict_proba"):
        probas = model.predict_proba(df)[:, 1]
    elif clf is not None and hasattr(clf, "decision_function"):
        scores = model.decision_function(df)
        probas = 1 / (1 + np.exp(-scores))
    else:
        probas = preds.astype(float)

    return preds, probas


def risk_label(p: float) -> str:
    if p >= 0.7: return "🔴 Tinggi"
    if p >= 0.4: return "🟡 Sedang"
    return "🟢 Rendah"

# ─────────────────────────────────────────────
# Hero Header — nama model dari metadata/pipeline
# ─────────────────────────────────────────────
pipeline_line = f"Pipeline: {PIPELINE_DESC}" if PIPELINE_DESC else ""
st.markdown(f"""
<div class="hero-card">
    <h1>🛒 Customer Churn Predictor</h1>
    <p>E-Commerce · Capstone ML Project · Model: <b>{MODEL_NAME}</b>
    {"&nbsp;|&nbsp;" + pipeline_line if pipeline_line else ""}
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Info Aplikasi")

    # Tabel fitur dinamis
    feature_table_rows = "\n".join(
        f"    | {i} | {col} |" for i, col in enumerate(FEATURE_COLS, 1)
    )
    st.markdown(f"""
Prediksi **customer churn** menggunakan model **{MODEL_NAME}** yang dilatih dengan {len(FEATURE_COLS)} fitur:

| # | Fitur |
|---|-------|
{feature_table_rows}
""")

    st.divider()

    if model is None:
        st.error("⚠️ Model tidak ditemukan!\n\nLetakkan `best_churn_model.pkl` di folder `models/`")
    else:
        src = "metadata" if metadata else "introspeksi pipeline"
        st.success(f"✅ Model berhasil dimuat  \n`{MODEL_NAME}` _(dari {src})_")

    st.divider()
    st.markdown("**Performa Model (Test Set)**")

    if METRICS:
        # Tampilkan metrik dari metadata JSON (otomatis dari notebook)
        metric_keys = [
            ("Recall",    "recall"),
            ("Precision", "precision"),
            ("F1-Score",  "f1"),
            ("FBeta β=5", "fbeta"),
            ("Accuracy",  "accuracy"),
        ]
        cols_per_row = 2
        keys_to_show = [(label, key) for label, key in metric_keys if key in METRICS]
        for i in range(0, len(keys_to_show), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, (label, key) in enumerate(keys_to_show[i:i + cols_per_row]):
                val = METRICS[key]
                row_cols[j].metric(label, f"{val:.3f}" if isinstance(val, float) else str(val))
    else:
        # Fallback: metrik hardcode (akan hilang setelah metadata tersedia)
        c1, c2 = st.columns(2)
        c1.metric("Recall",   "—")
        c2.metric("Presisi",  "—")
        c1.metric("F1-Score", "—")
        c2.metric("Fokus",    "Recall ↑")
        st.caption("💡 Tambahkan kode penyimpan metadata di notebook untuk menampilkan metrik aktual.")

    if metadata.get("saved_at"):
        st.caption(f"🕒 Disimpan: {metadata['saved_at']}")

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
        # Pastikan urutan kolom sesuai FEATURE_COLS
        df_single = pd.DataFrame([{k: row[k] for k in FEATURE_COLS}])

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
    st.markdown(f"""
    <div class="info-box">
    📁 Upload file CSV dengan <b>{len(FEATURE_COLS)} kolom fitur</b> sesuai format training.
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

            # Kolom extra → abaikan
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
    st.markdown('<p class="section-title">Format Kolom CSV (Fitur Wajib)</p>', unsafe_allow_html=True)

    # Tipe & keterangan kolom — dinamis dari FEATURE_COLS, meta jika ada
    col_meta = metadata.get("column_info", {})

    _default_col_meta = {
        "Tenure":                   ("float",       "12.0",               "Lama berlangganan dalam bulan"),
        "WarehouseToHome":          ("float",       "20.0",               "Jarak gudang ke rumah (km)"),
        "NumberOfDeviceRegistered": ("int",         "3",                  "Jumlah perangkat terdaftar di akun"),
        "PreferedOrderCat":         ("object",      "Laptop & Accessory", "Laptop & Accessory | Mobile Phone | Fashion | Grocery | Others"),
        "SatisfactionScore":        ("int (1–5)",   "3",                  "Skala 1 (sangat tidak puas) sampai 5 (sangat puas)"),
        "MaritalStatus":            ("object",      "Single",             "Single | Divorced | Married"),
        "NumberOfAddress":          ("int",         "3",                  "Jumlah alamat pengiriman terdaftar"),
        "Complain":                 ("int (0/1)",   "0 atau 1",           "0 = tidak pernah komplain · 1 = pernah komplain"),
        "DaySinceLastOrder":        ("float",       "7.0",                "Hari sejak terakhir kali melakukan order"),
        "CashbackAmount":           ("float",       "150.0",              "Rata-rata nominal cashback yang diterima"),
    }

    guide_rows = []
    for i, col in enumerate(FEATURE_COLS, 1):
        if col in col_meta:
            tipe     = col_meta[col].get("dtype", "—")
            contoh   = str(col_meta[col].get("example", "—"))
            ket      = col_meta[col].get("description", "—")
        elif col in _default_col_meta:
            tipe, contoh, ket = _default_col_meta[col]
        else:
            tipe, contoh, ket = ("—", "—", "—")
        guide_rows.append({"No": i, "Nama Kolom": col, "Tipe": tipe,
                           "Contoh": contoh, "Keterangan / Nilai yang Valid": ket})

    guide = pd.DataFrame(guide_rows)
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📥 Download Template CSV Siap Pakai")
    st.markdown("Berisi **5 baris contoh data**. Hapus atau ganti dengan data pelanggan asli.")

    template = pd.DataFrame({
        "Tenure":                   [12.0, 3.0, 24.0, 1.0, 18.0],
        "WarehouseToHome":          [20.0, 45.0, 10.0, 70.0, 30.0],
        "NumberOfDeviceRegistered": [3, 2, 4, 1, 3],
        "PreferedOrderCat":         [PREF_ORDER_CATS[0], PREF_ORDER_CATS[1 % len(PREF_ORDER_CATS)],
                                     PREF_ORDER_CATS[2 % len(PREF_ORDER_CATS)], PREF_ORDER_CATS[3 % len(PREF_ORDER_CATS)],
                                     PREF_ORDER_CATS[-1]],
        "SatisfactionScore":        [4, 1, 5, 2, 3],
        "MaritalStatus":            [MARITAL_STATUSES[0], MARITAL_STATUSES[1 % len(MARITAL_STATUSES)],
                                     MARITAL_STATUSES[2 % len(MARITAL_STATUSES)], MARITAL_STATUSES[0],
                                     MARITAL_STATUSES[1 % len(MARITAL_STATUSES)]],
        "NumberOfAddress":          [3, 5, 2, 8, 3],
        "Complain":                 [0, 1, 0, 1, 0],
        "DaySinceLastOrder":        [7.0, 30.0, 3.0, 45.0, 5.0],
        "CashbackAmount":           [150.0, 80.0, 220.0, 50.0, 175.0],
    })[FEATURE_COLS]   # urutan sesuai FEATURE_COLS

    st.dataframe(template, use_container_width=True, hide_index=True)

    csv_tmpl = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download template_churn_prediction.csv",
        data=csv_tmpl,
        file_name="template_churn_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )