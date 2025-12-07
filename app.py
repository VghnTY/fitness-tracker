import streamlit as st
import pandas as pd
import joblib
import time

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Fitness Tracker",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Background sidebar lebih lembut */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Card Utama (Kalori) */
    .metric-card-main {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5253 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        box-shadow: 0 4px 15px rgba(238, 82, 83, 0.3);
        margin-bottom: 20px;
    }
    .metric-card-main h3 { color: rgba(255,255,255,0.9); margin: 0; font-size: 16px; }
    .metric-card-main h1 { color: white; margin: 10px 0; font-size: 48px; font-weight: 800; }
    
    /* Card Info (Putih) */
    .metric-card-info {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #eee;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        height: 100%;
    }
    .metric-card-info h4 { color: #888; margin: 0; font-size: 14px; text-transform: uppercase; }
    .metric-card-info h2 { color: #333; margin: 5px 0; font-size: 28px; }
    
    /* Status Badge BMI */
    .badge {
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-normal { background-color: #e6fffa; color: #00b894; }
    .badge-warning { background-color: #fffbe6; color: #fdcb6e; }
    .badge-danger { background-color: #fff5f5; color: #d63031; }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INISIALISASI STATE
# ==========================================
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'bmi' not in st.session_state:
    st.session_state['bmi'] = 0
if 'bmi_status' not in st.session_state:
    st.session_state['bmi_status'] = "-"
if 'food' not in st.session_state:
    st.session_state['food'] = "-"

# ==========================================
# 3. LOAD ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('final_model.joblib')
        scaler = joblib.load('final_scaler.joblib')
        cols = joblib.load('final_columns.joblib')
        return model, scaler, cols
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_artifacts()

# ==========================================
# 4. FUNGSI HELPER
# ==========================================
def calculate_bmi(w, h):
    h_meter = h / 100
    bmi = w / (h_meter ** 2)
    if bmi < 18.5: return bmi, "Underweight", "badge-warning"
    elif bmi < 24.9: return bmi, "Normal", "badge-normal"
    elif bmi < 29.9: return bmi, "Overweight", "badge-warning"
    else: return bmi, "Obesity", "badge-danger"

def get_food_equivalent(calories):
    foods = [
        (100, "🍎 1 Buah Apel Besar"), (200, "🥛 1 Gelas Susu"),
        (300, "🍕 1 Potong Pizza"), (400, "🍜 1 Mangkok Mie Instan"),
        (500, "🍔 1 Burger"), (700, "🍛 1 Porsi Nasi Padang")
    ]
    closest = min(foods, key=lambda x: abs(x[0] - calories))
    return closest[1]

# ==========================================
# 5. SIDEBAR (INPUT)
# ==========================================
with st.sidebar:
    st.title("⚙️ Parameter")
    
    # PERUBAHAN DISINI: Menambahkan step=1.0 agar naik/turun per 1 angka
    with st.expander("👤 Profil Fisik", expanded=True):
        weight = st.number_input("Berat (kg)", 30.0, 150.0, 70.0, step=1.0)
        height = st.number_input("Tinggi (cm)", 120.0, 220.0, 170.0, step=1.0)
    
    with st.expander("❤️ Kondisi & Aktivitas", expanded=True):
        activity = st.selectbox("Jenis Olahraga", ['Cycling', 'Running', 'Walking', 'Workout', 'Yoga'])
        heart_rate = st.slider("Detak Jantung (bpm)", 60, 200, 120)
        duration = st.slider("Durasi (menit)", 10, 180, 45, step=5)
        met = st.slider("Intensitas (MET)", 1.0, 15.0, 5.0, help="Semakin tinggi = semakin berat latihan")

    btn_predict = st.button("🔥 Hasil Analisis", type="primary", use_container_width=True)

# ==========================================
# 6. HALAMAN UTAMA
# ==========================================
st.title("🧘 Fitness Tracker Dashboard")
st.write("Analisis pembakaran kalori berbasis **Machine Learning**")

if model is None:
    st.error("⚠️ File model tidak ditemukan! Harap jalankan training.py terlebih dahulu.")
else:
    # --- LOGIKA PREDIKSI ---
    if btn_predict:
        with st.spinner('Sedang menghitung...'):
            time.sleep(0.5) # Efek loading
            
            # 1. Hitung BMI
            bmi, status, badge = calculate_bmi(weight, height)
            
            # 2. Prediksi Kalori
            input_df = pd.DataFrame([{
                'Duration_min': duration, 'HeartRate_bpm': heart_rate,
                'Weight_kg': weight, 'Height_cm': height, 'MET': met
            }])
            input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            
            final_input = {col: 0 for col in model_columns}
            for col in input_df.columns: final_input[col] = input_scaled.iloc[0][col]
            if f"Activity_{activity}" in final_input: final_input[f"Activity_{activity}"] = 1
            
            cal = model.predict(pd.DataFrame([final_input]))[0]
            
            # 3. Simpan ke Session State
            st.session_state['result'] = cal
            st.session_state['bmi'] = bmi
            st.session_state['bmi_status'] = status
            st.session_state['bmi_badge'] = badge
            st.session_state['food'] = get_food_equivalent(cal)
            st.session_state['inputs'] = {'dur': duration, 'act': activity}

    # --- TAMPILAN HASIL ---
    if st.session_state['result'] is not None:
        res = st.session_state['result']
        bmi_val = st.session_state['bmi']
        bmi_stat = st.session_state['bmi_status']
        food_eq = st.session_state['food']
        
        st.divider()
        
        # BARIS 1: KARTU UTAMA
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-main">
                <h3>ESTIMASI KALORI TERBAKAR</h3>
                <h1>{res:.0f} <span style="font-size:24px">kkal</span></h1>
                <p>Setara dengan: <b>{food_eq}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card-info">
                <h4>BODY MASS INDEX (BMI)</h4>
                <h2>{bmi_val:.1f}</h2>
                <span class="badge {st.session_state.get('bmi_badge', 'badge-normal')}">{bmi_stat}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card-info">
                <h4>Durasi Latihan</h4>
                <h2>{st.session_state['inputs']['dur']} <span style="font-size:16px">mnt</span></h2>
                <p style="color:#888; margin:0;">{st.session_state['inputs']['act']}</p>
            </div>
            """, unsafe_allow_html=True)

        # BARIS 2: PENJELASAN BMI & TIPS
        col_info, col_tips = st.columns([1.5, 1])
        
        with col_info:
            st.subheader("📌 Apa itu BMI?")
            st.info(f"""
            **Body Mass Index (BMI)** adalah perkiraan lemak tubuh berdasarkan tinggi dan berat badan.
            Nilai BMI Anda saat ini adalah **{bmi_val:.1f}** ({bmi_stat}).
            """)
            
            # Progress bar untuk visualisasi posisi BMI
            bmi_progress = min(max((bmi_val - 10) / 30, 0.0), 1.0)
            st.write("Posisi BMI Anda pada skala umum:")
            st.progress(bmi_progress)
            st.caption("Skala: Kiri (Sangat Kurus) --- Kanan (Obesitas)")

        with col_tips:
            st.subheader("💡 Saran Kesehatan")
            if bmi_stat == "Normal":
                st.success("Berat badan Anda ideal! Pertahankan pola latihan ini untuk menjaga kesehatan jantung dan stamina.")
            elif bmi_stat == "Underweight":
                st.warning("Anda termasuk kurus. Fokus pada peningkatan asupan nutrisi dan latihan beban untuk membangun massa otot.")
            else:
                st.warning("BMI menunjukkan indikasi berat berlebih. Latihan kardio rutin seperti ini sangat bagus untuk membakar lemak.")

    else:
        # Tampilan Awal Kosong
        st.info("👈 Masukkan data diri dan latihan Anda di panel sebelah kiri, lalu klik tombol **Hasil Analisis**.")