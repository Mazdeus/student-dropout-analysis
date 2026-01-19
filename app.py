import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan preprocessing objects
model = joblib.load("model/rdf_model.joblib")
scaler = joblib.load("model/scaler.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# Daftar fitur yang digunakan (sesuai urutan saat training)
selected_features = [
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_approved', 
    'Curricular_units_2nd_sem_grade',
    'Tuition_fees_up_to_date',
    'Debtor',
    'Age_at_enrollment',
]

# ===================== HEADER =====================
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=130)
with col2:
    st.title("🎓 Student Dropout Prediction")
    st.write("Jaya Jaya Institut - Early Warning System")

st.markdown("---")

# ===================== DESCRIPTION =====================
st.markdown("""
### Tentang Aplikasi
Aplikasi ini digunakan untuk memprediksi status siswa berdasarkan data akademik dan finansial.
Hasil prediksi dapat membantu institusi dalam mengidentifikasi siswa yang berisiko **Dropout** 
sehingga dapat dilakukan intervensi dini.
""")

st.markdown("---")

# ===================== INPUT FORM =====================
st.subheader("Masukkan Data Siswa")

# Buat DataFrame untuk menampung input
data = pd.DataFrame()

# Row 1: Faktor Akademik - Semester 1
st.markdown("#### Performa Akademik Semester 1")
col1, col2 = st.columns(2)

with col1:
    units_approved_sem1 = st.number_input(
        label="Jumlah Mata Kuliah Lulus Semester 1",
        min_value=0,
        max_value=30,
        value=5,
        help="Jumlah mata kuliah yang berhasil diluluskan di semester 1"
    )
    data["Curricular_units_1st_sem_approved"] = [units_approved_sem1]

with col2:
    grade_sem1 = st.number_input(
        label="Rata-rata Nilai Semester 1",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1,
        help="Rata-rata nilai semester 1 (skala 0-20)"
    )
    data["Curricular_units_1st_sem_grade"] = [grade_sem1]

# Row 2: Faktor Akademik - Semester 2
st.markdown("#### Performa Akademik Semester 2")
col1, col2 = st.columns(2)

with col1:
    units_approved_sem2 = st.number_input(
        label="Jumlah Mata Kuliah Lulus Semester 2",
        min_value=0,
        max_value=30,
        value=5,
        help="Jumlah mata kuliah yang berhasil diluluskan di semester 2"
    )
    data["Curricular_units_2nd_sem_approved"] = [units_approved_sem2]

with col2:
    grade_sem2 = st.number_input(
        label="Rata-rata Nilai Semester 2",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1,
        help="Rata-rata nilai semester 2 (skala 0-20)"
    )
    data["Curricular_units_2nd_sem_grade"] = [grade_sem2]

# Row 3: Faktor Finansial
st.markdown("#### Faktor Finansial")
col1, col2 = st.columns(2)

with col1:
    tuition_fees = st.selectbox(
        label="Pembayaran Biaya Kuliah Tepat Waktu?",
        options=[1, 0],
        format_func=lambda x: "Ya" if x == 1 else "Tidak",
        help="Apakah siswa membayar biaya kuliah tepat waktu?"
    )
    data["Tuition_fees_up_to_date"] = [tuition_fees]

with col2:
    debtor = st.selectbox(
        label="Status Hutang",
        options=[0, 1],
        format_func=lambda x: "Tidak Punya Hutang" if x == 0 else "Memiliki Hutang",
        help="Apakah siswa memiliki hutang?"
    )
    data["Debtor"] = [debtor]

# Row 4: Faktor Demografi
st.markdown("#### Faktor Demografi")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        label="Usia Saat Mendaftar",
        min_value=17,
        max_value=70,
        value=20,
        help="Usia siswa saat pertama kali mendaftar"
    )
    data["Age_at_enrollment"] = [age]

st.markdown("---")

# ===================== VIEW RAW DATA =====================
with st.expander("Lihat Data Input"):
    st.dataframe(data, use_container_width=True)

# ===================== PREDICTION =====================
st.subheader("🔮 Hasil Prediksi")

if st.button("Prediksi Status Siswa", type="primary"):
    # Preprocessing: pastikan urutan kolom sesuai
    data_ordered = data[selected_features]
    
    # Scaling
    data_scaled = scaler.transform(data_ordered)
    
    # Prediksi
    prediction = model.predict(data_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    
    # Probabilitas (jika model mendukung)
    try:
        prediction_proba = model.predict_proba(data_scaled)[0]
        proba_dict = dict(zip(label_encoder.classes_, prediction_proba))
    except:
        proba_dict = None
    
    # Tampilkan hasil
    st.markdown("---")
    
    # Warna berdasarkan hasil
    if prediction_label == "Dropout":
        st.error(f"### Prediksi: **{prediction_label}**")
        st.warning("""
        **Rekomendasi Tindakan:**
        - Hubungi siswa untuk konseling akademik
        - Tawarkan program bimbingan belajar
        - Evaluasi kondisi finansial siswa
        - Pertimbangkan pemberian beasiswa atau keringanan biaya
        """)
    elif prediction_label == "Graduate":
        st.success(f"### Prediksi: **{prediction_label}**")
        st.info("""
        **Insight:**
        - Siswa diprediksi akan menyelesaikan studi dengan baik
        - Tetap monitor perkembangan akademik secara berkala
        """)
    else:  # Enrolled
        st.info(f"### Prediksi: **{prediction_label}**")
        st.info("""
        **Insight:**
        - Siswa masih dalam proses studi
        - Perlu pemantauan lebih lanjut untuk menentukan status akhir
        """)
    
    # Tampilkan probabilitas jika ada
    if proba_dict:
        st.markdown("#### Probabilitas Prediksi:")
        prob_df = pd.DataFrame({
            'Status': proba_dict.keys(),
            'Probabilitas': [f"{p*100:.2f}%" for p in proba_dict.values()]
        })
        st.dataframe(prob_df, use_container_width=True)
    
    # Tampilkan data yang sudah di-scale
    with st.expander("Lihat Data Setelah Preprocessing"):
        scaled_df = pd.DataFrame(data_scaled, columns=selected_features)
        st.dataframe(scaled_df, use_container_width=True)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 Student Dropout Prediction - Jaya Jaya Institut</p>
    <p>Dibuat dengan Streamlit | Model: Random Forest</p>
</div>
""", unsafe_allow_html=True)

