# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

Proyek ini bertujuan untuk menganalisis data siswa, mengidentifikasi faktor-faktor yang mempengaruhi dropout, serta membangun model machine learning untuk memprediksi status siswa (Dropout, Enrolled, atau Graduate). Selain itu, akan dibuat business dashboard untuk membantu institusi dalam memonitor performa siswa.

### Permasalahan Bisnis

1. Tingginya tingkat dropout siswa yang berdampak negatif pada reputasi dan keberlanjutan institusi.
2. Ketidakmampuan untuk mendeteksi secara dini siswa yang berpotensi dropout.
3. Kurangnya pemahaman terhadap faktor-faktor yang mempengaruhi keputusan siswa untuk dropout.
4. Tidak adanya sistem monitoring yang memadai untuk memantau performa dan risiko dropout siswa.

### Cakupan Proyek

Cakupan proyek pada studi kasus ini dibatasi pada:
- Melakukan analisis data siswa untuk memahami karakteristik dan pola dropout.
- Mengidentifikasi faktor-faktor yang berhubungan dengan dropout:
    - **Faktor Demografi**: Age at enrollment, Gender, Marital Status, Nationality
    - **Faktor Akademik**: Previous qualification, Admission grade, Course, Application mode/order
    - **Faktor Performa Akademik**: Curricular units (credited, enrolled, evaluations, approved, grade) pada semester 1 dan 2
    - **Faktor Sosial-Ekonomi**: Mother's/Father's qualification & occupation, Scholarship holder, Tuition fees up to date, Debtor
    - **Faktor Ekonomi Makro**: Unemployment rate, Inflation rate, GDP
- Membangun model machine learning untuk memprediksi status siswa (Dropout, Enrolled, Graduate).
- Membuat business dashboard yang menampilkan metrik dan visualisasi utama terkait performa siswa.
- Membuat prototype sistem machine learning menggunakan Streamlit.
- Memberikan kesimpulan dan rekomendasi action items untuk membantu institusi dalam pengambilan keputusan.

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

Dataset ini berisi data siswa dari institusi pendidikan tinggi yang terdaftar di berbagai program sarjana. Dataset mencakup informasi yang diketahui pada saat pendaftaran siswa (jalur akademik, demografi, dan faktor sosial-ekonomi) serta performa akademik siswa pada akhir semester pertama dan kedua.

**Deskripsi Variabel:**

| Variabel | Deskripsi |
|----------|-----------|
| Marital_status | Status pernikahan (1-single, 2-married, 3-widower, 4-divorced, 5-facto union, 6-legally separated) |
| Application_mode | Mode pendaftaran |
| Application_order | Urutan pilihan pendaftaran (0-pilihan pertama sampai 9-pilihan terakhir) |
| Course | Program studi yang diambil |
| Daytime_evening_attendance | Waktu kuliah (1-siang, 0-malam) |
| Previous_qualification | Kualifikasi pendidikan sebelumnya |
| Previous_qualification_grade | Nilai kualifikasi sebelumnya (0-200) |
| Nacionality | Kewarganegaraan |
| Mothers_qualification | Kualifikasi pendidikan ibu |
| Fathers_qualification | Kualifikasi pendidikan ayah |
| Mothers_occupation | Pekerjaan ibu |
| Fathers_occupation | Pekerjaan ayah |
| Admission_grade | Nilai masuk (0-200) |
| Displaced | Apakah siswa pindahan (1-ya, 0-tidak) |
| Educational_special_needs | Kebutuhan pendidikan khusus (1-ya, 0-tidak) |
| Debtor | Status hutang (1-ya, 0-tidak) |
| Tuition_fees_up_to_date | Biaya kuliah terbayar tepat waktu (1-ya, 0-tidak) |
| Gender | Jenis kelamin (1-pria, 0-wanita) |
| Scholarship_holder | Penerima beasiswa (1-ya, 0-tidak) |
| Age_at_enrollment | Usia saat mendaftar |
| International | Siswa internasional (1-ya, 0-tidak) |
| Curricular_units_1st_sem_credited | Jumlah unit kurikuler yang dikreditkan di semester 1 |
| Curricular_units_1st_sem_enrolled | Jumlah unit kurikuler yang terdaftar di semester 1 |
| Curricular_units_1st_sem_evaluations | Jumlah evaluasi unit kurikuler di semester 1 |
| Curricular_units_1st_sem_approved | Jumlah unit kurikuler yang lulus di semester 1 |
| Curricular_units_1st_sem_grade | Rata-rata nilai di semester 1 (0-20) |
| Curricular_units_1st_sem_without_evaluations | Jumlah unit kurikuler tanpa evaluasi di semester 1 |
| Curricular_units_2nd_sem_credited | Jumlah unit kurikuler yang dikreditkan di semester 2 |
| Curricular_units_2nd_sem_enrolled | Jumlah unit kurikuler yang terdaftar di semester 2 |
| Curricular_units_2nd_sem_evaluations | Jumlah evaluasi unit kurikuler di semester 2 |
| Curricular_units_2nd_sem_approved | Jumlah unit kurikuler yang lulus di semester 2 |
| Curricular_units_2nd_sem_grade | Rata-rata nilai di semester 2 (0-20) |
| Curricular_units_2nd_sem_without_evaluations | Jumlah unit kurikuler tanpa evaluasi di semester 2 |
| Unemployment_rate | Tingkat pengangguran (%) |
| Inflation_rate | Tingkat inflasi (%) |
| GDP | Produk Domestik Bruto |
| Status | Target variable - status siswa (Dropout, Enrolled, Graduate) |

**Setup environment:**

**Opsi 1: Menggunakan Anaconda**
```bash
conda create --name student-performance python=3.9
conda activate student-performance
pip install -r requirements.txt
```

**Opsi 2: Menggunakan Shell/Terminal**
```bash
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
pip install notebook
jupyter-notebook .
```

### Metabase dengan Docker

Langkah-langkah:

1. Install Docker
2. Jalankan perintah berikut pada Terminal/Command Prompt/PowerShell guna memanggil (pull) Docker image untuk menjalankan Metabase.
    ```
    docker pull metabase/metabase:v0.46.4
    ```
3. Selanjutnya,
    ```
    docker run -p 3000:3000 --name metabase metabase/metabase:v0.46.4
    ```
    atau:
    ```
    docker run -p 3000:3000 --name metabase metabase/metabase
    ```
4. Masuk ke metabase dengan URL berikut : http://localhost:3000/setup
    Gunakan akun login berikut ini:
    ```
    Email : root@gmail.com
    Password : root123
    ```

### Mengirim Dataset ke Database Supabase 

1. Siapkan supabase, pastikan sudah login dan membuat Project
2. Untuk mengirim dataset ke database, ke bagian menu "Project Overview", lalu scroll kebawah, akan ada "connect", Ubah dulu Methodnya jadi Transaction Pooler, lalu salin DATABASE_URL nya.
3. Setelah itu, gunakan perintah berikut ini:
    ```
    from sqlalchemy import create_engine
 
    URL = "DATABASE_URL"
 
    engine = create_engine(URL)
    df.to_sql('orders', engine)

    ```  

### Hubungkan Metabase dengan Databse

1. Pastikan sudah masuk Admin setting pada metabase
2. Hubungkan dengan isi field seperti dibawah:
    ```
    Connection string (optional) : postgresql://postgres.kwvzemdwsatjahcwrogm:AndikaFadhil@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres
    Display name : Attrition-Analysis
    ```
    
    Ketika connection string diisi, otomatis semua field dibawahnya terisi

## Business Dashboard

Business dashboard yang dibuat bertujuan untuk membantu Jaya Jaya Institut dalam memantau tingkat dropout siswa serta memahami faktor-faktor yang memengaruhinya. Dashboard ini disusun ke dalam beberapa bagian agar mudah dipahami dan dianalisis oleh pihak manajemen institusi.

![Students Dashboard](Mohammad_Amadeus_Andika_Fadhil-dashboard.png)

### Ringkasan Statistik Siswa

Bagian atas dashboard menampilkan KPI cards yang memberikan gambaran umum:
- **Total Siswa**: 4,424 siswa
- **Jumlah Dropout**: 1,421 siswa
- **Jumlah Graduate**: 2,209 siswa

### Distribusi Status Siswa (Dropout Rate)

Visualisasi donut chart menunjukkan distribusi status siswa:
- **Graduate**: 50% (2,209 siswa)
- **Dropout**: 32% (1,421 siswa)
- **Enrolled**: 18% (794 siswa)

Dengan tingkat dropout sebesar **32%**, ini menunjukkan bahwa hampir sepertiga siswa tidak menyelesaikan pendidikannya, yang merupakan angka yang cukup tinggi dan perlu mendapat perhatian serius dari institusi.

### Pengaruh Faktor Finansial terhadap Status Siswa

Dashboard menampilkan tiga faktor finansial utama yang mempengaruhi status siswa:

**1. Pengaruh Beasiswa terhadap Status**

Visualisasi menunjukkan bahwa siswa penerima beasiswa (1) memiliki proporsi Graduate yang lebih tinggi dibandingkan yang tidak menerima beasiswa (0). Dari donut chart "Dropout berdasarkan Beasiswa", terlihat bahwa **90.6% siswa yang dropout tidak menerima beasiswa**, sementara hanya **9.4% yang menerima beasiswa**. Hal ini mengindikasikan bahwa beasiswa berperan penting dalam mendukung keberhasilan studi siswa.

**2. Pengaruh Pembayaran terhadap Status**

Siswa yang membayar biaya kuliah tepat waktu (1) menunjukkan tingkat kelulusan yang jauh lebih tinggi. Dari visualisasi "Graduate berdasarkan Pembayaran", **98.7% siswa yang lulus membayar tepat waktu**. Sebaliknya, siswa yang tidak membayar tepat waktu cenderung mengalami dropout, menjadikan faktor pembayaran sebagai indikator penting risiko dropout.

**3. Pengaruh Hutang terhadap Status**

Visualisasi menunjukkan bahwa siswa yang memiliki hutang (1) memiliki proporsi dropout yang lebih tinggi dibandingkan kategori lain.

### Kombinasi Faktor Finansial

Grafik bar di bagian bawah menunjukkan kombinasi antara status beasiswa dan status pembayaran. Terlihat bahwa kelompok siswa terbesar adalah yang **tidak menerima beasiswa dan membayar tepat waktu**. Sementara itu, kelompok penerima beasiswa memiliki jumlah yang lebih kecil namun dengan tingkat keberhasilan yang relatif baik.

### Insight Utama dari Dashboard

1. **Faktor pembayaran tepat waktu** merupakan indikator paling kuat terhadap keberhasilan studi siswa (98.7% Graduate membayar tepat waktu).
2. **Beasiswa** berperan signifikan dalam mengurangi risiko dropout (90.6% Dropout tidak menerima beasiswa).
3. **Status hutang** juga berkorelasi dengan risiko dropout yang lebih tinggi.
4. Kombinasi faktor finansial (beasiswa + pembayaran tepat waktu) dapat digunakan sebagai early warning system untuk mengidentifikasi siswa berisiko dropout.

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning untuk memprediksi status siswa (Dropout, Enrolled, Graduate) telah dibuat menggunakan **Streamlit**. Aplikasi ini menerima input data siswa dan menampilkan hasil prediksi beserta probabilitasnya.

### Cara Menjalankan Secara Lokal

1. **Pastikan semua dependencies terinstall:**
```bash
pip install -r requirements.txt
```

2. **Jalankan aplikasi Streamlit:**
```bash
streamlit run app.py
```

3. **Akses aplikasi melalui browser:**
   - Buka browser dan akses `http://localhost:8501`

### Fitur Aplikasi

Aplikasi ini memiliki fitur:
- **Input Data Siswa**: Form untuk memasukkan data akademik (jumlah MK lulus dan nilai semester 1 & 2), faktor finansial (status pembayaran, status hutang), dan demografi (usia saat mendaftar).
- **Prediksi Status**: Menampilkan hasil prediksi status siswa (Dropout/Enrolled/Graduate).
- **Probabilitas Prediksi**: Menampilkan probabilitas untuk setiap kelas.
- **Rekomendasi Tindakan**: Memberikan rekomendasi intervensi jika siswa diprediksi Dropout.

### Link Akses Prototype (Deployment)

🔗 **Link Streamlit Cloud:** [Student Dropout Prediction App](https://student-dropout-prediction-jayajayainstitut.streamlit.app/)

*Catatan: Jika link tidak tersedia, jalankan aplikasi secara lokal menggunakan langkah-langkah di atas.*

## Conclusion

Berdasarkan analisis data, business dashboard, dan model machine learning yang telah dikembangkan, diperoleh kesimpulan sebagai berikut:

### Performa Model Machine Learning

Model **Random Forest** dipilih sebagai model terbaik dengan performa:
- **Akurasi**: 66%
- **Precision untuk Dropout**: 83% (artinya ketika model memprediksi seorang siswa akan Dropout, 83% prediksi tersebut benar)
- **F1-Score untuk Dropout**: 0.71

Model ini menggunakan **7 fitur** yang dipilih berdasarkan EDA dan domain knowledge:
1. Jumlah mata kuliah lulus semester 1 & 2
2. Rata-rata nilai semester 1 & 2
3. Status pembayaran biaya kuliah tepat waktu
4. Status hutang
5. Usia saat mendaftar

### Faktor yang Berpengaruh terhadap Dropout

- **Performa akademik rendah** (jumlah MK lulus sedikit dan nilai semester rendah) merupakan indikator kuat siswa akan Dropout.
- **Pembayaran biaya kuliah tidak tepat waktu** sangat berkorelasi dengan tingkat Dropout yang tinggi (98.7% siswa Graduate membayar tepat waktu).
- **Tidak menerima beasiswa** meningkatkan risiko Dropout secara signifikan (90.6% siswa Dropout tidak menerima beasiswa).
- **Memiliki hutang** berkorelasi dengan risiko Dropout yang lebih tinggi.
- **Usia lebih tua saat mendaftar** cenderung memiliki risiko Dropout yang lebih besar.

### Faktor yang Kurang Berpengaruh Signifikan terhadap Dropout

- **Faktor ekonomi makro** (unemployment rate, inflation rate, GDP) tidak menunjukkan korelasi yang kuat dengan status Dropout.
- **Kualifikasi dan pekerjaan orang tua** tidak menjadi faktor dominan dalam menentukan keberhasilan studi siswa.
- **Nilai masuk (Admission grade)** memiliki pengaruh yang relatif kecil dibandingkan performa akademik selama kuliah.

### Statistik Utama

- **Total siswa**: 4,424
- **Tingkat Dropout**: 32% (1,421 siswa)
- **Tingkat Graduate**: 50% (2,209 siswa)
- **Tingkat Enrolled**: 18% (794 siswa)

### Rekomendasi Action Items

Berdasarkan hasil analisis dan temuan dari proyek ini, berikut rekomendasi action items yang dapat dilakukan Jaya Jaya Institut untuk mengurangi tingkat Dropout:

1. **Implementasi Early Warning System**
   - Menggunakan model machine learning yang telah dikembangkan untuk mengidentifikasi siswa berisiko Dropout secara dini di awal semester.
   - Melakukan monitoring performa akademik siswa secara berkala, terutama setelah ujian semester 1 dan 2.

2. **Program Intervensi Akademik**
   - Menyediakan program bimbingan belajar atau tutoring bagi siswa dengan performa akademik rendah (jumlah MK lulus sedikit atau nilai rendah).
   - Memberikan konseling akademik untuk membantu siswa merencanakan strategi belajar yang lebih efektif.

3. **Dukungan Finansial yang Lebih Terarah**
   - Memperluas program beasiswa untuk siswa yang berpotensi namun memiliki keterbatasan finansial, mengingat 90.6% siswa Dropout tidak menerima beasiswa.
   - Menyediakan opsi cicilan atau keringanan biaya kuliah bagi siswa yang kesulitan membayar tepat waktu.
   - Memberikan bantuan atau konseling finansial bagi siswa yang memiliki hutang.

4. **Monitoring Pembayaran dan Intervensi Proaktif**
   - Memantau status pembayaran biaya kuliah secara real-time dan menghubungi siswa yang mulai telat membayar sebelum situasi memburuk.
   - Menawarkan solusi alternatif pembayaran sebelum siswa mengambil keputusan untuk Dropout.

5. **Program Khusus untuk Siswa dengan Usia Lebih Tua**
   - Mengembangkan program fleksibilitas jadwal atau kelas khusus bagi siswa yang mendaftar di usia lebih tua, yang mungkin memiliki tanggung jawab lain (pekerjaan, keluarga).
   - Menyediakan layanan konseling untuk membantu menyeimbangkan antara studi dan tanggung jawab lainnya.

6. **Evaluasi dan Perbaikan Berkelanjutan**
   - Melakukan evaluasi berkala terhadap efektivitas program intervensi yang dilakukan.
   - Mengupdate model machine learning secara periodik dengan data terbaru untuk meningkatkan akurasi prediksi.
