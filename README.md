# Factors That Influence Career Success

Dashboard interaktif berbasis **Streamlit + Plotly** untuk menganalisis faktor-faktor yang memengaruhi kesuksesan karier awal lulusan, dilengkapi **kuesioner prediksi jumlah Job Offers** menggunakan model machine learning.

> Mendukung **SDG 4 (Quality Education)** dan **SDG 8 (Decent Work and Economic Growth)**
> Live demo: http://joboffer-prediction.streamlit.app/

---

## Fitur Utama

**Dashboard analisis (7 halaman):**

| Halaman | Fokus |
|---|---|
| Exploratory Data Analysis (EDA) | Profil IPK, sebaran bidang studi, portofolio proyek/magang/sertifikasi |
| 1. Recruitment Trends | Pengaruh GPA vs soft skills terhadap job offers |
| 2. Career Well-Being | Gaji, work-life balance, dan kepuasan karier |
| 3. Skills Gap | Dampak magang, proyek, sertifikasi vs pendidikan formal |
| 4. Gender Parity | Perbandingan gaji awal, waktu promosi, dan job offers antar gender |
| 5. Unequal Opportunities | Ketimpangan hasil karier pada GPA setara dan antar bidang studi |
| 6. Conclusion | Ringkasan temuan dan implikasi |
| 7. Prediksi Job Offers | Kuesioner untuk memprediksi jumlah job offers |

Semua halaman analisis mendukung **filter global** di sidebar (Gender, Field of Study, University GPA, Usia, Starting Salary).

**Kuesioner prediksi** menerima 10 input: `Age`, `Gender`, `University_GPA`, `Field_of_Study`, `Internships_Completed`, `Projects_Completed`, `Certifications`, `Soft_Skills_Score`, `Networking_Score`, dan `Starting_Salary`, lalu menampilkan estimasi jumlah Job Offers.

---

## Struktur Repository

```
.
├── .vscode
├── dashboard/
│   ├── dashboard.py              # Dashboard Streamlit (7 halaman)
├── dataset/
│   ├── education_career_success.csv # Dataset organik
│   ├── education_career_success_cleaned.csv #Dataset bersih
├── notebook/
│   ├── eda_analysis.ipynb     # Analisis EDA dataset
├── modelling.ipynb #Perbandingan dan pemilihan model
├── model_career_prediction2.pkl # Model prediksi
├── requirements.txt
└── README.md
```

> `dashboard.py` membaca dataset dari `../dataset/education_career_success_cleaned.csv` dan model dari `model_career_prediction2.pkl`. Path ini dapat diubah lewat konstanta `DEFAULT_PATH` dan `MODEL_PATH` di bagian atas `dashboard.py`.

---

##  Instalasi & Menjalankan

**1. Clone repository**

```bash
git clone <url-repository-kamu>
cd <nama-folder>
```

**2. Install dependensi**

```bash
pip install streamlit pandas numpy plotly scikit-learn joblib
```

**3. Jalankan dashboard**

```bash
cd dashboard
streamlit run app.py
```

---

## Model Machine Learning

Target prediksi: **`Job_Offers`** (jumlah tawaran kerja, bilangan bulat kecil → diperlakukan sebagai regresi/count).

Lima model dibandingkan pada test set 20% (`random_state=42`):

| Model | MAE | MSE | RMSE | R² | Accuracy | F1 Score |
|---|---|---|---|---|---|---|
| Random Forest | 0.012 | 0.002 | 0.046 | 0.999 | 1.000 | 1.000 |
| Gradient Boosting | 0.007 | 0.002 | 0.046 | 0.999 | 1.000 | 1.000 |
| Ridge Regression | 0.142 | 0.033 | 0.180 | 0.983 | 0.988 | 0.987 |
| Linear Regression | 0.143 | 0.033 | 0.181 | 0.983 | 0.988 | 0.987 |
| Poisson Regression | 0.248 | 0.092 | 0.303 | 0.953 | 0.950 | 0.938 |

> MAE, MSE, RMSE, dan R² dihitung dari prediksi asli (regresi). Accuracy dan F1 Score (weighted) dihitung setelah prediksi dibulatkan ke bilangan bulat terdekat, karena `Job_Offers` berupa bilangan bulat kecil.


Model yang dipakai di aplikasi: **Gradient Boosting Regressor** (dilatih hanya dari 10 fitur kuesioner).

**Preprocessing:** `OneHotEncoder(min_frequency=10)` untuk fitur kategorikal (`Gender`, `Field_of_Study`), sehingga kategori dengan sampel sangat sedikit (Finance, Education, Nursing) digabung otomatis.

Untuk menjalankan perbandingan seluruh model beserta metrik tambahan (MSE, Accuracy, F1 pada prediksi yang dibulatkan):

```bash
python modelling.ipynb
```

---

## Catatan & Keterbatasan

- Hampir semua fitur numerik pada dataset berkorelasi sangat kuat (≈0.95–0.98) dengan `Job_Offers`. Pola sebersih ini jarang muncul di data dunia nyata, sehingga dataset kemungkinan **sintetis atau dibuat dengan rumus**. Karena itu, nilai R² yang mendekati 1 **tidak boleh dianggap sebagai bukti performa model di dunia nyata**.
- Hasil prediksi adalah estimasi statistik untuk keperluan akademik/pembelajaran, bukan jaminan hasil aktual.
- Dataset hanya berisi 400 baris, dengan beberapa kategori bidang studi yang sangat sedikit sampelnya.

---

## Teknologi

Python · Streamlit · Plotly · pandas · NumPy · scikit-learn · joblib

---
