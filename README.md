
# Deteksi Berita Clickbait Berbahasa Indonesia dengan IndoELECTRA

Sistem klasifikasi otomatis untuk mendeteksi berita **clickbait** dalam bahasa Indonesia, menggunakan model **IndoELECTRA** yang di-fine-tune pada dataset ekonomi digital dari media daring Indonesia. Proyek ini bertujuan untuk mendukung peningkatan literasi digital konsumen dalam menghadapi banjir informasi di ekosistem ekonomi digital.

## 📌 Ringkasan

- **Model**: IndoELECTRA (oleh ChristopherA08)
- **Dataset**: 1.250 judul berita dari DetikFinance, Kompas, dan CNN Indonesia
- **Metodologi**: Fine-tuning IndoELECTRA menggunakan PyTorch + HuggingFace Transformers
- **Akurasi**: 87.2%
- **F1-score**: 85.84%
- **Tujuan**: Meningkatkan literasi digital dan deteksi otomatis konten menyesatkan (clickbait)

---

## 🧠 Latar Belakang

Judul clickbait sering kali mengecoh konsumen berita daring. Dalam konteks ekonomi digital, hal ini bisa mengarah pada misinformasi yang memengaruhi keputusan finansial. Oleh karena itu, dibutuhkan sistem deteksi otomatis berbasis NLP yang mampu mengenali pola-pola provokatif atau manipulatif dalam judul berita.

---


## 📊 Evaluasi Model

| Metrik     | Nilai     |
|------------|-----------|
| Akurasi    | 87.2%     |
| Precision  | 86.4%     |
| Recall     | 85.8%     |
| F1-Score   | 85.84%    |

Model juga berhasil mengklasifikasikan judul baru yang belum pernah dilatih, membuktikan kemampuan generalisasinya dalam konteks nyata.

---

## 🎯 Kontribusi dan Implikasi

- 📌 Membantu konsumen berita daring untuk **lebih kritis dan terlindungi**
- 🔎 Memperkuat **literasi digital dan ketahanan informasi masyarakat**
- 💡 Menjadi langkah awal dalam pengembangan alat bantu berbasis AI untuk menghadapi **disinformasi ekonomi digital**

---

## 👨‍💻 Tim Pengembang

- **Muhammad Zuhdi** (muhammadzuhdi391@gmail.com)  
- Deananda  
- Mahmuddin  
Program Studi Informatika  
Universitas Sulawesi Barat

---
