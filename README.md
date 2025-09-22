## AI Chatbot (Google Gemini)

Aplikasi chatbot berbasis Streamlit yang menggunakan Google AI Studio (Gemini) sebagai model bahasa (LLM). API key dimasukkan melalui UI (sidebar), tanpa disimpan di kode. Chatbot juga dilengkapi retrieval pengetahuan sederhana untuk memberi konteks tambahan.

### Fitur
- Gaya bahasa: formal atau santai
- Retrieval knowledge base (FAISS + SentenceTransformers)
- Memori percakapan singkat (riwayat terbaru)
- Rekomendasi tindak lanjut otomatis

### Persiapan
1. Buat virtualenv (opsional):
```bash
python -m venv .venv
```
2. Aktifkan:
```bash
# PowerShell
.venv\Scripts\Activate.ps1
```
3. Install dependensi:
```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi (Gemini via Google AI Studio)
```bash
streamlit run app.py
```
Pertama kali jalan akan menginisialisasi koneksi Gemini. Buka URL yang ditampilkan (biasanya `http://localhost:8501`).

Untuk memakai Google Gemini:
1. Dapatkan API Key dari Google AI Studio (aktifkan Generative Language API).
2. Isi API key langsung di sidebar (rekomendasi) atau set `GOOGLE_API_KEY` di environment.
   - PowerShell:
   ```powershell
   setx GOOGLE_API_KEY "<API_KEY>"
   $env:GOOGLE_API_KEY = "<API_KEY>"
   ```
3. Di sidebar, pastikan model: `gemini-1.5-flash` atau `gemini-1.5-pro`.

### Screenshots UI
Ambil tangkapan layar dari UI Streamlit (halaman utama dan contoh percakapan). Simpan untuk pelaporan tugas.


```

### Kustomisasi Lanjutan
- Tambahkan file di folder `data/` untuk domain baru atau memperkaya konten.
- Ubah `utils.py` untuk memakai model lain (mis. `mistralai/Mistral-7B-Instruct` via API) bila diperlukan.
- Integrasikan API eksternal (cuaca, rute) di domain Travel.



