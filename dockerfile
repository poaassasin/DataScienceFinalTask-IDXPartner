# Gunakan image Python yang ringan
FROM python:3.11-slim

# Set folder kerja
WORKDIR /app

# --- BAGIAN BARU: Instalasi dependensi sistem ---
# Kita butuh libgomp1 agar LightGBM bisa berjalan
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Salin daftar library dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek
COPY . .

# Jalankan API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]