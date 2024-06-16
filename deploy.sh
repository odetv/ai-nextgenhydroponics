#!/bin/bash

# Direktori project di VPS
PROJECT_DIR=~/project/api-model-nextgenhydroponics

# Navigasi ke direktori project
cd $PROJECT_DIR

# Tarik perubahan terbaru dari GitHub
git pull origin main

# Hentikan aplikasi yang sedang berjalan
PIDS=$(ps aux | grep "python api.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

# Tambahkan echo untuk memverifikasi direktori saat ini
echo "Sedang berada di direktori: $(pwd)"

# Pastikan Anda berada di direktori yang benar sebelum menjalankan aplikasi
# Kembali ke direktori project untuk memastikan
cd $PROJECT_DIR

echo "Sekarang sedang berada di direktori: $(pwd)"

# Jalankan kembali aplikasi dengan nohup
nohup python api.py > output.log 2>&1 &

echo "Deploy selesai."
