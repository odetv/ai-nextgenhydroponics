#!/bin/bash

# Direktori project di VPS
PROJECT_DIR=~/project/api-model-nextgenhydroponics

# Navigasi ke direktori project
cd $PROJECT_DIR

# Tarik perubahan terbaru dari GitHub
git pull origin main

# Pastikan Anda berada di direktori yang benar sebelum menjalankan aplikasi
echo "Sedang berada di direktori: $(pwd)"

# Hentikan aplikasi yang sedang berjalan
PIDS=$(ps aux | grep "python api.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

# Jalankan kembali aplikasi dengan nohup menggunakan python3
nohup python api.py > output.log 2>&1 &

sleep 3

echo "Deploy selesai. Aplikasi FastAPI sudah berjalan."
