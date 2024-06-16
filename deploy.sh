#!/bin/bash

# Direktori project
PROJECT_DIR=~/project/api-model-nextgenhydroponics

# Navigasi ke direktori project
cd $PROJECT_DIR

# Tarik perubahan terbaru dari GitHub
git pull origin main

# Hentikan aplikasi yang sedang berjalan
PIDS=$(ps aux | grep "python index.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

# Jalankan kembali aplikasi
nohup python index.py > output.log 2>&1 &

echo "Deploy selesai."
