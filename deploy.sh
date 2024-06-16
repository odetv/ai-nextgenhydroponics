PROJECT_DIR=~/project/api-model-nextgenhydroponics

cd $PROJECT_DIR

git pull origin main

PIDS=$(ps aux | grep "python api.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

nohup python api.py > output.log 2>&1 &

echo "Deploy selesai."
