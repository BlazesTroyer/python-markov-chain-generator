# 7/24 ev kamerası: 10_dagitim/kamera_7x24.py'yi küçük bir CPU imajında çalıştırır.
# Derle (cv-yol-haritasi/ klasöründen):   docker build -f 10_dagitim/Dockerfile -t kisisel-goz .
# Çalıştır:                               docker compose -f 10_dagitim/docker-compose.yml up -d
FROM python:3.12-slim

# OpenCV'nin sistem bağımlılıkları (headless sürüm GUI gerektirmez)
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /uygulama
# Sadece CPU torch (GPU imajı ~5 GB daha büyük olurdu); onnxruntime: export edilmiş modeli hızlı çalıştırmak için
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "ultralytics>=8.3" onnxruntime opencv-python-headless numpy scipy lap

# Kod (veri/ ve modeller imaja GİRMEZ; volume olarak bağlanır — bkz. docker-compose.yml)
COPY ortak/ ortak/
COPY 00_temeller/ 00_temeller/
COPY 01_veri_fabrikasi/ 01_veri_fabrikasi/
COPY 02_yolo_finetune/ 02_yolo_finetune/
COPY 03_takip/ 03_takip/
COPY 09_agent/ 09_agent/
COPY 10_dagitim/ 10_dagitim/

ENV CV_VERI_KOK=/veri PYTHONUNBUFFERED=1
# Varsayılan: webcam 0 + export edilmiş ONNX model (compose'ta değiştirilebilir)
ENTRYPOINT ["python", "10_dagitim/kamera_7x24.py"]
CMD ["--kaynak", "0", "--model", "/veri/modeller/yolo11n.onnx", "--nabiz-sn", "300", "--log-dosyasi", "/veri/kamera.log"]
