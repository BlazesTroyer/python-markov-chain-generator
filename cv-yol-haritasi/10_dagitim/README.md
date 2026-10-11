# 10 · Dağıtım ve Optimizasyon — 7/24 çalışan ev kamerası

Panelde çalışan bir demo ile günlerce kesintisiz çalışan bir sistem arasında fark büyüktür. Bu bölümde:
1. **Export**: modeli dağıtım formatına çevir ve **doğruluk kaybını ölç**,
2. **Benchmark**: hızı doğru ölç (ısınma, p50/p95, aşama dökümü),
3. **7/24 çalıştırıcı**: kare kuyruğu, kopan bağlantı, hata, kapanma sinyali, nabız log'u,
4. **Docker**: tek komutla, makine yeniden başlasa da çalışan kurulum.

## Kavramlar

| Kavram | Özet |
|---|---|
| ONNX / OpenVINO / TensorRT | Modeli PyTorch'suz, donanıma göre hızlı çalıştıran formatlar (CPU / Intel / NVIDIA) |
| FP16 / INT8 | Daha az bitle ağırlık: küçük ve hızlı model. INT8 kalibrasyon ister; doğruluk kaybı ölçülmeli |
| p50 / p95 | Medyan gecikme / en kötü %5'in sınırı. 7/24 sistemde "arada takılma" p95'te görünür |
| Aşama dökümü | Ön işleme + ağ + son işleme (NMS). Darboğazı bilmeden optimizasyon yapma |
| Son-kare okuyucu | İşleme yavaşsa eski kareleri biriktirme, en tazesini işle: gecikme sabit kalır |

## Bu ortamda ölçülen örnek (CPU, yolo11n, bus.jpg)

| model | imgsz | p50 ms | p95 ms | FPS | ön | ağ | son |
|---|---|---|---|---|---|---|---|
| yolo11n.pt | 320 | 25.8 | 33.4 | 38.7 | 1.2 | 23.3 | 1.0 |
| yolo11n.pt | 640 | 72.8 | 79.4 | 13.7 | 3.7 | 66.6 | 1.1 |
| yolo11n.onnx (640'ta export) | 640 | 59.3 | 95.5 | 16.9 | 7.0 | 45.9 | 1.5 |
| yolo11n.onnx (640'ta export) | 320 | 67.2 | 106.5 | 14.9 | 11.4 | 51.0 | 1.8 |

Çıkarılacak dersler:
- **imgsz en büyük kaldıraç**: .pt'de 640 → 320 yaklaşık 3 kat hız kazandırdı.
- **ONNX, 640'ta .pt'den hızlı** (ağ 66.6 → 45.9 ms).
- **Sabit boyutla export edilen ONNX, başka imgsz'de hızlanmaz**: 640'a sabitlenmiş model 320'de daha yavaş çalıştı.
  320 istiyorsan `--imgsz 320` ile ayrıca export et.
- **ONNX dosyası (10.7 MB) .pt'den (5.6 MB) büyük**: .pt yarım hassasiyetli (FP16) ağırlık saklar, ONNX varsayılan FP32.
- **Export doğruluk kontrolü**: kutuların %100'ü eşleşti, ortalama IoU 0.977, ortalama güven farkı 0.046. Küçük ama sıfır değil.

Kendi bilgisayarında sonuçlar farklı olacak. Bu tabloyu kendi ölçümlerinle doldur.

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `export.py` | Eylem (uzun) | Export + dosya boyutu + **tahmin uyumu** (eşleşme, IoU, güven farkı) + isteğe bağlı val mAP |
| `benchmark.py` | Eylem (uzun) | Model × imgsz; ısınma + tekrar; p50/p95, FPS, aşama dökümü |
| `kamera_7x24.py` | betik | Dayanıklı çalıştırıcı: `SonKareOkuyucu`, yeniden bağlanma, hata izolasyonu, SIGTERM, NABIZ |
| `Dockerfile`, `docker-compose.yml` | dağıtım | CPU imajı; `veri/` volume (panel/agent aynı veriyi görür); `restart: unless-stopped` |

## Çalıştır

```bash
pip install -r 10_dagitim/requirements.txt
python 10_dagitim/export.py --model veri/egitimler/esyalarim/weights/best.pt --format onnx --imgsz 416 --anlat
python 10_dagitim/benchmark.py --modeller "veri/egitimler/esyalarim/weights/best.pt, veri/egitimler/esyalarim/weights/best.onnx" --boyutlar "320,416,640"
python 10_dagitim/kamera_7x24.py --kaynak 0 --model veri/egitimler/esyalarim/weights/best.onnx --nabiz-sn 60

# Docker (cv-yol-haritasi/ içinden):
cp veri/egitimler/esyalarim/weights/best.onnx veri/modeller/yolo11n.onnx    # ya da compose'taki --model yolunu değiştir
docker compose -f 10_dagitim/docker-compose.yml up -d
docker compose -f 10_dagitim/docker-compose.yml logs -f
```

Raspberry Pi 5 / Jetson notu: Pi'de `--format openvino` değil `onnx` ya da `ncnn` dene; Jetson'da `--format engine --half`
(TensorRT) en hızlısıdır ama **o cihazın üzerinde** export edilmelidir.

## Neye bakmalısın?

- **NABIZ satırı**: `kameradan okunan` ile `işlenen` arasındaki fark, sistemin yetişemediğini gösterir. Atlanan kare çok ise imgsz'i düşür ya da export et.
- **Benchmark'ta aşama dökümü**: "ön" sütunu "ağ"a yakınsa darboğaz model değil, görüntü hazırlığıdır.
- **Export uyumu**: INT8'de eşleşme oranı ve IoU ne kadar düşüyor? Senin kullanımın için kabul edilebilir mi?

## Deneyler

1. Kendi eşya modelini `.pt` / `.onnx` / `.onnx --half` / `openvino --int8 --data ...` olarak export et; uyum ve hız tablosunu doldur.
2. `kamera_7x24.py`'yi bir gece çalıştır; sabah `veri/kamera.log`'daki NABIZ satırlarından gece boyunca atlanan kare oranını ve hareket kapısının kazandırdığını hesapla.
3. Kablosuz IP kamerada `--yeniden-baglan-sn`'i dene: router'ı yeniden başlat, sistem kendiliğinden toparlanıyor mu?
4. Hareket kapısını kapat (`--no-hareket-kapisi`) ve aynı gece CPU kullanımını karşılaştır (enerji tasarrufu).
