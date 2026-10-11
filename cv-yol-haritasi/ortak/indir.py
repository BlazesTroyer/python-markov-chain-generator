"""
indir.py — Model dosyalarını (ONNX, MediaPipe .task) bir kez indirip veri/modeller/ altında saklar.

NEDEN?
    YuNet (yüz tespiti), SFace (yüz tanıma), MediaPipe el/poz modelleri pip paketi değil, AYRI
    dosyalardır. Her script'te "şuradan indir, şuraya koy" demek yerine tek yerden yönetiyoruz:

        yol = model_dosyasi("yunet")      # yoksa indirir, varsa doğrudan yolu verir

    Hangi modelin nereden geldiği aşağıdaki MODELLER tablosunda açıkça yazılı.
    İnternetsiz makinede: dosyayı başka yerden indirip veri/modeller/<dosya_adi> olarak koy, yeter.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from .yukle import VERI

MODELLER = {
    # ad: (dosya adı, URL, açıklama)
    "yunet": ("face_detection_yunet_2023mar.onnx",
              "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              "OpenCV YuNet yüz dedektörü (~230 KB): kutu + 5 nokta (gözler, burun, ağız köşeleri)"),
    "sface": ("face_recognition_sface_2021dec.onnx",
              "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
              "OpenCV SFace yüz tanıma (~37 MB): hizalanmış yüz -> 128 boyutlu embedding"),
    "el": ("hand_landmarker.task",
           "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
           "MediaPipe el: 21 nokta / el"),
    "poz": ("pose_landmarker_lite.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "MediaPipe poz (lite): 33 vücut noktası"),
}


def model_dosyasi(ad: str) -> Path:
    """Model dosyasının yerel yolu; yoksa indirir."""
    dosya, url, aciklama = MODELLER[ad]
    yol = VERI / "modeller" / dosya
    if yol.exists() and yol.stat().st_size > 1000:
        return yol
    yol.parent.mkdir(parents=True, exist_ok=True)
    print(f"[indir] {aciklama}\n        {url}\n     -> {yol}", flush=True)
    gecici = yol.with_suffix(yol.suffix + ".part")
    urllib.request.urlretrieve(url, gecici)   # yarım kalırsa .part kalır, bir sonraki seferde yeniden denenir
    gecici.rename(yol)
    return yol
