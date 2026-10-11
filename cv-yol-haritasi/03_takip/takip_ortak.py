"""
takip_ortak.py — Takip demolarının paylaştığı parçalar: tespit kaynağı + görselleştirme.

TESPİT KAYNAĞI (dedektor ayarı):
    "yolo" : Ultralytics YOLO (hazır COCO ya da kendi best.pt). Gerçek kullanım.
    "renk" : HSV renk eşiği + kontur (00_temeller/klasik_islemler.py'deki yöntem).
             Model GEREKTİRMEZ: ekranda renkli bir top/kapak oynatarak takibi hemen deneyebilirsin.

Takipçiye giden tespit formatı (hepsinde aynı):
    np.ndarray (N, 6) -> x1, y1, x2, y2, guven, sinif_no

GÖRSELLEŞTİRME:
    izleri_ciz            : kutu + ID + iz (geçmiş merkezler) + Kalman hız oku
    maliyet_matrisi_gorsel: Tespit × iz IoU matrisi ısı haritası; seçilen eşleşmeler çerçeveli
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import ayar  # noqa: E402
from ortak.cizim import id_rengi, kutu_ciz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

DOSYA = "03_takip/takip_ortak.py"


@dataclass
class TespitAyarlari:
    dedektor: str = ayar("yolo", "Tespit kaynağı: yolo (model) | renk (HSV eşiği, model gerektirmez)", secenekler=["yolo", "renk"])
    model: str = ayar("yolo11n.pt", "YOLO modeli (dedektor=yolo)", tur="dosya")
    guven: float = ayar(0.3, "Tespit güven eşiği", min=0.05, max=0.95, adim=0.05)
    siniflar: str = ayar("person", "Takip edilecek sınıflar (virgülle; boş = hepsi). Ör: 'person,cell phone'")
    h_min: int = ayar(100, "renk modu: HSV ton alt (mavi≈100-130, yeşil≈40-80, kırmızı≈0-10)", min=0, max=179)
    h_max: int = ayar(130, "renk modu: HSV ton üst", min=0, max=179)
    s_min: int = ayar(120, "renk modu: doygunluk alt", min=0, max=255)
    min_alan: int = ayar(300, "renk modu: en küçük blob alanı", min=10, max=20000, adim=10)


def tespit_al(kare, a: TespitAyarlari, anlat, model=None) -> tuple[np.ndarray, dict]:
    """Seçili dedektörle (N,6) tespit dizisi ve sınıf isimlerini döndür."""
    if a.dedektor == "renk":
        hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
        maske = cv2.inRange(hsv, np.array([a.h_min, a.s_min, 50]), np.array([a.h_max, 255, 255]))
        maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tespitler = []
        for c in konturlar:
            if cv2.contourArea(c) >= a.min_alan:
                x, y, w, h = cv2.boundingRect(c)
                tespitler.append([x, y, x + w, y + h, 1.0, 0])
        anlat.adim(f"Renk dedektörü: {len(tespitler)} blob", kod=f"{DOSYA}:tespit_al(renk)")
        return np.array(tespitler, np.float32).reshape(-1, 6), {0: "renkli"}

    yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
    th = modul_yukle("02_yolo_finetune/tahmin.py")
    model = model or yy.model_yukle(a.model)
    secili = th.sinif_filtresi(a.siniflar, model.names)
    r = model.predict(kare, conf=a.guven, classes=secili, verbose=False)[0]
    k, s, g, isimler = yy.sonucu_ayikla(r)
    anlat.adim(f"YOLO: {len(k)} tespit", kod=f"{DOSYA}:tespit_al(yolo)")
    return np.hstack([k, g[:, None], s[:, None]]).astype(np.float32).reshape(-1, 6), isimler


def izleri_ciz(img, izler: list[dict], isimler: dict, tespitler: np.ndarray | None = None):
    """izler: [{'id', 'kutu', 'sinif', 'gecmis': [(x,y),...], 'hiz': (vx,vy)}]"""
    if tespitler is not None:
        for t in tespitler:   # ham tespitler ince beyaz
            kutu_ciz(img, t[:4], "", (255, 255, 255), 1)
    for iz in izler:
        renk = id_rengi(iz["id"])
        kutu_ciz(img, iz["kutu"], f"ID {iz['id']} {isimler.get(int(iz['sinif']), '')}", renk)
        gecmis = iz.get("gecmis", [])
        for p1, p2 in zip(gecmis[:-1], gecmis[1:]):
            cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), renk, 2)
        if "hiz" in iz and gecmis:
            cx, cy = gecmis[-1]
            vx, vy = iz["hiz"]
            cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx + vx * 5), int(cy + vy * 5)), (0, 255, 255), 2, tipLength=0.3)
    return img


def maliyet_matrisi_gorsel(iou: np.ndarray, eslesmeler: list[tuple[int, int]], hucre: int = 46) -> np.ndarray:
    """Satır = tespit, sütun = iz. Hücre rengi = IoU (koyu=0, parlak=1). Seçilen eşleşmeler yeşil çerçeveli."""
    n, m = iou.shape
    if n == 0 or m == 0:
        img = np.full((80, 360, 3), 255, np.uint8)
        cv2.putText(img, f"Matris bos ({n} tespit x {m} iz)", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img
    img = np.full((n * hucre + 30, m * hucre + 40, 3), 255, np.uint8)
    for i in range(n):
        cv2.putText(img, f"T{i}", (4, 30 + i * hucre + hucre // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for j in range(m):
            v = float(iou[i, j])
            x0, y0 = 40 + j * hucre, 30 + i * hucre
            cv2.rectangle(img, (x0, y0), (x0 + hucre - 2, y0 + hucre - 2), (int(255 * (1 - v)), int(200 * v + 30), int(255 * (1 - v))), -1)
            cv2.putText(img, f"{v:.2f}", (x0 + 4, y0 + hucre // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            if (i, j) in eslesmeler:
                cv2.rectangle(img, (x0, y0), (x0 + hucre - 2, y0 + hucre - 2), (0, 160, 0), 3)
    for j in range(m):
        cv2.putText(img, f"iz{j}", (44 + j * hucre, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img
