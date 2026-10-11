"""
metrikler.py — Tespit metriklerini SIFIRDAN: IoU, Precision, Recall, AP, mAP@50, mAP@50-95.

NEDEN SIFIRDAN?
    YOLO eğitiminin sonunda "mAP50-95: 0.62" yazar. Bu sayının ne demek olduğunu bilmeden
    "augmentation işe yaradı mı?" sorusunu cevaplayamazsın. Burada her adım açık.

KAVRAMLAR (sırayla):
    IoU (Intersection over Union)
        İki kutunun örtüşme oranı = kesişim alanı / birleşim alanı. 0 (hiç) .. 1 (birebir).
    Eşleştirme
        Bir tahmin, aynı sınıftan ve IoU >= eşik olan, daha önce eşleşmemiş bir gerçek kutuyla
        eşleşirse TP (doğru pozitif), yoksa FP (yanlış pozitif). Eşleşmeyen gerçek kutular FN.
        Tahminler GÜVEN skoruna göre büyükten küçüğe işlenir (önemli!).
    Precision = TP / (TP + FP)   "Dediklerimin ne kadarı doğru?"
    Recall    = TP / (TP + FN)   "Gerçekte olanların ne kadarını buldum?"
    PR eğrisi : Güven eşiğini yukarıdan aşağı indirirken (P, R) noktaları.
    AP        : PR eğrisinin altındaki alan (COCO usulü: 101 noktalı interpolasyon).
    mAP@50    : IoU eşiği 0.5 iken sınıfların AP ortalaması.
    mAP@50-95 : IoU eşiği 0.50, 0.55, ..., 0.95 için mAP'lerin ortalaması (konum hassasiyetini de ölçer).

VERİ FORMATI:
    gercekler : [{"kare": 0, "sinif": "kupa", "kutu": [x1,y1,x2,y2]}, ...]
    tahminler : [{"kare": 0, "sinif": "kupa", "kutu": [...], "guven": 0.87}, ...]

Bu dosyanın altında bir "Metrik Oyun Alanı" Eylem'i var: sentetik veri üretip gürültüyü
değiştirerek mAP'in nasıl tepki verdiğini görebilirsin (panelde 00 · Temeller sekmesi).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import kutu_ciz, yazi_yaz  # noqa: E402

DOSYA = "00_temeller/metrikler.py"


# ============================================================================ IoU
def iou(a, b) -> float:
    """İki kutunun IoU'su. a, b: [x1, y1, x2, y2]."""
    kx1, ky1 = max(a[0], b[0]), max(a[1], b[1])       # kesişimin sol-üst köşesi
    kx2, ky2 = min(a[2], b[2]), min(a[3], b[3])       # kesişimin sağ-alt köşesi
    kesisim = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
    alan_a = (a[2] - a[0]) * (a[3] - a[1])
    alan_b = (b[2] - b[0]) * (b[3] - b[1])
    birlesim = alan_a + alan_b - kesisim
    return float(kesisim / birlesim) if birlesim > 0 else 0.0


def iou_matrisi(A, B) -> np.ndarray:
    """A (N,4) ve B (M,4) arasındaki tüm IoU'lar -> (N, M). Vektörize sürüm (takipte de kullanılır)."""
    A = np.asarray(A, np.float32).reshape(-1, 4)
    B = np.asarray(B, np.float32).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), np.float32)
    kx1 = np.maximum(A[:, None, 0], B[None, :, 0])
    ky1 = np.maximum(A[:, None, 1], B[None, :, 1])
    kx2 = np.minimum(A[:, None, 2], B[None, :, 2])
    ky2 = np.minimum(A[:, None, 3], B[None, :, 3])
    kesisim = np.clip(kx2 - kx1, 0, None) * np.clip(ky2 - ky1, 0, None)
    alan_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    alan_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return kesisim / np.maximum(alan_a[:, None] + alan_b[None, :] - kesisim, 1e-9)


# ============================================================================ eşleştirme
def eslestir(gercekler: list[dict], tahminler: list[dict], sinif: str, iou_esigi: float):
    """Bir sınıf için tahminleri TP/FP olarak işaretle.

    Döndürür: (tp_dizisi, guven_dizisi, gercek_sayisi)
      tp_dizisi[i] = 1 ise güvene göre sıralı i. tahmin doğru pozitif.
    """
    g_kare = defaultdict(list)                      # kare -> o karedeki gerçek kutular
    for g in gercekler:
        if g["sinif"] == sinif:
            g_kare[g["kare"]].append(g["kutu"])
    kullanildi = {k: [False] * len(v) for k, v in g_kare.items()}
    gercek_sayisi = sum(len(v) for v in g_kare.values())

    t = sorted((p for p in tahminler if p["sinif"] == sinif), key=lambda p: -p["guven"])
    tp = np.zeros(len(t))
    for i, p in enumerate(t):
        adaylar = g_kare.get(p["kare"], [])
        en_iyi, en_iyi_j = 0.0, -1
        for j, gk in enumerate(adaylar):
            o = iou(p["kutu"], gk)
            if o > en_iyi:
                en_iyi, en_iyi_j = o, j
        if en_iyi >= iou_esigi and not kullanildi[p["kare"]][en_iyi_j]:
            tp[i] = 1
            kullanildi[p["kare"]][en_iyi_j] = True   # bir gerçek kutu sadece bir kez eşleşir
        # aksi halde FP (ya IoU düşük ya da o gerçek kutu zaten eşleşmiş -> çift tespit)
    return tp, np.array([p["guven"] for p in t]), gercek_sayisi


def pr_egrisi(tp: np.ndarray, gercek_sayisi: int):
    """Kümülatif TP/FP'den precision ve recall dizileri."""
    if len(tp) == 0:
        return np.array([0.0]), np.array([0.0])
    tp_kum = np.cumsum(tp)
    fp_kum = np.cumsum(1 - tp)
    recall = tp_kum / max(gercek_sayisi, 1)
    precision = tp_kum / (tp_kum + fp_kum)
    return precision, recall


def ortalama_hassasiyet(precision: np.ndarray, recall: np.ndarray) -> float:
    """AP: COCO tarzı 101 noktalı interpolasyon.

    1) Precision'ı 'zarf' haline getir: her recall seviyesinde, o recall'dan SONRAKİ en yüksek
       precision kullanılır (eğri monoton azalan olur).
    2) recall = 0.00, 0.01, ..., 1.00 noktalarında bu zarfın değerini oku ve ortalamasını al.
    """
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])                # zarf
    noktalar = np.linspace(0, 1, 101)
    idx = np.searchsorted(r, noktalar, side="left")
    return float(np.mean([p[i] if i < len(p) else 0.0 for i in idx]))


def map_hesapla(gercekler, tahminler, iou_esikleri=None) -> dict:
    """mAP@50, mAP@50-95 ve sınıf bazında AP/P/R'yi hesapla."""
    iou_esikleri = iou_esikleri if iou_esikleri is not None else np.round(np.arange(0.5, 0.96, 0.05), 2)
    siniflar = sorted({g["sinif"] for g in gercekler})
    tablo = {}
    for s in siniflar:
        ap_list = []
        for e in iou_esikleri:
            tp, _, n = eslestir(gercekler, tahminler, s, e)
            p, r = pr_egrisi(tp, n)
            ap_list.append(ortalama_hassasiyet(p, r))
        tp50, _, n50 = eslestir(gercekler, tahminler, s, 0.5)
        p50, r50 = pr_egrisi(tp50, n50)
        tablo[s] = {"AP50": ap_list[0], "AP50_95": float(np.mean(ap_list)),
                    "P_son": float(p50[-1]), "R_son": float(r50[-1]), "gercek": n50, "tahmin": len(tp50)}
    return {
        "mAP50": float(np.mean([v["AP50"] for v in tablo.values()])) if tablo else 0.0,
        "mAP50_95": float(np.mean([v["AP50_95"] for v in tablo.values()])) if tablo else 0.0,
        "siniflar": tablo,
    }


# ============================================================================ oyun alanı
def sentetik_veri(kare_sayisi, konum_gurultusu, kacirma, yanlis_pozitif, rng, g=640, y=480):
    """Gerçek kutular üret, sonra 'kusurlu bir model' gibi tahminler türet."""
    siniflar = ["kupa", "anahtar", "telefon"]
    gercekler, tahminler = [], []
    for k in range(kare_sayisi):
        for _ in range(rng.integers(1, 4)):
            w, h = rng.uniform(60, 180), rng.uniform(60, 180)
            x1, y1 = rng.uniform(0, g - w), rng.uniform(0, y - h)
            s = str(rng.choice(siniflar))
            kutu = [x1, y1, x1 + w, y1 + h]
            gercekler.append({"kare": k, "sinif": s, "kutu": kutu})
            if rng.random() < kacirma:
                continue                                   # model bunu kaçırdı (FN)
            oynama = rng.normal(0, konum_gurultusu, 4) * np.array([w, h, w, h])
            tahminler.append({"kare": k, "sinif": s, "kutu": list(np.array(kutu) + oynama),
                              "guven": float(np.clip(rng.normal(0.75, 0.15), 0.05, 0.99))})
        for _ in range(rng.poisson(yanlis_pozitif)):       # hiç olmayan şeyi "gördü" (FP)
            w, h = rng.uniform(40, 150), rng.uniform(40, 150)
            x1, y1 = rng.uniform(0, g - w), rng.uniform(0, y - h)
            tahminler.append({"kare": k, "sinif": str(rng.choice(siniflar)), "kutu": [x1, y1, x1 + w, y1 + h],
                              "guven": float(np.clip(rng.normal(0.45, 0.15), 0.05, 0.99))})
    return gercekler, tahminler


def pr_cizimi(egriler: dict[str, tuple[np.ndarray, np.ndarray]], boyut=(520, 400)) -> np.ndarray:
    """PR eğrilerini matplotlib olmadan OpenCV ile çiz (bağımlılık az, kod okunur)."""
    G, Y = boyut
    img = np.full((Y, G, 3), 255, np.uint8)
    sol, alt, ust, sag = 50, Y - 40, 20, G - 20
    cv2.rectangle(img, (sol, ust), (sag, alt), (0, 0, 0), 1)
    for v in np.linspace(0, 1, 6):
        x = int(sol + v * (sag - sol)); yy = int(alt - v * (alt - ust))
        cv2.putText(img, f"{v:.1f}", (x - 10, alt + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, f"{v:.1f}", (sol - 35, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "Recall", (G // 2 - 20, Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Precision", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    renkler = [(200, 80, 0), (0, 140, 0), (0, 0, 200), (150, 0, 150)]
    for i, (isim, (p, r)) in enumerate(egriler.items()):
        noktalar = np.stack([sol + r * (sag - sol), alt - p * (alt - ust)], 1).astype(np.int32)
        cv2.polylines(img, [noktalar], False, renkler[i % 4], 2)
        # açıklama sol-altta: PR eğrileri genelde sağ-üstte yoğunlaştığı için orası boş kalır
        cv2.putText(img, isim, (sol + 12, alt - 12 - 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renkler[i % 4], 1)
    return img


@dataclass
class OyunAyarlari:
    kare_sayisi: int = ayar(50, "Kaç sentetik kare", min=5, max=500, adim=5)
    konum_gurultusu: float = ayar(0.05, "Tahmin kutularının ne kadar kaydığı (kutu boyuna oran)", min=0.0, max=0.4, adim=0.01,
                                  etki="Artırırsan mAP50-95 mAP50'den çok daha hızlı düşer (hassas konum cezalanır)")
    kacirma: float = ayar(0.1, "Gerçek nesneleri kaçırma oranı (FN)", min=0.0, max=0.9, adim=0.05,
                          etki="Artırırsan Recall düşer, PR eğrisi erken biter")
    yanlis_pozitif: float = ayar(0.5, "Kare başına ortalama hayali tespit (FP)", min=0.0, max=5.0, adim=0.1,
                                 etki="Artırırsan Precision düşer (özellikle düşük güven bölgesinde)")
    tohum: int = ayar(42, "Rastgelelik tohumu", min=0, max=9999, adim=1)


class MetrikOyunAlani(Eylem):
    ISIM = "Metrik oyun alanı (IoU / P / R / mAP)"
    BOLUM = "00"
    DOSYA = DOSYA
    ACIKLAMA = ("Sentetik 'gerçek' kutular ve 'kusurlu model' tahminleri üretir; sonra mAP'i **sıfırdan yazılmış** "
                "fonksiyonlarla hesaplar. Gürültü/kaçırma/yanlış pozitif ayarlarını değiştirip metriklerin nasıl tepki verdiğini gör.")
    Ayarlar = OyunAyarlari

    def calistir(self, ayarlar: OyunAyarlari, anlat):
        rng = np.random.default_rng(ayarlar.tohum)
        G, T = sentetik_veri(ayarlar.kare_sayisi, ayarlar.konum_gurultusu, ayarlar.kacirma, ayarlar.yanlis_pozitif, rng)
        anlat.adim(f"{len(G)} gerçek kutu, {len(T)} tahmin üretildi", kod=f"{DOSYA}:sentetik_veri")

        sonuc = map_hesapla(G, T)
        anlat.adim("Her sınıf ve 10 IoU eşiği için eşleştir → PR eğrisi → AP", kod=f"{DOSYA}:map_hesapla",
                   veri={"mAP50": sonuc["mAP50"], "mAP50_95": sonuc["mAP50_95"]})
        for s, v in sonuc["siniflar"].items():
            anlat.bilgi(f"{s}: AP50={v['AP50']:.3f} AP50-95={v['AP50_95']:.3f} P={v['P_son']:.2f} R={v['R_son']:.2f}",
                        kod=f"{DOSYA}:eslestir")

        egriler = {}
        for s in sonuc["siniflar"]:
            tp, _, n = eslestir(G, T, s, 0.5)
            egriler[s] = pr_egrisi(tp, n)              # (precision, recall)
        pr_img = pr_cizimi(egriler)
        anlat.adim("PR eğrileri çizildi (IoU=0.5)", kod=f"{DOSYA}:pr_cizimi")

        # Örnek kare: yeşil = gerçek, kırmızı = tahmin (IoU yazılı)
        ornek = np.full((480, 640, 3), 40, np.uint8)
        for g in [g for g in G if g["kare"] == 0]:
            kutu_ciz(ornek, g["kutu"], f"GT {g['sinif']}", (0, 200, 0))
        for t in [t for t in T if t["kare"] == 0]:
            en_iyi = max((iou(t["kutu"], g["kutu"]) for g in G if g["kare"] == 0), default=0)
            kutu_ciz(ornek, t["kutu"], f"{t['sinif']} {t['guven']:.2f} IoU={en_iyi:.2f}", (0, 0, 230))
        yazi_yaz(ornek, ["Yesil: gercek (GT)", "Kirmizi: tahmin + en iyi IoU"])

        tablo = "| sınıf | AP50 | AP50-95 | P | R | gerçek | tahmin |\n|---|---|---|---|---|---|---|\n"
        tablo += "\n".join(f"| {s} | {v['AP50']:.3f} | {v['AP50_95']:.3f} | {v['P_son']:.2f} | {v['R_son']:.2f} | {v['gercek']} | {v['tahmin']} |"
                           for s, v in sonuc["siniflar"].items())
        return Sonuc(
            goruntu=pr_img,
            ara_gorseller={"PR eğrileri (IoU 0.5)": pr_img, "Kare 0: GT vs tahmin": ornek},
            metin=f"**mAP50 = {sonuc['mAP50']:.3f}**, **mAP50-95 = {sonuc['mAP50_95']:.3f}**\n\n{tablo}",
            veri=sonuc,
        )


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(MetrikOyunAlani)
