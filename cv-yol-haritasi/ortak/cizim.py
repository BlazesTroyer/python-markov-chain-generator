"""
cizim.py — Görselleştirme yardımcıları (kutu, ID, maske, yazı, ızgara).

NEDEN VAR?
    "Ne oluyor?" sorusunun görsel cevabı. Bütün projeler aynı renk/ID düzenini kullanır,
    böylece bir projeden diğerine geçince gözün alışık olur.

KOORDİNAT KURALI (bütün projede aynı):
    kutu = (x1, y1, x2, y2) piksel, sol-üst ve sağ-alt köşe. (YOLO formatı ayrı; bkz. 01)
"""
from __future__ import annotations

import cv2
import numpy as np


def id_rengi(kimlik: int) -> tuple[int, int, int]:
    """Aynı ID her zaman aynı rengi alsın (takipte ID değişimi gözle hemen fark edilir)."""
    rng = np.random.default_rng(int(kimlik) * 9973 + 17)
    r, g, b = rng.integers(60, 255, size=3)
    return int(b), int(g), int(r)


def kutu_ciz(img, kutu, etiket: str = "", renk=(0, 220, 0), kalinlik: int = 2, kesikli: bool = False):
    """Bir kutu ve üstüne etiket çiz. kesikli=True: tahmin/hayalet kutular için (ör. Kalman)."""
    x1, y1, x2, y2 = map(int, kutu)
    if kesikli:
        for i in range(x1, x2, 10):
            cv2.line(img, (i, y1), (min(i + 5, x2), y1), renk, kalinlik)
            cv2.line(img, (i, y2), (min(i + 5, x2), y2), renk, kalinlik)
        for i in range(y1, y2, 10):
            cv2.line(img, (x1, i), (x1, min(i + 5, y2)), renk, kalinlik)
            cv2.line(img, (x2, i), (x2, min(i + 5, y2)), renk, kalinlik)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, kalinlik)
    if etiket:
        (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), renk, -1)
        cv2.putText(img, etiket, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def yazi_yaz(img, satirlar: list[str] | str, konum=(10, 24), renk=(255, 255, 255)):
    """Sol üstte yarı saydam zemin üzerine çok satırlı bilgi yazısı (FPS, sayaç...)."""
    if isinstance(satirlar, str):
        satirlar = [satirlar]
    x, y = konum
    genislik = max((cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for s in satirlar), default=0)
    katman = img.copy()
    cv2.rectangle(katman, (x - 6, y - 18), (x + genislik + 6, y + 22 * (len(satirlar) - 1) + 8), (0, 0, 0), -1)
    cv2.addWeighted(katman, 0.5, img, 0.5, 0, dst=img)
    for i, s in enumerate(satirlar):
        cv2.putText(img, s, (x, y + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 1, cv2.LINE_AA)
    return img


def maske_ortu(img, maske: np.ndarray, renk=(0, 0, 255), saydamlik: float = 0.45):
    """İkili maskeyi (Y,G) renkli yarı saydam örtü olarak bindir."""
    m = maske.astype(bool)
    katman = img.copy()
    katman[m] = renk
    return cv2.addWeighted(katman, saydamlik, img, 1 - saydamlik, 0)


def izgara(goruntuler: list[np.ndarray], basliklar: list[str] | None = None, sutun: int = 4, hucre: int = 220):
    """Birden çok görüntüyü başlıklı bir ızgarada birleştir (augmentation örnekleri vb.)."""
    if not goruntuler:
        return np.zeros((hucre, hucre, 3), np.uint8)
    kucukler = []
    for g in goruntuler:
        if g.ndim == 2:
            g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        oran = hucre / max(g.shape[:2])
        kucukler.append(cv2.resize(g, (int(g.shape[1] * oran), int(g.shape[0] * oran))))
    satir = (len(kucukler) + sutun - 1) // sutun
    sutun = min(sutun, len(kucukler))
    yukseklik = max(k.shape[0] for k in kucukler) + 24      # satır yüksekliği = en uzun hücre + başlık
    tuval = np.full((satir * yukseklik, sutun * hucre, 3), 30, np.uint8)
    for i, k in enumerate(kucukler):
        r, c = divmod(i, sutun)
        y0, x0 = r * yukseklik + 24, c * hucre
        tuval[y0:y0 + k.shape[0], x0:x0 + k.shape[1]] = k
        if basliklar and i < len(basliklar):
            cv2.putText(tuval, basliklar[i][:30], (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tuval


_TR = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")


def dosya_adi(metin: str) -> str:
    """Metni güvenli dosya adına çevir: Türkçe harfler -> ASCII, diğer işaretler -> '_'.

    NEDEN? cv2.imwrite/imread Windows'ta ASCII olmayan yollarda sessizce BAŞARISIZ olur.
    'şişe' -> 'sise', 'loss ve mAP eğrileri' -> 'loss_ve_mAP_egrileri'
    """
    temiz = "".join(c if (c.isascii() and c.isalnum()) or c in "-." else "_" for c in metin.translate(_TR))
    while "__" in temiz:
        temiz = temiz.replace("__", "_")
    return temiz.strip("_") or "dosya"


def bgr2rgb(img):
    """OpenCV (BGR) -> Gradio/PIL (RGB). Panel her görüntüyü göstermeden önce bunu çağırır."""
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2bgr(img):
    """Gradio/PIL (RGB) -> OpenCV (BGR). Panelden gelen görüntüler buradan geçer."""
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def egri_ciz(seriler: dict[str, list[float]], baslik: str = "", G: int = 560, Y: int = 300) -> np.ndarray:
    """Eğitim eğrileri (loss/accuracy) — matplotlib'siz basit çizgi grafik.

    seriler: {"train_loss": [...], "val_loss": [...]}  (hepsi aynı x ekseninde: epoch)
    """
    img = np.full((Y, G, 3), 255, np.uint8)
    sol, sag, ust, alt = 50, G - 130, 28, Y - 30
    cv2.rectangle(img, (sol, ust), (sag, alt), (0, 0, 0), 1)
    cv2.putText(img, baslik, (sol, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    degerler = [v for s in seriler.values() for v in s]
    if not degerler:
        return img
    en_az, en_cok = min(degerler), max(degerler)
    aralik = (en_cok - en_az) or 1.0
    n = max(len(s) for s in seriler.values())
    for k in range(5):
        v = en_az + aralik * k / 4
        yy = int(alt - (v - en_az) / aralik * (alt - ust))
        cv2.putText(img, f"{v:.2f}", (4, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    for i, (ad, s) in enumerate(seriler.items()):
        renk = id_rengi(i + 3)
        noktalar = [(int(sol + (j / max(n - 1, 1)) * (sag - sol)), int(alt - (v - en_az) / aralik * (alt - ust))) for j, v in enumerate(s)]
        for p1, p2 in zip(noktalar[:-1], noktalar[1:]):
            cv2.line(img, p1, p2, renk, 2)
        cv2.putText(img, ad, (sag + 8, ust + 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1, cv2.LINE_AA)
    cv2.putText(img, f"epoch (1..{n})", (sol + (sag - sol) // 2 - 30, Y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def matris_gorsel(matris: np.ndarray, etiketler: list[str], baslik: str = "satir=gercek, sutun=tahmin", hucre: int = 48) -> np.ndarray:
    """Karışıklık matrisi: satır = gerçek sınıf, sütun = tahmin. Hücrede adet; renk satır içi orana göre."""
    n = len(etiketler)
    sol, ust = 90, 70
    img = np.full((ust + n * hucre + 10, sol + n * hucre + 10, 3), 255, np.uint8)
    cv2.putText(img, baslik, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    satir_top = matris.sum(axis=1, keepdims=True).clip(min=1)
    for i in range(n):
        cv2.putText(img, etiketler[i][:10], (4, ust + i * hucre + hucre // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, etiketler[i][:6], (sol + i * hucre + 2, ust - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        for j in range(n):
            oran = matris[i, j] / satir_top[i, 0]
            renk = (int(255 - 200 * oran), int(255 - 120 * oran), 255) if i != j else (int(255 - 200 * oran), 255, int(255 - 200 * oran))
            x0, y0 = sol + j * hucre, ust + i * hucre
            cv2.rectangle(img, (x0, y0), (x0 + hucre - 2, y0 + hucre - 2), renk, -1)
            cv2.putText(img, str(int(matris[i, j])), (x0 + 6, y0 + hucre // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img
