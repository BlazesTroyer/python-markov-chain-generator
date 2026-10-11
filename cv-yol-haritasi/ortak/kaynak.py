"""
kaynak.py — Görüntü kaynağını tek tip hale getirir.

NEDEN VAR?
    Aynı proje bazen webcam'den, bazen bir video dosyasından, bazen de bir klasör
    dolusu fotoğraftan beslenir. Her projede bunu ayrı ayrı yazmamak ve WEBCAM OLMADAN
    da inceleyebilmek için hepsini tek bir üretece (generator) çeviriyoruz.

KULLANIM:
    for no, kare in kareleri_oku("0"):            # webcam 0
    for no, kare in kareleri_oku("video.mp4"):    # video dosyası
    for no, kare in kareleri_oku("fotolar/"):     # klasördeki .jpg/.png'ler (alfabetik)
    for no, kare in kareleri_oku("foto.jpg"):     # tek görüntü

ÇIKTI: (kare_no, kare) — kare BGR renk düzeninde, numpy uint8 dizisi, şekli (Y, G, 3).
       Not: OpenCV BGR kullanır; Gradio/PIL/matplotlib RGB kullanır. Dönüşüm cizim.py'de.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def kareleri_oku(kaynak: str, en_fazla: int | None = None, atla: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Kaynaktan kare üret.

    kaynak   : "0", "1" (webcam numarası) | video yolu | görüntü yolu | klasör yolu
    en_fazla : en fazla kaç kare okunacak (None = hepsi)
    atla     : her 'atla' karede bir kare ver (2 -> yarı FPS; hız için)
    """
    yol = Path(kaynak)

    # 1) Klasör -> içindeki görüntüleri sırayla oku
    if yol.is_dir():
        dosyalar = sorted(p for p in yol.iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI)
        for i, p in enumerate(dosyalar[::atla]):
            if en_fazla is not None and i >= en_fazla:
                break
            kare = cv2.imread(str(p))
            if kare is not None:
                yield i, kare
        return

    # 2) Tek görüntü
    if yol.suffix.lower() in GORUNTU_UZANTILARI:
        kare = cv2.imread(str(yol))
        if kare is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {yol}")
        yield 0, kare
        return

    # 3) Webcam ("0") ya da video dosyası -> cv2.VideoCapture ikisini de açar
    cap = cv2.VideoCapture(int(kaynak) if kaynak.isdigit() else str(yol))
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {kaynak} (webcam takılı mı / dosya var mı?)")
    try:
        okunan, verilen = 0, 0
        while True:
            ok, kare = cap.read()
            if not ok:
                break  # video bitti ya da kamera koptu
            if okunan % atla == 0:
                yield verilen, kare
                verilen += 1
                if en_fazla is not None and verilen >= en_fazla:
                    break
            okunan += 1
    finally:
        cap.release()  # kamerayı/dosyayı mutlaka serbest bırak


def video_fps(kaynak: str, varsayilan: float = 30.0) -> float:
    """Video dosyasının FPS'ini oku (süre hesapları için). Okunamazsa varsayılanı döndür."""
    if Path(kaynak).is_dir() or Path(kaynak).suffix.lower() in GORUNTU_UZANTILARI:
        return varsayilan
    cap = cv2.VideoCapture(int(kaynak) if kaynak.isdigit() else kaynak)
    fps = cap.get(cv2.CAP_PROP_FPS) or varsayilan
    cap.release()
    return fps if fps > 1 else varsayilan


class VideoYazici:
    """İşlenmiş kareleri .mp4 olarak kaydeder (panelde video çıktısı, CLI'de --cikti)."""

    def __init__(self, yol: str | Path, fps: float = 20.0):
        self.yol = str(yol)
        self.fps = fps
        self._yazici = None

    def yaz(self, kare: np.ndarray) -> None:
        if self._yazici is None:  # boyutu ilk kareden öğren
            y, g = kare.shape[:2]
            self._yazici = cv2.VideoWriter(self.yol, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (g, y))
        self._yazici.write(kare)

    def kapat(self) -> None:
        if self._yazici is not None:
            self._yazici.release()
