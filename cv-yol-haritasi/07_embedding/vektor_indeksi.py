"""
vektor_indeksi.py — En basit vektör veritabanı: numpy ile "en yakın k komşu" araması (+ isteğe bağlı FAISS).

VEKTÖR ARAMASI NEDİR?
    Elinde N tane D boyutlu (normalize) vektör var. Sorgu vektörü q için:
        benzerlik = V @ q        (N,) — normalize vektörlerde iç çarpım = kosinüs benzerliği
        en_iyi_k  = en büyük k benzerliğin indeksleri
    Bu kadar! 100.000 fotoğrafa kadar numpy milisaniyeler içinde yapar.
    FAISS (Facebook) aynı işi milyonlarca vektörde, yaklaşık (ANN) yöntemlerle çok daha hızlı yapar.
    faiss kuruluysa 'faiss_kullan=True' ile AYNI sonucu FAISS'in IndexFlatIP'si verir (karşılaştırma için).

DOSYA FORMATI (veri/indeksler/<isim>.npz):
    vektorler : (N, D) float32
    yollar    : (N,) dosya yolları
    zamanlar  : (N,) dosya değiştirilme zamanı (yeniden indekslemede değişmeyenleri atlamak için)
    gomucu    : hangi gömücüyle üretildi (farklı gömücülerin vektörleri KARIŞTIRILAMAZ)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.yukle import VERI  # noqa: E402


class VektorIndeksi:
    def __init__(self, gomucu: str, vektorler=None, yollar=None, zamanlar=None):
        self.gomucu = gomucu
        self.vektorler = np.zeros((0, 0), np.float32) if vektorler is None else np.asarray(vektorler, np.float32)
        self.yollar = [] if yollar is None else list(yollar)
        self.zamanlar = [] if zamanlar is None else list(zamanlar)

    def __len__(self):
        return len(self.yollar)

    def ekle(self, vektorler: np.ndarray, yollar: list[str], zamanlar: list[float]):
        """Yeni vektörleri ekle; aynı yol zaten varsa eskisinin yerine geçer."""
        if len(self) == 0:
            self.vektorler = np.asarray(vektorler, np.float32)
            self.yollar, self.zamanlar = list(yollar), list(zamanlar)
            return
        sira = {y: i for i, y in enumerate(self.yollar)}
        yeni_v, yeni_y, yeni_z = [], [], []
        for v, y, z in zip(vektorler, yollar, zamanlar):
            if y in sira:
                self.vektorler[sira[y]], self.zamanlar[sira[y]] = v, z
            else:
                yeni_v.append(v); yeni_y.append(y); yeni_z.append(z)
        if yeni_v:
            self.vektorler = np.vstack([self.vektorler, np.stack(yeni_v)])
            self.yollar += yeni_y
            self.zamanlar += yeni_z

    def ara(self, sorgu: np.ndarray, k: int = 10, faiss_kullan: bool = False) -> list[tuple[str, float]]:
        """Sorgu vektörüne en benzer k kayıt: [(yol, benzerlik), ...]"""
        if len(self) == 0:
            return []
        q = np.asarray(sorgu, np.float32).reshape(1, -1)
        k = min(k, len(self))
        if faiss_kullan:
            import faiss
            ind = faiss.IndexFlatIP(self.vektorler.shape[1])      # IP = inner product (iç çarpım)
            ind.add(self.vektorler)
            skor, idx = ind.search(q, k)
            return [(self.yollar[i], float(s)) for i, s in zip(idx[0], skor[0])]
        benz = self.vektorler @ q[0]                               # (N,)
        en_iyi = np.argsort(-benz)[:k]
        return [(self.yollar[i], float(benz[i])) for i in en_iyi]

    # ------------------------------------------------------------------ disk
    @staticmethod
    def yol(isim: str) -> Path:
        return VERI / "indeksler" / f"{isim}.npz"

    def kaydet(self, isim: str) -> Path:
        p = self.yol(isim)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, vektorler=self.vektorler, yollar=np.array(self.yollar), zamanlar=np.array(self.zamanlar), gomucu=self.gomucu)
        return p

    @classmethod
    def yukle(cls, isim: str) -> "VektorIndeksi":
        d = np.load(cls.yol(isim))
        return cls(str(d["gomucu"]), d["vektorler"], d["yollar"].tolist(), d["zamanlar"].tolist())
