"""
kalman.py — Bir kutunun hareketini tahmin eden Kalman filtresi, SIFIRDAN (sadece numpy).

KALMAN FİLTRESİ NE İŞE YARAR? (takipteki rolü)
    Tespit (YOLO) her karede "nesneler şurada" der ama:
      - bazen bir karede nesneyi kaçırır,
      - kutular titrer (gürültülü ölçüm),
      - hangi kutunun önceki karedeki hangi nesne olduğunu söylemez.
    Kalman, her nesne için "konum + HIZ" tutar. Böylece:
      1. TAHMİN: Yeni kare gelmeden "bu nesne şimdi muhtemelen şurada" der.
      2. GÜNCELLEME: Tespit gelince tahminle ölçümü, ikisinin GÜVENİLİRLİĞİNE göre harmanlar.
    Eşleştirme, tespitleri önceki kutuyla değil bu TAHMİNLE karşılaştırarak yapılır -> hızlı
    hareket eden nesnelerde ve kısa kaçırmalarda ID korunur.

DURUM VEKTÖRÜ (SORT makalesindeki gibi, 7 boyut):
    x = [cx, cy, s, r, vx, vy, vs]
        cx, cy : kutu merkezi            vx, vy : merkezin kare başına hızı
        s      : alan (genişlik*yükseklik) vs   : alanın değişim hızı (yaklaşma/uzaklaşma)
        r      : en/boy oranı (sabit kabul edilir, hızı yok)
ÖLÇÜM VEKTÖRÜ (tespitten gelen, 4 boyut):  z = [cx, cy, s, r]

MATRİSLER (her birinin anlamı):
    F (7x7) geçiş     : "sabit hız" modeli -> yeni_cx = cx + vx  (vb.)
    H (4x7) ölçüm     : durumdan ölçülebilen kısmı seçer (ilk 4 eleman)
    P (7x7) belirsizlik: Durum tahminimize ne kadar güvenmiyoruz. Büyük = emin değiliz.
    Q (7x7) süreç gürültüsü : Model ne kadar yanılabilir? (nesne aniden hızlanabilir)
    R (4x4) ölçüm gürültüsü : Dedektör kutuları ne kadar titrek?

DENKLEMLER:
    Tahmin:     x = F x               P = F P Fᵀ + Q
    Güncelleme: y = z - H x           (yenilik: ölçüm ile tahmin farkı)
                S = H P Hᵀ + R        (yeniliğin belirsizliği)
                K = P Hᵀ S⁻¹          (Kalman kazancı: ölçüme ne kadar güvenelim? 0..1 gibi düşün)
                x = x + K y           P = (I - K H) P

    Sezgi: R büyükse (dedektör titrek) K küçülür -> tahmine daha çok güvenilir, kutu yumuşar.
           Q büyükse (nesne öngörülemez) K büyür -> ölçüme daha çok güvenilir, kutu çabuk tepki verir.
"""
from __future__ import annotations

import numpy as np


def kutu_to_z(kutu) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx, cy, s, r] (4,1)"""
    x1, y1, x2, y2 = kutu[:4]
    w, h = x2 - x1, y2 - y1
    return np.array([[x1 + w / 2], [y1 + h / 2], [w * h], [w / max(h, 1e-6)]], dtype=float)


def x_to_kutu(x) -> np.ndarray:
    """[cx, cy, s, r, ...] -> [x1,y1,x2,y2]"""
    cx, cy, s, r = x[0, 0], x[1, 0], max(x[2, 0], 1e-6), x[3, 0]
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanKutu:
    """Tek bir nesnenin Kalman filtresi."""

    def __init__(self, kutu, surec_gurultusu: float = 1.0, olcum_gurultusu: float = 1.0):
        # F: sabit hız modeli. Satır 0: cx' = cx + vx ; satır 1: cy' = cy + vy ; satır 2: s' = s + vs
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        # H: ölçüm sadece ilk 4 durumu görür (hızları doğrudan ölçemeyiz)
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        # R: ölçüm gürültüsü. Alan ve oran ölçümleri konumdan daha gürültülü kabul edilir.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]) * olcum_gurultusu
        # P: başlangıç belirsizliği. Hızları hiç bilmiyoruz -> çok büyük.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        # Q: süreç gürültüsü. Hızlar küçük oynayabilir.
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4]) * surec_gurultusu
        self.x = np.zeros((7, 1))
        self.x[:4] = kutu_to_z(kutu)
        self.son_yenilik = np.zeros((4, 1))   # şeffaflık: son güncellemedeki y
        self.son_kazanc = None                # şeffaflık: son K

    def tahmin(self) -> np.ndarray:
        """Bir kare ileri: x = F x, P = F P Fᵀ + Q. Tahmini kutuyu döndürür."""
        if self.x[2, 0] + self.x[6, 0] <= 0:   # alan negatife düşmesin (nesne "küçülüp yok olmasın")
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_kutu(self.x)

    def guncelle(self, kutu) -> np.ndarray:
        """Ölçümle düzelt. Düzeltilmiş kutuyu döndürür."""
        z = kutu_to_z(kutu)
        y = z - self.H @ self.x                          # yenilik
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman kazancı (7x4)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.son_yenilik, self.son_kazanc = y, K
        return x_to_kutu(self.x)

    def kutu(self) -> np.ndarray:
        return x_to_kutu(self.x)

    def hiz(self) -> tuple[float, float]:
        """Merkezin kare başına hızı (vx, vy) — çizimde ok olarak gösterilir."""
        return float(self.x[4, 0]), float(self.x[5, 0])
