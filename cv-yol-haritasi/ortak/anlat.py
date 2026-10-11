"""
anlat.py — "Şu an kod nerede, ne yapıyor?" sorusunun cevabını toplayan küçük logger.

NEDEN VAR?
    Görüntü işleme kodunda en çok kaybolunan yer "bu kare için hangi adımlar çalıştı,
    hangi sayılar çıktı?" sorusudur. Her proje adımlarını buraya yazar:

        anlat.adim("Kalman tahmini yapıldı", kod="03_takip/kalman.py:KalmanKutu.tahmin",
                   veri={"iz_sayisi": 3})

    - Komut satırında `--anlat` verirsen satırlar anında konsola basılır.
    - Kontrol panelinde "Ne oluyor?" kutusuna aynı satırlar düşer.
    - "Kod nerede?" satırı `kod=` parametresinden toplanır (hangi dosya:fonksiyon çalıştı).

GİRDİ : adım mesajları
ÇIKTI : metin() -> tüm satırlar, kod_yerleri() -> çalışan dosya:fonksiyon listesi

KİM KULLANIR? Bütün projeler (isle(...) fonksiyonlarına parametre olarak gelir).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _Satir:
    zaman: float           # time.perf_counter() — adımlar arası süreyi göstermek için
    seviye: str            # "ADIM" | "BILGI" | "UYARI"
    mesaj: str
    kod: str | None        # "dosya.py:Fonksiyon" — panelde "Kod nerede?" satırı buradan gelir
    veri: dict | None      # küçük sayısal ayrıntılar (ör. {"tespit": 4})


@dataclass
class Anlatici:
    """Adım adım açıklama toplayıcı.

    konsola_bas=True  -> her satırı anında print eder (CLI'de --anlat).
    konsola_bas=False -> sessizce toplar (panel kutuya kendisi yazar).
    """

    konsola_bas: bool = False
    satirlar: list[_Satir] = field(default_factory=list)
    _adim_no: int = 0
    _baslangic: float = field(default_factory=time.perf_counter)

    # ------------------------------------------------------------------ yazma
    def adim(self, mesaj: str, kod: str | None = None, veri: dict | None = None) -> None:
        """Numaralı bir işlem adımı kaydet (ör. 'Tespit yapıldı')."""
        self._adim_no += 1
        self._ekle("ADIM", f"[{self._adim_no}] {mesaj}", kod, veri)

    def bilgi(self, mesaj: str, kod: str | None = None, veri: dict | None = None) -> None:
        """Numarasız ek bilgi (ör. bir eşiğin neden seçildiği)."""
        self._ekle("BILGI", mesaj, kod, veri)

    def uyari(self, mesaj: str, kod: str | None = None, veri: dict | None = None) -> None:
        """Bir şey beklenmedik gittiğinde (ör. model bulunamadı, kare boş)."""
        self._ekle("UYARI", mesaj, kod, veri)

    def _ekle(self, seviye, mesaj, kod, veri):
        satir = _Satir(time.perf_counter(), seviye, mesaj, kod, veri)
        self.satirlar.append(satir)
        if self.konsola_bas:
            print(self._bicimle(satir), flush=True)

    # ------------------------------------------------------------------ okuma
    def _bicimle(self, s: _Satir) -> str:
        ms = (s.zaman - self._baslangic) * 1000
        on_ek = {"ADIM": "  ", "BILGI": "  · ", "UYARI": "  ⚠ "}[s.seviye]
        metin = f"{ms:8.1f} ms {on_ek}{s.mesaj}"
        if s.veri:
            metin += "  " + ", ".join(f"{k}={_kisa(v)}" for k, v in s.veri.items())
        if s.kod:
            metin += f"   ← {s.kod}"
        return metin

    def metin(self) -> str:
        """Bütün satırları tek metin olarak döndür (panelin 'Ne oluyor?' kutusu)."""
        return "\n".join(self._bicimle(s) for s in self.satirlar)

    def kod_yerleri(self) -> list[str]:
        """Bu çalıştırmada dokunulan dosya:fonksiyon listesi (sırayla, tekrarsız)."""
        gorulen: list[str] = []
        for s in self.satirlar:
            if s.kod and s.kod not in gorulen:
                gorulen.append(s.kod)
        return gorulen

    def sifirla(self) -> None:
        """Yeni kare/çalıştırma için temizle (zaman sayacı da sıfırlanır)."""
        self.satirlar.clear()
        self._adim_no = 0
        self._baslangic = time.perf_counter()


def _kisa(v):
    """Uzun listeleri/float'ları log satırında okunur tut."""
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)) and len(v) > 6:
        return f"[{', '.join(map(str, v[:6]))}, …({len(v)})]"
    return v
