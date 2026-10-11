"""
isler.py — Uzun süren işleri (model eğitimi, indeksleme, export) ayrı süreçte çalıştırır.

NEDEN AYRI SÜREÇ?
    Eğitim dakikalar/saatler sürebilir. Panelin içinde çalıştırırsak panel donar.
    Bunun yerine projeyi KOMUT SATIRINDAN çalıştırır gibi başlatırız:

        python 02_yolo_finetune/egit.py --epochs 50 --model yolo11n.pt ...

    Çıktısını bir log dosyasına yazdırır, paneldan o dosyanın sonunu okuruz.
    Böylece:
      - Panelde gördüğün komutu terminale yapıştırıp AYNISINI çalıştırabilirsin (şeffaflık).
      - Panel kapansa bile log dosyası kalır:  veri/isler/<zaman>_<isim>/log.txt
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from ortak.cizim import dosya_adi  # noqa: E402
from ortak.yukle import VERI  # noqa: E402
ISLER_KLASORU = VERI / "isler"


@dataclass
class Is:
    kimlik: str
    isim: str
    komut: list[str]
    klasor: Path
    surec: subprocess.Popen | None = field(default=None, repr=False)
    baslangic: float = field(default_factory=time.time)

    @property
    def log_yolu(self) -> Path:
        return self.klasor / "log.txt"

    @property
    def durum(self) -> str:
        if self.surec is None:
            return "bilinmiyor"
        kod = self.surec.poll()
        if kod is None:
            return "çalışıyor"
        return "bitti ✔" if kod == 0 else f"hata ✘ (çıkış kodu {kod})"

    def komut_metni(self) -> str:
        """Terminale yapıştırılabilir komut (cv-yol-haritasi/ klasöründen çalıştırılır)."""
        parcalar = ["python"] + [str(Path(self.komut[1]).relative_to(KOK))] + self.komut[2:]
        return " ".join(shlex.quote(p) for p in parcalar)

    def log_sonu(self, satir: int = 300) -> str:
        if not self.log_yolu.exists():
            return ""
        return "\n".join(self.log_yolu.read_text(encoding="utf-8", errors="replace").splitlines()[-satir:])


class IsYoneticisi:
    def __init__(self):
        self.isler: dict[str, Is] = {}

    def baslat(self, isim: str, dosya: str, argv: list[str]) -> Is:
        """dosya: KOK'a göre .py yolu; argv: ayarlardan üretilmiş argümanlar."""
        kimlik = time.strftime("%Y%m%d_%H%M%S") + "_" + dosya_adi(isim)[:30]   # ASCII: Windows yolları için
        klasor = ISLER_KLASORU / kimlik
        klasor.mkdir(parents=True, exist_ok=True)
        komut = [sys.executable, str(KOK / dosya), *argv, "--cikti-klasor", str(klasor), "--anlat"]
        log = open(klasor / "log.txt", "w", encoding="utf-8")
        ortam = dict(os.environ, PYTHONUNBUFFERED="1")  # log anında akabilsin
        surec = subprocess.Popen(komut, cwd=KOK, stdout=log, stderr=subprocess.STDOUT, env=ortam)
        is_ = Is(kimlik, isim, komut, klasor, surec)
        self.isler[kimlik] = is_
        return is_

    def listele(self) -> list[Is]:
        return sorted(self.isler.values(), key=lambda i: i.baslangic, reverse=True)

    def durdur(self, kimlik: str) -> None:
        is_ = self.isler.get(kimlik)
        if is_ and is_.surec and is_.surec.poll() is None:
            is_.surec.terminate()
