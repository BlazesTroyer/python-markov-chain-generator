"""
yukle.py — İki küçük ama önemli yardımcı.

1) modul_yukle("03_takip/sort_sifirdan.py")
   Klasör adları rakamla başladığı için (00_temeller, 03_takip...) Python'un normal
   `import` komutu bunları paket olarak göremez. Bu fonksiyon dosyayı yolundan yükler.
   Böylece 09_agent, 03_takip'teki SORT'u AYNEN kullanabilir — kod kopyalanmaz.

2) gerekli("ultralytics", "pip install ultralytics")
   Ağır/opsiyonel bağımlılıklar (YOLO, MediaPipe, SAM 2, InsightFace...) dosyanın en
   üstünde değil, kullanıldığı fonksiyonun içinde istenir. Yüklü değilse ne
   kurman gerektiğini söyleyen açık bir hata verir. Böylece panel, bir bölümün paketi
   eksik olsa bile açılır; sadece o sekme "şunu kur" der.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]   # .../cv-yol-haritasi


def kok_yolu_ekle() -> None:
    """`import ortak` her yerden çalışsın diye proje kökünü sys.path'e ekle."""
    if str(KOK) not in sys.path:
        sys.path.insert(0, str(KOK))


def modul_yukle(goreli_yol: str):
    """KOK'a göre verilen .py dosyasını modül olarak yükle (aynı dosya bir kez yüklenir)."""
    yol = (KOK / goreli_yol).resolve()
    isim = "cvyh_" + goreli_yol.replace("/", "_").replace(".py", "")
    if isim in sys.modules:
        return sys.modules[isim]
    spec = importlib.util.spec_from_file_location(isim, yol)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[isim] = modul          # dataclass'ların doğru çalışması için önce kaydet
    spec.loader.exec_module(modul)
    return modul


class EksikBagimlilik(ImportError):
    """Panel bu hatayı yakalayıp sekmede 'şunu kur' mesajı gösterir."""


def gerekli(paket: str, kurulum: str | None = None):
    """Opsiyonel bir paketi içe aktar; yoksa ne kurulacağını söyleyen hata ver."""
    try:
        return importlib.import_module(paket)
    except ImportError as e:
        kok_paket = paket.split(".")[0]
        raise EksikBagimlilik(
            f"Bu adım için '{kok_paket}' paketi gerekli.\n"
            f"Kurmak için:  {kurulum or 'pip install ' + kok_paket}\n(Orijinal hata: {e})"
        ) from e
