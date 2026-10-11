"""
arayuz.py — Her projenin uyduğu ortak SÖZLEŞME.

NEDEN VAR?
    Kullanıcı isteği: "Her şey şeffaf olsun + hepsine kontrol paneli olsun."
    Bunu tek bir kuralla çözüyoruz:

        Her proje ayarlarını bir dataclass'ta tanımlar (Ayarlar).
        Her alanın açıklaması, min/max değeri ve "artırırsan ne olur" notu alanın yanındadır.

    Sonra:
        - ortak/cli.py      bu dataclass'tan otomatik komut satırı argümanları üretir,
        - panel/otomatik_form.py aynı dataclass'tan otomatik slider/checkbox üretir.
    Yani paneldeki her kontrol, koddaki bir alana BİREBİR karşılık gelir. Gizli ayar yok.

İKİ TÜR PROJE VAR:

    KareIsleyici  -> Kare kare çalışır (tespit, takip, jest, yüz...).
                     isle(kare, ayarlar, anlat) -> Sonuc
                     Durum (state) tutabilir: ör. takipçi önceki karelerin izlerini hatırlar.

    Eylem         -> Bir kere çalışan iş (dataset böl, model eğit, indeksle, export...).
                     calistir(ayarlar, anlat) -> Sonuc

ÖRNEK:
    @dataclass
    class Ayarlar:
        esik: float = ayar(0.5, "Güven eşiği", min=0, max=1, adim=0.05,
                           etki="Artırırsan daha az ama daha emin tespit.")

    class Dedektor(KareIsleyici):
        ISIM = "Basit dedektör"
        Ayarlar = Ayarlar
        def isle(self, kare, ayarlar, anlat):
            anlat.adim("Kare alındı", kod="dosya.py:Dedektor.isle")
            return Sonuc(goruntu=kare)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .anlat import Anlatici


# ----------------------------------------------------------------------------- ayar alanı
def ayar(
    varsayilan: Any,
    aciklama: str,
    *,
    min: float | None = None,
    max: float | None = None,
    adim: float | None = None,
    secenekler: list[str] | None = None,
    tur: str | None = None,
    etki: str | None = None,
):
    """Açıklamalı bir ayar alanı tanımla.

    varsayilan : başlangıç değeri (tipi buradan anlaşılır: bool / int / float / str)
    aciklama   : "Bu ayar nedir?" — panelde etiket altında, CLI'de --help'te görünür
    min/max/adim : sayısal alanlar için slider sınırları
    secenekler : str alanı için açılır liste seçenekleri
    tur        : özel ipucu: "dosya" | "klasor" | "uzun_metin"
    etki       : "Artırırsan / azaltırsan ne olur?" notu (öğrenmenin asıl kısmı)
    """
    return field(
        default=varsayilan,
        metadata=dict(
            aciklama=aciklama, min=min, max=max, adim=adim,
            secenekler=secenekler, tur=tur, etki=etki,
        ),
    )


@dataclass
class AlanBilgisi:
    """Bir ayar alanının panel/CLI için çıkarılmış hali."""
    isim: str
    tip: type             # bool | int | float | str
    varsayilan: Any
    aciklama: str
    min: float | None
    max: float | None
    adim: float | None
    secenekler: list[str] | None
    tur: str | None
    etki: str | None

    def yardim_metni(self) -> str:
        """CLI --help ve panel tooltip'i için tek satırlık açıklama."""
        metin = self.aciklama
        if self.etki:
            metin += f" | Etki: {self.etki}"
        return metin


def ayar_alanlari(ayar_sinifi) -> list[AlanBilgisi]:
    """Bir Ayarlar dataclass'ını okuyup alan listesini döndür.

    Tip, varsayılan değerin tipinden alınır (from __future__ annotations'ta
    field.type bir string olduğu için bu daha güvenilir ve okunur).
    """
    alanlar = []
    for f in dataclasses.fields(ayar_sinifi):
        m = f.metadata or {}
        alanlar.append(
            AlanBilgisi(
                isim=f.name,
                tip=type(f.default),
                varsayilan=f.default,
                aciklama=m.get("aciklama", f.name),
                min=m.get("min"), max=m.get("max"), adim=m.get("adim"),
                secenekler=m.get("secenekler"), tur=m.get("tur"), etki=m.get("etki"),
            )
        )
    return alanlar


def ayarlari_olustur(ayar_sinifi, degerler: dict | list | tuple):
    """Panelden/CLI'den gelen ham değerleri doğru tiplere çevirip Ayarlar nesnesi yap.

    degerler bir sözlük {isim: değer} ya da alan sırasıyla bir liste olabilir
    (Gradio fonksiyonlara girdileri sırayla verir).
    """
    alanlar = ayar_alanlari(ayar_sinifi)
    if not isinstance(degerler, dict):
        degerler = {a.isim: v for a, v in zip(alanlar, degerler)}
    temiz = {}
    for a in alanlar:
        v = degerler.get(a.isim, a.varsayilan)
        if v is None:
            v = a.varsayilan
        if a.tip is bool:
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "evet", "yes")
        elif a.tip in (int, float, str):
            v = a.tip(v)
        temiz[a.isim] = v
    return ayar_sinifi(**temiz)


# ----------------------------------------------------------------------------- sonuç
@dataclass
class Sonuc:
    """Her projenin döndürdüğü tek tip sonuç.

    goruntu       : Ana çıktı görüntüsü (BGR, uint8) — kutular/ID'ler çizilmiş hali.
    ara_gorseller : {"isim": görüntü} — öğretici ara adımlar (maske, ızgara, Kalman tahmini...).
    metin         : İnsan için özet (Markdown). Panelde "Sonuç" kutusunda görünür.
    olaylar       : [{"tur": "phone_pickup", ...}] — olay deposuna / agent'a gidecek kayıtlar.
    veri          : Sayısal çıktılar (tablo/metrik) — panelde JSON olarak görünür.
    """
    goruntu: np.ndarray | None = None
    ara_gorseller: dict[str, np.ndarray] = field(default_factory=dict)
    metin: str = ""
    olaylar: list[dict] = field(default_factory=list)
    veri: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- proje tipleri
class _ProjeTabani:
    """Ortak üst bilgiler — panel sekme başlığı ve açıklaması buradan gelir."""
    ISIM: str = "isimsiz"
    BOLUM: str = "??"
    ACIKLAMA: str = ""       # Markdown: ne yapar, ne öğretir
    DOSYA: str = ""          # "03_takip/sort_sifirdan.py" — panelde "Kod nerede?" için
    Ayarlar: type = None     # dataclass sınıfı
    GEREKSINIMLER: tuple[str, ...] = ()  # ek pip paketleri (panelde gösterilir)


class KareIsleyici(_ProjeTabani):
    """Kare kare çalışan projeler için taban sınıf.

    Alt sınıf isle(...) yazar. Durum tutan projeler (takip, sayaç) sifirla()'yı da yazar.
    """

    def isle(self, kare: np.ndarray, ayarlar, anlat: Anlatici) -> Sonuc:  # pragma: no cover
        raise NotImplementedError

    def sifirla(self) -> None:
        """Yeni video/oturum başlarken iç durumu temizle (varsayılan: durum yok)."""


class Eylem(_ProjeTabani):
    """Bir kez çalışan iş (eğitim, dataset hazırlama, export...).

    UZUN_SURER=True ise panel bunu ayrı bir süreçte (subprocess) çalıştırır ve
    log'u canlı akıtır; ayrıca "bu, şu komutla aynıdır: python ..." satırını gösterir.
    """
    UZUN_SURER: bool = False

    def calistir(self, ayarlar, anlat: Anlatici) -> Sonuc:  # pragma: no cover
        raise NotImplementedError
