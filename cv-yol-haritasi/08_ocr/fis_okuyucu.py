"""
fis_okuyucu.py — Fiş/fatura fotoğrafı -> {mağaza, tarih, toplam, kategori, kalemler} -> gider kaydı.

HAT:
    1. DÜZLEŞTİR : belge_duzlestir.py (kontur + homografi). Belge bulunamazsa görüntü olduğu gibi kullanılır.
    2. OCR       : EasyOCR (Türkçe + İngilizce). Her satır için: kutu, metin, güven.
                   OCR = (a) metin BÖLGELERİNİ bul (tespit)  (b) her bölgedeki karakterleri oku (tanıma).
    3. AYRIŞTIR  : iki mod:
        kural  : Düzenli ifadeler (regex). "TOPLAM/TUTAR/TOTAL" geçen satırdaki son tutar, ilk tarih
                 kalıbı, ilk satır = mağaza. Hızlı, ücretsiz, yerel — ama fiş formatı değişince kırılgan.
        claude : OCR metni + düzleştirilmiş görüntü Claude'a gönderilir; JSON ŞEMASINA uyan yanıt döner
                 (ortak/claude_istemci.py). Farklı formatlara dayanıklı, kategori de tahmin eder.
                 ⚠ Fiş görüntüsü bilgisayarından çıkar (Anthropic API'ye gider). ANTHROPIC_API_KEY gerekir.
    4. KAYDET    : Olay deposuna {"tur": "gider", "nesne": mağaza, "ek": {...}} (panelde 'Olayları depoya yaz').
                   aylik_harcama() bu kayıtlardan özet çıkarır; 09'daki agent'ın expenses() aracı bunu kullanır.

Her iki modun sonucu da log'da yan yana görülebilir (mod="ikisi"): kuralın nerede yanıldığını gör.
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak import claude_istemci as ci  # noqa: E402
from ortak.arayuz import Eylem, KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import kutu_ciz  # noqa: E402
from ortak.olay_deposu import OlayDeposu  # noqa: E402
from ortak.yukle import gerekli, modul_yukle  # noqa: E402

bd = modul_yukle("08_ocr/belge_duzlestir.py")
ek = modul_yukle("01_veri_fabrikasi/etiket_kontrol.py")
DOSYA = "08_ocr/fis_okuyucu.py"
KATEGORILER = ["market", "restoran", "kafe", "ulasim", "fatura", "saglik", "giyim", "elektronik", "diger"]

# Claude'dan istenen yanıtın JSON şeması (structured outputs: yanıt bu şemaya UYMAK ZORUNDA)
FIS_SEMASI = {
    "type": "object",
    "properties": {
        "magaza": {"type": "string", "description": "İşletme adı"},
        "tarih": {"type": "string", "description": "YYYY-AA-GG; okunamıyorsa boş"},
        "toplam": {"type": "number", "description": "Ödenen genel toplam"},
        "para_birimi": {"type": "string"},
        "kategori": {"type": "string", "enum": KATEGORILER},
        "kalemler": {"type": "array", "items": {"type": "object",
                                                "properties": {"ad": {"type": "string"}, "tutar": {"type": "number"}},
                                                "required": ["ad", "tutar"], "additionalProperties": False}},
        "emin_degilim": {"type": "string", "description": "Okunamayan/şüpheli alanlar; yoksa boş"},
    },
    "required": ["magaza", "tarih", "toplam", "para_birimi", "kategori", "kalemler", "emin_degilim"],
    "additionalProperties": False,
}

TUTAR = r"(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})"


def tutar_coz(metin: str) -> float:
    """'1.234,56' -> 1234.56 ; '1,234.56' -> 1234.56 ; '123,45' -> 123.45 (son ayraç = ondalık)."""
    t = metin.replace(" ", "")
    son = max(t.rfind(","), t.rfind("."))
    tam, ondalik = re.sub(r"[.,]", "", t[:son]), t[son + 1:]
    return float(f"{tam}.{ondalik}")


def kural_ayristir(satirlar: list[str]) -> dict:
    """Regex tabanlı ayrıştırıcı (saf fonksiyon — test edilir)."""
    sonuc = {"magaza": satirlar[0].strip() if satirlar else "", "tarih": "", "toplam": None, "para_birimi": "TRY",
             "kategori": "diger", "kalemler": [], "emin_degilim": ""}
    for s in satirlar:
        m = re.search(r"(\d{2})[./-](\d{2})[./-](\d{2,4})", s)
        if m:
            g, a, y = m.groups()
            sonuc["tarih"] = f"{'20' + y if len(y) == 2 else y}-{a}-{g}"
            break
    for i, s in enumerate(satirlar):
        if re.search(r"\b(GENEL\s*TOPLAM|TOPLAM|TOP|TUTAR|TOTAL)\b", s.upper()):
            adaylar = re.findall(TUTAR, s) or (re.findall(TUTAR, satirlar[i + 1]) if i + 1 < len(satirlar) else [])
            if adaylar:
                sonuc["toplam"] = tutar_coz(adaylar[-1])      # son eşleşen: "TOPLAM" satırları genelde en alttadır
    if sonuc["toplam"] is None:
        tum = [tutar_coz(t) for s in satirlar for t in re.findall(TUTAR, s)]
        sonuc["toplam"] = max(tum) if tum else 0.0
        sonuc["emin_degilim"] = "TOPLAM satırı bulunamadı; en büyük tutar alındı"
    return sonuc


def aylik_harcama(yil_ay: str | None = None, depo: OlayDeposu | None = None) -> dict:
    """'2026-10' ayının kategori bazında toplamı (09 agent expenses() bunu kullanır)."""
    depo = depo or OlayDeposu()
    yil_ay = yil_ay or time.strftime("%Y-%m")
    toplam, sayi = defaultdict(float), 0
    for o in depo.sorgula(tur="gider", limit=100000):
        tarih = o["ek"].get("tarih") or time.strftime("%Y-%m-%d", time.localtime(o["zaman"]))
        if tarih.startswith(yil_ay):
            toplam[o["ek"].get("kategori", "diger")] += float(o["ek"].get("toplam") or 0)
            sayi += 1
    return {"ay": yil_ay, "fis_sayisi": sayi, "kategoriler": {k: round(v, 2) for k, v in sorted(toplam.items())},
            "genel_toplam": round(sum(toplam.values()), 2)}


_OKUYUCU = {}


def ocr_oku(img: np.ndarray, diller: tuple = ("tr", "en")) -> list[dict]:
    """EasyOCR: [{'kutu': (4,2), 'metin': str, 'guven': float}] (yukarıdan aşağı sıralı)."""
    if diller not in _OKUYUCU:
        _OKUYUCU[diller] = gerekli("easyocr", "pip install easyocr").Reader(list(diller), gpu=False, verbose=False)
    sonuc = _OKUYUCU[diller].readtext(img)
    satirlar = [{"kutu": np.array(k, np.float32), "metin": m, "guven": float(g)} for k, m, g in sonuc]
    return sorted(satirlar, key=lambda s: (s["kutu"][:, 1].mean(), s["kutu"][:, 0].min()))


def satirlara_birlestir(parcalar: list[dict], tolerans: float = 0.6) -> list[str]:
    """OCR parçalarını aynı satırda olanları birleştirerek metin satırlarına çevir (fişte sol: ad, sağ: tutar)."""
    satirlar, simdiki, son_y, son_h = [], [], None, None
    for p in parcalar:
        y = p["kutu"][:, 1].mean()
        h = float(np.ptp(p["kutu"][:, 1]))          # parçanın yüksekliği
        if son_y is not None and abs(y - son_y) > tolerans * max(son_h, h, 1):
            satirlar.append(" ".join(x["metin"] for x in sorted(simdiki, key=lambda q: q["kutu"][:, 0].min())))
            simdiki = []
        simdiki.append(p)
        son_y, son_h = y, h
    if simdiki:
        satirlar.append(" ".join(x["metin"] for x in sorted(simdiki, key=lambda q: q["kutu"][:, 0].min())))
    return satirlar


@dataclass
class Ayarlar:
    mod: str = ayar("kural", "Ayrıştırma: kural (regex, yerel) | claude (LLM, görüntü API'ye gider) | ikisi (karşılaştır)",
                    secenekler=["kural", "claude", "ikisi"])
    duzlestir: bool = ayar(True, "Önce belgeyi bul ve düzleştir")
    min_guven: float = ayar(0.2, "Bu güvenin altındaki OCR parçalarını at", min=0.0, max=0.9, adim=0.05)


class FisOkuyucu(KareIsleyici):
    ISIM = "Fiş okuyucu → gider kaydı"
    BOLUM = "08"
    DOSYA = DOSYA
    ACIKLAMA = ("Fişi düzleştirir, EasyOCR ile okur, **regex kuralları** ya da **Claude (JSON şemalı)** ile mağaza/tarih/toplam/kategori "
                "çıkarır ve `gider` olayı üretir. 'Görüntü' sekmesiyle bir fiş fotoğrafı yükle; 'Olayları depoya yaz'ı aç.")
    GEREKSINIMLER = ("easyocr", "anthropic (claude modu)")
    Ayarlar = Ayarlar

    def isle(self, kare, a: Ayarlar, anlat):
        ara, img = {}, kare
        if a.duzlestir:
            koseler, _, yontem = bd.belge_bul(kare)
            if koseler is not None:
                img, _ = bd.duzlestir(kare, koseler)
                anlat.adim(f"Düzleştirildi ({yontem})", kod="08_ocr/belge_duzlestir.py:duzlestir")
            else:
                anlat.uyari("Belge kenarı bulunamadı — görüntü olduğu gibi okunacak")
        parcalar = [p for p in ocr_oku(img) if p["guven"] >= a.min_guven]
        satirlar = satirlara_birlestir(parcalar)
        anlat.adim(f"OCR: {len(parcalar)} parça → {len(satirlar)} satır", kod=f"{DOSYA}:ocr_oku / satirlara_birlestir",
                   veri={"ort_guven": round(float(np.mean([p['guven'] for p in parcalar])), 3) if parcalar else 0})
        ocr_img = img.copy()
        for p in parcalar:
            renk = (0, 200, 0) if p["guven"] > 0.6 else (0, 165, 255) if p["guven"] > 0.3 else (0, 0, 255)
            kutu_ciz(ocr_img, [*p["kutu"].min(0), *p["kutu"].max(0)], f"{p['guven']:.2f}", renk, 1)
        ara["OCR kutuları (yeşil=emin, kırmızı=şüpheli)"] = ocr_img
        metin = "\n".join(satirlar)

        sonuclar = {}
        if a.mod in ("kural", "ikisi"):
            sonuclar["kural"] = kural_ayristir(satirlar)
            anlat.adim(f"Kural sonucu: {sonuclar['kural']}", kod=f"{DOSYA}:kural_ayristir")
        if a.mod in ("claude", "ikisi"):
            sonuclar["claude"] = ci.sor(
                [ci.goruntu_blogu(img), {"type": "text", "text": f"Bu bir alışveriş fişi. OCR metni (hatalı olabilir):\n{metin}\n\n"
                                                                    "Görüntüyü esas alarak alanları çıkar. Tutarlar sayı olsun."}],
                sistem="Fişlerden harcama bilgisi çıkaran dikkatli bir asistansın. Emin olmadığın alanları 'emin_degilim'de belirt.",
                sema=FIS_SEMASI, anlat=anlat)
            anlat.adim(f"Claude sonucu: {sonuclar['claude']}", kod="ortak/claude_istemci.py:sor")
        secilen = sonuclar.get("claude") or sonuclar["kural"]
        olay = {"tur": "gider", "kaynak": DOSYA, "nesne": secilen["magaza"],
                "ek": {**secilen, "mod": "claude" if "claude" in sonuclar else "kural", "ocr_metni": metin[:2000]}}
        tablo = "| alan | " + " | ".join(sonuclar) + " |\n|---|" + "---|" * len(sonuclar) + "\n" + "\n".join(
            f"| {k} | " + " | ".join(str(s.get(k, ""))[:60] for s in sonuclar.values()) + " |"
            for k in ("magaza", "tarih", "toplam", "kategori", "emin_degilim"))
        return Sonuc(goruntu=img, ara_gorseller=ara, olaylar=[olay], veri={"sonuclar": sonuclar, "ocr_satirlari": satirlar},
                     metin=tablo + f"\n\n<details><summary>OCR metni</summary>\n\n```\n{metin}\n```\n</details>")


@dataclass
class RaporAyarlari:
    ay: str = ayar("", "YYYY-AA (boş = bu ay)")


class GiderRaporu(Eylem):
    ISIM = "Aylık gider raporu"
    BOLUM = "08"
    DOSYA = DOSYA
    ACIKLAMA = "Olay deposundaki `gider` kayıtlarından kategori bazında aylık toplam (09 agent'ın expenses() aracıyla aynı fonksiyon)."
    Ayarlar = RaporAyarlari

    def calistir(self, a: RaporAyarlari, anlat):
        r = aylik_harcama(a.ay or None)
        anlat.adim(f"{r['ay']}: {r['fis_sayisi']} fiş, toplam {r['genel_toplam']}", kod=f"{DOSYA}:aylik_harcama")
        grafik = ek.sutun_grafigi({k: int(v) for k, v in r["kategoriler"].items()} or {"-": 0}, f"{r['ay']} harcama (kategori)")
        return Sonuc(goruntu=grafik, veri=r, metin=f"**{r['ay']}**: {r['fis_sayisi']} fiş, toplam **{r['genel_toplam']}**\n\n" +
                     "\n".join(f"- {k}: {v}" for k, v in r["kategoriler"].items()))


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(FisOkuyucu)
