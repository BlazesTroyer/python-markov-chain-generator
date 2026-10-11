"""
araclar.py — Kişisel agent'ın ARAÇLARI: önceki bölümlerin ürettiği olayları sorgulayan saf Python fonksiyonları.

TEK DOĞRULUK KAYNAĞI:
    Bu fonksiyonlar iki yerden çağrılır, kopyası YOKTUR:
        mcp_server.py : Claude Desktop / Claude Code gibi MCP istemcilerine araç olarak sunar
        asistan.py    : Kendi yazdığımız küçük agent (Anthropic tool runner) bunları çağırır
    Her fonksiyonun docstring'i, LLM'in aracı ne zaman kullanacağına karar verirken okuduğu AÇIKLAMADIR —
    bu yüzden net ve kısa yazıldı.

ARAÇ -> VERİ KAYNAĞI (hangi bölümün hangi olayını okur):
    where_is            -> 02 esya_hafizasi  ('nesne_goruldu')     -> nerede()
    who_was_here        -> 04 yuz_tanima      ('kisi_goruldu', 'tanimadik_yuz')
    screen_time_today   -> 03 ekran_suresi    ('telefon_kullanimi') -> gunluk_ozet()
    expenses            -> 08 fis_okuyucu     ('gider')             -> aylik_harcama()
    search_photos       -> 07 foto_hafiza     (vektör indeksi)      -> fotograf_ara()
    recent_events       -> hepsi              (olay deposu, filtreli)
    describe_now        -> 09 olay_ureteci'nin kaydettiği son kare + vlm_yorumla (Claude vision)

Hepsi JSON'a çevrilebilir dict döndürür; "cumle" alanı insan dilinde özet içerir.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak.olay_deposu import OlayDeposu  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

SON_KARE = VERI / "son_kare.jpg"   # olay_ureteci.py düzenli olarak buraya yazar


def where_is(nesne: str) -> dict:
    """Bir eşyanın (anahtar, cüzdan, telefon, kupa...) kamerada EN SON nerede ve ne zaman görüldüğünü söyler."""
    return modul_yukle("02_yolo_finetune/esya_hafizasi.py").nerede(nesne)


def who_was_here(son_saat: float = 24) -> dict:
    """Son N saatte kameranın tanıdığı kişileri (ve tanınmayan yüz sayısını) listeler."""
    bas = time.time() - son_saat * 3600
    depo = OlayDeposu()
    taninan = depo.sorgula(tur="kisi_goruldu", baslangic=bas, limit=1000)
    yabanci = depo.sorgula(tur="tanimadik_yuz", baslangic=bas, limit=1000)
    kisiler = {}
    for o in sorted(taninan, key=lambda o: o["zaman"]):
        kisiler.setdefault(o["nesne"], {"ilk": o["saat"], "son": o["saat"], "kez": 0})
        kisiler[o["nesne"]]["son"] = o["saat"]
        kisiler[o["nesne"]]["kez"] += 1
    cumle = (", ".join(f"{k} (son: {v['son']})" for k, v in kisiler.items()) or "Tanınan kimse görülmedi")
    cumle += f". Tanınmayan yüz olayı: {len(yabanci)}." if yabanci else "."
    return {"kisiler": kisiler, "tanimadik_yuz": len(yabanci), "cumle": f"Son {son_saat:g} saatte: {cumle}"}


def screen_time_today() -> dict:
    """Bugün telefonla geçen toplam süreyi, oturum sayısını ve en uzun oturumu verir."""
    oz = modul_yukle("03_takip/ekran_suresi.py").gunluk_ozet()
    oz["cumle"] = f"Bugün telefonda toplam {oz['toplam_dk']} dk ({oz['oturum']} oturum, en uzun {oz['en_uzun_dk']} dk)."
    return oz


def expenses(ay: str = "") -> dict:
    """Bir ayın (YYYY-AA; boşsa bu ay) fişlerden kaydedilmiş harcamalarını kategori bazında toplar."""
    r = modul_yukle("08_ocr/fis_okuyucu.py").aylik_harcama(ay or None)
    r["cumle"] = (f"{r['ay']}: {r['fis_sayisi']} fiş, toplam {r['genel_toplam']}. " +
                  ", ".join(f"{k} {v}" for k, v in r["kategoriler"].items()))
    return r


def search_photos(sorgu: str, adet: int = 5, indeks: str = "galerim_clip") -> dict:
    """Kişisel fotoğraf arşivinde doğal dille arar (CLIP). Sorguyu İngilizce yazmak daha iyi sonuç verir."""
    sonuc = modul_yukle("07_embedding/foto_hafiza/ara.py").fotograf_ara(sorgu, indeks, adet)
    return {"sonuclar": [{"yol": y, "benzerlik": round(s, 3)} for y, s in sonuc],
            "cumle": f"'{sorgu}' için en benzer {len(sonuc)} fotoğraf bulundu."}


def recent_events(tur: str = "", son_saat: float = 6, adet: int = 20) -> dict:
    """Son N saatteki olayları listeler; tur boşsa hepsi (örn. jest, kotu_durus_uyari, tekrar, cizgi_gecildi, duygu_degisti, sahne_aciklamasi)."""
    olaylar = OlayDeposu().sorgula(tur=tur or None, baslangic=time.time() - son_saat * 3600, limit=adet)
    sade = [{"saat": o["saat"], "tur": o["tur"], "nesne": o["nesne"], "ek": o["ek"]} for o in olaylar]
    sayim = Counter(o["tur"] for o in olaylar)
    return {"olaylar": sade, "cumle": f"Son {son_saat:g} saatte {len(sade)} olay: " + (", ".join(f"{t} ×{n}" for t, n in sayim.items()) or "yok")}


def describe_now(soru: str = "Odada şu an ne görüyorsun? Kısaca anlat.") -> dict:
    """Kameranın EN SON karesini Claude'a gösterip soruyu yanıtlatır (görüntü API'ye gönderilir)."""
    if not SON_KARE.exists():
        return {"cumle": "Henüz kaydedilmiş kare yok (olay_ureteci.py çalışıyor mu?)."}
    import cv2
    yas = time.time() - SON_KARE.stat().st_mtime
    cevap = modul_yukle("09_agent/vlm_yorumla.py").sahneyi_acikla(cv2.imread(str(SON_KARE)), soru)
    return {"kare_yasi_sn": round(yas), "cumle": cevap, "kare": str(SON_KARE)}


ARACLAR = [where_is, who_was_here, screen_time_today, expenses, search_photos, recent_events, describe_now]
