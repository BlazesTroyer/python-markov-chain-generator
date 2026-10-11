"""
olay_ureteci.py — Agent'ın "her zaman açık gözü": kamera karelerini ANLAMLI OLAYLARA çevirir.

BÖLÜMLERİN BİRLEŞTİĞİ YER (her kare, sırayla):
    1. HAREKET KAPISI (00)   : Sahne durgunsa ve takip edilen kimse yoksa pahalı adımları ATLA.
                               (Log'da "atlanan kare oranı" = kazanılan hesaplama / enerji.)
    2. TESPİT (02)           : Tek bir YOLO çıkarımı.
    3. TAKİP (03)            : 'person' tespitleri SORT'a -> kişi ID'leri.
                               Yeni onaylanan iz  -> {"tur": "kisi_girdi"}
                               Silinen (onaylı) iz -> {"tur": "kisi_cikti", "ek": {"sure_sn": ...}}
    4. EŞYA HAFIZASI (02)    : Aynı tespitlerden eşya kayıtları (EsyaHafizasi.tespitleri_isle) -> 'nesne_goruldu'
    5. VLM (09, opsiyonel)   : Yeni kişi girdiğinde (bekleme süreli) kareyi Claude'a sor -> 'sahne_aciklamasi'
    6. SON KARE              : Düzenli olarak veri/son_kare.jpg'ye yazılır -> agent'ın describe_now() aracı okur.

Bu olaylar olay deposuna yazılınca (panelde 'Olayları depoya yaz' / CLI'de --olaylari-kaydet),
mcp_server.py ve asistan.py onları sorgulayabilir. Yani: KAMERA -> OLAY -> HAFIZA -> AGENT.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

hk = modul_yukle("00_temeller/hareket_kapisi.py")
yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
eh = modul_yukle("02_yolo_finetune/esya_hafizasi.py")
st = modul_yukle("03_takip/sort_sifirdan.py")
to = modul_yukle("03_takip/takip_ortak.py")
ar = modul_yukle("09_agent/araclar.py")
DOSYA = "09_agent/olay_ureteci.py"


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n.pt", "YOLO modeli (kendi eşya modelin de olabilir; kişiler için 'person' sınıfı gerekir)", tur="dosya")
    guven: float = ayar(0.4, "Tespit güven eşiği", min=0.05, max=0.95, adim=0.05)
    hareket_kapisi: bool = ayar(True, "Durgun sahnede tespiti atla (enerji/hesap tasarrufu)")
    esya_esleme: str = ayar("cell phone=telefon, cup=kupa, book=kitap, remote=kumanda, bottle=şişe",
                            "Eşya hafızası için sınıf=ad eşlemesi (02 ile aynı biçim)", tur="uzun_metin")
    esya_tekrar_sn: float = ayar(60.0, "Aynı eşya en fazla bu sürede bir kaydedilsin", min=1, max=3600)
    vlm: bool = ayar(False, "Yeni kişi girince kareyi Claude'a yorumlat (⚠ görüntü API'ye gider, ücretli)")
    vlm_bekleme_sn: float = ayar(300.0, "İki VLM çağrısı arası en az süre", min=10, max=86400)
    kare_kaydet_sn: float = ayar(5.0, "Son kareyi (describe_now için) kaç saniyede bir kaydet", min=0.5, max=600)


class OlayUreteci(KareIsleyici):
    ISIM = "Olay üretici (kamera → olay → hafıza)"
    BOLUM = "09"
    DOSYA = DOSYA
    ACIKLAMA = ("Hareket kapısı + YOLO + SORT + eşya hafızası (+ opsiyonel VLM) tek hatta: kişi girdi/çıktı, eşya görüldü, "
                "sahne açıklaması olayları üretir ve son kareyi agent için kaydeder. 'Olayları depoya yaz'ı aç.")
    GEREKSINIMLER = ("ultralytics", "anthropic (vlm)")
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.kapi, self.sort, self.esya = hk.HareketKapisi(), st.Sort(max_yas=30, min_vurus=3, iou_esigi=0.3), eh.EsyaHafizasi()
        self.onayli: dict[int, float] = {}       # kişi ID -> giriş zamanı
        self.son_vlm, self.son_kare_kayit, self.kare_sayisi, self.atlanan = 0.0, 0.0, 0, 0

    def isle(self, kare, a: Ayarlar, anlat):
        self.kare_sayisi += 1
        simdi, olaylar = time.time(), []
        if simdi - self.son_kare_kayit >= a.kare_kaydet_sn:
            ar.SON_KARE.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ar.SON_KARE), kare)
            self.son_kare_kayit = simdi

        # 1. Hareket kapısı: sahne durgun VE içeride takip edilen kimse yoksa ağır adımları atla
        if a.hareket_kapisi:
            hareket = self.kapi.isle(kare, hk.Ayarlar(), anlat).veri.get("hareket", True)
            if not hareket and not self.onayli:
                self.atlanan += 1
                cikti = yazi_yaz(kare.copy(), [f"durgun sahne - tespit atlandi", f"atlanan: %{100 * self.atlanan / self.kare_sayisi:.0f}"])
                anlat.adim("Durgun sahne, takip edilen kişi yok → tespit ATLANDI", kod="00_temeller/hareket_kapisi.py")
                return Sonuc(goruntu=cikti, veri={"atlanan_orani": self.atlanan / self.kare_sayisi})

        # 2. Tek YOLO çıkarımı
        model = yy.model_yukle(a.model)
        r = model.predict(kare, conf=a.guven, verbose=False)[0]
        kutular, siniflar, guvenler, isimler = yy.sonucu_ayikla(r)
        kisi_no = next((k for k, v in isimler.items() if v == "person"), -1)
        kisi = siniflar == kisi_no
        anlat.adim(f"YOLO: {len(kutular)} tespit ({int(kisi.sum())} kişi)", kod="ultralytics:YOLO.predict")

        # 3. Kişi takibi -> giriş/çıkış olayları
        izler = self.sort.guncelle(np.hstack([kutular[kisi], guvenler[kisi, None], siniflar[kisi, None]]).astype(np.float32))
        for iz in izler:
            if iz["id"] not in self.onayli:
                self.onayli[iz["id"]] = simdi
                olaylar.append({"tur": "kisi_girdi", "kaynak": DOSYA, "nesne": "kişi", "kimlik": iz["id"], "kutu": iz["kutu"].tolist()})
                anlat.adim(f"Kişi {iz['id']} GİRDİ", kod="03_takip/sort_sifirdan.py:Sort.guncelle")
        for kid in self.sort.son_adim["silinen"]:
            if kid in self.onayli:
                sure = simdi - self.onayli.pop(kid)
                olaylar.append({"tur": "kisi_cikti", "kaynak": DOSYA, "nesne": "kişi", "kimlik": kid, "ek": {"sure_sn": round(sure)}})
                anlat.adim(f"Kişi {kid} ÇIKTI ({sure:.0f} sn kaldı)")

        # 4. Eşya hafızası (aynı tespitler)
        cikti = to.izleri_ciz(kare.copy(), izler, isimler)
        esya_ayar = eh.Ayarlar(model=a.model, guven=a.guven, isim_esleme=a.esya_esleme, tekrar_sn=a.esya_tekrar_sn)
        esya = self.esya.tespitleri_isle(kare, kutular[~kisi], siniflar[~kisi], guvenler[~kisi], isimler, esya_ayar, anlat, cikti=cikti)
        olaylar += esya.olaylar

        # 5. VLM (opsiyonel, bekleme süreli)
        if a.vlm and any(o["tur"] == "kisi_girdi" for o in olaylar) and simdi - self.son_vlm >= a.vlm_bekleme_sn:
            self.son_vlm = simdi
            cevap = modul_yukle("09_agent/vlm_yorumla.py").sahneyi_acikla(kare, "Biri odaya girdi. Ne yapıyor? Kısaca.", anlat=anlat)
            olaylar.append({"tur": "sahne_aciklamasi", "kaynak": DOSYA, "ek": {"cevap": cevap}})

        yazi_yaz(cikti, [f"iceride: {len(self.onayli)} kisi", f"bu kare: {len(olaylar)} olay",
                         f"atlanan kare: %{100 * self.atlanan / self.kare_sayisi:.0f}"], konum=(10, 70))   # eşya yazısının altına
        return Sonuc(goruntu=cikti, olaylar=olaylar,
                     veri={"iceride": list(self.onayli), "olaylar": [o["tur"] for o in olaylar], "atlanan_orani": self.atlanan / self.kare_sayisi},
                     metin=f"İçeride {len(self.onayli)} kişi; bu karede olaylar: {', '.join(o['tur'] for o in olaylar) or '—'}")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(OlayUreteci)
