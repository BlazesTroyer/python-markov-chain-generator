"""
ergonomi_kocu.py — "30 dakikadır kambur oturuyorsun" diyen duruş koçu.

ÖLÇÜM (kameraya YANDAN otururken en doğru):
    boyun eğimi = dikeyle_aci(omuz, kulak)    : kulak omzun tam üstündeyse ≈ 0°, kafa öne uzadıkça büyür
    gövde eğimi = dikeyle_aci(kalça, omuz)    : dik oturuşta ≈ 0°, öne kambur eğildikçe büyür
    (Kalça görünmüyorsa — masada sık olur — sadece boyun kullanılır.)
    Eşikler (başlangıç önerisi, kendine göre ayarla): boyun < 40°, gövde < 15°  -> "iyi"

ZAMAN MANTIĞI:
    1. Her karenin kararı ("iyi"/"kotu") DurumMakinesi'nden geçer (kısa öne eğilmeler sayılmaz).
    2. Kararlı durum "kotu" iken kronometre işler.
    3. Kesintisiz kötü duruş `uyari_sn`'yi geçince olay: {"tur": "kotu_durus_uyari", "ek": {"sure_sn": ...}}
       Aynı kötü duruş sürerse her `uyari_sn`'de bir tekrar uyarır. İyi duruşa dönünce sıfırlanır.
    09_agent bu olayla sana bildirim gönderebilir.

fps=0 -> gerçek saat (canlı); fps>0 -> kayıtlı video (kare sayısından süre).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

nk = modul_yukle("05_el_poz/noktalar.py")
tel = modul_yukle("02_yolo_finetune/telefon_dedektoru_v2.py")
DOSYA = "05_el_poz/ergonomi_kocu.py"


@dataclass
class Ayarlar:
    boyun_esik: float = ayar(40.0, "Boyun eğimi bunun üstündeyse kötü (derece)", min=5, max=90, adim=1)
    govde_esik: float = ayar(15.0, "Gövde eğimi bunun üstündeyse kötü (derece)", min=3, max=60, adim=1)
    uyari_sn: float = ayar(1800.0, "Kesintisiz kötü duruş bu kadar sürerse uyar (sn). Deneme için 10 yap", min=5, max=7200)
    kararlilik: int = ayar(15, "Durum değişimi için art arda kare", min=1, max=120)
    fps: float = ayar(0.0, "0 = gerçek saat; >0 = kayıtlı videonun FPS'i", min=0.0, max=120.0, adim=1.0)


class ErgonomiKocu(KareIsleyici):
    ISIM = "Ergonomi / duruş koçu"
    BOLUM = "05"
    DOSYA = DOSYA
    ACIKLAMA = ("Yandan bakan kamerada boyun ve gövde eğimini ölçer; kararlı kötü duruş süresi `uyari_sn`'yi geçince "
                "`kotu_durus_uyari` olayı üretir.")
    GEREKSINIMLER = ("mediapipe",)
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.makine, self.kotu_baslangic, self.son_uyari, self.kare_no = tel.DurumMakinesi("iyi"), None, None, 0

    def isle(self, kare, a: Ayarlar, anlat):
        self.kare_no += 1
        simdi = self.kare_no / a.fps if a.fps > 0 else time.time()
        p = nk.poz_noktalari(kare)
        cikti, olaylar = kare.copy(), []
        if p is None:
            yazi_yaz(cikti, "kisi yok")
            return Sonuc(goruntu=cikti, metin="Kişi yok.")
        # kameraya yakın (görünürlüğü yüksek) taraf: sol (7 kulak, 11 omuz, 23 kalça) ya da sağ (8, 12, 24)
        kulak, omuz, kalca = (7, 11, 23) if p[[7, 11], 3].mean() >= p[[8, 12], 3].mean() else (8, 12, 24)
        boyun = nk.dikeyle_aci(p[omuz], p[kulak])
        govde = nk.dikeyle_aci(p[kalca], p[omuz]) if p[kalca, 3] > 0.5 else None
        kotu = boyun > a.boyun_esik or (govde is not None and govde > a.govde_esik)
        nk.iskelet_ciz(cikti, p, nk.POZ_BAGLANTILARI, vurgula=(kulak, omuz, kalca))
        degisti, _ = self.makine.guncelle("kotu" if kotu else "iyi", a.kararlilik)
        if degisti:
            self.kotu_baslangic = simdi if self.makine.durum == "kotu" else None
            self.son_uyari = None
        sure = simdi - self.kotu_baslangic if self.kotu_baslangic is not None else 0.0
        if self.makine.durum == "kotu" and sure >= a.uyari_sn and (self.son_uyari is None or simdi - self.son_uyari >= a.uyari_sn):
            self.son_uyari = simdi
            olaylar.append({"tur": "kotu_durus_uyari", "kaynak": DOSYA, "ek": {"sure_sn": round(sure), "boyun": round(boyun), "govde": govde}})
            anlat.adim(f"UYARI: {sure / 60:.1f} dk kesintisiz kötü duruş")
        anlat.adim(f"boyun={boyun:.0f}° (eşik {a.boyun_esik}), gövde={'—' if govde is None else f'{govde:.0f}°'} (eşik {a.govde_esik}) → "
                   f"kare: {'kötü' if kotu else 'iyi'}, kararlı: {self.makine.durum}, süre {sure:.0f} sn",
                   kod="05_el_poz/noktalar.py:dikeyle_aci")
        renk = (0, 0, 255) if self.makine.durum == "kotu" else (0, 220, 0)
        yazi_yaz(cikti, [f"DURUS: {self.makine.durum.upper()}", f"boyun {boyun:.0f}  govde {'-' if govde is None else f'{govde:.0f}'}",
                         f"kotu sure: {sure:.0f} sn"], renk=renk)
        return Sonuc(goruntu=cikti, olaylar=olaylar, veri={"boyun": round(boyun, 1), "govde": govde, "durum": self.makine.durum, "kotu_sure_sn": round(sure)},
                     metin=f"Duruş: **{self.makine.durum}** — boyun {boyun:.0f}°, kötü süre {sure:.0f} sn")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(ErgonomiKocu)
