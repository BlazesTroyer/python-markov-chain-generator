"""
tani.py — "Tanıdık yüz karşılayıcı": karedeki yüzleri kayıtlı kişilerle karşılaştırır.

AKIŞ (her kare):
    1. YuNet ile yüzleri bul                         (yuz_ortak.yuzleri_bul)
    2. Her yüzü hizala + embedding çıkar             (yuz_ortak.hizala / embedding)
    3. Veritabanındaki HER vektörle kosinüs benzerliği; her kişi için en yüksek değeri al
    4. En iyi kişi benzerliği >= esik ise o kişi, değilse "tanınmadı"
    5. Olay (bekleme süreli — aynı kişi her karede değil, `bekleme_sn`'de bir):
         {"tur": "kisi_goruldu", "nesne": "Uğur", "guven": 0.62}
         {"tur": "tanimadik_yuz"}  -> agent "kapıda tanımadığım biri var" diyebilir
    09_agent/mcp_server.py'deki who_was_here() bu olayları okur.

Ara görselde her yüz için tüm kişilerin benzerlik çubukları görünür: kararın "ne kadar kıl payı"
olduğunu görürsün (0.37 vs eşik 0.36 = şüpheli; 0.70 = net).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

yo = modul_yukle("04_yuz/yuz_ortak.py")
vt = modul_yukle("04_yuz/yuz_tanima/veritabani.py")
DOSYA = "04_yuz/yuz_tanima/tani.py"


@dataclass
class Ayarlar:
    esik: float = ayar(0.363, "Bu benzerliğin üstü 'aynı kişi' (SFace önerisi 0.363; kendi eşik analizine bak)", min=0.1, max=0.9, adim=0.005,
                       etki="Artırırsan yabancıyı tanıdık sanma azalır ama seni tanımama artar")
    guven: float = ayar(0.8, "YuNet yüz güven eşiği", min=0.3, max=0.99, adim=0.01)
    bekleme_sn: float = ayar(60.0, "Aynı kişi için iki olay arası en az süre", min=1, max=3600)


def benzerlik_cubuklari(skorlar: dict[str, float], esik: float, G=320) -> np.ndarray:
    """Her kişinin benzerliği yatay çubuk; eşik dikey çizgi."""
    Y = 30 + 24 * max(len(skorlar), 1)
    img = np.full((Y, G, 3), 255, np.uint8)
    olcek = lambda v: int(100 + max(v, 0) * (G - 120))  # noqa: E731
    for i, (ad, s) in enumerate(sorted(skorlar.items(), key=lambda x: -x[1])):
        yy = 20 + i * 24
        cv2.putText(img, ad[:10], (4, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.rectangle(img, (100, yy), (olcek(s), yy + 16), (60, 170, 60) if s >= esik else (60, 60, 220), -1)
        cv2.putText(img, f"{s:.2f}", (olcek(s) + 4, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.line(img, (olcek(esik), 10), (olcek(esik), Y - 4), (0, 0, 0), 2)
    return img


class YuzTani(KareIsleyici):
    ISIM = "Tanıdık yüz karşılayıcı (tanıma)"
    BOLUM = "04"
    DOSYA = DOSYA
    ACIKLAMA = ("Yüzleri kayıtlı kişilerle karşılaştırır (YuNet → hizalama → SFace embedding → kosinüs). Ara görselde her kişinin "
                "benzerlik çubuğu ve eşik. Olaylar: `kisi_goruldu`, `tanimadik_yuz` (bekleme süreli).")
    Ayarlar = Ayarlar

    def __init__(self):
        self.db = None
        self.son_olay: dict[str, float] = {}

    def sifirla(self):
        self.db, self.son_olay = None, {}

    def isle(self, kare, a: Ayarlar, anlat):
        if self.db is None:
            if not vt.DB_YOLU.exists():
                raise FileNotFoundError(f"{vt.DB_YOLU} yok — önce 'Yüz veritabanı oluştur' çalıştır")
            d = np.load(vt.DB_YOLU)
            self.db = (d["isimler"], d["vektorler"])
            anlat.bilgi(f"Veritabanı yüklendi: {len(self.db[0])} vektör", kod=f"{DOSYA}:YuzTani.isle")
        isimler, vektorler = self.db
        yuzler = yo.yuzleri_bul(kare, a.guven)
        anlat.adim(f"{len(yuzler)} yüz bulundu", kod="04_yuz/yuz_ortak.py:yuzleri_bul")
        cikti, ara, olaylar, simdi, sonuc = kare.copy(), {}, [], time.time(), []
        for n, y in enumerate(yuzler):
            v = yo.embedding(yo.hizala(kare, y))
            benz = vektorler @ v                                   # (N,) tüm kayıtlara benzerlik
            kisi_skor = {k: float(benz[isimler == k].max()) for k in np.unique(isimler)}
            en_iyi = max(kisi_skor, key=kisi_skor.get)
            karar = en_iyi if kisi_skor[en_iyi] >= a.esik else "tanınmadı"
            anlat.adim(f"Yüz {n}: en yakın '{en_iyi}' = {kisi_skor[en_iyi]:.3f} (eşik {a.esik}) → {karar}",
                       kod="04_yuz/yuz_ortak.py:embedding → vektorler @ v", veri={k: round(s, 3) for k, s in kisi_skor.items()})
            renk = (0, 200, 0) if karar != "tanınmadı" else (0, 0, 255)
            kutu_ciz(cikti, yo.kutu_xyxy(y), f"{karar} {kisi_skor[en_iyi]:.2f}", renk)
            yo.noktalari_ciz(cikti, y)
            ara[f"yüz {n}: benzerlikler"] = benzerlik_cubuklari(kisi_skor, a.esik)
            anahtar = karar if karar != "tanınmadı" else "_yabanci"
            if simdi - self.son_olay.get(anahtar, 0) >= a.bekleme_sn:
                self.son_olay[anahtar] = simdi
                olaylar.append({"tur": "kisi_goruldu", "kaynak": DOSYA, "nesne": karar, "guven": kisi_skor[en_iyi]}
                               if karar != "tanınmadı" else
                               {"tur": "tanimadik_yuz", "kaynak": DOSYA, "guven": kisi_skor[en_iyi], "ek": {"en_yakin": en_iyi}})
            sonuc.append((karar, round(kisi_skor[en_iyi], 3)))
        yazi_yaz(cikti, [f"{len(yuzler)} yuz", ", ".join(k for k, _ in sonuc)[:60]])
        return Sonuc(goruntu=cikti, ara_gorseller=ara, olaylar=olaylar, veri={"yuzler": sonuc},
                     metin=", ".join(f"**{k}** ({s})" for k, s in sonuc) or "Yüz yok.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(YuzTani)
