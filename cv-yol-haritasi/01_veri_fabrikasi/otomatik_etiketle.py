"""
otomatik_etiketle.py — Açık kelime dağarcıklı (open-vocabulary) YOLO-World ile ÖN-etiketleme.

FİKİR:
    Yüzlerce kareyi elle kutulamak saatler sürer. YOLO-World, eğitim yapmadan METİNLE
    verilen sınıfları ("keys", "wallet", "mug") bulabilir. Bu dosya onu bir "yardımcı
    etiketçi" olarak kullanır:
        1. Model ön-etiketleri yazar           (bu dosya)
        2. Sen hatalılarını düzeltirsin          (CVAT / Label Studio / Roboflow — "YOLO" formatında içe/dışa aktar)
        3. Düzeltilmiş veriyle KENDİ küçük, hızlı modelini eğitirsin (02)
    Neden doğrudan YOLO-World kullanmıyoruz? Daha yavaş ve senin özel eşyalarında
    (SENİN anahtarın, SENİN kupan) fine-tune edilmiş küçük bir modelden daha az isabetli.

SINIF YAZIMI:  "anahtar=keys, cuzdan=wallet, kupa=mug"
    Sol: dataset'te görünecek ad (Türkçe olabilir). Sağ: modele verilen İngilizce istem.
    İstem kalitesi çok önemli: "keychain", "set of keys", "car key" farklı sonuç verir — dene!

GÜVENLİK AĞI:
    uzerine_yaz=False iken etiket dosyası OLAN görüntülere dokunulmaz. Yani elle
    düzelttiğin etiketleri yanlışlıkla ezmezsin.

ÇIKTI:
    veri/ham/<isim>/labels/*.txt     (YOLO formatı)
    veri/ham/<isim>/onizleme/*.jpg   (kutular çizilmiş; hızlı göz kontrolü için)
    veri/ham/<isim>/siniflar.txt
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, izgara, kutu_ciz  # noqa: E402
from ortak.yukle import gerekli, modul_yukle, yol_coz  # noqa: E402

yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
DOSYA = "01_veri_fabrikasi/otomatik_etiketle.py"


@dataclass
class Ayarlar:
    klasor: str = ayar("veri/ham/esyalarim", "İçinde images/ olan ham dataset klasörü (köke göre ya da tam yol)", tur="klasor")
    siniflar: str = ayar("anahtar=keys, cuzdan=wallet, kupa=mug, telefon=cell phone",
                         "ad=İngilizce istem, virgülle ayrılmış", tur="uzun_metin",
                         etki="İstemi değiştirmek (ör. 'keys' → 'keychain') sonucu ciddi değiştirir")
    model: str = ayar("yolov8s-worldv2.pt", "YOLO-World ağırlığı (ilk çalıştırmada otomatik indirilir)",
                      secenekler=["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt"])
    guven: float = ayar(0.25, "Bu güvenin altındaki tespitleri yazma", min=0.01, max=0.9, adim=0.01,
                        etki="Düşürürsen daha çok kutu (ve daha çok yanlış); sonra elle silmek, elle çizmekten kolaydır")
    uzerine_yaz: bool = ayar(False, "Etiketi zaten olan görüntüleri de yeniden etiketle",
                             etki="Açarsan elle düzelttiğin etiketler SİLİNİR")


def siniflari_coz(metin: str) -> tuple[list[str], list[str]]:
    """'anahtar=keys, kupa=mug' -> (['anahtar','kupa'], ['keys','mug'])"""
    adlar, istemler = [], []
    for parca in metin.replace("\n", ",").split(","):
        if not parca.strip():
            continue
        ad, _, istem = parca.partition("=")
        adlar.append(ad.strip())
        istemler.append((istem or ad).strip())
    return adlar, istemler


def klasoru_coz(yol: str) -> Path:
    """'veri/ham/x' gibi yolları gerçek klasöre çevir (bkz. ortak/yukle.py:yol_coz)."""
    return yol_coz(yol)


class OtomatikEtiketle(Eylem):
    ISIM = "Otomatik ön-etiketleme (YOLO-World)"
    BOLUM = "01"
    DOSYA = DOSYA
    ACIKLAMA = ("Eğitimsiz, metinle tespit yapan YOLO-World ile ham görüntülere **ön-etiket** yazar. "
                "Sonra CVAT/Label Studio'da düzeltip 02'de kendi modelini eğitirsin. Elle düzeltilmiş etiketleri ezmez.")
    GEREKSINIMLER = ("ultralytics",)
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        ultra = gerekli("ultralytics", "pip install ultralytics")
        kok = klasoru_coz(ayarlar.klasor)
        goruntuler = yf.goruntuleri_listele(kok / "images")
        if not goruntuler:
            raise FileNotFoundError(f"{kok / 'images'} içinde görüntü yok")
        adlar, istemler = siniflari_coz(ayarlar.siniflar)
        (kok / "siniflar.txt").write_text("\n".join(adlar) + "\n", encoding="utf-8")
        anlat.adim(f"{len(goruntuler)} görüntü, sınıflar: {dict(zip(adlar, istemler))}", kod=f"{DOSYA}:siniflari_coz")

        model = ultra.YOLOWorld(ayarlar.model)
        model.set_classes(istemler)   # metin istemleri -> CLIP metin embedding'leri -> sınıf başlığı
        anlat.adim(f"YOLO-World yüklendi ({ayarlar.model}); istemler metin kodlayıcıdan geçirildi",
                   kod="ultralytics:YOLOWorld.set_classes")

        onizleme_k = kok / "onizleme"
        onizleme_k.mkdir(exist_ok=True)
        sayac = {"etiketlenen": 0, "atlanan_mevcut": 0, "bos": 0, "kutu": 0}
        ornekler, basliklar = [], []
        for p in goruntuler:
            txt = kok / "labels" / (p.stem + ".txt")
            if txt.exists() and not ayarlar.uzerine_yaz:
                sayac["atlanan_mevcut"] += 1
                continue
            r = model.predict(str(p), conf=ayarlar.guven, verbose=False)[0]
            kutular = r.boxes.xyxy.cpu().numpy()
            siniflar = r.boxes.cls.cpu().numpy().astype(int).tolist()
            guvenler = r.boxes.conf.cpu().numpy()
            y, g = r.orig_shape
            yf.etiket_yaz(txt, kutular, siniflar, g, y)
            sayac["etiketlenen"] += 1
            sayac["kutu"] += len(kutular)
            sayac["bos"] += int(len(kutular) == 0)
            img = cv2.imread(str(p))
            for k, s, c in zip(kutular, siniflar, guvenler):
                kutu_ciz(img, k, f"{adlar[s]} {c:.2f}", id_rengi(s))
            cv2.imwrite(str(onizleme_k / p.name), img)
            if len(ornekler) < 12:
                ornekler.append(img); basliklar.append(p.name)
            anlat.bilgi(f"{p.name}: {len(kutular)} kutu " + ", ".join(f"{adlar[s]}:{c:.2f}" for s, c in zip(siniflar, guvenler)),
                        kod="ultralytics:YOLOWorld.predict → yolo_format.etiket_yaz")
        anlat.adim("Bitti", veri=sayac)
        izg = izgara(ornekler, basliklar) if ornekler else None
        if izg is not None:
            cv2.imwrite(str(onizleme_k / "_izgara.jpg"), izg)
        return Sonuc(goruntu=izg, veri=sayac, metin=(
            f"{sayac['etiketlenen']} görüntü etiketlendi ({sayac['kutu']} kutu, {sayac['bos']} boş). "
            f"{sayac['atlanan_mevcut']} görüntünün etiketi zaten vardı, dokunulmadı.\n\n"
            f"**Şimdi:** `{onizleme_k}` klasörüne göz at; hatalıları CVAT/Label Studio'da düzelt "
            "(içe aktarma formatı: *YOLO 1.1*). Sonra **Etiket kontrol** → **Dataset böl**."))


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(OtomatikEtiketle)
