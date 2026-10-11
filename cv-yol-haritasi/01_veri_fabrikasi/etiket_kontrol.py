"""
etiket_kontrol.py — Eğitimden ÖNCE dataset sağlık raporu. ("Çöp girerse çöp çıkar.")

NE KONTROL EDER?
    1. Eşleşme      : Etiketi olmayan görüntü / görüntüsü olmayan etiket var mı?
    2. Format       : Her satır 5 değer mi? Değerler 0..1 içinde mi? Sınıf no geçerli mi?
    3. Sınıf dengesi: Her sınıftan kaç kutu var? (Biri diğerinin 10 katıysa model ona kayar.)
    4. Kutu boyutları: Çok küçük kutular (görüntünün %1'inden küçük) — 640px'te birkaç piksel kalır.
    5. Negatifler   : Boş etiketli görüntü oranı (biraz olması iyidir: %5-15).
    6. Göz kontrolü : Rastgele örnekleri kutularıyla çizer — yanlış etiketi en hızlı göz yakalar.

ÇALIŞTIR:
    python 01_veri_fabrikasi/etiket_kontrol.py --klasor veri/ham/sekiller
    python 01_veri_fabrikasi/etiket_kontrol.py --klasor veri/datasetler/sekiller   # bölünmüş dataset de olur
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, izgara, kutu_ciz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
oe = modul_yukle("01_veri_fabrikasi/otomatik_etiketle.py")
DOSYA = "01_veri_fabrikasi/etiket_kontrol.py"


@dataclass
class Ayarlar:
    klasor: str = ayar("veri/ham/sekiller", "Dataset klasörü (images/ ve labels/ içeren)", tur="klasor")
    ornek_sayisi: int = ayar(12, "Göz kontrolü için kaç rastgele örnek çizilsin", min=0, max=48)
    kucuk_kutu_orani: float = ayar(0.01, "Alanı görüntünün bu oranından küçük kutular 'çok küçük' sayılır",
                                   min=0.001, max=0.1, adim=0.001)
    tohum: int = ayar(0, "Örnek seçimi tohumu", min=0, max=9999)


def sinif_isimleri(kok: Path) -> list[str]:
    """siniflar.txt (ham) ya da data.yaml (bölünmüş) — hangisi varsa."""
    if (kok / "siniflar.txt").exists():
        return [s.strip() for s in (kok / "siniflar.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    if (kok / "data.yaml").exists():
        names = yf.data_yaml_oku(kok / "data.yaml")["names"]
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    return []


def sutun_grafigi(sayilar: dict[str, int], baslik: str, g=520, y=260) -> np.ndarray:
    """Basit yatay çubuk grafik (OpenCV ile — ek bağımlılık yok)."""
    img = np.full((y, g, 3), 255, np.uint8)
    cv2.putText(img, baslik, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    if not sayilar:
        return img
    en_cok = max(sayilar.values()) or 1
    satir_y = (y - 40) // max(len(sayilar), 1)
    for i, (ad, n) in enumerate(sayilar.items()):
        yy = 40 + i * satir_y
        uzunluk = int((g - 200) * n / en_cok)
        cv2.rectangle(img, (120, yy), (120 + uzunluk, yy + satir_y - 6), id_rengi(i), -1)
        cv2.putText(img, ad[:14], (8, yy + satir_y // 2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(n), (126 + uzunluk, yy + satir_y // 2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


class EtiketKontrol(Eylem):
    ISIM = "Etiket kontrol (dataset sağlık raporu)"
    BOLUM = "01"
    DOSYA = DOSYA
    ACIKLAMA = "Eşleşme, format, sınıf dengesi, kutu boyutları ve negatif oranını kontrol eder; rastgele örnekleri kutularıyla çizer."
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        kok = oe.klasoru_coz(ayarlar.klasor)
        isimler = sinif_isimleri(kok)
        goruntuler = yf.goruntuleri_listele(kok / "images")
        etiketler = sorted((kok / "labels").rglob("*.txt")) if (kok / "labels").exists() else []
        anlat.adim(f"{len(goruntuler)} görüntü, {len(etiketler)} etiket dosyası, sınıflar={isimler}", kod=f"{DOSYA}:EtiketKontrol.calistir")

        etiket_kokleri = {e.relative_to(kok / "labels").with_suffix("") for e in etiketler}
        goruntu_kokleri = {g.relative_to(kok / "images").with_suffix("") for g in goruntuler}
        etiketsiz = sorted(goruntu_kokleri - etiket_kokleri)
        sahipsiz = sorted(etiket_kokleri - goruntu_kokleri)
        anlat.adim(f"Eşleşme: etiketsiz görüntü={len(etiketsiz)}, görüntüsüz etiket={len(sahipsiz)}", kod=f"{DOSYA}:(küme farkı)")

        sinif_sayaci, sorunlar, alan_oranlari = Counter(), [], []
        bos, kucuk = 0, 0
        okunanlar = []
        for gp in goruntuler:
            img = cv2.imread(str(gp))
            if img is None:
                sorunlar.append(f"{gp.name}: görüntü okunamadı"); continue
            y, g = img.shape[:2]
            kutular, siniflar, s = yf.etiket_oku(yf.etiket_yolu(gp), g, y)
            sorunlar += s
            if len(kutular) == 0:
                bos += 1
            for k, c in zip(kutular, siniflar):
                if isimler and c >= len(isimler):
                    sorunlar.append(f"{gp.name}: sınıf {c} tanımsız (sadece {len(isimler)} sınıf var)")
                sinif_sayaci[isimler[c] if c < len(isimler) else f"#{c}"] += 1
                oran = (k[2] - k[0]) * (k[3] - k[1]) / (g * y)
                alan_oranlari.append(oran)
                kucuk += oran < ayarlar.kucuk_kutu_orani
            okunanlar.append((gp, kutular, siniflar))
        anlat.adim("Sınıf dağılımı", kod="01_veri_fabrikasi/yolo_format.py:etiket_oku", veri=dict(sinif_sayaci))
        if sinif_sayaci:
            oran = max(sinif_sayaci.values()) / max(min(sinif_sayaci.values()), 1)
            (anlat.uyari if oran > 5 else anlat.bilgi)(f"En çok/en az sınıf oranı: {oran:.1f}x" + (" — DENGESİZ! az olan sınıftan veri topla" if oran > 5 else ""))
        bos_orani = bos / max(len(goruntuler), 1)
        anlat.bilgi(f"Negatif (boş) görüntü oranı: %{bos_orani * 100:.1f} (öneri %5-15)")
        if kucuk:
            anlat.uyari(f"{kucuk} kutu çok küçük (< %{ayarlar.kucuk_kutu_orani * 100:.1f} alan) — daha yüksek imgsz gerekebilir")
        for s in sorunlar[:20]:
            anlat.uyari(s, kod="01_veri_fabrikasi/yolo_format.py:etiket_oku")

        rng = np.random.default_rng(ayarlar.tohum)
        secim = rng.choice(len(okunanlar), size=min(ayarlar.ornek_sayisi, len(okunanlar)), replace=False) if okunanlar else []
        ornekler, basliklar = [], []
        for i in secim:
            gp, kutular, siniflar = okunanlar[i]
            img = cv2.imread(str(gp))
            for k, c in zip(kutular, siniflar):
                kutu_ciz(img, k, isimler[c] if c < len(isimler) else f"#{c}", id_rengi(c))
            ornekler.append(img); basliklar.append(gp.name)
        hist = np.histogram(alan_oranlari, bins=[0, 0.005, 0.01, 0.05, 0.1, 0.25, 1.0])[0] if alan_oranlari else []
        boyut_sayilari = dict(zip(["<0.5%", "0.5-1%", "1-5%", "5-10%", "10-25%", ">25%"], [int(v) for v in hist]))
        ara = {"sınıf dağılımı": sutun_grafigi(dict(sinif_sayaci), "Sinif basina kutu sayisi"),
               "kutu alanı / görüntü alanı": sutun_grafigi(boyut_sayilari, "Kutu boyutu dagilimi")}
        izg = izgara(ornekler, basliklar) if ornekler else None
        if izg is not None:
            ara["rastgele örnekler"] = izg
        saglikli = not sorunlar and not etiketsiz and not sahipsiz
        return Sonuc(
            goruntu=izg if izg is not None else ara["sınıf dağılımı"], ara_gorseller=ara,
            metin=(f"### {'✅ Sağlıklı' if saglikli else '⚠ Sorunlar var'}\n"
                   f"- Görüntü: {len(goruntuler)}, etiketsiz: {len(etiketsiz)}, görüntüsüz etiket: {len(sahipsiz)}\n"
                   f"- Kutu: {sum(sinif_sayaci.values())}, çok küçük: {kucuk}, negatif görüntü: %{bos_orani * 100:.1f}\n"
                   f"- Format sorunu: {len(sorunlar)}" + ("".join(f"\n  - {s}" for s in sorunlar[:10]))),
            veri={"sinif_sayilari": dict(sinif_sayaci), "etiketsiz": [str(e) for e in etiketsiz[:50]],
                  "sorunlar": sorunlar[:100], "kutu_boyutlari": boyut_sayilari, "negatif_orani": bos_orani})


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(EtiketKontrol)
