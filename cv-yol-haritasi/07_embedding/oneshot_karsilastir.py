"""
oneshot_karsilastir.py — "Elimde nesnemin 6 fotoğrafı var; onu yeni görüntülerde nasıl tanırım?" Yöntemleri YARIŞTIR.

SORU: Hangi yöntem ne zaman? (Senin v1 one-shot projenin bilimsel değerlendirmesi)
    renk       : Referansların renk histogramlarına en yakın benzerlik         (model yok — v1'e benzer kaba yöntem)
    yolo_cls   : ImageNet sınıflandırıcı özelliklerine en yakın benzerlik     (küçük, hızlı)
    dinov2     : DINOv2 özelliklerine en yakın benzerlik                       (aynı nesneyi farklı açıda tanımada güçlü)
    clip       : CLIP görüntü özelliklerine en yakın benzerlik
    yolo_world : Metin istemiyle tespit ("red mug"); skor = en yüksek tespit güveni  (referans fotoğrafı bile gerekmez!)
    yolo_model : 02/06'da eğittiğin modelin bu sınıf için en yüksek güveni      (en çok emek, genelde en iyi)

SKOR -> KARAR: Her test görüntüsü bir skor alır. Eşiğin üstü "nesnem var".
    Eşikten bağımsız karşılaştırma için ROC eğrisi ve AUC (sıfırdan):
        Eşiği yukarıdan aşağı indir; her eşikte TPR = doğru bulunan pozitif oranı, FPR = yanlış alarm oranı.
        AUC = ROC eğrisi altındaki alan (1 = mükemmel ayırma, 0.5 = yazı tura).
    Ayrıca: en iyi eşikteki doğruluk ve görüntü başına süre (ms).

VERİ DÜZENİ: veri/oneshot/<nesne>/referans/*.jpg  (6 fotoğraf)
                               test/pozitif/*.jpg  (nesnenin olduğu YENİ fotoğraflar)
                               test/negatif/*.jpg  (olmadığı fotoğraflar — benzer nesneler dahil!)
             ornek_uret=True ile sentetik bir örnek veri seti üretilir (hattı denemek için).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi  # noqa: E402
from ortak.yukle import gerekli, modul_yukle, yol_coz  # noqa: E402

gm = modul_yukle("07_embedding/gomucu.py")
yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
od = modul_yukle("01_veri_fabrikasi/ornek_dataset.py")
DOSYA = "07_embedding/oneshot_karsilastir.py"


def roc_auc(skorlar: np.ndarray, etiketler: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """ROC (FPR, TPR), AUC, en iyi eşik ve o eşikteki doğruluk — sıfırdan."""
    sira = np.argsort(-skorlar)
    s, e = skorlar[sira], etiketler[sira]
    P, N = e.sum(), (1 - e).sum()
    tpr = np.concatenate([[0], np.cumsum(e) / max(P, 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - e) / max(N, 1)])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    dogruluklar = [((skorlar >= t) == etiketler.astype(bool)).mean() for t in s]
    en_iyi = int(np.argmax(dogruluklar))
    return fpr, tpr, auc, float(s[en_iyi]), float(dogruluklar[en_iyi])


def roc_cizimi(egriler: dict[str, tuple[np.ndarray, np.ndarray, float]], G=480, Y=400) -> np.ndarray:
    img = np.full((Y, G, 3), 255, np.uint8)
    sol, alt, ust, sag = 45, Y - 35, 15, G - 15
    cv2.rectangle(img, (sol, ust), (sag, alt), (0, 0, 0), 1)
    cv2.line(img, (sol, alt), (sag, ust), (200, 200, 200), 1)            # yazı-tura çizgisi
    cv2.putText(img, "FPR (yanlis alarm)", (G // 2 - 60, Y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "TPR", (5, ust + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for i, (ad, (fpr, tpr, auc)) in enumerate(egriler.items()):
        p = np.stack([sol + fpr * (sag - sol), alt - tpr * (alt - ust)], 1).astype(np.int32)
        cv2.polylines(img, [p], False, id_rengi(i + 1), 2)
        cv2.putText(img, f"{ad} AUC={auc:.2f}", (sol + 120, alt - 12 - 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, id_rengi(i + 1), 1)
    return img


def ornek_veri_uret(kok: Path, rng) -> None:
    """Hedef: kırmızı gövdeli, kulplu 'kupa'. Negatif: benzer renk/şekilli başka nesneler + boş sahneler."""
    def kupa(img, x, y, s, renk=(30, 30, 210)):
        cv2.rectangle(img, (x, y), (x + s, y + int(s * 1.2)), renk, -1)
        cv2.ellipse(img, (x + s, y + int(s * 0.6)), (s // 3, s // 3), 0, -90, 90, renk, max(3, s // 10))
        cv2.rectangle(img, (x + s // 4, y + s // 3), (x + 3 * s // 4, y + s // 2), (240, 240, 240), -1)   # logo

    def sahne(hedef: bool):
        img = od._arka_plan(rng, 224, rng.uniform(60, 200, 3))
        s = int(rng.uniform(40, 80))
        x, y = int(rng.uniform(10, 224 - 2 * s)), int(rng.uniform(10, 224 - 1.4 * s))
        if hedef:
            kupa(img, x, y, s)
        elif rng.random() < 0.5:   # zor negatif: aynı renkte ama kulpsuz/logosuz kutu
            cv2.rectangle(img, (x, y), (x + s, y + int(s * 1.2)), (30, 30, 210), -1)
        else:                       # başka renkte kupa
            kupa(img, x, y, s, tuple(int(v) for v in rng.uniform(0, 255, 3)))
        return img
    for alt, n, hedef in (("referans", 6, True), ("test/pozitif", 30, True), ("test/negatif", 30, False)):
        (kok / alt).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(kok / alt / f"{i:03d}.jpg"), sahne(hedef))


@dataclass
class Ayarlar:
    klasor: str = ayar("veri/oneshot/kupa", "referans/ ve test/{pozitif,negatif}/ içeren klasör", tur="klasor")
    ornek_uret: bool = ayar(True, "Klasör boşsa sentetik örnek veri üret")
    yontemler: str = ayar("renk,yolo_cls", "Virgülle: renk, yolo_cls, dinov2, clip, yolo_world, yolo_model",
                          etki="dinov2/clip HuggingFace'ten indirir; yolo_world/yolo_model ultralytics gerektirir")
    istem: str = ayar("red mug", "yolo_world için metin istemi")
    yolo_model: str = ayar("", "yolo_model için kendi best.pt yolun", tur="dosya")
    sinif: str = ayar("", "yolo_model'de hedef sınıf adı")
    tohum: int = ayar(0, "Örnek veri tohumu", min=0, max=9999)


class OneShotKarsilastir(Eylem):
    ISIM = "One-shot yöntem yarışı (ROC / AUC / hız)"
    BOLUM = "07"
    DOSYA = DOSYA
    ACIKLAMA = ("Aynı referans/test fotoğraflarında renk histogramı, YOLO-cls, DINOv2, CLIP embedding'leri, YOLO-World ve kendi "
                "YOLO modelini karşılaştırır: ROC eğrileri, AUC, en iyi eşikte doğruluk ve görüntü başına süre.")
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        kok = yol_coz(a.klasor)
        if a.ornek_uret and not (kok / "referans").exists():
            ornek_veri_uret(kok, np.random.default_rng(a.tohum))
            anlat.adim(f"Sentetik örnek veri üretildi: {kok}", kod=f"{DOSYA}:ornek_veri_uret")
        oku = lambda alt: [cv2.imread(str(p)) for p in yf.goruntuleri_listele(kok / alt)]  # noqa: E731
        ref, poz, neg = oku("referans"), oku("test/pozitif"), oku("test/negatif")
        test = poz + neg
        etiket = np.array([1] * len(poz) + [0] * len(neg))
        anlat.adim(f"{len(ref)} referans, {len(poz)} pozitif, {len(neg)} negatif test", kod=f"{DOSYA}:OneShotKarsilastir.calistir")

        sonuclar, egriler, atlananlar = {}, {}, {}
        for y in [s.strip() for s in a.yontemler.split(",") if s.strip()]:
            try:
                t0 = time.perf_counter()
                if y in gm.GOMUCULER:
                    g = gm.gomucu_al(y)
                    R, T = g.goruntu(ref), g.goruntu(test)
                    skor = (T @ R.T).max(axis=1)              # her test görüntüsü: en benzer referansa kosinüs
                elif y == "yolo_world":
                    m = gerekli("ultralytics").YOLOWorld("yolov8s-worldv2.pt")
                    m.set_classes([a.istem])
                    skor = np.array([float(r.boxes.conf.max()) if len(r.boxes) else 0.0
                                     for r in (m.predict(img, conf=0.01, verbose=False)[0] for img in test)])
                elif y == "yolo_model":
                    m = gerekli("ultralytics").YOLO(str(yol_coz(a.yolo_model)))
                    no = [k for k, v in m.names.items() if v == a.sinif]
                    skor = np.array([float(r.boxes.conf.max()) if len(r.boxes) else 0.0
                                     for r in (m.predict(img, conf=0.01, classes=no or None, verbose=False)[0] for img in test)])
                else:
                    anlat.uyari(f"Bilinmeyen yöntem: {y}"); continue
                ms = (time.perf_counter() - t0) * 1000 / len(test)
                fpr, tpr, auc, esik, dogruluk = roc_auc(skor, etiket)
                sonuclar[y] = {"AUC": round(auc, 3), "en_iyi_esik": round(esik, 3), "dogruluk": round(dogruluk, 3), "ms_goruntu": round(ms, 1),
                               "poz_ort": round(float(skor[etiket == 1].mean()), 3), "neg_ort": round(float(skor[etiket == 0].mean()), 3)}
                egriler[y] = (fpr, tpr, auc)
                anlat.adim(f"{y}: AUC={auc:.3f}, doğruluk={dogruluk:.3f} (eşik {esik:.3f}), {ms:.1f} ms/görüntü",
                           kod=f"{DOSYA}:roc_auc", veri={"poz_ort": sonuclar[y]["poz_ort"], "neg_ort": sonuclar[y]["neg_ort"]})
            except Exception as e:  # noqa: BLE001 — bir yöntemin eksik paketi/indirmesi diğerlerini durdurmasın
                ipucu = " (YOLO-World için: pip install git+https://github.com/ultralytics/CLIP.git)" if "clip" in str(e).lower() else ""
                atlananlar[y] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}{ipucu}"
                anlat.uyari(f"{y} atlandı: {atlananlar[y]}")
        tablo = "| yöntem | AUC ↑ | doğruluk | eşik | ms/görüntü | ort. skor (poz / neg) |\n|---|---|---|---|---|---|\n" + "\n".join(
            f"| {y} | {s['AUC']} | {s['dogruluk']} | {s['en_iyi_esik']} | {s['ms_goruntu']} | {s['poz_ort']} / {s['neg_ort']} |" for y, s in sonuclar.items())
        roc = roc_cizimi(egriler)
        if atlananlar:
            tablo += "\n\n**Atlanan yöntemler:**\n" + "\n".join(f"- `{y}`: {n}" for y, n in atlananlar.items())
        return Sonuc(goruntu=roc, ara_gorseller={"ROC eğrileri": roc}, veri={"sonuclar": sonuclar, "atlananlar": atlananlar}, metin=tablo)


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(OneShotKarsilastir)
