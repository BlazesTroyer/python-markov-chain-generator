"""
benchmark.py — Modelleri/formatları/boyutları aynı koşullarda HIZ açısından karşılaştır.

DOĞRU ÖLÇÜM NASIL YAPILIR? (en sık yapılan hatalar ve önlemleri)
    1. ISINMA (warmup): İlk birkaç çıkarım yavaştır (bellek ayırma, JIT, önbellek). Ölçmeden ÖNCE at.
    2. TEKRAR: Tek ölçüm gürültülüdür. N kez ölç, ortalama değil MEDYAN (p50) ve KUYRUK (p95) raporla.
       7/24 sistemde p95 önemlidir: "çoğu zaman hızlı ama arada takılıyor" p50'de görünmez.
    3. AYRIŞTIR: Toplam süre = ön işleme + ağ (inference) + son işleme (NMS). Ultralytics her birini
       ayrı verir (r.speed). Darboğazın NEREDE olduğunu bilmeden optimizasyon yapma.
    4. AYNI KOŞUL: Aynı görüntü, aynı imgsz, aynı cihaz, başka ağır iş çalışmazken.

ÇIKTI: Model × imgsz tablosu (p50/p95 ms, FPS, ön/ağ/son işleme payı) + çubuk grafik.
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
from ortak.yukle import gerekli, modul_yukle, yol_coz  # noqa: E402

ek = modul_yukle("01_veri_fabrikasi/etiket_kontrol.py")
DOSYA = "10_dagitim/benchmark.py"


def olc(fonk, isinma: int, tekrar: int) -> dict:
    """fonk()'u ısıtıp 'tekrar' kez çalıştır; ms cinsinden p50/p95/ortalama (saf — test edilir)."""
    for _ in range(isinma):
        fonk()
    sureler = []
    for _ in range(tekrar):
        t0 = time.perf_counter()
        fonk()
        sureler.append((time.perf_counter() - t0) * 1000)
    s = np.array(sureler)
    return {"p50": float(np.percentile(s, 50)), "p95": float(np.percentile(s, 95)), "ort": float(s.mean()), "fps": 1000 / float(np.median(s))}


@dataclass
class Ayarlar:
    modeller: str = ayar("yolo11n.pt, yolo11n.onnx", "Virgülle model dosyaları (.pt, .onnx, _openvino_model/ ...)", tur="uzun_metin")
    boyutlar: str = ayar("320, 640", "Virgülle imgsz değerleri")
    goruntu: str = ayar("", "Test görüntüsü (boş = Ultralytics bus.jpg)", tur="dosya")
    isinma: int = ayar(3, "Isınma çalıştırması", min=0, max=50)
    tekrar: int = ayar(20, "Ölçüm tekrarı", min=3, max=1000, etki="Artırırsan sonuç daha güvenilir ama süre uzar")


class Benchmark(Eylem):
    ISIM = "Hız benchmark'ı (p50/p95, aşama dökümü)"
    BOLUM = "10"
    DOSYA = DOSYA
    ACIKLAMA = "Model × görüntü boyutu kombinasyonlarını ısınma + tekrarla ölçer; p50/p95 gecikme, FPS ve ön/ağ/son işleme dökümü."
    GEREKSINIMLER = ("ultralytics",)
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        ultra = gerekli("ultralytics")
        img = cv2.imread(str(yol_coz(a.goruntu))) if a.goruntu else cv2.imread(str(Path(ultra.__file__).parent / "assets" / "bus.jpg"))
        satirlar = []
        for ad in [m.strip() for m in a.modeller.split(",") if m.strip()]:
            yol = yol_coz(ad) if "/" in ad else Path(ad)
            if not yol.exists() and not ad.endswith(".pt"):
                anlat.uyari(f"{ad} bulunamadı (önce export.py) — atlandı")
                continue
            m = ultra.YOLO(str(yol), task="detect")
            for b in [int(x) for x in a.boyutlar.split(",") if x.strip()]:
                asamalar = []

                def calistir_bir():
                    asamalar.append(m.predict(img, imgsz=b, verbose=False)[0].speed)
                s = olc(calistir_bir, a.isinma, a.tekrar)
                asamalar = asamalar[a.isinma:]
                on, ag, son = (float(np.median([x[k] for x in asamalar])) for k in ("preprocess", "inference", "postprocess"))
                satirlar.append({"model": Path(ad).name, "imgsz": b, **{k: round(v, 1) for k, v in s.items()},
                                 "on_isleme": round(on, 1), "ag": round(ag, 1), "son_isleme": round(son, 1)})
                anlat.adim(f"{Path(ad).name} @ {b}: p50 {s['p50']:.1f} ms, p95 {s['p95']:.1f} ms ({s['fps']:.1f} FPS) | "
                           f"ön {on:.1f} / ağ {ag:.1f} / son {son:.1f} ms", kod=f"{DOSYA}:olc")
        tablo = "| model | imgsz | p50 ms | p95 ms | FPS | ön | ağ | son (NMS) |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(
            f"| {r['model']} | {r['imgsz']} | {r['p50']} | {r['p95']} | {r['fps']} | {r['on_isleme']} | {r['ag']} | {r['son_isleme']} |" for r in satirlar)
        grafik = ek.sutun_grafigi({f"{r['model'][:10]}@{r['imgsz']}": int(r["fps"]) for r in satirlar} or {"-": 0}, "FPS (p50)")
        return Sonuc(goruntu=grafik, veri={"sonuclar": satirlar}, metin=tablo + "\n\n> Toplam süre ≈ ön + ağ + son. Hangi sütun baskın? Orayı optimize et.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(Benchmark)
