"""
export.py — PyTorch modelini (.pt) dağıtım formatlarına çevir ve DOĞRULUK KAYBINI ölç.

NEDEN EXPORT?
    .pt dosyası eğitim için idealdir ama çalıştırmak için PyTorch'un tamamı gerekir (ağır, yavaş başlar).
    Dağıtım formatları:
        onnx        : Evrensel ara format. ONNX Runtime ile CPU'da genelde PyTorch'tan hızlı; her yerde çalışır.
        openvino    : Intel CPU / iGPU için optimize (dizüstü, mini PC'ler).
        engine      : NVIDIA TensorRT (Jetson, RTX). En hızlısı ama sadece NVIDIA GPU + o makinede üretilmeli.
        torchscript : PyTorch'suz Python'a gerek kalmadan (C++ vb.) çalıştırmak için.
    Hassasiyet:
        half (FP16) : Ağırlıklar 16 bit -> ~2x küçük, GPU'da hızlı. Doğruluk kaybı genelde ihmal edilebilir.
        int8        : 8 bit tamsayı (kuantizasyon) -> ~4x küçük, en hızlı; KALİBRASYON verisi ister (data=)
                      ve doğruluk kaybı ölçülmeli. Kalibrasyon: aktivasyon aralıklarını gerçek görüntülerle öğrenmek.

DOĞRULUK KONTROLÜ (bu dosyanın asıl dersi):
    Export "çalıştı" demek "aynı sonucu veriyor" demek değildir. Bu yüzden:
    1. Aynı örnek görüntülerde orijinal (.pt) ve export edilmiş model çalıştırılır.
    2. Kutular IoU ile eşleştirilir (00 metrikler) -> uyum oranı, ortalama IoU, güven farkı.
    3. data verilirse ikisinin de val mAP'i ölçülür.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.yukle import gerekli, modul_yukle, yol_coz  # noqa: E402

met = modul_yukle("00_temeller/metrikler.py")
it = modul_yukle("03_takip/iou_takipci.py")
yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
DOSYA = "10_dagitim/export.py"


def tahmin_uyumu(m1, m2, goruntuler, imgsz: int, guven: float = 0.25) -> dict:
    """İki modelin aynı görüntülerdeki tespitlerini karşılaştır (saf mantık + modeller)."""
    eslesen = toplam1 = toplam2 = 0
    iou_list, guven_farki = [], []
    for img in goruntuler:
        a = yy.sonucu_ayikla(m1.predict(img, imgsz=imgsz, conf=guven, verbose=False)[0])
        b = yy.sonucu_ayikla(m2.predict(img, imgsz=imgsz, conf=guven, verbose=False)[0])
        toplam1, toplam2 = toplam1 + len(a[0]), toplam2 + len(b[0])
        iou = met.iou_matrisi(a[0], b[0])
        for i in range(len(a[1])):                       # farklı sınıflar eşleşmesin
            iou[i, b[1] != a[1][i]] = 0
        eslesme, _, _ = it.macar_eslestir(iou, 0.5)
        eslesen += len(eslesme)
        iou_list += [float(iou[i, j]) for i, j in eslesme]
        guven_farki += [abs(float(a[2][i] - b[2][j])) for i, j in eslesme]
    return {"orijinal_kutu": toplam1, "export_kutu": toplam2, "eslesen": eslesen,
            "uyum_orani": eslesen / max(toplam1, toplam2, 1),
            "ort_iou": float(np.mean(iou_list)) if iou_list else 0.0,
            "ort_guven_farki": float(np.mean(guven_farki)) if guven_farki else 0.0}


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n.pt", "Kaynak .pt modeli", tur="dosya")
    format: str = ayar("onnx", "Hedef format", secenekler=["onnx", "openvino", "torchscript", "engine"],
                       etki="engine (TensorRT) sadece NVIDIA GPU'lu makinede üretilebilir")
    imgsz: int = ayar(640, "Sabit girdi boyutu", min=160, max=1280, adim=32, etki="Küçültürsen hızlanır, küçük nesneler kaybolur")
    half: bool = ayar(False, "FP16 (yarım hassasiyet)")
    int8: bool = ayar(False, "INT8 kuantizasyon (data ile kalibrasyon gerekir; openvino/engine)")
    data: str = ayar("", "Kalibrasyon ve mAP karşılaştırması için data.yaml (opsiyonel)", tur="dosya")
    ornek_klasor: str = ayar("", "Tahmin uyumu için örnek görüntü klasörü (boş = Ultralytics örnekleri)", tur="klasor")


class Export(Eylem):
    ISIM = "Model export + doğruluk kaybı kontrolü"
    BOLUM = "10"
    DOSYA = DOSYA
    ACIKLAMA = ("`.pt` modelini ONNX/OpenVINO/TensorRT/TorchScript'e çevirir; dosya boyutunu, örnek görüntülerde **tahmin uyumunu** "
                "(eşleşen kutu oranı, ortalama IoU, güven farkı) ve isteğe bağlı val mAP farkını raporlar.")
    GEREKSINIMLER = ("ultralytics", "onnx + onnxruntime (onnx)", "openvino (openvino)")
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        ultra = gerekli("ultralytics")
        kaynak = yol_coz(a.model) if "/" in a.model else Path(a.model)
        m_pt = ultra.YOLO(str(kaynak))
        argumanlar = dict(format=a.format, imgsz=a.imgsz, half=a.half, int8=a.int8, verbose=False)
        if a.data:
            argumanlar["data"] = str(yol_coz(a.data))
        anlat.adim(f"Export argümanları: {argumanlar}", kod="ultralytics:YOLO.export")
        cikti = Path(m_pt.export(**argumanlar))
        boyut = lambda p: sum(f.stat().st_size for f in p.rglob("*")) if p.is_dir() else p.stat().st_size  # noqa: E731
        mb_pt, mb_ex = Path(m_pt.ckpt_path).stat().st_size / 1e6, boyut(cikti) / 1e6
        anlat.adim(f"{cikti.name}: {mb_pt:.1f} MB → {mb_ex:.1f} MB", kod=f"{DOSYA}:Export.calistir")

        m_ex = ultra.YOLO(str(cikti), task=m_pt.task)
        if a.ornek_klasor:
            dosyalar = yf.goruntuleri_listele(yol_coz(a.ornek_klasor))[:20]
        else:
            dosyalar = yf.goruntuleri_listele(Path(ultra.__file__).parent / "assets")
        goruntuler = [cv2.imread(str(p)) for p in dosyalar]
        uyum = tahmin_uyumu(m_pt, m_ex, goruntuler, a.imgsz)
        anlat.adim(f"Tahmin uyumu ({len(goruntuler)} görüntü): %{uyum['uyum_orani'] * 100:.1f} eşleşme, ort. IoU {uyum['ort_iou']:.3f}, "
                   f"ort. güven farkı {uyum['ort_guven_farki']:.4f}", kod=f"{DOSYA}:tahmin_uyumu")
        veri = {"cikti": str(cikti), "mb_pt": round(mb_pt, 2), "mb_export": round(mb_ex, 2), "uyum": uyum}
        if a.data:
            v1 = m_pt.val(data=str(yol_coz(a.data)), imgsz=a.imgsz, verbose=False, plots=False)
            v2 = m_ex.val(data=str(yol_coz(a.data)), imgsz=a.imgsz, verbose=False, plots=False)
            veri["mAP50_95"] = {"pt": float(v1.box.map), "export": float(v2.box.map)}
            anlat.adim(f"val mAP50-95: pt {v1.box.map:.4f} → export {v2.box.map:.4f}", kod="ultralytics:YOLO.val")
        return Sonuc(veri=veri, metin=(
            f"✔ `{cikti}` ({mb_ex:.1f} MB; .pt {mb_pt:.1f} MB)\n\n"
            f"| ölçüt | değer |\n|---|---|\n| eşleşen kutu oranı | %{uyum['uyum_orani'] * 100:.1f} |\n"
            f"| ortalama IoU | {uyum['ort_iou']:.3f} |\n| ortalama güven farkı | {uyum['ort_guven_farki']:.4f} |"
            + (f"\n| val mAP50-95 (pt → export) | {veri['mAP50_95']['pt']:.4f} → {veri['mAP50_95']['export']:.4f} |" if a.data else "")
            + "\n\nSıradaki: **Benchmark** ile hızını ölç."))


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(Export)
