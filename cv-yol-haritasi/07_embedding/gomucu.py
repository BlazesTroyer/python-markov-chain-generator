"""
gomucu.py — Görüntüyü (ve CLIP'te metni) bir VEKTÖRE çeviren "gömücüler" (embedder), tek arayüzle.

EMBEDDING NEDİR?
    Bir görüntüyü D sayılık bir vektöre çevirmek; öyle ki BENZER görüntüler uzayda YAKIN olsun.
    "Benzer"in ne demek olduğu modele göre değişir — bu dosyadaki 4 seçenek bunu görmeni sağlar:

    renk     : HSV renk histogramı (512 boyut). Model yok, eğitim yok. "Aynı renkler" = benzer.
               Taban çizgisi: senin v1 one-shot yöntemine benzer kaba bir benzerlik.
    yolo_cls : Ultralytics yolo11n-cls (ImageNet sınıflandırıcısı) iç katman özellikleri (256 boyut).
               "Aynı tür nesne" = benzer. Küçük ve hızlı.
    dinov2   : Meta DINOv2 (kendi kendine denetimli; 384 boyut, timm). Görsel ayrıntıda çok güçlü:
               "aynı nesne, farklı açı" eşleştirmesinde genelde en iyisi. (HuggingFace'ten iner.)
    clip     : OpenAI/LAION CLIP (512 boyut, open_clip). Görüntü ve METİN AYNI uzaya gömülür:
               "denizde köpek" yazıp fotoğraf arayabilirsin. (HuggingFace'ten iner.)

ORTAK ARAYÜZ:
    g = gomucu_al("clip")
    G = g.goruntu([img1, img2])     -> (2, D)  L2 normalize (kosinüs = iç çarpım)
    T = g.metin(["denizde köpek"])  -> (1, D)  (sadece clip)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.yukle import gerekli  # noqa: E402

DOSYA = "07_embedding/gomucu.py"
GOMUCULER = ["renk", "yolo_cls", "dinov2", "clip"]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class RenkGomucu:
    ad, metin_destekli = "renk", False

    def goruntu(self, goruntuler: list[np.ndarray]) -> np.ndarray:
        cikti = []
        for img in goruntuler:
            hsv = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
            cikti.append(np.sqrt(h))           # karekök: baskın rengin her şeyi ezmesini azaltır (Hellinger)
        return _normalize(np.stack(cikti))


class YoloClsGomucu:
    ad, metin_destekli = "yolo_cls", False

    def __init__(self, model: str = "yolo11n-cls.pt"):
        self.model = gerekli("ultralytics").YOLO(model)

    def goruntu(self, goruntuler):
        vekt = self.model.embed(list(goruntuler), verbose=False)    # her biri (D,) torch tensörü
        return _normalize(np.stack([v.cpu().numpy() for v in vekt]))


class Dinov2Gomucu:
    ad, metin_destekli = "dinov2", False

    def __init__(self, model: str = "vit_small_patch14_dinov2.lvd142m"):
        timm, self.torch = gerekli("timm", "pip install timm"), gerekli("torch")
        self.model = timm.create_model(model, pretrained=True, num_classes=0).eval()   # num_classes=0 -> özellik çıkar
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)

    def goruntu(self, goruntuler):
        from PIL import Image
        x = self.torch.stack([self.tf(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))) for g in goruntuler])
        with self.torch.no_grad():
            return _normalize(self.model(x).numpy())


class ClipGomucu:
    ad, metin_destekli = "clip", True

    def __init__(self, model: str = "ViT-B-32", agirlik: str = "laion2b_s34b_b79k"):
        oc, self.torch = gerekli("open_clip", "pip install open_clip_torch"), gerekli("torch")
        self.model, _, self.tf = oc.create_model_and_transforms(model, pretrained=agirlik)
        self.model.eval()
        self.tokenizer = oc.get_tokenizer(model)

    def goruntu(self, goruntuler):
        from PIL import Image
        x = self.torch.stack([self.tf(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))) for g in goruntuler])
        with self.torch.no_grad():
            return _normalize(self.model.encode_image(x).numpy())

    def metin(self, metinler: list[str]) -> np.ndarray:
        """CLIP İngilizce eğitildi: Türkçe sorgular çalışır ama İngilizce çok daha iyi sonuç verir."""
        with self.torch.no_grad():
            return _normalize(self.model.encode_text(self.tokenizer(metinler)).numpy())


_ONBELLEK: dict[str, object] = {}


def gomucu_al(ad: str):
    """Gömücüyü bir kez kur, sonra önbellekten ver."""
    if ad not in _ONBELLEK:
        _ONBELLEK[ad] = {"renk": RenkGomucu, "yolo_cls": YoloClsGomucu, "dinov2": Dinov2Gomucu, "clip": ClipGomucu}[ad]()
    return _ONBELLEK[ad]
