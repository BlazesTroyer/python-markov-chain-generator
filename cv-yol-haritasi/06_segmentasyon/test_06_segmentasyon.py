"""06_segmentasyon testleri. Saf maske işlemleri ve copy-paste (model gerektirmez).
CV_AGIR_TEST=1: gerçek SAM ile maske testi (sam2.1_t.pt indirir)."""
import os

import cv2
import numpy as np
import pytest

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.yukle import VERI, modul_yukle

mo = modul_yukle("06_segmentasyon/maske_ortak.py")
cp = modul_yukle("06_segmentasyon/copy_paste_aug.py")
yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")


def _daire(boyut=(100, 100), merkez=(50, 50), r=20):
    m = np.zeros(boyut, np.uint8)
    cv2.circle(m, merkez, r, 1, -1)
    return m.astype(bool)


def test_iou_ve_dice():
    a, b = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    a[:, :5], b[:, 2:7] = True, True             # 50 px ve 50 px, kesişim 30 px
    assert abs(mo.maske_iou(a, b) - 30 / 70) < 1e-9 and abs(mo.dice(a, b) - 60 / 100) < 1e-9
    assert mo.maske_iou(a, a) == 1.0 and mo.dice(a, ~a) == 0.0


def test_kutu_ve_en_buyuk_parca():
    m = _daire() | _daire(merkez=(90, 10), r=3)
    temiz = mo.en_buyuk_parca(m)
    assert temiz.sum() == _daire().sum()
    assert mo.kutu_from_maske(temiz).tolist() == [30, 30, 71, 71]
    assert mo.kutu_from_maske(np.zeros((5, 5), bool)) is None


def test_el_maskesi_noktalari_kapsar():
    n = np.array([[0.2, 0.2, 0], [0.4, 0.2, 0], [0.4, 0.4, 0], [0.2, 0.4, 0]] + [[0.3, 0.3, 0]] * 17, np.float32)
    m = mo.el_maskesi([n], (100, 100), genislet=0)
    assert m[30, 30] and not m[60, 60]


def test_kesit_rgba_alfa():
    kare = np.full((100, 100, 3), 200, np.uint8)
    k = mo.kesit_rgba(kare, _daire())
    assert k.shape == (41, 41, 4) and k[20, 20, 3] == 255 and k[0, 0, 3] == 0


def test_copy_paste_etiketleri_maskeden_kesin():
    rng = np.random.default_rng(0)
    for s, renk in (("kirmizi", (0, 0, 255)), ("mavi", (255, 0, 0))):
        d = VERI / "nesneler_t" / s
        d.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            bgra = np.zeros((60, 60, 4), np.uint8)
            cv2.circle(bgra, (30, 30), 25, (*renk, 255), -1)
            cv2.imwrite(str(d / f"{i}.png"), bgra)
    a = ayarlari_olustur(cp.Ayarlar, {"kesit_klasoru": "veri/nesneler_t", "isim": "cp_t", "goruntu_sayisi": 15,
                                      "boyut": 256, "en_fazla_nesne": 1, "aci_max": 0.0, "bos_orani": 0.0})
    s = cp.CopyPasteAug().calistir(a, Anlatici())
    assert s.veri["siniflar"] == ["kirmizi", "mavi"]
    for img_yolu in sorted((VERI / "ham/cp_t/images").glob("*.jpg"))[:5]:
        img = cv2.imread(str(img_yolu))
        kutular, siniflar, _ = yf.etiket_oku(VERI / "ham/cp_t/labels" / (img_yolu.stem + ".txt"), 256, 256)
        assert len(kutular) == 1
        x1, y1, x2, y2 = kutular[0].astype(int)
        merkez = img[(y1 + y2) // 2, (x1 + x2) // 2]
        beklenen = 2 if siniflar[0] == 0 else 0          # kırmızı -> R kanalı (BGR'de 2), mavi -> B (0)
        assert merkez.argmax() == beklenen               # kutunun merkezi gerçekten o renkte nesne


@pytest.mark.skipif(not os.environ.get("CV_AGIR_TEST"), reason="CV_AGIR_TEST=1 ile çalışır (SAM indirir)")
def test_gercek_sam_kutu_istemi():
    import ultralytics
    from pathlib import Path
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    m = mo.sam_maske(img, kutu=[20, 220, 810, 750])          # otobüsün yaklaşık kutusu
    assert 0.2 < m.mean() < 0.6
