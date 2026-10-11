"""04_yuz testleri. Saf mantık testleri model gerektirmez.
CV_AGIR_TEST=1 ile: YuNet/SFace indirip gerçek yüz tespiti/tanıma testi ve küçük bir duygu eğitimi."""
import os

import cv2
import numpy as np
import pytest

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.yukle import VERI, modul_yukle

canli = modul_yukle("04_yuz/emotion_v2/canli.py")
eg = modul_yukle("04_yuz/emotion_v2/egit.py")
vt = modul_yukle("04_yuz/yuz_tanima/veritabani.py")
AGIR = pytest.mark.skipif(not os.environ.get("CV_AGIR_TEST"), reason="CV_AGIR_TEST=1 ile çalışır (model indirir)")


def test_ema_yumusatma():
    y = canli.Yumusatici("ema", alfa=0.5)
    assert np.allclose(y.guncelle([1, 0]), [1, 0])           # ilk kare aynen
    assert np.allclose(y.guncelle([0, 1]), [0.5, 0.5])       # 0.5*yeni + 0.5*eski
    assert np.allclose(y.guncelle([0, 1]), [0.25, 0.75])


def test_pencere_oyu_titremeyi_bastirir():
    y = canli.Yumusatici("pencere_oy", pencere=5)
    for p in ([1, 0], [1, 0], [0, 1], [1, 0], [1, 0]):        # tek karelik "titreme"
        sonuc = y.guncelle(p)
    assert sonuc.argmax() == 0 and np.isclose(sonuc[0], 0.8)


def test_esik_analizi_ayrik_dagilimlarda_sifir_hata():
    rng = np.random.default_rng(0)
    merkezler = rng.normal(size=(3, 16))
    vek, isim = [], []
    for k, m in enumerate(merkezler):
        for _ in range(4):
            v = m + rng.normal(0, 0.05, 16)
            vek.append(v / np.linalg.norm(v)); isim.append(f"k{k}")
    ea = vt.esik_analizi(np.array(isim), np.array(vek))
    assert ea["onerilen"]["FAR"] == 0 and ea["onerilen"]["FRR"] == 0
    assert ea["ayni"].min() > ea["farkli"].max()
    assert vt.histogram_gorsel(ea["ayni"], ea["farkli"], 0.5).shape[2] == 3


def test_oturuma_gore_bolme_sizinti_yok():
    ornekler = [(f"{g}_{i}.jpg", i % 2, g) for g in ("a", "b", "c", "d", "e") for i in range(6)]
    tr, val, gruplu = eg.oturuma_gore_bol(ornekler, 0.4, np.random.default_rng(0))
    assert gruplu and not ({o[2] for o in tr} & {o[2] for o in val}) and len(tr) + len(val) == 30


def test_tek_oturumda_uyari_icin_gruplu_false():
    ornekler = [(f"x_{i}.jpg", i % 2, "x") for i in range(10)]
    _, val, gruplu = eg.oturuma_gore_bol(ornekler, 0.2, np.random.default_rng(0))
    assert not gruplu and len(val) == 2


def test_kucuk_duygu_egitimi_calisir():
    pytest.importorskip("timm")
    rng = np.random.default_rng(0)
    for s, r in {"a": (0, 0, 255), "b": (255, 0, 0)}.items():
        d = VERI / "duygu_t" / s
        d.mkdir(parents=True, exist_ok=True)
        for ot in ("o1", "o2"):
            for i in range(6):
                cv2.imwrite(str(d / f"{ot}_{i}.jpg"), np.clip(rng.normal(0, 20, (48, 48, 3)) + r, 0, 255).astype(np.uint8))
    a = ayarlari_olustur(eg.Ayarlar, {"klasor": "veri/duygu_t", "isim": "t", "model": "resnet18", "onegitimli": False,
                                      "augmentation": False, "epochs": 2, "boyut": 48, "batch": 8, "lr": 0.003, "val_orani": 0.5})
    s = eg.DuyguEgit().calistir(a, Anlatici())
    assert (VERI / "modeller/duygu_t/model.pt").exists() and "karışıklık matrisi (en iyi epoch)" in s.ara_gorseller


@AGIR
def test_gercek_yuz_tespit_ve_tanima():
    import ultralytics
    from pathlib import Path
    yo = modul_yukle("04_yuz/yuz_ortak.py")
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
    yuzler = yo.yuzleri_bul(img)
    assert len(yuzler) == 2
    v = [yo.embedding(yo.hizala(img, y)) for y in yuzler]
    assert yo.kosinus(v[0], v[0]) > 0.99 and yo.kosinus(v[0], v[1]) < 0.363
