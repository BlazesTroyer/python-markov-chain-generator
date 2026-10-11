"""05_el_poz testleri — saf hesaplar ve küçük bir GRU eğitimi (MediaPipe gerektirmez).
CV_AGIR_TEST=1: gerçek MediaPipe poz modeliyle tekrar sayacı/ergonomi duman testi."""
import os

import numpy as np
import pytest

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.yukle import VERI, modul_yukle

nk = modul_yukle("05_el_poz/noktalar.py")
ts = modul_yukle("05_el_poz/tekrar_sayaci.py")
jc = modul_yukle("05_el_poz/jest_canli.py")
je = modul_yukle("05_el_poz/jest_egit.py")


def _el(rng):
    return rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32)


def test_el_normalize_konum_olcek_taraf_bagimsiz():
    rng = np.random.default_rng(0)
    n = _el(rng)
    v = nk.el_normalize(n)
    kaydirilmis = n.copy(); kaydirilmis[:, :2] += 0.1
    assert np.allclose(v, nk.el_normalize(kaydirilmis), atol=1e-5)
    buyutulmus = n.copy(); buyutulmus = (buyutulmus - n[0]) * 2 + n[0]
    assert np.allclose(v, nk.el_normalize(buyutulmus), atol=1e-5)
    ayna = n.copy(); ayna[:, 0] = 1 - ayna[:, 0]
    assert np.allclose(v, nk.el_normalize(ayna, "Left"), atol=1e-5)


def test_nokta_augment_sekli_korur_ve_degistirir():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(30, 65)).astype(np.float32)
    a = nk.nokta_augment(d, rng)
    assert a.shape == d.shape and not np.allclose(a, d)


def test_acilar():
    assert abs(nk.aci([1, 0], np.array([0, 0]), [0, 1]) - 90) < 1e-6
    assert abs(nk.aci([-1, 0], np.array([0, 0]), [1, 0]) - 180) < 1e-6
    assert abs(nk.dikeyle_aci((0, 1), (0, 0))) < 1e-6        # tam dik
    assert abs(nk.dikeyle_aci((0, 1), (1, 0)) - 45) < 1e-6   # 45° eğik
    assert abs(nk.dikeyle_aci((0, 0), (1, 0)) - 90) < 1e-6   # yatay


def test_tekrar_sayaci_histerezis():
    s = ts.TekrarSayaci(100, 160)
    acilar = [170, 150, 120, 95, 101, 99, 105, 150, 159, 161, 158, 162, 90, 165]
    tamam = [s.guncelle(a) for a in acilar]
    assert s.sayi == 2 and sum(tamam) == 2    # 99-105 arası titreme ve 158-162 titremesi fazladan saymaz


def test_statik_el_ve_kaydirma():
    n = np.zeros((21, 3), np.float32)
    for u, p in zip(jc.PARMAK_UC, jc.PARMAK_PIP):
        n[p, 1], n[u, 1] = -0.1, -0.2        # uçlar bilekten daha uzak -> avuç
    assert jc.statik_el_durumu(n) == "avuc"
    for u, p in zip(jc.PARMAK_UC, jc.PARMAK_PIP):
        n[p, 1], n[u, 1] = -0.2, -0.05       # uçlar içe kıvrık -> yumruk
    assert jc.statik_el_durumu(n) == "yumruk"
    assert jc.kaydirma([0.2, 0.3, 0.5], 0.25) == "sag_kaydir" and jc.kaydirma([0.5, 0.45], 0.25) is None


def test_dizi_hazirla_bilegi_goreli_yapar():
    d = np.zeros((5, 65), np.float32); d[:, 63] = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert np.allclose(je.dizi_hazirla(d)[:, 63], [0, 0.1, 0.2, 0.3, 0.4])


def test_gru_yon_ogrenir():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    for etiket, yon in (("sag", 1), ("sol", -1)):
        k = VERI / "jestler_t" / etiket
        k.mkdir(parents=True, exist_ok=True)
        for ot in ("o1", "o2", "o3"):
            for i in range(10):
                d = rng.normal(0, 0.05, (16, 65)).astype(np.float32)
                d[:, 63] = 0.5 + yon * np.linspace(0, 0.3, 16) + rng.normal(0, 0.01, 16)
                np.save(k / f"{ot}_{i}.npy", d)
    a = ayarlari_olustur(je.Ayarlar, {"klasor": "veri/jestler_t", "isim": "t", "epochs": 25, "val_orani": 0.34})
    s = je.JestEgit().calistir(a, Anlatici())
    assert s.veri["val_dogruluk"] >= 0.9


@pytest.mark.skipif(not os.environ.get("CV_AGIR_TEST"), reason="CV_AGIR_TEST=1 ile çalışır (MediaPipe modeli indirir)")
def test_poz_modulleri_gercek_goruntude():
    import cv2
    import ultralytics
    from pathlib import Path
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    s = ts.TekrarSayaciDemo().isle(img, ayarlari_olustur(ts.Ayarlar, {}), Anlatici())
    assert "aci" in s.veri and s.veri["aci"] > 150           # ayakta duran kişi: diz düz
    ek = modul_yukle("05_el_poz/ergonomi_kocu.py")
    s2 = ek.ErgonomiKocu().isle(img, ayarlari_olustur(ek.Ayarlar, {}), Anlatici())
    assert s2.veri["durum"] == "iyi"
