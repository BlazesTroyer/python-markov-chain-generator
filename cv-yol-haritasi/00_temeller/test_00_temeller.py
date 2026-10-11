"""00_temeller testleri. Çalıştır: pytest 00_temeller/"""
import cv2
import numpy as np

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.yukle import modul_yukle

met = modul_yukle("00_temeller/metrikler.py")
aug = modul_yukle("00_temeller/augment_sifirdan.py")
gors = modul_yukle("00_temeller/augmentation_gorsellestir.py")
hk = modul_yukle("00_temeller/hareket_kapisi.py")
kl = modul_yukle("00_temeller/klasik_islemler.py")


def _ornek_goruntu():
    img = np.full((240, 320, 3), 80, np.uint8)
    cv2.rectangle(img, (100, 80), (220, 180), (0, 0, 255), -1)   # kırmızı kare
    return img


# ------------------------------------------------------------------ metrikler
def test_iou_temel():
    assert met.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert met.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(met.iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-6


def test_iou_matrisi_tekli_ile_ayni():
    A = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], float)
    B = np.array([[0, 0, 10, 10], [6, 4, 18, 22], [50, 50, 60, 60]], float)
    M = met.iou_matrisi(A, B)
    for i in range(2):
        for j in range(3):
            assert abs(M[i, j] - met.iou(A[i], B[j])) < 1e-5


def test_mukemmel_tahmin_map_1():
    G = [{"kare": 0, "sinif": "a", "kutu": [0, 0, 10, 10]}, {"kare": 1, "sinif": "b", "kutu": [5, 5, 50, 50]}]
    T = [dict(g, guven=0.9) for g in G]
    s = met.map_hesapla(G, T)
    assert abs(s["mAP50"] - 1) < 1e-9 and abs(s["mAP50_95"] - 1) < 1e-9


def test_cift_tespit_fp_sayilir():
    G = [{"kare": 0, "sinif": "a", "kutu": [0, 0, 10, 10]}]
    T = [{"kare": 0, "sinif": "a", "kutu": [0, 0, 10, 10], "guven": 0.9},
         {"kare": 0, "sinif": "a", "kutu": [0, 0, 10, 10], "guven": 0.8}]
    tp, _, n = met.eslestir(G, T, "a", 0.5)
    assert tp.tolist() == [1, 0] and n == 1


def test_gurultu_map_dusurur():
    rng = np.random.default_rng(0)
    iyi = met.map_hesapla(*met.sentetik_veri(40, 0.01, 0.0, 0.0, rng))
    rng = np.random.default_rng(0)
    kotu = met.map_hesapla(*met.sentetik_veri(40, 0.2, 0.3, 2.0, rng))
    assert iyi["mAP50_95"] > kotu["mAP50_95"] + 0.2


def test_metrik_oyun_alani_calisir():
    s = met.MetrikOyunAlani().calistir(ayarlari_olustur(met.OyunAyarlari, {}), Anlatici())
    assert s.goruntu is not None and "mAP50" in s.veri


# ------------------------------------------------------------------ augmentation
def test_yatay_cevir_iki_kez_kimlik():
    img = _ornek_goruntu()
    k = np.array([[100, 80, 220, 180]], np.float32)
    rng = np.random.default_rng(0)
    i1, k1, _ = aug.yatay_cevir(img, k, rng)
    assert k1.tolist() == [[100, 80, 220, 180]] or k1[0, 0] == 320 - 220
    i2, k2, _ = aug.yatay_cevir(i1, k1, rng)
    assert np.array_equal(i2, img) and np.allclose(k2, k)


def test_ceviride_kutu_kirmizi_kareyi_takip_eder():
    img = _ornek_goruntu()
    i1, k1, _ = aug.yatay_cevir(img, np.array([[100, 80, 220, 180]], np.float32), np.random.default_rng(0))
    x1, y1, x2, y2 = k1[0].astype(int)
    merkez = i1[(y1 + y2) // 2, (x1 + x2) // 2]
    assert merkez.tolist() == [0, 0, 255]


def test_politika_kutulari_goruntu_icinde_tutar():
    img = _ornek_goruntu()
    rng = np.random.default_rng(1)
    for _ in range(30):
        yeni, k, e, _ = aug.uygula(img, [[100, 80, 220, 180]], ["kare"], "agir", rng)
        assert len(k) == len(e)
        if len(k):
            assert (k[:, [0, 2]] >= 0).all() and (k[:, [0, 2]] <= 320).all()
            assert (k[:, [1, 3]] >= 0).all() and (k[:, [1, 3]] <= 240).all()


def test_gorsellestirici_iki_mod():
    p = gors.AugmentationGorsellestir()
    for mod in ("tek_tek", "politika"):
        a = ayarlari_olustur(gors.Ayarlar, {"mod": mod, "tohum": 3, "ornek_sayisi": 4})
        s = p.isle(_ornek_goruntu(), a, Anlatici())
        assert s.goruntu is not None and len(s.ara_gorseller) >= 5


def test_gorsellestirici_albumentations():
    import pytest
    pytest.importorskip("albumentations")
    a = ayarlari_olustur(gors.Ayarlar, {"mod": "politika", "yontem": "albumentations", "tohum": 3, "ornek_sayisi": 3})
    an = Anlatici()
    s = gors.AugmentationGorsellestir().isle(_ornek_goruntu(), a, an)
    assert s.veri["ornek"] == 3 and "Örnek 1" in an.metin()


# ------------------------------------------------------------------ hareket kapısı
def test_hareket_kapisi_hareketli_kareyi_bulur():
    for yontem in ("fark", "akis"):
        p = hk.HareketKapisi()
        a = ayarlari_olustur(hk.Ayarlar, {"yontem": yontem, "min_alan": 200, "olcek": 1.0})
        bos = np.full((120, 160, 3), 50, np.uint8)
        p.isle(bos, a, Anlatici())
        s1 = p.isle(bos, a, Anlatici())
        assert s1.veri["hareket"] is False
        kare = bos.copy(); cv2.rectangle(kare, (40, 40), (90, 90), (255, 255, 255), -1)
        s2 = p.isle(kare, a, Anlatici())
        assert s2.veri["hareket"] is True
        assert s2.olaylar and s2.olaylar[0]["tur"] == "hareket_basladi"


# ------------------------------------------------------------------ klasik
def test_klasik_hepsi_calisir_ve_kirmiziyi_bulur():
    p = kl.KlasikIslemler()
    for islem in ("hsv_renk", "canny", "kontur", "adaptif"):
        s = p.isle(_ornek_goruntu(), ayarlari_olustur(kl.Ayarlar, {"islem": islem}), Anlatici())
        assert s.goruntu is not None
    s = p.isle(_ornek_goruntu(), ayarlari_olustur(kl.Ayarlar, {"islem": "hsv_renk"}), Anlatici())
    assert s.veri["kontur"] == 1
