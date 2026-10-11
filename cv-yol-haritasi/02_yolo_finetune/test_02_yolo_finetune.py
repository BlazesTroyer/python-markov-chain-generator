"""02 testleri. Saf mantık testleri ultralytics gerektirmez.
Gerçek model testleri için:  CV_AGIR_TEST=1 pytest 02_yolo_finetune   (yolo11n.pt indirir)"""
import os
import time

import numpy as np
import pytest

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.olay_deposu import OlayDeposu
from ortak.yukle import modul_yukle

eg = modul_yukle("02_yolo_finetune/egit.py")
th = modul_yukle("02_yolo_finetune/tahmin.py")
tel = modul_yukle("02_yolo_finetune/telefon_dedektoru_v2.py")
eh = modul_yukle("02_yolo_finetune/esya_hafizasi.py")


def _csv_yaz(yol, train, val, mapler):
    basliklar = "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss\n"
    satirlar = [f"{i + 1},{t},{m + 0.1},{m},{v}" for i, (t, v, m) in enumerate(zip(train, val, mapler))]
    yol.write_text(basliklar + "\n".join(satirlar) + "\n")


def test_egitim_yorumla_overfitting_bulur(tmp_path):
    n = 12
    _csv_yaz(tmp_path / "r.csv", np.linspace(2, 0.5, n), [1.0] * 6 + list(np.linspace(1.0, 1.6, 6)), [0.3] * 5 + [0.5] + [0.45] * 6)
    o = eg.egitim_yorumla(tmp_path / "r.csv")
    assert o["overfitting"] and o["en_iyi_epoch"] == 6 and any("best.pt" in y for y in o["yorum"])


def test_egitim_yorumla_saglikli(tmp_path):
    n = 12
    _csv_yaz(tmp_path / "r.csv", np.linspace(2, 0.5, n), np.linspace(2, 0.6, n), np.linspace(0.1, 0.7, n))
    o = eg.egitim_yorumla(tmp_path / "r.csv")
    assert not o["overfitting"] and o["en_iyi_epoch"] == n


def test_nms_sifirdan():
    kutular = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], np.float32)
    skor = np.array([0.9, 0.8, 0.7, 0.95])
    sinif = np.array([0, 0, 0, 1])        # son kutu farklı sınıf -> silinmemeli
    assert sorted(th.nms_sifirdan(kutular, skor, sinif, 0.5)) == [0, 2, 3]
    assert sorted(th.nms_sifirdan(kutular, skor, sinif, 0.95)) == [0, 1, 2, 3]   # eşik yüksek -> hepsi kalır


def test_durum_makinesi_titremeyi_yutar():
    m = tel.DurumMakinesi()
    dizi = ["kulakta", "yok", "kulakta", "kulakta", "kulakta"]
    degisimler = [m.guncelle(g, 3)[0] for g in dizi]
    assert degisimler == [False, False, False, False, True] and m.durum == "kulakta"


def test_mesafe_olcekten_bagimsiz():
    kps = np.zeros((17, 2)); conf = np.ones(17)
    kps[3] = [100, 100]
    d1 = tel.en_yakin_mesafe(np.array([110, 100]), kps, conf, (3, 4), 50)
    d2 = tel.en_yakin_mesafe(np.array([220, 200]), kps * 2, conf, (3, 4), 100)   # 2 kat yakın kamera
    assert abs(d1 - d2) < 1e-6
    conf[3] = conf[4] = 0
    assert tel.en_yakin_mesafe(np.array([0, 0]), kps, conf, (3, 4), 50) == float("inf")


def test_konum_ve_esleme():
    assert eh.konum_adi([0, 0, 10, 10], 300, 300) == "sol üst"
    assert eh.konum_adi([140, 140, 160, 160], 300, 300) == "orta"
    assert eh.konum_adi([280, 280, 300, 300], 300, 300) == "sağ alt"
    assert eh.esleme_coz("cell phone=telefon, cup = kupa") == {"cell phone": "telefon", "cup": "kupa"}


def test_nerede(tmp_path):
    depo = OlayDeposu(tmp_path / "o.db")
    assert not eh.nerede("anahtar", depo)["bulundu"]
    depo.kaydet("nesne_goruldu", nesne="anahtar", ek={"konum": "sol alt"}, zaman=time.time() - 300)
    c = eh.nerede("anahtar", depo)
    assert c["bulundu"] and "sol alt" in c["cumle"] and "5 dk önce" in c["cumle"]


@pytest.mark.skipif(not os.environ.get("CV_AGIR_TEST"), reason="CV_AGIR_TEST=1 ile çalışır (model indirir)")
def test_gercek_tahmin_ve_nms_esitligi():
    import ultralytics
    from pathlib import Path
    import cv2
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    s = th.Tahmin().isle(img, ayarlari_olustur(th.Ayarlar, {"nms_goster": True, "guven": 0.1}), Anlatici())
    assert s.veri["nms"]["bizim"] == s.veri["nms"]["ultralytics"]
