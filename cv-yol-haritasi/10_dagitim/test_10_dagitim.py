"""10_dagitim testleri. Ölçüm mantığı ve son-kare okuyucu (model gerektirmez).
CV_AGIR_TEST=1: gerçek export + tahmin uyumu (ONNX, onnxruntime gerekir)."""
import os
import time

import cv2
import numpy as np
import pytest

from ortak.yukle import modul_yukle

bm = modul_yukle("10_dagitim/benchmark.py")
k7 = modul_yukle("10_dagitim/kamera_7x24.py")


def test_olc_isinmayi_atar_ve_yuzdelik_verir():
    sayac = {"n": 0}

    def fonk():
        sayac["n"] += 1
        time.sleep(0.05 if sayac["n"] <= 2 else 0.001)        # ilk 2 çağrı yavaş (ısınma)
    s = bm.olc(fonk, isinma=2, tekrar=10)
    assert sayac["n"] == 12 and s["p50"] < 20 and s["fps"] > 50


def _video(yol, n=15):
    yaz = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(n):
        yaz.write(np.full((48, 64, 3), (i * 10) % 256, np.uint8))
    yaz.release()


def test_son_kare_okuyucu_her_kare_modu(tmp_path):
    _video(tmp_path / "v.mp4")
    ok = k7.SonKareOkuyucu(str(tmp_path / "v.mp4"), gercek_zaman=False).baslat()
    nolar = []
    t0 = time.time()
    while time.time() - t0 < 5:
        no, kare = ok.al()
        if kare is not None:
            nolar.append(no)
        elif ok.bitti:
            break
        time.sleep(0.002)
    ok.durdur()
    assert nolar == list(range(1, 16))                          # her-kare modunda hiçbiri atlanmaz


def test_son_kare_okuyucu_gercek_zamanda_eskileri_atar(tmp_path):
    _video(tmp_path / "v.mp4", n=60)
    ok = k7.SonKareOkuyucu(str(tmp_path / "v.mp4"), gercek_zaman=True).baslat()
    while not ok.bitti:
        time.sleep(0.01)
    no, kare = ok.al()
    ok.durdur()
    assert no == 60 and kare is not None                        # yavaş işleyici sadece EN SON kareyi görür


@pytest.mark.skipif(not os.environ.get("CV_AGIR_TEST"), reason="CV_AGIR_TEST=1 ile çalışır (export yapar)")
def test_gercek_onnx_export_uyumu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = modul_yukle("10_dagitim/export.py")
    from ortak.anlat import Anlatici
    from ortak.arayuz import ayarlari_olustur
    s = ex.Export().calistir(ayarlari_olustur(ex.Ayarlar, {"format": "onnx", "imgsz": 320}), Anlatici())
    assert s.veri["uyum"]["uyum_orani"] > 0.8 and s.veri["uyum"]["ort_iou"] > 0.9
