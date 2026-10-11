"""03_takip testleri — model gerektirmez (sentetik kutular ve 'renk' dedektörü)."""
import time

import cv2
import numpy as np

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.olay_deposu import OlayDeposu
from ortak.yukle import modul_yukle

kal = modul_yukle("03_takip/kalman.py")
it = modul_yukle("03_takip/iou_takipci.py")
st = modul_yukle("03_takip/sort_sifirdan.py")
sen = modul_yukle("03_takip/sentetik_test.py")
bs = modul_yukle("03_takip/bolge_sayaci.py")
es = modul_yukle("03_takip/ekran_suresi.py")


def _kutu(x, y, w=40, h=40):
    return [x, y, x + w, y + h]


def test_kalman_hizi_ogrenir():
    k = kal.KalmanKutu(_kutu(0, 0))
    for t in range(1, 15):
        k.tahmin()
        k.guncelle(_kutu(5 * t, 2 * t))
    vx, vy = k.hiz()
    assert abs(vx - 5) < 0.5 and abs(vy - 2) < 0.5
    tahmin = k.tahmin()
    assert abs(tahmin[0] - 75) < 3          # bir sonraki karede x ≈ 5*15


def test_kalman_donusum_gidis_donus():
    k = np.array([10, 20, 50, 100.0])
    assert np.allclose(kal.x_to_kutu(np.vstack([kal.kutu_to_z(k), np.zeros((3, 1))])), k)


def test_sort_kacirmada_id_korur():
    s = st.Sort(max_yas=5, min_vurus=1, iou_esigi=0.2)
    idler = []
    for t in range(20):
        tespit = np.zeros((0, 6)) if 8 <= t <= 10 else np.array([[*_kutu(10 * t, 50), 0.9, 0]])   # 3 kare kaçırma
        izler = s.guncelle(tespit)
        idler += [iz["id"] for iz in izler]
    assert set(idler) == {1}


def test_kacirmadan_sonra_iou_koparken_sort_dayanir():
    # Kare başı 15px hareket, kutu 40px. 10-11. karelerde dedektör kaçırıyor:
    # 3 kare sonra nesne 45px ilerlemiş -> eski kutuyla örtüşme 0 -> IoU takipçi YENİ ID verir.
    # SORT, Kalman'ın öğrendiği hızla tahmini ileri taşıdığı için aynı ID ile yakalar.
    iou_t, sort_t = it.IouTakipci(0.3, 5), st.Sort(5, 1, 0.3)
    for t in range(20):
        tespit = np.zeros((0, 6)) if t in (10, 11) else np.array([[*_kutu(15 * t, 50), 0.9, 0]], np.float32)
        iou_t.guncelle(tespit)
        sort_t.guncelle(tespit)
    assert iou_t.sonraki_id - 1 == 2
    assert sort_t.sonraki_id - 1 == 1


def test_sort_ilk_hiz_siniri():
    # Belgelenen sınır: ilk kareden itibaren kare başı 30px (kutu 40px) -> ilk tahmin (hız=0) örtüşmez, iz tutunamaz.
    s = st.Sort(5, 1, 0.3)
    for t in range(10):
        s.guncelle(np.array([[*_kutu(30 * t, 50), 0.9, 0]], np.float32))
    assert s.sonraki_id - 1 >= 5
    s = st.Sort(5, 1, 0.1)   # eşiği düşürmek bu durumda kurtarır (IoU≈0.14 > 0.1)
    for t in range(10):
        s.guncelle(np.array([[*_kutu(30 * t, 50), 0.9, 0]], np.float32))
    assert s.sonraki_id - 1 == 1


def test_metrik_mukemmel_takip():
    gt = [[(1, np.array(_kutu(t, 0))), (2, np.array(_kutu(200, t)))] for t in range(10)]
    iz = [[(7, np.array(_kutu(t, 0))), (9, np.array(_kutu(200, t)))] for t in range(10)]
    m = sen.takip_metrikleri(gt, iz)
    assert m["MOTA"] == 1.0 and m["IDSW"] == 0 and m["parcalanma"] == 1.0


def test_sentetik_yaris_sort_daha_az_id_switch():
    a = ayarlari_olustur(sen.Ayarlar, {"hiz": 15, "kare_sayisi": 80, "tohum": 1})
    s = sen.SentetikTest().calistir(a, Anlatici())
    assert s.veri["SORT"]["IDSW"] < s.veri["IoU"]["IDSW"]


def test_taraf_ve_segment():
    assert bs.taraf((0, 0), (0, 10), (5, 5)) == -bs.taraf((0, 0), (0, 10), (-5, 5))
    assert bs.segment_icinde((0, 0), (0, 10), (3, 5)) and not bs.segment_icinde((0, 0), (0, 10), (3, 50))


def test_bolge_sayaci_renk_dedektoruyle():
    p = bs.BolgeSayaci()
    a = ayarlari_olustur(bs.Ayarlar, {"dedektor": "renk", "h_min": 100, "h_max": 130, "s_min": 100})
    for x in range(20, 280, 15):          # mavi kare soldan sağa geçer
        kare = np.full((200, 320, 3), 40, np.uint8)
        cv2.rectangle(kare, (x, 80), (x + 30, 110), (255, 0, 0), -1)
        s = p.isle(kare, a, Anlatici())
    assert s.veri == {"giris": 1, "cikis": 0}


def test_sort_demo_renk_modu_calisir():
    p = st.SortDemo()
    a = ayarlari_olustur(st.Ayarlar, {"dedektor": "renk"})
    kare = np.full((200, 320, 3), 40, np.uint8)
    cv2.rectangle(kare, (50, 50), (90, 90), (255, 0, 0), -1)
    s = p.isle(kare, a, Anlatici())
    assert s.goruntu is not None and len(s.ara_gorseller) == 2


def test_gunluk_ozet(tmp_path):
    depo = OlayDeposu(tmp_path / "o.db")
    simdi = time.time()
    depo.kaydet("telefon_kullanimi", ek={"sure_sn": 120}, zaman=simdi)
    depo.kaydet("telefon_kullanimi", ek={"sure_sn": 60}, zaman=simdi)
    depo.kaydet("telefon_kullanimi", ek={"sure_sn": 999}, zaman=simdi - 3 * 86400)   # eski gün sayılmamalı
    oz = es.gunluk_ozet(depo)
    assert oz["toplam_dk"] == 3.0 and oz["oturum"] == 2 and oz["en_uzun_dk"] == 2.0
