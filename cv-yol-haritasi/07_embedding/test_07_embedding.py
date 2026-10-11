"""07_embedding testleri — 'renk' gömücüsüyle (model indirmeden) indeks, arama ve ROC/AUC."""
import cv2
import numpy as np

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.yukle import VERI, modul_yukle

gm = modul_yukle("07_embedding/gomucu.py")
vi = modul_yukle("07_embedding/vektor_indeksi.py")
ara = modul_yukle("07_embedding/foto_hafiza/ara.py")
idx = modul_yukle("07_embedding/foto_hafiza/indeksle.py")
os_ = modul_yukle("07_embedding/oneshot_karsilastir.py")


def test_renk_gomucu_benzerlik_mantigi():
    kirmizi = np.zeros((50, 50, 3), np.uint8); kirmizi[:] = (0, 0, 220)
    kirmizi2 = kirmizi.copy(); kirmizi2[:10] = (0, 0, 180)
    mavi = np.zeros((50, 50, 3), np.uint8); mavi[:] = (220, 0, 0)
    V = gm.gomucu_al("renk").goruntu([kirmizi, kirmizi2, mavi])
    assert np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-5)
    assert V[0] @ V[1] > V[0] @ V[2]


def test_vektor_indeksi_ekle_ara_kaydet(tmp_path):
    ind = vi.VektorIndeksi("renk")
    ind.ekle(np.eye(3, dtype=np.float32), ["a", "b", "c"], [1, 1, 1])
    ind.ekle(np.array([[0, 1, 0]], np.float32) * -1, ["b"], [2])     # aynı yol -> güncellenir, eklenmez
    assert len(ind) == 3 and ind.ara(np.array([1, 0, 0]), 1)[0] == ("a", 1.0)
    assert ind.ara(np.array([0, -1, 0]), 1)[0][0] == "b"
    ind.kaydet("t_idx")
    geri = vi.VektorIndeksi.yukle("t_idx")
    assert geri.yollar == ["a", "b", "c"] and geri.gomucu == "renk"


def test_roc_auc():
    e = np.array([1, 1, 0, 0])
    assert os_.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), e)[2] == 1.0
    assert os_.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), e)[2] == 0.0
    _, _, auc, esik, dogruluk = os_.roc_auc(np.array([0.9, 0.3, 0.5, 0.1]), e)
    assert auc == 0.75 and dogruluk == 0.75


def test_indeksle_ve_ornekle_ara():
    k = VERI / "foto_t"
    k.mkdir(parents=True, exist_ok=True)
    for i, renk in enumerate([(0, 0, 220), (0, 0, 200), (220, 0, 0), (0, 220, 0)]):
        img = np.zeros((40, 40, 3), np.uint8); img[:] = renk
        cv2.imwrite(str(k / f"{i}.png"), img)
    s = idx.Indeksle().calistir(ayarlari_olustur(idx.Ayarlar, {"klasor": "veri/foto_t", "gomucu": "renk", "isim": "t"}), Anlatici())
    assert s.veri["yeni"] == 4
    s2 = idx.Indeksle().calistir(ayarlari_olustur(idx.Ayarlar, {"klasor": "veri/foto_t", "gomucu": "renk", "isim": "t"}), Anlatici())
    assert s2.veri["yeni"] == 0                                       # artımlı: değişmeyen atlanır
    sonuc = ara.fotograf_ara("", "t_renk", 2, ornek_goruntu=str(k / "0.png"))
    assert sonuc[0][0].endswith("0.png") and sonuc[1][0].endswith("1.png")


def test_oneshot_yarisi_sentetik():
    a = ayarlari_olustur(os_.Ayarlar, {"klasor": "veri/oneshot_t", "yontemler": "renk,olmayan_yontem"})
    s = os_.OneShotKarsilastir().calistir(a, Anlatici())
    assert "renk" in s.veri["sonuclar"] and 0.0 <= s.veri["sonuclar"]["renk"]["AUC"] <= 1.0
