"""01_veri_fabrikasi testleri: format dönüşümü ve uçtan uca hat (sentetik veri → kontrol → böl → çoğalt)."""
from pathlib import Path

import cv2
import numpy as np

from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.yukle import VERI, modul_yukle

yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
od = modul_yukle("01_veri_fabrikasi/ornek_dataset.py")
ek = modul_yukle("01_veri_fabrikasi/etiket_kontrol.py")
db = modul_yukle("01_veri_fabrikasi/dataset_bol.py")
au = modul_yukle("01_veri_fabrikasi/augment_uret.py")
kc = modul_yukle("01_veri_fabrikasi/kare_cikar.py")
oe = modul_yukle("01_veri_fabrikasi/otomatik_etiketle.py")


def test_yolo_donusum_gidis_donus(tmp_path):
    k = [32, 48, 96, 160]
    n = yf.piksel_to_yolo(k, 320, 240)
    assert np.allclose(yf.yolo_to_piksel(*n, 320, 240), k)
    yf.etiket_yaz(tmp_path / "a.txt", [k], [2], 320, 240)
    kutular, siniflar, sorun = yf.etiket_oku(tmp_path / "a.txt", 320, 240)
    assert siniflar == [2] and np.allclose(kutular[0], k, atol=0.01) and not sorun


def test_bozuk_etiket_raporlanir(tmp_path):
    (tmp_path / "b.txt").write_text("0 0.5 0.5 0.2\n1 1.5 0.5 0.2 0.2\n")
    _, siniflar, sorun = yf.etiket_oku(tmp_path / "b.txt", 100, 100)
    assert len(sorun) == 2 and siniflar == [1]


def test_etiket_yolu():
    assert yf.etiket_yolu(Path("d/images/train/x.jpg")) == Path("d/labels/train/x.txt")


def test_sinif_coz():
    assert oe.siniflari_coz("anahtar=keys, kupa") == (["anahtar", "kupa"], ["keys", "kupa"])


def test_uctan_uca_hat():
    an = Anlatici()
    od.OrnekDataset().calistir(ayarlari_olustur(od.Ayarlar, {"isim": "t", "goruntu_sayisi": 60, "grup_sayisi": 10, "boyut": 160}), an)
    rapor = ek.EtiketKontrol().calistir(ayarlari_olustur(ek.Ayarlar, {"klasor": "veri/ham/t"}), an)
    assert rapor.veri["sorunlar"] == [] and sum(rapor.veri["sinif_sayilari"].values()) > 0

    s = db.DatasetBol().calistir(ayarlari_olustur(db.Ayarlar, {"kaynak": "veri/ham/t", "isim": "t"}), an)
    assert sum(s.veri["bolmeler"].values()) == 60
    # Sızıntı yok: hiçbir grup iki bölmede birden olmamalı
    bolme_grup = {b: {db.grup_adi(p) for p in (VERI / "datasetler/t/images" / b).glob("*.jpg")}
                  for b in ("train", "val", "test")}
    assert not (bolme_grup["train"] & bolme_grup["val"]) and not (bolme_grup["train"] & bolme_grup["test"])
    assert (VERI / "datasetler/t/data.yaml").read_text().count(": daire") == 1

    a = au.AugmentUret().calistir(ayarlari_olustur(au.Ayarlar, {"dataset": "veri/datasetler/t", "politika": "agir", "kopya": 2}), an)
    assert a.veri["uretilen"] == 2 * a.veri["train_once"]
    yeni = Path(a.veri["dataset"])
    assert len(list((yeni / "images/val").glob("*.jpg"))) == len(list((VERI / "datasetler/t/images/val").glob("*.jpg")))
    rapor2 = ek.EtiketKontrol().calistir(ayarlari_olustur(ek.Ayarlar, {"klasor": str(yeni)}), an)
    assert rapor2.veri["sorunlar"] == []


def test_kare_cikar_bulanik_ve_tekrari_atar(tmp_path):
    video = tmp_path / "v.mp4"
    yaz = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for i in range(40):
        if i < 10:
            kare = np.full((120, 160, 3), 90, np.uint8)
            cv2.rectangle(kare, (20, 20), (80, 80), (255, 255, 255), -1)              # aynı kare (tekrar)
        elif i < 20:
            kare = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (31, 31), 0)  # bulanık
        else:
            kare = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)               # keskin, hep farklı
        yaz.write(kare)
    yaz.release()
    s = kc.KareCikar().calistir(ayarlari_olustur(kc.Ayarlar, {"video": str(video), "isim": "v", "her_n_kare": 1}), Anlatici())
    assert s.veri["bulanik"] >= 8 and s.veri["tekrar"] >= 8 and s.veri["kaydedilen"] >= 15
