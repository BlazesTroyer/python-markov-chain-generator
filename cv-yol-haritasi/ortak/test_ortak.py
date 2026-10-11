"""ortak/ modüllerinin testleri: sözleşme, CLI, olay deposu, kaynak. Çalıştır: pytest ortak/"""
from dataclasses import dataclass

import cv2
import numpy as np

from ortak.anlat import Anlatici
from ortak.arayuz import Eylem, Sonuc, ayar, ayar_alanlari, ayarlari_olustur
from ortak.cli import _parser_olustur, ayarlari_argv, calistir
from ortak.kaynak import kareleri_oku
from ortak.olay_deposu import OlayDeposu


@dataclass
class _Ayar:
    esik: float = ayar(0.5, "eşik", min=0, max=1)
    adet: int = ayar(3, "adet")
    acik: bool = ayar(True, "açık mı")
    mod: str = ayar("a", "mod", secenekler=["a", "b"])


class _Eylem(Eylem):
    ISIM, BOLUM, DOSYA, Ayarlar = "test", "99", "x.py", _Ayar

    def calistir(self, ayarlar, anlat):
        anlat.adim("çalıştı", kod="x.py:calistir")
        return Sonuc(metin=f"{ayarlar.esik}-{ayarlar.adet}-{ayarlar.acik}-{ayarlar.mod}")


def test_ayar_alanlari_ve_tip_donusumu():
    alanlar = ayar_alanlari(_Ayar)
    assert [a.isim for a in alanlar] == ["esik", "adet", "acik", "mod"]
    assert alanlar[0].tip is float and alanlar[2].tip is bool
    a = ayarlari_olustur(_Ayar, ["0.7", "5", "false", "b"])   # panel sırayla verir
    assert (a.esik, a.adet, a.acik, a.mod) == (0.7, 5, False, "b")


def test_argv_gidis_donus_ayni_ayarlari_verir():
    a = ayarlari_olustur(_Ayar, {"esik": 0.2, "adet": 9, "acik": False, "mod": "b"})
    argv = ayarlari_argv(_Ayar, a)
    geri = ayarlari_olustur(_Ayar, vars(_parser_olustur(_Eylem).parse_args(argv)))
    assert geri == a


def test_cli_eylem_calisir(capsys):
    sonuc = calistir(_Eylem, ["--esik", "0.9", "--no-acik"])
    assert sonuc.metin == "0.9-3-False-a"
    assert "Proje   : [99] test" in capsys.readouterr().out


def test_anlatici_kod_yerleri():
    an = Anlatici()
    an.adim("a", kod="f.py:x"); an.bilgi("b", kod="f.py:x"); an.adim("c", kod="g.py:y")
    assert an.kod_yerleri() == ["f.py:x", "g.py:y"]
    assert "[2] c" in an.metin()


def test_olay_deposu(tmp_path):
    depo = OlayDeposu(tmp_path / "o.db")
    depo.kaydet("nesne_goruldu", nesne="anahtar", kutu=[1, 2, 3, 4], zaman=100)
    depo.kaydet("nesne_goruldu", nesne="Anahtar", kutu=[5, 6, 7, 8], zaman=200)
    depo.toplu_kaydet([{"tur": "jest", "nesne": "sag_kaydir"}])
    son = depo.son_gorulme("anahtar")
    assert son["kutu"] == [5, 6, 7, 8]                 # büyük/küçük harf duyarsız, en yenisi
    assert dict(depo.turler()) == {"nesne_goruldu": 2, "jest": 1}
    depo.temizle()
    assert depo.sorgula() == []


def test_kaynak_klasor_ve_video(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((20, 30, 3), i * 50, np.uint8))
    kareler = list(kareleri_oku(str(tmp_path)))
    assert len(kareler) == 3 and kareler[0][1].shape == (20, 30, 3)
