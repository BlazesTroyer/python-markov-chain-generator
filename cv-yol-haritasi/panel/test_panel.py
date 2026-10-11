"""Panel testleri: form eşlemesi, tüm projelerin yüklenmesi, panelin kurulabilmesi. Çalıştır: pytest panel/"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kayit import PROJELER, projeleri_yukle  # noqa: E402
from otomatik_form import bilesen_tanimi  # noqa: E402
from ortak.arayuz import AlanBilgisi  # noqa: E402


def _alan(**kw):
    taban = dict(isim="x", tip=int, varsayilan=1, aciklama="a", min=None, max=None, adim=None,
                 secenekler=None, tur=None, etki=None)
    taban.update(kw)
    return AlanBilgisi(**taban)


def test_form_eslemesi():
    assert bilesen_tanimi(_alan(tip=bool, varsayilan=True))[0] == "Checkbox"
    ad, p = bilesen_tanimi(_alan(min=0, max=10, etki="çok olur"))
    assert ad == "Slider" and p["step"] == 1 and "çok olur" in p["info"]
    assert bilesen_tanimi(_alan())[0] == "Number"
    assert bilesen_tanimi(_alan(tip=str, varsayilan="a", secenekler=["a", "b"]))[0] == "Dropdown"
    assert bilesen_tanimi(_alan(tip=str, varsayilan="", tur="uzun_metin"))[1]["lines"] == 4


def test_butun_projeler_yuklenir():
    projeler = projeleri_yukle()
    assert len(projeler) == len(PROJELER)
    hatalar = {p.dosya: p.hata for p in projeler if p.hata}
    assert not hatalar, hatalar
    for p in projeler:
        assert p.sinif.DOSYA == p.dosya, f"{p.dosya}: DOSYA alanı yanlış"
        assert p.sinif.BOLUM == p.bolum


def test_sonucu_panele_cevir():
    import app  # gradio gerektirir
    from ortak.anlat import Anlatici
    from ortak.arayuz import Sonuc

    an = Anlatici(); an.adim("x", kod="a.py:f")
    out = app.sonucu_panele_cevir(Sonuc(goruntu=np.zeros((4, 4, 3), np.uint8), ara_gorseller={"m": np.zeros((4, 4, 3), np.uint8)}), an, "a.py")
    assert out[0].shape == (4, 4, 3) and out[1][0][1] == "m" and "a.py:f" in out[5]


def test_panel_kurulur():
    pytest.importorskip("gradio")
    import app
    demo = app.uygulama_olustur()
    assert demo is not None
