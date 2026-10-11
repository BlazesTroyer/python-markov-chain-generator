"""08_ocr testleri. Düzleştirme, regex ayrıştırma, gider özeti ve Claude çağrısı (SAHTE istemciyle — ağ yok).
CV_AGIR_TEST=1: gerçek EasyOCR ile sentetik eğik fiş okuma."""
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ortak import claude_istemci as ci
from ortak.anlat import Anlatici
from ortak.arayuz import ayarlari_olustur
from ortak.olay_deposu import OlayDeposu
from ortak.yukle import modul_yukle

bd = modul_yukle("08_ocr/belge_duzlestir.py")
fo = modul_yukle("08_ocr/fis_okuyucu.py")


def _egik_fis():
    fis = np.full((300, 200, 3), 250, np.uint8)
    cv2.putText(fis, "TOPLAM 12,50", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    masa = np.full((500, 500, 3), (60, 90, 130), np.uint8)
    hedef = np.float32([[120, 60], [330, 100], [300, 420], [90, 380]])
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [200, 0], [200, 300], [0, 300]]), hedef)
    cv2.warpPerspective(fis, H, (500, 500), dst=masa, borderMode=cv2.BORDER_TRANSPARENT)
    return masa, hedef


def test_kose_sirala():
    p = np.array([[10, 90], [90, 10], [10, 10], [90, 90]])
    assert bd.kose_sirala(p).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_belge_bul_ve_duzlestir():
    masa, hedef = _egik_fis()
    koseler, _, yontem = bd.belge_bul(masa)
    assert yontem == "4 köşeli kontur" and np.abs(koseler - hedef).max() < 8
    duz, _ = bd.duzlestir(masa, koseler)
    assert 180 < duz.shape[1] < 240 and 280 < duz.shape[0] < 340      # oranlar ~ orijinal fiş (200x300)
    assert duz[5:20, 5:60].mean() > 200                                  # köşe artık beyaz kâğıt


@pytest.mark.parametrize("metin,beklenen", [("1.234,56", 1234.56), ("1,234.56", 1234.56), ("123,45", 123.45),
                                            ("12.50", 12.5), ("1 234,00", 1234.0)])
def test_tutar_coz(metin, beklenen):
    assert fo.tutar_coz(metin) == beklenen


def test_kural_ayristir():
    r = fo.kural_ayristir(["BIM A.S.", "09/10/26 14:22", "SUT 32,50", "EKMEK 10,00", "TOPLAM *42,50", "NAKIT 50,00"])
    assert r["magaza"] == "BIM A.S." and r["tarih"] == "2026-10-09" and r["toplam"] == 42.5
    r2 = fo.kural_ayristir(["KAFE", "LATTE 85,00", "KURABIYE 40,00"])
    assert r2["toplam"] == 85.0 and "bulunamadı" in r2["emin_degilim"]


def test_satirlara_birlestir():
    def p(x, y, m):
        return {"kutu": np.array([[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]], np.float32), "metin": m, "guven": 1.0}
    satirlar = fo.satirlara_birlestir([p(0, 0, "SUT"), p(200, 2, "32,50"), p(0, 40, "TOPLAM"), p(200, 41, "32,50")])
    assert satirlar == ["SUT 32,50", "TOPLAM 32,50"]


def test_aylik_harcama(tmp_path):
    depo = OlayDeposu(tmp_path / "o.db")
    depo.kaydet("gider", nesne="BIM", ek={"tarih": "2026-10-03", "toplam": 100, "kategori": "market"})
    depo.kaydet("gider", nesne="Kafe", ek={"tarih": "2026-10-05", "toplam": 50.5, "kategori": "kafe"})
    depo.kaydet("gider", nesne="Eski", ek={"tarih": "2026-09-30", "toplam": 999, "kategori": "market"})
    r = fo.aylik_harcama("2026-10", depo)
    assert r == {"ay": "2026-10", "fis_sayisi": 2, "kategoriler": {"kafe": 50.5, "market": 100.0}, "genel_toplam": 150.5}


class _SahteClaude:
    """anthropic istemcisini taklit eder: gönderilen isteği saklar, hazır yanıt döner."""

    def __init__(self, metin, stop_reason="end_turn"):
        self.istekler, self._metin, self._stop = [], metin, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **istek):
        self.istekler.append(istek)
        return SimpleNamespace(stop_reason=self._stop, model=istek["model"], content=[SimpleNamespace(type="text", text=self._metin)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                               stop_details=SimpleNamespace(category="cyber", explanation="test"))


def test_claude_istegi_dogru_kurulur_ve_json_doner():
    sahte = _SahteClaude(json.dumps({"a": 1}))
    an = Anlatici()
    blok = ci.goruntu_blogu(np.zeros((2000, 1000, 3), np.uint8))
    r = ci.sor([blok, {"type": "text", "text": "x"}], sistem="s", sema={"type": "object"}, anlat=an, client=sahte)
    istek = sahte.istekler[0]
    assert r == {"a": 1} and istek["model"] == "claude-opus-5-5" and istek["fallbacks"] == "default"
    assert istek["output_config"]["format"]["type"] == "json_schema" and istek["system"] == "s"
    assert "1 görüntü" in an.metin()                                    # gönderilen her şey log'da
    goruntu = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(blok["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert max(goruntu.shape[:2]) == 1280                               # büyük görüntü küçültüldü


def test_claude_reddi_sessiz_gecmez():
    with pytest.raises(ci.ClaudeReddetti):
        ci.sor("x", client=_SahteClaude("", stop_reason="refusal"))


def test_fis_okuyucu_claude_modu_sahte(monkeypatch):
    masa, _ = _egik_fis()
    monkeypatch.setattr(fo, "ocr_oku", lambda img: [{"kutu": np.array([[0, 0], [9, 0], [9, 9], [0, 9]], np.float32),
                                                       "metin": "TOPLAM 12,50", "guven": 0.9}])
    cevap = {"magaza": "Kafe X", "tarih": "2026-10-09", "toplam": 12.5, "para_birimi": "TRY", "kategori": "kafe",
             "kalemler": [], "emin_degilim": ""}
    monkeypatch.setattr(ci, "istemci", lambda: _SahteClaude(json.dumps(cevap)))
    s = fo.FisOkuyucu().isle(masa, ayarlari_olustur(fo.Ayarlar, {"mod": "ikisi"}), Anlatici())
    assert s.veri["sonuclar"]["claude"]["kategori"] == "kafe" and s.veri["sonuclar"]["kural"]["toplam"] == 12.5
    assert s.olaylar[0]["tur"] == "gider" and s.olaylar[0]["ek"]["mod"] == "claude"


@pytest.mark.skipif(not os.environ.get("CV_AGIR_TEST"), reason="CV_AGIR_TEST=1 ile çalışır (EasyOCR modeli indirir)")
def test_gercek_easyocr():
    masa, _ = _egik_fis()
    s = fo.FisOkuyucu().isle(masa, ayarlari_olustur(fo.Ayarlar, {}), Anlatici())
    assert s.veri["sonuclar"]["kural"]["toplam"] == 12.5
