"""09_agent testleri — araçlar, MCP sunucusu, asistan döngüsü ve VLM (Claude çağrıları SAHTE istemciyle)."""
import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ortak.anlat import Anlatici
from ortak.olay_deposu import OlayDeposu
from ortak.yukle import modul_yukle

ar = modul_yukle("09_agent/araclar.py")
asis = modul_yukle("09_agent/asistan.py")
vlm = modul_yukle("09_agent/vlm_yorumla.py")


@pytest.fixture
def depo():
    d = OlayDeposu()
    d.temizle()
    simdi = time.time()
    d.kaydet("nesne_goruldu", nesne="anahtar", ek={"konum": "sol alt"}, zaman=simdi - 600)
    d.kaydet("kisi_goruldu", nesne="Uğur", guven=0.7, zaman=simdi - 3600)
    d.kaydet("kisi_goruldu", nesne="Uğur", guven=0.7, zaman=simdi - 60)
    d.kaydet("tanimadik_yuz", zaman=simdi - 120)
    d.kaydet("telefon_kullanimi", ek={"sure_sn": 300}, zaman=simdi)
    d.kaydet("gider", nesne="BIM", ek={"tarih": time.strftime("%Y-%m-%d"), "toplam": 99.9, "kategori": "market"})
    yield d
    d.temizle()


def test_araclar_cumleleri(depo):
    assert "sol alt" in ar.where_is("anahtar")["cumle"]
    w = ar.who_was_here(24)
    assert w["kisiler"]["Uğur"]["kez"] == 2 and w["tanimadik_yuz"] == 1
    assert ar.screen_time_today()["toplam_dk"] == 5.0
    assert ar.expenses()["genel_toplam"] == 99.9
    assert "kisi_goruldu ×2" in ar.recent_events(son_saat=2)["cumle"]
    assert "kare yok" in ar.describe_now()["cumle"] or ar.SON_KARE.exists()


def test_arac_semalari_imzadan_uretilir():
    pytest.importorskip("anthropic")
    gunluk = []
    araclar = asis.kayitli_araclar(gunluk)
    assert [t.name for t in araclar] == [f.__name__ for f in ar.ARACLAR]
    sema = araclar[0].to_dict()["input_schema"]
    assert sema["required"] == ["nesne"] and sema["properties"]["nesne"]["type"] == "string"
    assert json.loads(araclar[0].call({"nesne": "yok_boyle_bir_sey"}))["bulundu"] is False
    assert gunluk[0]["arac"] == "where_is"            # her çağrı günlüğe yazılıyor


def test_mcp_sunucusu_araclari_kaydeder():
    pytest.importorskip("mcp")
    sunucu = modul_yukle("09_agent/mcp_server.py").sunucu_kur()
    araclar = asyncio.run(sunucu.list_tools())
    assert {t.name for t in araclar} == {f.__name__ for f in ar.ARACLAR}


def test_asistan_dongusu_sahte_runner(depo):
    pytest.importorskip("anthropic")

    class SahteRunner:
        """tool_runner'ı taklit eder: 1. tur araç ister (ve aracı çalıştırır), 2. tur cevap yazar."""

        def __init__(self, tools, **_):
            self.tools = tools

        def __iter__(self):
            yield SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(type="tool_use", name="where_is", input={"nesne": "anahtar"})])
            sonuc = json.loads(next(t for t in self.tools if t.name == "where_is").call({"nesne": "anahtar"}))
            yield SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=f"Cevap: {sonuc['cumle']}")])

    istekler = []

    def tool_runner(**kw):
        istekler.append(kw)
        return SahteRunner(**kw)

    sahte = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner)))
    an = Anlatici()
    cevap, gunluk = asis.sor("Anahtarım nerede?", anlat=an, client=sahte)
    assert "sol alt" in cevap and gunluk[0]["arac"] == "where_is"
    assert istekler[0]["model"] == "claude-opus-5-5" and len(istekler[0]["tools"]) == len(ar.ARACLAR)
    assert "Claude araç istedi: where_is" in an.metin()


def test_vlm_sahte_istemci():
    gonderilen = []

    def create(**kw):
        gonderilen.append(kw)
        return SimpleNamespace(stop_reason="end_turn", model=kw["model"], usage=None,
                               content=[SimpleNamespace(type="text", text="Masada bir kupa var.")])
    sahte = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    cevap = vlm.sahneyi_acikla(np.zeros((480, 640, 3), np.uint8), "Ne var?", client=sahte)
    icerik = gonderilen[0]["messages"][0]["content"]
    assert cevap == "Masada bir kupa var." and icerik[0]["type"] == "image" and icerik[1]["text"] == "Ne var?"
