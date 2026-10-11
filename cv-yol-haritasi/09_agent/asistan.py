"""
asistan.py — "Gözleri olan kişisel asistan": araclar.py'deki araçları kullanan küçük bir Claude agent'ı.

AGENT DÖNGÜSÜ (Anthropic SDK'nın tool runner'ı bu döngüyü bizim yerimize çalıştırır):
    1. Soru + araç listesi (ad + açıklama + parametre şeması) Claude'a gönderilir.
    2. Claude ya doğrudan cevap verir ya da "where_is(nesne='anahtar') çağır" der (tool_use).
    3. Runner aracı ÇALIŞTIRIR (bizim Python fonksiyonumuz, yerelde) ve sonucu Claude'a geri verir.
    4. Claude yeterli bilgiye ulaşana kadar 2-3 tekrar eder, sonra cevabı yazar.

ŞEFFAFLIK: Her araç çağrısı (ad, argümanlar, dönen sonuç) log'a yazılır — Claude'un cevabının hangi veriye
dayandığını görürsün. Araçların kendileri araclar.py'de; şemaları fonksiyon imzalarından OTOMATİK üretilir.

KİMLİK: ANTHROPIC_API_KEY ya da `ant auth login`. Model: claude-opus-5-5 (ortak/claude_istemci.MODEL).
VERİ: Sadece araçların döndürdüğü özetler (ve describe_now çağrılırsa son kare) API'ye gider.
"""
from __future__ import annotations

import functools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak import claude_istemci as ci  # noqa: E402
from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.yukle import gerekli, modul_yukle  # noqa: E402

ar = modul_yukle("09_agent/araclar.py")
DOSYA = "09_agent/asistan.py"
SISTEM = ("Kullanıcının kişisel asistanısın. Evdeki kamera ve fotoğraf arşivinden üretilmiş bir olay hafızasına "
          "araçlarla erişebilirsin. Kullanıcının sorusunu yanıtlamak için gereken araçları kullan; araç sonuçlarında "
          "olmayan bilgiyi uydurma. Türkçe, kısa ve net cevap ver. Bugünün tarihi: {tarih}.")


def kayitli_araclar(gunluk: list[dict]):
    """araclar.ARACLAR'ı @beta_tool'a çevir; her çağrıyı 'gunluk' listesine kaydeden sarmalayıcıyla."""
    beta_tool = gerekli("anthropic", "pip install anthropic").beta_tool

    def sar(fonk):
        @functools.wraps(fonk)                     # ad, docstring ve imza korunur -> şema aynı kalır
        def sarmalayici(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                sonuc = fonk(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — hatayı Claude'a geri ver, döngü çökmesin
                sonuc = {"hata": f"{type(e).__name__}: {e}"}
            gunluk.append({"arac": fonk.__name__, "argumanlar": kwargs or list(args), "sonuc": sonuc,
                           "ms": round((time.perf_counter() - t0) * 1000)})
            return json.dumps(sonuc, ensure_ascii=False, default=str)
        return beta_tool(sarmalayici)
    return [sar(f) for f in ar.ARACLAR]


def sor(soru: str, anlat=None, client=None, effort: str = "medium") -> tuple[str, list[dict]]:
    """Asistana soru sor. Döndürür: (cevap metni, araç çağrıları günlüğü)."""
    gunluk: list[dict] = []
    client = client or ci.istemci()
    runner = client.beta.messages.tool_runner(
        model=ci.MODEL,
        max_tokens=4000,
        betas=[ci.FALLBACK_BETA],
        fallbacks="default",
        output_config={"effort": effort},
        system=SISTEM.format(tarih=time.strftime("%Y-%m-%d %H:%M")),
        tools=kayitli_araclar(gunluk),
        messages=[{"role": "user", "content": soru}],
    )
    cevap, son = "", None
    for mesaj in runner:                                     # her tur: Claude'un bir yanıtı
        son = mesaj
        for blok in mesaj.content:
            if blok.type == "tool_use" and anlat is not None:
                anlat.adim(f"Claude araç istedi: {blok.name}({json.dumps(blok.input, ensure_ascii=False)})", kod=f"09_agent/araclar.py:{blok.name}")
            elif blok.type == "text":
                cevap = blok.text
    if son is not None and son.stop_reason == "refusal":
        raise ci.ClaudeReddetti("Model isteği reddetti")
    if anlat is not None:
        for g in gunluk:
            anlat.bilgi(f"{g['arac']} → {g['ms']} ms: {str(g['sonuc'].get('cumle', g['sonuc']))[:200]}", kod=DOSYA)
    return cevap, gunluk


@dataclass
class Ayarlar:
    soru: str = ayar("Anahtarlarımı en son nerede gördün? Bugün ne kadar telefonda kaldım?", "Asistana sorun", tur="uzun_metin")
    effort: str = ayar("medium", "Düşünme derinliği", secenekler=["low", "medium", "high"],
                       etki="low: hızlı/ucuz, basit sorular için yeterli; high: çok adımlı sorularda daha titiz")


class Asistan(Eylem):
    ISIM = "Kişisel asistan (Claude + araçlar)"
    BOLUM = "09"
    DOSYA = DOSYA
    ACIKLAMA = ("Sorunu Claude'a sorar; Claude araclar.py'deki araçları (where_is, who_was_here, screen_time_today, expenses, "
                "search_photos, recent_events, describe_now) çağırarak olay hafızandan cevap üretir. Her araç çağrısı ve sonucu görünür.")
    GEREKSINIMLER = ("anthropic",)
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        cevap, gunluk = sor(a.soru, anlat=anlat, effort=a.effort)
        tablo = "\n".join(f"| `{g['arac']}` | `{json.dumps(g['argumanlar'], ensure_ascii=False)}` | {str(g['sonuc'].get('cumle', ''))[:120]} | {g['ms']} |"
                          for g in gunluk)
        return Sonuc(veri={"cevap": cevap, "arac_cagrilari": gunluk},
                     metin=f"**Soru:** {a.soru}\n\n**Cevap:** {cevap}\n\n#### Kullanılan araçlar\n| araç | argümanlar | sonuç özeti | ms |\n|---|---|---|---|\n{tablo}")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(Asistan)
