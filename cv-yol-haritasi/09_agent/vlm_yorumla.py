"""
vlm_yorumla.py — Vision-Language Model (VLM): bir kareye bakıp DOĞAL DİLDE yorum yapmak (Claude vision).

MİMARİ FİKİR — "hızlı ve ucuz tetikler, yavaş ve akıllı yorumlar":
    Lokal modeller (YOLO, takip, poz) her karede çalışır: hızlı, bedava, ama sadece BİLDİKLERİNİ söyler
    ("person 0.91"). VLM ise sahneyi ANLAR ("Birisi masaya bir paket bırakıyor") ama yavaş ve ücretlidir.
    Bu yüzden VLM'i her karede değil, sadece ÖNEMLİ ANLARDA çağırırız:
        - olay_ureteci.py yeni bir kişi izi onaylandığında (bekleme süreli)
        - agent'ın describe_now() aracı çağrıldığında
        - panelde elle (bu dosyanın kare işleyicisi)

GİZLİLİK: Kare Anthropic API'ye gönderilir. ortak/claude_istemci.py her gönderimi log'a yazar.
Kimlik bilgisi: ANTHROPIC_API_KEY ya da `ant auth login` (koda gömülmez).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak import claude_istemci as ci  # noqa: E402
from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402

DOSYA = "09_agent/vlm_yorumla.py"
SISTEM = ("Bir ev kamerasının karelerini yorumlayan kişisel asistansın. Kısa ve somut ol (en fazla 3 cümle). "
          "Görmediğin şeyi uydurma; emin değilsen söyle. Kişileri yüz özelliklerinden tanımlamaya çalışma.")


def sahneyi_acikla(kare, soru: str = "Bu karede ne oluyor?", anlat=None, client=None) -> str:
    """Kareyi + soruyu Claude'a gönder, kısa açıklamayı döndür."""
    return ci.sor([ci.goruntu_blogu(kare, en_uzun_kenar=1024), {"type": "text", "text": soru}],
                  sistem=SISTEM, effort="low", max_tokens=1000, anlat=anlat, client=client)


@dataclass
class Ayarlar:
    soru: str = ayar("Bu karede ne oluyor? Dikkat çeken bir şey var mı?", "Claude'a sorulacak soru", tur="uzun_metin")


class VlmYorumla(KareIsleyici):
    ISIM = "VLM yorumla (Claude vision)"
    BOLUM = "09"
    DOSYA = DOSYA
    ACIKLAMA = ("Bir kareyi Claude'a gösterip soruyu yanıtlatır. ⚠ Görüntü API'ye gider; `ANTHROPIC_API_KEY` gerekir. "
                "**Görüntü** sekmesiyle kullan (canlı akışta her kare ücretli bir çağrı olur!).")
    GEREKSINIMLER = ("anthropic",)
    Ayarlar = Ayarlar

    def isle(self, kare, a: Ayarlar, anlat):
        cevap = sahneyi_acikla(kare, a.soru, anlat=anlat)
        cikti = yazi_yaz(kare.copy(), "Claude yorumu asagida")
        return Sonuc(goruntu=cikti, metin=f"**Soru:** {a.soru}\n\n**Claude:** {cevap}",
                     olaylar=[{"tur": "sahne_aciklamasi", "kaynak": DOSYA, "ek": {"soru": a.soru, "cevap": cevap}}])


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(VlmYorumla)
