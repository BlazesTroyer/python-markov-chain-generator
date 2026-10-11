"""
otomatik_form.py — Ayarlar dataclass'ından OTOMATİK panel kontrolleri üretir.

NEDEN VAR?
    Paneldeki her slider/checkbox, proje dosyasındaki Ayarlar sınıfındaki bir alanın
    birebir karşılığıdır. Elle ayrı bir form yazmıyoruz; böylece "panelde başka, kodda başka"
    durumu oluşamaz. Kontrolün etiketi = alan adı, altındaki bilgi = açıklama + etki notu.

EŞLEME KURALLARI (bilesen_tanimi):
    bool                          -> Checkbox
    int/float + min & max         -> Slider
    int/float (sınırsız)          -> Number
    str + secenekler              -> Dropdown
    str + tur="uzun_metin"        -> çok satırlı Textbox
    str (tur="dosya"/"klasor" dahil) -> Textbox (yol yazılır; yerel kullanım için en şeffafı)

bilesen_tanimi() gradio'ya dokunmaz (saf fonksiyon) -> birim testi kolay.
bilesenleri_olustur() gerçek gradio bileşenlerini üretir.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 'ortak' paketini bulmak için

from ortak.arayuz import AlanBilgisi, ayar_alanlari  # noqa: E402


def bilesen_tanimi(a: AlanBilgisi) -> tuple[str, dict]:
    """Bir alan için (gradio bileşen adı, parametreler) döndür."""
    etiket = a.isim
    bilgi = a.aciklama + (f"  →  {a.etki}" if a.etki else "")
    ortak = dict(label=etiket, info=bilgi, value=a.varsayilan)

    if a.tip is bool:
        return "Checkbox", ortak
    if a.tip in (int, float):
        if a.min is not None and a.max is not None:
            adim = a.adim if a.adim is not None else (1 if a.tip is int else (a.max - a.min) / 100)
            return "Slider", dict(ortak, minimum=a.min, maximum=a.max, step=adim)
        return "Number", dict(ortak, precision=0 if a.tip is int else None)
    if a.secenekler:
        return "Dropdown", dict(ortak, choices=list(a.secenekler))
    if a.tur == "uzun_metin":
        return "Textbox", dict(ortak, lines=4)
    if a.tur in ("dosya", "klasor"):
        return "Textbox", dict(ortak, placeholder=f"{a.tur} yolu (ör. veri/...)")
    return "Textbox", ortak


def bilesenleri_olustur(ayar_sinifi) -> list:
    """Gradio Blocks bağlamı içinde çağrılır; bileşenleri sırayla oluşturup listeler."""
    import gradio as gr

    bilesenler = []
    for a in ayar_alanlari(ayar_sinifi):
        ad, parametreler = bilesen_tanimi(a)
        bilesenler.append(getattr(gr, ad)(**parametreler))
    return bilesenler
