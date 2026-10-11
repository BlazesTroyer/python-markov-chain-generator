"""
ortak/ — Bütün projelerin paylaştığı küçük yardımcılar.

Buradaki her dosya kısa ve tek iş yapar; "sihirli" bir şey yoktur:

    anlat.py        -> "Şu an ne oluyor?" açıklamalarını toplayan logger (--anlat)
    arayuz.py       -> Her projenin uyduğu sözleşme: AYARLAR dataclass'ı, Sonuc, KareIsleyici, Eylem
    kaynak.py       -> Webcam / video / görüntü klasörü -> tek tip kare üreteci
    cizim.py        -> Kutu, ID, maske, yazı, ızgara çizimi
    olay_deposu.py  -> SQLite olay kaydı (agent'ın hafızası)
    cli.py          -> AYARLAR dataclass'ından otomatik komut satırı (argparse) üretir ve çalıştırır
    yukle.py        -> Rakamla başlayan klasörlerdeki (03_takip/...) modülleri yükler,
                       opsiyonel bağımlılıkları anlaşılır hata mesajıyla ister

Akış (her proje için aynı):

    kaynak.py ──kare──> Proje.isle(kare, ayarlar, anlat) ──Sonuc──> cizim / panel / CLI penceresi
                                   │
                                   └──olaylar──> olay_deposu.py ──> 09_agent/mcp_server.py
"""
