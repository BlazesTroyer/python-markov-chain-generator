"""
app.py — Bütün projeler için tek KONTROL PANELİ (Gradio, yerel).

ÇALIŞTIR:
    cd cv-yol-haritasi
    python panel/app.py            # http://127.0.0.1:7860
    python panel/app.py --port 8000

NASIL ÇALIŞIR? (yukarıdan aşağı)
    1. kayit.py'deki PROJELER listesindeki her proje dosyası yüklenir.
    2. Her bölüm (00..10) bir sekme olur; içindeki her proje bir alt sekme.
    3. Projenin Ayarlar dataclass'ından otomatik form üretilir (otomatik_form.py).
    4. Proje türüne göre:
         KareIsleyici -> "Görüntü", "Video", "Canlı (webcam)" girişleri.
                         Her kare proje.isle(kare, ayarlar, anlat)'tan geçer.
         Eylem        -> "Çalıştır" butonu. UZUN_SURER ise ayrı süreçte (isler.py) çalışır.
    5. Her çalıştırmadan sonra şunlar görünür:
         Çıktı görüntüsü | Ara adımlar galerisi | Sonuç özeti | Veri (JSON)
         "Ne oluyor?" (anlat.py log'u) | "Kod nerede?" (çalışan dosya:fonksiyon listesi)
    6. "Olay Deposu" sekmesi agent hafızasını (SQLite) tablo olarak gösterir.
    7. "İşler" sekmesi arka planda çalışan eğitim/indeksleme işlerinin log'larını gösterir.

GİZLİLİK: server_name=127.0.0.1 ve share=False -> panel sadece senin bilgisayarından açılır.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 'ortak' için
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 'kayit', 'isler' için

import gradio as gr  # noqa: E402

from isler import IsYoneticisi  # noqa: E402
from kayit import BOLUMLER, YuklenenProje, projeleri_yukle  # noqa: E402
from otomatik_form import bilesenleri_olustur  # noqa: E402
from ortak.anlat import Anlatici  # noqa: E402
from ortak.arayuz import Eylem, KareIsleyici, Sonuc, ayarlari_olustur  # noqa: E402
from ortak.cizim import bgr2rgb, rgb2bgr  # noqa: E402
from ortak.cli import ayarlari_argv  # noqa: E402
from ortak.olay_deposu import OlayDeposu  # noqa: E402

IS_YONETICISI = IsYoneticisi()

GIRIS_METNI = """
# 🎛️ Görüntü İşleme Yol Haritası — Kontrol Paneli
Her sekme bir bölüm, her alt sekme bir proje. **Paneldeki her ayar, proje dosyasının tepesindeki
`Ayarlar` sınıfındaki bir alanın aynısıdır** (üzerindeki açıklama da oradan gelir).
Çalıştırdıktan sonra aşağıdaki **"Ne oluyor?"** kutusu adım adım ne yapıldığını,
**"Kod nerede?"** satırı hangi dosyadaki hangi fonksiyonun çalıştığını gösterir.
"""

# ============================================================================ yardımcılar


def sonucu_panele_cevir(sonuc: Sonuc | None, anlat: Anlatici, proje_dosyasi: str):
    """Sonuc nesnesini panelin 6 çıktı kutusuna dağıt (saf fonksiyon; test edilebilir)."""
    sonuc = sonuc or Sonuc()
    galeri = [(bgr2rgb(g), isim) for isim, g in sonuc.ara_gorseller.items() if g is not None]
    kod_satirlari = anlat.kod_yerleri() or [proje_dosyasi]
    kod_md = "**Kod nerede?**  " + "  →  ".join(f"`{k}`" for k in kod_satirlari)
    if sonuc.olaylar:
        kod_md += f"\n\n**Üretilen olaylar:** {len(sonuc.olaylar)} adet ({', '.join(sorted({o['tur'] for o in sonuc.olaylar}))})"
    return (
        bgr2rgb(sonuc.goruntu) if sonuc.goruntu is not None else None,
        galeri,
        sonuc.metin or "_(özet yok)_",
        sonuc.veri or {},
        anlat.metin(),
        kod_md,
    )


def _hata_ciktisi(e: Exception, anlat: Anlatici):
    anlat.uyari(f"{type(e).__name__}: {e}")
    return (None, [], f"### ⚠ Hata\n```\n{e}\n```", {"hata": str(e)},
            anlat.metin() + "\n\n" + traceback.format_exc(), "")


def _cikti_bilesenleri():
    """Her proje alt sekmesinde aynı düzende sonuç kutuları."""
    with gr.Row():
        cikti = gr.Image(label="Çıktı", type="numpy", interactive=False)
        ara = gr.Gallery(label="Ara adımlar (debug görselleri)", columns=3, height=360)
    with gr.Row():
        ozet = gr.Markdown(label="Sonuç")
        veri = gr.JSON(label="Veri (sayılar / metrikler)")
    log = gr.Textbox(label="Ne oluyor? (adım adım)", lines=12, max_lines=30)
    kod = gr.Markdown()
    return [cikti, ara, ozet, veri, log, kod]


# ============================================================================ KareIsleyici arayüzü


def _kare_isleyici_arayuzu(p: YuklenenProje):
    cls: type[KareIsleyici] = p.sinif
    durum = gr.State(None)  # bu tarayıcı oturumuna ait proje örneği (takip izleri vb. burada yaşar)

    with gr.Accordion("⚙️ Ayarlar (dosyadaki Ayarlar sınıfından)", open=True):
        ayar_bil = bilesenleri_olustur(cls.Ayarlar)
    with gr.Row():
        olay_yaz = gr.Checkbox(False, label="Olayları depoya yaz", info="Sonuc.olaylar -> veri/olaylar.db (09_agent okur)")
        sifirla = gr.Button("🔄 Durumu sıfırla (takip izleri, sayaçlar...)")

    with gr.Tabs():
        with gr.Tab("🖼️ Görüntü"):
            girdi = gr.Image(label="Girdi görüntüsü (yükle / webcam fotoğrafı)", type="numpy",
                             sources=["upload", "webcam", "clipboard"])
            isle_btn = gr.Button("İşle", variant="primary")
        with gr.Tab("🎞️ Video"):
            vgirdi = gr.Video(label="Video yükle", sources=["upload"])
            en_fazla = gr.Slider(10, 3000, value=300, step=10, label="En fazla kare")
            vbtn = gr.Button("Videoyu işle", variant="primary")
            vdurum = gr.Markdown()
            vdosya = gr.File(label="İşlenmiş video (indir)")
        with gr.Tab("📷 Canlı (webcam)"):
            gr.Markdown("Tarayıcı kameraya izin ister. Kareler sırayla `isle()`'den geçer; durum (takip izleri) korunur.")
            canli = gr.Image(label="Webcam", type="numpy", sources=["webcam"], streaming=True)

    ciktilar = _cikti_bilesenleri()

    def tek_kare(img_rgb, ornek, olay_yaz_mi, *degerler):
        anlat = Anlatici()
        if img_rgb is None:
            return (ornek, None, [], "Önce bir görüntü ver.", {}, "", "")
        try:
            ornek = ornek or cls()
            ayarlar = ayarlari_olustur(cls.Ayarlar, list(degerler))
            sonuc = ornek.isle(rgb2bgr(img_rgb), ayarlar, anlat)
            if olay_yaz_mi and sonuc.olaylar:
                OlayDeposu().toplu_kaydet(sonuc.olaylar)
            return (ornek, *sonucu_panele_cevir(sonuc, anlat, cls.DOSYA))
        except Exception as e:  # noqa: BLE001 — paneli çökertme, hatayı göster
            return (ornek, *_hata_ciktisi(e, anlat))

    def video_isle(video_yolu, ornek, olay_yaz_mi, en_fazla_kare, *degerler):
        from ortak.kaynak import VideoYazici, kareleri_oku, video_fps

        if not video_yolu:
            yield (ornek, "Önce video yükle.", None, None, [], "", {}, "", "")
            return
        anlat = Anlatici()
        try:
            ornek = cls()  # her video temiz durumla başlar
            ayarlar = ayarlari_olustur(cls.Ayarlar, list(degerler))
            cikis = Path(tempfile.mkdtemp()) / "islenmis.mp4"
            yazici = VideoYazici(cikis, video_fps(video_yolu))
            depo = OlayDeposu() if olay_yaz_mi else None
            sonuc, toplam_olay, t0 = None, 0, time.perf_counter()
            for no, kare in kareleri_oku(video_yolu, en_fazla=int(en_fazla_kare)):
                anlat.sifirla()   # log'da sadece SON karenin adımları kalsın (okunur olsun)
                sonuc = ornek.isle(kare, ayarlar, anlat)
                yazici.yaz(sonuc.goruntu if sonuc.goruntu is not None else kare)
                toplam_olay += len(sonuc.olaylar)
                if depo and sonuc.olaylar:
                    depo.toplu_kaydet(sonuc.olaylar)
                if no % 15 == 0:
                    fps = (no + 1) / (time.perf_counter() - t0)
                    yield (ornek, f"⏳ {no + 1} kare işlendi ({fps:.1f} FPS)…", None,
                           *sonucu_panele_cevir(sonuc, anlat, cls.DOSYA))
            yazici.kapat()
            durum_md = f"✔ Bitti. Toplam olay: {toplam_olay}. (Log'da son karenin adımları görünüyor.)"
            yield (ornek, durum_md, str(cikis), *sonucu_panele_cevir(sonuc, anlat, cls.DOSYA))
        except Exception as e:  # noqa: BLE001
            yield (ornek, "⚠ Hata", None, *_hata_ciktisi(e, anlat))

    girdiler = [durum, olay_yaz, *ayar_bil]
    isle_btn.click(tek_kare, [girdi, *girdiler], [durum, *ciktilar])
    vbtn.click(video_isle, [vgirdi, durum, olay_yaz, en_fazla, *ayar_bil], [durum, vdurum, vdosya, *ciktilar])
    try:  # Gradio 5+: stream_every ile kare aralığı
        canli.stream(tek_kare, [canli, *girdiler], [durum, *ciktilar], stream_every=0.15, concurrency_limit=1)
    except TypeError:
        canli.stream(tek_kare, [canli, *girdiler], [durum, *ciktilar])

    def sifirla_fn(ornek):
        if ornek is not None:
            ornek.sifirla()
        return ornek, "Durum sıfırlandı (izler/sayaçlar temizlendi)."
    sifirla.click(sifirla_fn, [durum], [durum, ciktilar[2]])


# ============================================================================ Eylem arayüzü


def _eylem_arayuzu(p: YuklenenProje):
    cls: type[Eylem] = p.sinif
    with gr.Accordion("⚙️ Ayarlar (dosyadaki Ayarlar sınıfından)", open=True):
        ayar_bil = bilesenleri_olustur(cls.Ayarlar)
    if cls.UZUN_SURER:
        gr.Markdown("ℹ️ Bu iş uzun sürebilir; **ayrı bir süreçte** çalışır. Aşağıda eşdeğer terminal komutu ve canlı log görünür. "
                    "İş, panel kapansa da `veri/isler/` altında log bırakır.")
    btn = gr.Button("▶ Çalıştır", variant="primary")
    komut_md = gr.Markdown()
    ciktilar = _cikti_bilesenleri()

    def hemen_calistir(*degerler):
        anlat = Anlatici()
        try:
            ayarlar = ayarlari_olustur(cls.Ayarlar, list(degerler))
            komut = "python " + cls.DOSYA + " " + " ".join(ayarlari_argv(cls.Ayarlar, ayarlar))
            sonuc = cls().calistir(ayarlar, anlat)
            return (f"**Eşdeğer komut:** `{komut}`", *sonucu_panele_cevir(sonuc, anlat, cls.DOSYA))
        except Exception as e:  # noqa: BLE001
            return ("", *_hata_ciktisi(e, anlat))

    def arka_planda_calistir(*degerler):
        import cv2

        try:
            ayarlar = ayarlari_olustur(cls.Ayarlar, list(degerler))
        except Exception as e:  # noqa: BLE001
            yield ("", *_hata_ciktisi(e, Anlatici()))
            return
        is_ = IS_YONETICISI.baslat(cls.ISIM, cls.DOSYA, ayarlari_argv(cls.Ayarlar, ayarlar))
        bas = f"**Eşdeğer komut** (cv-yol-haritasi/ içinden):\n```bash\n{is_.komut_metni()}\n```\nLog: `{is_.log_yolu}`"
        while is_.durum == "çalışıyor":
            yield (bas + f"\n\n⏳ **{is_.durum}**", None, [], "", {}, is_.log_sonu(), f"`{cls.DOSYA}`")
            time.sleep(1.0)
        resimler = sorted(is_.klasor.glob("*.jpg")) + sorted(is_.klasor.glob("*.png"))
        galeri = [(bgr2rgb(cv2.imread(str(r))), r.name) for r in resimler[:24]]
        yield (bas + f"\n\n**Durum:** {is_.durum}", galeri[0][0] if galeri else None, galeri,
               f"İş bitti: **{is_.durum}**. Tüm çıktılar: `{is_.klasor}`", {}, is_.log_sonu(), f"`{cls.DOSYA}`")

    btn.click(arka_planda_calistir if cls.UZUN_SURER else hemen_calistir, ayar_bil, [komut_md, *ciktilar])


# ============================================================================ özel sekmeler


def _olay_deposu_sekmesi():
    gr.Markdown("Agent'ın hafızası: projelerin ürettiği olaylar (`ortak/olay_deposu.py`, dosya: `veri/olaylar.db`).")
    with gr.Row():
        tur = gr.Textbox(label="Tür filtresi (boş = hepsi)", placeholder="nesne_goruldu, telefon_alindi, ...")
        nesne = gr.Textbox(label="Nesne/kişi filtresi", placeholder="anahtar")
        limit = gr.Slider(10, 1000, value=100, step=10, label="En fazla satır")
    with gr.Row():
        yenile = gr.Button("🔄 Yenile", variant="primary")
        temizle = gr.Button("🗑️ Tüm olayları sil")
    ozet = gr.Markdown()
    tablo = gr.Dataframe(headers=["saat", "tur", "nesne", "kimlik", "guven", "kaynak", "ek"], interactive=False, wrap=True)

    def oku(t, n, lim):
        depo = OlayDeposu()
        satirlar = depo.sorgula(tur=t or None, nesne=n or None, limit=int(lim))
        ozet_md = "**Türlere göre:** " + (", ".join(f"`{a}`: {s}" for a, s in depo.turler()) or "henüz olay yok")
        return ozet_md, [[r["saat"], r["tur"], r["nesne"], r["kimlik"], r["guven"], r["kaynak"], str(r["ek"])] for r in satirlar]

    yenile.click(oku, [tur, nesne, limit], [ozet, tablo])
    temizle.click(lambda: (OlayDeposu().temizle(), "Silindi.", [])[1:], None, [ozet, tablo])


def _isler_sekmesi():
    gr.Markdown("Arka planda çalışan/çalışmış işler (eğitim, indeksleme, export). Her işin log'u `veri/isler/<iş>/log.txt`.")
    secim = gr.Dropdown(label="İş", choices=[])
    with gr.Row():
        yenile = gr.Button("🔄 Listeyi yenile")
        durdur = gr.Button("⏹ Seçili işi durdur")
    bilgi = gr.Markdown()
    log = gr.Textbox(label="Log (son 300 satır)", lines=20)

    def listele():
        return gr.update(choices=[(f"{i.isim} — {i.durum}", i.kimlik) for i in IS_YONETICISI.listele()])

    def goster(kimlik):
        i = IS_YONETICISI.isler.get(kimlik)
        if not i:
            return "", ""
        return f"**{i.isim}** — {i.durum}\n```bash\n{i.komut_metni()}\n```", i.log_sonu()

    yenile.click(listele, None, secim)
    secim.change(goster, secim, [bilgi, log])
    durdur.click(lambda k: (IS_YONETICISI.durdur(k), *goster(k))[1:], secim, [bilgi, log])


# ============================================================================ ana uygulama


def uygulama_olustur() -> gr.Blocks:
    projeler = projeleri_yukle()
    with gr.Blocks(title="CV Yol Haritası Paneli") as demo:
        gr.Markdown(GIRIS_METNI)
        with gr.Tabs():
            for kod, baslik in BOLUMLER.items():
                bolum_projeleri = [p for p in projeler if p.bolum == kod]
                if not bolum_projeleri:
                    continue
                with gr.Tab(baslik):
                    with gr.Tabs():
                        for p in bolum_projeleri:
                            ad = p.sinif.ISIM if p.sinif else p.sinif_adi
                            with gr.Tab(ad):
                                _proje_sekmesi(p)
            with gr.Tab("🗄️ Olay Deposu"):
                _olay_deposu_sekmesi()
            with gr.Tab("🧵 İşler"):
                _isler_sekmesi()
    return demo


def _proje_sekmesi(p: YuklenenProje):
    if p.sinif is None:
        gr.Markdown(f"### ⚠ `{p.dosya}` yüklenemedi\n```\n{p.hata}\n```")
        return
    cls = p.sinif
    gr.Markdown(
        f"### {cls.ISIM}\n{cls.ACIKLAMA}\n\n"
        f"**Dosya:** `{cls.DOSYA}`"
        + (f"  ·  **Ek paketler:** `{' '.join(cls.GEREKSINIMLER)}`" if cls.GEREKSINIMLER else "")
        + f"  ·  **Tür:** {'Kare işleyici' if issubclass(cls, KareIsleyici) else 'Eylem (tek seferlik iş)'}"
    )
    if issubclass(cls, KareIsleyici):
        _kare_isleyici_arayuzu(p)
    else:
        _eylem_arayuzu(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CV Yol Haritası kontrol paneli")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="Sadece yerel: 127.0.0.1 (önerilen)")
    a = ap.parse_args()
    uygulama_olustur().queue().launch(server_name=a.host, server_port=a.port, share=False)
