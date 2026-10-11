# 09 · Agent — Gözleri olan kişisel asistan (capstone)

Önceki bölümlerin hepsi burada birleşir: kamera karelerinden **anlamlı olaylar** üretilir, olaylar **hafızaya**
yazılır, bir LLM agent'ı da bu hafızayı **araçlarla** sorgulayıp sorularına cevap verir.

## Mimari

```
                    ┌──────────────── HER KARE (lokal, hızlı, ücretsiz) ────────────────┐
kamera ─▶ olay_ureteci.py:  hareket kapısı(00) ─▶ YOLO(02) ─▶ SORT(03) ─▶ eşya hafızası(02)
                    │            │ durgunsa atla                   │ kisi_girdi / kisi_cikti   │ nesne_goruldu
                    │            └─▶ veri/son_kare.jpg             ▼                           ▼
diğer projeler:  yüz tanıma(04) · telefon/ekran süresi(02,03) · jest/duruş/tekrar(05) · fiş(08)
                    └──────────────────────────────┬────────────────────────────────────┘
                                                   ▼
                                     ortak/olay_deposu.py (SQLite: veri/olaylar.db)
                                                   ▲
                        ┌──────────────────────────┴──────────────────────────┐
                 araclar.py  (where_is, who_was_here, screen_time_today, expenses,
                              search_photos, recent_events, describe_now)
                        │                                                      │
             mcp_server.py (MCP / stdio)                          asistan.py (Anthropic tool runner)
                        │                                                      │
          Claude Desktop / Claude Code / başka MCP istemcisi        panel / komut satırı
                                                   │
                  ÖNEMLİ ANLARDA (yavaş, ücretli, akıllı): vlm_yorumla.py ─▶ Claude vision
```

**Ana fikir — "hızlı model tetikler, akıllı model yorumlar":** YOLO/SORT/MediaPipe her karede çalışır ve sadece
*olay* üretir. LLM yalnızca sen bir şey sorduğunda ya da önemli bir anda devreye girer. Maliyet, gecikme ve gizlilik
bu sayede kontrol altında kalır.

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `araclar.py` | kütüphane | **Agent araçlarının tek kaynağı.** Docstring = LLM'in okuduğu araç açıklaması |
| `olay_ureteci.py` | Kare işleyici | Hareket kapısı + YOLO + SORT + eşya hafızası (+VLM); kişi girdi/çıktı olayları; son kare |
| `vlm_yorumla.py` | Kare işleyici + fonksiyon | `sahneyi_acikla()`: kareyi Claude'a gösterip kısa yorum |
| `mcp_server.py` | MCP sunucusu | `araclar.py`'yi MCP istemcilerine açar (mcp 1.x ve 2.x uyumlu) |
| `asistan.py` | Eylem | Tool runner ile çalışan küçük agent; her araç çağrısı ve sonucu tabloda |
| `../ortak/claude_istemci.py` | kütüphane | Claude'a giden tek kapı (model, refusal kontrolü, gönderim log'u) |

## Kurulum ve çalıştırma

```bash
pip install -r 09_agent/requirements.txt
export ANTHROPIC_API_KEY=...            # ya da: ant auth login   (sadece asistan / VLM / describe_now için)

# 1) Gözü çalıştır (olayları depoya yazarak)
python 09_agent/olay_ureteci.py --kaynak 0 --olaylari-kaydet --goster
#    (yanında istersen: 04 tani.py, 03 ekran_suresi.py, 05 ergonomi_kocu.py ... hepsi --olaylari-kaydet ile)

# 2a) Kendi asistanınla sor
python 09_agent/asistan.py --soru "Anahtarlarımı en son nerede gördün?" --anlat

# 2b) ...ya da Claude Code'a MCP sunucusu olarak ekle
claude mcp add kisisel-goz -- python /TAM/YOL/cv-yol-haritasi/09_agent/mcp_server.py
#    sonra Claude Code'da: "kisisel-goz araçlarıyla bugün ne kadar telefonda kaldığımı söyle"

python 09_agent/mcp_server.py --listele          # araçları ve açıklamalarını gör
```

Panelde: **09 · Agent** → *Olay üretici* (Canlı/Video, "Olayları depoya yaz" açık) → *Kişisel asistan* (soru yaz, Çalıştır).
**🗄️ Olay Deposu** sekmesinden agent'ın neye dayanarak cevap verdiğini satır satır görebilirsin.

## Neye bakmalısın?

- **Asistan çıktısındaki "Kullanılan araçlar" tablosu**: Claude hangi aracı hangi argümanla çağırdı, ne döndü? Cevap gerçekten bu veriye mi dayanıyor?
- **Olay üreticideki "atlanan kare" oranı**: Boş odada hareket kapısı ağır işlemlerin ne kadarını kurtarıyor?
- **Log'daki "Claude'a GÖNDERİLİYOR" satırları**: Hangi adım ne zaman veri gönderiyor?

## Deneyler

1. `araclar.py`'ye yeni bir araç ekle (ör. `posture_today()`: bugünkü `kotu_durus_uyari` sayısı). `ARACLAR` listesine eklemen yeterli: hem MCP hem asistan otomatik görür.
2. Bir aracın docstring'ini belirsizleştir ("bilgi verir"). Claude o aracı doğru zamanda çağırmayı bırakıyor mu? (Araç açıklamalarının önemi.)
3. `olay_ureteci.py`'de `--vlm` aç; kişi girişlerinde gelen sahne açıklamalarını Olay Deposu'nda incele. Hangi bekleme süresi makul?
4. Proaktif agent: Olay deposunu dakikada bir okuyup `kotu_durus_uyari` ya da `tanimadik_yuz` gelince sana bildirim (ör. masaüstü bildirimi / Telegram) gönderen küçük bir döngü yaz.
5. Jestten komuta: 05'in `jest` olaylarını okuyup komutu gerçekten çalıştıran (müzik, ışık) bir "eyleyici" yaz. Agent artık hem görüyor hem eyliyor.

## Gizlilik

Olaylar, kareler ve indeksler yerelde kalır. API'ye giden veriler yalnızca şunlardır: (a) asistana sorduğun soru ve
araçların döndürdüğü **özetler**, (b) `describe_now` / `--vlm` / VLM sekmesi kullanıldığında **kareler**.
`mcp_server.py` salt-okunurdur.
