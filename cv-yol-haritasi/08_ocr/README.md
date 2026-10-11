# 08 · OCR — Fiş/Fatura → Gider Takip Agent'ı

Telefonla çektiğin fiş fotoğrafından **mağaza, tarih, toplam, kategori** çıkarıp gider kaydı oluşturuyoruz. Sonra
09'daki agent'a "bu ay markete ne harcadım?" diye sorabilirsin.

## Kavramlar

- **Homografi**: Bir düzlemin (fiş) başka bir düzleme (ekran) perspektif dönüşümü. 4 köşe eşleşmesi yeter → 3×3 matris.
- **OCR iki aşamadır**: (a) metin bölgelerini **tespit** et, (b) her bölgedeki karakterleri **tanı**. EasyOCR ikisini birden yapar.
- **Kural vs LLM ayrıştırma**: Regex hızlı, ücretsiz ve yereldir ama formata bağımlıdır. LLM (Claude) farklı formatlara
  dayanıklıdır ve kategori gibi anlam gerektiren alanları doldurur, ama ücretlidir ve veri dışarı çıkar.
- **Structured outputs**: Claude'a bir **JSON şeması** verilir (`FIS_SEMASI`). Yanıt bu şemaya uyan geçerli JSON olmak
  zorundadır, ayrıca "JSON'u ayrıştıramadım" hatası olmaz.

## Hat

```
fiş fotoğrafı ─▶ belge_bul: Canny → kontur → approxPolyDP (4 köşe) ─▶ kose_sirala ─▶ getPerspectiveTransform ─▶ düz fiş
                                                                                                                │
                           EasyOCR (tr+en): parçalar [kutu, metin, güven] ─▶ satirlara_birlestir ◀──────────────┘
                                                             │
                     ┌───────────────── mod=kural ───────────┴──────── mod=claude ──────────────────┐
                     ▼                                                                             ▼
          kural_ayristir (regex: TOPLAM, tarih)                 claude_istemci.sor(görüntü + OCR metni, şema)
                     └──────────────────────────────┬──────────────────────────────────────────────┘
                                                    ▼
                         olay {"tur": "gider", "ek": {magaza, tarih, toplam, kategori, ...}} ─▶ olay deposu
                                                    ▼
                         aylik_harcama("2026-10") ─▶ Gider raporu / 09 agent expenses()
```

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `belge_duzlestir.py` | Kare işleyici | Kenar → kontur → 4 köşe → homografi; ara adımlar ve H matrisi log'da |
| `fis_okuyucu.py` | Kare işleyici + Eylem | OCR + kural/Claude ayrıştırma + `gider` olayı; `GiderRaporu`; `aylik_harcama()` |
| `../ortak/claude_istemci.py` | kütüphane | Claude'a giden tek kapı: ne gönderildiği log'da; refusal kontrolü; JSON şeması |

## Çalıştır

```bash
pip install -r 08_ocr/requirements.txt
python 08_ocr/belge_duzlestir.py --kaynak fis.jpg --debug-klasor dbg/belge --anlat
python 08_ocr/fis_okuyucu.py --kaynak fis.jpg --anlat --olaylari-kaydet                   # kural modu (yerel)
export ANTHROPIC_API_KEY=...                                                                # ya da: ant auth login
python 08_ocr/fis_okuyucu.py --kaynak fis.jpg --mod ikisi --anlat --olaylari-kaydet        # kural vs Claude yan yana
python 08_ocr/fis_okuyucu.py --kaynak fisler/ --mod claude --olaylari-kaydet               # klasördeki bütün fişler
```

Panelde: **08 · OCR → Fiş okuyucu → 🖼️ Görüntü** sekmesine fiş yükle, "Olayları depoya yaz"ı işaretle, **İşle**.
Sonra **Aylık gider raporu**.

## Neye bakmalısın?

- **"OCR kutuları" ara görseli**: Kırmızı (düşük güven) kutular nerede? Genelde buruşuk/silik kısımlar → `tarama_gorunumu` dene.
- **`mod=ikisi` tablosu**: Kural nerede yanılıyor? (Ör. "TOPLAM KDV" satırını toplam sanmak, "*" işaretli tutarlar.)
- **Log'daki "Claude'a GÖNDERİLİYOR" satırı**: Kaç KB görüntü ve kaç karakter metin gittiğini her seferinde gör.
- **`emin_degilim` alanı**: Claude okuyamadığı alanı uydurmak yerine burada söylemeli.

## Deneyler

1. 10 kendi fişinle `mod=ikisi` çalıştır; her alan için kural ve Claude doğruluğunu say.
2. Düzleştirme kapalıyken (`--no-duzlestir`) OCR ortalama güveni ve doğruluk ne kadar düşüyor?
3. `kural_ayristir`'a kendi marketinin özel formatı için bir kural ekle; testine bir satır ekle.
4. Sadece OCR metnini gönder (görüntüsüz): `fis_okuyucu.py`'de görüntü bloğunu çıkar. Maliyet düşer; doğruluk ne olur?
5. Gizlilik ve maliyet: Kural modunu varsayılan bırakıp sadece `emin_degilim` dolu olduğunda Claude'a soran hibrit bir mod yaz.
