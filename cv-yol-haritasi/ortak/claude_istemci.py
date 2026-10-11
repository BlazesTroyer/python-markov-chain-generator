"""
claude_istemci.py — Claude API'ye giden BÜTÜN çağrıların geçtiği tek, küçük yardımcı.

NEDEN TEK YER?
    Şeffaflık + gizlilik: Görüntülerin/metinlerin bilgisayarından ÇIKTIĞI tek nokta burası.
    Her çağrıda ne gönderildiği (görüntü sayısı/boyutu, metin uzunluğu, şema) anlat log'una yazılır.
    Bu projede Claude'u kullanan yerler:
        08_ocr/fis_okuyucu.py   (mod="claude")  : OCR metni + fiş görüntüsü -> yapılandırılmış JSON
        09_agent/vlm_yorumla.py                 : önemli anlarda kare -> kısa sahne açıklaması
        09_agent/asistan.py                     : araçlarla (where_is...) konuşan agent

KİMLİK BİLGİSİ: Kod anahtar İÇERMEZ. anthropic.Anthropic() sırasıyla şunları dener:
    ANTHROPIC_API_KEY ortam değişkeni -> ANTHROPIC_AUTH_TOKEN -> `ant auth login` profili.

MODEL ve AYARLAR:
    MODEL = "claude-opus-5-5"  (Claude Opus 5.5)
    output_config.effort : "low" (basit çıkarım/açıklama için yeterli ve ucuz; zor işlerde artır)
    output_config.format : JSON şeması verilirse yanıt şemaya UYAN geçerli JSON olur (structured outputs)
    fallbacks="default"  : Güvenlik sınıflandırıcısı nadiren bir isteği reddederse ("refusal"), API isteği
                           uygun bir modelle otomatik yeniden dener (beta: server-side-fallback-2026-07-01).
    stop_reason == "refusal" ise ClaudeReddetti hatası atılır (sessizce boş sonuç dönmez).
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from .yukle import gerekli

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeReddetti(RuntimeError):
    """Model isteği reddetti (stop_reason='refusal'); stop_details bilgisini taşır."""


def goruntu_blogu(bgr: np.ndarray, en_uzun_kenar: int = 1280, kalite: int = 85) -> dict:
    """OpenCV görüntüsü -> Claude 'image' içerik bloğu (base64 JPEG).

    Büyük görüntüler küçültülür: daha az token = daha ucuz ve hızlı; okunabilirlik için 1280 px yeterli.
    """
    y, g = bgr.shape[:2]
    oran = min(1.0, en_uzun_kenar / max(y, g))
    if oran < 1.0:
        bgr = cv2.resize(bgr, (int(g * oran), int(y * oran)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, kalite])
    if not ok:
        raise ValueError("Görüntü JPEG'e çevrilemedi")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(jpg.tobytes()).decode("ascii")}}


def istemci():
    """anthropic.Anthropic() — kimlik bilgisi ortamdan okunur (bkz. dosya başı)."""
    return gerekli("anthropic", "pip install anthropic").Anthropic()


def sor(icerik: list[dict] | str, *, sistem: str = "", sema: dict | None = None, effort: str = "low",
        max_tokens: int = 4000, anlat=None, client=None):
    """Tek bir Claude çağrısı. sema verilirse dict, verilmezse metin döndürür.

    icerik : "metin" ya da [görüntü_blogu, {"type": "text", "text": ...}, ...]
    client : testlerde sahte istemci vermek için (verilmezse istemci() kurulur)
    """
    if isinstance(icerik, str):
        icerik = [{"type": "text", "text": icerik}]
    if anlat is not None:
        n_goruntu = sum(1 for b in icerik if b.get("type") == "image")
        kb = sum(len(b["source"]["data"]) * 3 // 4 // 1024 for b in icerik if b.get("type") == "image")
        n_metin = sum(len(b.get("text", "")) for b in icerik if b.get("type") == "text")
        anlat.adim(f"Claude'a GÖNDERİLİYOR ({MODEL}, effort={effort}): {n_goruntu} görüntü (~{kb} KB), {n_metin} karakter metin"
                   + (", JSON şeması ile" if sema else ""), kod="ortak/claude_istemci.py:sor")
    output_config: dict = {"effort": effort}
    if sema is not None:
        output_config["format"] = {"type": "json_schema", "schema": sema}
    istek = dict(
        model=MODEL,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        output_config=output_config,
        messages=[{"role": "user", "content": icerik}],
    )
    if sistem:
        istek["system"] = sistem
    client = client or istemci()
    yanit = client.beta.messages.create(**istek)
    if yanit.stop_reason == "refusal":
        detay = getattr(yanit, "stop_details", None)
        raise ClaudeReddetti(f"Model isteği reddetti: {getattr(detay, 'category', None)} — {getattr(detay, 'explanation', '')}")
    metin = "".join(b.text for b in yanit.content if b.type == "text")
    if anlat is not None:
        kullanim = getattr(yanit, "usage", None)
        anlat.adim(f"Claude yanıtı alındı ({len(metin)} karakter; giriş/çıkış token: "
                   f"{getattr(kullanim, 'input_tokens', '?')}/{getattr(kullanim, 'output_tokens', '?')}; model: {getattr(yanit, 'model', MODEL)})",
                   kod="ortak/claude_istemci.py:sor")
    return json.loads(metin) if sema is not None else metin
