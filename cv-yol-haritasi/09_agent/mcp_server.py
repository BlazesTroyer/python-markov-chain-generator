"""
mcp_server.py — Kişisel görüntü işleme hafızanı bir MCP SUNUCUSU olarak dışarı aç.

MCP (Model Context Protocol) NEDİR?
    LLM uygulamalarının (Claude Desktop, Claude Code, kendi agent'ların...) dış araçlara standart bir
    yolla bağlanmasını sağlayan protokol. Bu dosya araclar.py'deki fonksiyonları "araç" olarak sunar;
    istemci (ör. Claude Code) bunları listeler, açıklamalarını okur ve gerektiğinde çağırır.

    "Anahtarlarımı en son nerede gördün?"
        -> Claude, where_is aracının açıklamasını okur ve where_is(nesne="anahtar") çağırır
        -> bu sunucu olay deposunu sorgular -> {"cumle": "'anahtar' en son 18:42'de ... sol alt ..."}
        -> Claude cevabı sana kendi cümleleriyle söyler.

ÇALIŞTIRMA (stdio: istemci bu süreci kendisi başlatır, stdin/stdout üzerinden konuşur):
    Claude Code:     claude mcp add kisisel-goz -- python /TAM/YOL/cv-yol-haritasi/09_agent/mcp_server.py
    Claude Desktop:  claude_desktop_config.json ->
                     {"mcpServers": {"kisisel-goz": {"command": "python",
                                                     "args": ["/TAM/YOL/cv-yol-haritasi/09_agent/mcp_server.py"]}}}
    Elle deneme:     python 09_agent/mcp_server.py --listele      (araçları ve açıklamalarını yazar, çıkar)

GÜVENLİK: Bu sunucu SADECE OKUR (olay deposu, indeksler). Hiçbir aracı bir şeyi silmez/değiştirmez.
          describe_now ve search_photos dışındaki araçlar internete hiçbir şey göndermez.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak.yukle import gerekli, modul_yukle  # noqa: E402

ar = modul_yukle("09_agent/araclar.py")


def sunucu_kur():
    """MCP sunucusunu kur ve araclar.ARACLAR'daki her fonksiyonu araç olarak kaydet."""
    gerekli("mcp", "pip install mcp")
    try:                                   # mcp 2.x
        from mcp.server.mcpserver import MCPServer
    except ImportError:                    # mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer
    sunucu = MCPServer("kisisel-goz", instructions=(
        "Kullanıcının kişisel kamera/fotoğraf hafızası. Eşyaların yeri, kimin geldiği, ekran süresi, harcamalar, "
        "fotoğraf arama ve son olaylar için bu araçları kullan. Sonuçlardaki 'cumle' alanı hazır özettir."))
    for fonk in ar.ARACLAR:
        sunucu.tool()(fonk)               # @sunucu.tool() dekoratörünün elle uygulanmış hali (şema imzadan üretilir)
    return sunucu


if __name__ == "__main__":
    if "--listele" in sys.argv:
        for f in ar.ARACLAR:
            print(f"- {f.__name__}: {(f.__doc__ or '').strip().splitlines()[0]}")
        if "--dene" in sys.argv:
            print(json.dumps(ar.recent_events(), ensure_ascii=False, indent=2, default=str))
        sys.exit(0)
    sunucu_kur().run(transport="stdio")
