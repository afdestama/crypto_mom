# telegram_bot.py — One-shot Telegram signal sender untuk crypto factor scanner.
# Dipanggil oleh Railway Cron tiap 4 jam (UTC). Tidak ada loop internal.

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone

# Load .env untuk dev lokal; di Railway env sudah di-inject jadi no-op
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from main import main as run_scanner
import config

TOKEN   = os.environ['TELEGRAM_BOT_TOKEN']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
TOP_N   = int(os.environ.get('TOP_N', '20'))
TG_URL  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

TG_MAX_LEN = 4000  # Telegram max 4096 chars; sisakan buffer untuk safety


def send(text: str):
    if len(text) > TG_MAX_LEN:
        text = text[:TG_MAX_LEN] + "\n…(truncated)"
    r = requests.post(TG_URL, data={
        'chat_id': CHAT_ID,
        'text': text,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True,
    }, timeout=20)
    r.raise_for_status()


def format_message(ranking: pd.DataFrame) -> str:
    ts = ranking['datetime'].iloc[0]

    lines = []
    lines.append(f"<b>📊 TOP {TOP_N}</b> · 4H close {ts}")
    lines.append("<pre>")
    lines.append(" # Symbol     Score  Dir   Conv   Fund     L/S   OI(M)")
    for _, r in ranking.head(TOP_N).iterrows():
        sym  = r['symbol'].replace('/USDT:USDT', '')
        d    = r.get('direction', 'WAIT')
        icon = '▲' if d == 'LONG' else '▼' if d == 'SHORT' else '─'
        fr   = r.get('funding_rate_raw')
        ls   = r.get('ls_ratio_raw')
        oi   = r.get('oi_raw')
        fr_s = f"{fr*100:+.4f}%" if pd.notna(fr) else '   N/A '
        ls_s = f"{ls:.2f}"       if pd.notna(ls) else ' N/A'
        oi_s = f"{oi/1e6:>5.1f}" if pd.notna(oi) else '  N/A'
        lines.append(
            f"{int(r['rank']):>2} {sym:<10} "
            f"{r['composite_score']:+.2f}  "
            f"{icon}{d:<4} "
            f"{r['conviction']:+.2f}  "
            f"{fr_s:>7}  {ls_s:>4}  {oi_s}"
        )
    lines.append("</pre>")

    ls_col = ranking.get('ls_ratio_raw')
    ls_ok  = ls_col.isna() | (ls_col < config.LS_EXTREME_THRESH) if ls_col is not None else True
    buy = ranking[(ranking['direction'] == 'LONG') & (ranking['conviction'] > 0.5) & ls_ok]
    if not buy.empty:
        lines.append(f"\n<b>▲ BELI (conv &gt; 0.5, L/S &lt; {config.LS_EXTREME_THRESH})</b>")
        for _, r in buy.iterrows():
            sym = r['symbol'].replace('/USDT:USDT', '')
            fr  = r.get('funding_rate_raw')
            ls  = r.get('ls_ratio_raw')
            fr_s = f"{fr*100:+.4f}%" if pd.notna(fr) else 'N/A'
            ls_s = f"{ls:.2f}"       if pd.notna(ls) else 'N/A'
            lines.append(
                f"• <code>{sym}</code> conv {r['conviction']:+.2f}  "
                f"fr {fr_s}  L/S {ls_s}"
            )

    counts = ranking['direction'].value_counts().to_dict()
    lines.append(
        f"\nCounts: LONG {counts.get('LONG', 0)} · "
        f"WAIT {counts.get('WAIT', 0)}"
    )
    return "\n".join(lines)


def main():
    try:
        results = run_scanner(force_refresh=False)
        ranking = (results or {}).get('latest_ranking')
        if ranking is None or ranking.empty:
            send(
                f"⚠ Scan {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — "
                f"tidak ada coin lolos gate."
            )
            return 0
        send(format_message(ranking))
        return 0
    except Exception as e:
        err = f"❌ Scanner error: <code>{type(e).__name__}: {e}</code>"
        try:
            send(err)
        except Exception:
            pass
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
