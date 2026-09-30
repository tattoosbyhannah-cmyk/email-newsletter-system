#!/usr/bin/env python3
"""
Step 1: Parse a Google Takeout .mbox file (no LLM needed).

Outputs (in data/processed/ by default):
  messages.jsonl  - one cleaned message per line: headers + plain-text body,
                    automated/marketing mail left out
  contacts.csv    - one row per email address: name, how often they emailed you,
                    how often you emailed them, first/last seen, sample subjects

Usage:
  python src/mbox_extract.py "data/raw/All mail Including Spam and Trash.mbox" --me you@gmail.com
  (repeat --me for any other addresses/aliases you send from)
"""
import argparse
import csv
import json
import mailbox
import re
from collections import Counter, defaultdict
from datetime import timezone
from email.header import decode_header, make_header
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

import html2text
from tqdm import tqdm

# Sender addresses that are almost never a real person
AUTOMATED_ADDR = re.compile(
    r"(no-?reply|do-?not-?reply|mailer-daemon|postmaster|bounce|notifications?@|"
    r"alerts?@|updates?@|news(letter)?@|marketing@|billing@|receipts?@)",
    re.I,
)

# Where quoted reply history usually starts
QUOTE_MARKER = re.compile(r"^(On .+wrote:|-{3,}\s*Original Message\s*-{3,}|From: .+)$", re.M)

H2T = html2text.HTML2Text()
H2T.ignore_links = True
H2T.ignore_images = True
H2T.body_width = 0


def decode(value):
    """Decode MIME-encoded headers like =?UTF-8?B?...?= into readable text."""
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value))))
    except Exception:
        return str(value)


def addresses(msg, *fields):
    """Return [(name, email)] from the given header fields, lowercased emails."""
    raw = [str(v) for f in fields for v in msg.get_all(f, [])]
    out = []
    for name, addr in getaddresses(raw):
        addr = addr.strip().lower()
        if "@" in addr:
            out.append((decode(name).strip().strip('"'), addr))
    return out


def part_text(part):
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def clean(text):
    """Drop quoted reply history and '>' lines so each message is only what the sender wrote."""
    m = QUOTE_MARKER.search(text)
    if m and m.start() > 0:
        text = text[: m.start()]
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(">"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def get_body(msg):
    """Prefer the text/plain part; fall back to converting HTML to text. Skip attachments."""
    plain, html = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        if "attachment" in (part.get("Content-Disposition") or "").lower():
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(part_text(part))
        elif ctype == "text/html":
            html.append(part_text(part))
    if plain:
        return clean("\n".join(plain))
    if html:
        return clean(H2T.handle("\n".join(html)))
    return ""


def to_utc_iso(date_header):
    try:
        dt = parsedate_to_datetime(date_header)
    except Exception:
        return ""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def is_automated(msg, addr):
    if msg.get("List-Unsubscribe") or msg.get("List-Id"):
        return True
    if (msg.get("Precedence") or "").lower() in {"bulk", "list", "junk"}:
        return True
    auto = (msg.get("Auto-Submitted") or "").lower()
    if auto and auto != "no":
        return True
    return bool(AUTOMATED_ADDR.search(addr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mbox", help="Path to the Takeout .mbox file")
    ap.add_argument("--me", action="append", default=[], help="Your own address (repeat for aliases)")
    ap.add_argument("--out", default="data/processed", help="Output folder")
    ap.add_argument("--max-body", type=int, default=3000, help="Truncate each body to this many characters")
    args = ap.parse_args()

    me = {a.lower() for a in args.me}
    if not me:
        print("Warning: no --me given. Sent mail will only be detected via Gmail's 'Sent' label.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    contacts = defaultdict(lambda: {"names": Counter(), "received": 0, "sent_to": 0,
                                    "automated": 0, "first": "", "last": "", "subjects": []})
    stats = Counter()
    box = mailbox.mbox(args.mbox, create=False)

    with open(out / "messages.jsonl", "w", encoding="utf-8") as fh:
        for msg in tqdm(box, desc="Reading mbox", unit=" msgs"):
            stats["total"] += 1
            try:
                labels = [l.strip() for l in decode(msg.get("X-Gmail-Labels")).lower().split(",")]
                if "spam" in labels or "trash" in labels:
                    stats["skipped spam/trash"] += 1
                    continue

                sender = addresses(msg, "From")
                if not sender:
                    stats["no sender"] += 1
                    continue
                s_name, s_addr = sender[0]
                outgoing = s_addr in me or "sent" in labels
                date = to_utc_iso(msg.get("Date"))
                subject = decode(msg.get("Subject"))
                recipients = addresses(msg, "To", "Cc")

                if outgoing:
                    targets = [(n, a) for n, a in recipients if a not in me]
                    auto = False
                else:
                    targets = [(s_name, s_addr)]
                    auto = is_automated(msg, s_addr)

                for name, addr in targets:
                    c = contacts[addr]
                    if name and name.lower() != addr:
                        c["names"][name] += 1
                    if outgoing:
                        c["sent_to"] += 1
                    else:
                        c["received"] += 1
                        c["automated"] += int(auto)
                    if date:
                        c["first"] = date if not c["first"] else min(c["first"], date)
                        c["last"] = max(c["last"], date)
                    if subject and len(c["subjects"]) < 5 and subject not in c["subjects"]:
                        c["subjects"].append(subject)

                if auto:
                    stats["automated (body not saved)"] += 1
                    continue

                record = {
                    "direction": "out" if outgoing else "in",
                    "date": date,
                    "from_name": s_name,
                    "from_email": s_addr,
                    "to": [a for _, a in recipients],
                    "subject": subject,
                    "body": get_body(msg)[: args.max_body],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["saved"] += 1
            except Exception:
                stats["errors"] += 1

    rows = []
    for addr, c in contacts.items():
        human_received = c["received"] - c["automated"]
        rows.append({
            "email": addr,
            "name": c["names"].most_common(1)[0][0] if c["names"] else "",
            "received": c["received"],
            "sent_to": c["sent_to"],
            "likely_automated": c["sent_to"] == 0 and c["received"] > 0 and human_received == 0,
            "first_seen": c["first"],
            "last_seen": c["last"],
            "sample_subjects": " | ".join(c["subjects"]),
        })
    rows.sort(key=lambda r: (r["likely_automated"], -(r["received"] + r["sent_to"])))

    with open(out / "contacts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["email"])
        w.writeheader()
        w.writerows(rows)

    real = sum(1 for r in rows if not r["likely_automated"])
    print("\nDone.")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  unique addresses: {len(rows)}  (likely real people: {real})")
    print(f"  wrote {out/'messages.jsonl'} and {out/'contacts.csv'}")


if __name__ == "__main__":
    main()
