#!/usr/bin/env python3
"""
Step 2: Use a local LLM (Qwen via Ollama, llama.cpp server, or LM Studio) to flag
potential clients among the contacts found in step 1.

Reads data/processed/messages.jsonl + contacts.csv, sends each real person's recent
emails to the model, and writes:
  classified_contacts.csv  - every contact the model looked at
  potential_clients.csv    - just the ones flagged as potential clients

Safe to stop (Ctrl+C) and rerun: contacts already classified are skipped.

Usage:
  python classify_contacts.py --model qwen2.5:7b --limit 20 \
      --business "tattoo studio and tattoo-education courses"

Endpoints (--base-url):
  Ollama     http://localhost:11434/v1   (default)
  llama.cpp  http://localhost:8080/v1
  LM Studio  http://localhost:1234/v1
"""
import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

CATEGORIES = ["prospective_client", "past_client", "industry_peer", "vendor_or_business",
              "personal", "automated_or_marketing", "other"]

SYSTEM = """You sort email contacts for a small business owner who is building a newsletter list.
The business: {business}

You will see emails exchanged with one contact. Reply with ONLY a JSON object, no other text:
{{"category": one of {cats},
  "potential_client": true or false,
  "confidence": a number from 0 to 1,
  "name": the contact's name if you can tell, otherwise "",
  "reason": one short sentence}}

potential_client is true for people who asked about, booked, bought, or showed interest in
the business's services or courses. It is false for vendors, spam, and automated mail."""

OUT_FIELDS = ["email", "name", "category", "potential_client", "confidence", "reason",
              "received", "sent_to", "last_seen"]


def call_llm(base_url, model, system, user, timeout):
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # Qwen3 reasoning blocks
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in model output")
    return json.loads(m.group(0))


def build_prompt(contact, msgs, per_msg_chars, no_think):
    lines = [f"Contact: {contact['name']} <{contact['email']}>",
             f"They emailed the owner {contact['received']} times; "
             f"the owner emailed them {contact['sent_to']} times.", ""]
    for m in msgs:
        who = "FROM CONTACT" if m["direction"] == "in" else "FROM OWNER"
        lines.append(f"--- {who} | {m['date'][:10]} | Subject: {m['subject']}")
        lines.append(m["body"][:per_msg_chars] or "(empty body)")
        lines.append("")
    if no_think:
        lines.append("/no_think")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model name as your server knows it (see `ollama list`)")
    ap.add_argument("--business", required=True, help="One line describing what you sell/offer")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--min-received", type=int, default=1,
                    help="Only contacts who emailed you at least this many times (0 = include people you only emailed)")
    ap.add_argument("--max-msgs", type=int, default=4, help="Most recent messages sent to the model per contact")
    ap.add_argument("--per-msg-chars", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N contacts (for testing)")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--no-think", action="store_true", help="Append /no_think (faster on Qwen3 models)")
    args = ap.parse_args()

    data = Path(args.data)
    out_path = data / "classified_contacts.csv"

    with open(data / "contacts.csv", encoding="utf-8") as fh:
        contacts = [r for r in csv.DictReader(fh)
                    if r["likely_automated"] != "True" and int(r["received"]) >= args.min_received]

    # Index messages by contact: what they sent you, and what you sent them
    by_contact = defaultdict(list)
    with open(data / "messages.jsonl", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if m["direction"] == "in":
                by_contact[m["from_email"]].append(m)
            else:
                for addr in m["to"]:
                    by_contact[addr].append(m)

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            done = {r["email"] for r in csv.DictReader(fh)}

    todo = [c for c in contacts if c["email"] not in done and by_contact.get(c["email"])]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(contacts)} candidate contacts, {len(done)} already done, {len(todo)} to classify now.")

    system = SYSTEM.format(business=args.business, cats=CATEGORIES)
    new_file = not out_path.exists()
    failures = 0

    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        if new_file:
            w.writeheader()
        for c in tqdm(todo, desc="Classifying", unit=" contacts"):
            msgs = sorted(by_contact[c["email"]], key=lambda m: m["date"], reverse=True)[: args.max_msgs]
            prompt = build_prompt(c, msgs, args.per_msg_chars, args.no_think)
            try:
                result = parse_json(call_llm(args.base_url, args.model, system, prompt, args.timeout))
            except urllib.error.URLError as e:
                if isinstance(getattr(e, "reason", None), ConnectionRefusedError):
                    sys.exit(f"\nCan't reach {args.base_url} - is your model server running?")
                failures += 1
                tqdm.write(f"  failed {c['email']}: {e}")
                continue
            except Exception as e:
                failures += 1
                tqdm.write(f"  failed {c['email']}: {e}")
                continue
            w.writerow({
                "email": c["email"],
                "name": result.get("name") or c["name"],
                "category": result.get("category", ""),
                "potential_client": bool(result.get("potential_client")),
                "confidence": result.get("confidence", ""),
                "reason": result.get("reason", ""),
                "received": c["received"],
                "sent_to": c["sent_to"],
                "last_seen": c["last_seen"],
            })
            fh.flush()

    # Rebuild the shortlist from everything classified so far
    with open(out_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["potential_client"] == "True"]
    rows.sort(key=lambda r: float(r["confidence"] or 0), reverse=True)
    with open(data / "potential_clients.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {failures} failures (rerun to retry them).")
    print(f"  {len(rows)} potential clients so far -> {data/'potential_clients.csv'}")


if __name__ == "__main__":
    main()
