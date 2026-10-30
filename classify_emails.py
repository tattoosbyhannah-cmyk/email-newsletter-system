#!/usr/bin/env python3
"""
Read every email from step 1 (data/processed/messages.jsonl) and have a local LLM
(Qwen via Ollama / llama.cpp / LM Studio) decide whether it is a tattoo inquiry or
tattoo correspondence.

Outputs (in data/processed/):
  email_labels.csv     - one row per email the model read, with its label
  tattoo_contacts.csv  - one row per person who sent a tattoo-related email

Safe to stop (Ctrl+C) and rerun: emails already labeled are skipped.

Usage:
  python classify_emails.py --model qwen2.5:7b --limit 25     # test run
  python classify_emails.py --model qwen2.5:7b                # full run
  python classify_emails.py --model qwen2.5:7b --prefilter    # faster: only emails with tattoo-ish words
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

LABELS = ["tattoo_inquiry", "tattoo_correspondence", "not_tattoo"]

SYSTEM = """You read emails sent to a professional tattoo artist and label each one.

Labels:
- tattoo_inquiry: someone asking about getting a tattoo - ideas, pricing, availability,
  booking, consultations, cover-ups, touch-ups, "do you do this style?"
- tattoo_correspondence: an ongoing exchange with a tattoo client - scheduling or
  rescheduling, deposits, design revisions, reference images, aftercare questions,
  healed photos, thank-yous after a session
- not_tattoo: anything else - vendors, supply companies, spam, personal mail,
  newsletters, receipts, business admin

Reply with ONLY a JSON object, no other text:
{"label": "tattoo_inquiry" | "tattoo_correspondence" | "not_tattoo",
 "confidence": a number from 0 to 1,
 "sender_name": the sender's name if it appears in the email, otherwise "",
 "summary": one short sentence describing what they want}"""

# Used only with --prefilter to skip obviously unrelated mail before the LLM sees it
TATTOO_WORDS = re.compile(
    r"tattoo|tatt|ink|appointment|appt|booking|book (a|an|in)|consult|deposit|flash|"
    r"sleeve|piece|design|sketch|stencil|cover.?up|touch.?up|aftercare|healed|session|"
    r"placement|forearm|shoulder|ribs|back piece|quote|how much|availability|available",
    re.I,
)

OUT_FIELDS = ["id", "date", "from_email", "sender_name", "subject", "label",
              "confidence", "summary"]


def msg_id(m):
    key = f"{m['from_email']}|{m['date']}|{m['subject']}"
    return hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:16]


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
    data = json.loads(m.group(0))
    if data.get("label") not in LABELS:
        raise ValueError(f"unexpected label: {data.get('label')!r}")
    return data


def build_prompt(m, max_chars, no_think):
    prompt = (f"From: {m['from_name']} <{m['from_email']}>\n"
              f"Date: {m['date'][:10]}\n"
              f"Subject: {m['subject']}\n\n"
              f"{m['body'][:max_chars] or '(empty body)'}")
    return prompt + ("\n\n/no_think" if no_think else "")


def write_contacts(labels_path, out_path, min_conf):
    people = defaultdict(lambda: {"name": "", "inquiries": 0, "correspondence": 0,
                                  "first": "", "last": "", "latest_summary": ""})
    with open(labels_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["label"] == "not_tattoo" or float(r["confidence"] or 0) < min_conf:
                continue
            p = people[r["from_email"]]
            p["name"] = p["name"] or r["sender_name"]
            p["inquiries" if r["label"] == "tattoo_inquiry" else "correspondence"] += 1
            p["first"] = r["date"] if not p["first"] else min(p["first"], r["date"])
            if r["date"] >= p["last"]:
                p["last"], p["latest_summary"] = r["date"], r["summary"]

    rows = [{"email": e, **p} for e, p in people.items()]
    rows.sort(key=lambda r: r["last"], reverse=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["email", "name", "inquiries", "correspondence",
                                           "first", "last", "latest_summary"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model name as your server knows it (see `ollama list`)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="Ollama :11434/v1, llama.cpp :8080/v1, LM Studio :1234/v1")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--max-chars", type=int, default=1500, help="Characters of each body sent to the model")
    ap.add_argument("--min-confidence", type=float, default=0.6,
                    help="Minimum confidence to count an email in tattoo_contacts.csv")
    ap.add_argument("--prefilter", action="store_true",
                    help="Only send emails containing tattoo-related words to the LLM (much faster, may miss a few)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N emails (for testing)")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--no-think", action="store_true", help="Append /no_think (faster on Qwen3 models)")
    args = ap.parse_args()

    data = Path(args.data)
    labels_path = data / "email_labels.csv"

    done = set()
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as fh:
            done = {r["id"] for r in csv.DictReader(fh)}

    todo, skipped_filter = [], 0
    with open(data / "messages.jsonl", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if m["direction"] != "in":          # only emails people sent to you
                continue
            m["id"] = msg_id(m)
            if m["id"] in done:
                continue
            if args.prefilter and not TATTOO_WORDS.search(m["subject"] + " " + m["body"]):
                skipped_filter += 1
                continue
            todo.append(m)
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(done)} already labeled, {len(todo)} to label now"
          + (f", {skipped_filter} skipped by prefilter" if args.prefilter else "") + ".")

    new_file = not labels_path.exists()
    counts, failures = defaultdict(int), 0

    with open(labels_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        if new_file:
            w.writeheader()
        for m in tqdm(todo, desc="Reading emails", unit=" emails"):
            try:
                result = parse_json(call_llm(args.base_url, args.model, SYSTEM,
                                             build_prompt(m, args.max_chars, args.no_think),
                                             args.timeout))
            except urllib.error.URLError as e:
                if isinstance(getattr(e, "reason", None), ConnectionRefusedError):
                    sys.exit(f"\nCan't reach {args.base_url} - is your model server running?")
                failures += 1
                tqdm.write(f"  failed: {m['subject'][:60]!r}: {e}")
                continue
            except Exception as e:
                failures += 1
                tqdm.write(f"  failed: {m['subject'][:60]!r}: {e}")
                continue

            counts[result["label"]] += 1
            w.writerow({
                "id": m["id"],
                "date": m["date"],
                "from_email": m["from_email"],
                "sender_name": result.get("sender_name") or m["from_name"],
                "subject": m["subject"],
                "label": result["label"],
                "confidence": result.get("confidence", ""),
                "summary": result.get("summary", ""),
            })
            fh.flush()
            if result["label"] != "not_tattoo":
                tqdm.write(f"  [{result['label']}] {m['from_email']}: {result.get('summary', '')[:80]}")

    n = write_contacts(labels_path, data / "tattoo_contacts.csv", args.min_confidence)
    print("\nThis run:", dict(counts), f"| failures: {failures} (rerun to retry)")
    print(f"{n} tattoo contacts -> {data/'tattoo_contacts.csv'}")


if __name__ == "__main__":
    main()
