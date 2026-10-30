# email-newsletter-system

**Turn a Google Takeout mailbox into a clean newsletter contact list using a local LLM.**
Three dependency-light Python scripts that parse an `.mbox` export, strip the automated noise, and let a self-hosted Qwen model decide who is a real client. Nothing leaves your machine.

Built to recover ~40,000 emails' worth of client history from a small business inbox and produce something a newsletter tool can import. Written for a tattoo studio, but the business description is a command-line argument, so it works for any service business that lives in its inbox.

---

## Why this exists

Email providers do not give you a contact list. They give you an inbox, and after a decade that inbox is 90% receipts, notifications and vendor mail with the actual clients scattered through it. Exporting a CSV of "everyone who ever emailed you" produces a list that is mostly bounce addresses and marketing senders.

The traditional fix is keyword filters, which miss the client who wrote "hey, still thinking about that forearm piece" and catch the supply company whose subject line says "tattoo". The fix here is to let a language model read each thread and make the call, while keeping every step resumable and every decision inspectable in a CSV.

Design constraints that shaped it:

- **Local only.** Client email is private. The LLM runs on your own hardware via Ollama, llama.cpp or LM Studio; the scripts speak the OpenAI-compatible `/v1/chat/completions` API and never touch a cloud endpoint.
- **Cheap to run, cheap to stop.** Every output is append-only CSV/JSONL. Hit Ctrl+C mid-run and rerun: already-processed rows are skipped by ID.
- **No model needed for the boring part.** Parsing, deduplication, quoted-reply stripping and automated-sender detection are deterministic Python. The model only sees mail that survived the filters.
- **Pure standard library for the LLM calls.** `urllib` only, so there is no client SDK to pin or break.

## Pipeline

```
Google Takeout .mbox
        │
        ▼
 mbox_extract.py ──────────────► messages.jsonl   (cleaned human mail, in + out)
   • drops Spam/Trash labels     contacts.csv     (one row per address, with
   • flags automated senders                       received / sent_to counts,
   • strips quoted reply history                   first/last seen, sample subjects,
   • HTML → text fallback                          likely_automated flag)
        │
        ├─────────────────────────────┐
        ▼                             ▼
 classify_emails.py            classify_contacts.py
   per-email labeling            per-contact labeling
   (3 labels)                    (7 categories + potential_client bool)
        │                             │
        ▼                             ▼
 email_labels.csv              classified_contacts.csv
 tattoo_contacts.csv           potential_clients.csv
```

The two classifiers answer different questions and can be used independently:

| Script | Unit of work | Question | Good for |
|---|---|---|---|
| `classify_emails.py` | one email | Is this message a tattoo inquiry, ongoing client correspondence, or neither? | Auditing individual messages; finding inquiries you never answered |
| `classify_contacts.py` | one contact (their last N messages, both directions) | Who is this person to the business, and are they a potential client? | Building the actual newsletter list |

### What the extractor filters out before any model runs

- Anything Gmail labeled Spam or Trash
- Senders matching `no-reply`, `noreply`, `mailer-daemon`, `notifications@`, `billing@`, `receipts@`, `newsletter@` and similar
- Messages carrying `List-Unsubscribe`, `List-Id`, `Precedence: bulk/list/junk` or `Auto-Submitted` headers
- Quoted reply history (`On ... wrote:`, `-----Original Message-----`, `>` lines) so each record is only what that sender wrote
- Attachments (bodies only; text/plain preferred, HTML converted via `html2text`)

Automated senders are still counted in `contacts.csv` so you can see them, but their bodies are not stored and they are flagged `likely_automated`.

## Usage

### 1. Export your mail

Google Takeout → Mail → "All mail Including Spam and Trash" → `.mbox`. Put it in `data/raw/`.

### 2. Start a local model server

Any of these work; the scripts default to Ollama.

```bash
ollama pull qwen2.5:7b          # http://localhost:11434/v1  (default)
# llama.cpp server               # http://localhost:8080/v1
# LM Studio                      # http://localhost:1234/v1
```

### 3. Run

```bash
pip install html2text tqdm

# Parse (no LLM). Repeat --me for every address you send from.
python mbox_extract.py "data/raw/All mail Including Spam and Trash.mbox" \
    --me you@gmail.com --me you@yourdomain.com

# Test the classifier on 25 emails before committing to the full run
python classify_emails.py --model qwen2.5:7b --limit 25

# Full run; --prefilter skips mail with no tattoo-ish words (much faster, may miss a few)
python classify_emails.py --model qwen2.5:7b --prefilter

# Contact-level classification for the newsletter list
python classify_contacts.py --model qwen2.5:7b \
    --business "tattoo studio and tattoo-education courses"
```

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--base-url` | `http://localhost:11434/v1` | Point at llama.cpp or LM Studio instead |
| `--no-think` | off | Appends `/no_think` for Qwen3 models; the parser also strips `<think>` blocks if they appear |
| `--max-chars` / `--per-msg-chars` | 1500 / 800 | How much of each body the model sees |
| `--max-msgs` | 4 | Most recent messages per contact (both directions) |
| `--min-confidence` | 0.6 | Threshold for inclusion in `tattoo_contacts.csv` |
| `--min-received` | 1 | Skip contacts who never emailed you |
| `--limit` | 0 | Stop after N items, for testing |

All runs are resumable. Delete the output CSV to start a stage over.

## Model contract

Both classifiers ask the model for **JSON only**, at `temperature=0`, and validate the result in Python: a response with no JSON object, or a label outside the allowed set, is a failure that gets logged and retried on the next run rather than written as a bad row. Confidence is the model's self-report and is used only as a filter threshold; it is not calibrated.

`classify_emails.py` labels: `tattoo_inquiry` · `tattoo_correspondence` · `not_tattoo`
`classify_contacts.py` categories: `prospective_client` · `past_client` · `industry_peer` · `vendor_or_business` · `personal` · `automated_or_marketing` · `other`

## Output files

All in `data/processed/` (git-ignored — nothing under `data/` is ever committed).

| File | Produced by | Contents |
|---|---|---|
| `messages.jsonl` | extractor | One cleaned message per line: direction, date, from, to, subject, body (truncated) |
| `contacts.csv` | extractor | Every address seen, with counts and `likely_automated` flag, sorted real-people-first |
| `email_labels.csv` | email classifier | Per-email label, confidence, summary |
| `tattoo_contacts.csv` | email classifier | Per-sender rollup of tattoo-related mail, newest first |
| `classified_contacts.csv` | contact classifier | Every contact the model examined |
| `potential_clients.csv` | contact classifier | Just the `potential_client=True` rows, sorted by confidence — the newsletter import |

## Limitations

- **No ground-truth evaluation yet.** There is no hand-labeled sample to measure precision/recall against, so the quality of the list depends on spot-checking `email_labels.csv`. Adding a 200-row labeled set is the obvious next step.
- **Prefilter is lossy by design.** `--prefilter` keyword-gates mail before the model and will miss inquiries that don't use any of the listed words.
- **Message ID is a hash of sender + date + subject**, which is fine for resumability but will collide on two different emails with identical headers.
- **Single-threaded.** Throughput is bounded by the model server; a 7B model on consumer hardware handles roughly 1–3 emails/sec.
- **English-centric prompts and keyword lists.**

## Requirements

Python 3.9+, `html2text`, `tqdm`, and a local OpenAI-compatible model server. No other dependencies.

## License

MIT

---

Built by **Hannah Cowan** — [LinkedIn](https://linkedin.com/in/hannah-m-cowan)
