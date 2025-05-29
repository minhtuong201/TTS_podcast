# TTS‑Podcast Pipeline (Local Prototype)

---

## 0  Quick Recap of Requirements

* **LLM**: OpenRouter → *gpt‑4o‑mini* for summarizing + script generation.
* **TTS**: Start with free‑tier or OSS voices; later switchable. Two speakers (♀, ♂) auto‑matched to detected language.
* **Runtime**: Local only (Linux/Windows). Python 3.10+.
* **Target length**: \~8‑min dialogue (≈ 900 words). Cover all main points of the PDF.
* **Output**: Single mixed MP3. (Optionally keep per‑speaker tracks.)

---

## 1  Project Layout

```text
podcast_pipeline/
├── .env                 # API keys & secrets
├── README.md            # usage & setup docs
├── requirements.txt     # pinned libs
├── src/
│   ├── main.py          # CLI entry‑point / DAG orchestrator
│   ├── pdf_ingest.py    # text extraction
│   ├── lang_detect.py   # language + script direction
│   ├── summarizer.py    # OpenRouter wrapper
│   ├── script_gen.py    # dialogue generator
│   ├── tts/
│   │   ├── base.py      # common interface (synthesize())
│   │   ├── eleven.py    # ElevenLabs Free‑Tier  (β voices)
│   │   ├── openai.py    # OpenAI TTS  (preview, 2 voices)
│   │   ├── azure.py     # Azure Neural TTS  (Free‑F0 voices)
│   │   └── coqui.py     # Coqui‑XTTS local (offline)
│   ├── audio_mixer.py   # concat & level‑match
│   └── utils/
│       └── log_cfg.py
└── tests/               # pytest smoke tests
```

---

## 2  Environment & Credentials

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit keys
```

`requirements.txt` (excerpt):

```
openrouter      # → gpt‑4o‑mini wrapper
python‑dotenv
pdfminer.six
langdetect
pydub           # audio concat
requests
```

(TTS SDKs added conditionally.)

---

## 3  Processing DAG (main.py)

1. **Extract PDF text** → `pdf_ingest.extract(path)`
2. **Detect language** → `lang_detect.detect(text)` → ISO code.
3. **Summarize**

   ```python
   summary = summarizer.llm_summary(text, target_len="short")
   ```
4. **Generate 2‑person script**

   ```python
   script = script_gen.dialogue(summary,
                                lang=iso,
                                target_words≈900)
   ```
5. **Choose TTS backend** (CLI flag or auto choose cheapest that supports `iso`).
6. **Split script by speaker** → iterate through dialogue lines.
7. **Synthesize** each line → `tts.synthesize(text, voice_id)` ⇒ WAV/MP3 snippets.
8. **Mix** in order, normalise loudness → single MP3.
9. **Write** `output/<pdfstem>_podcast.mp3`.

CLI example:

```bash
python src/main.py my_paper.pdf --tts eleven --voices female_1 male_2
```

---

## 4  TTS Back‑ends & Cost Snapshot (May 2025)

* **ElevenLabs Free‑Tier**: ‑ 10k chars ≈ 6 min audio/month. Extra \$5‑\$11 per 100k chars. Clear podcast‑style voices; multilingual.
* **OpenAI TTS (beta)**: first 10 min free; then \~\$0.015 / min (mono). Fewer voices but natural prosody; easy auth (same OpenRouter key soon).
* **Azure Neural TTS (F0)**: 0.5M chars free 30 days; then \~\$16 per 1M chars. 400+ voices. Needs Azure Sub.
* **Coqui‑XTTS local**: OSS, no cost, GPU helpful. Slightly lower quality, but unlimited.

All support MP3; ElevenLabs provides automatic breath/pauses best‑suited for conversational podcasts.

---

## 5  Prompt Design (core snippets)

```python
SUMMARIZE_PROMPT = (
  "You are an expert editor. Summarize the key ideas of the following document in under 400 words, retain technical essence, use the same language: \n\n{text}\n")

SCRIPT_PROMPT = (
  "Turn the summary into an 8‑minute dialogue between a curious host (Female) and an expert guest (Male). Use friendly tone, occasional humor, natural pauses. Each line starts with 'HOST:' or 'GUEST:'. Aim ≈900 words total.\n\nSummary:\n{summary}\n")
```

---

## 6  Audio Mixer Details

```python
segments = [AudioSegment.from_file(p) for p in snippet_paths]
combined = sum(segments)
combined.export(outfile, format="mp3", bitrate="192k")
```

---

## 7  Error Handling & Retries

* Wrap each network call with `tenacity.retry(stop=stop_after_attempt(3))`.
* Fallback TTS order: Eleven → OpenAI → Azure → Coqui.
* Log JSON per run: timings, cost, chars, failures.

---

## 8  Deployment Notes

* **Offline demo**: swap summarizer with local LLM (e.g., Phi‑3‑mini) + Coqui TTS.
* **Container**: provide `Dockerfile` with GPU optional.
* Future: Airflow DAG or FastAPI micro‑service wrapping `main.py` for batch.

---

**Next step**: approve or request edits; then I’ll flesh out code skeletons file‑by‑file.
