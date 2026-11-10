# Nyaya — Complete Product Demo Video Script (Loom)

**Target Video Duration:** 5 to 6 minutes  
**Format:** Screen recording with webcam bubble in the bottom corner  
**Presenter:** Engineering / Product Lead  
**Tone:** Confident, mission-driven, technically rigorous, and user-centric  

---

## Pre-Recording Checklist & Setup

1. **Browser Tab 1**: Nyaya Web Application running at `http://127.0.0.1:8000` (Clean browser window, zoomed to ~110% for legibility).
2. **Browser Tab 2 / PDF Viewer**: Have a pre-downloaded or ready-to-open generated legal notice PDF.
3. **Terminal Window**: Prepared terminal in the project directory ready to show `.venv/bin/python -m pytest -q` and `evals/run_evals.py`.
4. **Microphone**: Working headset or desktop microphone with low background noise.
5. **Initial State**: Clear any previous session so you start on a fresh, clean conversation (`Click "New" or refresh`).

---

## Scene-by-Scene Script

### Scene 1: The Problem & Product Thesis (0:00 – 0:45)
**Visual Setup:**  
* Webcam centered or bottom-left bubble.  
* Screen shows the clean Nyaya interface (`http://127.0.0.1:8000`) with the glowing microphone orb, language indicator, and quick-start legal topic chips.

**Spoken Narration:**
> "Hi everyone. Over 40 million cases are pending in Indian courts today, and hundreds of millions of citizens face everyday legal disputes — from unpaid wages and arbitrary tenant evictions to police refusing to register an FIR or cheque bounce issues.
>
> But for the average citizen, the legal system is completely inaccessible: legal counsel is expensive, the jargon is intimidating, and services rarely work in their mother tongue. 
>
> When people turn to generic AI chatbots, the results are dangerous: LLMs routinely hallucinate non-existent statutory sections or deliver ungrounded advice.
>
> That’s why we built **Nyaya** — an open, multilingual, voice-first citizen legal advocate designed specifically for India. 
>
> Nyaya is governed by an uncompromising engineering invariant: **no legal proposition ever reaches a user without a verbatim quote from an approved, in-date statutory source.** Let’s see it in action."

---

### Scene 2: Voice-First, Multilingual Interaction & Real Barge-In (0:45 – 2:00)
**Visual Setup:**  
* Screen focused on the web interface.  
* Click the microphone orb. Speak a natural query in Hindi or Telugu (or code-mixed English).  
* Show real-time streaming transcription in the composer.  
* While Nyaya is speaking back via TTS, speak or tap to demonstrate instantaneous local barge-in.

**Action:**
1. Click the **Mic Orb** (or type if audio environment is noisy).
2. Speak:  
   *"Mera landlord deposit refund nahi kar raha hai, 6 mahine ho gaye hain aur ab phone bhi nahi utha raha. Mujhe kya karna chahiye?"*  
   *(Or in Telugu: "నా యజమాని నా సెక్యూరిటీ డిపాజిట్ తిరిగి ఇవ్వడం లేదు...")*
3. Watch the live transcript appear with auto-detected language (`hi-en` / `te`).
4. As the synthesized voice begins speaking the answer, say firmly: *"Wait, can I send him a legal notice?"*
5. Show that the speech immediately halts (zero echo, zero double-play).

**Spoken Narration:**
> "Nyaya is built voice-first. Over half of India's population prefers speaking in their native dialect over typing. 
> 
> Nyaya integrates streaming speech recognition powered by Gnani Prisma across 10 Indian languages — including Hindi, Telugu, Tamil, Kannada, Marathi, Bengali, and code-mixed Hinglish.
>
> Notice what happened:
> 1. There was no manual language selector required — Prisma's `indic-auto` real-time streaming detects the script and dialect dynamically.
> 2. When Nyaya began responding with streaming Timbre audio, I interrupted it mid-sentence. Nyaya features local RMS energy barge-in backed by SQLite state. It immediately silenced playback without lag, truncated the conversation record to what was actually spoken, and seamlessly accepted my follow-up."

---

### Scene 3: Evidence-Gated Architecture & Fact Ledger (2:00 – 3:15)
**Visual Setup:**  
* Expand the **"Trace / Evidence"** panel or inspect the message card.  
* Highlight the verbatim statutory source link and the structured **Fact Ledger**.

**Action:**
1. Point the cursor to the citation badge and the statutory quote block.
2. Show the Fact Ledger section (showing captured dispute facts: amount, landlord, 6-month delay).

**Spoken Narration:**
> "Now let’s look under the hood at why Nyaya is safe for high-stakes legal information.
>
> Look at this response. Notice that every single claim cites an exact statutory source. In our engine, we run strict substring verification: if a model proposes a claim that does not contain a verbatim, character-for-character quote from an approved legal gazette or judgment, the claim is stripped before display.
>
> We also run monotone safety triage:
> - PII like Aadhaar numbers, PAN cards, and bank account numbers are redacted at the boundary before any prompt reaches the LLM.
> - Facts are never guessed: they enter the session ledger only when grounded by a verbatim span from the user's own words.
> - And if an external LLM provider goes down, Nyaya doesn’t crash — it gracefully falls back to deterministic statutory and procedural guidance."

---

### Scene 4: Reviewed Procedural Workflows & Safety Net (3:15 – 4:00)
**Visual Setup:**  
* Click on one of the reviewed procedural guides from the sidebar (e.g., **"Police Refused FIR"** or **"Cheque Bounce Notice"**).  
* Show the structured step-by-step intake without hallucinations.

**Action:**
1. Click **"Police Refused FIR"** in the sidebar.
2. Show the official statutory entry points: Section 154(3) CrPC / Section 175(3) BNSS (Superintendent of Police complaint), followed by Section 156(3) Magistrate application.

**Spoken Narration:**
> "Not every citizen issue requires generating free-form text. For standard processes — like filing an FIR when a police station turns you away, seeking free legal aid through NALSA, or handling a Section 138 cheque bounce — Nyaya uses human-reviewed procedural state machines.
>
> The steps, statutory citations, and required documents are loaded from reviewed knowledge bases. The model cannot hallucinate or reorder steps, ensuring citizens get legally sound, court-recognized pathways every single time."

---

### Scene 5: Court-Grade Legal Document Drafting & PDF Engine (4:00 – 5:00)
**Visual Setup:**  
* Type or ask: *"Please draft a formal legal notice for recovery of money due under promissory note."*  
* Show the interactive legal draft editor in the UI.  
* Click **"Download PDF"**.  
* Open the downloaded PDF in full-screen.

**Action:**
1. Send: *"Draft a legal notice for recovery of debt of Rs. 1,00,000 under promissory note."*
2. The draft appears in the styled legal draft viewer with editable text areas.
3. Show the disclaimer: *"This draft is a general template and must be reviewed by a qualified legal professional."*
4. Click **Download PDF**.
5. Switch to the PDF viewer and zoom into:
   - Centered formal court title: `"LEGAL NOTICE FOR RECOVERY OF MONEY DUE UNDER PROMISSORY NOTE"`
   - Formal 0.75" legal margins.
   - Dynamic leading with clean paragraph spacing (no overlapping lines).
   - Running header rule (`LEGAL NOTICE`) and footer rule (`Page 1 of 2` with statutory disclaimer).
   - Formatted signature and verification block.

**Spoken Narration:**
> "Now, one of the most powerful features of Nyaya: automated, court-grade document generation.
>
> Citizens often need formal legal notices or representations, but existing tools generate poorly formatted raw text files.
>
> In Nyaya, we built an Indian court-compliant PDF engine using ReportLab.
> Look at the exported PDF:
> - It formats to standard A4 dimensions with formal 54-point legal court margins.
> - It features automated two-pass running headers and footers with dynamic page counters (`Page X of Y`).
> - The typesetting engine dynamically calculates line height and leading to completely eliminate text collisions.
> - Numbered legal clauses, schedules, and verification clauses are automatically structured.
> - And every draft carries mandatory statutory disclaimers so users know to review it with qualified legal counsel before filing."

---

### Scene 6: Engineering Rigor, Test Suites & Architecture (5:00 – 5:45)
**Visual Setup:**  
* Switch to the terminal window.  
* Run the test suite: `.venv/bin/python -m pytest -q`  
* Run the behavioral evaluation: `.venv/bin/python evals/run_evals.py`  
* Show GitHub repository structure or `README.md`.

**Action:**
1. In terminal, run:
   ```bash
   .venv/bin/python -m pytest -q
   ```
   *(Show 359 tests passing in seconds)*
2. In terminal, run:
   ```bash
   .venv/bin/python evals/run_evals.py
   ```
   *(Show 18/18 behavioral legal evals passing)*

**Spoken Narration:**
> "Finally, let's look at the engineering discipline behind Nyaya. 
> 
> This is not a hackathon prototype wrapped around an API call:
> - We maintain a test suite of **359 automated unit and API tests**, covering everything from routing, PII redaction, session compaction, and audio worklets to the PDF generator.
> - We run **18 behavioral evaluation benchmarks** on every build — testing active safety threats in English and Devanagari, prompt injections, and fact-capture boundaries.
> - Our hybrid search engine combines SQLite FTS5 BM25 with local int8 dense vector embeddings running on CPU with ONNX — searching over **10,588 judgments and 94,843 passages** without external vector database dependencies.
> - The entire frontend is an installable PWA that works offline on both iOS and Android."

---

### Scene 7: Wrap-up & Call to Action (5:45 – 6:00)
**Visual Setup:**  
* Return to the browser showing the GitHub repository or the clean Nyaya interface.  
* Presenter webcam full or smiling.

**Spoken Narration:**
> "Nyaya bridges the justice gap by putting safe, evidence-grounded, multilingual legal intelligence into the hands of every Indian citizen.
>
> The entire project is open-source, thoroughly documented, and ready for deployment. Check out our GitHub repository, review the architecture documentation, and run it locally with a single script.
>
> Thank you for watching!"

---

## Pro-Tips for Delivering the Loom Demo

1. **Energy & Pacing**: Speak at a steady, engaging pace (~130–140 words per minute). Don't rush through the PDF demo — let the viewer see the typography and formatting.
2. **Audio Setup**: If recording live microphone input, ensure your speaker audio doesn't loop back into the microphone. Using headphones is strongly recommended.
3. **If You Prefer Not to Speak Hindi/Telugu Live**: You can click the language chip or paste a pre-prepared regional sentence into the input bar — the STT auto-detection and speech synthesis will still demonstrate the multilingual capability flawlessly.
4. **Resolution**: Record at 1080p (1920x1080) for sharp text rendering of legal clauses and terminal outputs.
