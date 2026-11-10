const $ = (selector) => document.querySelector(selector);
const messages = $('#messages');
const input = $('#messageInput');
const languageSelect = $('#languageSelect');
const jurisdictionSelect = $('#jurisdictionSelect');
const sendButton = $('#sendButton');
const stopButton = $('#stopButton');
const micButton = $('#micButton');
const consentModal = $('#consentModal');
const toast = $('#toast');
const sheetBackdrop = $('#sheetBackdrop');
const sheetBody = $('#sheetBody');
const sheetTitle = $('#sheetTitle');
const appShell = $('.app-shell');

const state = {
  sessionId: null, sessionToken: null, config: null, capabilities: [], language: 'auto', consent: false,
  lastUserText: '', pendingService: null, busy: false, autoSpeak: false,
  recording: false, micStream: null, audioContext: null, workletNode: null,
  sttSocket: null, transcript: '', reviewSummary: '', reviewLanguage: 'en-IN',
  abort: null, speaking: null, ttsNodes: [],
  ttsSpeech: null, speechStartedAt: 0, speakNotBefore: 0,
  vadOnsetAt: 0, endpointTimer: null,
  // Voice output is a queue of sentences, not one blob: the server may send a
  // safe opening sentence long before the turn has finished validating, and
  // those sentences have to play in order over one AudioContext.
  voiceQueue: [], voiceSpeaking: false, voiceNextTime: 0, spokenSoFar: '',
  // Bumped on every barge-in. Sentences already in flight for the previous
  // generation are dropped, so an interruption is not undone by the tail of
  // the answer arriving a moment later.
  speechGeneration: 0, turnSpeechGeneration: 0,
  // Language follows the caller: the override is only honoured when the caller
  // actually chose one. `detectedLanguage` is what Prisma heard.
  detectedLanguage: null, sttRestarts: 0,
};

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function showToast(text) {
  toast.textContent = text;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 3800);
}
function setListening(active) {
  // The orb is the only state indicator most callers will look at, so the
  // microphone state is expressed on the body and read by CSS; the hint beside
  // it carries the same fact in words for anyone who cannot see the colour.
  document.body.classList.toggle('listening', !!active);
  micButton.setAttribute('aria-pressed', active ? 'true' : 'false');
  const hint = $('#voiceHint');
  // The hint earns its line only while it is telling the caller something
  // live: that the microphone is open, or which language was recognised. At
  // rest it is empty and CSS removes the row, so the composer stays small.
  if (hint) hint.textContent = active ? 'Listening — pause to send.' : '';
}
function setSpeaking(active) {
  document.body.classList.toggle('speaking', !!active);
}
function setMicLevel(level) {
  document.documentElement.style.setProperty('--mic-level', Math.max(0, Math.min(1, level)).toFixed(2));
}
function capabilityLabel(id) { return state.capabilities.find(item => item.id === id)?.label || id.replaceAll('_', ' '); }
// Fact values arrive from the ledger and can be enum tokens (`police_refused`,
// `cheque_returned`). The caller never typed those, so they are never shown.
function humanise(value) { return String(value ?? '').replaceAll('_', ' ').replace(/\s+/g, ' ').trim(); }
function setWelcomeHidden() { $('#welcome').classList.add('hidden'); }
// The three-step "progress" card is gone. It advanced on a guess — one step
// for any answer, a third for anything with a source — so it told the caller
// they were making progress the app had not measured.
function scrollMessages() { messages.scrollTop = messages.scrollHeight; }

// A fold that opens near the bottom of the transcript grows downward, outside
// the scroll port, so the revealed steps look cut off. `toggle` does not
// bubble, hence the capture listener. Reduced-motion users get the jump
// without the glide.
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
messages.addEventListener('toggle', event => {
  const details = event.target;
  if (!(details instanceof HTMLDetailsElement) || !details.open) return;
  requestAnimationFrame(() => {
    const port = messages.getBoundingClientRect();
    const box = details.getBoundingClientRect();
    const behavior = reduceMotion.matches ? 'auto' : 'smooth';
    // If the opened panel is taller than the port, align its top so the first
    // step is readable. Otherwise reveal its bottom, which is what looked cut
    // off before.
    if (box.height > port.height - 48) {
      messages.scrollBy({ top: box.top - port.top - 8, behavior });
    } else if (box.bottom > port.bottom - 8) {
      messages.scrollBy({ top: box.bottom - port.bottom + 24, behavior });
    }
  });
}, true);

function addUserMessage(text) {
  setWelcomeHidden();
  const article = document.createElement('article');
  article.className = 'message user';
  article.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  messages.append(article); scrollMessages();
}
function addTyping(label = 'Listening…') {
  removeTyping();
  const article = document.createElement('article');
  article.className = 'message assistant'; article.id = 'typingMessage';
  // The work is real and pending, so the indicator is a thin indeterminate
  // line under the stage label rather than three dots simulating liveness.
  article.innerHTML = `<div class="answer-head"><span class="status-pill" id="typingLabel">${esc(label)}</span></div><div class="pending-line" aria-hidden="true"></div>`;
  messages.append(article); scrollMessages();
}
function removeTyping() { $('#typingMessage')?.remove(); }
function setTypingLabel(text) {
  const label = $('#typingLabel');
  if (label && text) label.textContent = String(text).slice(0, 90);
}

/* Consent is remembered per notice version, not per page load. Re-asking on
 * every visit trained people to dismiss the one screen that explains what the
 * service is; a new notice version (the copy changed) asks again. Only the
 * version and a timestamp are stored — no identifiers. */
const CONSENT_KEY = 'nyaya.consent';
function storedConsent(version) {
  if (!version) return false;
  try {
    const saved = JSON.parse(localStorage.getItem(CONSENT_KEY) || 'null');
    return !!saved && saved.version === version;
  } catch { return false; }
}
function rememberConsent(version) {
  try { localStorage.setItem(CONSENT_KEY, JSON.stringify({ version, at: Date.now() })); } catch {}
}

async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    state.config = await response.json();
    if (storedConsent(state.config.consent_notice_version)) {
      state.consent = true;
      hideDialog(consentModal);
    }
    const supported = new Set(state.config.conversation_languages || []);
    [...languageSelect.options].forEach(option => {
      if (option.value !== 'auto') option.disabled = !supported.has(option.value);
    });
    const codes = state.config.jurisdictions || [];
    jurisdictionSelect.innerHTML = '<option value="auto">State · not set</option>' +
      codes.filter(item => item.code !== 'IN').map(item => `<option value="${esc(item.code)}">${esc(item.label)}</option>`).join('');
    // The rail states what the caller can rely on, in the caller's words. An
    // environment variable name is a setup instruction for the operator, so it
    // lives in "How this works" where it can explain itself.
    const speechReady = !!state.config.speech?.gnani_configured;
    const provider = $('#providerState');
    provider.classList.toggle('is-off', !speechReady);
    provider.lastElementChild.textContent = speechReady
      ? 'Voice ready — speak any language'
      : 'Voice not set up · typing works';
    const modelOn = !!state.config.reasoning?.model_configured;
    $('#modeChip').classList.toggle('is-limited', !modelOn);
    $('#modeText').textContent = modelOn ? 'Full assistant' : 'Sources only';
  } catch {
    const provider = $('#providerState');
    provider.classList.add('is-off');
    provider.lastElementChild.textContent = 'Service unreachable';
    $('#modeChip').classList.add('is-limited');
    $('#modeText').textContent = 'Service unreachable';
  }
}

async function loadCapabilities() {
  try {
    const data = await (await fetch('/api/workflows')).json();
    state.capabilities = data.capabilities || [];
  } catch { state.capabilities = []; }
}

// Guides, approved sources, saved work and the explanation all live in one
// sheet. Pushed into the transcript they were mistakable for answers, and they
// took the width the answer needed.
/* ---------- modal focus ------------------------------------------------- *
 * The sheet and both dialogs are modal, so the page behind them is inert, Tab
 * stays inside, and focus returns to whatever opened it. Without this a
 * keyboard user tabs into the conversation behind an open sheet and cannot
 * tell where they are.
 */
let opener = null;

const focusable = root => [...root.querySelectorAll(
  'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),summary,[tabindex]:not([tabindex="-1"])',
)].filter(el => el.getClientRects().length > 0);

function modalRoot() {
  for (const element of [$('#reviewModal'), consentModal, sheetBackdrop]) {
    if (!element || element.classList.contains('hidden')) continue;
    return element === sheetBackdrop ? element.querySelector('.sheet') : element;
  }
  return null;
}

function showDialog(element, preferred) {
  if (!opener) opener = document.activeElement;
  element.classList.remove('hidden');
  if ('inert' in HTMLElement.prototype && appShell) appShell.inert = true;
  const root = element === sheetBackdrop ? element.querySelector('.sheet') : element;
  (preferred || root.querySelector('.modal') || root).focus();
}

function hideDialog(element) {
  if (element.classList.contains('hidden')) return;
  element.classList.add('hidden');
  if ('inert' in HTMLElement.prototype && appShell) appShell.inert = false;
  if (modalRoot()) return;
  if (opener && document.contains(opener)) opener.focus();
  opener = null;
}

// Tab wraps inside whichever layer is open. Escape is handled separately.
document.addEventListener('keydown', event => {
  if (event.key !== 'Tab') return;
  const root = modalRoot();
  if (!root) return;
  const items = focusable(root);
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  if (!root.contains(document.activeElement)) { event.preventDefault(); first.focus(); return; }
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
});

function openSheet(title, body) {
  sheetTitle.textContent = title;
  sheetBody.innerHTML = body;
  sheetBody.scrollTop = 0;
  // Focus the dialog itself, not its close button: the panel is announced by
  // its own label, and a ring around the X the instant it opens reads as an
  // error. Tab still enters at the first control inside.
  showDialog(sheetBackdrop);
}
function closeSheet() {
  if (sheetBackdrop.classList.contains('hidden')) return;
  hideDialog(sheetBackdrop);
  setNav('talk');
}

async function openGuides() {
  openSheet('Guides and sources', '<p class="sheet-note">Loading the approved material…</p>');
  const [procedures, sources] = await Promise.all([
    fetch('/api/procedures').then(response => response.ok ? response.json() : { procedures: [] }).catch(() => ({ procedures: [] })),
    fetch('/api/sources').then(response => response.ok ? response.json() : []).catch(() => []),
  ]);
  const guides = (procedures.procedures || []).map(item => `<button class="guide-item" data-procedure="${esc(item.id)}" data-label="${esc(item.label)}">
      <b>${esc(item.label)}</b><small>${esc(item.summary || '')}</small></button>`).join('');
  const records = (Array.isArray(sources) ? sources : []).map(source => `<a class="source-card" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">
      <span class="source-icon"><svg class="icon" aria-hidden="true"><use href="#i-doc"/></svg></span><span><b>${esc(source.title)}</b><small>${esc(source.authority)}${source.review_status ? ` · ${esc(source.review_status)}` : ''}</small></span></a>`).join('');
  openSheet('Guides and sources', `
    <section class="sheet-section"><h3>Step-by-step guides</h3>
      ${guides ? `<div class="sheet-list">${guides}</div>` : '<p class="sheet-empty">No guides are loaded.</p>'}
      <p class="sheet-note">A guide marked draft still has to be confirmed at the official source.</p>
    </section>
    <section class="sheet-section"><h3>Approved sources</h3>
      ${records ? `<div class="sheet-list">${records}</div>` : '<p class="sheet-empty">No source records are approved yet.</p>'}
      <p class="sheet-note">${Array.isArray(sources) ? sources.length : 0} primary records are approved for retrieval. This is not complete coverage of Indian law.</p>
    </section>`);
  sheetBody.querySelectorAll('[data-procedure]').forEach(button => {
    button.addEventListener('click', () => {
      closeSheet();
      sendTurn(`How do I ${button.dataset.label.toLowerCase()}?`, null, false);
    });
  });
}

async function openSaved() {
  openSheet('Saved', '<p class="sheet-note">Loading…</p>');
  if (!state.sessionId) { openSheet('Saved', '<p class="sheet-empty">Nothing is saved yet. A draft you save stays in this conversation.</p>'); return; }
  try {
    const response = await fetch(`/api/session/${encodeURIComponent(state.sessionId)}/artifacts`, { headers: authHeaders() });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not load saved work.');
    const items = (result.artifacts || []).map((item, idx) => `<div class="claim-note"><b>${esc(item.payload?.title || item.action || item.type || 'Saved item')}</b>
      <p>${esc(String(item.payload?.content || JSON.stringify(item.payload || item)).slice(0, 300))}</p>
      <div style="margin-top: 8px;"><button class="action-chip download-saved-pdf" data-index="${idx}" type="button"><svg class="icon" aria-hidden="true"><use href="#i-download"/></svg> Download Court PDF</button></div></div>`).join('');
    openSheet('Saved', items
      ? `<div class="sheet-list spaced">${items}</div><p class="sheet-note">Held in this conversation only, and deleted when it expires or you start a new one.</p>`
      : '<p class="sheet-empty">Nothing is saved yet.</p>');
    document.querySelectorAll('.download-saved-pdf').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = btn.dataset.index;
        showToast('Downloading court PDF...');
        try {
          const resp = await fetch(`/api/session/${encodeURIComponent(state.sessionId)}/artifact/${idx}/pdf`, {
            headers: authHeaders(),
          });
          if (!resp.ok) throw new Error('Could not download PDF.');
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `saved_legal_draft_${idx}.pdf`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          showToast('Court PDF downloaded.');
        } catch (e) {
          showToast(e.message || 'Error downloading PDF');
        }
      });
    });
  } catch (error) {
    openSheet('Saved', `<p class="sheet-empty">${esc(error.message)}</p>`);
  }
}

function openHelp() {
  const speech = state.config?.speech || {};
  const voiceLine = speech.gnani_configured
    ? 'Voice is on. The microphone is open only while you are speaking, and a tap stops it.'
    : 'Voice is not available right now. Everything else works: type your situation and the answer still carries sources.';
  openSheet('How this works', `
    <section class="sheet-section"><h3>A turn, in order</h3>
      <p class="sheet-note">Your message is routed before any writing happens: a safety script, a reviewed procedure, or a search of the approved legal sources. Only then is an answer phrased.</p>
      <p class="sheet-note">A legal proposition is shown with a verbatim quote from an approved, in-date, jurisdiction-matching source. When nothing supports it, the app says so instead of guessing.</p>
    </section>
    <section class="sheet-section"><h3>What is kept</h3>
      <p class="sheet-note">The conversation is kept for four hours, then deleted. Identity and account numbers are masked before the message reaches the model, and in the logs.</p>
      <p class="sheet-note">Nothing is filed, sent to a court, or shown to a person without a second confirmation from you.</p>
    </section>
    <section class="sheet-section"><h3>Voice</h3>
      <p class="sheet-note">${esc(voiceLine)}</p>
      <p class="sheet-note">Language is detected from what you actually say, so a mixed sentence is normal. Choosing a language in the composer only overrides that detection.</p>
    </section>
    <section class="sheet-section"><h3>Limits</h3>
      <p class="sheet-note">This is legal information, not legal advice, and not a substitute for a lawyer. Source coverage is small, so "I could not verify this" is a real answer.</p>
    </section>`);
}

async function createSession() {
  if (state.sessionId) return state.sessionId;
  const response = await fetch('/api/session', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({language:state.language, consent_confirmed:state.consent, consent_version:state.config?.consent_notice_version}),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || 'Could not start a conversation.');
  state.sessionId = result.session_id;
  // The id is public to this page; the token is the credential. It stays in
  // memory and is never written to storage, a URL or the service worker cache.
  state.sessionToken = result.session_token;
  return state.sessionId;
}

function authHeaders() {
  const headers = {'Content-Type':'application/json'};
  if (state.sessionToken) headers.Authorization = `Bearer ${state.sessionToken}`;
  return headers;
}

function sessionSocketQuery() {
  if (!state.sessionId) return '';
  const id = encodeURIComponent(state.sessionId);
  return state.sessionToken ? `&session_id=${id}&token=${encodeURIComponent(state.sessionToken)}` : `&session_id=${id}`;
}

function jurisdictionValue() {
  return jurisdictionSelect?.value || 'auto';
}

function parseSSE(chunk) {
  const lines = chunk.split('\n');
  let event = 'message';
  const data = [];
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  if (!data.length) return null;
  try { return { event, data: JSON.parse(data.join('\n')) }; } catch { return null; }
}

async function sendTurn(rawText, service = null, speakReply = false) {
  const text = rawText.trim();
  if (!text || state.busy) return;
  if (!state.consent) { showDialog(consentModal); return; }
  if (state.ttsSpeech) interruptSpeech('superseded');
  state.busy = true; state.autoSpeak = speakReply; state.lastUserText = text;
  // A new turn starts a new spoken reply: what was said belongs to the last one.
  if (speakReply) {
    state.spokenSoFar = ''; state.speechPromise = null; state.ttsSpeech = null;
    state.turnSpeechGeneration = state.speechGeneration;
  }
  state.abort = new AbortController();
  sendButton.disabled = true; stopButton.classList.remove('hidden');
  input.value = ''; autosize(); addUserMessage(text); addTyping();
  let final = null;
  try {
    const sessionId = await createSession();
    const response = await fetch('/api/turn/stream', {
      method:'POST', headers: authHeaders(),
      signal: state.abort.signal,
      body:JSON.stringify({
        session_id: sessionId, text, language: state.language,
        jurisdiction: jurisdictionValue(), service, consent_confirmed: true,
      }),
    });
    if (!response.ok || !response.body) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || 'The assistant could not complete this turn.');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf('\n\n')) >= 0) {
        const chunk = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        const parsed = parseSSE(chunk);
        if (!parsed) continue;
        if (parsed.event === 'ack') {
          setTypingLabel(parsed.data.text);
          // Said immediately: the acknowledgement exists to replace dead air
          // with the caller's own language while the reader and retrieval run.
          if (speakReply) speakSentence(parsed.data.text, parsed.data.language || state.language, false);
        }
        if (parsed.event === 'speak' && speakReply) {
          speakSentence(parsed.data.text, parsed.data.language || state.language);
        }
        if (parsed.event === 'stage') {
          if (parsed.data.stage === 'routed') setTypingLabel('Working out what this is');
          if (parsed.data.stage === 'evidence') setTypingLabel(`Checking ${parsed.data.sources || 0} sources`);
          if (parsed.data.stage === 'procedure') setTypingLabel('Found a reviewed procedure');
        }
        if (parsed.event === 'final') final = parsed.data;
        if (parsed.event === 'error') throw new Error(parsed.data.message || 'Turn failed.');
      }
    }
    if (!final) throw new Error('The assistant stopped before answering.');
    removeTyping();
    addAssistant(final);
    // The document language drives pronunciation and the screen-reader voice,
    // so it follows the answer instead of staying en-IN for a Tamil caller.
    const answerLanguage = final.tts_language || final.language;
    if (answerLanguage) document.documentElement.lang = answerLanguage;
    if (speakReply && state.speechGeneration === state.turnSpeechGeneration) {
      // Sentences already streamed were spoken; only the tail is left. When the
      // provider could not stream, nothing was said yet and this says it all.
      // A barge-in mid-answer changes the generation, and then nothing more is
      // spoken: the caller has already taken the floor.
      const rest = remainingSpeech(final.spoken || final.response || '', state.spokenSoFar);
      if (rest) speakAnswer(rest, final.tts_language || final.language);
    }
  } catch (error) {
    removeTyping();
    if (error.name === 'AbortError') {
      addAssistant(note('Stopped. Nothing further was sent to the model.', 'You stopped this answer.'));
    } else {
      addAssistant(note(error.message, 'Your message stays in this session. No external person or service was contacted.'));
    }
  } finally {
    state.busy = false; state.abort = null;
    sendButton.disabled = false; stopButton.classList.add('hidden'); input.focus();
  }
}

function note(response, disclaimer) {
  return {
    response, spoken: response, language: state.language, sources: [], claims: [], case_context: [],
    services_used: [], risk_flags: [], verification: { status: 'not_run' }, answer_status: 'not_run',
    human_connection: { offered: false }, disclaimer, mode: 'client',
  };
}

// Two strings that carry the same words are one thing to a reader. The engine
// phrases its own question at the end of its reply and also reports the same
// question as `next_question`; printing both made every intake turn look like
// it was repeating itself. Script-aware, so Devanagari, Tamil and Bengali are
// compared as words too.
function words(value) {
  return String(value || '').toLowerCase().replace(/[^\p{L}\p{N} ]+/gu, ' ').replace(/\s+/g, ' ').trim();
}
function pullQuestion(response, question) {
  const text = String(response || '').trim();
  const asked = words(question);
  if (!text || !asked) return { body: text, question: '' };
  const sentences = text.match(/[^.!?।\n]+[.!?।]*/gu) || [];
  let taken = 0;
  let tail = '';
  for (let i = sentences.length - 1; i >= 0; i -= 1) {
    const candidate = `${sentences[i]}${tail}`.trim();
    const candidateWords = words(candidate);
    if (!candidateWords || !asked.includes(candidateWords)) break;
    tail = ` ${sentences[i]}`;
    taken += 1;
  }
  if (!taken) return { body: text, question: '' };
  const body = sentences.slice(0, sentences.length - taken).join('').trim();
  // An answer that is nothing but the question keeps its text in the bubble.
  if (!body) return { body: text, question: '' };
  return { body, question: tail.trim() };
}

function renderProcedure(procedure) {
  if (!procedure) return '';
  const steps = (procedure.steps || []).map(step => `<li>
      <b>${esc(step.action)}</b>
      ${step.detail ? `<p>${esc(step.detail)}</p>` : ''}
      ${step.where ? `<small><span class="step-where">Where:</span> ${esc(step.where)}</small>` : ''}
      ${(step.bring || []).length ? `<small><span class="step-where">Bring:</span> ${esc(step.bring.join(' · '))}</small>` : ''}
    </li>`).join('');
  const review = procedure.reviewed
    ? '<span class="review-ok">reviewed</span>'
    : '';
  const verify = (procedure.verify_at_source || []).length
    ? `<div class="procedure-verify"><b>Confirm before relying on it:</b> ${esc(procedure.verify_at_source.join(' · '))}</div>` : '';
  const handoff = procedure.handoff
    ? `<a class="procedure-link" href="${esc(procedure.handoff.url)}" target="_blank" rel="noopener noreferrer">${esc(procedure.handoff.label)} →</a>` : '';
  return `<div class="procedure-card">
      <div class="procedure-head"><b>${esc(procedure.label)}</b>${review}</div>
      <ol class="procedure-steps">${steps}</ol>
      ${procedure.if_refused ? `<div class="procedure-note"><b>If it is refused:</b> ${esc(procedure.if_refused)}</div>` : ''}
      ${verify}${handoff}
      <small class="procedure-source">Source: <a href="${esc(procedure.source.url)}" target="_blank" rel="noopener noreferrer">${esc(procedure.source.authority)}</a></small>
    </div>`;
}

function renderTrace(trace, disclaimer) {
  const rows = (Array.isArray(trace) ? trace : []).map(item => `<div class="trace-row"><span>${esc(item.stage)}</span><span>${Number(item.ms) || 0} ms</span><span>${esc(Object.entries(item).filter(([k]) => !['stage','ms'].includes(k)).map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ')).slice(0, 160)}</span></div>`).join('');
  if (!rows && !disclaimer) return '';
  // The full disclaimer and the timings stay available, but folded away: said
  // once on every answer it stopped being read, and the sentence that matters
  // ("this is information, not advice") is now always under the composer.
  return `<details class="trace"><summary><svg class="icon disclosure" aria-hidden="true"><use href="#i-chevron"/></svg>Sources and reasoning</summary>${rows}${disclaimer ? `<p class="trace-note">${esc(disclaimer)}</p>` : ''}</details>`;
}

function addAssistant(result) {
  removeTyping(); setWelcomeHidden();
  const article = document.createElement('article'); article.className = 'message assistant';
  const sourceCards = (result.sources || []).map(source => `<a class="source-card" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer"><span class="source-icon" aria-hidden="true"><svg class="icon"><use href="#i-doc"/></svg></span><span><b>${esc(source.title)}</b><small>${esc(source.authority)}${source.citation ? ` · ${esc(source.citation)}` : ''}${source.decision_date ? ` · decided ${esc(source.decision_date)}` : ''}${source.review_status ? ` · ${esc(source.review_status)}` : ''}${source.treatment_status ? ' · treatment not checked' : ''}</small></span></a>`).join('');
  const claims = (result.claims || []).map(claim => `<div class="claim-note"><b>${esc(claim.text)}</b><details><summary><svg class="icon disclosure" aria-hidden="true"><use href="#i-chevron"/></svg>Source passage · ${esc(claim.authority || '')}</summary><p>${esc(claim.quote)}</p></details></div>`).join('');
  const caseNotes = (result.case_context || []).map(item => `<div class="claim-note"><b>Why it may be similar</b><p>${esc(item.why_similar)}</p><small>${esc(item.limits)}</small></div>`).join('');
  // Judgments retrieval actually returned. They used to sit in the payload and
  // render nowhere, so a passages-only answer said "no source matched" while
  // four decisions sat unread in the response. They are leads, not authority:
  // the tier is always shown, `treatment_status` is always unchecked, and the
  // quoted passage is the one that matched, verbatim.
  const precedentNotes = (result.case_precedents || []).slice(0, 3).map(item => `<div class="claim-note precedent">
      <b>${esc(item.title || 'Judgment')}</b>
      <small>${esc(item.court || 'Court not recorded')}${item.decision_date ? ` · decided ${esc(item.decision_date)}` : ''}${item.citation ? ` · ${esc(item.citation)}` : ''} · ${esc(item.court_level || 'unknown')}-court authority · treatment not checked</small>
      ${item.excerpt ? `<details><summary><svg class="icon disclosure" aria-hidden="true"><use href="#i-chevron"/></svg>Matched passage</summary><p>${esc(item.excerpt)}</p></details>` : ''}
      ${item.url ? `<a class="precedent-link" href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">Open the judgment</a>` : ''}
    </div>`).join('');
  const safety = result.safety || {};
  const safetyNotice = safety.notice
    ? `<div class="danger-callout safety-${esc(safety.level)}"><b>${safety.level === 'critical' ? 'SAFETY FIRST' : 'URGENT'}</b> · ${esc(safety.notice)}</div>` : '';
  const riskList = (result.risk_flags || []).filter(flag => !String(flag).startsWith('triage:'));
  const risk = riskList.length ? `<div class="danger-callout">${esc(riskList.join(' · '))}</div>` : '';
  const slotLabels = {jurisdiction:'state', when:'when', amount:'amount', notice_or_order:'notice', outcome:'outcome', stage:'what was tried', recipient:'recipient', situation:'issue'};
  const knownFacts = result.conversation_state?.known_facts || [];
  const workflow = result.workflow;
  const asking = !!(workflow && workflow.ask && workflow.status !== 'completed');
  // No step counter. "Question 1 of 4" told the caller they were in a form,
  // which is the opposite of a conversation, and the intake label repeated the
  // source card title directly below it. The question itself is the signal.
  const workflowStep = '';
  const factsBlock = knownFacts.length
    ? `<div class="fact-line"><span class="fact-label">Known</span>${knownFacts.map(fact => `<span class="fact-chip">${esc(slotLabels[fact.slot] || fact.slot)}: ${esc(humanise(fact.value))}</span>`).join('')}</div>` : '';
  const conflict = result.conversation_state?.conflict_note
    ? `<div class="fact-conflict">${esc(result.conversation_state.conflict_note)}</div>` : '';
  const identifiers = result.identifiers_in_message || [];
  const piiNote = identifiers.length
    ? `<div class="pii-note">Your message contains ${esc(identifiers.map(item => item.kind).join(', '))}. These are masked in server logs, and masked again in any summary you send for review unless you tick the option to include them.</div>` : '';
  const services = (result.services_used || []).map(id => `<button class="action-chip" data-service="${esc(id)}">${esc(capabilityLabel(id))}</button>`).join('');
  // When the relevance check rejected every candidate, the two closest are
  // still shown (a caller is better served by a lead than by a blank panel)
  // but the label says what they are: retrieved, not confirmed to answer.
  const relevanceNote = result.corpus?.case_relevance === 'none_on_point'
    ? '<div class="fact-conflict">Closest decisions retrieved — none is confirmed to address your question directly.</div>' : '';
  const citationBlock = claims || caseNotes || precedentNotes
    ? `<div class="claim-evidence">${claims}${caseNotes}${relevanceNote}${precedentNotes}</div>` : '';
  const nextQuestion = result.next_question || result.follow_up || '';
  const said = pullQuestion(result.response, nextQuestion);
  // Mid-intake the engine prefixes the review caveat to its question, and
  // pulling the question out can leave that caveat alone as the body. It is
  // already stated by the status pill and the source card, so repeating it as
  // the answer buries the question the caller has to reply to.
  const body = asking && said.body.length < 220 && /signed off|official source/i.test(said.body)
    ? '' : said.body;
  const followUp = said.question ? `<div class="follow-up-question">${esc(said.question)}</div>` : '';
  // While an intake is asking, the question is the answer and the whole
  // procedure is reference material one tap away. Printing five steps above a
  // one-line question buried the thing the caller has to reply to.
  const procedureBlock = result.procedure
    ? (asking
      ? `<details class="fold"><summary><svg class="icon disclosure" aria-hidden="true"><use href="#i-chevron"/></svg>See all ${(result.procedure.steps || []).length} steps for this</summary>${renderProcedure(result.procedure)}</details>`
      : renderProcedure(result.procedure))
    : '';
  const warning = result.artifact?.warning ? `<div class="draft-warning">${esc(result.artifact.warning)}</div>` : '';
  const reviewButton = result.human_connection?.offered
    ? '<button class="action-chip request-review" type="button">Talk to a qualified person</button>' : '';
  const statusMap = {
    source_linked: ['Source-linked answer', 'ok'], procedure_grounded: ['Reviewed procedure', 'ok'],
    emergency_safety: ['Safety first', 'danger'],    handoff_ready: ['Ready to connect a person', 'ok'],
    no_supported_legal_claim: ['Could not verify', 'warn'], not_run: ['Not completed', 'plain'],
    intake_turn: ['Question', 'plain'],
  };
  let [status, tone] = statusMap[result.answer_status] || ['Answer', 'plain'];
  // The pill is a legal-trust label, so it follows the attached procedure's own
  // review state, not the route that produced the turn. A procedure-grounded
  // turn with no approved procedure behind it must never read "Reviewed".
  if (result.procedure) {
    const approved = result.procedure.reviewed === true || result.procedure.review_status === 'approved';
    status = approved ? 'Reviewed procedure' : 'Draft procedure';
    tone = approved ? 'ok' : 'warn';
  } else if (result.answer_status === 'procedure_grounded') {
    status = 'Procedure';
    tone = 'plain';
  }
  const modeBadge = result.mode === 'grounded_only' ? '<span class="mode-badge">quoted sources</span>' : '';
  article.innerHTML = `<div class="answer-head"><span class="status-pill tone-${tone}">${esc(status)}</span>${workflowStep}${modeBadge}</div>
    ${safetyNotice}
    ${body ? `<div class="bubble">${esc(body)}</div>` : ''}
    ${risk}${piiNote}${conflict}${followUp}${factsBlock}${procedureBlock}${citationBlock}
    ${sourceCards ? `<div class="source-list">${sourceCards}</div>` : ''}
    ${result.artifact ? renderArtifact(result.artifact, warning, reviewButton) : ''}
    ${services || (!result.artifact && reviewButton) ? `<div class="action-chips">${services}${!result.artifact ? reviewButton : ''}</div>` : ''}
    ${renderTrace(result.trace, result.disclaimer)}
    <div class="message-actions">
      <button class="small-action speak-answer" type="button"><svg class="icon" aria-hidden="true"><use href="#i-volume"/></svg>Speak</button>
      <button class="small-action stop-speak" type="button"><svg class="icon" aria-hidden="true"><use href="#i-stop"/></svg>Stop voice</button>
      <button class="small-action copy-answer" type="button"><svg class="icon" aria-hidden="true"><use href="#i-copy"/></svg>Copy</button>
    </div>`;
  article.dataset.answer = result.spoken || result.response || '';
  article.dataset.language = result.tts_language || result.language || state.language;
  messages.append(article); scrollMessages();
  article.querySelectorAll('[data-service]').forEach(button => button.addEventListener('click', () => runService(button.dataset.service)));
  article.querySelector('.speak-answer')?.addEventListener('click', () => speakAnswer(article.dataset.answer, article.dataset.language));
  article.querySelector('.stop-speak')?.addEventListener('click', stopSpeaking);
  article.querySelector('.copy-answer')?.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(result.response || ''); showToast('Answer copied'); }
    catch { showToast('Clipboard access is unavailable in this browser.'); }
  });
  article.querySelector('.request-review')?.addEventListener('click', () => {
    const summary = result.artifact?.content || (result.reported_facts || []).join('\n') || state.lastUserText;
    openReview(summary, result.language || state.language);
  });
  article.querySelector('.save-artifact')?.addEventListener('click', () => saveArtifact(result.artifact, article.querySelector('.editable-draft')));
  article.querySelector('.download-pdf')?.addEventListener('click', async () => {
    const draftText = article.querySelector('.editable-draft')?.value || result.artifact?.content || '';
    const title = result.artifact?.title || 'legal_draft';
    const type = result.artifact?.type || 'general_legal_draft';
    const warning = result.artifact?.warning || '';
    showToast('Generating court-formatted PDF...');
    try {
      const resp = await fetch('/api/export/pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(state.sessionToken ? { 'Authorization': `Bearer ${state.sessionToken}` } : {})
        },
        body: JSON.stringify({
          title,
          content: draftText,
          document_type: type,
          warning
        })
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to generate PDF.');
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const safeTitle = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 45) || 'court_document';
      a.download = `${safeTitle}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('Court PDF downloaded.');
    } catch (e) {
      showToast(e.message || 'Could not download PDF.');
    }
  });
  article.querySelector('.copy-draft')?.addEventListener('click', async () => {
    const draftText = article.querySelector('.editable-draft')?.value || result.artifact?.content || '';
    try { await navigator.clipboard.writeText(draftText); showToast('Draft copied to clipboard.'); }
    catch { showToast('Clipboard access unavailable.'); }
  });
  article.querySelector('.download-draft')?.addEventListener('click', () => {
    const draftText = article.querySelector('.editable-draft')?.value || result.artifact?.content || '';
    const title = result.artifact?.title || 'legal_draft';
    const blob = new Blob([draftText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 40)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Draft downloaded.');
  });
}

function renderArtifact(artifact, warning, reviewButton) {
  const action = `<div class="artifact-buttons"><button class="action-chip download-pdf primary-action" type="button"><svg class="icon" aria-hidden="true"><use href="#i-download"/></svg> Download Court PDF</button><button class="action-chip copy-draft" type="button">Copy draft</button><button class="action-chip download-draft" type="button">Download text</button><button class="action-chip save-artifact" type="button">Save in this session</button>${reviewButton}</div>`;
  return `<div class="artifact-card"><div class="artifact-title">${esc(artifact.title || 'Prepared work')}</div><textarea class="editable-draft" rows="12" aria-label="Editable generated document">${esc(artifact.content || '')}</textarea>${warning}${action}</div>`;
}

async function runService(service) {
  if (!state.lastUserText) { state.pendingService = service; input.focus(); return; }
  await sendTurn(state.lastUserText, service, false);
}

async function saveArtifact(artifact, draftField) {
  const content = draftField?.value || artifact?.content || '';
  try {
    const response = await fetch('/api/action/confirm', {
      method:'POST',headers: authHeaders(),
      body:JSON.stringify({session_id:await createSession(),action:'save_artifact',payload:{type:artifact?.type,title:artifact?.title,content},user_confirmed:true}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not save.');
    showToast('Saved in this session only.');
  } catch (error) { showToast(error.message); }
}

function openReview(summary, language) {
  state.reviewSummary = String(summary || '').slice(0, 3000);
  state.reviewLanguage = language && language !== 'auto' ? language : state.language;
  const include = $('#includeIdentifiers');
  if (include) include.checked = false;
  $('#reviewSummary').textContent = state.reviewSummary || 'No summary selected.';
  $('#shareReview').checked = false; $('#confirmReview').disabled = true;
  $('#reviewResult').textContent = ''; showDialog($('#reviewModal'));
  refreshReviewPreview();
}

async function refreshReviewPreview() {
  const note = $('#reviewMaskNote');
  const include = !!$('#includeIdentifiers')?.checked;
  try {
    const response = await fetch('/api/review/preview', {
      method:'POST', headers: authHeaders(),
      body:JSON.stringify({session_id:await createSession(),summary:state.reviewSummary,include_identifiers:include}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not build the summary.');
    $('#reviewSummary').textContent = result.body || 'No summary selected.';
    if (note) {
      note.textContent = result.notice
        || (result.identifiers.length && include ? 'Identity numbers are included because you chose to share them.' : '');
    }
  } catch (error) {
    if (note) note.textContent = `Preview unavailable (${error.message}). The text above is what would be sent.`;
  }
}
async function requestHumanReview() {
  if (!$('#shareReview').checked) return;
  $('#confirmReview').disabled = true; $('#reviewResult').textContent = 'Sending your explicit request…';
  try {
    const response = await fetch('/api/review/request', {
      method:'POST',headers: authHeaders(),
      body:JSON.stringify({session_id:await createSession(),language:state.reviewLanguage,user_requested:true,consent_to_share:true,summary:state.reviewSummary,include_identifiers:!!$('#includeIdentifiers')?.checked}),
    });
    const result = await response.json();
    $('#reviewResult').textContent = result.connected
      ? `Connection request accepted in ${result.language}.`
      : `${result.message || 'No person was connected and no information was sent.'}${result.fallback ? ` ${result.fallback}` : ''}`;
  } catch (error) { $('#reviewResult').textContent = `No person was connected: ${error.message}`; }
  finally { $('#confirmReview').disabled = false; }
}

function voicePlan() { return state.config?.voice_control || {}; }

function stopAudio() {
  setSpeaking(false);
  state.speechGeneration += 1;
  state.voiceQueue = [];
  for (const node of state.ttsNodes) { try { node.stop(); } catch {} }
  state.ttsNodes = [];
  for (const socket of state.ttsSockets || []) { try { socket.close(); } catch {} }
  state.ttsSockets = [];
  state.ttsSocket = null;
  // The AudioContext is kept alive for the call. Rebuilding it for every
  // sentence costs more than the sentence took to synthesise, and a stopped
  // playhead only needs to be moved forward, not reset.
  if (state.speaking) state.voiceNextTime = state.speaking.currentTime + 0.06;
  state.ttsSpeech = null; state.speechStartedAt = 0;
}

async function interruptSpeech(reason = 'button', partial = '') {
  const speech = state.ttsSpeech;
  const onset = state.vadOnsetAt || performance.now();
  const heard = state.speechStartedAt ? Math.round(performance.now() - state.speechStartedAt) : 0;
  const stopLatency = Math.round(performance.now() - onset);
  // Audio goes quiet locally first: the stop must not wait for a round trip.
  stopAudio();
  if (!speech) return;
  try {
    const response = await fetch('/api/voice/interrupt', {
      method:'POST', headers: authHeaders(),
      body:JSON.stringify({
        session_id: state.sessionId || await createSession(), speech_id: speech.speech_id,
        partial_text: partial || state.transcript || '', heard_ms: heard, reason,
      }),
    });
    const result = await response.json();
    if (result.resume_after_ms) state.speakNotBefore = performance.now() + result.resume_after_ms;
    if (result.speech?.truncated) {
      const bubbles = messages.querySelectorAll('.message.assistant .bubble');
      const last = bubbles[bubbles.length - 1];
      if (last && result.speech.text) last.textContent = result.speech.text;
      showToast('Stopped. Go ahead — I am listening.');
    }
    fetch('/api/voice/metrics', {
      method:'POST', headers: authHeaders(),
      body:JSON.stringify({speech_id: speech.speech_id, stopped_ms: stopLatency}),
    }).catch(() => {});
  } catch { /* the caller already has silence; recording it can fail quietly */ }
}

function stopSpeaking() {
  if (state.ttsSpeech) interruptSpeech('button');
  else stopAudio();
}

async function beginSpeech(text) {
  try {
    const response = await fetch('/api/voice/speech', {
      method:'POST', headers: authHeaders(),
      body:JSON.stringify({session_id: state.sessionId || await createSession(), text}),
    });
    return response.ok ? await response.json() : null;
  } catch { return null; }
}

// --- spoken output ---------------------------------------------------------
// One AudioContext per call, a playhead that only moves forward, and one Timbre
// socket at a time. Sentences are synthesised in order while the previous one
// is still playing, which is what removes the gap between them — and it is why
// the server's streamed sentences are worth anything at all.

function voiceContext() {
  if (!state.speaking) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    state.speaking = new AudioCtx({sampleRate: 24000});
    state.voiceNextTime = state.speaking.currentTime + 0.06;
  }
  if (state.speaking.state === 'suspended') state.speaking.resume().catch(() => {});
  return state.speaking;
}

// sentenceChunks and remainingSpeech live in voice-logic.js so they can be
// tested with `node --test`; the page loads that file first.
const { sentenceChunks, remainingSpeech } = window.VoiceLogic;

async function ensureSpeech(text) {
  if (state.ttsSpeech) return state.ttsSpeech;
  // A speech turn has to exist before the first sentence is queued, so a
  // barge-in can truncate the record and stop synthesis at the source.
  if (!state.speechPromise) state.speechPromise = beginSpeech(text);
  const speech = await state.speechPromise;
  state.speechPromise = null;
  if (speech) state.ttsSpeech = speech;
  return speech;
}

// Serialised so two sentences arriving in the same millisecond cannot reorder.
let speechChain = Promise.resolve();
function speakSentence(text, language, track = true) {
  const clean = String(text || '').trim();
  if (!clean) return speechChain;
  // Synchronously update spokenSoFar so when 'final' arrives at stream end,
  // streamed sentences are already accounted for and never repeated twice.
  if (track) {
    state.spokenSoFar = state.spokenSoFar ? `${state.spokenSoFar} ${clean}` : clean;
  }
  const generation = state.speechGeneration;
  speechChain = speechChain.then(async () => {
    if (generation !== state.speechGeneration) return;  // the caller has taken the floor
    await ensureSpeech(clean);
    if (generation !== state.speechGeneration) return;
    for (const chunk of sentenceChunks(clean)) state.voiceQueue.push({ text: chunk, language });
    drainVoice();
  });
  return speechChain;
}

// Audio is scheduled ahead of the playhead, so "synthesis finished" is not
// "the caller has stopped hearing it". The speaking state and the speech turn
// are cleared only once the playhead has actually run out — otherwise the orb
// stays lit and a stray cough is treated as an interruption of a finished
// answer.
function settleVoiceState() {
  setTimeout(() => {
    if (state.voiceSpeaking || state.voiceQueue.length) return;
    if (!state.speaking) return;
    const remaining = (state.voiceNextTime - state.speaking.currentTime) * 1000;
    if (remaining > 80) { settleVoiceState(); return; }
    setSpeaking(false);
    state.ttsSpeech = null;
    state.speechStartedAt = 0;
  }, 250);
}

async function drainVoice() {
  if (state.voiceSpeaking) return;
  if (!state.config?.speech?.gnani_configured) {
    state.voiceQueue = [];
    showToast('Voice playback is not available here. The answer stays on screen.');
    return;
  }
  state.voiceSpeaking = true;
  const generation = state.speechGeneration;
  try {
    while (state.voiceQueue.length && generation === state.speechGeneration) {
      const item = state.voiceQueue.shift();
      await speakChunk(item.text, item.language);
    }
  } finally {
    state.voiceSpeaking = false;
    settleVoiceState();
  }
}

function speakChunk(text, language) {
  return new Promise(resolve => {
    let settled = false;
    let socket = null;
    const done = () => {
      if (settled) return;
      settled = true;
      state.ttsSockets = (state.ttsSockets || []).filter(item => item !== socket);
      resolve();
    };
    try {
      const context = voiceContext();
      const requestedAt = performance.now();
      const waitMs = Math.max(0, (voicePlan().start_speaking?.wait_seconds ?? 0.4) * 1000);
      const speakAt = Math.max(requestedAt + waitMs, state.speakNotBefore || 0);
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const speech = state.ttsSpeech;
      const query = `?language=${encodeURIComponent(language || 'auto')}` +
        (speech && state.sessionId ? `${sessionSocketQuery()}&speech_id=${encodeURIComponent(speech.speech_id)}` : '');
      socket = new WebSocket(`${protocol}//${location.host}/api/voice/timbre${query}`);
      state.ttsSocket = socket;
      if (!Array.isArray(state.ttsSockets)) state.ttsSockets = [];
      state.ttsSockets.push(socket);
      let firstChunk = true;
      const close = () => { try { socket.close(); } catch {} };
      socket.onopen = () => socket.send(JSON.stringify({
        text, language: language || 'auto', speed: 1.0,
        audio_config:{sample_rate:24000,num_channels:1,sample_width:2,encoding:'linear_pcm',container:'raw'},
      }));
      socket.onmessage = event => {
        let packet; try { packet = JSON.parse(event.data); } catch { return; }
        if (packet.type === 'error') { showToast('Voice playback failed. The answer stays on screen.'); close(); done(); return; }
        const encoded = packet.data?.audio || packet.audio;
        const final = packet.data?.is_final || packet.is_final || packet.type === 'complete';
        if (!encoded) { if (final) { close(); done(); } return; }
        if (firstChunk) {
          firstChunk = false;
          // Measured from the request, not the first packet: the caller waits
          // for the whole path, not for the last hop.
          if (!state.speechStartedAt) state.speechStartedAt = performance.now();
          setSpeaking(true);
          const turn = state.ttsSpeech;
          if (turn) fetch('/api/voice/metrics', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify({speech_id: turn.speech_id, first_audio_ms: Math.round(performance.now() - requestedAt)}),
          }).catch(() => {});
        }
        const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
        const view = new DataView(bytes.buffer); const floats = new Float32Array(Math.floor(bytes.byteLength / 2));
        for (let i=0;i<floats.length;i++) floats[i] = view.getInt16(i*2,true)/32768;
        const buffer = context.createBuffer(1,floats.length,24000); buffer.copyToChannel(floats,0);
        const source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination);
        const earliest = context.currentTime + Math.max(0.02, (speakAt - performance.now()) / 1000);
        const startAt = Math.max(earliest, state.voiceNextTime);
        source.start(startAt); state.voiceNextTime = startAt + buffer.duration;
        state.ttsNodes.push(source);
        if (final) { close(); done(); }
      };
      socket.onerror = () => done();
      socket.onclose = () => done();
    } catch { showToast('Voice is unavailable right now. The answer stays on screen.'); done(); }
  });
}

async function speakAnswer(text, language) {
  if (!state.config?.speech?.gnani_configured) { showToast('Voice playback is not available here. The answer stays on screen.'); return; }
  if (!language || language === 'unavailable') { showToast('There is no voice for this language yet. The answer stays on screen.'); return; }
  await speakSentence(text, language);
}

function onMicLevel(frame) {
  // Cheap RMS over the captured frame. This is the local VAD spike the stop
  // plan uses, and it has to be local: waiting for a transcript round trip to
  // notice that someone started talking is exactly the delay people hate.
  let rms = 0;
  try {
    const ints = frame instanceof Int16Array ? frame : new Int16Array(frame instanceof ArrayBuffer ? frame : frame.buffer);
    let sum = 0;
    for (let i = 0; i < ints.length; i += 4) { const value = ints[i] / 32768; sum += value * value; }
    rms = Math.sqrt(sum / Math.max(1, Math.ceil(ints.length / 4)));
  } catch { return; }
  if (state.recording) {
    const now = performance.now();
    if (now - (onMicLevel.lastPaint || 0) > 60) {
      onMicLevel.lastPaint = now;
      setMicLevel(rms * 5);
    }
  }
  if (rms > 0.055) {
    if (!state.vadOnsetAt) state.vadOnsetAt = performance.now();
  } else {
    state.vadOnsetAt = 0;
  }
  if (!state.ttsSpeech || !state.vadOnsetAt) return;
  const required = (voicePlan().stop_speaking?.voice_seconds ?? 0.2) * 1000;
  if (performance.now() - state.vadOnsetAt >= required) interruptSpeech('caller_spoke', state.transcript || '');
}

function maybeBargeInByText(text) {
  const plan = voicePlan().stop_speaking || {};
  if (!state.ttsSpeech || !plan.num_words) return;
  const words = String(text || '').trim().split(/\s+/).filter(Boolean).length;
  if (words >= plan.num_words) interruptSpeech('caller_spoke', text);
}

function scheduleEndpoint(packet) {
  // Natural turn-taking: the caller does not press a button to finish.
  //
  // Prisma has already endpointed — a `transcript` frame is final and only
  // arrives after the server VAD closed the segment. Waiting the full
  // "unfinished clause" delay on top of that doubled the silence the caller
  // heard, so once there is a transcript only a short merge window remains, to
  // catch a second clause of the same breath.
  clearTimeout(state.endpointTimer);
  if (!state.recording) return;
  const plan = voicePlan().start_speaking || {};
  const text = state.transcript.trim();
  if (!text) return;
  const last = text.slice(-1);
  let delay = (plan.on_final_segment_seconds ?? 0.35) * 1000;
  if (/\d/.test(last)) delay = (plan.on_number_seconds ?? 0.6) * 1000;   // still reading a number
  else if (/[.!?।]/.test(last)) delay = (plan.on_punctuation_seconds ?? 0.3) * 1000;
  state.endpointTimer = setTimeout(() => {
    // Never fire a second turn into a turn in flight; the text stays in the
    // composer so it can be sent once the answer lands.
    if (state.recording && state.transcript.trim() && !state.ttsSpeech && !state.busy) stopMic(true);
  }, delay);
}

function noteDetectedLanguage(code) {
  if (!code) return;
  state.detectedLanguage = code;
  // An override is honoured only when the caller actually chose one.
  if (languageSelect.value === 'auto') state.language = code;
  const labels = state.config?.language_labels || {};
  const hint = $('#voiceHint');
  if (hint) hint.textContent = `Following you in ${labels[code] || code} — speak naturally, I will wait for you.`;
  document.documentElement.lang = code;
}

function openTranscriptStream(override) {
  // No language is required. `auto` asks Prisma v3 to identify the language of
  // each utterance and report it, which is what removes the question entirely.
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/voice/prisma?language=${encodeURIComponent(override || 'auto')}&sample_rate=${state.audioContext.sampleRate}${sessionSocketQuery()}`);
  socket.binaryType = 'arraybuffer';
  state.sttSocket = socket;
  socket.onmessage = event => {
    let packet; try { packet = JSON.parse(event.data); } catch { return; }
    if (packet.type === 'error') { showToast(packet.message); return; }
    if (packet.type === 'transcript' && typeof packet.text === 'string') {
      const part = packet.text.trim();
      if (packet.detected_language_code) noteDetectedLanguage(packet.detected_language_code);
      if (part) {
        // A transcript is proof the stream is healthy, so the reconnect budget
        // is restored here rather than on open: a socket that opens and closes
        // without producing speech must still run out of retries.
        state.sttRestarts = 0;
        state.transcript = `${state.transcript} ${part}`.trim();
        input.value = state.transcript;
        maybeBargeInByText(part);
        scheduleEndpoint(packet);
      }
    }
  };
  socket.onerror = () => { /* onclose handles recovery */ };
  socket.onclose = () => {
    // The server re-dials Prisma itself; this covers the browser's own socket
    // dying (Wi-Fi handover, tab throttling). Losing the microphone silently
    // mid-sentence is worse than a short gap.
    if (!state.recording) return;
    if (state.sttRestarts >= 2) { showToast('Speech recognition stopped. Tap the microphone to continue.'); stopMic(false); return; }
    state.sttRestarts += 1;
    setTimeout(() => { if (state.recording) openTranscriptStream(override); }, 250);
  };
}

async function startMic() {
  if (!state.config?.speech?.gnani_configured) {
    // An operator's setup instruction is not a caller's error message.
    showToast('Voice is not available here. You can type instead.');
    input.focus();
    return;
  }
  const override = languageSelect.value || 'auto';
  stopSpeaking();
  try {
    // Open the session before the microphone so Prisma gets the call's previous
    // transcript as `previous_text` from the very first utterance, and so a
    // barge-in during that utterance has a speech turn to bind to.
    await createSession().catch(() => null);
    state.micStream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioCtx();
    if (!state.audioContext.audioWorklet) throw new Error('This browser does not support live audio capture. Type your message instead.');
    await state.audioContext.audioWorklet.addModule('/static/pcm-worklet.js');
    // The microphone is captured once; reconnects only replace the socket. The
    // worklet writes to whichever stream is current, so a reconnect does not
    // duplicate audio nodes.
    const source = state.audioContext.createMediaStreamSource(state.micStream);
    const node = new AudioWorkletNode(state.audioContext, 'pcm-tap');
    const mute = state.audioContext.createGain(); mute.gain.value = 0;
    node.port.onmessage = event => {
      if (state.sttSocket && state.sttSocket.readyState === WebSocket.OPEN) state.sttSocket.send(event.data);
      onMicLevel(event.data);
    };
    source.connect(node); node.connect(mute); mute.connect(state.audioContext.destination);
    state.micSource = source; state.workletNode = node;
    state.transcript = ''; state.sttRestarts = 0;
    state.recording = true; micButton.classList.add('recording'); setListening(true);
    const auto = state.config?.speech?.auto_detect !== false && override === 'auto';
    showToast(auto ? 'Listening — speak in any language, in any mix.' : 'Listening now.');
    openTranscriptStream(override);
  } catch (error) { showToast(error.message || 'Microphone permission is required.'); stopMic(false); }
}
function stopMic(sendTranscript = false) {
  clearTimeout(state.endpointTimer); state.endpointTimer = null; state.vadOnsetAt = 0;
  const transcript = state.transcript.trim();
  state.recording = false;
  if (state.workletNode) { try { state.workletNode.disconnect(); } catch {} state.workletNode = null; }
  if (state.micSource) { try { state.micSource.disconnect(); } catch {} state.micSource = null; }
  if (state.micStream) { state.micStream.getTracks().forEach(track => track.stop()); state.micStream = null; }
  if (state.audioContext) { state.audioContext.close().catch(() => {}); state.audioContext = null; }
  if (state.sttSocket && state.sttSocket.readyState < WebSocket.CLOSING) state.sttSocket.close();
  state.sttSocket = null; micButton.classList.remove('recording'); setListening(false); setMicLevel(0);
  // `state.language` already follows what the caller spoke; the selector is
  // only an override and must not overwrite a detected language here.
  if (sendTranscript && transcript) sendTurn(transcript, state.pendingService, true);
  state.transcript = '';
}

// The composer's hint has to fit the field it sits in. On a 320px screen the
// long sentence wraps and is clipped by the single-row textarea, so the narrow
// viewport gets the short question instead.
function fitPlaceholder() {
  input.placeholder = window.innerWidth < 380 ? 'What happened?' : 'Speak or type what happened…';
}

function autosize() {
  // One row when empty, up to the cap in the stylesheet. Re-measured from
  // `auto` so deleting text shrinks it again — a textarea that only grows
  // leaves the composer permanently taller than its content.
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.3)}px`;
}

$('#composer').addEventListener('submit', event => {
  event.preventDefault();
  sendTurn(input.value, state.pendingService, false);
});
stopButton.addEventListener('click', () => {
  if (state.abort) state.abort.abort();
  stopSpeaking();
});
input.addEventListener('input', autosize);
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendTurn(input.value, state.pendingService, false);
  }
});
micButton.addEventListener('click', () => state.recording ? stopMic(true) : startMic());

// Starters open the conversation in one tap. Filling the composer first made
// every example a two-step action, and the caller can still correct the facts
// in their next turn — which is what the intake is for.
document.querySelectorAll('.starter').forEach(button => {
  button.addEventListener('click', () => {
    sendTurn(button.dataset.starter || '', state.pendingService, false);
  });
});
languageSelect.addEventListener('change', () => { state.language = languageSelect.value; });
$('#consentCheck').addEventListener('change', () => { $('#beginButton').disabled = !$('#consentCheck').checked; });
$('#beginButton').addEventListener('click', () => {
  state.consent = $('#consentCheck').checked; state.language = languageSelect.value;
  if (state.consent) rememberConsent(state.config?.consent_notice_version);
  hideDialog(consentModal); input.focus();
});
async function startNewConversation() {
  if (state.sessionId) {
    try { await fetch(`/api/session/${encodeURIComponent(state.sessionId)}`, { method: 'DELETE', headers: authHeaders() }); } catch {}
  }
  location.reload();
}
$('#newCase').addEventListener('click', startNewConversation);
$('#newCaseTop').addEventListener('click', startNewConversation);
$('#sourceButton').addEventListener('click', openGuides);
$('#savedItemsButton').addEventListener('click', openSaved);
$('#closeSheet').addEventListener('click', closeSheet);
sheetBackdrop.addEventListener('click', event => { if (event.target === sheetBackdrop) closeSheet(); });
$('#helpButton').addEventListener('click', openHelp);
$('#safetyLink').addEventListener('click', openHelp);
$('#shareReview').addEventListener('change', () => { $('#confirmReview').disabled = !$('#shareReview').checked; });
$('#includeIdentifiers')?.addEventListener('change', refreshReviewPreview);
$('#confirmReview').addEventListener('click', requestHumanReview);
$('#closeReview').addEventListener('click', () => hideDialog($('#reviewModal')));
$('#reviewModal').addEventListener('click', event => { if (event.target.id === 'reviewModal') hideDialog($('#reviewModal')); });
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  closeSheet();
  hideDialog($('#reviewModal'));
  stopSpeaking();
});
window.addEventListener('beforeunload', stopSpeaking);

/* ---------- app shell: install, navigation, keyboard ---------------------- */
// Registered from the root so it controls the whole app, not /static/.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {
      /* Private mode and insecure origins refuse service workers; the page
         works without one, it just stops being installable. */
    });
  });
}

const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
const iOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
let installEvent = null;

function showInstall(visible) {
  const button = $('#installButton');
  if (!button) return;
  button.classList.toggle('hidden', !visible || standalone);
}

window.addEventListener('beforeinstallprompt', event => {
  // Chrome's own mini-infobar is easier to miss than a button in the app.
  event.preventDefault();
  installEvent = event;
  showInstall(true);
});

window.addEventListener('appinstalled', () => {
  installEvent = null;
  showInstall(false);
  showToast('Nyaya is installed. Open it from your home screen.');
});

async function promptInstall() {
  if (installEvent) {
    installEvent.prompt();
    const choice = await installEvent.userChoice.catch(() => ({ outcome: 'dismissed' }));
    if (choice.outcome === 'accepted') { installEvent = null; showInstall(false); }
    return;
  }
  // Safari has no install prompt event at all.
  showToast(iOS
    ? 'To install: tap Share, then “Add to Home Screen”.'
    : 'Use your browser menu → “Install app” or “Add to Home screen”.');
}
$('#installButton')?.addEventListener('click', promptInstall);
// Safari only exposes the affordance through the menu, but the button is still
// the place people look, so it explains the two taps rather than hiding.
if (iOS && !standalone) showInstall(true);

function setNav(name) {
  document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('is-active', item.dataset.nav === name));
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const target = item.dataset.nav;
    if (target === 'guides') { setNav('guides'); openGuides(); return; }
    if (target === 'saved') { setNav('saved'); openSaved(); return; }
    closeSheet();
    setNav('talk');
    input.focus();
  });
});

// The on-screen keyboard shrinks the visual viewport without resizing fixed
// elements on iOS, which would leave the navigation bar sitting on top of the
// composer. This measures the overlap and lifts the bar by exactly that much.
if (window.visualViewport) {
  const viewport = window.visualViewport;
  const syncKeyboard = () => {
    const overlap = Math.max(0, window.innerHeight - viewport.height - viewport.offsetTop);
    document.documentElement.style.setProperty('--keyboard-inset', `${Math.round(overlap)}px`);
    // The tab bar is replaced by the keyboard rather than stacked above it, so
    // the composer stays the last thing on the screen while typing.
    document.body.classList.toggle('keyboard-open', overlap > 80);
  };
  viewport.addEventListener('resize', syncKeyboard);
  viewport.addEventListener('scroll', syncKeyboard);
  syncKeyboard();
}

// Manifest shortcuts launch with a query so they land on the right section.
// The query is cleared first: a reload must not re-run the shortcut, which
// would make "New conversation" reload itself forever.
const launchParams = new URLSearchParams(location.search);
if (launchParams.get('new') === '1') {
  history.replaceState(null, '', '/');
  startNewConversation();
}
if (launchParams.get('sources') === '1') {
  history.replaceState(null, '', '/');
  window.addEventListener('load', openGuides);
}

// The consent notice ships visible in the markup so it cannot be missed while
// scripts load. That means the first-run path has to join the same focus model
// as every other dialog: background inert, focus on the notice.
if (!consentModal.classList.contains('hidden')) {
  if ('inert' in HTMLElement.prototype && appShell) appShell.inert = true;
  consentModal.querySelector('.modal')?.focus();
}

loadConfig(); loadCapabilities(); fitPlaceholder(); autosize();
window.addEventListener('resize', fitPlaceholder);
