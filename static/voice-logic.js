/* Speech-queue logic that is pure enough to test without a browser.
 *
 * These two functions decide what gets synthesised and what still needs to be
 * said; a bug in either is audible — half a sentence spoken twice, or the tail
 * of an answer lost after a barge-in. They live here so `node --test` can
 * exercise them, and the page loads this file before app.js.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.VoiceLogic = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function sentenceChunks(text, limit = 220) {
    const clean = String(text || '').replace(/\s+/g, ' ').trim();
    if (!clean) return [];
    // A sentence ends at . ! ? or a Devanagari danda. Whatever trails the last
    // stop is its own chunk, so a truncated stream still speaks what it has.
    const parts = clean.match(/[^.!?।॥…]+[.!?।॥…]+|\S[^.!?।॥…]*$/g) || [clean];
    const chunks = [];
    let buffer = '';
    for (const part of parts) {
      const piece = part.trim();
      if (!piece) continue;
      buffer = buffer ? `${buffer} ${piece}` : piece;
      if (buffer.length >= limit || /[.!?।॥…]$/.test(piece)) { chunks.push(buffer); buffer = ''; }
    }
    if (buffer) chunks.push(buffer);
    return chunks;
  }

  function remainingSpeech(full, already) {
    const text = String(full || '').replace(/\s+/g, ' ').trim();
    const heard = String(already || '').replace(/\s+/g, ' ').trim();
    if (!text) return '';
    if (!heard) return text;
    if (text === heard || heard.includes(text)) return '';
    // Streamed sentences come from the same reply, so they are a prefix. If
    // they are not (the turn was regenerated), saying the rest would contradict
    // what the caller already heard, so nothing more is said.
    return text.startsWith(heard) ? text.slice(heard.length).trim() : '';
  }

  return { sentenceChunks, remainingSpeech };
});
