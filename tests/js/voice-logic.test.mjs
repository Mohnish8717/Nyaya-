// Browser-side speech logic, tested without a browser.
//
//   node --test tests/js
//
// These functions decide what is synthesised and what still needs saying. A
// bug here is audible: half a sentence spoken twice, or the tail of an answer
// lost after a barge-in.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { sentenceChunks, remainingSpeech } = require('../../static/voice-logic.js');

test('an empty or whitespace reply produces no speech', () => {
  assert.deepEqual(sentenceChunks(''), []);
  assert.deepEqual(sentenceChunks('   \n '), []);
  assert.deepEqual(sentenceChunks(null), []);
});

test('each sentence becomes its own chunk', () => {
  assert.deepEqual(
    sentenceChunks('You can file it. Keep a copy. Ask for a receipt.'),
    ['You can file it.', 'Keep a copy.', 'Ask for a receipt.'],
  );
});

test('a Devanagari danda ends a sentence too', () => {
  assert.deepEqual(sentenceChunks('पहले शिकायत लिखें। फिर जमा करें।'), [
    'पहले शिकायत लिखें।',
    'फिर जमा करें।',
  ]);
});

test('short clauses merge until the limit is reached', () => {
  const chunks = sentenceChunks('One. Two. Three. Four. Five.', 22);
  assert.ok(chunks.length >= 2);
  assert.ok(chunks.every((chunk) => chunk.length <= 22), JSON.stringify(chunks));
  assert.equal(chunks.join(' ').replace(/\s+/g, ' '), 'One. Two. Three. Four. Five.');
});

test('an unterminated tail is still spoken', () => {
  assert.deepEqual(sentenceChunks('The office is open until'), ['The office is open until']);
});

test('a streamed prefix is not repeated', () => {
  const full = 'You can file a complaint. Keep a copy of the receipt.';
  assert.equal(remainingSpeech(full, 'You can file a complaint.'), 'Keep a copy of the receipt.');
  assert.equal(remainingSpeech(full, ''), full);
  assert.equal(remainingSpeech(full, full), '');
});

test('a regenerated reply says nothing rather than contradicting what was heard', () => {
  const full = 'A different answer entirely.';
  assert.equal(remainingSpeech(full, 'You can file a complaint.'), '');
});

test('remainingSpeech on an empty reply is empty', () => {
  assert.equal(remainingSpeech('', 'anything'), '');
  assert.equal(remainingSpeech(null, null), '');
});
