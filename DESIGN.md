# Nyaya — design system

Recorded from the shipped CSS (`static/styles.css`) and markup (`static/index.html`).
Mode: **Operate** — the visitor is completing a task under stress, often at night,
often one-handed on a phone. Precision and restraint outrank expression.

The world: **a case file under a desk lamp.** Warm ink, paper, a wax seal, and one
warm bulb. Colour is chosen from that scene, not from a UI kit, which is why there
is no mint, no frosted glass, no gradient, and no cool grey anywhere in the file.

## Principles

1. **The answer is the interface.** Structure carries hierarchy: status, the one
   question to answer, then evidence folded away. No decorative container competes
   with the transcript.
2. **One piece of light.** Amber is reserved for what is live (an open microphone,
   a focus ring) and for the one control that moves a case to a person. Everything
   else is a border on a surface.
3. **Red means harm, never brand.** The seal vermilion appears only on danger and
   on the full stop in the wordmark.
4. **Honest state.** No progress the app cannot measure, no "reviewed" label on an
   unreviewed source, no fake liveness. A disabled control is an outline, not a
   faded slab.

## Surfaces

| Token | Value | Use |
| --- | --- | --- |
| `--bg` | `#14100d` | Page (warm ink) |
| `--bg-2` | `#1b1612` | Rail, sheets, modals, composer |
| `--surface` | `rgba(242,234,220,.045)` | Cards, chips, quiet buttons |
| `--surface-2` | `rgba(242,234,220,.075)` | User bubble, hover |
| `--surface-3` | `rgba(242,234,220,.11)` | Pressed / selected |
| `--line` | `rgba(242,234,220,.11)` | Default border |
| `--line-strong` | `rgba(242,234,220,.20)` | Focusable borders, composer |

Text: `--text #f2eadc` · `--text-2 #c6b8a5` · `--text-3 #9c8e7c`.
Material: `--paper #f2eadc`, `--paper-strong #fbf5e9`, `--ink #14100d`.
Accent (the lamp): `--accent #d9a361`, `--accent-strong #e8b77a`, `--accent-ink #1a1208`.
Semantic: `--seal #d6634b`, `--warn #d9a361`, `--danger #e0664e`.

Body and placeholder text stay ≥ 4.5:1 against their own surface; icons and
borders ≥ 3:1. Secondary text is tinted warm, never grey. No gradient behind text,
no blur behind a text field.

**Paper is the primary-action material.** A filled paper slab (ink text) is used for
the consent button and for "Talk to a qualified person" — signing something, or
handing a case to a human. Amber is never a button fill; it is a ring, a dot, a rule.

## Type

- UI: system sans (`ui-sans-serif` …). Display: a serif with drawn letterforms —
  Iowan Old Style, Palatino, Georgia, Noto Serif — used for the wordmark, the
  welcome line, and dialog titles.
- Scale: `12 / 13 / 16 / 17 / 20 / clamp(32–44)`. Sizes are declared on the tags
  (`h1` clamp, `h2` 26, `h3` 20) so no heading ever falls back to body size.
- Body is 16px. Intro and modal lead 17px. Sheet prose is 16px: a panel that is
  read, not scanned, does not get fine print.
- Measure: 68ch for prose, 52ch for the question, 46ch for the welcome lead.
- Headings balanced (`text-wrap: balance`), prose `pretty`.
- Numerals: tabular in procedure steps and step counters.
- No webfonts: a legal answer must not wait on a font download.

## Space, shape, depth

- 4px rhythm: `4 / 8 / 12 / 16 / 24 / 32 / 48 / 64`.
- Radii: `6 / 8 / 12 / 14 / 16`. Squared by temperament; no 24px capsule anywhere,
  circles only for the microphone and send control.
- Elevation is a 1px border, never a shadow — the one exception is the toast, which
  floats and carries `0 12px 32px rgba(0,0,0,.45)`.
- No card contains a card.
- Layout constants: rail 236px, content column 744px, top bar 52px, mobile nav 56px.

## Components

| Component | Rules |
| --- | --- |
| Wordmark | `Nyaya.` in the display serif, the full stop in seal vermilion. No second lockup line. |
| Rail | Wordmark, new conversation, three reference links, status at the foot. Hidden below 1024px. |
| Top bar | Mode chip (`Full assistant` / `Sources only`), install, new conversation on mobile. No bottom rule on desktop. |
| Welcome | One serif sentence, one lead, then an index of three ruled rows (label left, drawn chevron right). Not chips. |
| Transcript | One scrolling column. User turns right-aligned in a bordered bubble; answers full-width, status first. |
| Status pill | Legal-trust label derived from the answer's own review state: `Source-linked`, `Draft procedure`, `Reviewed procedure`, `Could not verify`, `Safety first`. |
| Disclosure | A quiet inline row with a 14px chevron that rotates on open; the panel below aligns to the column. One pattern for steps, passages and reasoning. |
| Source card | Document icon, title, authority · citation · review state. Whole card is the link. |
| Actions | Speak / Stop / Copy are borderless utilities. The human handoff is the only filled control in the transcript. |
| Pending | A fixed segment that breathes. No travelling bar, no dot pulse. |
| Composer | Opaque ink panel, 44px paper orb, 40px outlined send, two 11.5rem selects. Collapses to one visible row; the hint row exists only while the microphone is open. |
| Sheet | Right drawer ≥1024px, bottom sheet below. One focus trap for sheets and dialogs; focus goes to the panel, not its close button. |
| Modal | Blocking consent and handoff only. Serif title, ruled points, paper signature button. |
| Toast | Transient confirmation, never the only record of an outcome. |

Icons: one authored 24px set, 1.5 stroke, `currentColor`. No glyphs or emoji as icons.
Browser surfaces are themed too: selection, caret, scrollbars, focus ring, and the
service worker's offline page use the same four colours as the app.

## Motion

- Easings: `--ease-out cubic-bezier(.23,1,.32,1)`, `--ease-in-out cubic-bezier(.77,0,.175,1)`.
- Durations: press 120ms, UI 180ms, panel 240ms.
- Entrances: 6px rise + fade, 180ms. One authored moment per view, not one identical
  entrance on every element.
- Loops are permitted only for a genuinely pending request (the breathing rule).
- All motion stops under `prefers-reduced-motion`; fold scrolling switches to instant.

## Accessibility

- Visible focus ring (2px amber, 2px offset) on every control; dialogs and the sheet
  focus on open, trap Tab, set the app shell `inert`, restore focus to the opener.
- Skip link to the message box, revealed on focus.
- `lang` follows the detected/selected language so screen readers switch voice.
- Touch targets ≥ 36px on phones (44px for the orb), 48px in the bottom nav.
- Live regions: transcript is `aria-live="polite"`; the pending label carries stage text.
- No inline `style` attributes: the CSP is `style-src 'self'`, so spacing lives in classes.

## Responsive

| Width | Behaviour |
| --- | --- |
| ≥ 1024 | Rail visible, sheet is a right drawer, composer holds both 11.5rem selects on one row |
| 600–1023 | Rail becomes a bottom tab bar; sheet becomes a bottom sheet; selects share a row |
| ≤ 599 | Centred welcome, compact composer, select row shared, thumb-sized actions |
| ≤ 380 | Selects take a row each; placeholder shortens to "What happened?"; safety icon drops |

Verified widths: 320, 390, 768, 1440 — no horizontal overflow.

## Not allowed here

Eyebrow labels above headings; coloured 2px left stripes; unicode glyphs as icons;
gradient text; glass or blur as decoration; nested cards; hard offset shadows;
looping decorative animation; fabricated progress; a "reviewed" claim not backed by
`review_status`; a second green accent; raw ISO codes shown to a citizen.
