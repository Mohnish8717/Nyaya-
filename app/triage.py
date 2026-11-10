"""Risk triage: decide *how dangerous* a turn is before deciding what to say.

Keyword routing for "emergency" is not enough for a legal-aid product. The
best clinical and legal triage systems separate three questions:

  1. what is the risk (category),
  2. how urgent is it (severity),
  3. what does that forbid the system from doing (policy).

Only the third one prevents harm. A drafting request during an active threat
must not produce a document; a fraud still in progress must prioritise the
1930 call over legal analysis; a caller in custody or with a child at risk
must not be told to "collect evidence" first.

Everything here is deterministic and unit-tested. A model may add flags, but
it cannot lower a severity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SEVERITY_ORDER = {"low": 0, "normal": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class Category:
    key: str
    severity: str
    patterns: tuple[str, ...]
    policies: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        "immediate_danger", "critical",
        (
            r"\b(he|she|they|husband|father|brother|uncle|landlord|partner)\b[^.]{0,40}\b(is coming|will kill|threaten\w*|beating|hit me|attacked)\b",
            r"\b(coming (right )?now|outside my (house|door)|locked (me )?in|not allowed to leave)\b",
            r"\b(maar raha|maar diya|jaan se|dhamki de raha)\b",
            r"\b(acid attack|molest|rape|kidnap|abduct)\b",
            # A caller in danger does not switch script for our convenience.
            # Every language the speech models serve has its own threat verbs
            # here, because this net is the safety path that must work with no
            # model configured.
            r"(मार रह|मारता|मारती|मार डाल|जान से मार|पीट रह|पीटता|धमकी दे रह)",
            r"(बलात्कार|छेड़छाड़|अपहरण)",
            r"(మారుస్త|చంపేస్త|చంపుతా|కొడుతున్న|ప్రాణం పోత)",           # te
            r"(கொலை|கொல்லுவேன்|கொல்கிறான்|அடிக்கிறான்|உயிருக்கு ஆபத்து)",   # ta
            r"(মেরে ফেল|মারছে|মারধর|খুন)",                                  # bn
            r"(મારી નાખ|મારે છે|કતલ)",                                     # gu
            r"(ಕೊಲ್ಲುತ್ತ|ಹೊಡೆಯುತ್ತ|ಕೊಲೆ)",                                  # kn
            r"(കൊല്ലും|കൊല്ലുക|അടിക്കുന്നു|ജീവന് ഭീഷണി)",                     # ml
            r"(ਮਾਰ ਦੇਵੇ|ਮਾਰ ਰਿਹਾ|ਕਤਲ)",                                    # pa
            r"(मारून टाक|मारहाण|जीवाला धोका)",                              # mr
        ),
        ("safety_first", "block_drafting", "offer_emergency", "no_legal_analysis"),
    ),
    Category(
        "self_harm", "critical",
        (
            r"\b(suicide|kill myself|end my life|self[- ]harm|khudkhushi)\b",
            r"(आत्महत्या|खुदकुशी|जान दे दूँ)",
            r"(ఆత్మహత్య|చనిపోవాలి|బతకాలని లేదు)",        # te
            r"(தற்கொலை|வாழ்க்கை வேண்டாம்|சாக வேண்டும்)",    # ta
            r"(আত্মহত্যা|বাঁচতে চাই না)",                 # bn
            r"(આત્મહત્યા|જીવવું નથી)",                   # gu
            r"(ಆತ್ಮಹತ್ಯೆ|ಬದುಕಲು ಇಷ್ಟವಿಲ್ಲ)",              # kn
            r"(ആത്മഹത്യ|ജീവിക്കണ്ട)",                    # ml
            r"(ਖੁਦਕੁਸ਼ੀ|ਆਤਮਹੱਤਿਆ|ਜੀਣਾ ਨਹੀਂ)",            # pa
            r"(आत्महत्या|जगायचं नाही|जीव द्यायचा)",      # mr
        ),
        ("safety_first", "block_drafting", "offer_emergency", "no_legal_analysis"),
    ),
    Category(
        "child_at_risk", "critical",
        (r"\b(child|minor|baby|bachch?a)\b[^.]{0,40}\b(beat|hit|hurt|missing|taken|abduct)\b",),
        ("safety_first", "block_drafting", "offer_emergency"),
    ),
    Category(
        "fraud_in_progress", "high",
        (
            r"\b(otp|upi|bank|card|account)\b[^.]{0,40}\b(debited|transferred|withdrawn|hacked|fraud)\b",
            r"\b(money (is )?(being |just )?(taken|transferred)|abhi paisa)\b",
        ),
        ("preserve_evidence", "report_now", "block_drafting"),
    ),
    Category(
        "domestic_violence", "high",
        (
            r"\b(domestic violence|husband beats|in[- ]laws|dowry|gharelu hinsa|abuse at home)\b",
            r"\b(violence|maarta hai|maar peet)\b",
            r"(घरेलू हिंसा|दहेज)",
        ),
        ("safety_first", "offer_support_helpline"),
    ),
    Category(
        "eviction_or_lockout", "high",
        (
            r"\b(evict\w*|lock(ed)?\b[^.]{0,14}\bout|thrown out|papers? on (the )?door|makan se nikal)\b",
            r"\b(lock(ed)? (me )?(in|outside)|changed the locks)\b",
        ),
        ("urgent_deadline",),
    ),
    Category(
        "deadline_pressure", "normal",
        (
            r"\b(tomorrow|today|tonight|within \d+ (hours|days)|last date|kal|aaj)\b[^.]{0,60}\b(court|hearing|notice|deadline|submit|file)\b",
            r"\b(court (is )?(tomorrow|today)|hearing (is )?(tomorrow|today))\b",
        ),
        ("mention_deadline_without_computing",),
    ),
    Category(
        "in_custody", "high",
        (r"\b(in (police )?custody|arrested|detained|thana me bandh)\b",),
        ("rights_first", "no_delay"),
    ),
)


CATEGORY_BY_KEY: dict[str, Category] = {category.key: category for category in CATEGORIES}


def policies_for(key: str) -> tuple[str, ...]:
    """Policies a risk category imposes.

    Used by the semantic reader: the model names the category, the policy is
    looked up here. A model cannot invent a policy, and an unknown category
    carries none.
    """
    category = CATEGORY_BY_KEY.get(str(key or "").strip())
    return category.policies if category else ()


@dataclass
class Assessment:
    level: str = "normal"
    categories: list[dict[str, str]] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def has(self, policy: str) -> bool:
        return policy in self.policies

    @property
    def critical(self) -> bool:
        return self.level == "critical"

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "categories": self.categories,
            "policies": self.policies,
            "reasons": self.reasons,
        }


def assess(text: str) -> Assessment:
    """Classify risk. The highest severity wins; policies accumulate."""
    lowered = (text or "").lower()
    assessment = Assessment()
    for category in CATEGORIES:
        hits = [match.group(0).strip() for pattern in category.patterns for match in [re.search(pattern, lowered)] if match]
        if not hits:
            continue
        assessment.categories.append({"key": category.key, "severity": category.severity, "evidence": hits[0][:120]})
        for policy in category.policies:
            if policy not in assessment.policies:
                assessment.policies.append(policy)
        assessment.reasons.append(f"{category.key}: {hits[0][:80]}")
        if SEVERITY_ORDER[category.severity] > SEVERITY_ORDER[assessment.level]:
            assessment.level = category.severity
    for policy in ("block_drafting", "no_legal_analysis", "safety_first"):
        if assessment.has(policy):
            break
    return assessment


def is_emergency(assessment: Assessment) -> bool:
    return assessment.critical and (
        assessment.has("safety_first") or assessment.has("offer_emergency")
    )


def sort_questions(questions: list[str], assessment: Assessment) -> list[str]:
    """Safety questions go first; everything else keeps its order."""
    if not assessment.has("safety_first"):
        return questions
    safety = [q for q in questions if "safe" in q.lower()]
    return safety + [q for q in questions if q not in safety]
