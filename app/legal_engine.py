"""Turn orchestration: route, retrieve, compose, verify.

The turn is deliberately split into `analyze` (cheap, deterministic, no
generation) and `compose` (the only stage that may call a model). That split
exists for three reasons:

  * a voice turn must be able to start speaking before generation finishes;
  * safety routes (emergency, handoff) must work with no model at all;
  * procedural answers come from a reviewed file, so they need no model
    unless the caller's language or register requires one.

The model gets **one** call per turn, and that call is the consultation. It
returns the answer together with its reading of the turn -- intent, grounded
facts, risk, the search phrasing, the one question worth asking next -- and
`_apply_consultation` validates that reading through the same closed
vocabularies, grounding checks and monotone safety net a separate reader used
to. What it does not do is let the reading replace the answer: facts it
proposed are committed to the ledger only with a verbatim span from the
caller, claims only with an exact quote from a retrieved source, and an
ungrounded emergency only raises urgency rather than triggering the script.

Nothing here invents legal authority. Claims are displayed only when a
verbatim quote from a retrieved, approved, in-date source is present, and the
verification block says plainly that entailment was not reviewed.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from . import (
    case_corpus, facts, knowledge, language as language_layer, model_gateway, privacy, procedural,
    retrieval, routing, tools as tool_layer, triage, understanding, workflow,
)
from .model_gateway import configured as model_configured
from .model_gateway import generate_turn, generate_turn_stream

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "data" / "service_catalog.json").open(encoding="utf-8") as f:
    SERVICE_CATALOG = json.load(f)


# Language identity lives in `app/language.py`: the audio provider's codes, the
# script heuristics and the caller-override rule all belong in one place. These
# names are re-exported because the API layer and the browser both read them.
TEXT_LANGUAGES = language_layer.TEXT_LANGUAGES
# What Timbre can actually voice, and what this proxy will accept on the STT
# socket (`auto`/`indic-auto` are accepted values, not languages).
TTS_LANGUAGES = language_layer.SPEECH_LANGUAGES | {language_layer.AUTO, language_layer.HINGLISH}
STT_LANGUAGES = language_layer.SPEECH_LANGUAGES | {language_layer.AUTO, language_layer.PRISMA_AUTO, language_layer.HINGLISH}
LANGUAGE_LABELS = language_layer.LABELS
INDIAN_JURISDICTIONS = {
    "IN", "IN-AN", "IN-AP", "IN-AR", "IN-AS", "IN-BR", "IN-CH", "IN-CT", "IN-DH", "IN-DL", "IN-GA",
    "IN-GJ", "IN-HR", "IN-HP", "IN-JH", "IN-JK", "IN-KA", "IN-KL", "IN-LA", "IN-LD", "IN-MH",
    "IN-ML", "IN-MN", "IN-MP", "IN-MZ", "IN-NL", "IN-OR", "IN-PB", "IN-PY", "IN-RJ", "IN-SK",
    "IN-TS", "IN-TN", "IN-TR", "IN-UP", "IN-UT", "IN-WB",
}
JURISDICTION_LABELS = {
    "IN": "India (national)", "IN-AP": "Andhra Pradesh", "IN-AR": "Arunachal Pradesh", "IN-AS": "Assam",
    "IN-BR": "Bihar", "IN-CT": "Chhattisgarh", "IN-DL": "Delhi", "IN-GA": "Goa", "IN-GJ": "Gujarat",
    "IN-HR": "Haryana", "IN-HP": "Himachal Pradesh", "IN-JH": "Jharkhand", "IN-KA": "Karnataka",
    "IN-KL": "Kerala", "IN-MP": "Madhya Pradesh", "IN-MH": "Maharashtra", "IN-OR": "Odisha",
    "IN-PB": "Punjab", "IN-RJ": "Rajasthan", "IN-TN": "Tamil Nadu", "IN-TS": "Telangana",
    "IN-UP": "Uttar Pradesh", "IN-UT": "Uttarakhand", "IN-WB": "West Bengal", "IN-SK": "Sikkim",
    # Every code in INDIAN_JURISDICTIONS needs a label: the picker falls back to
    # the raw code, and "IN-ML" is not something to show a citizen.
    "IN-AN": "Andaman and Nicobar Islands", "IN-CH": "Chandigarh",
    "IN-DH": "Dadra and Nagar Haveli and Daman and Diu", "IN-JK": "Jammu and Kashmir",
    "IN-LA": "Ladakh", "IN-LD": "Lakshadweep", "IN-ML": "Meghalaya", "IN-MN": "Manipur",
    "IN-MZ": "Mizoram", "IN-NL": "Nagaland", "IN-PY": "Puducherry", "IN-TR": "Tripura",
}

NEXT_QUESTION = {
    "police-fir": "Has the police station refused to register it yet?",
    "free-legal-aid": "Which district are you in?",
    "consumer-complaint": "What did you pay for, and roughly how much?",
    "cybercrime-report": "Was money already transferred, and have you called 1930?",
    "rti-request": "Which department holds the record you need?",
    "unpaid-wages": "How long did you work there, and what is still unpaid?",
    "domestic-violence-protection": "Are you somewhere safe right now?",
    "cheque-bounce": "When did the bank return the cheque, and do you have the memo?",
}

# The safety line is the one utterance that must never arrive in a language the
# caller does not speak. It is written for every language the speech models
# serve, and it is short enough to be the first thing heard.
SAFETY_SCRIPT = {
    "en-IN": "If you are in danger right now, get to a safe place and call 112. If it is safe to keep talking, tell me what is happening and where you are.",
    "hi-IN": "अगर आप अभी खतरे में हैं, तो सुरक्षित जगह जाएँ और 112 पर कॉल करें। अगर बोलना सुरक्षित है, तो बताइए कि क्या हो रहा है और आप कहाँ हैं।",
    "hi-en": "Agar aap abhi khatre mein hain, to safe jagah jaiye aur 112 par call kijiye. Agar bolna safe hai, to bataiye kya ho raha hai aur aap kahan hain.",
    "bn-IN": "আপনি যদি এখনই বিপদে থাকেন, নিরাপদ জায়গায় যান এবং ১১২ নম্বরে ফোন করুন। কথা বলা নিরাপদ হলে বলুন কী হচ্ছে এবং আপনি কোথায় আছেন।",
    "te-IN": "మీరు ఇప్పుడు ప్రమాదంలో ఉంటే, సురక్షితమైన చోటుకు వెళ్లి 112కు కాల్ చేయండి. మాట్లాడటం సురక్షితమైతే, ఏం జరుగుతోంది మరియు మీరు ఎక్కడ ఉన్నారో చెప్పండి.",
    "ta-IN": "நீங்கள் இப்போது ஆபத்தில் இருந்தால், பாதுகாப்பான இடத்திற்குச் சென்று 112-ஐ அழைக்கவும். பேசுவது பாதுகாப்பாக இருந்தால், என்ன நடக்கிறது, நீங்கள் எங்கே இருக்கிறீர்கள் என்று சொல்லுங்கள்.",
    "mr-IN": "तुम्ही आत्ता धोक्यात असाल तर सुरक्षित जागी जा आणि ११२ वर कॉल करा. बोलणे सुरक्षित असेल तर काय होत आहे आणि तुम्ही कुठे आहात ते सांगा.",
    "gu-IN": "તમે અત્યારે જોખમમાં હો તો સુરક્ષિત જગ્યાએ જાઓ અને ૧૧૨ પર કૉલ કરો. વાત કરવી સુરક્ષિત હોય તો શું થઈ રહ્યું છે અને તમે ક્યાં છો તે જણાવો.",
    "kn-IN": "ನೀವು ಈಗ ಅಪಾಯದಲ್ಲಿದ್ದರೆ, ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ಹೋಗಿ 112 ಗೆ ಕರೆ ಮಾಡಿ. ಮಾತನಾಡುವುದು ಸುರಕ್ಷಿತವಾದರೆ, ಏನಾಗುತ್ತಿದೆ ಮತ್ತು ನೀವು ಎಲ್ಲಿದ್ದೀರಿ ಎಂದು ಹೇಳಿ.",
    "ml-IN": "നിങ്ങൾ ഇപ്പോൾ അപകടത്തിലാണെങ്കിൽ, സുരക്ഷിതമായ സ്ഥലത്തേക്ക് പോയി 112 വിളിക്കുക. സംസാരിക്കാൻ സുരക്ഷിതമാണെങ്കിൽ, എന്താണ് സംഭവിക്കുന്നതെന്നും നിങ്ങൾ എവിടെയാണെന്നും പറയുക.",
    "pa-IN": "ਜੇ ਤੁਸੀਂ ਹੁਣ ਖ਼ਤਰੇ ਵਿੱਚ ਹੋ, ਤਾਂ ਸੁਰੱਖਿਅਤ ਥਾਂ 'ਤੇ ਜਾਓ ਅਤੇ 112 'ਤੇ ਕਾਲ ਕਰੋ। ਜੇ ਗੱਲ ਕਰਨਾ ਸੁਰੱਖਿਅਤ ਹੈ, ਤਾਂ ਦੱਸੋ ਕੀ ਹੋ ਰਿਹਾ ਹੈ ਅਤੇ ਤੁਸੀਂ ਕਿੱਥੇ ਹੋ।",
}

# The first sound a caller hears. It exists to remove dead air, so it is one
# short line in the caller's language, not an English placeholder.
ACK = {
    "en-IN": "Let me check the current sources for that.",
    "hi-IN": "मैं इसके लिए मौजूदा स्रोत देख रहा हूँ।",
    "hi-en": "Main iske liye current sources check kar raha hoon.",
    "bn-IN": "আমি এর জন্য বর্তমান সূত্রগুলো দেখছি।",
    "te-IN": "దీనికి ప్రస్తుత మూలాలు చూస్తున్నాను.",
    "ta-IN": "இதற்கான தற்போதைய ஆதாரங்களைப் பார்க்கிறேன்.",
    "mr-IN": "मी यासाठी सध्याचे स्रोत बघत आहे.",
    "gu-IN": "હું આ માટેના વર્તમાન સ્રોતો જોઈ રહ્યો છું.",
    "kn-IN": "ಇದಕ್ಕಾಗಿ ಪ್ರಸ್ತುತ ಮೂಲಗಳನ್ನು ನೋಡುತ್ತಿದ್ದೇನೆ.",
    "ml-IN": "ഇതിനായി നിലവിലുള്ള സ്രോതസ്സുകൾ നോക്കുകയാണ്.",
    "pa-IN": "ਮੈਂ ਇਸ ਲਈ ਮੌਜੂਦਾ ਸਰੋਤ ਵੇਖ ਰਿਹਾ ਹਾਂ।",
}


# The degraded paths used to speak English at a caller who had just been
# answered in Telugu, which is the one moment the app must not sound like a
# different product. Each sentence below is written for the language the turn
# resolved to; a language without an entry falls back rather than inventing.
FLOOR = {
    "en-IN": {
        "no_source_state": "I could not find an approved source that covers this, so I will not guess at a legal rule. Tell me the state you are in and what happened, and I will show you what is verifiable — or a procedure I can walk you through.",
        "no_source_more": "I could not find an approved source that covers this, so I will not guess at a legal rule. Tell me a little more about what happened, and I will show you what is verifiable — or a procedure I can walk you through.",
        "no_model_lead": "The model endpoint is not configured on this server, so I will not generate a legal conclusion. These are the closest approved passages I retrieved, quoted exactly:",
        "no_verified_claim_lead": "I could not verify an authoritative source that directly covers this situation, so I will not guess at a legal conclusion. These are the closest approved passages I retrieved, quoted exactly:",
        "case_only_lead": "I could not verify a statutory provision that settles this. The closest decision I have is this, and I am quoting it exactly:",
        "case_only_tail": "That is a quotation from the judgment, not advice, and I have not checked how later courts treated it.",
        "case_only_caveat": " I have not confirmed that it addresses your question, so read it as the nearest decision rather than an answer.",
        "no_model_tail": "They may not answer your question; coverage is incomplete and I have not interpreted them for your facts.",
    },
    "hi-IN": {
        "no_source_state": "मुझे इस विषय पर कोई स्वीकृत स्रोत नहीं मिला, इसलिए मैं कानूनी नियम का अनुमान नहीं लगाऊँगा। आप किस राज्य में हैं और क्या हुआ, यह बताइए — मैं दिखाऊँगा कि क्या सत्यापित किया जा सकता है, या कोई प्रक्रिया जिसमें मैं आपको चरण-दर-चरण ले चलूँ।",
        "no_source_more": "मुझे इस विषय पर कोई स्वीकृत स्रोत नहीं मिला, इसलिए मैं कानूनी नियम का अनुमान नहीं लगाऊँगा। थोड़ा और बताइए कि क्या हुआ — मैं दिखाऊँगा कि क्या सत्यापित किया जा सकता है, या कोई प्रक्रिया जिसमें मैं आपको चरण-दर-चरण ले चलूँ।",
        "no_model_lead": "इस सर्वर पर भाषा मॉडल जुड़ा नहीं है, इसलिए मैं कानूनी निष्कर्ष नहीं दूँगा। ये सबसे निकटतम स्वीकृत अंश हैं, जैसे हैं वैसे उद्धृत:",
        "no_verified_claim_lead": "मैं इस प्रश्न के लिए स्वीकृत स्रोतों से किसी विशिष्ट कानूनी निष्कर्ष का अनुमान नहीं लगाऊँगा। ये सबसे निकटतम स्वीकृत अंश हैं, जैसे हैं वैसे उद्धृत:",
        "case_only_lead": "मुझे इस पर कोई स्वीकृत धारा नहीं मिली जो इसे तय करे। जो सबसे निकटतम निर्णय मेरे पास है, वह यह है, और मैं उसे जैसा है वैसा उद्धृत कर रहा हूँ:",
        "case_only_tail": "यह फैसले से लिया गया उद्धरण है, कानूनी सलाह नहीं; बाद की अदालतों ने इसे कैसे लिया, यह मैंने जाँचा नहीं है।",
        "case_only_caveat": " मैंने यह पुष्टि नहीं की कि यह आपके प्रश्न को संबोधित करता है, इसलिए इसे उत्तर नहीं, सबसे निकटतम निर्णय समझिए।",
        "no_model_tail": "ये आपके प्रश्न का उत्तर न दें; कवरेज अधूरा है और मैंने इन्हें आपके तथ्यों पर लागू करके नहीं पढ़ा है।",
    },
    "hi-en": {
        "no_source_state": "Mujhe is vishay par koi approved source nahi mila, isliye main kanoon ka andaza nahi lagaunga. Aap kis state mein hain aur kya hua, bataiye — main dikhaunga kya verify ho sakta hai, ya koi procedure jisme main step-by-step le chalu.",
        "no_source_more": "Mujhe is vishay par koi approved source nahi mila, isliye main kanoon ka andaza nahi lagaunga. Thoda aur bataiye kya hua — main dikhaunga kya verify ho sakta hai, ya koi procedure jisme main step-by-step le chalu.",
        "no_model_lead": "Is server par language model connected nahi hai, isliye main legal conclusion nahi dunga. Ye sabse kareeb approved passages hain, exactly quote kiye gaye:",
        "no_verified_claim_lead": "Main is vishay par approved sources se bina verification ke kanoon ka andaza nahi lagaunga. Ye sabse kareeb approved passages hain, exactly quote kiye gaye:",
        "case_only_lead": "Mujhe is par koi approved section nahi mila jo ise tay kare. Jo sabse kareeb decision mere paas hai wahi ye hai, aur main use exactly quote kar raha hoon:",
        "case_only_tail": "Ye faisle se liya gaya quote hai, legal advice nahi; baad ki adalaton ne ise kaise liya, ye maine check nahi kiya.",
        "case_only_caveat": " Maine confirm nahi kiya ki ye aapke sawaal ko address karta hai, isliye ise jawab nahi, sabse kareeb faisla samajhiye.",
        "no_model_tail": "Ye aapke sawaal ka jawab na dein; coverage adhura hai aur maine inhe aapke facts par laga kar nahi padha.",
    },
    "bn-IN": {
        "no_source_state": "এই বিষয়ে অনুমোদিত কোনো সূত্র পাইনি, তাই আমি আইনের নিয়ম অনুমান করব না। আপনি কোন রাজ্যে আছেন এবং কী হয়েছে বলুন — কী যাচাই করা যায়, বা কোন প্রক্রিয়া ধাপে ধাপে দেখাতে পারি তা দেখাব।",
        "no_source_more": "এই বিষয়ে অনুমোদিত কোনো সূত্র পাইনি, তাই আমি আইনের নিয়ম অনুমান করব না। কী হয়েছে একটু বেশি বলুন — কী যাচাই করা যায়, বা কোন প্রক্রিয়া ধাপে ধাপে দেখাতে পারি তা দেখাব।",
        "no_model_lead": "এই সার্ভারে কোনো ভাষা মডেল সংযুক্ত নেই, তাই আমি আইনি সিদ্ধান্ত দেব না। এগুলো আমার পাওয়া নিকটতম অনুমোদিত অংশ, হুবহু উদ্ধৃত:",
        "no_verified_claim_lead": "এই পরিস্থিতি সরাসরি আচ্ছাদন করে এমন কোনো অনুমোদিত সূত্র আমি যাচাই করতে পারিনি, তাই আমি আইনি সিদ্ধান্ত অনুমান করব না। এগুলো নিকটতম অনুমোদিত অংশ, হুবহু উদ্ধৃত:",
        "case_only_lead": "এই প্রশ্নটি নিষ্পত্তি করে এমন কোনো অনুমোদিত বিধান আমি যাচাই করতে পারিনি। আমার কাছে থাকা সবচেয়ে কাছের রায়টি এই, এবং আমি এটি হুবহু উদ্ধৃত করছি:",
        "case_only_tail": "এটি রায় থেকে নেওয়া উদ্ধৃতি, আইনি পরামর্শ নয়; পরবর্তী আদালতগুলি এটিকে কীভাবে দেখেছে তা আমি যাচাই করিনি।",
        "case_only_caveat": " এটি আপনার প্রশ্নের উত্তর দেয় কি না তা আমি নিশ্চিত করিনি, তাই এটি উত্তর নয়, নিকটতম রায় হিসেবে পড়ুন।",
        "no_model_tail": "এগুলো আপনার প্রশ্নের উত্তর নাও দিতে পারে; কভারেজ অসম্পূর্ণ এবং আপনার তথ্যের সঙ্গে মিলিয়ে আমি এগুলো পড়িনি।",
    },
    "te-IN": {
        "no_source_state": "ఈ విషయంపై ఆమోదించిన మూలం నాకు దొరకలేదు, అందుకే నేను చట్ట నియమాన్ని ఊహించను. మీరు ఏ రాష్ట్రంలో ఉన్నారు, ఏం జరిగిందో చెప్పండి — ఏది నిర్ధారించగలమో, లేదా నేను దశలవారీగా చెప్పగల ప్రక్రియను చూపిస్తాను.",
        "no_source_more": "ఈ విషయంపై ఆమోదించిన మూలం నాకు దొరకలేదు, అందుకే నేను చట్ట నియమాన్ని ఊహించను. ఏం జరిగిందో కొంచెం ఎక్కువ చెప్పండి — ఏది నిర్ధారించగలమో, లేదా నేను దశలవారీగా చెప్పగల ప్రక్రియను చూపిస్తాను.",
        "no_model_lead": "ఈ సర్వర్‌లో భాషా మోడల్ కనెక్ట్ కాలేదు, అందుకే నేను చట్టపరమైన నిర్ణయం చెప్పను. ఇవి నాకు దొరకిన దగ్గరి ఆమోదిత భాగాలు, యథాతథంగా:",
        "no_verified_claim_lead": "ఈ పరిస్థితిని నేరుగా కవర్ చేసే ఆమోదిత మూలాన్ని నేను నిర్ధారించలేకపోయాను, అందుకే చట్టపరమైన నిర్ణయాన్ని ఊహించను. ఇవి నాకు దొరకిన దగ్గరి ఆమోదిత భాగాలు, యథాతథంగా:",
        "case_only_lead": "ఈ ప్రశ్నను తీర్పు చేసే ఆమోదిత శాసన భాగం నాకు నిర్ధారణ కాలేదు. నా దగ్గర ఉన్న అతి దగ్గరి తీర్పు ఇది, దాన్ని యథాతథంగా ఉద్ధరిస్తున్నాను:",
        "case_only_tail": "ఇది తీర్పు నుండి తీసుకున్న ఉద్ధరణ, న్యాయ సలహా కాదు; తర్వాతి కోర్టులు దాన్ని ఎలా పరిగణించాయో నేను పరిశీలించలేదు.",
        "case_only_caveat": " ఇది మీ ప్రశ్నకు సమాధానం ఇస్తుందో లేదో నేను నిర్ధారించలేదు, కాబట్టి దీన్ని సమాధానంగా కాకుండా అతి దగ్గరి తీర్పుగా చదవండి.",
        "no_model_tail": "ఇవి మీ ప్రశ్నకు సమాధానం ఇవ్వకపోవచ్చు; కవరేజ్ అసంపూర్ణం, మీ వాస్తవాలకు వర్తించి నేను వీటిని చదవలేదు.",
    },
    "ta-IN": {
        "no_source_state": "இந்த விஷயத்திற்கு அங்கீகரிக்கப்பட்ட ஆதாரம் கிடைக்கவில்லை, எனவே சட்ட விதியை ஊகிக்க மாட்டேன். நீங்கள் எந்த மாநிலத்தில் இருக்கிறீர்கள், என்ன நடந்தது என்று சொல்லுங்கள் — எதை உறுதிப்படுத்த முடியும், அல்லது நான் படிப்படியாக வழிகாட்டக்கூடிய ஒரு நடைமுறையைக் காட்டுகிறேன்.",
        "no_source_more": "இந்த விஷயத்திற்கு அங்கீகரிக்கப்பட்ட ஆதாரம் கிடைக்கவில்லை, எனவே சட்ட விதியை ஊகிக்க மாட்டேன். என்ன நடந்தது என்று சற்று கூடுதலாகச் சொல்லுங்கள் — எதை உறுதிப்படுத்த முடியும், அல்லது ஒரு நடைமுறையைக் காட்டுகிறேன்.",
        "no_model_lead": "இந்த சேவையகத்தில் மொழி மாதிரி இணைக்கப்படவில்லை, எனவே சட்ட முடிவு சொல்ல மாட்டேன். இவை நான் பெற்ற அருகிலுள்ள அங்கீகரிக்கப்பட்ட பகுதிகள், அப்படியே மேற்கோள்:",
        "no_verified_claim_lead": "இந்தச் சூழலை நேரடியாக ஆதரிக்கும் அங்கீகரிக்கப்பட்ட ஆதாரத்தை என்னால் உறுதிப்படுத்த முடியவில்லை, எனவே சட்ட முடிவை ஊகிக்க மாட்டேன். இவை நான் பெற்ற அருகிலுள்ள அங்கீகரிக்கப்பட்ட பகுதிகள், அப்படியே மேற்கோள்:",
        "case_only_lead": "இந்தக் கேள்வியைத் தீர்க்கும் அங்கீகரிக்கப்பட்ட சட்டப் பிரிவை என்னால் உறுதிப்படுத்த முடியவில்லை. என்னிடம் உள்ள மிக நெருக்கமான தீர்ப்பு இது, அதை அப்படியே மேற்கோள் காட்டுகிறேன்:",
        "case_only_tail": "இது தீர்ப்பிலிருந்து எடுக்கப்பட்ட மேற்கோள், சட்ட ஆலோசனை அல்ல; பின்னர் வந்த நீதிமன்றங்கள் இதை எப்படிக் கருதின என்பதை நான் சரிபார்க்கவில்லை.",
        "case_only_caveat": " இது உங்கள் கேள்விக்குப் பதில் அளிக்கிறதா என்பதை நான் உறுதி செய்யவில்லை, எனவே இதைப் பதிலாக அல்ல, மிக நெருக்கமான தீர்ப்பாகப் படியுங்கள்.",
        "no_model_tail": "இவை உங்கள் கேள்விக்கு பதில் அளிக்காமல் இருக்கலாம்; பரப்பு முழுமையில்லை, உங்கள் உண்மைகளுக்குப் பொருத்தி நான் இவற்றைப் படிக்கவில்லை.",
    },
    "mr-IN": {
        "no_source_state": "या विषयावर मंजूर स्रोत सापडला नाही, म्हणून मी कायद्याचा अंदाज लावणार नाही. तुम्ही कोणत्या राज्यात आहात आणि काय झाले ते सांगा — काय पडताळता येईल, किंवा मी टप्प्याटप्प्याने सांगू शकतो अशी प्रक्रिया दाखवतो.",
        "no_source_more": "या विषयावर मंजूर स्रोत सापडला नाही, म्हणून मी कायद्याचा अंदाज लावणार नाही. काय झाले थोडे अधिक सांगा — काय पडताळता येईल, किंवा मी टप्प्याटप्प्याने सांगू शकतो अशी प्रक्रिया दाखवतो.",
        "no_model_lead": "या सर्व्हरवर भाषा मॉडेल जोडलेले नाही, म्हणून मी कायदेशीर निष्कर्ष देणार नाही. हे मला मिळालेले जवळचे मंजूर उतारे आहेत, तसेच उद्धृत:",
        "no_verified_claim_lead": "हे परिस्थिती थेट हाताळणारा मंजूर स्रोत मला पडताळता आला नाही, म्हणून मी कायदेशीर निष्कर्षाचा अंदाज लावणार नाही. हे मला मिळालेले जवळचे मंजूर उतारे आहेत, तसेच उद्धृत:",
        "case_only_lead": "हा प्रश्न निकाली काढणारी मंजूर कायदेशीर तरतूद मला पडताळता आली नाही. माझ्याकडे असलेला सर्वात जवळचा निर्णय हा आहे, आणि मी तो जसाच्या तसा उद्धृत करत आहे:",
        "case_only_tail": "हे निर्णयातील उद्धरण आहे, कायदेशीर सल्ला नाही; नंतरच्या न्यायालयांनी त्याकडे कसे पाहिले हे मी तपासलेले नाही.",
        "case_only_caveat": " हा निर्णय तुमच्या प्रश्नाला उत्तर देतो का हे मी तपासलेले नाही, म्हणून हे उत्तर नाही, सर्वात जवळचा निर्णय असे वाचा.",
        "no_model_tail": "हे तुमच्या प्रश्नाचे उत्तर देतीलच असे नाही; व्याप्ती अपूर्ण आहे आणि तुमच्या तथ्यांना लावून मी हे वाचलेले नाही.",
    },
    "gu-IN": {
        "no_source_state": "આ વિષય પર મંજૂર સ્રોત મળ્યો નથી, તેથી હું કાયદાનો અંદાજ લગાવીશ નહીં. તમે કયા રાજ્યમાં છો અને શું થયું તે જણાવો — શું ચકાસી શકાય, અથવા હું પગલે પગલે કહી શકું તેવી પ્રક્રિયા બતાવીશ.",
        "no_source_more": "આ વિષય પર મંજૂર સ્રોત મળ્યો નથી, તેથી હું કાયદાનો અંદાજ લગાવીશ નહીં. શું થયું તે થોડું વધુ જણાવો — શું ચકાસી શકાય, અથવા પ્રક્રિયા બતાવીશ.",
        "no_model_lead": "આ સર્વર પર ભાષા મોડેલ જોડાયેલું નથી, તેથી હું કાનૂની નિષ્કર્ષ આપીશ નહીં. આ મને મળેલા નજીકના મંજૂર ફકરા છે, જેમ છે તેમ ટાંકેલા:",
        "no_verified_claim_lead": "આ પરિસ્થિતિને સીધું આવરી લેતો મંજૂર સ્રોત હું ચકાસી શક્યો નથી, તેથી હું કાનૂની નિષ્કર્ષનો અંદાજ લગાવીશ નહીં. આ મને મળેલા નજીકના મંજૂર ફકરા છે, જેમ છે તેમ ટાંકેલા:",
        "case_only_lead": "આ પ્રશ્ન નક્કી કરતી મંજૂર કાયદેસર જોગવાઈ મને ચકાસી શકાઈ નથી. મારી પાસેનો સૌથી નજીકનો નિર્ણય આ છે, અને હું તેને જેમ છે તેમ ટાંકું છું:",
        "case_only_tail": "આ ચુકાદામાંથી લીધેલો ઉતારો છે, કાનૂની સલાહ નથી; પાછળથી આવેલી અદાલતોએ તેને કેવી રીતે ગણ્યું તે મેં ચકાસ્યું નથી.",
        "case_only_caveat": " આ તમારા પ્રશ્નનો જવાબ આપે છે કે નહીં તે મેં ચકાસ્યું નથી, તેથી આને જવાબ નહીં, સૌથી નજીકનો નિર્ણય ગણો.",
        "no_model_tail": "આ તમારા પ્રશ્નનો જવાબ ન પણ આપે; આવરી લેવાનું અધૂરું છે અને તમારા તથ્યો પર લગાવીને મેં આ વાંચ્યા નથી.",
    },
    "kn-IN": {
        "no_source_state": "ಈ ವಿಷಯಕ್ಕೆ ಅನುಮೋದಿತ ಮೂಲ ಸಿಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ನಾನು ಕಾನೂನು ನಿಯಮವನ್ನು ಊಹಿಸುವುದಿಲ್ಲ. ನೀವು ಯಾವ ರಾಜ್ಯದಲ್ಲಿದ್ದೀರಿ ಮತ್ತು ಏನಾಯಿತು ಎಂದು ಹೇಳಿ — ಏನು ಪರಿಶೀಲಿಸಬಹುದು, ಅಥವಾ ನಾನು ಹಂತ ಹಂತವಾಗಿ ಹೇಳಬಹುದಾದ ಕ್ರಮವನ್ನು ತೋರಿಸುತ್ತೇನೆ.",
        "no_source_more": "ಈ ವಿಷಯಕ್ಕೆ ಅನುಮೋದಿತ ಮೂಲ ಸಿಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ನಾನು ಕಾನೂನು ನಿಯಮವನ್ನು ಊಹಿಸುವುದಿಲ್ಲ. ಏನಾಯಿತು ಎಂದು ಸ್ವಲ್ಪ ಹೆಚ್ಚು ಹೇಳಿ — ಏನು ಪರಿಶೀಲಿಸಬಹುದು, ಅಥವಾ ಕ್ರಮವನ್ನು ತೋರಿಸುತ್ತೇನೆ.",
        "no_model_lead": "ಈ ಸರ್ವರ್‌ನಲ್ಲಿ ಭಾಷಾ ಮಾದರಿ ಸಂಪರ್ಕಗೊಂಡಿಲ್ಲ, ಆದ್ದರಿಂದ ನಾನು ಕಾನೂನು ತೀರ್ಮಾನ ನೀಡುವುದಿಲ್ಲ. ಇವು ನನಗೆ ಸಿಕ್ಕ ಹತ್ತಿರದ ಅನುಮೋದಿತ ಭಾಗಗಳು, ಹಾಗೆಯೇ ಉಲ್ಲೇಖಿಸಲಾಗಿದೆ:",
        "no_verified_claim_lead": "ಈ ಪರಿಸ್ಥಿತಿಯನ್ನು ನೇರವಾಗಿ ಒಳಗೊಳ್ಳುವ ಅನುಮೋದಿತ ಮೂಲವನ್ನು ನಾನು ಪರಿಶೀಲಿಸಲಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಕಾನೂನು ತೀರ್ಮಾನವನ್ನು ಊಹಿಸುವುದಿಲ್ಲ. ಇವು ನನಗೆ ಸಿಕ್ಕ ಹತ್ತಿರದ ಅನುಮೋದಿತ ಭಾಗಗಳು, ಹಾಗೆಯೇ ಉಲ್ಲೇಖಿಸಲಾಗಿದೆ:",
        "case_only_lead": "ಈ ಪ್ರಶ್ನೆಯನ್ನು ನಿರ್ಣಯಿಸುವ ಅನುಮೋದಿತ ಕಾನೂನು ನಿಬಂಧನೆಯನ್ನು ನಾನು ಪರಿಶೀಲಿಸಲಾಗಲಿಲ್ಲ. ನನ್ನ ಬಳಿ ಇರುವ ಅತ್ಯಂತ ಹತ್ತಿರದ ತೀರ್ಪು ಇದು, ಅದನ್ನು ಹಾಗೆಯೇ ಉಲ್ಲೇಖಿಸುತ್ತಿದ್ದೇನೆ:",
        "case_only_tail": "ಇದು ತೀರ್ಪಿನಿಂದ ತೆಗೆದ ಉಲ್ಲೇಖ, ಕಾನೂನು ಸಲಹೆ ಅಲ್ಲ; ನಂತರದ ನ್ಯಾಯಾಲಯಗಳು ಇದನ್ನು ಹೇಗೆ ಪರಿಗಣಿಸಿವೆ ಎಂದು ನಾನು ಪರಿಶೀಲಿಸಿಲ್ಲ.",
        "case_only_caveat": " ಇದು ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡುತ್ತದೆಯೇ ಎಂದು ನಾನು ಖಚಿತಪಡಿಸಿಲ್ಲ, ಆದ್ದರಿಂದ ಇದನ್ನು ಉತ್ತರವಲ್ಲ, ಹತ್ತಿರದ ತೀರ್ಪು ಎಂದು ಓದಿ.",
        "no_model_tail": "ಇವು ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡದಿರಬಹುದು; ವ್ಯಾಪ್ತಿ ಅಪೂರ್ಣ ಮತ್ತು ನಿಮ್ಮ ಸಂಗತಿಗಳಿಗೆ ಅನ್ವಯಿಸಿ ನಾನು ಇವನ್ನು ಓದಿಲ್ಲ.",
    },
    "ml-IN": {
        "no_source_state": "ഈ വിഷയത്തിന് അംഗീകൃത സ്രോതസ്സ് കണ്ടെത്തിയില്ല, അതിനാൽ ഞാൻ നിയമം ഊഹിക്കുന്നില്ല. നിങ്ങൾ ഏത് സംസ്ഥാനത്താണ്, എന്താണ് സംഭവിച്ചത് എന്ന് പറയുക — എന്ത് സ്ഥിരീകരിക്കാം, അല്ലെങ്കിൽ ഞാൻ ഘട്ടം ഘട്ടമായി പറയാവുന്ന ഒരു നടപടിക്രമം കാണിക്കാം.",
        "no_source_more": "ഈ വിഷയത്തിന് അംഗീകൃത സ്രോതസ്സ് കണ്ടെത്തിയില്ല, അതിനാൽ ഞാൻ നിയമം ഊഹിക്കുന്നില്ല. എന്താണ് സംഭവിച്ചത് അൽപ്പം കൂടി പറയുക — എന്ത് സ്ഥിരീകരിക്കാം എന്ന് കാണിക്കാം.",
        "no_model_lead": "ഈ സെർവറിൽ ഭാഷാ മോഡൽ ബന്ധിപ്പിച്ചിട്ടില്ല, അതിനാൽ ഞാൻ നിയമപരമായ നിഗമനം പറയില്ല. ഇവയാണ് എനിക്ക് ലഭിച്ച അടുത്തുള്ള അംഗീകൃത ഭാഗങ്ങൾ, അതേപടി ഉദ്ധരിച്ചത്:",
        "no_verified_claim_lead": "ഈ സാഹചര്യം നേരിട്ട് ഉൾക്കൊള്ളുന്ന അംഗീകൃത സ്രോതസ്സ് എനിക്ക് സ്ഥിരീകരിക്കാനായില്ല, അതിനാൽ ഞാൻ നിയമപരമായ നിഗമനം ഊഹിക്കുന്നില്ല. ഇവയാണ് എനിക്ക് ലഭിച്ച അടുത്തുള്ള അംഗീകൃത ഭാഗങ്ങൾ, അതേപടി ഉദ്ധരിച്ചത്:",
        "case_only_lead": "ഈ ചോദ്യം തീർപ്പാക്കുന്ന അംഗീകൃത നിയമവ്യവസ്ഥ എനിക്ക് സ്ഥിരീകരിക്കാനായില്ല. എന്റെ പക്കലുള്ള ഏറ്റവും അടുത്ത വിധി ഇതാണ്, അത് അതേപടി ഉദ്ധരിക്കുന്നു:",
        "case_only_tail": "ഇത് വിധിയിൽ നിന്നുള്ള ഉദ്ധരണിയാണ്, നിയമോപദേശമല്ല; പിന്നീടുള്ള കോടതികൾ ഇതിനെ എങ്ങനെ കണ്ടു എന്ന് ഞാൻ പരിശോധിച്ചിട്ടില്ല.",
        "case_only_caveat": " ഇത് നിങ്ങളുടെ ചോദ്യത്തിന് ഉത്തരം നൽകുന്നുണ്ടോ എന്ന് ഞാൻ ഉറപ്പാക്കിയിട്ടില്ല, അതിനാൽ ഇത് ഉത്തരമല്ല, ഏറ്റവും അടുത്ത വിധിയായി വായിക്കുക.",
        "no_model_tail": "ഇവ നിങ്ങളുടെ ചോദ്യത്തിന് ഉത്തരം നൽകണമെന്നില്ല; കവറേജ് അപൂർണമാണ്, നിങ്ങളുടെ വസ്തുതകളിൽ പ്രയോഗിച്ച് ഞാൻ ഇവ വായിച്ചിട്ടില്ല.",
    },
    "pa-IN": {
        "no_source_state": "ਇਸ ਵਿਸ਼ੇ ਲਈ ਕੋਈ ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਸਰੋਤ ਨਹੀਂ ਮਿਲਿਆ, ਇਸ ਲਈ ਮੈਂ ਕਾਨੂੰਨ ਦਾ ਅੰਦਾਜ਼ਾ ਨਹੀਂ ਲਾਵਾਂਗਾ। ਤੁਸੀਂ ਕਿਸ ਰਾਜ ਵਿੱਚ ਹੋ ਅਤੇ ਕੀ ਹੋਇਆ ਦੱਸੋ — ਮੈਂ ਦਿਖਾਵਾਂਗਾ ਕਿ ਕੀ ਤਸਦੀਕ ਹੋ ਸਕਦਾ ਹੈ, ਜਾਂ ਕੋਈ ਪ੍ਰਕਿਰਿਆ ਜਿਸ ਵਿੱਚ ਮੈਂ ਪੜਾਅ-ਦਰ-ਪੜਾਅ ਲੈ ਜਾ ਸਕਦਾ ਹਾਂ।",
        "no_source_more": "ਇਸ ਵਿਸ਼ੇ ਲਈ ਕੋਈ ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਸਰੋਤ ਨਹੀਂ ਮਿਲਿਆ, ਇਸ ਲਈ ਮੈਂ ਕਾਨੂੰਨ ਦਾ ਅੰਦਾਜ਼ਾ ਨਹੀਂ ਲਾਵਾਂਗਾ। ਕੀ ਹੋਇਆ ਥੋੜ੍ਹਾ ਹੋਰ ਦੱਸੋ — ਮੈਂ ਦਿਖਾਵਾਂਗਾ ਕਿ ਕੀ ਤਸਦੀਕ ਹੋ ਸਕਦਾ ਹੈ।",
        "no_model_lead": "ਇਸ ਸਰਵਰ 'ਤੇ ਕੋਈ ਭਾਸ਼ਾ ਮਾਡਲ ਜੁੜਿਆ ਨਹੀਂ ਹੈ, ਇਸ ਲਈ ਮੈਂ ਕਾਨੂੰਨੀ ਸਿੱਟਾ ਨਹੀਂ ਦਿਆਂਗਾ। ਇਹ ਮੈਨੂੰ ਮਿਲੇ ਸਭ ਤੋਂ ਨੇੜਲੇ ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਹਿੱਸੇ ਹਨ, ਜਿਵੇਂ ਹਨ ਤਿਵੇਂ ਹਵਾਲਾ:",
        "no_verified_claim_lead": "ਇਸ ਹਾਲਤ ਨੂੰ ਸਿੱਧਾ ਕਵਰ ਕਰਨ ਵਾਲਾ ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਸਰੋਤ ਮੈਂ ਤਸਦੀਕ ਨਹੀਂ ਕਰ ਸਕਿਆ, ਇਸ ਲਈ ਮੈਂ ਕਾਨੂੰਨੀ ਸਿੱਟੇ ਦਾ ਅੰਦਾਜ਼ਾ ਨਹੀਂ ਲਾਵਾਂਗਾ। ਇਹ ਮੈਨੂੰ ਮਿਲੇ ਸਭ ਤੋਂ ਨੇੜਲੇ ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਹਿੱਸੇ ਹਨ, ਜਿਵੇਂ ਹਨ ਤਿਵੇਂ ਹਵਾਲਾ:",
        "case_only_lead": "ਇਸ ਸਵਾਲ ਨੂੰ ਤੈਅ ਕਰਨ ਵਾਲੀ ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਕਾਨੂੰਨੀ ਧਾਰਾ ਮੈਂ ਤਸਦੀਕ ਨਹੀਂ ਕਰ ਸਕਿਆ। ਮੇਰੇ ਕੋਲ ਸਭ ਤੋਂ ਨੇੜਲਾ ਫੈਸਲਾ ਇਹ ਹੈ, ਅਤੇ ਮੈਂ ਇਸਨੂੰ ਜਿਵੇਂ ਹੈ ਤਿਵੇਂ ਹਵਾਲਾ ਦੇ ਰਿਹਾ ਹਾਂ:",
        "case_only_tail": "ਇਹ ਫੈਸਲੇ ਵਿੱਚੋਂ ਲਿਆ ਹਵਾਲਾ ਹੈ, ਕਾਨੂੰਨੀ ਸਲਾਹ ਨਹੀਂ; ਬਾਅਦ ਦੀਆਂ ਅਦਾਲਤਾਂ ਨੇ ਇਸਨੂੰ ਕਿਵੇਂ ਲਿਆ, ਇਹ ਮੈਂ ਨਹੀਂ ਵੇਖਿਆ।",
        "case_only_caveat": " ਮੈਂ ਪੱਕਾ ਨਹੀਂ ਕੀਤਾ ਕਿ ਇਹ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦਿੰਦਾ ਹੈ, ਇਸ ਲਈ ਇਸਨੂੰ ਜਵਾਬ ਨਹੀਂ, ਸਭ ਤੋਂ ਨੇੜਲਾ ਫੈਸਲਾ ਸਮਝੋ।",
        "no_model_tail": "ਇਹ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਨਾ ਵੀ ਦੇਣ; ਕਵਰੇਜ ਅਧੂਰੀ ਹੈ ਅਤੇ ਤੁਹਾਡੇ ਤੱਥਾਂ 'ਤੇ ਲਾ ਕੇ ਮੈਂ ਇਹ ਨਹੀਂ ਪੜ੍ਹੇ।",
    },
}


def _floor(language: str, key: str) -> str:
    """One degraded-path sentence, in the caller's language when it exists."""
    return FLOOR.get(language, FLOOR["en-IN"]).get(key) or FLOOR["en-IN"][key]


def _text(value: Any, limit: int = 5000) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _normalize_language(code: Any, preferred: str) -> str:
    """Map a language the generator returned onto a code this build speaks.

    `preferred` is the language the turn was already resolved to, so an
    unparseable model answer cannot silently switch the reply to English.
    """
    resolved = language_layer.normalize(code)
    if resolved not in {language_layer.AUTO, language_layer.PRISMA_AUTO}:
        return resolved
    fallback = language_layer.normalize(preferred)
    return fallback if fallback not in {language_layer.AUTO, language_layer.PRISMA_AUTO} else "auto"


def speakable(text: str, limit: int = 340) -> str:
    """Flatten a reply into something that sounds human when spoken.

    Voice output must not read out markdown, bullets or bracketed metadata,
    and a caller should hear the first useful thing quickly. The full text
    stays available on screen.
    """
    cleaned = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"[*_#>`~]+", "", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if boundary > limit * 0.4:
        return cut[: boundary + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + "…"


# Speech that may start before the turn is validated. Anything resembling a
# legal proposition waits for claim validation; a greeting, an acknowledgement
# or a question does not need to. This is the safety cost of the latency win,
# and it is deliberately biased towards silence.
EARLY_SPEECH_BLOCK = re.compile(
    r"\d|\b(section|article|act|code|ipc|crpc|bnss|bns|schedule|penalty|fine|imprison\w*|"
    r"liable|entitled|rights?|deadline|compensation|damages|court|tribunal|notice period)\b|https?://",
    re.I,
)


def _early_speak_ok(sentence: str) -> bool:
    """Whether a sentence can be spoken before the turn finishes validating."""
    text = " ".join(sentence.split())
    if not text or len(text) > 240:
        return False
    return not EARLY_SPEECH_BLOCK.search(text)


def _focus_terms(text: str, jurisdiction: str) -> tuple[str, ...]:
    """Reviewed bench vocabulary for the issue this message is about.

    A caller asking about a refused FIR will not say "cognizable offence", and
    no judgment says "refused to register an FIR": the two vocabularies have to
    be joined somewhere explicit. The procedure card's `case_terms` are that
    bridge, reviewed by a human, and `best_match` is pure Python over twelve
    records so it costs microseconds inside a thread already doing I/O.
    """
    card = procedural.best_match(text, jurisdiction)
    return tuple(
        str(term) for term in (card or {}).get("case_terms", ()) if str(term).strip()
    )


# Words a follow-up uses to *point at* the matter instead of describing it.
# "What can I do about the money?" carries one content word, and that word is
# the caller's shorthand for something they already told the call. Searching
# it literally answers a question about money in the abstract.
POINTER_TERMS = frozenset({
    "money", "case", "matter", "problem", "issue", "thing", "things", "help",
    "next", "now", "then", "also", "same", "there", "situation", "option",
    "options", "step", "steps", "process", "law", "legal",
    "kya", "karu", "karun", "kare", "karein", "aage", "ab", "batao", "matlab",
    "phir", "aur", "hai", "kaise", "ena", "pannu", "cheyyu", "yenu", "madbeku",
})


def _carried_matter(
    stored_state: Any,
    ledger: facts.Ledger,
    text: str,
    history: list[dict[str, Any]] | None = None,
    earlier: str = "",
) -> tuple[str, tuple[str, ...] | None]:
    """The matter this call is already about, when the turn only points at it.

    A follow-up is not a question in its own right. "What can I do about the
    money?" has one content word, and that word is the caller's shorthand for
    something they already described; searched literally it answers a question
    about money in the abstract, which is how an on-point authority turned into
    a bank judgment. So the thread is what gets searched: the caller's own
    recent sentences, taken from the transcript rather than summarised, then
    the ledger's description slots, then the intake's name.

    A turn that brings its own subject -- "my landlord changed the lock" -- is
    left alone, because there the words *are* the evidence, and appending an
    older matter would pull the ranking back to it.

    Returns `(query, focus)`: `query` is empty when nothing is being carried,
    `focus` is the reviewed bench vocabulary (`case_terms`) of the procedure the
    call is in, so the relevance gate measures the right matter even though the
    caller's four words say nothing about it.
    """
    substantive = [
        term for term in retrieval.content_terms(text)
        if term.lower() not in POINTER_TERMS
    ]
    # Two words cannot be a question on their own: the relevance gate needs two
    # matching terms, so "keep deposit" alone asks the store to find judgments
    # containing both words and gets nothing. Three substantive terms is the
    # same floor that already decides whether a turn is a real question.
    if len(substantive) >= 3:
        return "", None

    procedure_id = ""
    if isinstance(stored_state, dict):
        procedure_id = str(stored_state.get("procedure_id") or "")
        # The workflow is the fallback, not the override. An intake left open
        # after the caller moved on stays "active" in storage, and letting it
        # win measured the new matter against the old matter's vocabulary --
        # a deposit question inside an eviction dispute was scored as an
        # unpaid-wages question.
        if not procedure_id:
            workflow_state = stored_state.get("workflow")
            if isinstance(workflow_state, dict) and workflow_state.get("status") == "active":
                procedure_id = str(workflow_state.get("id") or "")

    # What the caller actually said, newest last. Two messages is enough thread
    # to disambiguate a pointer and short enough not to bury it in one topic:
    # "my employer stopped paying" then "he says I resigned" is one matter.
    # Accumulate up to 3 recent substantive user turns. One message is not
    # enough when the caller narrates across multiple turns: "my employer did
    # not pay" then "I resigned two months ago" then "what can I do?" needs
    # all three to know this is a wages-after-resignation question. Three is
    # the ceiling — beyond that the query grows stale matter that hurts more
    # than it helps.
    said: list[str] = []
    for item in reversed(list(history or [])):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = " ".join(str(item.get("content") or item.get("text") or "").split())
        if not content:
            continue
        if len([
            term for term in retrieval.content_terms(content)
            if term.lower() not in POINTER_TERMS
        ]) < 2:
            continue
        said.append(content)
        if len(said) >= 3:
            break
    said.reverse()

    # When the live window has nothing (all turns were pointer-only or the
    # conversation is short), scan the compacted earlier digest for the first
    # substantive line. This is what keeps turn 15 retrieving on the same
    # matter as turn 2 even after the live window has rolled past it.
    if not said and earlier:
        for line in earlier.splitlines():
            # Earlier lines are prefixed "caller: …" or "assistant: …".
            if not line.startswith("caller:"):
                continue
            content = line[len("caller:"):].strip()
            if len([
                term for term in retrieval.content_terms(content)
                if term.lower() not in POINTER_TERMS
            ]) >= 2:
                said.append(content)
                break

    # What they just said beats what the ledger recorded earlier: the ledger's
    # `situation` slot may describe a *previous* matter in the same call, and
    # mixing two matters into one query is how a deposit question came back
    # with a company-law judgment.
    spoken = said or [
        value for value in (ledger.value("issue"), ledger.value("situation"))
        if isinstance(value, str) and value.strip()
    ]
    if not spoken:
        # Nothing was described in words they chose: fall back to the matter
        # held while another intake was running, then to the procedure's name.
        noted = [item for item in ledger.matters if item.get("issue")]
        if noted:
            spoken = [str(noted[-1]["issue"])]
    card = procedural.BY_ID.get(procedure_id) or {}
    if not spoken and card:
        spoken = [str(card.get("label") or "")]

    # The vocabulary to measure coverage by: the card this turn's own words
    # match wins ("deposit" inside an eviction matter is a deposit question),
    # then the intake the call is in.
    own = procedural.best_match(text, "IN") or {}
    # A turn too short to stand alone is not a new matter, so the intake the
    # call is in decides the vocabulary the candidates must cover.
    focus = tuple(
        str(term) for term in (card.get("case_terms") or ()) if str(term).strip()
    ) or None
    query = " ".join(dict.fromkeys(" ".join(part.split()) for part in spoken if part)).strip()
    # This text reaches the search and, when it runs, the case judge's question.
    # It is masked at the same boundary as the live message: an identifier read
    # out on turn two must not become a model-facing query on turn five, and a
    # number is a rare-term anchor that would pull the ranking to it anyway.
    return privacy.for_model(query)[:320], focus


def _turn_search(
    stored_state: Any,
    ledger: facts.Ledger,
    text: str,
    history: list[dict[str, Any]] | None,
    earlier: str,
) -> tuple[str, str, tuple[str, ...] | None]:
    """What this turn searches with: the call, not the sentence.

    Three sources, in the order they earn their place: the turn's own
    substantive words, the thread it points at (`_carried_matter`), and the
    legal-vocabulary phrasing the previous consultation wrote. The last one is
    only folded in when the turn does not bring five substantive terms of its
    own, so a new matter still leads its own search.

    Returns `(search_text, carried_query, focus)`.
    """
    carried_query, carried_focus = _carried_matter(stored_state, ledger, text, history, earlier)
    if carried_query:
        # The thread, not the sentence, is what gets searched: "money" is a
        # pointer, and putting it in front of the query is what ranks bank and
        # tax judgments above the wage dispute the caller is actually in. The
        # matter leads, and only the turn's substantive words follow it.
        residual = " ".join(
            term for term in retrieval.content_terms(text)
            if term.lower() not in POINTER_TERMS
        )
        # The new aspect the caller just named leads; the thread follows it so
        # the two words have a matter to attach to. Order also decides what the
        # cap drops, and it must never be the newest words.
        search_text = f"{residual} {carried_query}".strip()[:360]
    else:
        search_text = text
    # The model's own phrasing from the previous turn is the one sentence in
    # the call written in the corpus's vocabulary. A follow-up ("what can I do
    # about the money?") otherwise throws it away.
    previous_query = " ".join(
        str((stored_state or {}).get("search_query") or "").split()
    ) if isinstance(stored_state, dict) else ""
    if previous_query and (
        carried_query or len(retrieval.content_terms(search_text)) < 5
    ) and previous_query.lower() not in search_text.lower():
        search_text = f"{search_text} {previous_query[:160]}".strip()[:420]
    return search_text, carried_query, carried_focus


def _evidence_block(
    text: str,
    jurisdiction: str,
    want_cases: bool,
    focus: tuple[str, ...] | None = None,
):
    """Retrieval in one call, so it can run off the event loop.

    `focus` is passed in when the caller already computed it, so a rewrite
    spends no time re-matching procedure signals; None means "derive it here",
    which is what the speculative pre-reader call needs.
    """
    sources = knowledge.retrieve(text, jurisdiction, limit=8)
    cases: list[dict[str, Any]] = []
    if want_cases:
        terms = _focus_terms(text, jurisdiction) if focus is None else focus
        cases = case_corpus.retrieve_cases(text, jurisdiction, limit=6, focus=terms)
    return sources, cases


def _merge_ranked(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Interleave two result lists by id, first list keeping its order."""
    key = lambda item: str(item.get("id") or "")
    order = retrieval.interleave([key(item) for item in primary], [key(item) for item in secondary], limit)
    by_id: dict[str, dict[str, Any]] = {}
    for item in [*primary, *secondary]:
        by_id.setdefault(key(item), item)
    return [by_id[item] for item in order if item in by_id]


def _merge_sources(existing: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add tool-retrieved sources to the turn's evidence, without duplicates."""
    merged = list(existing)
    seen = {source.get("id") for source in merged}
    for source in extra:
        if isinstance(source, dict) and source.get("id") and source["id"] not in seen:
            merged.append(source)
            seen.add(source["id"])
    return merged[:12]


def _json_tool_requests(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Accept in-band tool requests from a server without native tool calling."""
    raw = (payload or {}).get("tool_requests")
    if not isinstance(raw, list):
        return []
    requests: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:4]):
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]:
            requests.append({
                "id": f"json_{index}",
                "name": item["name"][:64],
                "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
            })
    return requests


# Numeric tokens a reply may carry without a source: emergency and helpline
# short codes, which the caller needs to hear even when nothing in the corpus
# mentions them.
SAFE_SHORT_CODES = {
    "112", "100", "101", "102", "103", "104", "108", "1090", "1091", "1098",
    "181", "1915", "1930", "1073", "1075", "14416", "14417",
}

_SECTION_TOKEN = re.compile(
    r"\b(?:section|secs?\.?|article|rule|regulation|schedule|धारा|अनुच्छेद|सेक्शन|"
    r"సెక్షన్|பிரிவு|ধারা|ಕಲಂ|വകുപ്പ്|ਧਾਰਾ)\s*([0-9]{1,4}[A-Za-z]?)",
    re.I,
)
_UNIT_TOKEN = re.compile(
    r"\b([0-9]{1,3})\s*(?:days?|months?|years?|weeks?|hours?|दिन|महीन\w*|साल|वर्ष|"
    r"రోజు\w*|నెల\w*|సంవత్సర\w*|நாட\w*|மாத\w*|ஆண்ட\w*|দিন|মাস|বছর|"
    r"ದಿನ\w*|ತಿಂಗಳ\w*|ವರ್ಷ|ദിവസ\w*|മാസ\w*|വർഷ\w*|ਦਿਨ|ਮਹੀਨ\w*|ਸਾਲ)\b",
    re.I,
)
_PERCENT_TOKEN = re.compile(r"\b([0-9]{1,3})\s*(?:%|per ?cent|percent|प्रतिशत)", re.I)
_MONEY_TOKEN = re.compile(r"(?:₹|rs\.?|inr)\s*([0-9][0-9,]{2,})", re.I)


def _numeric_tokens(text: str) -> set[str]:
    """Numbers in a reply that only a source or the caller can make true.

    A section number, a limitation period, a rate or a sum is the kind of
    statement a caller acts on, so each one is reduced to a comparable token
    and checked against what the turn actually has.
    """
    if not text:
        return set()
    tokens = {f"s:{match.group(1).lower()}" for match in _SECTION_TOKEN.finditer(text)}
    tokens |= {f"u:{match.group(1)}{match.group(0).lstrip()[len(match.group(1)):].strip()[:1].lower()}"
               for match in _UNIT_TOKEN.finditer(text)}
    tokens |= {f"p:{match.group(1)}" for match in _PERCENT_TOKEN.finditer(text)}
    tokens |= {f"m:{match.group(1).replace(',', '')}" for match in _MONEY_TOKEN.finditer(text)}
    return tokens


def _prose_gaps(
    reply: str,
    claims: list[dict[str, str]],
    caller_text: str,
) -> list[str]:
    """Legal numbers stated in the spoken reply that nothing supports.

    Claim-level checking validated `legal_claims`; the sentence the caller
    actually hears is `reply`. A model that writes "section 138" or "within 30
    days" into prose without a claim item used to reach the caller unverified.
    A number is acceptable when a verified quote, a verified claim or the
    caller's own words contain it, or when it is an emergency short code.
    """
    # A number inside a question is not a legal proposition being asserted:
    # "was that 30 days ago?" is the model trying to establish a fact, and
    # treating it as a citation would replace a good intake question with a
    # wall of passages. An assertion in the same reply is still checked.
    asserted = " ".join(
        sentence for sentence in re.split(r"(?<=[.!?\u0964])\s+", reply)
        if not sentence.strip().endswith("?")
    )
    spoken = _numeric_tokens(asserted)
    if not spoken:
        return []
    supported = _numeric_tokens(caller_text)
    for claim in claims:
        supported |= _numeric_tokens(claim.get("quote", ""))
        supported |= _numeric_tokens(claim.get("text", ""))
    unsupported = spoken - supported
    return sorted(token for token in unsupported if token.split(":", 1)[-1] not in SAFE_SHORT_CODES)


def _grounded_prose_scope(context: Any) -> str:
    """Collect caller words, recorded facts, and reviewed procedure tokens that legitimately ground prose numbers."""
    parts = [getattr(context, "text", "")]
    for h in getattr(context, "history", []) or []:
        if isinstance(h, dict) and h.get("role") in {"user", "caller"}:
            parts.append(str(h.get("content", "")))
    ledger = getattr(context, "ledger", None)
    if ledger and hasattr(ledger, "facts"):
        for f in ledger.facts.values():
            parts.append(str(getattr(f, "value", "")))
            parts.append(str(getattr(f, "evidence", "")))
    procedure = getattr(context, "procedure", None)
    if isinstance(procedure, dict):
        parts.append(str(procedure.get("label", "")))
        parts.append(str(procedure.get("summary", "")))
        for step in procedure.get("steps", []):
            if isinstance(step, dict):
                parts.append(str(step.get("action", "")))
                parts.append(str(step.get("detail", "")))
    for s in getattr(context, "sources", []) or []:
        if isinstance(s, dict):
            parts.append(str(s.get("title", "")))
            parts.append(str(s.get("id", "")))
    return " ".join(parts)


def _claims(raw_claims: Any, sources: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    if not raw_claims:
        return [], []
    if not isinstance(raw_claims, list):
        return [], ["legal_claims_not_a_list"]
    by_id = {source["id"]: source for source in sources}
    clean: list[dict[str, str]] = []
    rejected: list[str] = []
    for index, claim in enumerate(raw_claims[:20]):
        if not isinstance(claim, dict):
            rejected.append(f"claim_{index}_invalid_shape")
            continue
        claim_text = _text(claim.get("text"), 800)
        source_id = _text(claim.get("source_id"), 120)
        quote = _text(claim.get("quote"), 1200)
        source = by_id.get(source_id)
        source_text = " ".join(str(source.get("text", "")).split()) if source else ""
        if not claim_text or not source or not quote or quote not in source_text:
            rejected.append(f"claim_{index}_not_exactly_supported")
            continue
        # The quote is real, but is it the paragraph the claim is about? Every
        # number the claim asserts has to be inside the quoted span itself;
        # otherwise a quote from the neighbour provision passes on substring
        # identity while saying something else.
        # A section number asserted by the claim is also supported if the source itself
        # (its statutory title or provision id) is explicitly that section and the quote is
        # from that source:
        source_section_numbers = {
            tok for tok in (
                _numeric_tokens(source.get("title", ""))
                | _numeric_tokens(source.get("id", "").replace("-", " ").replace("_", " "))
                | _numeric_tokens(source_text[:140])
            )
            if tok.startswith("s:")
        }
        unsupported_numbers = _numeric_tokens(claim_text) - (_numeric_tokens(quote) | source_section_numbers)
        if unsupported_numbers:
            rejected.append(
                f"claim_{index}_quote_omits:{','.join(sorted(unsupported_numbers))[:60]}"
            )
            continue
        clean.append({"text": claim_text, "source_id": source_id, "quote": quote})
    return clean, rejected


@dataclass
class TurnContext:
    text: str
    language: str
    jurisdiction: str
    jurisdiction_basis: str
    route: dict[str, Any]
    procedure: dict[str, Any] | None
    plan: dict[str, Any] | None
    # The caller's message with identity numbers masked, which is what any
    # model in this turn is allowed to receive.
    model_text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    secondary: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    # Bounded digest of the turns that fell out of the live history window,
    # privacy-masked like everything else the models receive.
    earlier: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    requested_service: str | None = None
    ledger: facts.Ledger = field(default_factory=facts.Ledger)
    risk: dict[str, Any] = field(default_factory=dict)
    reading: dict[str, Any] | None = None
    workflow_state: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None
    # True when this turn left the same step open for a second time, so the
    # question can be re-asked in different words rather than repeated.
    repeated_question: bool = False
    # How the turn's language was decided: caller_selected, speech_recognition,
    # script, romanised_hindi, reader, default. Reported so a wrong guess is
    # visible and correctable instead of quietly shaping the answer.
    language_basis: str = "auto"
    tool_audit: list[dict[str, Any]] = field(default_factory=list)
    tool_artifact: dict[str, Any] | None = None
    # Set when the post-retrieval judge rejected every candidate and the two
    # closest were shown anyway; the interface says so instead of presenting
    # them as related.
    case_relevance_note: str = ""
    # What the model wrote as this turn's search query. It is kept in the
    # session and folded into the next turn's retrieval, so a call's legal
    # vocabulary accumulates instead of being re-derived from four words.
    search_query: str = ""
    # 1-based position of this turn in the call.
    turn: int = 1

    @property
    def route_name(self) -> str:
        return self.route.get("name", "unclear")


# What retrieval may put in front of the model. This is a prompt-size bound,
# not a legal one: a land matter can need a central Act, a State amendment, a
# notification and a judgment, and the old "two statutes and one judgment"
# ceiling silently made that answer impossible. Counts are the cheap guard;
# the character caps per item are the real budget.
MAX_PROMPT_SOURCES = max(2, min(10, int(os.getenv("LLM_MAX_SOURCES", "6"))))
MAX_PROMPT_CASES = max(1, min(8, int(os.getenv("LLM_MAX_CASES", "5"))))


def _stage(trace: list[dict[str, Any]], name: str, started: float, **detail: Any) -> None:
    trace.append({"stage": name, "ms": round((time.perf_counter() - started) * 1000), **detail})


async def analyze(
    text: str,
    language: str = "auto",
    jurisdiction: str = "auto",
    service: str | None = None,
    history: list[dict[str, str]] | None = None,
    session: dict[str, Any] | None = None,
) -> TurnContext:
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []
    clean_text = _text(text, 6000)
    requested_language = language_layer.normalize(language)
    # The caller's language is decided from evidence before anything is
    # generated, so no later stage has to ask. An explicit choice wins, then the
    # recogniser's detected code for this utterance, then the script, then
    # Romanised Hindi markers.
    # What the model and the retriever may see. Identity numbers are masked
    # here, at the boundary, not only in the logs: a model that never receives
    # an Aadhaar number cannot repeat it in a draft.
    model_text = privacy.for_model(clean_text)
    # Compacted earlier turns ride the same mask as the live message: the
    # digest contains nothing the models may read in the clear.
    earlier_text = privacy.for_model(_earlier_digest((session or {}).get("earlier")))
    turn_language, language_basis = language_layer.resolve(clean_text, override=requested_language)
    hint = _text(jurisdiction, 40).upper()
    hint = hint if hint in INDIAN_JURISDICTIONS else "auto"
    capabilities = SERVICE_CATALOG["capabilities"]
    capability_ids = {item["id"] for item in capabilities}
    history = (history or [])[-12:]

    turn_number = int((session or {}).get("turns", 0)) + 1

    # The ledger is rebuilt from the stored session, not from the transcript:
    # a fact the caller stated eight turns ago stays known, and nothing the
    # model merely implied is treated as stated.
    ledger = facts.Ledger.from_dict((session or {}).get("facts"))

    # The lexical net runs first and costs microseconds. If it already sees an
    # emergency, the safety script must not wait for a model: a person in
    # danger gets the script now, and the reader is skipped for this turn.
    net_route = routing.route_turn(clean_text, hint)

    # Retrieval does not depend on the reader, so it starts now and runs in a
    # worker thread while the model is reading the turn. The jurisdiction guess
    # is the same deterministic rule the reader's facts go through, so a
    # mismatch is rare and is corrected below.
    suspicion_guess = hint if hint != "auto" else (ledger.value("jurisdiction") or "IN")
    if suspicion_guess not in INDIAN_JURISDICTIONS:
        suspicion_guess = "IN"
    # The call's stored state, read here because a bare follow-up has to be
    # searched as the matter it continues rather than as its own four words.
    stored_state = (session or {}).get("last_state")
    search_text, carried_query, carried_focus = _turn_search(
        stored_state, ledger, model_text, history, earlier_text,
    )
    speculation: "asyncio.Task[tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None" = None
    if not net_route.emergency:
        # Precedent is worth looking for on any substantive legal turn, not only
        # when the caller says "similar case". "My landlord kept my deposit"
        # benefits from a judgment the caller never names, and the regex cue is
        # only the strongest signal, not the only one. A bare greeting still
        # skips the search: three content terms is the floor for a real question.
        # This runs in a worker thread alongside the reader call, so the cost
        # is overlap rather than added latency.
        wants_cases = bool(
            any(re.search(pattern, clean_text, re.I) for pattern in routing.PRECEDENT_CUES)
            or len(retrieval.content_terms(search_text)) >= 3
            or bool(carried_query)
        )
        speculation = asyncio.create_task(
            asyncio.to_thread(
                _evidence_block, search_text, suspicion_guess, wants_cases, carried_focus,
            )
        )

    if net_route.emergency:
        reading = understanding.combine(None, clean_text)
        _stage(trace, "understand", started, source="lexical", skipped="emergency fast path")
        route = net_route
    else:
        # There is no separate reading call. The model reads the turn inside the
        # consultation that answers it, and `_apply_consultation` validates
        # that reading afterwards. Before the call there is no model, so the
        # retrieval query is built from the call itself -- the caller's recent
        # substantive turns, the open intake and last turn's legal phrasing.
        reading = understanding.combine(None, clean_text)
        reading.source = "deferred" if model_gateway.configured() else "lexical"
        reading.intent = net_route.name
        _stage(trace, "understand", started, source=reading.source)
        route = net_route

    # When the language came from a heuristic rather than from the caller or the
    # recogniser, the reader is the better judge: it sees the whole sentence, and
    # it can tell Marathi from Hindi in the same script. A caller's explicit
    # choice is never overwritten.
    if language_basis in {"script", "romanised_hindi", "default"}:
        read_language = _normalize_language(getattr(reading, "language", None), turn_language)
        if read_language not in {language_layer.AUTO, language_layer.PRISMA_AUTO} and read_language != turn_language:
            turn_language, language_basis = read_language, "reader"
    _stage(
        trace, "language", started, language=turn_language, basis=language_basis,
        requested=requested_language,
    )
    _stage(trace, "route", started, route=route.name, confidence=route.confidence, reasons=route.reasons[:5])

    # The deterministic extraction is the floor, not the reading. It runs on
    # every turn so a provider outage cannot leave the app with an empty
    # ledger, and so a state named in the message is known *before* retrieval:
    # a Karnataka question must retrieve Karnataka's material. Facts the model
    # reports later are recorded with source `reader` and replace these (`net`)
    # values, because a grounded reading of the caller's sentence beats a
    # pattern match on it.
    prior_jurisdiction = ledger.value("jurisdiction")
    fragment, extracted = facts.extract(clean_text, turn_number)
    ledger = facts.merge(ledger, fragment, turn_number)
    if extracted:
        _stage(trace, "facts", started, mode="lexical_net", slots=extracted, known=len(ledger.facts))

    # Injection attempts are counted deterministically even when the reader is
    # available: the reader is told to ignore them, the counter records them.
    if facts.INJECTION_RE.search(clean_text):
        ledger.injections += 1
        _stage(trace, "injection_attempt", started, count=ledger.injections)

    read_jurisdiction = next((f["value"] for f in reading.facts if f["slot"] == "jurisdiction"), None)
    stored_jurisdiction = ledger.value("jurisdiction")
    if hint != "auto":
        resolved, basis = hint, "caller_selected"
    elif isinstance(read_jurisdiction, str) and read_jurisdiction in INDIAN_JURISDICTIONS:
        resolved, basis = read_jurisdiction, "caller_location_in_message"
    elif isinstance(stored_jurisdiction, str) and stored_jurisdiction in INDIAN_JURISDICTIONS:
        # A state stated earlier in the same call still governs later turns. A
        # lawyer does not re-ask which state you are in on every sentence, and
        # forgetting it silently narrowed every answer to national law. A state
        # named in *this* message is reported as such: the caller is owed the
        # reason the answer is Karnataka-specific.
        resolved = stored_jurisdiction
        basis = (
            "stated_earlier_in_this_call"
            if stored_jurisdiction == prior_jurisdiction
            else "caller_location_in_message"
        )
    else:
        resolved, basis = "IN", "national_default_location_not_established"

    if hint != "auto":
        # A jurisdiction the caller picked in the interface is a stated fact,
        # not a model inference, and it must survive into the ledger.
        ledger.record("jurisdiction", hint, "ui", turn_number, "jurisdiction")

    # --- which matter this turn belongs to -----------------------------------
    # Entry is decided before the turn's facts are written, because a fact has
    # to be scoped to the intake it answers: the dates of a police complaint
    # must not count as the dates of a wages dispute.
    workflow_state = workflow.load_state(stored_state.get("workflow") if isinstance(stored_state, dict) else None)
    workflow_events: list[dict[str, Any]] = []
    if route.name == "emergency":
        workflow_state = workflow.abort(workflow_state, "immediate_danger")
        if workflow_state:
            _stage(trace, "workflow", started, id=workflow_state["id"], status=workflow_state.get("status"))
    else:
        # The reader's chosen procedure is preferred, but a lexical match must
        # also open the intake: workflows cannot depend on a provider being up.
        procedure_id = reading.procedure_id or (route.procedure or {}).get("id")
        if not procedure_id and isinstance(stored_state, dict) and stored_state.get("procedure_id"):
            prev_proc_id = stored_state.get("procedure_id")
            if procedural.BY_ID.get(prev_proc_id) and route.name in {"unclear", "procedural"}:
                procedure_id = prev_proc_id
                chosen = procedural.BY_ID[procedure_id]
                route.procedure = {**chosen, "score": 0.0, "matched_signals": ["ongoing_consultation"]}
                route.name = "procedural"
                route.confidence = max(route.confidence, 0.7)
                route.reasons.append("ongoing procedure consultation maintained")
        closed = workflow.is_closed(workflow_state)
        # A finished intake is announced once, not repeated at the end of every
        # later answer. It stays in the transcript; the session stops steering.
        if closed and workflow_state and workflow_state.get("turn", 0) < turn_number and not procedure_id:
            workflow_state, closed = None, False
        answered = (route.procedure or {}).get("id")
        if workflow_state and workflow_state.get("status") == "active" and procedure_id and procedure_id != workflow_state["id"]:
            # A second matter raised mid-call is held, not dropped. Naming the
            # process does not mean the caller wants it now, so the intake in
            # progress keeps the floor and the completion turn offers this one.
            if procedural.BY_ID.get(procedure_id) and ledger.note_matter(procedure_id, reading.issue or "", turn_number):
                _stage(trace, "matter_noted", started, procedure=procedure_id, while_active=workflow_state["id"])
        if (workflow_state is None or closed) and procedure_id and route.name != "human":
            candidate = workflow.for_procedure(procedure_id)
            restarts = workflow_state is None or workflow_state.get("status") == "aborted" or candidate["id"] != workflow_state["id"]
            if candidate and restarts:
                workflow_state = workflow.start(candidate["id"], turn_number)
                _stage(trace, "workflow", started, id=workflow_state["id"], started_at_node=workflow_state["node"])

    matter = workflow_state["id"] if workflow_state and workflow_state.get("status") == "active" else ""
    written = facts.apply_reading(ledger, reading.facts, turn_number, matter)
    if reading.issue:
        ledger.record("situation", reading.issue, "reader", turn_number, matter=matter)
    if matter:
        # Facts the lexical net extracted (no reader available) belong to this
        # intake too; without this they could not satisfy its own steps.
        ledger.tag_matter(turn_number, matter)

    # Downstream stages expect a plan-shaped object. It is derived from the
    # reading rather than from a second model call: one interpretation, one
    # place where it can be wrong, one trace entry.
    plan: dict[str, Any] | None = None
    if reading.source != "lexical":
        plan = {
            "issue": reading.issue,
            "jurisdiction": read_jurisdiction,
            "needs_precedents": reading.needs_precedents or route.name == "precedent",
            "needs_primary_law": route.name in {"statute", "draft", "unclear"},
            "services": [],
            "reported_facts": [fact["value"] for fact in reading.facts],
        }
    if written:
        _stage(trace, "facts_written", started, slots=written, conflicts=len(ledger.contradictions))

    # --- advance the intake with the facts now on record ---------------------
    if workflow_state and workflow_state.get("status") == "active":
        ledger.resolve_matter(workflow_state["id"])
        workflow_state, workflow_events = workflow.advance(workflow_state, ledger, turn=turn_number)
        for event in workflow_events:
            if event.get("event") == "tool_call":
                outcome = tool_layer.execute(
                    event["tool"], event["arguments"],
                    tool_layer.ToolContext(
                        text=clean_text, jurisdiction=resolved, ledger=ledger,
                        workflow_state=workflow_state, turn=turn_number,
                    ),
                )
                if outcome.ok and outcome.name == "get_procedure":
                    chosen = procedural.BY_ID.get(outcome.data.get("id", ""))
                    if chosen:
                        route.procedure = {**chosen, "score": 0.0, "matched_signals": ["workflow"]}
                        route.name = "procedural"
        _stage(
            trace, "workflow", started, id=workflow_state["id"], node=workflow_state["node"],
            status=workflow_state["status"], events=[event["event"] for event in workflow_events][:6],
        )

    # An intake that is running keeps its procedure attached, even when the
    # answer is a terse "haan" the reader could not classify. Losing the
    # context mid-intake is what made a conversation feel like a form that
    # silently reset itself.
    if workflow_state and workflow_state.get("status") in {"active", "completed"} and route.name not in {"emergency", "human"}:
        active_procedure_id = workflow.BY_ID[workflow_state["id"]].get("procedure_id") or ""
        chosen = procedural.BY_ID.get(active_procedure_id)
        answered = (route.procedure or {}).get("id")
        # A caller who asks about a *different* process gets that process
        # answered; the running intake is paused, not overwritten. Otherwise
        # "how do I file an RTI application" would be answered with a question
        # about the wages dispute.
        if chosen and (not answered or answered == active_procedure_id):
            if route.name == "unclear":
                route.name = "procedural"
                route.confidence = max(route.confidence, 0.6)
                route.reasons.append("mid-intake turn keeps the intake context")
            route.procedure = {**chosen, "score": 0.0, "matched_signals": ["workflow"]}

    # "Asked twice" is about the same step being left open for a second turn,
    # not about a slot asked long ago and answered since. It is decided once,
    # here, so every caller of `_question_for` gets the same answer.
    repeated_question = False
    if workflow_state and workflow_state.get("status") == "active":
        node = str(workflow_state.get("node") or "")
        active_procedure_id = workflow.BY_ID[workflow_state["id"]].get("procedure_id") or ""
        answered_procedure = (route.procedure or {}).get("id")
        on_topic = route.name in {"procedural", "unclear"} and (not answered_procedure or answered_procedure == active_procedure_id)
        repeated_question = on_topic and workflow_state.get("asked_node") == node
        workflow_state["asked_node"] = node

    context = TurnContext(
        text=clean_text,
        model_text=model_text,
        language=turn_language,
        language_basis=language_basis,
        jurisdiction=resolved,
        jurisdiction_basis=basis,
        route=route.as_dict(),
        procedure=(procedural.public(route.procedure) if route.procedure else None),
        plan=plan,
        history=history,
        earlier=earlier_text,
        trace=trace,
        requested_service=service if service in capability_ids else None,
        ledger=ledger,
        risk=route.risk or triage.assess(clean_text).as_dict(),
        reading=reading.as_dict(),
        workflow_state=workflow_state,
        workflow=workflow.public(workflow_state, ledger),
        repeated_question=repeated_question,
        search_query=str((stored_state or {}).get("search_query") or "") if isinstance(stored_state, dict) else "",
        turn=turn_number,
    )

    if route.name in {"emergency", "human"}:
        if speculation is not None:
            speculation.cancel()
        return context

    # A process question with no curated match is still a law question worth
    # retrieving for: the reader understood the intent, the catalogue simply
    # does not cover that process yet. Retrieval also runs *during* an intake,
    # because callers ask a real question in the same breath as an answer
    # ("he has given nothing in writing — can he just deny I worked there?"),
    # and the reply needs evidence to answer it without guessing.
    retrieval_started = time.perf_counter()
    if route.name in {"statute", "draft", "unclear", "precedent", "procedural"}:
        if speculation is not None:
            context.sources, context.cases = await speculation
            if suspicion_guess != resolved:
                # The reader resolved a different state than the guess; state
                # sources are excluded by a national query, so re-run once.
                context.sources = knowledge.retrieve(model_text, resolved, limit=8)
                context.cases = []
        else:
            context.sources = knowledge.retrieve(search_text, resolved, limit=8)
        if (route.name == "precedent" or (plan or {}).get("needs_precedents")) and not context.cases:
            # Same reviewed vocabulary as the speculative pass. Without it this
            # branch searched the caller's framing alone, which is how a refused
            # FIR landed on a Lokayukta corruption case.
            context.cases = case_corpus.retrieve_cases(
                search_text, resolved, limit=6,
                focus=carried_focus or _focus_terms(search_text, resolved),
            )
        if route.name in {"unclear", "statute"}:
            context.secondary = knowledge.retrieve_secondary_qa(model_text, limit=2)
        # There is no second search here any more. The model's legal-vocabulary
        # phrasing is written by the consultation, which runs *after* this
        # retrieval; it is persisted in the session and folded into the next
        # turn's query by `_turn_search`, and the model can ask for a search in
        # this turn through `search_law` / `search_judgments` when the seeded
        # evidence is missing.
        _stage(
            trace, "evidence", retrieval_started,
            mode="curated_procedure+retrieval" if context.procedure else "retrieval",
            steps=len(context.procedure["steps"]) if context.procedure else 0,
            sources=len(context.sources), cases=len(context.cases),
            secondary=len(context.secondary),
            terms=retrieval.content_terms(search_text)[:8],
            carried=bool(carried_query),
        )
        # A judgment is retrieved because it shares vocabulary with the
        # question, and sharing vocabulary is not addressing the question. The
        # reader's own words are the best evidence available here, so one small
        # call decides which of the top candidates are actually about it. It can
        # only remove: a candidate the judge did not answer stays, and a judge
        # that is down or slow leaves the list untouched rather than quietly
        # deleting a correct decision.
        if context.cases and model_gateway.case_judge_available():
            judged_started = time.perf_counter()
            # The judge is asked about the matter, not about the sentence: a
            # follow-up ("what can I do about the money?") is unjudgeable on its
            # own words, and judging it alone is how on-point judgments for the
            # actual dispute were dropped for not answering four words.
            case_question = context.model_text
            if carried_query:
                case_question = f"{context.model_text} (matter: {carried_query})"
            relevance = await model_gateway.case_relevance(case_question, context.cases)
            if relevance is not None:
                # The judge reads the same passage the caller will read, so its
                # score is both a filter and an ordering: a 3 the retriever put
                # fourth is more useful than a 2 it put first. A candidate the
                # judge skipped keeps a neutral 2 and its retrieval order,
                # because silence is not a negative finding.
                ordered = sorted(
                    enumerate(context.cases),
                    key=lambda pair: (-relevance.get(pair[0], 2), pair[0]),
                )
                kept = [case for index, case in ordered if relevance.get(index, 2) >= 2]
                # A judge that rejects every candidate is usually right and
                # occasionally too strict; the caller is left with nothing to
                # read in the second case. Keep the two closest decisions,
                # mark the turn so the interface says they are not confirmed to
                # address the question, and let the caller judge for themselves
                # instead of pretending nothing was retrieved.
                reinstated = 0
                if not kept:
                    kept = [case for _, case in ordered[:2]]
                    reinstated = len(kept)
                    context.case_relevance_note = "none_on_point"
                _stage(
                    trace, "case_relevance", judged_started,
                    judged=len(relevance), kept=len(kept),
                    dropped=len(context.cases) - len(kept),
                    unanswered=len(context.cases) - len(relevance),
                    reinstated=reinstated,
                    best=max(relevance.values()) if relevance else 0,
                )
                context.cases = kept
    return context


def _base_result(context: TurnContext, reply: str, mode: str, status: str) -> dict[str, Any]:
    language_out = context.language if context.language != "auto" else "auto"
    return {
        "response": reply,
        "spoken": speakable(reply),
        "language": language_out,
        # Timbre's mixed-content path is `auto`, and `hi-en` is a real code, so
        # the voice language is resolved through the same rule that decided the
        # reply language rather than being copied blindly.
        "tts_language": language_layer.tts_language(language_out),
        "language_basis": context.language_basis,
        "language_label": LANGUAGE_LABELS.get(language_out, language_out),
        "issue": _text((context.plan or {}).get("issue"), 120) or context.route_name,
        # The model's legal-vocabulary phrasing of this turn, kept so the next
        # turn's retrieval can search the call rather than one sentence.
        "search_query": context.search_query or None,
        "jurisdiction": context.jurisdiction,
        "jurisdiction_label": JURISDICTION_LABELS.get(context.jurisdiction, context.jurisdiction),
        "jurisdiction_basis": context.jurisdiction_basis,
        "route": context.route,
        "procedure": context.procedure,
        "reported_facts": [
            _text(x, 500) for x in (context.plan or {}).get("reported_facts", [])[:20] if isinstance(x, str)
        ] if isinstance((context.plan or {}).get("reported_facts"), list) else [],
        "facts_status": "caller_reported_not_independently_verified",
        "claims": [],
        "case_context": [],
        "verification": {
            "status": "not_applicable",
            "entailment": "not_reviewed",
            "rejected_claims": 0,
            "source_scope": "quotes are matched exactly; legal interpretation is not reviewed",
        },
        "sources": [],
        "follow_up": None,
        "next_question": None,
        "services_used": [],
        "artifact": None,
        "risk_flags": [],
        "source_gap": None,
        "safety": _safety_block(context),
        "reading": context.reading,
        "workflow": context.workflow,
        "tool_calls": context.tool_audit,
        "conversation_state": {
            "known_facts": [fact.as_dict() for fact in context.ledger.facts.values()],
            "conflicts": context.ledger.contradictions,
            "conflict_note": facts.conflict_note(context.ledger),
            "turn": int((context.ledger.facts and max(f.turn for f in context.ledger.facts.values())) or 0),
        },
        "identifiers_in_message": privacy.find(context.text),
        "answer_status": status,
        "mode": mode,
        "human_connection": {"offered": False, "connected": False, "requires_user_choice": True, "language": language_out, "reason": None},
        "disclaimer": "AI-generated legal information can be wrong or incomplete. Source coverage, currentness and interpretation need checking; this is not a lawyer's advice.",
        "corpus": {"legal_source_candidates": len(context.sources), "case_judgment_candidates": len(context.cases), "secondary_qa_candidates_not_cited": len(context.secondary), "case_relevance": context.case_relevance_note or None},
        "trace": context.trace,
    }


def _safety_block(context: TurnContext) -> dict[str, Any]:
    risk = context.risk or {"level": "normal", "categories": [], "policies": []}
    level = risk.get("level", "normal")
    policies = risk.get("policies", [])
    notice = None
    if level == "critical":
        notice = "Immediate safety comes before legal steps. Nothing here replaces emergency help."
    elif level == "high" and "block_drafting" in policies:
        notice = "This looks like something happening now. Doing the urgent reporting step comes before any document."
    elif level == "high":
        notice = "There may be a deadline or urgent step. Confirm dates before relying on this."
    return {
        "level": level,
        "categories": risk.get("categories", []),
        "policies": policies,
        "notice": notice,
        "blocks_drafting": "block_drafting" in policies,
    }


def _completion_question(context: TurnContext) -> str:
    """What a lawyer asks when the intake is finished: what happens next.

    Asking another fact here was the old behaviour, and it re-asked something
    the caller had already answered — the clearest sign a conversation is
    being driven by a checklist instead of by a person.
    """
    matter = context.ledger.pending_matter()
    if matter:
        procedure = procedural.BY_ID.get(str(matter.get("procedure_id"))) or {}
        label = procedure.get("label") or "the other problem you mentioned"
        return f"You also raised this on the call: {label}. Do you want to take that up now?"
    return (
        "That is everything I need for this. Shall I prepare the first document you will need, "
        "or would you rather go through the steps one at a time?"
    )


def _question_for(context: TurnContext) -> str | None:
    if context.route_name == "emergency":
        return "Are you somewhere safe right now?"
    state = context.workflow_state or {}
    if state.get("status") == "completed":
        return _completion_question(context)
    # Inside an active workflow the next required question is not a judgement
    # call: it is the step's declared requirement, and skipping it would leave
    # the intake incomplete while looking finished. When a model is available
    # it phrases this question in the caller's language (see `_intake_block`).
    # A question the reader formed from this specific story beats any template.
    # The reviewed flow says which fact the step still wants; the model decides
    # whether *asking* it now is the useful move, and how to phrase it for this
    # caller. Reversing these two is what made the intake feel like a form: the
    # script won every time, even when the caller had just asked something else.
    phrased = (context.reading or {}).get("next_question")
    if isinstance(phrased, str) and phrased.strip():
        return phrased.strip()
    if state.get("status") == "active" and context.workflow and context.workflow.get("ask"):
        ask = str(context.workflow["ask"])
        slot = context.workflow.get("needs")
        if slot:
            context.ledger.mark_asked(slot)
        if context.repeated_question:
            # The fact still did not arrive. Repeating the same sentence word
            # for word is how a form fails; a person says they missed it.
            return f"Sorry, I did not catch that. {ask}"
        active_procedure_id = (context.workflow or {}).get("procedure_id")
        if context.procedure and active_procedure_id and context.procedure.get("id") != active_procedure_id:
            return f"Back to the earlier matter: {ask}"
        return ask
    if context.procedure:
        return NEXT_QUESTION.get(context.procedure["id"], "What have you already tried so far?")
    missing = (context.plan or {}).get("missing_material_facts")
    if isinstance(missing, list):
        for item in missing:
            if isinstance(item, str) and item.strip():
                cleaned = item.strip().rstrip(".")
                return cleaned if cleaned.endswith("?") else f"Could you tell me {cleaned[0].lower() + cleaned[1:]}?"
    slot, question = facts.next_missing(context.ledger, context.route_name)
    if question:
        context.ledger.mark_asked(slot or "")
        return question
    if context.jurisdiction_basis == "national_default_location_not_established":
        return "Which state are you in?"
    if context.route_name == "precedent":
        return "Do you already have a notice or court order about this?"
    return "What would you like to happen next?"


def _procedure_reply(context: TurnContext) -> dict[str, Any]:
    """Deterministic reply for a process turn, with no model available.

    Mid-intake this is one question, not a re-reading of the procedure: the
    steps are a reference the interface keeps visible, and repeating them on
    every turn is what made the conversation sound like a form being read out.
    """
    procedure = context.procedure or {}
    steps = procedure.get("steps", [])
    label = procedure.get("label", "this process")
    state = context.workflow_state or {}
    workflow_block = context.workflow or {}
    same_matter = bool(procedure.get("id")) and procedure.get("id") == workflow_block.get("procedure_id")
    question = workflow_block.get("ask")
    # A different process always shows its own card; the intake's card is shown
    # once, at the start and at the end.
    card_is_shown = not same_matter or state.get("status") != "active" or not state.get("shown")
    caveat = (
        ""
        if procedure.get("reviewed") or not card_is_shown
        else "This procedure is a draft and has not been signed off by a reviewer; confirm it at the official source. "
    )
    if same_matter and state.get("status") == "completed":
        reply = f"That is everything I needed for {label}. {caveat}{_completion_question(context)}"
    elif same_matter and question:
        reply = f"{caveat}{_question_for(context) or question}"
    else:
        first = steps[0]["action"] if steps else ""
        # The summary is on the card directly below, so saying it here printed
        # the same sentence twice on screen. The label stays: it is what the
        # caller hears, and the audio has no card to read.
        reply = (
            f"Here is the route for {label}. "
            f"Start with this: {first}. There are {len(steps)} steps in total, and I can go through them one at a time. "
            f"{caveat}".rstrip()
        )
    result = _base_result(context, reply, "procedure", "procedure_grounded")
    result["procedure"] = procedure
    result["next_question"] = _question_for(context)
    result["follow_up"] = result["next_question"]
    result["sources"] = [{
        "id": procedure["id"],
        "title": procedure["source"]["title"],
        "authority": procedure["source"]["authority"],
        "url": procedure["source"]["url"],
        "kind": "curated_procedure",
        "review_status": procedure.get("review_status"),
    }]
    result["verification"] = {
        "status": "curated_procedure",
        "entailment": "reviewer_signoff_" + ("complete" if procedure.get("reviewed") else "pending"),
        "rejected_claims": 0,
        "source_scope": "steps come from a reviewed file; the model cannot add or reorder them",
    }
    result["human_connection"] = {
        "offered": bool(procedure.get("handoff")),
        "connected": False,
        "requires_user_choice": True,
        "language": result["language"],
        "reason": "procedure_handoff" if procedure.get("handoff") else None,
        "handoff": procedure.get("handoff"),
    }
    return result


def _emergency_reply(context: TurnContext) -> dict[str, Any]:
    script = SAFETY_SCRIPT.get(context.language, SAFETY_SCRIPT["en-IN"])
    result = _base_result(context, script, "emergency", "emergency_safety")
    result["spoken"] = speakable(script, 260)
    result["next_question"] = "Are you somewhere safe right now?"
    result["follow_up"] = result["next_question"]
    flags = ["immediate_safety"] + [f"triage:{item.get('key')}" for item in context.risk.get("categories", [])]
    result["risk_flags"] = list(dict.fromkeys(flags))
    result["verification"] = {
        "status": "safety_script",
        "entailment": "not_applicable",
        "rejected_claims": 0,
        "source_scope": "no legal analysis is performed on a safety route",
    }
    result["human_connection"] = {
        "offered": True, "connected": False, "requires_user_choice": True,
        "language": result["language"], "reason": "immediate_safety",
        "handoff": {"kind": "emergency", "label": "Emergency 112 and the 181 women's helpline", "url": "https://nalsa.gov.in/"},
    }
    return result


def _handoff_reply(context: TurnContext) -> dict[str, Any]:
    reply = (
        "I can connect you to a person, but nothing is sent until you choose it and approve the summary. "
        "I will show you exactly what would be shared first."
    )
    result = _base_result(context, reply, "handoff", "handoff_ready")
    result["next_question"] = "Shall I prepare a short summary you can review before anything is shared?"
    result["follow_up"] = result["next_question"]
    result["human_connection"] = {
        "offered": True, "connected": False, "requires_user_choice": True,
        "language": result["language"], "reason": "caller_requested",
        "handoff": {"kind": "legal_aid", "label": "Free legal aid (NALSA / DLSA)", "url": "https://nalsa.gov.in/"},
    }
    return result


def _fallback_question(context: TurnContext, reply: str) -> str | None:
    """The follow-up chip, only where it will be in a language the caller reads.

    The slot questions are written in English. Handing a Telugu caller a
    Telugu paragraph and then an English suggestion to tap is the same failure
    as answering in the wrong language — and the degraded reply already ends
    with a question of its own in every language it supports.
    """
    if context.language in {"en-IN", "hi-en"}:
        return _question_for(context)
    return None


def _case_precedents(context: TurnContext) -> list[dict[str, Any]]:
    """Retrieved judgments as structured data, for the degraded path.

    Without a model there is nothing to say whether a judgment applies, so these
    are offered as related decisions with the passage quoted verbatim and
    `treatment_status` untouched.

    There used to be a second filter here: keep only candidates containing one
    of the caller's own content words. It was a duplicate relevance test, and a
    worse one than the retrievers already agreed on -- on "the refusal of FIR"
    the anchor set was {refusal, fir}, so it discarded Lalita Kumari (whose
    holding uses "cognizable offence" and "section 154") and kept income-tax
    cases that happened to say "refusal". Retrieval, the focus gate and the
    model judge decide relevance; this function formats the result.
    """
    out: list[dict[str, Any]] = []
    for case in context.cases[:4]:
        raw_date = str(case.get("decision_date") or "").strip()
        out.append({
            "id": case["id"],
            "title": case.get("title"),
            "court": case.get("court"),
            "court_level": case.get("court_level"),
            "citation": case.get("citation"),
            "decision_date": None if raw_date.lower() in {"", "unknown", "none", "null"} else raw_date,
            "jurisdiction": case.get("jurisdiction"),
            "url": case.get("source_url"),
            "excerpt": _text(case.get("excerpt"), 900),
            "treatment_status": "not_checked",
        })
    return out


def _case_led_sentence(context: TurnContext) -> str:
    """One sentence a lawyer would actually say, built from a real passage.

    The degraded path used to answer "I could not find an approved source"
    while the interface below it listed a judgment that decides the exact
    question. That is honest about the statute pack and misleading about the
    call: the caller asked why the police will not register an FIR and the best
    material in the building was *Lalita Kumari*. This names it, quotes it
    exactly, and says what has not been checked -- nothing more is claimed.
    """
    if not context.cases:
        return ""
    case = context.cases[0]
    title = _text(case.get("title"), 160)
    excerpt = _text(case.get("excerpt"), 420)
    if not title or not excerpt:
        return ""
    court = _text(case.get("court"), 90)
    decided = _text(case.get("decision_date"), 40)
    # Unimported records carry the literal string "unknown"; a caller hearing
    # "the Supreme Court of India, decided unknown" is hearing a data gap
    # spoken as if it were a date.
    if decided.strip().lower() in {"unknown", "none", "null", "n/a"}:
        decided = ""
    where = f"{court}, decided {decided}" if court and decided else (court or f"decided {decided}")
    # Every sentence around the quote is in the caller's language. The quote
    # itself stays verbatim in whatever language the judgment was written in --
    # translating it would break exact-substring claim checking -- so a Telugu
    # caller hears Telugu framing around an English holding, which is what an
    # interpreter does, rather than an English paragraph from an app they were
    # speaking Telugu to.
    caveat = (
        _floor(context.language, "case_only_caveat")
        if getattr(context, "case_relevance_note", None) == "none_on_point"
        else ""
    )
    return (
        f"{title}" + (f" ({where})" if where else "") + f': "{excerpt}" '
        + _floor(context.language, "case_only_tail")
        + caveat
    )


def _grounded_only(context: TurnContext) -> dict[str, Any]:
    """Answer with retrieved passages and no generated legal conclusion.

    This is the honest degraded mode: with no model configured the app still
    shows what it actually has, instead of pretending or refusing entirely.
    """
    if not context.sources:
        # Do not ask for a state the caller already gave earlier in the call.
        key = (
            "no_source_state"
            if context.jurisdiction_basis == "national_default_location_not_established"
            else "no_source_more"
        )
        # An unclear turn has not been established as a legal question at all
        # ("what is the weather in mumbai today" routes there). Quoting a
        # judgment back as "the closest decision I have" answers a question the
        # caller did not ask, so this path asks what happened instead -- the
        # same question the route already wanted to ask.
        case_line = _case_led_sentence(context) if context.route_name != "unclear" else ""
        reply = (
            f"{_floor(context.language, 'case_only_lead')} {case_line}"
            if case_line else _floor(context.language, key)
        )
        result = _base_result(context, reply, "grounded_only", "no_supported_legal_claim")
        result["verification"] = {
            "status": "no_supported_legal_claim",
            "entailment": "not_attempted",
            "rejected_claims": 0,
            "prose_checked": True,
            "source_scope": (
                "judgment passage quoted verbatim; no statutory provision matched and nothing was generated"
                if case_line else "no approved source matched; nothing generated"
            ),
        }
        result["next_question"] = _fallback_question(context, reply)
        result["follow_up"] = result["next_question"]
        result["source_gap"] = (
            "no verified statutory provision; the closest decision is quoted instead"
            if case_line else "no approved source matched this query"
        )
        # A judgment can be relevant even when no statute passage matched; the
        # alternative was reporting four candidates in `corpus` and showing none.
        # Not for an unclear turn: those candidates were retrieved speculatively
        # before the route existed, and four cards under "what happened?" reads
        # as an answer.
        if context.route_name != "unclear":
            result["case_precedents"] = _case_precedents(context)
        return result
    passages = " ".join(
        f"From {source['authority']}: \"{_text(source['text'], 420)}\"" for source in context.sources[:2]
    )
    lead_key = "no_model_lead" if not model_configured() else "no_verified_claim_lead"
    reply = f"{_floor(context.language, lead_key)} {passages} {_floor(context.language, 'no_model_tail')}"
    result = _base_result(context, reply, "grounded_only", "no_supported_legal_claim")
    result["sources"] = [knowledge.source_evidence(source) for source in context.sources[:4]]
    result["sources"] = [{**item, "passage": _text(source["text"], 900)} for item, source in zip(result["sources"], context.sources[:4])]
    result["case_precedents"] = _case_precedents(context)
    result["verification"] = {
        "status": "retrieved_uninterpreted",
        "entailment": "not_attempted",
        "rejected_claims": 0,
        "source_scope": "verbatim passages only; no legal conclusion generated",
    }
    result["next_question"] = _fallback_question(context, reply)
    result["follow_up"] = result["next_question"]
    result["source_gap"] = "no model configured; passage retrieval only" if not model_configured() else "no approved statutory claim verified; closest passages retrieved"
    return result


def _intake_block(context: TurnContext) -> dict[str, Any]:
    """What the model needs to continue an intake like a person would.

    The checklist is the application's, not the model's: it says which single
    fact the process still needs and lets the model phrase that request in the
    caller's own language. The model cannot add, skip or reorder a step.
    """
    state = context.workflow_state or {}
    block = context.workflow or {}
    procedure = context.procedure or {}
    label = block.get("label") or procedure.get("label") or procedure.get("title") or ((context.reading or {}).get("issue") if isinstance(context.reading, dict) else "this matter")
    ask = block.get("ask") or _question_for(context) or ((context.reading or {}).get("next_question") if isinstance(context.reading, dict) else None)
    needed = block.get("needs") or ("jurisdiction" if context.jurisdiction == "IN" else "stage")
    return {
        # Advisory, not a gate. A missing slot is a gap in the notes, never a
        # reason to withhold an answer the evidence already supports.
        "advisory": True,
        "guidance": (
            "Use this as background. Answer the caller's legal question first when the "
            "supplied sources support it; ask this step's question only if it is still "
            "needed and the caller has not already answered it."
        ),
        "procedure_label": label,
        "needed_slot": needed,
        "ask": ask,
        "step": block.get("step", 1),
        "total": block.get("total", len(procedure.get("steps", [])) or 3),
        "status": state.get("status", "active"),
        "steps_are_shown_in_the_interface": bool(state.get("shown") or procedure),
        "other_matters_held": [
            {"procedure_id": item.get("procedure_id"), "issue": item.get("issue")}
            for item in context.ledger.matters if item.get("status") == "noted"
        ],
    }


def _earlier_digest(items: Any) -> str:
    """Fold compacted earlier turns into one bounded block for the models.

    Each stored entry is a single labelled line (`caller: ...`). The digest is
    the join of those lines, capped so even a very long call contributes a
    fixed maximum to context. Masking of identity numbers happens on the
    whole string at the boundary in `analyze`.
    """
    lines: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        content = _text(item.get("content"), 240)
        if content:
            lines.append(content)
    return " | ".join(lines)[:2000]


def _masked_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """The transcript as the model may see it: identifiers masked.

    A caller who reads out an account number in turn two should not have it
    replayed to a model in every later turn of the same call.
    """
    masked: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            masked.append({**item, "content": privacy.for_model(item["content"])})
        elif isinstance(item, dict):
            masked.append(item)
    return masked


def _consultation_transcript(context: TurnContext) -> str:
    """The caller's own earlier words, masked, for grounding a late fact.

    A fact the model reports may cite a turn that has scrolled out of the live
    window. It still has to be a verbatim span of something the caller said,
    and this is the same text the model was given, so a model quoting its own
    paraphrase fails the check.
    """
    lines = [
        str(item.get("content") or "")
        for item in context.history
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    return privacy.for_model("\n".join(lines))


def _open_consultation_matter(context: TurnContext, reading: understanding.Interpretation) -> None:
    """Open or hold the matter the model named inside its answer.

    The lexical pass cannot match every way a caller names a process, and a
    model that recognises one should not have to wait a turn for the card. The
    reverse is also true: naming a second process mid-intake does not take the
    floor from the one in progress -- it is held (`note_matter`) and offered
    when the first finishes.
    """
    procedure_id = reading.procedure_id
    if not procedure_id:
        return
    card = procedural.BY_ID.get(procedure_id)
    if not card:
        return
    state = context.workflow_state
    active = bool(state and state.get("status") == "active")
    if active and state and state.get("id") != procedure_id:
        if context.ledger.note_matter(procedure_id, reading.issue, context.turn):
            _stage(
                context.trace, "matter_noted", time.perf_counter(),
                procedure=procedure_id, while_active=state["id"],
            )
    else:
        candidate = workflow.for_procedure(procedure_id)
        # Same rule the lexical pass uses: a finished intake of the same matter
        # is not silently restarted, and an aborted one may be.
        restarts = (
            state is None
            or state.get("status") == "aborted"
            or procedure_id != state.get("id")
        )
        if candidate and restarts and context.route_name not in {"emergency", "human"}:
            context.workflow_state = workflow.start(candidate["id"], context.turn)
            context.workflow = workflow.public(context.workflow_state, context.ledger)
    if context.route_name not in {"emergency", "human"}:
        if reading.intent == "procedural":
            context.route["name"] = "procedural"
            context.route["procedure_id"] = procedure_id
        if not context.procedure or context.procedure.get("id") != procedure_id:
            context.procedure = procedural.public(card)


def _advance_consultation_matter(context: TurnContext) -> None:
    """Move the intake forward with the facts the answer just committed.

    The lexical pass advanced it before the model spoke, so a fact that only
    arrived in the consultation left the step open for one more turn and
    re-asked a question the caller had just answered. This is the same advance,
    run once more after the ledger changed.
    """
    state = context.workflow_state
    if not state or state.get("status") != "active":
        return
    context.ledger.resolve_matter(state["id"])
    state, events = workflow.advance(state, context.ledger, turn=context.turn)
    for event in events:
        if event.get("event") != "tool_call":
            continue
        # `get_procedure` is the only tool node the reviewed flows use, and its
        # arguments are static, so the card is a catalog lookup. Nothing here
        # can reach the network or the filesystem.
        wanted = str((event.get("arguments") or {}).get("procedure_id") or state.get("id") or "")
        chosen = procedural.BY_ID.get(wanted)
        if chosen and context.route_name not in {"emergency", "human"}:
            context.procedure = procedural.public(chosen)
            if context.route_name in {"unclear", "procedural"}:
                context.route["name"] = "procedural"
                context.route["procedure_id"] = chosen["id"]
    context.workflow_state = state
    context.workflow = workflow.public(state, context.ledger)
    _stage(
        context.trace, "workflow", time.perf_counter(), id=state["id"],
        node=state.get("node"), status=state.get("status"),
        events=[event["event"] for event in events][:6],
    )


def _apply_consultation(
    context: TurnContext, payload: dict[str, Any] | None,
) -> understanding.Interpretation | None:
    """Fold the reading carried by the reply into the turn, after validation.

    This is where the model's judgement enters session state, and every path
    in it is checked: an intent outside the closed set, a fact without a
    verbatim span, a slot the ledger does not have, a risk level below the
    lexical net, or an emergency with no grounded evidence -- each is rejected
    or capped exactly as it would have been in a separate reading call.
    """
    if not isinstance(payload, dict) or not payload:
        return None
    reading = understanding.from_consultation(
        payload,
        context.model_text or context.text,
        jurisdictions=INDIAN_JURISDICTIONS,
        asked=context.ledger.asked,
        transcript=_consultation_transcript(context),
    )
    context.reading = reading.as_dict()
    context.search_query = reading.search_query
    context.risk = reading.risk or context.risk
    context.route["risk_level"] = context.risk.get("level", "normal")
    context.route["risk_categories"] = [
        item.get("key") for item in context.risk.get("categories", [])
    ]
    if reading.emergency:
        context.route["name"] = "emergency"
        context.route["emergency"] = True
        context.route["confidence"] = max(float(context.route.get("confidence") or 0.0), 0.97)
        context.route.setdefault("reasons", []).insert(
            0, "model: emergency grounded in the caller's words"
        )
        return reading

    # The consultation has now read the turn. Its validated intent is better
    # evidence than the phrase net that had to decide before the call ran, so
    # the route is settled here rather than left reporting the earlier guess.
    # Safety is never lowered by this: `settle` refuses to leave `emergency`.
    context.route = routing.settle(context.route, reading)

    state = context.workflow_state or {}
    matter = str(state.get("id")) if state.get("status") == "active" else ""
    written = facts.apply_reading(context.ledger, reading.facts, context.turn, matter)
    if reading.issue:
        context.ledger.record("situation", reading.issue, "reader", context.turn, matter=matter)
    if matter:
        context.ledger.tag_matter(context.turn, matter)
    if written:
        _stage(context.trace, "facts_written", time.perf_counter(), slots=written, conflicts=len(context.ledger.contradictions))

    _open_consultation_matter(context, reading)
    _advance_consultation_matter(context)

    # A state the lexical pass could not read (another script, another name for
    # it) still becomes the call's jurisdiction -- for the next turn's search,
    # and for the label. The passages in hand were retrieved under the earlier
    # resolution, so the trace says so rather than implying they were local.
    if context.jurisdiction_basis == "national_default_location_not_established":
        reported = context.ledger.value("jurisdiction")
        if isinstance(reported, str) and reported in INDIAN_JURISDICTIONS:
            context.jurisdiction = reported
            context.jurisdiction_basis = "caller_location_in_message"
            _stage(context.trace, "jurisdiction_late", time.perf_counter(), code=reported)

    context.plan = {
        "issue": reading.issue,
        "jurisdiction": context.jurisdiction,
        "needs_precedents": reading.needs_precedents,
        "needs_primary_law": context.route_name in {"statute", "draft", "unclear"},
        "services": [],
        "reported_facts": [f"{fact['slot']}: {fact['value']}" for fact in reading.facts],
        "risk_flags": [
            str(item.get("key")) for item in context.risk.get("categories", [])
            if item.get("source") == "model"
        ],
    }
    _stage(
        context.trace, "consultation", time.perf_counter(),
        intent=reading.intent, confidence=round(reading.confidence, 2),
        source=reading.source, facts=written, rejected=reading.rejected[:6],
        emergency=False,
    )
    return reading


def _resolve_draft_artifact(context: TurnContext, tool_context: Any = None) -> dict[str, Any] | None:
    """Ensure a court-standard legal draft artifact is built when a document or PDF is requested."""
    prior_text = " ".join([m.get("text", "") or m.get("content", "") for m in (context.history or [])[-6:]]).lower()
    full_matter_context = f"{context.text} {prior_text}".lower()

    workflow_proc = (context.workflow_state or {}).get("procedure_id") if isinstance(context.workflow_state, dict) else None
    if not workflow_proc and isinstance(context.workflow, dict):
        workflow_proc = context.workflow.get("procedure_id")
    if not workflow_proc and context.procedure:
        workflow_proc = context.procedure.get("id")

    is_fir_or_156 = (
        "156" in full_matter_context
        or "magistrate" in full_matter_context
        or "మజిస్ట్రేట్" in full_matter_context
        or "మేజిస్ట్రేట్" in full_matter_context
        or "fir" in full_matter_context
        or "ఎఫ్ఐఆర్" in full_matter_context
        or "एफआईआर" in full_matter_context
        or "పోలీస్" in full_matter_context
        or "పోలీసులు" in full_matter_context
        or "police" in full_matter_context
        or workflow_proc == "police-fir"
        or "police" in str(context.ledger.facts).lower()
        or "refus" in str(context.ledger.facts).lower()
    )
    is_sp = "sp" in full_matter_context or "superintendent" in full_matter_context or "ఎస్పీ" in full_matter_context
    is_138 = "138" in full_matter_context or "cheque" in full_matter_context or "బౌన్స్" in full_matter_context or "చెక్" in full_matter_context
    is_rti = "rti" in full_matter_context or "సమాచార హక్కు" in full_matter_context or "right to information" in full_matter_context
    is_consumer = "consumer" in full_matter_context or "వినియోగదారు" in full_matter_context or "వినియోగదారుల" in full_matter_context
    is_debt = any(k in full_matter_context for k in [
        "promissory", "pro-note", "pronote", "ప్రోనోట్", "బాకీ", "అప్పు", "loan", "debt", "recovery", "money", "రూపాయలు"
    ])

    if is_fir_or_156:
        if is_sp and "refus" not in context.text.lower() and "court" not in full_matter_context:
            doc_type = "complaint_to_sp"
            doc_title = "Complaint to the Superintendent of Police under Section 154(3) CrPC / Section 173(4) BNSS"
            prayer = "Direct the registration of FIR and investigation by police under Section 154(3) CrPC / Section 173(4) BNSS."
        else:
            doc_type = "application_156_3"
            doc_title = "Application under Section 156(3) CrPC / Section 175(3) BNSS before the Judicial Magistrate"
            prayer = "Direct the registration of FIR and investigation by police under Section 156(3) CrPC / Section 175(3) BNSS."
    elif is_138:
        doc_type = "legal_notice_138"
        doc_title = "Statutory Demand Notice under Section 138 of the Negotiable Instruments Act, 1881"
        prayer = "Payment of the dishonoured cheque amount within 15 days of receipt of this notice."
    elif is_rti:
        doc_type = "rti_application"
        doc_title = "Application for Information under Section 6(1) of the Right to Information Act, 2005"
        prayer = "Supply the requested certified copies and information within the statutory 30-day period."
    elif is_consumer:
        doc_type = "consumer_complaint"
        doc_title = "Consumer Complaint under Section 35 of the Consumer Protection Act, 2019"
        prayer = "Refund of amount paid, replacement of defective goods, and compensation for deficiency in service."
    elif is_debt:
        doc_type = "legal_notice_debt"
        doc_title = "Legal Notice for Recovery of Money Due Under Promissory Note / Debt"
        prayer = "Demand for payment of outstanding sum with interest within 15 days of notice receipt."
    else:
        doc_type = "general_legal_draft"
        doc_title = "Formal Legal Application / Representation"
        prayer = "Appropriate statutory relief and legal redressal."

    known_facts = []
    for slot, fact_obj in context.ledger.facts.items():
        if fact_obj.value:
            known_facts.append(f"{slot.replace('_', ' ').capitalize()}: {fact_obj.value}")
    grounds = ("Recorded facts:\n" + "\n".join(f"- {f}" for f in known_facts)) if known_facts else context.text

    if tool_context is None:
        tool_context = tool_layer.ToolContext(
            text=context.model_text or context.text,
            jurisdiction=context.jurisdiction,
            ledger=context.ledger,
            workflow_state=context.workflow_state if isinstance(context.workflow_state, dict) else {},
            turn=len(context.history) + 1,
        )

    draft_res = tool_layer.execute(
        "draft_legal_document",
        {
            "document_type": doc_type,
            "title": doc_title,
            "complainant_or_applicant": "The Complainant / Applicant",
            "respondent_or_opposite_party": "Station House Officer / Accused / Opposite Party",
            "facts_and_grounds": grounds,
            "prayer": prayer,
            "jurisdiction": context.jurisdiction,
        },
        tool_context,
    )
    if draft_res.ok:
        return draft_res.data.get("artifact")
    return None


async def _generated(context: TurnContext, intake: bool = False, on_sentence: Any = None) -> dict[str, Any] | None:
    response_context = {
        "caller_message": context.model_text or context.text,
        "caller_note_on_identifiers": (
            "Identity and account numbers in this message were replaced with masked tokens "
            "before you saw it. Refer to them by kind — aadhaar, pan, account, phone — and "
            "never invent the digits."
            if context.model_text != context.text else None
        ),
        "language_hint": context.language,
        "jurisdiction": context.jurisdiction,
        "jurisdiction_label": JURISDICTION_LABELS.get(context.jurisdiction, context.jurisdiction),
        "jurisdiction_basis": context.jurisdiction_basis,
        "route": context.route,
        "retrieval_plan": {k: v for k, v in (context.plan or {}).items() if k != "history"},
        "authoritative_sources": [
            {
                "id": s["id"], "title": s["title"], "authority": s["authority"], "kind": s["kind"],
                "jurisdiction": s["jurisdiction"], "url": s["url"], "checked_at": s.get("checked_at"),
                # Long enough to hold a whole sub-section. 1,500 characters cut
                # the proviso off the rule it qualifies, which is the one part
                # a lawyer would not answer without.
                "text": s["text"][:2200],
            }
            for s in context.sources[:MAX_PROMPT_SOURCES]
        ],
        "case_precedents": [
            {
                "id": c["id"], "title": c["title"], "court": c.get("court"),
                "court_level": c.get("court_level"), "citation": c.get("citation"),
                "decision_date": c.get("decision_date"), "jurisdiction": c.get("jurisdiction"),
                "source_url": c.get("source_url"),
                # `excerpt` is what retrieval fills. Reading `text` here sent
                # every precedent to the model with an empty passage, so it saw
                # case names and had nothing to reason from -- the reason its
                # case discussion read like a summary of the titles.
                "text": _text(c.get("excerpt") or c.get("text"), 1600),
            }
            for c in context.cases[:MAX_PROMPT_CASES]
        ],
        "secondary_qa": context.secondary,
        "capabilities": SERVICE_CATALOG["capabilities"],
        # The reviewed catalogue, so the answer can name the process it is
        # describing. `procedure_id` is validated against this list; the model
        # cannot invent one.
        "procedures": [
            {"id": item["id"], "label": item["label"], "summary": item["summary"][:180]}
            for item in procedural.PROCEDURES
            if item.get("review_status") not in {"rejected", "expired"}
        ],
        "procedure": context.procedure,
        "known_facts": [fact.as_dict() for fact in context.ledger.facts.values()],
        "fact_conflicts": context.ledger.contradictions,
        "already_asked": context.ledger.asked,
        "previous_search": context.search_query or None,
        "safety": _safety_block(context),
        "validation_feedback": None,
        "history": _masked_history(context.history),
        "earlier": context.earlier or None,
        "intake": _intake_block(context) if (intake or context.workflow_state or context.procedure) else None,
    }
    started = time.perf_counter()
    tool_context = tool_layer.ToolContext(
        text=context.model_text or context.text,
        jurisdiction=context.jurisdiction,
        ledger=context.ledger,
        workflow_state=context.workflow_state if isinstance(context.workflow_state, dict) else {},
        turn=len(context.history) + 1,
    )
    tool_messages: list[dict[str, Any]] = []

    # Native tool calling. One probing attempt per turn: if the server cannot
    # accept `tools` at all, the answer falls back to a plain completion and
    # the trace records it rather than retrying on every turn.
    generated: dict[str, Any] | None = None
    if on_sentence is not None:
        # Stream the reply and let safe sentences start playing while the rest
        # is still being generated. A turn that becomes tool calls, or a
        # server that cannot stream, falls back to the ordinary call below.
        streamed: dict[str, Any] | None = None
        try:
            async for kind, payload in generate_turn_stream(response_context):
                if kind == "sentence":
                    if _early_speak_ok(str(payload)):
                        await on_sentence(str(payload))
                elif isinstance(payload, dict):
                    streamed = payload
        except Exception:  # streaming must never cost the turn
            streamed = None
        if streamed and not streamed.get("_tool_calls") and streamed.get("reply"):
            generated = streamed
            _stage(context.trace, "generate", started, ok=True, streamed=True)
    if generated is None:
        # One request, with internal retries bounded by the gateway. The old
        # fallback -- a second full call without tools when the first returned
        # nothing -- doubled the token charge of every failed turn, and it
        # could not succeed where the gateway's own 400 handling already
        # retried the call without tool definitions.
        generated = await generate_turn(
            response_context, tools=tool_layer.TOOL_SPECS, extra_messages=tool_messages,
        )

    # One tool round, not three. The seeded evidence is already in the prompt,
    # so a lookup is the exception -- the passage that was not retrieved -- and
    # one round is enough to fetch it and answer. Every extra round is another
    # full generation charged against the same minute, which is exactly how a
    # voice consult ends up throttled. The per-tool budgets still bound this.
    for round_index in range(1):
        calls = (generated or {}).get("_tool_calls") or _json_tool_requests(generated)
        if not calls:
            break
        assistant_calls: list[dict[str, Any]] = []
        tool_messages_out: list[dict[str, Any]] = []
        for index, call in enumerate(calls[:5]):
            result = tool_layer.execute(str(call.get("name")), call.get("arguments"), tool_context)
            context.tool_audit.append(result.as_dict())
            context.sources = _merge_sources(context.sources, result.sources)
            if result.ok and result.name == "draft_legal_document":
                context.tool_artifact = result.data.get("artifact") or {
                    "title": result.data.get("title"),
                    "type": result.data.get("document_type") or result.data.get("type"),
                    "content": result.data.get("content"),
                    "intended_for_submission": True,
                    "warning": result.data.get("warning"),
                }
            call_id = str(call.get("id") or f"call_{index}")
            assistant_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(call.get("name")),
                    "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                },
            })
            tool_messages_out.append({
                "role": "tool", "tool_call_id": call_id,
                "name": str(call.get("name")), "content": result.as_message(),
            })
        tool_messages = [*tool_messages, {"role": "assistant", "content": "", "tool_calls": assistant_calls}, *tool_messages_out]
        _stage(
            context.trace, "tool_calls", started, round=round_index + 1,
            tools=[str(call.get("name")) for call in calls[:5]],
            ok=[item["ok"] for item in context.tool_audit[-len(assistant_calls):]],
        )
        generated = await generate_turn(response_context, tools=tool_layer.TOOL_SPECS, extra_messages=tool_messages)

    _stage(context.trace, "generate", started, ok=bool(generated), tools=len(context.tool_audit))
    if not generated:
        return None

    # The reading the answer carries becomes session state here, and nowhere
    # earlier: facts, risk, the matter it names and the query it would search.
    reading = _apply_consultation(context, generated)
    if reading is not None and reading.emergency:
        # The model saw danger the lexical net did not name, and it pointed at
        # the caller's own words. Safety takes the turn: the answer it just
        # wrote is not spoken while someone is in danger.
        _stage(context.trace, "safety_override", time.perf_counter(), source="consultation")
        return _emergency_reply(context)

    async def _checked(payload: dict[str, Any] | None) -> tuple[list[dict[str, str]], list[str], bool]:
        """Claims that survive both the quote test and the entailment test.

        Returns (claims, rejections, entailment_ran).
        """
        found, bad = _claims((payload or {}).get("legal_claims"), context.sources)
        if not found:
            return found, bad, False
        verdicts = await model_gateway.verify_claims(found)
        if verdicts is None:
            return found, bad, False
        failed = {index for index, supported in verdicts.items() if not supported}
        if failed:
            bad = [*bad, "entailment_rejected:" + ",".join(str(i) for i in sorted(failed))]
            found = [claim for index, claim in enumerate(found) if index not in failed]
        return found, bad, True

    claims, rejected, entailment_ran = await _checked(generated)
    if rejected:
        # No repair round. A second full generation was the worst latency spike
        # in the product, doubled the token charge of the turn that had just hit
        # a quota wall, and produced nothing the caller does not already get
        # from the deterministic answer: the passages that were actually
        # retrieved, quoted exactly, labelled as uninterpreted. Fail closed.
        _stage(
            context.trace, "downgrade", time.perf_counter(),
            mode="claim_verification", rejected=rejected[:6],
        )
        return _grounded_only(context)

    case_ids = {case["id"] for case in context.cases}
    case_context: list[dict[str, str]] = []
    raw_cases = generated.get("case_context")
    if isinstance(raw_cases, list):
        for item in raw_cases[:8]:
            if not isinstance(item, dict) or item.get("source_id") not in case_ids:
                continue
            case_context.append({
                "source_id": item["source_id"],
                "why_similar": _text(item.get("why_similar"), 600),
                "limits": _text(item.get("limits"), 600),
            })

    reply = _text(generated.get("reply"), 12000)
    if not reply:
        return None

    # The caller hears `reply`, not `legal_claims`. A number in the spoken
    # sentence that no verified quote or the caller introduced is the same
    # unsupported legal proposition in a different field, so it gets the same
    # treatment: one repair, then the deterministic grounded reply.
    #
    # An intake turn is exempt: its job is to ask the one missing fact, and a
    # question about the caller's own timeline ("was that 30 days ago?") is not
    # a legal assertion. Downgrading there would replace the conversation with
    # passages, which is a worse failure than the one being guarded against.
    scope_text = _grounded_prose_scope(context)
    # Checked on intake turns too. The consultation answers the caller's
    # question before it asks the next one, so an intake reply can carry a
    # section number; the earlier blanket exemption let that through. Questions
    # are excluded inside `_prose_gaps`, which is what the exemption was for.
    gaps = _prose_gaps(reply, claims, scope_text)
    if gaps:
        # Same rule, same reason: a number no verified quote carries is not
        # spoken, and it is not retried at the cost of another full turn.
        _stage(context.trace, "prose_guard", time.perf_counter(), unsupported=gaps[:6])
        _stage(context.trace, "downgrade", time.perf_counter(), mode="prose_guard")
        return _grounded_only(context)

    language_out = _normalize_language(generated.get("language"), context.language)
    result = _base_result(
        context,
        reply,
        "intake" if intake else "model",
        "intake_turn" if intake else ("source_linked" if claims else "no_supported_legal_claim"),
    )
    result["language"] = language_out
    result["tts_language"] = language_layer.tts_language(language_out)
    result["language_label"] = LANGUAGE_LABELS.get(language_out, language_out)

    source_by_id = {source["id"]: source for source in context.sources}
    result["claims"] = [
        {**claim, "source_title": source_by_id[claim["source_id"]]["title"],
         "source_url": source_by_id[claim["source_id"]]["url"],
         "authority": source_by_id[claim["source_id"]]["authority"]}
        for claim in claims
    ]
    used = {claim["source_id"] for claim in claims}
    result["sources"] = [knowledge.source_evidence(s) for s in context.sources if s["id"] in used]
    for item in case_context:
        case = next((c for c in context.cases if c["id"] == item["source_id"]), None)
        if case:
            result["sources"].append({
                "id": case["id"], "title": case["title"], "authority": case["court"],
                "url": case["source_url"], "jurisdiction": case["jurisdiction"],
                "kind": "judgment_dataset_record", "checked_at": None,
                "citation": case.get("citation"), "decision_date": case.get("decision_date"),
                "treatment_status": "not_checked",
            })
    result["case_context"] = case_context
    result["verification"] = {
        "status": "exact_source_quote_pass" if claims else "no_legal_claims_emitted",
        "entailment": "model_checked" if entailment_ran else ("not_reviewed" if claims else "not_attempted"),
        "rejected_claims": len(rejected),
        "prose_checked": True,
        "source_scope": "retrieved source text only; exact quote matched; legal interpretation not reviewed",
    }
    result["corpus"].update({"legal_source_candidates": len(context.sources), "case_judgment_candidates": len(context.cases)})

    allowed = {item["id"] for item in SERVICE_CATALOG["capabilities"]}
    services = [s for s in generated.get("services_used", []) if isinstance(s, str) and s in allowed] if isinstance(generated.get("services_used"), list) else []
    if context.requested_service and context.requested_service not in services:
        services.append(context.requested_service)
    result["services_used"] = services

    artifact = generated.get("artifact") if isinstance(generated.get("artifact"), dict) else None
    if not artifact and getattr(context, "tool_artifact", None):
        artifact = context.tool_artifact
    if not artifact and (context.route.get("name") == "draft" or any(cue in context.text.lower() for cue in [
        "draft", "prepare", "application", "write", "make", "pdf",
        "దరఖాస్తు", "అప్లికేషన్", "పిడిఎఫ్", "పీడీఎఫ్", "आवेदन", "प्रारूप", "மனு", "அರ್ಜಿ"
    ])):
        artifact = _resolve_draft_artifact(context, tool_context)

    if artifact:
        from .pdf_generator import normalize_legal_draft_text
        raw_c = _text(artifact.get("content"), 15000)
        norm_c = normalize_legal_draft_text(raw_c) if raw_c else ""
        warn = _text(artifact.get("warning"), 1200) or "Statutory Notice: This draft must be reviewed, verified, and signed in consultation with an enrolled advocate before filing in court."
        artifact = {
            "type": _text(artifact.get("type"), 60) or "summary",
            "title": _text(artifact.get("title"), 180) or "Prepared draft",
            "content": norm_c,
            "intended_for_submission": artifact.get("intended_for_submission") is True,
            "warning": warn,
        }
        if not artifact["content"]:
            artifact = None
        identifiers = privacy.find(artifact["content"]) if artifact else []
        if artifact and identifiers:
            # A draft that carries an Aadhaar or account number should say so
            # rather than quietly text it to a court or a landlord.
            artifact["identifiers"] = identifiers
            artifact["warning"] = (artifact.get("warning") or "") + " This draft contains an identity or account number; share it only where it is required."
    if artifact and _safety_block(context)["blocks_drafting"]:
        # A document prepared while the caller is in danger can delay the one
        # step that matters. Withholding it is the safer failure.
        artifact = None
    result["artifact"] = artifact
    if artifact and ("IN THE COURT" in reply.upper() or "BEFORE THE" in reply.upper() or reply.count("\n") > 10):
        lang = context.language
        if lang.startswith("te"):
            reply = "మీరు అడిగిన దరఖాస్తును నేను PDF ఫార్మాట్‌లో సిద్ధం చేశాను. మీరు క్రింద ఉన్న బటన్ ద్వారా కోర్టు PDFని డౌన్‌లోడ్ చేసుకోవచ్చు మరియు అవసరమైతే సవరించవచ్చు."
        elif lang.startswith("hi"):
            reply = "मैंने आपका कानूनी आवेदन पत्र तैयार कर दिया है। आप नीचे दिए गए बटन से कोर्ट-फॉर्मेट पीडीएफ डाउनलोड कर सकते हैं।"
        else:
            reply = "I have prepared the legal draft for you. You can review the draft below and download the court-formatted PDF directly."
        result["response"] = reply
        result["spoken"] = reply

    risk = []
    for source in (context.plan or {}).get("risk_flags", []) if isinstance((context.plan or {}).get("risk_flags"), list) else []:
        if isinstance(source, str):
            risk.append(_text(source, 100))
    for source in generated.get("risk_flags", []) if isinstance(generated.get("risk_flags"), list) else []:
        if isinstance(source, str):
            risk.append(_text(source, 100))
    # Deterministic triage flags are added last and cannot be removed by the
    # model; the model may only add its own.
    for item in context.risk.get("categories", []):
        risk.append(f"triage:{item.get('key')}")
    if context.ledger.contradictions:
        risk.append("caller_stated_conflicting_values")
    result["risk_flags"] = list(dict.fromkeys(risk))[:12]
    result["source_gap"] = _text(generated.get("source_gap"), 700) or None
    # The reply already ends with the question in a conversational turn; a
    # separate follow-up chip asking the same thing again reads as a system
    # talking over itself. When it does not ask, the model's own question is
    # preferred to the intake's template: it is the one formed from this story.
    question = (reading.next_question if reading is not None else None) or _question_for(context)
    result["next_question"] = None if "?" in reply else question
    result["follow_up"] = result["next_question"]

    requested_human = (context.plan or {}).get("requested_human") is True or context.route.get("wants_human")
    submission = bool(artifact and artifact["intended_for_submission"])
    result["human_connection"] = {
        "offered": bool(requested_human or submission),
        "connected": False,
        "requires_user_choice": True,
        "language": language_out,
        "reason": "submission_document" if submission else "caller_requested" if requested_human else None,
        "handoff": {"kind": "legal_aid", "label": "Free legal aid (NALSA / DLSA)", "url": "https://nalsa.gov.in/"} if (requested_human or submission) else None,
    }
    return result


def _settle_procedure_display(context: TurnContext, result: dict[str, Any]) -> None:
    """Show the reviewed steps once per intake, and again when it finishes.

    Reprinting the same card under every answer is what made the exchange read
    as a form. The card stays in the transcript, and the interface keeps the
    current step visible between turns.
    """
    state = context.workflow_state
    procedure = result.get("procedure")
    if not isinstance(state, dict) or not procedure:
        return
    own_matter = procedure.get("id") == (context.workflow or {}).get("procedure_id")
    if own_matter and state.get("status") == "active" and state.get("shown"):
        result["procedure"] = None
    elif own_matter:
        state["shown"] = True


def _floor_reply(context: TurnContext) -> dict[str, Any]:
    """What the app says when no model answered: never a pretence of one.

    A reviewed procedure is still the best thing it has for a process turn; a
    retrieved passage is the best thing it has for everything else. Both are
    labelled with the mode that produced them, so the caller and the trace can
    tell a generated consultation from a deterministic fallback.
    """
    if context.route_name == "draft" or any(cue in context.text.lower() for cue in [
        "draft", "prepare", "application", "write", "make", "pdf",
        "దరఖాస్తు", "అప్లికేషన్", "పిడిఎఫ్", "పీడీఎఫ్", "आवेदन", "प्रारूप", "மனு", "அರ್ಜಿ"
    ]):
        artifact = _resolve_draft_artifact(context)
        if artifact:
            from .pdf_generator import normalize_legal_draft_text
            if artifact.get("content"):
                artifact["content"] = normalize_legal_draft_text(artifact["content"])
            lang = context.language
            if lang.startswith("te"):
                reply = "మీరు అడిగిన దరఖాస్తును నేను PDF ఫార్మాట్‌లో సిద్ధం చేశాను. మీరు క్రింద ఉన్న బటన్ ద్వారా కోర్టు PDFని డౌన్‌లోడ్ చేసుకోవచ్చు."
            elif lang.startswith("hi"):
                reply = "मैंने आपका कानूनी आवेदन पत्र तैयार कर दिया है। आप नीचे दिए गए बटन से कोर्ट-फॉर्मेट पीडीएफ डाउनलोड कर सकते हैं।"
            else:
                reply = "I have prepared the legal draft for you. You can review the draft below and download the court-formatted PDF directly."
            result = _base_result(context, reply, "draft", "source_linked")
            result["artifact"] = artifact
            result["spoken"] = reply
            result["source_gap"] = None
            return result

    if context.procedure and context.route_name in {"procedural", "unclear"}:
        # `mode` stays `procedure`: it is the API's name for "answered from the
        # reviewed file, no model". Whether the server had no model or the model
        # failed is said in `source_gap` instead, which is where a caller and a
        # reviewer actually look for it.
        result = _procedure_reply(context)
    else:
        result = _grounded_only(context)
    result["source_gap"] = result.get("source_gap") or (
        "no model configured on this server"
        if not model_configured()
        else "the configured model did not return a valid structured answer"
    )
    return result


async def compose(context: TurnContext, allow_generation: bool = True, on_sentence: Any = None) -> dict[str, Any]:
    """One consultation per turn, with the deterministic floors underneath it.

    There is no second pipeline for intake turns. A procedural turn, an intake
    turn and an ordinary question all run the same single consultation; the
    only differences are the status it reports and whether the opening
    sentence may be spoken early. The reviewed procedure is a *floor* -- what
    the app says if generation is unavailable -- never a competing answer.
    """
    state = context.workflow_state or {}
    # Intake framing only when *this* turn is about the intake's own matter; an
    # unrelated process question is answered on its own terms.
    same_matter = bool(context.procedure) and context.procedure.get("id") == (context.workflow or {}).get("procedure_id")
    in_intake = same_matter and state.get("status") in {"active", "completed"}
    if context.route_name == "emergency":
        result = _emergency_reply(context)
    elif context.route_name == "human":
        result = _handoff_reply(context)
    elif not allow_generation or not model_configured():
        result = _floor_reply(context)
    elif in_intake or (context.route_name == "procedural" and context.procedure):
        # A conversational turn: acknowledge what was said, answer the question
        # the caller slipped in, and ask the one fact that still changes the
        # route -- in their language. Early speech is allowed here because the
        # first sentence is acknowledgement, not a legal proposition.
        result = await _generated(context, intake=True, on_sentence=on_sentence)
        result = result or _floor_reply(context)
    else:
        # No early speech for a legal answer: a sentence of a statute answer is
        # a legal proposition, and those are only spoken after claim
        # validation. Interruption-free silence beats an unverified rule.
        result = await _generated(context)
        result = result or _floor_reply(context)
    _settle_procedure_display(context, result)
    # The ledger is finalised here, after any follow-up question has been
    # recorded as asked, so the next turn does not repeat it.
    result.setdefault("conversation_state", {})["ledger"] = context.ledger.as_dict()
    result["conversation_state"]["workflow"] = context.workflow_state
    return result


async def answer_turn(
    text: str,
    language: str = "auto",
    jurisdiction: str = "auto",
    service: str | None = None,
    history: list[dict[str, str]] | None = None,
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if history is None and session and session.get("history"):
        history = session["history"]
    context = await analyze(text, language, jurisdiction, service, history, session)
    routed = time.perf_counter()
    result = await compose(context)
    result["timing"] = {
        "route_ms": round((routed - started) * 1000),
        "first_speech_ms": None,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }
    return result


async def compose_stream(
    context: TurnContext,
    allow_generation: bool = True,
) -> AsyncIterator[tuple[str, Any]]:
    """Compose the turn, yielding speech as it becomes safe to speak.

    `compose` runs as a task while completed sentences arrive through a queue,
    so the caller hears the opening of a reply at the provider's first-sentence
    time rather than at the end of the whole object. The final event is still
    the fully validated result, which is what the transcript and the citations
    are built from.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_sentence(sentence: str) -> None:
        await queue.put(sentence)

    task = asyncio.create_task(compose(context, allow_generation, on_sentence=on_sentence))
    while True:
        if task.done() and queue.empty():
            break
        if queue.empty():
            await asyncio.wait({task}, timeout=0.05)
            continue
        yield "speak", {"text": await queue.get()}
    yield "final", task.result()


async def answer_stream(
    text: str,
    language: str = "auto",
    jurisdiction: str = "auto",
    service: str | None = None,
    history: list[dict[str, str]] | None = None,
    session: dict[str, Any] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield speech and stage events, then the final result.

    The acknowledgement is the first event, emitted before the reader runs.
    A caller should hear the line, not silence: at a 350 ms provider TTFT the
    old ordering left roughly 1.7 s of dead air before anything was said, and
    no stage improvement changes that as cheaply as speaking first.
    """
    started = time.perf_counter()
    first_speech_ms: int | None = None
    # The acknowledgement is chosen from the caller's own words, not from the
    # language selector: it has to be right before any model has run.
    spoken_language, _ = language_layer.resolve(text, override=language)
    if not triage.is_emergency(triage.assess(text)):
        yield "ack", {"text": ACK.get(spoken_language, ACK["en-IN"]), "language": spoken_language, "at": 0}
    context = await analyze(text, language, jurisdiction, service, history, session)
    routed_ms = round((time.perf_counter() - started) * 1000)
    yield "stage", {"stage": "routed", "route": context.route}
    if context.route_name == "procedural" and context.procedure:
        yield "stage", {"stage": "procedure", "procedure": {"id": context.procedure["id"], "label": context.procedure["label"], "steps": len(context.procedure["steps"])}}
    elif context.route_name not in {"emergency", "human"}:
        yield "stage", {"stage": "evidence", "sources": len(context.sources), "cases": len(context.cases)}
    yield "stage", {"stage": "composing"}
    async for kind, data in compose_stream(context, allow_generation=True):
        if kind == "speak" and first_speech_ms is None:
            first_speech_ms = round((time.perf_counter() - started) * 1000)
        if kind == "final":
            # Published per turn, the way a voice platform reports it: route,
            # first audible speech, and full answer. A median without the
            # stage split cannot be acted on.
            data["timing"] = {
                "route_ms": routed_ms,
                "first_speech_ms": first_speech_ms,
                "total_ms": round((time.perf_counter() - started) * 1000),
            }
        yield kind, data
