# # ============================================================
# # FILE    : generate_emotion_v4.py
# # PURPOSE : Phase-4 GOLD EMOTION ENGINE (7-Class, Hinglish)
# # INDUSTRY: LOGISTICS + BANKING
# # ============================================================

# import pandas as pd

# # ------------------------------------------------------------
# # PATHS
# # ------------------------------------------------------------

# INPUT_PATH  = r"D:/bert_data/logistics data/train+intent+aspects+aspects_sentiment.csv"
# OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspects+aspects_sentiment+emotions.csv"

# # ------------------------------------------------------------
# # KEYWORD BANKS (ENGLISH + HINGLISH)
# # ------------------------------------------------------------

# VERY_ANGRY_KW = [
#     # English
#     "fraud", "cheated", "scam", "court", "legal",
#     "harassment", "money stolen", "refund fraud",
#     "police", "fir", "legal action", "lawsuit",

#     # Hinglish
#     "dhokha", "cheating", "scammed",
#     "paise chori", "paise gayab",
#     "court case", "legal case",
#     "police complaint", "fir karunga"
# ]

# ANGRY_KW = [
#     "angry", "furious", "complaint",
#     "bad service", "poor service",
#     "rude", "refund pending",
#     "lost", "damaged", "wrong delivery",
#     "unacceptable",

#     # Hinglish
#     "gussa", "bahut gussa",
#     "bahut bura service",
#     "bakwaas service",
#     "refund nahi mila",
#     "galat delivery",
#     "rude behaviour"
# ]

# FRUSTRATED_KW = [
#     "waiting", "delay", "pending",
#     "no update", "follow up",
#     "concern", "confused",
#     "still waiting",

#     # Hinglish
#     "kab milega", "kitna late",
#     "abhi tak nahi",
#     "wait kar raha",
#     "response nahi",
#     "samajh nahi aa raha"
# ]

# FEAR_KW = [
#     "threat", "unsafe", "security risk",
#     "account hacked", "data leak",
#     "privacy issue", "fear",

#     # Hinglish
#     "darr lag raha",
#     "unsafe lag raha",
#     "security issue",
#     "account hack ho gaya"
# ]

# SATISFIED_KW = [
#     "good", "great", "excellent",
#     "happy", "satisfied",
#     "thanks", "thank you",
#     "resolved", "well done",
#     "smooth",

#     # Hinglish
#     "achha", "bahut achha",
#     "badiya", "shandaar",
#     "problem solve ho gaya",
#     "satisfied hoon"
# ]

# NEUTRAL_KW = [
#     "ok", "fine", "average",
#     "normal", "information",
#     "query", "status",
#     "details",

#     # Hinglish
#     "theek hai",
#     "sirf info chahiye",
#     "details batao"
# ]

# # ------------------------------------------------------------
# # EMOTION ASSIGNMENT LOGIC (ENTERPRISE SAFE)
# # ------------------------------------------------------------

# def assign_emotion(text, sentiment, aspect, aspect_sentiment):
#     t = str(text).lower()

#     # 6 = FEAR / THREAT
#     if any(k in t for k in FEAR_KW):
#         return 6

#     # 5 = VERY ANGRY (Legal / Fraud / Loss)
#     if any(k in t for k in VERY_ANGRY_KW):
#         return 5

#     if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
#         return 5

#     # 4 = ANGRY
#     if any(k in t for k in ANGRY_KW):
#         return 4

#     if sentiment == 0 and aspect_sentiment == 0:
#         return 4

#     # 3 = FRUSTRATED
#     if any(k in t for k in FRUSTRATED_KW) and sentiment != 2:
#         return 3

#     # 1 = SATISFIED
#     if any(k in t for k in SATISFIED_KW):
#         return 1

#     if sentiment == 2 and aspect_sentiment == 2:
#         return 1

#     # 2 = NEUTRAL
#     if any(k in t for k in NEUTRAL_KW):
#         return 2

#     # 0 = CALM (DEFAULT SAFE)
#     return 0


# # ------------------------------------------------------------
# # APPLY TO DATASET
# # ------------------------------------------------------------

# df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

# required_cols = ["text", "sentiment", "primary_aspect", "aspect_sentiment"]
# for c in required_cols:
#     if c not in df.columns:
#         raise ValueError(f"Missing column: {c}")

# df["emotion"] = df.apply(
#     lambda r: assign_emotion(
#         r["text"],
#         int(r["sentiment"]),
#         int(r["primary_aspect"]),
#         int(r["aspect_sentiment"])
#     ),
#     axis=1
# )

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Emotion v4 (7-class, Hinglish) generated successfully")
# print(df["emotion"].value_counts())
# ============================================================
# FILE    : generate_emotion_v5.py
# PURPOSE : Phase-4 GOLD EMOTION ENGINE (Regex + Keywords)
# INDUSTRY: LOGISTICS + BANKING
# ============================================================

import pandas as pd
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/data logistics new/train+intent+aspect.csv"
OUTPUT_PATH = r"D:/bert_data/data logistics new/train+intent+aspects+emotions.csv"

# ------------------------------------------------------------
# NORMALIZER
# ------------------------------------------------------------

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def regex_match(text, pattern):
    return re.search(pattern, text) is not None

def contains_any(text, keywords):
    return any(k in text for k in keywords)

# ------------------------------------------------------------
# REGEX PATTERNS
# ------------------------------------------------------------

PATTERN_FRAUD = r"(fraud|scam|cheat|legal|court|police|fir)"
PATTERN_ANGRY = r"(rude|misbehave|bad service|worst service|unacceptable)"
PATTERN_DELAY = r"(waiting|delay|pending|late delivery)"
PATTERN_FEAR = r"(unsafe|security risk|data leak|hack)"
PATTERN_SATISFIED = r"(thank|thanks|good service|great service|excellent)"
PATTERN_NEUTRAL = r"(ok|fine|status|information|details)"

# ------------------------------------------------------------
# KEYWORD BANKS
# ------------------------------------------------------------

VERY_ANGRY_KW = [
    "fraud","cheated","scam","court","legal",
    "harassment","money stolen","refund fraud",
    "police","fir","legal action","lawsuit",
    "dhokha","paise chori","paise gayab",
    "court case","legal case","police complaint"
]

ANGRY_KW = [
    "angry","furious","complaint",
    "bad service","poor service",
    "rude","refund pending","lost",
    "damaged","wrong delivery","unacceptable",
    "gussa","bahut gussa","bakwaas service",
    "galat delivery","rude behaviour"
]

FRUSTRATED_KW = [
    "waiting","delay","pending","no update",
    "follow up","confused","still waiting",
    "kab milega","kitna late","abhi tak nahi",
    "wait kar raha","response nahi"
]

FEAR_KW = [
    "threat","unsafe","security risk",
    "account hacked","data leak","privacy issue",
    "fear","darr lag raha","unsafe lag raha",
    "security issue","account hack ho gaya"
]

SATISFIED_KW = [
    "good","great","excellent",
    "happy","satisfied","thanks",
    "thank you","resolved","well done",
    "smooth","achha","bahut achha",
    "badiya","shandaar","problem solve ho gaya"
]

NEUTRAL_KW = [
    "ok","fine","average","normal",
    "information","query","status",
    "details","theek hai","details batao"
]

# ------------------------------------------------------------
# EMOTION ASSIGNMENT
# ------------------------------------------------------------

def assign_emotion(text, sentiment, aspect, aspect_sentiment):

    t = normalize(text)

    # --------------------------------------------------------
    # FEAR
    # --------------------------------------------------------
    if regex_match(t, PATTERN_FEAR) or contains_any(t, FEAR_KW):
        return 6

    # --------------------------------------------------------
    # VERY ANGRY
    # --------------------------------------------------------
    if regex_match(t, PATTERN_FRAUD) or contains_any(t, VERY_ANGRY_KW):
        return 5

    if sentiment == 0 and aspect in [2,7] and aspect_sentiment == 0:
        return 5

    # --------------------------------------------------------
    # ANGRY
    # --------------------------------------------------------
    if regex_match(t, PATTERN_ANGRY) or contains_any(t, ANGRY_KW):
        return 4

    if sentiment == 0 and aspect_sentiment == 0:
        return 4

    # --------------------------------------------------------
    # FRUSTRATED
    # --------------------------------------------------------
    if regex_match(t, PATTERN_DELAY) or contains_any(t, FRUSTRATED_KW):
        if sentiment != 2:
            return 3

    # --------------------------------------------------------
    # SATISFIED
    # --------------------------------------------------------
    if regex_match(t, PATTERN_SATISFIED) or contains_any(t, SATISFIED_KW):
        return 1

    if sentiment == 2 and aspect_sentiment == 2:
        return 1

    # --------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------
    if regex_match(t, PATTERN_NEUTRAL) or contains_any(t, NEUTRAL_KW):
        return 2

    # --------------------------------------------------------
    # CALM DEFAULT
    # --------------------------------------------------------
    return 0

# ------------------------------------------------------------
# APPLY
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

required_cols = ["text","sentiment","primary_aspect","aspect_sentiment"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["emotion"] = df.apply(
    lambda r: assign_emotion(
        r["text"],
        int(r["sentiment"]),
        int(r["primary_aspect"]),
        int(r["aspect_sentiment"])
    ),
    axis=1
)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Emotion v5 generated successfully")
print(df["emotion"].value_counts())