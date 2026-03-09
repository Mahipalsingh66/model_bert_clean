# # ============================================================
# # FILE    : banking_emotion_engine.py
# # PURPOSE : Banking Emotion Gold Label Generator (7-Class)
# # AUTHOR  : Locked for Training Data Generation
# #
# # INPUT   : text, sentiment, primary_aspect, aspect_sentiment
# # OUTPUT  : + emotion
# #
# # EMOTION LABELS:
# # 0 = Angry
# # 1 = Frustrated
# # 2 = Fear
# # 3 = Sad
# # 4 = Neutral
# # 5 = Happy
# # 6 = Satisfied
# # ============================================================

# import pandas as pd

# # ------------------------------------------------------------
# # PATHS (UPDATE IF REQUIRED)
# # ------------------------------------------------------------

# INPUT_PATH  = r"D:/bert_data/Banking_data/bank_train_with_aspect+aspect sentiment.csv"
# OUTPUT_PATH = r"D:/bert_data/Banking_data/bank_train_with_aspect+aspect sentiment+emotions.csv"

# # ------------------------------------------------------------
# # EMOTION KEYWORD BANKS (ENGLISH + HINGLISH)
# # ------------------------------------------------------------

# ANGRY_KW = [
#     "angry", "furious", "outraged", "irritated", "annoyed",
#     "worst", "pathetic", "horrible", "unacceptable",
#     "never again", "shameful", "disgusting",
#     "harassment", "harassed",
#     "rude", "misbehaved", "abused", "threatened",
#     "legal action", "consumer court", "complaint karunga",
#     "bahut gussa", "gussa aa raha hai",
#     "badtameez", "bakwaas", "ab bas ho gaya",
#     "case kar dunga", "legal notice bhejunga"
# ]

# FRUSTRATED_KW = [
#     "frustrated", "fed up", "tired of this",
#     "again and again", "multiple times",
#     "still waiting", "no resolution",
#     "issue not resolved", "pending since",
#     "long time", "why always me",
#     "pareshan ho gaya", "thak gaya",
#     "baar baar problem", "kab tak",
#     "abhi tak solve nahi hua",
#     "roz ka problem"
# ]

# FEAR_KW = [
#     "fraud", "scam", "hacked",
#     "unauthorized", "security issue",
#     "account hacked",
#     "money stolen", "funds stolen",
#     "unsafe", "risk", "phishing",
#     "otp fraud",
#     "fraud ho gaya", "scam hua",
#     "paise chori ho gaye",
#     "dar lag raha hai",
#     "account hack ho gaya"
# ]

# SAD_KW = [
#     "sad", "very sad",
#     "disappointed", "disappointing",
#     "unhappy", "upset",
#     "regret", "let down",
#     "dukhi hoon", "nirash hoon",
#     "mann kharab ho gaya",
#     "afsos hua"
# ]

# HAPPY_KW = [
#     "happy", "very happy",
#     "great", "awesome",
#     "excellent", "amazing",
#     "loved it", "fantastic",
#     "bahut khush", "khushi hui",
#     "maza aa gaya", "badiya laga"
# ]

# SATISFIED_KW = [
#     "satisfied", "issue resolved",
#     "problem solved", "resolved now",
#     "handled well", "good service",
#     "smooth experience",
#     "thank you", "thanks",
#     "satisfied hoon",
#     "issue solve ho gaya",
#     "problem theek ho gaya",
#     "acha service mila",
#     "shukriya"
# ]

# NEUTRAL_KW = [
#     "ok", "fine", "average",
#     "normal", "as expected",
#     "query", "question",
#     "information", "details",
#     "please confirm",
#     "theek hai",
#     "jaankari chahiye",
#     "confirm karna hai"
# ]

# # ------------------------------------------------------------
# # HIGH-RISK BANKING ASPECTS (FOR FALLBACK)
# # ------------------------------------------------------------

# HIGH_RISK_ASPECTS = {0, 1, 2, 7, 8, 10, 12}

# # ------------------------------------------------------------
# # EMOTION ASSIGNMENT (STRICT PRIORITY)
# # ------------------------------------------------------------

# def assign_banking_emotion(text, sentiment, aspect_sentiment, aspect):
#     if not isinstance(text, str) or not text.strip():
#         return 4  # Neutral

#     t = text.lower()

#     # Priority order is CRITICAL
#     if any(k in t for k in FEAR_KW):
#         return 2  # Fear

#     if any(k in t for k in ANGRY_KW):
#         return 0  # Angry

#     if any(k in t for k in FRUSTRATED_KW):
#         return 1  # Frustrated

#     if any(k in t for k in SAD_KW):
#         return 3  # Sad

#     if any(k in t for k in HAPPY_KW):
#         return 5  # Happy

#     if any(k in t for k in SATISFIED_KW):
#         return 6  # Satisfied

#     if any(k in t for k in NEUTRAL_KW):
#         return 4  # Neutral

#     # ---------------- FALLBACK LOGIC ----------------

#     if sentiment == 0 and aspect in HIGH_RISK_ASPECTS:
#         return 1  # Frustrated

#     if sentiment == 2 and aspect_sentiment == 2:
#         return 6  # Satisfied

#     if sentiment == 1:
#         return 4  # Neutral

#     return 4  # Neutral

# # ------------------------------------------------------------
# # APPLY TO DATASET
# # ------------------------------------------------------------

# df = pd.read_csv(
#     INPUT_PATH,
#     encoding="latin1",
#     engine="python",
#     on_bad_lines="warn"
# )

# df.columns = df.columns.str.strip().str.lower()

# required_cols = ["text", "sentiment", "primary_aspect", "aspect_sentiment"]
# for c in required_cols:
#     if c not in df.columns:
#         raise ValueError(f"Missing required column: {c}")

# for col in ["sentiment", "primary_aspect", "aspect_sentiment"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df = df.dropna(subset=["sentiment", "primary_aspect", "aspect_sentiment"])
# df[["sentiment", "primary_aspect", "aspect_sentiment"]] = (
#     df[["sentiment", "primary_aspect", "aspect_sentiment"]].astype(int)
# )

# df["emotion"] = df.apply(
#     lambda x: assign_banking_emotion(
#         x["text"],
#         x["sentiment"],
#         x["aspect_sentiment"],
#         x["primary_aspect"]
#     ),
#     axis=1
# )

# # ------------------------------------------------------------
# # SAVE OUTPUT
# # ------------------------------------------------------------

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Banking emotion labels generated successfully")
# print("Input :", INPUT_PATH)
# print("Output:", OUTPUT_PATH)

# print("\n--- Emotion Distribution ---")
# print(df["emotion"].value_counts())

# ============================================================
# FILE    : logistics_emotion_engine_v5.py
# PURPOSE : Logistics Emotion Gold Label Generator (Banking Style)
# INDUSTRY: LOGISTICS
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/logistics data/train+intent+aspect_v4.csv"
OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspects_v4+emotion.csv"


# ------------------------------------------------------------
# KEYWORD BANKS (ENGLISH + HINGLISH)
# ------------------------------------------------------------

ANGRY_KW = [
    "worst","pathetic","ridiculous","unacceptable",
    "rude","abusive","very bad","bad service",
    "poor service","never again","complaint",

    # Hinglish
    "bakwas","bakwaas","bahut bura",
    "gussa","bahut gussa",
    "bekaar service","faltu service"
]

FRUSTRATED_KW = [
    "delay","delayed","late","still waiting",
    "pending","no update","follow up",
    "not received yet","waiting",

    # Hinglish
    "abhi tak nahi",
    "kab milega",
    "wait kar raha",
    "response nahi",
    "kitna late"
]

FEAR_KW = [
    "fraud","scam","cheated","security risk",
    "unsafe","data leak","account hacked",
    "legal","police","fir","court",

    # Hinglish
    "dhokha","scam ho gaya",
    "police complaint",
    "court case",
    "legal action"
]

SAD_KW = [
    "lost parcel","package lost",
    "damaged parcel","broken item",
    "very disappointed",
    "sad experience",

    # Hinglish
    "parcel kho gaya",
    "damage ho gaya",
    "bahut disappointment"
]

HAPPY_KW = [
    "great","excellent","awesome",
    "very good","fantastic",

    # Hinglish
    "bahut achha",
    "badiya",
    "shandaar"
]

SATISFIED_KW = [
    "resolved","issue solved",
    "thank you","thanks",
    "satisfied","problem solved",
    "delivery received",

    # Hinglish
    "problem solve ho gaya",
    "mil gaya parcel",
    "satisfied hoon",
    "dhanyavaad"
]

NEUTRAL_KW = [
    "ok","fine","information",
    "status","details",
    "tracking info",

    # Hinglish
    "theek hai",
    "details batao",
    "status batao"
]


# ------------------------------------------------------------
# EMOTION ASSIGNMENT LOGIC
# ------------------------------------------------------------

def assign_emotion(text, sentiment, aspect):

    t = str(text).lower()

    # FEAR (2)
    if any(k in t for k in FEAR_KW):
        return 2

    # ANGRY (0)
    if any(k in t for k in ANGRY_KW):
        return 0

    if sentiment == 0 and aspect in [2,3]:  # agent behaviour / service failure
        return 0

    # FRUSTRATED (1)
    if any(k in t for k in FRUSTRATED_KW):
        return 1

    if sentiment == 0:
        return 1

    # SAD (3)
    if any(k in t for k in SAD_KW):
        return 3

    # HAPPY (5)
    if any(k in t for k in HAPPY_KW):
        return 5

    # SATISFIED (6)
    if any(k in t for k in SATISFIED_KW):
        return 6

    if sentiment == 2:
        return 6

    # NEUTRAL (4)
    if any(k in t for k in NEUTRAL_KW):
        return 4

    return 4


# ------------------------------------------------------------
# APPLY ENGINE
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

required_cols = ["text","sentiment","primary_aspect"]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["emotion"] = df.apply(
    lambda r: assign_emotion(
        r["text"],
        int(r["sentiment"]),
        int(r["primary_aspect"])
    ),
    axis=1
)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Logistics Emotion v5 generated successfully")
print(df["emotion"].value_counts())