# # ============================================================
# # FILE    : generate_priority_v2.py
# # PURPOSE : Phase-5.5 GOLD GENERATOR — PRIORITY ENGINE
# # INDUSTRY: LOGISTICS + BANKING
# # ============================================================

# import pandas as pd

# # ------------------------------------------------------------
# # PATHS
# # ------------------------------------------------------------

# INPUT_PATH  = r"D:/bert_data/logistics data/train+intent+aspects+aspects_sentiment+emotions.csv"
# OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspects+aspects_sentiment+emotions+priority.csv"

# # ------------------------------------------------------------
# # PRIORITY ASSIGNMENT LOGIC (ENTERPRISE SAFE)
# # ------------------------------------------------------------

# def assign_priority(sentiment, aspect, aspect_sentiment, emotion):
#     """
#     sentiment        : 0=Negative, 1=Neutral, 2=Positive
#     emotion          : 0=Calm, 1=Satisfied, 2=Neutral,
#                        3=Frustrated, 4=Angry,
#                        5=Very Angry, 6=Fear
#     aspect           : domain specific (logistics / banking)
#     """

#     # --------------------------------------------------------
#     # 🔴 CRITICAL (IMMEDIATE ESCALATION)
#     # --------------------------------------------------------

#     # Fear / Legal / Security threat
#     if emotion == 6:
#         return 3

#     # Very Angry always critical
#     if emotion == 5:
#         return 3

#     # Angry + financial / loss risk
#     if emotion == 4 and aspect in [2, 7]:   # Damage/Lost, Refund
#         return 3

#     # Strong negative loss case
#     if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
#         return 3

#     # --------------------------------------------------------
#     # 🔴 HIGH PRIORITY
#     # --------------------------------------------------------

#     # Angry complaints
#     if emotion == 4:
#         return 2

#     # Frustrated + operational failure
#     if emotion == 3 and aspect in [0, 1, 5]:  # Delay, Wrong, Tracking
#         return 2

#     # Negative behaviour / staff issue
#     if sentiment == 0 and aspect == 3:
#         return 2

#     # Banking: transaction / security escalation
#     if sentiment == 0 and aspect in [0, 12]:  # transaction / security
#         return 2

#     # --------------------------------------------------------
#     # 🟡 MEDIUM PRIORITY
#     # --------------------------------------------------------

#     # Frustrated normal cases
#     if emotion == 3:
#         return 1

#     # Neutral but operational
#     if sentiment == 1 and aspect in [0, 1, 5]:
#         return 1

#     # Mild negative support
#     if sentiment == 0 and aspect in [4, 6]:  # support / customer service
#         return 1

#     # --------------------------------------------------------
#     # 🟢 LOW PRIORITY
#     # --------------------------------------------------------

#     # Calm / satisfied
#     if emotion in [0, 1]:
#         return 0

#     # Positive sentiment
#     if sentiment == 2:
#         return 0

#     # Default safe
#     return 0


# # ------------------------------------------------------------
# # APPLY TO DATASET
# # ------------------------------------------------------------

# df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

# required = ["sentiment", "primary_aspect", "aspect_sentiment", "emotion"]
# for c in required:
#     if c not in df.columns:
#         raise ValueError(f"Missing column: {c}")

# df["priority"] = df.apply(
#     lambda r: assign_priority(
#         int(r["sentiment"]),
#         int(r["primary_aspect"]),
#         int(r["aspect_sentiment"]),
#         int(r["emotion"])
#     ),
#     axis=1
# )

# # SAVE
# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Priority v2 generated successfully")
# print("Input :", INPUT_PATH)
# print("Output:", OUTPUT_PATH)

# print("\n--- Priority Distribution ---")
# print(df["priority"].value_counts())

# print("\n--- Priority by Emotion ---")
# print(pd.crosstab(df["emotion"], df["priority"]))

# print("\n--- Priority by Aspect ---")
# print(pd.crosstab(df["primary_aspect"], df["priority"]))

# ============================================================
# FILE    : generate_priority_v4.py
# PURPOSE : GOLD PRIORITY ENGINE (Keywords + Regex + Signals)
# INDUSTRY: LOGISTICS + BANKING
# ============================================================

import pandas as pd
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/data logistics new/train+intent+aspects+emotions.csv"
OUTPUT_PATH = r"D:/bert_data/data logistics new/train+intent+aspects+emotions+priority.csv"


# ------------------------------------------------------------
# NORMALIZE TEXT
# ------------------------------------------------------------

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text, keywords):
    return any(k in text for k in keywords)


# ------------------------------------------------------------
# PRIORITY KEYWORD BANKS
# ------------------------------------------------------------

Complaint_Service_Failure = [
    "disgusting behaviour","rude behaviour","very rude staff",
    "rude staff","harassment experience","felt harassed",
    "arrogant behaviour","staff not polite","no courtesy",
    "staff attitude bad","worst behaviour"
]

Social_Media_Legal_Threats = [
    "consumer court","see you court","file case","legal action",
    "take legal action","case against you","report to court",
    "consumer forum","ombudsman complaint","regulatory complaint",
    "post on twitter","tweet complaint","tweet to company",
    "share on facebook","post on facebook","social media complaint",
    "write public review","google review","public review",
    "post bad review","online complaint","send legal notice",
    "lawyer will contact","case already filed"
]

Product_Service_Issue = [
    "damaged parcel","broken parcel","parcel broken","parcel damaged",
    "package damaged","package broken","box damaged","box broken",
    "torn package","open package","opened parcel","seal broken",
    "broken frame","glass broken","food leaked","food spilling",
    "fruits damaged","laptop damaged","damaged goods"
]

Churn_Risk = [
    "stop using","stop using service","never again","will not use",
    "never use","not use again","avoid bluedart","avoid this courier",
    "switch courier","change courier","lost my trust",
    "won't recommend","cannot recommend","use other courier"
]

Death_Threat = [
    "kill you","kill him","kill her","i will kill",
    "death threat","threat to kill","physical harm",
    "life threat","danger to life","murder threat",
    "violent threat","threaten violence"
]

Weak_Complaints = [
    "poor service","bad service","very poor service",
    "worst service","bad experience","poor experience",
    "delivery slow","service slow"
]


# ------------------------------------------------------------
# REGEX PATTERNS
# ------------------------------------------------------------

PATTERN_LEGAL = r"(court|legal action|consumer forum|ombudsman)"
PATTERN_THREAT = r"(kill|murder|death threat|harm)"
PATTERN_CHURN = r"(never use|stop using|not use again|switch courier)"


# ------------------------------------------------------------
# TEXT BASED PRIORITY
# ------------------------------------------------------------

def priority_from_text(text):

    t = normalize(text)

    # CRITICAL
    if contains_any(t, Social_Media_Legal_Threats):
        return 3

    if contains_any(t, Death_Threat):
        return 3

    if re.search(PATTERN_LEGAL, t):
        return 3

    if re.search(PATTERN_THREAT, t):
        return 3

    # HIGH
    if contains_any(t, Complaint_Service_Failure):
        return 2

    if contains_any(t, Product_Service_Issue):
        return 2

    if contains_any(t, Churn_Risk):
        return 2

    if re.search(PATTERN_CHURN, t):
        return 2

    # MEDIUM
    if contains_any(t, Weak_Complaints):
        return 1

    return 0


# ------------------------------------------------------------
# STRUCTURED PRIORITY
# ------------------------------------------------------------

def priority_from_signals(sentiment, aspect, aspect_sentiment, emotion):

    # CRITICAL
    if emotion in [5,6]:
        return 3

    if sentiment == 0 and aspect in [2,7] and aspect_sentiment == 0:
        return 3

    # HIGH
    if emotion == 4:
        return 2

    if emotion == 3 and aspect in [0,1,5]:
        return 2

    # MEDIUM
    if emotion == 3:
        return 1

    if sentiment == 1 and aspect in [0,1,5]:
        return 1

    # LOW
    if emotion in [0,1] or sentiment == 2:
        return 0

    return 0


# ------------------------------------------------------------
# FINAL PRIORITY (HIGHEST WINS)
# ------------------------------------------------------------

def assign_priority(text, sentiment, aspect, aspect_sentiment, emotion):

    p_text = priority_from_text(text)

    p_signal = priority_from_signals(
        sentiment,
        aspect,
        aspect_sentiment,
        emotion
    )

    return max(p_text, p_signal)


# ------------------------------------------------------------
# APPLY ENGINE
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

required = ["text","sentiment","primary_aspect","aspect_sentiment","emotion"]

for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["priority"] = df.apply(
    lambda r: assign_priority(
        r["text"],
        int(r["sentiment"]),
        int(r["primary_aspect"]),
        int(r["aspect_sentiment"]),
        int(r["emotion"])
    ),
    axis=1
)

# SAVE
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Priority v4 generated successfully")
print("\n--- Priority Distribution ---")
print(df["priority"].value_counts())

print("\n--- Priority by Emotion ---")
print(pd.crosstab(df["emotion"], df["priority"]))

print("\n--- Priority by Aspect ---")
print(pd.crosstab(df["primary_aspect"], df["priority"]))