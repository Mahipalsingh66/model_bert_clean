# # ============================================================
# # FILE    : run_full_multitask_pipeline.py
# # PURPOSE : Run full 6-step CX pipeline in ONE execution
# # NOTE    : LOGIC + KEYWORDS ARE UNCHANGED
# # ============================================================

# import pandas as pd

# # ============================================================
# # INPUT / OUTPUT
# # ============================================================

# INPUT_PATH  = r"D:/model_bert_copy\data/gold/multitask_raw_data/val.csv"
# OUTPUT_PATH = r"D:/model_bert_copy/data/gold/multitask_raw_data/val_multi_logistics_6step_final.csv"

# # ============================================================
# # STEP 1 — INTENT MAKER (UNCHANGED)
# # ============================================================

# def assign_intent(text, sentiment):
#     t = text.lower()

#     if sentiment == 0 or any(k in t for k in [
#         "delay","late","not delivered","no contact","no visit",
#         "wrong","issue","problem","bad","worst","rude",
#         "missing","damage","damaged","lost","refund"
#     ]):
#         return 0

#     if sentiment == 2 or any(k in t for k in [
#         "good","nice","excellent","happy","thanks","thank you",
#         "great","perfect","smooth","clear","awesome"
#     ]):
#         return 2

#     if any(k in t for k in [
#         "where","when","status","track","tracking","update","why"
#     ]):
#         return 1

#     return 1


# # ============================================================
# # STEP 2 — ASPECT V3 (UNCHANGED)
# # ============================================================

# DELAY_KW = [
#     "late","delay","delayed","one day late","two days late","three days late",
#     "not on time","not delivered on time","delivery pending","pending delivery",
#     "pending for days","still waiting","waiting for delivery","no delivery yet",
#     "after due date","delivery not attempted","attempted late",
#     "shipment delayed","out for delivery but not delivered",
#     "postponed","rescheduled delivery","delivery attempt failed",
#     "delivery failed","delivery postponed","delivery rescheduled",
#     "not delivered yet","delivery slow","delivery very late",
#     "took long time","taking too long"
# ]

# WRONG_KW = [
#     "wrong address","wrong location","wrong place","wrong area",
#     "delivered to wrong","delivered somewhere else","delivered other place",
#     "someone else received","unknown person received",
#     "maid received","neighbor received","security received",
#     "not my order","not my package","not my parcel",
#     "wrong pin code","wrong building","wrong flat",
#     "misdelivered","incorrect delivery","wrong house","wrong society",
#     "wrong recipient","delivered to another person"
# ]

# DAMAGE_KW = [
#     "damaged","damage","broken","cracked","leak","leaked",
#     "torn","open packet","opened parcel","empty box",
#     "missing item","item missing","parcel missing","content missing",
#     "lost","not received","did not receive","never received",
#     "package lost","shipment lost","product missing",
#     "stolen","tampered","seal broken","box damaged",
#     "package empty","nothing inside","product damaged"
# ]

# BEHAVIOUR_KW = [
#     "rude","misbehave","misbehavior","behaviour","behavior",
#     "bad attitude","unprofessional","shouted","argued",
#     "angry delivery","delivery boy rude","staff rude",
#     "threatened","abused","not polite","very arrogant",
#     "impolite","bad manners","harassment","unacceptable behaviour",
#     "did not contact","not contacted","no call","did not call",
#     "no intimation","did not inform","no message",
#     "delivery boy not cooperative","delivery agent rude",
#     "not responsive delivery boy"
# ]

# TRACKING_KW = [
#     "track","tracking","tracking not updated","status not updated",
#     "no update","no tracking","awb","waybill","consignment",
#     "where is my order","where is my parcel","where is shipment",
#     "no status","status wrong","location not updated",
#     "showing delivered but not received","fake delivery status",
#     "system shows delivered","tracking wrong","tracking error",
#     "not reflecting","tracking problem","tracking issue",
#     "no movement","stuck in transit"
# ]

# PRICING_KW = [
#     "charged","extra charge","overcharged","wrong amount",
#     "high charge","pricing issue","price mismatch",
#     "fee","service charge","cod charge","cash on delivery charge",
#     "billing issue","payment issue","amount deducted",
#     "double charged","refund amount wrong",
#     "cost issue","pricing problem","rate issue"
# ]

# REFUND_KW = [
#     "refund","refunded","money back","amount not credited",
#     "return","returned","return process","return completed",
#     "refund pending","refund delayed","not refunded",
#     "refund not received","replacement","exchange",
#     "cancelled order refund","reverse pickup",
#     "pickup completed but refund not received",
#     "waiting for refund","refund issue","refund problem"
# ]

# SUPPORT_KW = [
#     "call","called","contact","support","customer care",
#     "helpline","complaint","registered complaint",
#     "no response","not responding","no reply","waiting for response",
#     "ticket","case id","grievance","escalation","follow up",
#     "no callback","service issue","helpdesk","query",
#     "customer service","service center","support team"
# ]

# SOFT_DELAY_PATTERNS = [
#     "delivery very slow","still not received","long delay",
#     "taking long","delivery taking time","delay in delivery"
# ]

# SOFT_TRACKING_PATTERNS = [
#     "no update from courier","no update yet",
#     "waiting for update","status not clear",
#     "no information","no tracking update"
# ]

# SOFT_BEHAVIOUR_PATTERNS = [
#     "not contacted","no call received",
#     "did not inform","no intimation given",
#     "no message received"
# ]

# def assign_aspect(text):
#     t = text.lower()

#     if any(k in t for k in DAMAGE_KW):
#         return 2
#     if any(k in t for k in WRONG_KW):
#         return 1
#     if any(k in t for k in DELAY_KW):
#         return 0
#     if any(k in t for k in REFUND_KW):
#         return 7
#     if any(k in t for k in PRICING_KW):
#         return 6
#     if any(k in t for k in TRACKING_KW):
#         return 5
#     if any(k in t for k in BEHAVIOUR_KW):
#         return 3
#     if any(k in t for k in SOFT_DELAY_PATTERNS):
#         return 0
#     if any(k in t for k in SOFT_TRACKING_PATTERNS):
#         return 5
#     if any(k in t for k in SOFT_BEHAVIOUR_PATTERNS):
#         return 3
#     if any(k in t for k in SUPPORT_KW):
#         return 4

#     return 4


# # ============================================================
# # STEP 3 — ASPECT SENTIMENT (UNCHANGED)
# # ============================================================

# NEGATIVE_ASPECT_BIAS = {0:True,1:True,2:True,7:True}
# POSITIVE_ASPECT_ALLOW = {3:True,4:True}

# def assign_aspect_sentiment(text, sentiment, aspect):
#     t = text.lower()

#     if any(k in t for k in [
#         "late","delay","not delivered","damaged","lost","wrong",
#         "rude","refund","issue","problem","pending"
#     ]):
#         return 0

#     if any(k in t for k in [
#         "good","great","excellent","nice","perfect",
#         "thanks","resolved","very good"
#     ]):
#         return 2 if POSITIVE_ASPECT_ALLOW.get(aspect, False) else 1

#     if sentiment == 0 and NEGATIVE_ASPECT_BIAS.get(aspect, False):
#         return 0

#     if sentiment == 2 and POSITIVE_ASPECT_ALLOW.get(aspect, False):
#         return 2

#     return 1


# # ============================================================
# # STEP 4 — EMOTION V3 (UNCHANGED)
# # ============================================================

# VERY_ANGRY_KW = ["fraud","cheated","scam","court","legal","harassment","refund fraud","money stolen"]
# ANGRY_KW = ["angry","furious","complaint","bad service","poor service","rude","refund pending","lost","damaged","wrong delivery"]
# FRUSTRATED_KW = ["waiting","delay","pending","no update","follow up","concern","confused","still waiting"]
# SATISFIED_KW = ["good","great","excellent","happy","satisfied","thanks","resolved","well done"]

# def assign_emotion(text, sentiment, aspect, aspect_sentiment):
#     t = text.lower()

#     if any(k in t for k in VERY_ANGRY_KW):
#         return 4
#     if sentiment == 0 and aspect in [2,7] and aspect_sentiment == 0:
#         return 4
#     if any(k in t for k in ANGRY_KW):
#         return 3
#     if sentiment == 0 and aspect_sentiment == 0:
#         return 3
#     if any(k in t for k in FRUSTRATED_KW) and sentiment != 2:
#         return 2
#     if any(k in t for k in SATISFIED_KW) or (sentiment == 2 and aspect_sentiment == 2):
#         return 1
#     return 0


# # ============================================================
# # STEP 5 — PRIORITY (UNCHANGED)
# # ============================================================

# def assign_priority(sentiment, aspect, aspect_sentiment, emotion):
#     if emotion == 4:
#         return 3
#     if emotion == 3 and aspect in [2,7]:
#         return 3
#     if sentiment == 0 and aspect in [2,7] and aspect_sentiment == 0:
#         return 3
#     if emotion == 3:
#         return 2
#     if emotion == 2 and aspect in [0,1,5]:
#         return 2
#     if sentiment == 0 and aspect == 3:
#         return 2
#     if emotion == 2:
#         return 1
#     if sentiment == 1 and aspect in [0,1,5]:
#         return 1
#     if sentiment == 0 and aspect == 4:
#         return 1
#     return 0


# # ============================================================
# # STEP 6 — LOGISTICS ASPECT (UNCHANGED)
# # ============================================================

# def assign_logistics_aspect(text):
#     t = text.lower()

#     if any(k in t for k in DAMAGED_PACKAGE_KW):
#         return 5, "damaged_package"
#     if any(k in t for k in WRONG_DELIVERY_KW):
#         return 1, "wrong_delivery"
#     if any(k in t for k in DELIVERY_FAILURE_KW):
#         return 0, "delivery_failures"
#     if any(k in t for k in TRACKING_KW):
#         return 4, "tracking_updates"
#     if any(k in t for k in AGENT_BEHAVIOUR_KW):
#         return 2, "agent_behaviour"
#     if any(k in t for k in PAYMENT_ISSUES_KW):
#         return 6, "payment_issues"
#     if any(k in t for k in CUSTOMER_SUPPORT_KW):
#         return 3, "customer_support"
#     if any(k in t for k in SERVICE_QUALITY_KW):
#         return 7, "service_quality"

#     return 7, "service_quality"


# # ============================================================
# # MAIN EXECUTION
# # ============================================================

# df = pd.read_csv(INPUT_PATH)

# df["sentiment"] = df["label"]
# df["customer_intent"] = df.apply(lambda x: assign_intent(x["text"], x["sentiment"]), axis=1)
# df["primary_aspect"] = df["text"].apply(assign_aspect)
# df["aspect_sentiment"] = df.apply(lambda x: assign_aspect_sentiment(x["text"], x["sentiment"], x["primary_aspect"]), axis=1)
# df["emotion"] = df.apply(lambda x: assign_emotion(x["text"], x["sentiment"], x["primary_aspect"], x["aspect_sentiment"]), axis=1)
# df["priority"] = df.apply(lambda x: assign_priority(x["sentiment"], x["primary_aspect"], x["aspect_sentiment"], x["emotion"]), axis=1)

# log_aspects = df["text"].apply(assign_logistics_aspect)
# df["primary_aspect"], df["primary_aspect_name"] = zip(*log_aspects)

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ FULL 6-STEP PIPELINE COMPLETED")
# print("Final rows:", len(df))
# ============================================================
# FILE    : run_full_multitask_pipeline.py
# PURPOSE : Run full 6-step CX pipeline in ONE execution
# NOTE    : CORE LOGIC + KEYWORDS ARE UNCHANGED
# ============================================================

import pandas as pd

# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_PATH  = r"D:/model_bert_copy\data/gold/multitask_raw_data/val.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/multitask_raw_data/val_multi_logistics_6step_final.csv"

# ============================================================
# STEP 1 — INTENT MAKER (UNCHANGED)
# ============================================================

def assign_intent(text, sentiment):
    t = text.lower()

    if sentiment == 0 or any(k in t for k in [
        "delay", "late", "not delivered", "no contact", "no visit",
        "wrong", "issue", "problem", "bad", "worst", "rude",
        "missing", "damage", "damaged", "lost", "refund"
    ]):
        return 0

    if sentiment == 2 or any(k in t for k in [
        "good", "nice", "excellent", "happy", "thanks", "thank you",
        "great", "perfect", "smooth", "clear", "awesome"
    ]):
        return 2

    if any(k in t for k in [
        "where", "when", "status", "track", "tracking", "update", "why"
    ]):
        return 1

    return 1


# ============================================================
# STEP 2 — ASPECT V3 (UNCHANGED)
# ============================================================

DELAY_KW = [
    "late", "delay", "delayed", "one day late", "two days late", "three days late",
    "not on time", "not delivered on time", "delivery pending", "pending delivery",
    "pending for days", "still waiting", "waiting for delivery", "no delivery yet",
    "after due date", "delivery not attempted", "attempted late",
    "shipment delayed", "out for delivery but not delivered",
    "postponed", "rescheduled delivery", "delivery attempt failed",
    "delivery failed", "delivery postponed", "delivery rescheduled",
    "not delivered yet", "delivery slow", "delivery very late",
    "took long time", "taking too long"
]

WRONG_KW = [
    "wrong address", "wrong location", "wrong place", "wrong area",
    "delivered to wrong", "delivered somewhere else", "delivered other place",
    "someone else received", "unknown person received",
    "maid received", "neighbor received", "security received",
    "not my order", "not my package", "not my parcel",
    "wrong pin code", "wrong building", "wrong flat",
    "misdelivered", "incorrect delivery", "wrong house", "wrong society",
    "wrong recipient", "delivered to another person"
]

DAMAGE_KW = [
    "damaged", "damage", "broken", "cracked", "leak", "leaked",
    "torn", "open packet", "opened parcel", "empty box",
    "missing item", "item missing", "parcel missing", "content missing",
    "lost", "not received", "did not receive", "never received",
    "package lost", "shipment lost", "product missing",
    "stolen", "tampered", "seal broken", "box damaged",
    "package empty", "nothing inside", "product damaged"
]

BEHAVIOUR_KW = [
    "rude", "misbehave", "misbehavior", "behaviour", "behavior",
    "bad attitude", "unprofessional", "shouted", "argued",
    "angry delivery", "delivery boy rude", "staff rude",
    "threatened", "abused", "not polite", "very arrogant",
    "impolite", "bad manners", "harassment", "unacceptable behaviour",
    "did not contact", "not contacted", "no call", "did not call",
    "no intimation", "did not inform", "no message",
    "delivery boy not cooperative", "delivery agent rude",
    "not responsive delivery boy"
]

TRACKING_KW = [
    "track", "tracking", "tracking not updated", "status not updated",
    "no update", "no tracking", "awb", "waybill", "consignment",
    "where is my order", "where is my parcel", "where is shipment",
    "no status", "status wrong", "location not updated",
    "showing delivered but not received", "fake delivery status",
    "system shows delivered", "tracking wrong", "tracking error",
    "not reflecting", "tracking problem", "tracking issue",
    "no movement", "stuck in transit"
]

PRICING_KW = [
    "charged", "extra charge", "overcharged", "wrong amount",
    "high charge", "pricing issue", "price mismatch",
    "fee", "service charge", "cod charge", "cash on delivery charge",
    "billing issue", "payment issue", "amount deducted",
    "double charged", "refund amount wrong",
    "cost issue", "pricing problem", "rate issue"
]

REFUND_KW = [
    "refund", "refunded", "money back", "amount not credited",
    "return", "returned", "return process", "return completed",
    "refund pending", "refund delayed", "not refunded",
    "refund not received", "replacement", "exchange",
    "cancelled order refund", "reverse pickup",
    "pickup completed but refund not received",
    "waiting for refund", "refund issue", "refund problem"
]

SUPPORT_KW = [
    "call", "called", "contact", "support", "customer care",
    "helpline", "complaint", "registered complaint",
    "no response", "not responding", "no reply", "waiting for response",
    "ticket", "case id", "grievance", "escalation", "follow up",
    "no callback", "service issue", "helpdesk", "query",
    "customer service", "service center", "support team"
]

SOFT_DELAY_PATTERNS = [
    "delivery very slow", "still not received", "long delay",
    "taking long", "delivery taking time", "delay in delivery"
]

SOFT_TRACKING_PATTERNS = [
    "no update from courier", "no update yet",
    "waiting for update", "status not clear",
    "no information", "no tracking update"
]

SOFT_BEHAVIOUR_PATTERNS = [
    "not contacted", "no call received",
    "did not inform", "no intimation given",
    "no message received"
]

def assign_aspect(text):
    t = text.lower()

    if any(k in t for k in DAMAGE_KW):
        return 2
    if any(k in t for k in WRONG_KW):
        return 1
    if any(k in t for k in DELAY_KW):
        return 0
    if any(k in t for k in REFUND_KW):
        return 7
    if any(k in t for k in PRICING_KW):
        return 6
    if any(k in t for k in TRACKING_KW):
        return 5
    if any(k in t for k in BEHAVIOUR_KW):
        return 3
    if any(k in t for k in SOFT_DELAY_PATTERNS):
        return 0
    if any(k in t for k in SOFT_TRACKING_PATTERNS):
        return 5
    if any(k in t for k in SOFT_BEHAVIOUR_PATTERNS):
        return 3
    if any(k in t for k in SUPPORT_KW):
        return 4

    return 4


# ============================================================
# STEP 3 — ASPECT SENTIMENT (UNCHANGED)
# ============================================================

NEGATIVE_ASPECT_BIAS = {0: True, 1: True, 2: True, 7: True}
POSITIVE_ASPECT_ALLOW = {3: True, 4: True}

def assign_aspect_sentiment(text, sentiment, aspect):
    t = text.lower()

    if any(k in t for k in [
        "late", "delay", "not delivered", "damaged", "lost", "wrong",
        "rude", "refund", "issue", "problem", "pending"
    ]):
        return 0

    if any(k in t for k in [
        "good", "great", "excellent", "nice", "perfect",
        "thanks", "resolved", "very good"
    ]):
        return 2 if POSITIVE_ASPECT_ALLOW.get(aspect, False) else 1

    if sentiment == 0 and NEGATIVE_ASPECT_BIAS.get(aspect, False):
        return 0

    if sentiment == 2 and POSITIVE_ASPECT_ALLOW.get(aspect, False):
        return 2

    return 1


# ============================================================
# STEP 4 — EMOTION V3 (UNCHANGED)
# ============================================================

VERY_ANGRY_KW = ["fraud", "cheated", "scam", "court", "legal", "harassment", "refund fraud", "money stolen"]
ANGRY_KW = ["angry", "furious", "complaint", "bad service", "poor service", "rude", "refund pending", "lost", "damaged", "wrong delivery"]
FRUSTRATED_KW = ["waiting", "delay", "pending", "no update", "follow up", "concern", "confused", "still waiting"]
SATISFIED_KW = ["good", "great", "excellent", "happy", "satisfied", "thanks", "resolved", "well done"]

def assign_emotion(text, sentiment, aspect, aspect_sentiment):
    t = text.lower()

    if any(k in t for k in VERY_ANGRY_KW):
        return 4
    if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
        return 4
    if any(k in t for k in ANGRY_KW):
        return 3
    if sentiment == 0 and aspect_sentiment == 0:
        return 3
    if any(k in t for k in FRUSTRATED_KW) and sentiment != 2:
        return 2
    if any(k in t for k in SATISFIED_KW) or (sentiment == 2 and aspect_sentiment == 2):
        return 1
    return 0


# ============================================================
# STEP 5 — PRIORITY (UNCHANGED)
# ============================================================

def assign_priority(sentiment, aspect, aspect_sentiment, emotion):
    if emotion == 4:
        return 3
    if emotion == 3 and aspect in [2, 7]:
        return 3
    if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
        return 3
    if emotion == 3:
        return 2
    if emotion == 2 and aspect in [0, 1, 5]:
        return 2
    if sentiment == 0 and aspect == 3:
        return 2
    if emotion == 2:
        return 1
    if sentiment == 1 and aspect in [0, 1, 5]:
        return 1
    if sentiment == 0 and aspect == 4:
        return 1
    return 0


# ============================================================
# STEP 6 — LOGISTICS ASPECT (UNCHANGED, FIXED DEFINITIONS)
# ============================================================

DELIVERY_FAILURE_KW = [
    "late", "delay", "delayed", "not delivered", "delivery failed",
    "delivery pending", "pending delivery", "still waiting",
    "delivery attempt failed", "not attempted", "delivery not done",
    "delivery postponed", "delivery rescheduled",
    "out for delivery but not delivered",
    "shipment delayed", "delivery very slow", "took too long"
]

WRONG_DELIVERY_KW = [
    "wrong delivery", "wrong address", "wrong location", "wrong person",
    "delivered to someone else", "delivered to wrong person",
    "unknown person received", "maid received", "security received",
    "neighbor received", "not my order", "not my parcel",
    "wrong flat", "wrong building", "wrong pin code",
    "misdelivered", "incorrect delivery"
]

AGENT_BEHAVIOUR_KW = [
    "rude", "misbehave", "misbehavior", "unprofessional",
    "bad attitude", "shouted", "argued", "abused",
    "delivery boy rude", "agent rude", "staff rude",
    "not polite", "very arrogant", "harassment",
    "did not call", "no call", "not contacted",
    "no intimation", "did not inform", "no message",
    "agent not cooperative"
]

CUSTOMER_SUPPORT_KW = [
    "customer support", "customer care", "support team",
    "called support", "no response", "not responding",
    "no reply", "waiting for response", "complaint registered",
    "ticket raised", "case id", "grievance",
    "escalation", "follow up", "no callback",
    "helpline", "service center"
]

TRACKING_KW = [
    "tracking", "track", "tracking not updated",
    "status not updated", "no update", "no tracking",
    "awb", "waybill", "consignment",
    "where is my order", "where is my parcel",
    "showing delivered but not received",
    "fake delivery status", "tracking wrong",
    "stuck in transit", "no movement"
]

DAMAGED_PACKAGE_KW = [
    "damaged", "broken", "cracked", "leaked", "torn",
    "open packet", "opened parcel", "empty box",
    "item missing", "parcel missing", "content missing",
    "lost", "package lost", "shipment lost",
    "stolen", "tampered", "seal broken",
    "box damaged", "nothing inside"
]

PAYMENT_ISSUES_KW = [
    "payment issue", "extra charge", "charged extra",
    "overcharged", "wrong amount", "high charge",
    "cod charge", "cash on delivery charge",
    "amount deducted", "double charged",
    "refund issue", "refund pending", "refund delayed",
    "refund not received", "money not returned"
]

SERVICE_QUALITY_KW = [
    "poor service", "bad service", "service quality",
    "very bad experience", "disappointed",
    "not satisfied", "unsatisfactory service",
    "worst experience", "pathetic service",
    "service is bad"
]

def assign_logistics_aspect(text):
    t = text.lower()

    if any(k in t for k in DAMAGED_PACKAGE_KW):
        return 5, "damaged_package"
    if any(k in t for k in WRONG_DELIVERY_KW):
        return 1, "wrong_delivery"
    if any(k in t for k in DELIVERY_FAILURE_KW):
        return 0, "delivery_failures"
    if any(k in t for k in TRACKING_KW):
        return 4, "tracking_updates"
    if any(k in t for k in AGENT_BEHAVIOUR_KW):
        return 2, "agent_behaviour"
    if any(k in t for k in PAYMENT_ISSUES_KW):
        return 6, "payment_issues"
    if any(k in t for k in CUSTOMER_SUPPORT_KW):
        return 3, "customer_support"
    if any(k in t for k in SERVICE_QUALITY_KW):
        return 7, "service_quality"

    return 7, "service_quality"


# ============================================================
# MAIN EXECUTION
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["sentiment"] = df["label"]
df["customer_intent"] = df.apply(lambda x: assign_intent(x["text"], x["sentiment"]), axis=1)
df["primary_aspect"] = df["text"].apply(assign_aspect)
df["aspect_sentiment"] = df.apply(
    lambda x: assign_aspect_sentiment(x["text"], x["sentiment"], x["primary_aspect"]), axis=1
)
df["emotion"] = df.apply(
    lambda x: assign_emotion(x["text"], x["sentiment"], x["primary_aspect"], x["aspect_sentiment"]), axis=1
)
df["priority"] = df.apply(
    lambda x: assign_priority(x["sentiment"], x["primary_aspect"], x["aspect_sentiment"], x["emotion"]), axis=1
)

log_aspects = df["text"].apply(assign_logistics_aspect)
df["primary_aspect"], df["primary_aspect_name"] = zip(*log_aspects)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ FULL 6-STEP PIPELINE COMPLETED SUCCESSFULLY")
print("Rows:", len(df))
