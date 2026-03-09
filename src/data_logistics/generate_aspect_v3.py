# # ============================================================
# # FILE    : generate_aspect_v3.py
# # PURPOSE : FINAL Enterprise Aspect Rule Engine (Production)
# # INPUT   : text,sentiment,customer_intent
# # OUTPUT  : text,sentiment,customer_intent,primary_aspect,aspect_flag
# #
# # TAXONOMY:
# #   0 = Delay
# #   1 = Wrong Delivery
# #   2 = Damage / Lost
# #   3 = Behaviour
# #   4 = Support
# #   5 = Tracking
# #   6 = Pricing
# #   7 = Refund
# # ============================================================

# import pandas as pd

# # ------------------------------------------------------------
# # PATHS
# # ------------------------------------------------------------

# INPUT_PATH  = r"D:/bert_data/logistics data/train+intent.csv"
# OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspects.csv"

# # ------------------------------------------------------------
# # KEYWORD BANK (ENGLISH + HINGLISH)
# # ------------------------------------------------------------

# DELAY_KW = [
#     "late", "delay", "delayed", "delivery late", "delivery delay",
#     "not on time", "not delivered on time", "delivery pending",
#     "pending delivery", "still waiting", "no delivery yet",
#     "after due date", "delivery not attempted",
#     "shipment delayed", "out for delivery but not delivered",
#     "delivery slow", "took long time", "taking too long",

#     # Hinglish
#     "bahut late", "abhi tak deliver nahi hua",
#     "delivery abhi tak nahi aayi",
#     "kab deliver hoga", "delivery pending hai",
#     "der se aaya", "late aaya"
# ]

# WRONG_KW = [
#     "wrong address", "wrong location", "wrong place",
#     "delivered to wrong", "someone else received",
#     "unknown person received", "maid received",
#     "neighbor received", "security received",
#     "not my order", "not my parcel",
#     "wrong pin code", "wrong flat", "misdelivered",

#     # Hinglish
#     "galat address", "galat jagah deliver",
#     "kisi aur ko de diya",
#     "galat parcel", "mera order nahi hai"
# ]

# DAMAGE_KW = [
#     "damaged", "damage", "broken", "cracked", "leaked",
#     "torn", "open packet", "opened parcel", "empty box",
#     "missing item", "item missing", "parcel missing",
#     "lost", "not received", "never received",
#     "stolen", "tampered", "seal broken",

#     # Hinglish
#     "toota hua", "khula hua packet",
#     "samaan missing", "box khali tha",
#     "parcel nahi mila", "samaan chori ho gaya"
# ]

# BEHAVIOUR_KW = [
#     "rude", "misbehave", "bad behaviour",
#     "unprofessional", "argued", "abused",
#     "delivery boy rude", "agent rude",
#     "not polite", "harassment",
#     "did not call", "no call", "not contacted",

#     # Hinglish
#     "bad behaviour", "misbehave kiya",
#     "baat karne ka tareeka kharab",
#     "delivery boy badtameez",
#     "call nahi kiya", "baat nahi ki"
# ]

# TRACKING_KW = [
#     "track", "tracking", "tracking not updated",
#     "status not updated", "no update",
#     "awb", "waybill", "consignment",
#     "showing delivered but not received",
#     "fake delivery status", "stuck in transit",

#     # Hinglish
#     "tracking update nahi hai",
#     "status galat dikha raha",
#     "delivered dikha raha hai par mila nahi",
#     "tracking issue hai"
# ]

# PRICING_KW = [
#     "charged", "extra charge", "overcharged",
#     "wrong amount", "high charge",
#     "cod charge", "service charge",
#     "amount deducted", "double charged",

#     # Hinglish
#     "extra paisa liya",
#     "zyada charge kiya",
#     "galat amount kata",
#     "paise zyada kat gaye"
# ]

# REFUND_KW = [
#     "refund", "refunded", "money back",
#     "amount not credited", "refund pending",
#     "refund delayed", "not refunded",
#     "replacement", "return completed",

#     # Hinglish
#     "refund nahi mila",
#     "paise wapas nahi aaye",
#     "refund pending hai",
#     "return ka paisa nahi aaya"
# ]

# SUPPORT_KW = [
#     "call", "contact", "support",
#     "customer care", "helpline",
#     "complaint", "ticket raised",
#     "no response", "not responding",

#     # Hinglish
#     "customer care se baat nahi hui",
#     "call connect nahi ho raha",
#     "support se response nahi mila"
# ]

# # ------------------------------------------------------------
# # SOFT PATTERNS
# # ------------------------------------------------------------

# SOFT_DELAY_PATTERNS = [
#     "delivery slow", "still not received",
#     "abhi tak nahi mila"
# ]

# SOFT_TRACKING_PATTERNS = [
#     "no update yet", "tracking unclear",
#     "status clear nahi hai"
# ]
# u
# SOFT_BEHAVIOUR_PATTERNS = [
#     "not contacted", "call nahi aaya"
# ]

# # ------------------------------------------------------------
# # ASPECT ASSIGNMENT
# # ------------------------------------------------------------

# def assign_aspect(text):
#     if not isinstance(text, str) or not text.strip():
#         return 4, "fallback_support"

#     t = text.lower()

#     if any(k in t for k in DAMAGE_KW):
#         return 2, "damage_lost"

#     if any(k in t for k in WRONG_KW):
#         return 1, "wrong_delivery"

#     if any(k in t for k in DELAY_KW):
#         return 0, "delay"

#     if any(k in t for k in REFUND_KW):
#         return 7, "refund"

#     if any(k in t for k in PRICING_KW):
#         return 6, "pricing"

#     if any(k in t for k in TRACKING_KW):
#         return 5, "tracking"

#     if any(k in t for k in BEHAVIOUR_KW):
#         return 3, "behaviour"

#     if any(k in t for k in SOFT_DELAY_PATTERNS):
#         return 0, "soft_delay"

#     if any(k in t for k in SOFT_TRACKING_PATTERNS):
#         return 5, "soft_tracking"

#     if any(k in t for k in SOFT_BEHAVIOUR_PATTERNS):
#         return 3, "soft_behaviour"

#     if any(k in t for k in SUPPORT_KW):
#         return 4, "support"

#     return 4, "fallback_support"

# # ------------------------------------------------------------
# # APPLY
# # ------------------------------------------------------------

# df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

# aspects, flags = [], []

# for text in df["text"]:
#     a, f = assign_aspect(text)
#     aspects.append(a)
#     flags.append(f)

# df["primary_aspect"] = aspects
# df["aspect_flag"] = flags

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Logistics Aspect v3 (HINGLISH) generated")
# print(df["primary_aspect"].value_counts())

# ============================================================
# FILE    : generate_aspect_v4.py
# PURPOSE : ENTERPRISE Logistics Aspect Rule Engine (Gold Data)
# INPUT   : text, sentiment, customer_intent
# OUTPUT  : primary_aspect, aspect_flag
#
# DESIGN PRINCIPLES
# - Appreciation is POSITIVE-ONLY
# - Complaint > Praise in mixed feedback
# - No garbage fallback to Support
# - Aspect purity > coverage
#
# ============================================================

# import pandas as pd

# # ------------------------------------------------------------
# # PATHS
# # ------------------------------------------------------------

# INPUT_PATH  = r"D:/bert_data/logistics data/train+intent.csv"
# OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspect_v4.csv"

# # ------------------------------------------------------------
# # ASPECT IDS (LOCKED)
# # ------------------------------------------------------------
# """
# 0  = Delivery_Delay
# 1  = Delivery_Failure
# 2  = Package_Damage
# 3  = Package_Lost
# 4  = Staff_Behaviour
# 5  = Customer_Support
# 6  = Tracking_Issue
# 7  = Pricing_Issue
# 8  = Refund_Issue
# 9  = Pickup_Issue
# 10 = System_Technical
# 11 = Process_Complexity
# 12 = Policy_Issue
# 13 = Appreciation
# 14 = General_Feedback
# """

# # ------------------------------------------------------------
# # KEYWORD BANK (ENGLISH + HINGLISH)
# # ------------------------------------------------------------

# APPRECIATION_KW = [
#     "thank", "thanks", "thank you", "appreciate",
#     "good service", "great service", "excellent",
#     "awesome", "well done", "satisfied", "happy",

#     "dhanyavaad", "shukriya", "bahut accha",
#     "achhi service", "khush hoon"
# ]

# DELAY_KW = [
#     "late", "delay", "delayed", "pending delivery",
#     "not delivered on time", "still waiting",
#     "delivery slow", "took long time",

#     "bahut late", "der se aaya",
#     "abhi tak deliver nahi hua",
#     "delivery pending hai"
# ]

# FAILURE_KW = [
#     "wrong address", "wrong location",
#     "misdelivered", "delivered to wrong",
#     "not my parcel", "someone else received",

#     "galat address", "galat jagah deliver",
#     "kisi aur ko de diya"
# ]

# DAMAGE_KW = [
#     "damaged", "broken", "cracked",
#     "leaked", "torn", "open packet",
#     "seal broken",

#     "toota hua", "khula hua packet"
# ]

# LOST_KW = [
#     "lost", "missing", "not received",
#     "never received", "empty box",

#     "parcel nahi mila", "samaan missing"
# ]

# BEHAVIOUR_KW = [
#     "rude", "misbehave", "harassment",
#     "unprofessional", "abused",

#     "delivery boy rude",
#     "badtameez", "baat karne ka tareeka kharab"
# ]

# TRACKING_KW = [
#     "tracking", "awb", "waybill",
#     "status not updated",
#     "showing delivered but not received",

#     "tracking update nahi hai",
#     "status galat dikha raha"
# ]

# PRICING_KW = [
#     "charged", "extra charge",
#     "overcharged", "wrong amount",
#     "double charged",

#     "extra paisa liya",
#     "zyada charge kiya"
# ]

# REFUND_KW = [
#     "refund", "money back",
#     "refund pending", "not refunded",

#     "paise wapas nahi aaye",
#     "refund nahi mila"
# ]

# SUPPORT_KW = [
#     "customer care", "support",
#     "call center", "helpline",
#     "no response", "not responding",

#     "call connect nahi ho raha",
#     "response nahi mila"
# ]

# SYSTEM_KW = [
#     "app not working", "website issue",
#     "login problem", "technical issue",

#     "app crash", "system error"
# ]

# PROCESS_KW = [
#     "too complicated", "confusing process",
#     "many steps", "hard to understand",

#     "process samajh nahi aaya"
# ]

# POLICY_KW = [
#     "policy", "terms and conditions",
#     "rules", "policy issue",

#     "policy bekaar hai"
# ]

# # ------------------------------------------------------------
# # UTILITY
# # ------------------------------------------------------------

# def contains_any(text, keywords):
#     return any(k in text for k in keywords)

# # ------------------------------------------------------------
# # ASPECT ASSIGNMENT (CORE LOGIC)
# # ------------------------------------------------------------

# def assign_aspect(text, sentiment):
#     if not isinstance(text, str) or not text.strip():
#         return 14, "empty_text"

#     t = text.lower()

#     # 🔒 HARD RULE 1: Appreciation is POSITIVE-ONLY
#     if sentiment == "Positive" and contains_any(t, APPRECIATION_KW):
#         return 13, "appreciation"

#     # 🔒 HARD RULE 2: Complaint dominates praise
#     # (so DO NOT return appreciation after this point)

#     if contains_any(t, DAMAGE_KW):
#         return 2, "package_damage"

#     if contains_any(t, LOST_KW):
#         return 3, "package_lost"

#     if contains_any(t, FAILURE_KW):
#         return 1, "delivery_failure"

#     if contains_any(t, DELAY_KW):
#         return 0, "delivery_delay"

#     if contains_any(t, REFUND_KW):
#         return 8, "refund_issue"

#     if contains_any(t, PRICING_KW):
#         return 7, "pricing_issue"

#     if contains_any(t, TRACKING_KW):
#         return 6, "tracking_issue"

#     if contains_any(t, BEHAVIOUR_KW):
#         return 4, "staff_behaviour"

#     if contains_any(t, SYSTEM_KW):
#         return 10, "system_technical"

#     if contains_any(t, PROCESS_KW):
#         return 11, "process_complexity"

#     if contains_any(t, POLICY_KW):
#         return 12, "policy_issue"

#     if contains_any(t, SUPPORT_KW):
#         return 5, "customer_support"

#     # ✅ CLEAN fallback
#     return 14, "general_feedback"

# # ------------------------------------------------------------
# # APPLY
# # ------------------------------------------------------------

# df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

# aspects, flags = [], []

# for _, row in df.iterrows():
#     a, f = assign_aspect(row["text"], row["sentiment"])
#     aspects.append(a)
#     flags.append(f)

# df["primary_aspect"] = aspects
# df["aspect_flag"] = flags

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Logistics Aspect v4 generated successfully")
# print(df["primary_aspect"].value_counts())


# code for aspect sentiment assignment------1regex + KW based with hard rules

# # ============================================================
# # FILE    : logistics_aspect_engine_v6.py
# # PURPOSE : Logistics Aspect Gold Label Generator (Regex + KW)
# # ============================================================

# import pandas as pd
# import re

# INPUT_PATH  = r"D:/bert_data/logistics data/train+intent.csv"
# OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspect_v6.csv"

# # ------------------------------------------------------------
# # ASPECT IDS (LOCKED)
# # ------------------------------------------------------------
# """
# 0  Customer_Service_Support
# 1  Not_Delivered_But_Showing_Delivered
# 2  Product_Quality_Related
# 3  Poor_Delivery_Experience
# 4  High_Courier_Charges
# 5  Additional_Charges_Collected
# 6  Packaging
# 7  Return_Exchange_Order
# 8  Agent_Behaviour
# 9  Product_Deviation
# 10 Generic_Feedback
# 11 Refund_Issue
# 12 Doorstep_Delivery_Needed
# 13 Not_Delivered
# 14 Customer_Query
# 15 Delay_in_Delivery
# 16 Positive_Feedback
# 17 Order_Cancel
# 18 Suggestion
# 19 Delivered_Wrong_Address
# 20 Tampered_Box_Delivered
# 21 Partial_Delivery
# 22 Frequent_Calls_Notifications
# 23 Product_Size_Related
# 24 Pickup_Related
# """

# # ------------------------------------------------------------
# # NORMALIZER
# # ------------------------------------------------------------

# def normalize(text):
#     text = str(text).lower()
#     text = re.sub(r"[^\w\s]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def regex_match(text, pattern):
#     return re.search(pattern, text) is not None


# def contains_any(text, keywords):
#     return any(k in text for k in keywords)


# # ------------------------------------------------------------
# # REGEX PATTERNS (HIGH PRECISION)
# # ------------------------------------------------------------

# PATTERN_NOT_RECEIVED = r"(did\s*not|didnt|didn.?t|never|not)\s+(receive|received|deliver|delivered)"
# PATTERN_SHOWING_DELIVERED = r"(tracking|status|app).*deliver"
# PATTERN_DELAY = r"(late|delay|delayed|waiting|pending).*deliver"
# PATTERN_WRONG_ADDRESS = r"(deliver|delivered).*(wrong address|someone else)"
# PATTERN_REFUND = r"(refund|money back|paise wapas)"
# PATTERN_CANCEL = r"(cancel|cancelled).*(order)?"
# PATTERN_PICKUP = r"(pickup).*(issue|delay|not done|pending)"
# PATTERN_PARTIAL = r"(partial|missing item|half order)"
# PATTERN_TAMPERED = r"(tampered|seal broken|opened box)"
# PATTERN_AGENT_BEHAVIOUR = r"(rude|misbehave|abuse|bad behaviour)"
# PATTERN_CALL_SPAM = r"(too many calls|frequent calls|spam calls|bar bar call)"

# # ------------------------------------------------------------
# # KEYWORD BANKS
# # ------------------------------------------------------------

# POSITIVE_KW = [
#     "good service","great service","excellent","awesome",
#     "thank you","thanks","bahut accha","badiya service",
#     "bahut badiya service","excellent delivery",
#     "delivery fast thi","delivery on time thi",
#     "bahut achhi service","very fast delivery"
# ]

# SUPPORT_KW = [
#     "customer care","support team","helpline","no response",
#     "response nahi","support nahi mil raha",
#     "customer care respond nahi kar raha",
#     "support team reply nahi kar rahi",
#     "call center help nahi kar raha",
#     "helpline connect nahi ho raha",
#     "customer support useless"
# ]

# DELIVERY_KW = [
#     "poor delivery","bad delivery","delivery problem",
#     "delivery bakwas","delivery worst"
# ]

# CHARGE_KW = [
#     "high courier charge","delivery charge high",
#     "expensive delivery","mehenga courier",
#     "delivery bahut mehenga",
#     "courier charge zyada"
# ]

# EXTRA_CHARGE_KW = [
#     "extra charge","additional charge","overcharged",
#     "extra paisa","zyada paisa",
#     "extra paisa liya","double charge"
# ]

# PACKAGING_KW = [
#      "bad packaging","poor packaging","packing kharab",
#     "box phata hua","torn package",
#     "seal broken","open package","opened parcel",
#     "return order","exchange order","replace product",
#     "return karna hai","exchange karna hai",
#     "want to return","replace this item"
# ]

# RETURN_KW = [
#     "return order","exchange order","replace product",
#     "return karna hai","exchange karna hai",
#     "bad quality","poor quality","defective","faulty",
#     "quality kharab",
#     "parcel toot gaya","box toot gaya",
#     "package toot gaya","parcel kharab hai",
#     "box damage hai","product damage hai",
#     "item toot gaya","glass toot gaya",
#     "package leak ho gaya","food leak ho gaya"
# ]

# QUALITY_KW = [
#     "bad quality","poor quality","defective","faulty",
#     "quality kharab"
# ]

# PRODUCT_DEV_KW = [
#     "wrong product","different product","product mismatch",
#     "galat product","product alag hai"
# ]

# SIZE_KW = [
#     "wrong size","size issue","size mismatch"
# ]

# QUERY_KW = [
#     "where is my order","order status","tracking details",
#     "status batao","order kahan hai",
#         "where is my order","order status","tracking details",
#     "status batao","order kahan hai",
#     "parcel kaha hai","delivery kaha hai",
#     "tracking batao"
# ]

# SUGGESTION_KW = [
#     "suggestion","recommend","improve service",    "suggestion","recommend","improve service",
#     "service improve karo"
# ]

# DOORSTEP_KW = [
#     "doorstep delivery","deliver upstairs","ghar tak delivery"
# ]


# # ------------------------------------------------------------
# # ASPECT ASSIGNMENT
# # ------------------------------------------------------------

# def assign_aspect(text):

#     t = normalize(text)

#     # -------- POSITIVE FEEDBACK --------
#     if contains_any(t, POSITIVE_KW):
#         return 16

#     # -------- NOT DELIVERED BUT SHOWING --------
#     if regex_match(t, PATTERN_SHOWING_DELIVERED) and regex_match(t, PATTERN_NOT_RECEIVED):
#         return 1

#     # -------- WRONG ADDRESS --------
#     if regex_match(t, PATTERN_WRONG_ADDRESS):
#         return 19

#     # -------- NOT DELIVERED --------
#     if regex_match(t, PATTERN_NOT_RECEIVED):
#         return 13

#     # -------- DELAY --------
#     if regex_match(t, PATTERN_DELAY):
#         return 15

#     # -------- AGENT BEHAVIOUR --------
#     if regex_match(t, PATTERN_AGENT_BEHAVIOUR):
#         return 8

#     # -------- REFUND --------
#     if regex_match(t, PATTERN_REFUND):
#         return 11

#     # -------- CANCEL --------
#     if regex_match(t, PATTERN_CANCEL):
#         return 17

#     # -------- PICKUP --------
#     if regex_match(t, PATTERN_PICKUP):
#         return 24

#     # -------- PARTIAL DELIVERY --------
#     if regex_match(t, PATTERN_PARTIAL):
#         return 21

#     # -------- TAMPERED BOX --------
#     if regex_match(t, PATTERN_TAMPERED):
#         return 20

#     # -------- FREQUENT CALLS --------
#     if regex_match(t, PATTERN_CALL_SPAM):
#         return 22

#     # -------- KEYWORD FALLBACKS --------

#     if contains_any(t, RETURN_KW):
#         return 7

#     if contains_any(t, QUALITY_KW):
#         return 2

#     if contains_any(t, PACKAGING_KW):
#         return 6

#     if contains_any(t, PRODUCT_DEV_KW):
#         return 9

#     if contains_any(t, SIZE_KW):
#         return 23

#     if contains_any(t, CHARGE_KW):
#         return 4

#     if contains_any(t, EXTRA_CHARGE_KW):
#         return 5

#     if contains_any(t, SUPPORT_KW):
#         return 0

#     if contains_any(t, QUERY_KW):
#         return 14

#     if contains_any(t, SUGGESTION_KW):
#         return 18

#     if contains_any(t, DELIVERY_KW):
#         return 3

#     if contains_any(t, DOORSTEP_KW):
#         return 12

#     # -------- DEFAULT --------
#     return 10


# # ------------------------------------------------------------
# # APPLY ENGINE
# # ------------------------------------------------------------

# df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

# df["primary_aspect"] = df["text"].apply(assign_aspect)

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Logistics Aspect v6 generated successfully")
# print(df["primary_aspect"].value_counts())

# ============================================================
# FILE    : logistics_aspect_engine_v7.py
# PURPOSE : Logistics Aspect Gold Label Generator (Regex + KW)
# ============================================================

import pandas as pd
import re

INPUT_PATH  = r"D:/bert_data/data logistics new/train+intent.csv"
OUTPUT_PATH = r"D:/bert_data/data logistics new/train+intent+aspect.csv"

# ------------------------------------------------------------
# NORMALIZER
# ------------------------------------------------------------

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_any(text, keywords):
    return any(k in text for k in keywords)

# ------------------------------------------------------------
# REGEX PATTERNS (COMPILED)
# ------------------------------------------------------------

PATTERN_NOT_RECEIVED = re.compile(
    r"(did\s*not|didnt|didn.?t|never|still not)\s+(receive|received)"
)

PATTERN_SHOWING_DELIVERED = re.compile(
    r"(tracking|status|app).*(deliver|delivered)"
)

PATTERN_NOT_DELIVERED = re.compile(
    r"(not|still not|yet not)\s+(deliver|delivered)"
)

PATTERN_DELAY = re.compile(
    r"(late|delay|delayed|waiting|pending|overdue)"
)

PATTERN_WRONG_ADDRESS = re.compile(
    r"(wrong address|someone else|neighbour|security guard)"
)

PATTERN_REFUND = re.compile(
    r"(refund|money back|paise wapas)"
)

PATTERN_CANCEL = re.compile(
    r"(cancel|cancelled).*(order)?"
)

PATTERN_PICKUP = re.compile(
    r"(pickup).*(issue|delay|not done|pending)"
)

PATTERN_PARTIAL = re.compile(
    r"(partial|missing item|half order)"
)

PATTERN_TAMPERED = re.compile(
    r"(tampered|seal broken|opened box)"
)

PATTERN_AGENT_BEHAVIOUR = re.compile(
    r"(rude|misbehave|abuse|bad behaviour|unprofessional|badtameez)"
)

PATTERN_CALL_SPAM = re.compile(
    r"(too many calls|frequent calls|spam calls|bar bar call)"
)

# ------------------------------------------------------------
# KEYWORD BANKS (EXPANDED)
# ------------------------------------------------------------

POSITIVE_KW = [
    "good service","great service","excellent","awesome",
    "thank you","thanks","bahut accha","badiya service",
    "excellent delivery","delivery fast thi","delivery on time thi",
    "bahut achhi service","very fast delivery"
]

SUPPORT_KW = [
    "customer care","support team","helpline",
    "no response","response nahi",
    "support nahi mil raha","customer care respond nahi kar raha",
    "support team reply nahi kar rahi",
    "call center help nahi kar raha"
]

DELIVERY_KW = [
    "poor delivery","bad delivery","delivery problem",
    "delivery bakwas","delivery worst",
    "delivery slow","service slow"
]

CHARGE_KW = [
    "high courier charge","delivery charge high",
    "expensive delivery","mehenga courier",
    "courier charge zyada"
]

EXTRA_CHARGE_KW = [
    "extra charge","additional charge","overcharged",
    "extra paisa","zyada paisa",
    "extra paisa liya","double charge"
]

PACKAGING_KW = [
    "bad packaging","poor packaging",
    "packing kharab","torn package",
    "box phata hua","open package"
]

RETURN_KW = [
    "return order","exchange order","replace product",
    "return karna hai","exchange karna hai",
    "want to return","replace this item"
]

QUALITY_KW = [
    "bad quality","poor quality","defective","faulty",
    "quality kharab","parcel toot gaya",
    "box toot gaya","package toot gaya",
    "parcel kharab hai","product damage hai",
    "glass toot gaya","food leak ho gaya"
]

PRODUCT_DEV_KW = [
    "wrong product","different product",
    "product mismatch","galat product"
]

SIZE_KW = [
    "wrong size","size issue","size mismatch",
    "size galat","size problem"
]

QUERY_KW = [
    "where is my order","order status",
    "tracking details","status batao",
    "order kahan hai","parcel kaha hai",
    "delivery kaha hai"
]

SUGGESTION_KW = [
    "suggestion","recommend",
    "improve service","service improve karo"
]

DOORSTEP_KW = [
    "doorstep delivery","deliver upstairs",
    "ghar tak delivery","flat delivery"
]

NOT_DELIVERED_KW = [
    "parcel nahi mila","courier nahi mila",
    "delivery nahi hui","parcel missing hai",
    "item missing","didnt receive parcel",
    "never got parcel"
]

WRONG_ADDRESS_KW = [
    "galat address deliver","kisi aur ko deliver",
    "neighbour ko deliver","security guard ko parcel diya"
]

PICKUP_KW = [
    "pickup nahi hua","pickup pending hai",
    "pickup delay ho gaya","pickup agent nahi aaya"
]

# ------------------------------------------------------------
# ASPECT ASSIGNMENT
# ------------------------------------------------------------

def assign_aspect(text):

    t = normalize(text)

    # Positive feedback
    if contains_any(t, POSITIVE_KW):
        return 16

    # Showing delivered but not received
    if PATTERN_SHOWING_DELIVERED.search(t) and PATTERN_NOT_RECEIVED.search(t):
        return 1

    # Wrong address delivery
    if PATTERN_WRONG_ADDRESS.search(t) or contains_any(t, WRONG_ADDRESS_KW):
        return 19

    # Partial delivery
    if PATTERN_PARTIAL.search(t):
        return 21

    # Tampered package
    if PATTERN_TAMPERED.search(t):
        return 20

    # Not delivered
    if PATTERN_NOT_RECEIVED.search(t) or contains_any(t, NOT_DELIVERED_KW):
        return 13

    # Delay
    if PATTERN_DELAY.search(t):
        return 15

    # Agent behaviour
    if PATTERN_AGENT_BEHAVIOUR.search(t):
        return 8

    # Refund issue
    if PATTERN_REFUND.search(t):
        return 11

    # Cancel order
    if PATTERN_CANCEL.search(t):
        return 17

    # Pickup
    if PATTERN_PICKUP.search(t) or contains_any(t, PICKUP_KW):
        return 24

    # Frequent calls
    if PATTERN_CALL_SPAM.search(t):
        return 22

    # Keyword fallback rules

    if contains_any(t, RETURN_KW):
        return 7

    if contains_any(t, QUALITY_KW):
        return 2

    if contains_any(t, PACKAGING_KW):
        return 6

    if contains_any(t, PRODUCT_DEV_KW):
        return 9

    if contains_any(t, SIZE_KW):
        return 23

    if contains_any(t, CHARGE_KW):
        return 4

    if contains_any(t, EXTRA_CHARGE_KW):
        return 5

    if contains_any(t, SUPPORT_KW):
        return 0

    if contains_any(t, QUERY_KW):
        return 14

    if contains_any(t, SUGGESTION_KW):
        return 18

    if contains_any(t, DELIVERY_KW):
        return 3

    if contains_any(t, DOORSTEP_KW):
        return 12

    # Default
    return 10


# ------------------------------------------------------------
# APPLY ENGINE
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

df["primary_aspect"] = df["text"].apply(assign_aspect)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Logistics Aspect v7 generated successfully")
print(df["primary_aspect"].value_counts())