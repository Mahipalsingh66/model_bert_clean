# ============================================================
# FILE    : banking_aspect_engine_v2.py
# PURPOSE : Banking Industry Primary Aspect Classification
# AUTHOR  : Production Locked
#
# INDUSTRY : BANKING
#
# PRIMARY ASPECT TAXONOMY (LOCKED)
# 0  = Transaction_Issue
# 1  = Charges
# 2  = Loan_Credit
# 3  = Mobile_App
# 4  = Staff_Negative
# 5  = Appreciation
# 6  = Customer_Service
# 7  = ATM
# 8  = Minimum_Balance
# 9  = Branch
# 10 = Interest_Rate
# 11 = Offers
# 12 = Security
# 13 = General
#14 = Others (Fallback)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/Banking_data/Base_data.csv"
OUTPUT_PATH = r"D:/bert_data/Banking_data/bank_train_with_aspect_2.csv"

# ------------------------------------------------------------
# KEYWORD BANK (BANKING – ENGLISH + HINGLISH)
# ------------------------------------------------------------

TRANSACTION_KW = [
    # English
    "transaction failed", "amount debited", "amount credited",
    "money deducted", "double debit", "pending transaction",
    "reversal pending", "payment failed", "upi failed",
    "transfer failed", "imps failed", "neft failed",
    "rtgs failed", "wrong debit", "transaction stuck",
    # Hinglish
    "paise kat gaye", "amount kat gaya", "transaction atak gaya",
    "upi nahi chala", "payment nahi hua", "double paisa kata",
    "refund pending", "reversal nahi aaya"
]

CHARGES_KW = [
    # English
    "charge", "charges", "fee", "fees", "penalty",
    "hidden charges", "extra charges", "maintenance fee",
    "service charge", "sms charge", "annual charge",
    "overcharged", "wrong charges",
    # Hinglish
    "extra paisa", "faltu charges", "galat charge",
    "charges kaat liya", "penalty laga di",
    "balance charge", "fees zyada"
]

LOAN_KW = [
    # English
    "loan", "credit", "emi", "installment",
    "loan rejected", "loan approval", "loan disbursal",
    "credit limit", "credit card limit",
    "personal loan", "home loan", "education loan",
    "emi bounce", "emi issue",
    # Hinglish
    "loan reject", "loan approve nahi hua",
    "emi kat gayi", "emi bounce", "loan pass nahi hua",
    "credit limit kam", "loan amount nahi mila"
]

MOBILE_APP_KW = [
    # English
    "app", "mobile app", "banking app",
    "app crash", "app not working", "login issue",
    "otp not received", "app slow", "technical issue",
    "server down", "update issue",
    # Hinglish
    "app nahi chal raha", "login nahi ho raha",
    "otp nahi aaya", "app slow hai",
    "server down hai", "app crash ho raha"
]

STAFF_NEGATIVE_KW = [
    # English
    "rude staff", "staff rude", "misbehave",
    "bad behaviour", "unprofessional staff",
    "branch staff rude", "employee misbehaved",
    "manager rude", "harassment",
    # Hinglish
    "staff badtameez", "staff rude hai",
    "galat behaviour", "ache se baat nahi ki",
    "manager badtameez", "misbehave kiya"
]

APPRECIATION_KW = [
    # English
    "good service", "nice service", "excellent service",
    "very good", "happy", "satisfied",
    "great experience", "thank you",
    "staff was helpful", "awesome service",
    "smooth process",
    # Hinglish
    "bahut acha service", "badiya service",
    "kaafi acha", "staff helpful tha",
    "process smooth tha", "thank you bank"
]

CUSTOMER_SERVICE_KW = [
    # English
    "customer care", "customer support", "support team",
    "helpline", "call center", "complaint",
    "ticket raised", "case id", "no response",
    "not responding", "waiting for response",
    # Hinglish
    "customer care call", "support nahi mil raha",
    "call nahi uthaya", "response nahi aaya",
    "complaint ki thi", "helpline busy"
]

ATM_KW = [
    # English
    "atm", "atm card", "debit card",
    "card blocked", "card not working",
    "cash not dispensed", "atm failed",
    "card retained", "pin issue",
    # Hinglish
    "atm paisa nahi diya", "card block ho gaya",
    "debit card nahi chal raha",
    "pin issue hai", "atm ne card rakh liya"
]

MIN_BALANCE_KW = [
    # English
    "minimum balance", "min balance",
    "low balance", "balance requirement",
    "zero balance", "average balance",
    "penalty for low balance",
    # Hinglish
    "minimum balance", "kam balance",
    "zero balance account", "balance kam hai",
    "min balance maintain nahi hua"
]

BRANCH_KW = [
    # English
    "branch", "bank branch",
    "branch visit", "branch manager",
    "branch service", "local branch",
    "branch office",
    # Hinglish
    "branch gaya tha", "branch manager se baat",
    "bank branch me", "local branch"
]

INTEREST_RATE_KW = [
    # English
    "interest rate", "interest charged",
    "high interest", "low interest",
    "rate of interest", "roi",
    "interest calculation",
    # Hinglish
    "interest zyada", "interest kam",
    "byaj zyada", "byaj kam",
    "interest galat laga"
]

OFFERS_KW = [
    # English
    "offer", "offers", "cashback",
    "reward", "rewards", "discount",
    "promo", "promotion", "benefits",
    # Hinglish
    "cashback nahi mila", "offer nahi mila",
    "reward points", "discount nahi diya"
]

SECURITY_KW = [
    # English
    "fraud", "scam", "hacked",
    "unauthorized", "security issue",
    "account hacked", "phishing",
    "otp fraud", "cyber fraud",
    # Hinglish
    "fraud hua", "scam ho gaya",
    "account hack ho gaya",
    "otp fraud", "unauthorized transaction",
    "paise chori ho gaye"
]

# ------------------------------------------------------------
# ASPECT ASSIGNMENT (STRICT PRIORITY)
# ------------------------------------------------------------

def assign_banking_aspect(text: str):
    if not isinstance(text, str) or not text.strip():
        return 13, "general"

    t = text.lower()

    if any(k in t for k in SECURITY_KW):
        return 12, "security"

    if any(k in t for k in TRANSACTION_KW):
        return 0, "transaction_issue"

    if any(k in t for k in LOAN_KW):
        return 2, "loan_credit"

    if any(k in t for k in ATM_KW):
        return 7, "atm"

    if any(k in t for k in MOBILE_APP_KW):
        return 3, "mobile_app"

    if any(k in t for k in CHARGES_KW):
        return 1, "charges"

    if any(k in t for k in MIN_BALANCE_KW):
        return 8, "minimum_balance"

    if any(k in t for k in INTEREST_RATE_KW):
        return 10, "interest_rate"

    if any(k in t for k in OFFERS_KW):
        return 11, "offers"

    if any(k in t for k in STAFF_NEGATIVE_KW):
        return 4, "staff_negative"

    if any(k in t for k in CUSTOMER_SERVICE_KW):
        return 6, "customer_service"

    if any(k in t for k in BRANCH_KW):
        return 9, "branch"

    if any(k in t for k in APPRECIATION_KW):
        return 5, "appreciation"

    return 13, "general"

# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    encoding="latin1",
    sep=",",
    engine="python",
    on_bad_lines="warn"
)

df.columns = df.columns.str.strip().str.lower()

if "text" not in df.columns:
    raise ValueError("Missing required column: text")

aspects, flags = [], []

for text in df["text"]:
    asp, flag = assign_banking_aspect(text)
    aspects.append(asp)
    flags.append(flag)

df["primary_aspect"] = aspects
df["aspect_flag"] = flags

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Banking aspect labels (EN + Hinglish) generated")
print("\n--- Aspect Distribution ---")
print(df["primary_aspect"].value_counts())