# Golden Dataset v1 – Sentiment Labeling Rules

Task: Sentiment classification for customer feedback.

Labels:
0 = Negative
1 = Neutral
2 = Positive

## Positive (2)
Use when:
- Clear satisfaction or praise
- Successful delivery or service
- Polite/helpful behavior
- Words like: good, excellent, happy, smooth, fast, satisfied

## Negative (0)
Use when:
- Non-delivery, delay, damage, fraud
- Rude behavior or complaint
- Wrong item or missing item
- Strong dissatisfaction or anger

## Neutral (1)
Use ONLY when:
- No emotion (pure information)
- Status updates, IDs, names
- Suggestions without praise or anger

### Critical Rules
- Neutral is NOT a safe label
- If emotion exists → NOT Neutral
- Mixed but emotional → choose Positive or Negative
- If unsure → re-check, do NOT guess
