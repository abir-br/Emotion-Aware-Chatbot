import sys
import json
import logging
import pandas as pd
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────
# LOGGING
# ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────
# FSM STATES
# ─────────────────────────────
STATE_START = "START"
STATE_SUPPORT = "SUPPORTING"
STATE_DEESCALATE = "DEESCALATING"
STATE_PAUSE = "SUGGESTING_PAUSE"
STATE_END = "END"

# ─────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────
@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    vad: dict
    reaction: str

# ─────────────────────────────
# CSV (HUMAN POLICY)
# ─────────────────────────────
try:
    logic_df = pd.read_csv("My_Aggregated_Logic.csv")
    logging.info("CSV loaded (%d rules)", len(logic_df))
except FileNotFoundError:
    logging.warning("CSV not found")
    logic_df = pd.DataFrame()


def get_next_action_csv(current_action, reaction):

    if logic_df.empty:
        return None

    match = logic_df[
        (logic_df["Current_System_Action"] == current_action) &
        (logic_df["User_Reaction"] == reaction)
    ]

    if match.empty:
        return None

    best = match.sort_values(
        by="Agreement_Percentage",
        ascending=False
    ).iloc[0]

    return best["Next_System_Action"]

# ─────────────────────────────
# VAD-BERT MODEL
# ─────────────────────────────
MODEL_NAME = "RobroKools/vad-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ─────────────────────────────
# VAD → EMOTION
# ─────────────────────────────
def vad_to_emotion(v, a, d):

    if v < 2.5 and a > 3.5:
        return "anger"
    if v < 2.5 and d < 2.5:
        return "sadness"
    if v < 2.5:
        return "fear"
    if v > 3.5 and a > 3.5:
        return "surprise"
    if v < 2.0:
        return "disgust"

    return "joy"

# ─────────────────────────────
# VAD → USER REACTION (CSV)
# ─────────────────────────────
def vad_to_reaction(v, a, d):

    if a > 3.5 and d > 3.0:
        return "rejects_help"

    if v < 2.5:
        return "stays_negative"

    return "opens_up"

# ─────────────────────────────
# CONFIDENCE
# ─────────────────────────────
def compute_confidence(v, a, d):
    dist = ((v - 3)**2 + (a - 3)**2 + (d - 3)**2) ** 0.5
    return min(dist / 3.46, 1.0)

# ─────────────────────────────
# VAD PREDICTION
# ─────────────────────────────
def predict_vad(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    v, a, d = outputs.logits.squeeze().tolist()

    v, a, d = round(v, 2), round(a, 2), round(d, 2)

    emotion = vad_to_emotion(v, a, d)
    reaction = vad_to_reaction(v, a, d)
    confidence = compute_confidence(v, a, d)

    return EmotionResult(
        emotion=emotion,
        confidence=confidence,
        vad={"valence": v, "arousal": a, "dominance": d},
        reaction=reaction
    )

# ─────────────────────────────
# FSM FALLBACK
# ─────────────────────────────
def fsm_fallback(emotion, reaction):

    if emotion == "anger":
        return "slow_down"

    if emotion in ["sadness", "fear"]:
        return "offer_support"

    if reaction == "rejects_help":
        return "suggest_pause"

    return "ask_clarification"

# ─────────────────────────────
# MESSAGES
# ─────────────────────────────
def generate_message(action, emotion):

    messages = {
        "offer_support": f"I’m here for you. I sense {emotion}.",
        "slow_down": "Let’s slow down a bit.",
        "ask_clarification": "Could you clarify?",
        "suggest_pause": "Maybe take a break.",
        "encourage": "You’re doing well.",
        "continue": "Let’s continue."
    }

    return messages.get(action, "I'm listening.")

# ─────────────────────────────
# FSM TRANSITIONS
# ─────────────────────────────
TRANSITIONS = {
    (STATE_START, "offer_support"): STATE_SUPPORT,
    (STATE_START, "slow_down"): STATE_DEESCALATE,

    (STATE_SUPPORT, "slow_down"): STATE_DEESCALATE,
    (STATE_SUPPORT, "suggest_pause"): STATE_PAUSE,

    (STATE_DEESCALATE, "suggest_pause"): STATE_PAUSE,

    (STATE_PAUSE, "continue"): STATE_END,
}


def next_state(state, action):
    if state == STATE_END:
        return STATE_END
    return TRANSITIONS.get((state, action), state)

# ─────────────────────────────
# MEMORY
# ─────────────────────────────
history = []

def update_memory(text, emotion):
    history.append((text, emotion))
    if len(history) > 5:
        history.pop(0)

# ─────────────────────────────
# FUSION ENGINE
# ─────────────────────────────
def decide_action(result, current_action):

    csv_action = get_next_action_csv(current_action, result.reaction)
    fsm_action = fsm_fallback(result.emotion, result.reaction)

    final_action = csv_action if csv_action else fsm_action

    return final_action, fsm_action, csv_action

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():

    state = STATE_START
    current_action = "acknowledge"
    turn = 0

    print("=== FINAL VAD-BERT CHATBOT ===")

    for line in sys.stdin:

        text = line.strip()
        if not text:
            continue

        # 1. VAD + Emotion + Reaction
        result = predict_vad(text)

        # 2. Decision (CSV + FSM)
        action, fsm_action, csv_action = decide_action(result, current_action)

        # 3. FSM update
        state = next_state(state, action)

        # 4. Message
        message = generate_message(action, result.emotion)

        # 5. Memory
        update_memory(text, result.emotion)

        # 6. Output
        output = {
            "turn": turn + 1,
            "user_text": text,
            "vad_scores": result.vad,
            "emotion": result.emotion,
            "confidence": result.confidence,
            "user_reaction": result.reaction,
            "system_action": action,
            "fsm_action": fsm_action,
            "csv_action": csv_action,
            "state": state,
            "message": message,
            "history": history
        }

        print(json.dumps(output, indent=2))

        current_action = action
        turn += 1

        if state == STATE_END or turn >= 3:
            print("\n[SYSTEM] DONE")
            break


if __name__ == "__main__":
    main()