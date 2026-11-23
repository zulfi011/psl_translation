"""
step7_llm.py
=============
PURPOSE : Real-time PSL sign recognition from webcam/video PLUS an OpenRouter
          LLM post-processing layer that turns detected sign tokens into
          fluent, grammatically correct English sentences.

HOW IT WORKS
------------
  1. All the core capture / MediaPipe / TCN inference logic is identical to
     step6_realtime.py — nothing there has been changed.
  2. Every time a new high-confidence sign is accepted into the buffer it is
     accumulated alongside its confidence score.
  3. When the buffer holds at least MIN_SIGNS_FOR_LLM distinct signs, or when
     the user presses  L  (trigger LLM manually), the accumulated sign tokens
     are sent to an OpenRouter-hosted LLM with a carefully engineered prompt.
  4. The LLM returns THREE candidate sentences ranked by likelihood.  These
     are displayed in a dedicated HUD panel on the live video feed.
  5. The LLM is called asynchronously so the video feed never freezes.

Keys
----
  ESC  – quit
  C    – clear sentence, sign buffer, and LLM result
  L    – force an immediate LLM call with whatever signs are buffered

OpenRouter setup
----------------
  1. Create a free account at https://openrouter.ai
  2. Generate an API key from the dashboard.
  3. Put it in fyp/.env:  OPENROUTER_API_KEY=sk-or-...
     (see .env.example)

Model choice
------------
  The default model is "meta-llama/llama-3.3-70b:free" (free tier, no credits
  needed).  Other working free models as of May 2026:
      "meta-llama/llama-4-maverick:free"
      "deepseek/deepseek-r1:free"
      "mistralai/mistral-7b-instruct:free"
  Paid alternatives (require OpenRouter credits):
      "openai/gpt-4o-mini"
      "anthropic/claude-3-haiku"
  Full list: https://openrouter.ai/models  (filter by "free")

  NOTE: Free model availability changes without notice. If you get a 404,
  visit openrouter.ai/models, filter by ":free", and paste a working ID below.
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import cv2
import json
import threading
import time
import collections
import textwrap

# ── third-party ────────────────────────────────────────────────────────────────
import numpy as np
import torch
import mediapipe as mp
import requests          # pip install requests  (usually already present)

# ── local ──────────────────────────────────────────────────────────────────────
from psl_translator import config
from psl_translator.features import create_holistic, frame_to_features, fill_zero_frames
from psl_translator.model    import SignTCN


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIGURATION                                   ║
# ╠══════════════════════════════════════════════════════════════════════════════╣

OPENROUTER_API_KEY        = config.OPENROUTER_API_KEY   # set in .env / environment
OPENROUTER_MODEL          = config.OPENROUTER_MODEL
OPENROUTER_MODEL_FALLBACK = "openai/gpt-4o-mini"# used on 404

# ── paths ──────────────────────────────────────────────────────────────────────
LABELS_JSON = config.LABELS_JSON
MODEL_PATH  = config.MODEL_PATH

# ── video source ───────────────────────────────────────────────────────────────
# Use a file path for a recorded video, or 0 for live webcam
# webcam by default; set PSL_VIDEO_SOURCE=path/to/video.mp4 to use a recorded clip
VIDEO_SOURCE = config.video_source(0)

# ── recognition thresholds ─────────────────────────────────────────────────────
CONFIDENCE_THRESH  = 0.50   # minimum TCN softmax score to accept a sign token
MIN_SIGNS_FOR_LLM  = 2      # auto-trigger LLM when this many unique signs seen
                             # (set higher if you want to accumulate more tokens
                             #  before querying the LLM)
LLM_COOLDOWN_SEC   = 6.0    # minimum seconds between automatic LLM calls
                             # (prevents spamming the API every second)

# ── sliding window ─────────────────────────────────────────────────────────────
SEQUENCE_LEN    = 60     # frames sub-sampled for model input
SIGN_WINDOW_SEC = 3.0    # rolling window length in seconds
ADVANCE_SEC     = 1.0    # slide window every this many seconds

# ╚══════════════════════════════════════════════════════════════════════════════╝


if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set — LLM sentences will fail. See .env.example")

# ── label map ──────────────────────────────────────────────────────────────────
with open(LABELS_JSON) as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
num_classes  = len(label_map)

# ── model ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SignTCN(input_size=258, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── MediaPipe Holistic ─────────────────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_draw        = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

holistic = create_holistic()

# ── camera ─────────────────────────────────────────────────────────────────────
cap     = cv2.VideoCapture(VIDEO_SOURCE)
cam_fps = cap.get(cv2.CAP_PROP_FPS)
if cam_fps <= 0:
    cam_fps = 30.0

WINDOW_FRAMES  = int(cam_fps * SIGN_WINDOW_SEC)
ADVANCE_FRAMES = int(cam_fps * ADVANCE_SEC)

print(f"Camera FPS     : {cam_fps:.1f}")
print(f"Window frames  : {WINDOW_FRAMES}  ({SIGN_WINDOW_SEC}s)")
print(f"Advance frames : {ADVANCE_FRAMES} ({ADVANCE_SEC}s)")
print(f"LLM model      : {OPENROUTER_MODEL}")


# ══════════════════════════════════════════════════════════════════════════════
#  LLM INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(sign_tokens: list[dict]) -> str:
    """
    Build the system + user prompt sent to the LLM.

    sign_tokens is a list of dicts:
        [{"word": "i",    "confidence": 0.92},
         {"word": "go",   "confidence": 0.87},
         {"word": "home", "confidence": 0.78}, ...]

    The prompt is engineered so the model:
      • understands the input is raw sign-language token stream (not grammar)
      • knows PSL/ASL frequently omits articles, copulas, and tense markers
      • returns EXACTLY three candidate sentences, ranked 1-3
      • explains its reasoning briefly
    """

    token_str = ", ".join(
        f'"{t["word"]}" (conf {t["confidence"]:.0%})'
        for t in sign_tokens
    )

    system_prompt = textwrap.dedent("""\
        You are an expert interpreter specialising in Pakistani Sign Language (PSL)
        and American Sign Language (ASL).

        CONTEXT
        -------
        A computer-vision model has detected a sequence of signs performed by a signer.
        The detector outputs raw sign tokens — individual glosses (sign labels) in the
        rough temporal order they were performed, along with a confidence score for each.

        IMPORTANT LINGUISTIC NOTES
        --------------------------
        • Sign languages use a topic-comment grammar that often omits articles
          ("a", "the"), auxiliary verbs ("is", "are", "am"), and sometimes
          subject pronouns.
        • Tense is often implied by context rather than explicit verb morphology.
        • The confidence score indicates how certain the vision model is about
          each token.  A low-confidence token may be misidentified — consider
          plausible alternatives if a sentence would not make sense otherwise.
        • Tokens are listed in temporal order, but a signer may repeat a sign
          for emphasis, so identical consecutive tokens usually count as one sign.

        YOUR TASK
        ---------
        Given the detected sign tokens, reconstruct the most likely English
        sentence(s) the signer intended to communicate.

        OUTPUT FORMAT  (follow this EXACTLY — no extra text)
        -------------
        Sentence 1: <most likely English sentence>
        Sentence 2: <second most likely English sentence>
        Sentence 3: <third most likely English sentence>
        Reasoning: <one or two sentences explaining your choices>
    """)

    user_prompt = textwrap.dedent(f"""\
        Detected sign tokens (in order):
        {token_str}

        Please provide three candidate English sentences the signer is most
        likely saying, ranked from most to least probable.
    """)

    return system_prompt, user_prompt


def call_openrouter_llm(sign_tokens: list[dict]) -> str:
    """
    Send sign tokens to OpenRouter and return the raw response text.
    Runs in a background thread so it never blocks the video loop.
    """
    system_prompt, user_prompt = build_llm_prompt(sign_tokens)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/psl-sign-recognition",  # optional
        "X-Title":       "PSL Sign Recognition",                      # optional
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.3,   # low temp → more deterministic / consistent output
        "max_tokens":  300,
    }

    url = "https://openrouter.ai/api/v1/chat/completions"

    for attempt, model_id in enumerate([OPENROUTER_MODEL, OPENROUTER_MODEL_FALLBACK]):
        payload["model"] = model_id
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)

            # if primary model is gone, silently retry with fallback
            if resp.status_code == 404 and attempt == 0:
                print(f"[LLM] Primary model 404 — retrying with fallback: "
                      f"{OPENROUTER_MODEL_FALLBACK}")
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            return "LLM Error: Request timed out."
        except requests.exceptions.HTTPError:
            if attempt == 1:   # both models failed
                return f"LLM Error: HTTP {resp.status_code} — {resp.text[:160]}"
            continue
        except Exception as e:
            return f"LLM Error: {e}"

    return "LLM Error: All models unavailable — check openrouter.ai/models for free IDs."


# ── shared state for async LLM calls ──────────────────────────────────────────
llm_result_text   = ""        # latest LLM output (updated by background thread)
llm_is_running    = False     # True while a request is in-flight
llm_last_call_ts  = 0.0       # unix timestamp of last LLM call


def _llm_thread_fn(tokens_snapshot):
    """Background worker: call LLM, store result, clear running flag."""
    global llm_result_text, llm_is_running
    llm_result_text = "⏳ Thinking…"
    result          = call_openrouter_llm(tokens_snapshot)
    llm_result_text = result
    llm_is_running  = False
    print("\n── LLM Result ──────────────────────────────")
    print(result)
    print("────────────────────────────────────────────\n")


def trigger_llm(sign_token_buffer: list[dict]):
    """Spawn a background thread to call the LLM (non-blocking)."""
    global llm_is_running, llm_last_call_ts
    if llm_is_running:
        return   # already a call in-flight, skip
    llm_is_running   = True
    llm_last_call_ts = time.time()
    t = threading.Thread(
        target=_llm_thread_fn,
        args=(list(sign_token_buffer),),  # snapshot so main loop can keep going
        daemon=True,
    )
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (identical to step6_realtime.py)
# ══════════════════════════════════════════════════════════════════════════════

def draw_landmarks(frame, results):
    """Draw hand and pose skeletons. Face is intentionally NOT drawn."""
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style(),
        )
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style(),
        )
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw_styles.get_default_pose_landmarks_style(),
        )


def run_inference():
    """Sub-sample rolling buffer → model → (label, confidence)."""
    if len(frame_buffer) < WINDOW_FRAMES:
        return None, 0.0

    buf      = list(frame_buffer)
    indices  = np.linspace(0, len(buf) - 1, SEQUENCE_LEN, dtype=int)
    sequence = np.array([buf[i] for i in indices], dtype=np.float32)
    sequence = fill_zero_frames(sequence)

    x = torch.tensor([sequence], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits     = model(x)
        probs      = torch.softmax(logits, dim=1)
        confidence = probs.max().item()
        pred_idx   = probs.argmax().item()

    return idx_to_label[pred_idx], confidence


# ══════════════════════════════════════════════════════════════════════════════
#  HUD DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_multiline_text(frame, text, x, y, font_scale, color, thickness,
                        line_spacing=22, max_width=None):
    """
    Draw text that may contain '\n' line breaks, or wrap automatically if
    max_width (pixels) is given.  Returns the y coordinate after last line.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = []
    for raw_line in text.split("\n"):
        if max_width is None:
            lines.append(raw_line)
        else:
            # crude word-wrap
            words  = raw_line.split()
            cur    = ""
            for w in words:
                test = cur + (" " if cur else "") + w
                (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
                if tw <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)

    for line in lines:
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness,
                    cv2.LINE_AA)
        y += line_spacing
    return y


def draw_llm_panel(frame, llm_text, sign_tokens):
    """
    Draw a semi-transparent panel on the right side of the frame showing:
      • the sign tokens that were sent to the LLM
      • the three candidate sentences returned
    """
    h, w = frame.shape[:2]
    panel_x     = max(w - 460, 0)
    panel_y     = 160
    panel_w     = 455
    panel_h     = h - panel_y - 20

    # semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (80, 200, 120), 1)

    tx = panel_x + 8
    ty = panel_y + 22

    # header
    cv2.putText(frame, "LLM Sentence Reconstruction",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (80, 200, 120), 1, cv2.LINE_AA)
    ty += 4
    cv2.line(frame, (tx, ty), (panel_x + panel_w - 8, ty), (80, 200, 120), 1)
    ty += 14

    # tokens sent
    if sign_tokens:
        token_display = "Signs: " + " · ".join(
            f'{t["word"]}({t["confidence"]:.0%})' for t in sign_tokens
        )
        ty = draw_multiline_text(frame, token_display, tx, ty,
                                 0.38, (200, 200, 200), 1,
                                 line_spacing=16,
                                 max_width=panel_w - 16)
        ty += 6

    # LLM output lines
    if llm_text:
        ty = draw_multiline_text(frame, llm_text, tx, ty,
                                 0.42, (255, 255, 255), 1,
                                 line_spacing=18,
                                 max_width=panel_w - 16)

    if llm_is_running:
        cv2.putText(frame, "⏳ Calling LLM…",
                    (tx, panel_y + panel_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 200, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME STATE
# ══════════════════════════════════════════════════════════════════════════════

frame_buffer           = collections.deque(maxlen=WINDOW_FRAMES)
frames_since_inference = 0

sentence           = []       # accepted words (displayed top bar)
seen_words         = set()    # de-duplication set

current_prediction = ""
current_confidence = 0.0

# LLM token buffer: ordered list of {"word": ..., "confidence": ...}
# Each newly accepted unique word is appended here.
llm_token_buffer   = []


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("\nControls:  ESC = quit   |   C = clear   |   L = force LLM call\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    features = frame_to_features(results)
    frame_buffer.append(features)
    frames_since_inference += 1

    draw_landmarks(frame, results)

    # ── run TCN inference every ADVANCE_FRAMES ────────────────────────────────
    if frames_since_inference >= ADVANCE_FRAMES:
        frames_since_inference = 0

        word, confidence = run_inference()

        if word is not None:
            current_prediction = word
            current_confidence = confidence

            print(f"Prediction: {word:<14}  conf: {confidence:.2f}")

            if confidence >= CONFIDENCE_THRESH and word not in seen_words:
                sentence.append(word)
                seen_words.add(word)

                # also add to LLM token buffer (with confidence)
                llm_token_buffer.append({
                    "word":       word,
                    "confidence": confidence,
                })

                # auto-trigger LLM if we have enough tokens and cooldown passed
                enough_tokens = len(llm_token_buffer) >= MIN_SIGNS_FOR_LLM
                cooldown_ok   = (time.time() - llm_last_call_ts) >= LLM_COOLDOWN_SEC

                if enough_tokens and cooldown_ok and not llm_is_running:
                    print(f"[AUTO] Triggering LLM with tokens: "
                          f"{[t['word'] for t in llm_token_buffer]}")
                    trigger_llm(llm_token_buffer)

    # ── clamp sentence history ────────────────────────────────────────────────
    if len(sentence) > 10:
        sentence = sentence[-10:]

    # ── HUD: top bar — sentence history ──────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)
    cv2.putText(frame, "History: " + " ".join(sentence),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # ── HUD: live NOW prediction ──────────────────────────────────────────────
    box_color = (0, 200, 0) if current_confidence >= CONFIDENCE_THRESH else (0, 140, 255)
    cv2.rectangle(frame, (0, 80), (frame.shape[1], 145), (30, 30, 30), -1)
    cv2.putText(frame, f"NOW: {current_prediction}",
                (10, 118), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, box_color, 3)
    cv2.putText(frame, f"conf: {current_confidence:.2f}",
                (frame.shape[1] - 160, 118), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # ── HUD: LLM panel ───────────────────────────────────────────────────────
    draw_llm_panel(frame, llm_result_text, llm_token_buffer)

    # ── HUD: buffer progress bar ──────────────────────────────────────────────
    fill_ratio = min(len(frame_buffer) / WINDOW_FRAMES, 1.0)
    bar_width  = int(frame.shape[1] * fill_ratio)
    bar_y_top  = frame.shape[0] - 14
    bar_y_bot  = frame.shape[0]
    cv2.rectangle(frame, (0, bar_y_top), (bar_width, bar_y_bot),
                  (0, 200, 255), -1)

    secs_until = max(0, (ADVANCE_FRAMES - frames_since_inference) / cam_fps)
    cv2.putText(frame,
                f"Buffer: {len(frame_buffer)}/{WINDOW_FRAMES}  |  "
                f"Next infer in {secs_until:.1f}s  |  "
                f"LLM tokens: {len(llm_token_buffer)}",
                (8, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (255, 255, 255), 1)

    cv2.imshow("PSL + LLM — Sign Recognition", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == 27:       # ESC → quit
        break

    if key & 0xFF == ord("c"):  # C → clear everything
        sentence               = []
        seen_words             = set()
        llm_token_buffer       = []
        llm_result_text        = ""
        current_prediction     = ""
        current_confidence     = 0.0
        frame_buffer.clear()
        frames_since_inference = 0
        print("Cleared sentence, token buffer, and LLM result.")

    if key & 0xFF == ord("l"):  # L → force LLM call now
        if llm_token_buffer:
            print(f"[MANUAL] Triggering LLM with tokens: "
                  f"{[t['word'] for t in llm_token_buffer]}")
            trigger_llm(llm_token_buffer)
        else:
            print("[MANUAL] No tokens in buffer yet — keep signing!")

# ── cleanup ────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
holistic.close()