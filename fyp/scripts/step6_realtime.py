"""
step6_realtime.py
==================
PURPOSE : Real-time PSL sign recognition from webcam using a 5-second
          sliding window that advances by 1 second at a time.
RUN     : python scripts/step6_realtime.py

Sliding window logic
--------------------
  • A rolling buffer always holds the last (FPS × 5) frames.
  • Every (FPS × 1) frames a new frame-chunk is added and the oldest
    1-second chunk is dropped.
  • The full 5-second buffer is sub-sampled to 60 frames and sent to
    the model — so inference runs every 1 second, not every 5 seconds.
  • This gives smooth, responsive predictions.

Keys
----
  ESC  – quit
  C    – clear the recognised sentence

Visualisation
-------------
  • Left and right hand skeletons drawn in colour.
  • Full body pose skeleton drawn.
  • Face NOT drawn (intentionally excluded).
  • Bottom progress bar shows how many frames are in the current buffer.
  • Top black bar shows the growing recognised sentence.
"""

import cv2
import json
import numpy as np
import torch
import mediapipe as mp
import collections

from psl_translator.config   import LABELS_JSON, MODEL_PATH, video_source
from psl_translator.features import create_holistic, frame_to_features, fill_zero_frames
from psl_translator.model    import SignTCN

# ── label map ──────────────────────────────────────────────────────────────────
with open(LABELS_JSON) as f:
    label_map = json.load(f)       # {"eat": 0, "drink": 1, ...}

# invert: index → label name
idx_to_label = {v: k for k, v in label_map.items()}
num_classes  = len(label_map)

# ── model ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SignTCN(input_size=258, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()   # disable dropout, use frozen BN

# ── MediaPipe Holistic ─────────────────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_draw        = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

holistic = create_holistic()

# ── sliding window constants ───────────────────────────────────────────────────
SEQUENCE_LEN       = 60     # frames fed to the model after sub-sampling
SIGN_WINDOW_SEC    = 3.0    # rolling window length in seconds
ADVANCE_SEC        = 1.0    # slide window forward by this many seconds
CONFIDENCE_THRESH  = 0.50   # minimum softmax confidence to show prediction
                            # (lowered so continuous output is visible)

# ── camera ─────────────────────────────────────────────────────────────────────
# webcam by default; set PSL_VIDEO_SOURCE=path/to/video.mp4 to use a recorded clip
cap     = cv2.VideoCapture(video_source(0))

cam_fps = cap.get(cv2.CAP_PROP_FPS)
if cam_fps <= 0:
    cam_fps = 30.0   # fallback if driver reports 0

# derived frame counts
WINDOW_FRAMES  = int(cam_fps * SIGN_WINDOW_SEC)   # total frames in rolling buf
ADVANCE_FRAMES = int(cam_fps * ADVANCE_SEC)        # frames before next inference

print(f"Camera FPS     : {cam_fps:.1f}")
print(f"Window frames  : {WINDOW_FRAMES}  ({SIGN_WINDOW_SEC}s)")
print(f"Advance frames : {ADVANCE_FRAMES} ({ADVANCE_SEC}s) — inference every 1s")

# ── runtime state ──────────────────────────────────────────────────────────────
# deque with a fixed max length acts as an efficient rolling buffer:
# appending to a full deque automatically drops the oldest element.
frame_buffer     = collections.deque(maxlen=WINDOW_FRAMES)

frames_since_inference = 0        # counts frames since last inference call
sentence               = []       # growing list of accepted words (history bar)
seen_words             = set()    # tracks words already in sentence — prevents duplicates
current_prediction     = ""       # latest prediction shown live on camera
current_confidence     = 0.0      # confidence of latest prediction


# ── helper functions ───────────────────────────────────────────────────────────

def draw_landmarks(frame, results):
    """Draw hand and pose skeletons. Face is intentionally NOT drawn."""

    # left hand
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style()
        )

    # right hand
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style()
        )

    # full body pose (arms, torso, legs — no face mesh)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw_styles.get_default_pose_landmarks_style()
        )


def run_inference():
    """
    Sub-sample the rolling buffer to SEQUENCE_LEN frames and run the model.
    Returns (predicted_label, confidence) or (None, 0) if buffer not full.
    """
    if len(frame_buffer) < WINDOW_FRAMES:
        return None, 0.0   # not enough frames yet

    buf = list(frame_buffer)   # snapshot of deque as list

    # evenly-spaced sub-sample: picks SEQUENCE_LEN frames from the full window
    indices  = np.linspace(0, len(buf) - 1, SEQUENCE_LEN, dtype=int)
    sequence = np.array([buf[i] for i in indices], dtype=np.float32)
    # shape → (60, 258)
    sequence = fill_zero_frames(sequence) 

    x = torch.tensor([sequence], dtype=torch.float32).to(device)
    # shape → (1, 60, 258)

    with torch.no_grad():
        logits     = model(x)                       # (1, num_classes)
        probs      = torch.softmax(logits, dim=1)   # probabilities
        confidence = probs.max().item()
        pred_idx   = probs.argmax().item()

    return idx_to_label[pred_idx], confidence


# ── main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # run unified Holistic detector
    results = holistic.process(rgb)

    # build 258-value feature vector for this frame
    features = frame_to_features(results)

    # add to rolling buffer (deque automatically drops oldest when full)
    frame_buffer.append(features)
    frames_since_inference += 1

    # draw hand + pose skeletons on the live frame
    draw_landmarks(frame, results)

    # ── run inference every ADVANCE_FRAMES (= 1 second) ──────────────────────
    if frames_since_inference >= ADVANCE_FRAMES:
        frames_since_inference = 0

        word, confidence = run_inference()

        if word is not None:
            # ── always update the live prediction display ──────────────────
            # This shows continuously what the model is predicting RIGHT NOW,
            # regardless of confidence — so you can see the model "thinking".
            current_prediction = word
            current_confidence = confidence

            # print every prediction to terminal continuously
            print(f"Prediction: {word:<12}  conf: {confidence:.2f}")

            # ── only add to sentence history if confident + not a duplicate ──
            # Uses a set to track which words have already been added.
            # "eat eat" becomes just "eat" — same word won't appear twice.
            if confidence >= CONFIDENCE_THRESH and word not in seen_words:
                sentence.append(word)
                seen_words.add(word)

    # ── HUD: keep sentence history to last 10 words ───────────────────────────
    if len(sentence) > 10:
        sentence = sentence[-10:]

    # ── HUD: top bar — sentence history (accepted predictions) ───────────────
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "History: " + " ".join(sentence),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    # ── HUD: live prediction box — always shows current model output ──────────
    # colour: green if confident, orange if below threshold
    box_color  = (0, 200, 0) if current_confidence >= CONFIDENCE_THRESH else (0, 140, 255)
    text_color = (255, 255, 255)
    cv2.rectangle(frame, (0, 80), (frame.shape[1], 145), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"NOW: {current_prediction}",
        (10, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, box_color, 3
    )
    cv2.putText(
        frame,
        f"conf: {current_confidence:.2f}",
        (frame.shape[1] - 160, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, text_color, 2
    )

    # ── HUD: buffer fill bar (shows how full the 5-second window is) ─────────
    fill_ratio = min(len(frame_buffer) / WINDOW_FRAMES, 1.0)
    bar_width  = int(frame.shape[1] * fill_ratio)
    bar_y_top  = frame.shape[0] - 14
    bar_y_bot  = frame.shape[0]
    cv2.rectangle(frame, (0, bar_y_top), (bar_width, bar_y_bot), (0, 200, 255), -1)

    # label showing time-until-next-inference
    secs_until = max(0, (ADVANCE_FRAMES - frames_since_inference) / cam_fps)
    cv2.putText(
        frame,
        f"Buffer: {len(frame_buffer)}/{WINDOW_FRAMES}  |  "
        f"Next infer in {secs_until:.1f}s",
        (8, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1
    )

    cv2.imshow("PSL Real-Time — Hands + Pose (no face)", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:           # ESC → quit
        break
    if key & 0xFF == ord("c"):     # C   → clear sentence and buffer
        sentence               = []
        seen_words             = set()
        current_prediction     = ""
        current_confidence     = 0.0
        frame_buffer.clear()
        frames_since_inference = 0
        print("Sentence cleared.")

# ── cleanup ────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()