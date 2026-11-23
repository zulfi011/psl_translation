"""
server/app.py — Flask REST API wrapping the same pipeline as scripts/step7_llm.py.

POST /process   multipart form field "video"  → {"sentences": [...], "tokens": [...]}
GET  /health    → {"status": "ok", ...}

RUN : python server/app.py
"""
import os, json, tempfile, collections, textwrap
import numpy as np
import torch
import cv2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from psl_translator import config
from psl_translator.features import create_holistic, frame_to_features, fill_zero_frames
from psl_translator.model    import SignTCN

app = Flask(__name__)
CORS(app)

# ── config (identical to step7_llm.py) ────────────────────────────────────────
OPENROUTER_API_KEY  = config.OPENROUTER_API_KEY   # set in .env / environment
OPENROUTER_MODEL    = config.OPENROUTER_MODEL
LABELS_JSON         = config.LABELS_JSON
MODEL_PATH          = config.MODEL_PATH

CONFIDENCE_THRESH   = 0.50
SEQUENCE_LEN        = 60
SIGN_WINDOW_SEC     = 3.0
ADVANCE_SEC         = 1.0
MIN_SIGNS_FOR_LLM   = 1      # send to LLM even if only 1 sign detected

# ── load label map ─────────────────────────────────────────────────────────────
with open(LABELS_JSON) as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes  = len(label_map)

# ── load model once at startup ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcn_model = SignTCN(input_size=258, num_classes=num_classes)
tcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
tcn_model.to(device)
tcn_model.eval()
print(f"[STARTUP] Model loaded on {device}")
print(f"[STARTUP] {num_classes} classes: {list(label_map.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO PROCESSING  (same sliding-window logic as step7_llm.py)
# ══════════════════════════════════════════════════════════════════════════════

def process_video(video_path):
    cap     = cv2.VideoCapture(video_path)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps <= 0:
        cam_fps = 30.0

    window_frames  = int(cam_fps * SIGN_WINDOW_SEC)
    advance_frames = int(cam_fps * ADVANCE_SEC)

    print(f"[VIDEO] FPS={cam_fps:.1f}  window={window_frames}  advance={advance_frames}")

    holistic = create_holistic()

    frame_buffer   = collections.deque(maxlen=window_frames)
    frames_since   = 0
    seen_words     = set()
    token_buffer   = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        features = frame_to_features(results)
        frame_buffer.append(features)
        frames_since += 1

        # ── run inference every advance_frames, same as step7 ─────────────────
        if frames_since >= advance_frames and len(frame_buffer) >= window_frames:
            frames_since = 0

            buf     = list(frame_buffer)
            indices = np.linspace(0, len(buf) - 1, SEQUENCE_LEN, dtype=int)
            sequence = np.array([buf[i] for i in indices], dtype=np.float32)
            sequence = fill_zero_frames(sequence)

            x = torch.from_numpy(sequence).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = tcn_model(x)
                probs  = torch.softmax(logits, dim=1)
                conf   = probs.max().item()
                idx    = probs.argmax().item()
                word   = idx_to_label[idx]

            print(f"[INFER] frame={frame_count}  word={word:<14}  conf={conf:.2f}")

            if conf >= CONFIDENCE_THRESH and word not in seen_words:
                seen_words.add(word)
                token_buffer.append({"word": word, "confidence": conf})
                print(f"[TOKEN] Accepted: {word} ({conf:.0%})")

    cap.release()
    holistic.close()

    print(f"[VIDEO] Done. {frame_count} frames processed. Tokens: {token_buffer}")
    return token_buffer


# ══════════════════════════════════════════════════════════════════════════════
#  LLM CALL  (identical prompt engineering to step7_llm.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(sign_tokens):
    token_str = ", ".join(
        f'"{t["word"]}" (conf {t["confidence"]:.0%})'
        for t in sign_tokens
    )

    num_tokens = len(sign_tokens)

    if num_tokens == 1:
        word = sign_tokens[0]["word"]
        single_word_instruction = f"""
SPECIAL RULE — SINGLE TOKEN DETECTED
--------------------------------------
Only one sign was detected: "{word}"
You MUST follow this output format exactly:

Sentence 1: {word}
Sentence 2: <a short 2–4 word phrase using "{word}">
Sentence 3: <a full natural sentence using "{word}">
Reasoning: <one or two sentences explaining your choices>
"""
    elif num_tokens == 2:
        word1 = sign_tokens[0]["word"]
        word2 = sign_tokens[1]["word"]
        two_word_instruction = f"""
SPECIAL RULE — TWO TOKENS DETECTED
-------------------------------------
Exactly two signs were detected: "{word1}" and "{word2}"
You MUST follow this output format exactly:

Sentence 1: {word1}
Sentence 2: {word2}
Sentence 3: <a full natural English sentence combining both "{word1}" and "{word2}">
Reasoning: <one or two sentences explaining your choices>
"""
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
          each token. A low-confidence token may be misidentified — consider
          plausible alternatives if a sentence would not make sense otherwise.
        • Tokens are listed in temporal order, but a signer may repeat a sign
          for emphasis, so identical consecutive tokens usually count as one sign.

        YOUR TASK
        ---------
        Given the detected sign tokens, reconstruct the most likely English
        sentence(s) the signer intended to communicate.

        OUTPUT FORMAT (follow this EXACTLY — no extra text)
        -------------
        Sentence 1: <most likely English sentence>
        Sentence 2: <second most likely English sentence>
        Sentence 3: <third most likely English sentence>
        Reasoning: <one or two sentences explaining your choices>
    """)

    if num_tokens == 1:
        system_prompt = system_prompt + single_word_instruction
    elif num_tokens == 2:
        system_prompt = system_prompt + two_word_instruction

    user_prompt = textwrap.dedent(f"""\
        Detected sign tokens (in order):
        {token_str}

        Please provide three candidate English sentences the signer is most
        likely saying, ranked from most to least probable.
    """)

    return system_prompt, user_prompt


def call_llm(sign_tokens):
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY is not set — see .env.example")
    system_prompt, user_prompt = build_llm_prompt(sign_tokens)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/psl-sign-recognition",
        "X-Title":       "PSL Sign Recognition",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens":  300,
    }

    print(f"[LLM] Calling {OPENROUTER_MODEL} with tokens: "
          f"{[t['word'] for t in sign_tokens]}")

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[LLM] Response:\n{result}")
        return result
    except requests.exceptions.Timeout:
        raise Exception("LLM request timed out")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"LLM HTTP error: {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        raise Exception(f"LLM error: {e}")


def parse_sentences(llm_text):
    """Extract Sentence 1/2/3 lines from LLM output."""
    lines     = llm_text.split("\n")
    sentences = []
    for line in lines:
        for prefix in ["Sentence 1:", "Sentence 2:", "Sentence 3:"]:
            if line.strip().startswith(prefix):
                sentences.append(line.strip()[len(prefix):].strip())
    # fallback: return raw text as single sentence if parsing fails
    return sentences[:3] if sentences else [llm_text]


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/process", methods=["POST"])
def process():
    print("\n" + "="*60)
    print("[REQUEST] POST /process received")

    if "video" not in request.files:
        print("[ERROR] No video file in request")
        return jsonify({"error": "No video file attached"}), 400

    video_file = request.files["video"]
    suffix     = os.path.splitext(video_file.filename)[1] or ".mp4"

    # save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    file_size = os.path.getsize(tmp_path)
    print(f"[REQUEST] Video saved: {tmp_path}  size={file_size} bytes")

    try:
        # ── step 1: extract sign tokens ───────────────────────────────────────
        tokens = process_video(tmp_path)

        if not tokens:
            print("[RESULT] No signs detected above threshold")
            return jsonify({
                "sentences": ["No signs detected. Please try again."],
                "tokens": []
            })

        # ── step 2: call LLM ──────────────────────────────────────────────────
        llm_output = call_llm(tokens)
        sentences  = parse_sentences(llm_output)

        print(f"[RESULT] Returning {len(sentences)} sentences")
        return jsonify({
            "sentences": sentences,
            "tokens":    [t["word"] for t in tokens],
            "raw_llm":   llm_output,
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": num_classes, "device": str(device)})


if __name__ == "__main__":
    host = os.getenv("PSL_HOST", "0.0.0.0")
    port = int(os.getenv("PSL_PORT", "5000"))
    app.run(host=host, port=port, debug=False)