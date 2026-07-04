"""
SignAvatar Backend — FastAPI server.

Endpoints:
    POST /api/process-video   Upload & process a sign-language video (returns landmark JSON).
    GET  /api/demo-data       Returns synthetic landmark data for frontend testing.
    GET  /                    Serves the frontend SPA.
"""

import math
import os
import shutil
import tempfile
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from processing.pose_extractor import PoseExtractor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("signavatar")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="SignAvatar – ISL to 3D Avatar", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = PoseExtractor(model_complexity=1)

UPLOAD_DIR = tempfile.mkdtemp(prefix="signavatar_")
logger.info(f"Upload temp dir: {UPLOAD_DIR}")

ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/avi",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "application/octet-stream",  # fallback when MIME sniffing fails
}

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/process-video")
async def process_video(video: UploadFile = File(...)):
    """Accept a sign-language video, extract landmarks, return JSON."""
    logger.info(f"Received video: {video.filename}  type={video.content_type}")

    # Save to temp file
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    file_path = os.path.join(UPLOAD_DIR, f"upload_{id(video)}{suffix}")

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        logger.info(f"Saved to {file_path}, processing…")
        result = extractor.process_video(file_path, sample_rate=1)
        logger.info(
            f"Done — {result['total_frames']} frames, "
            f"{result['fps']:.1f} fps"
        )
        return JSONResponse(content=result)

    except ValueError as exc:
        logger.error(f"ValueError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Processing error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------------------------
# Demo data generator  — rich ISL-style multi-sign animation
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: tuple, b: tuple, t: float) -> tuple:
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t))


def _ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


# ── Base resting pose  (MediaPipe normalised 0-1 coords, 16:9 video) ──────────
# Landmarks are carefully proportioned to match real MediaPipe output
_BASE_POSE = [
    (0.500, 0.130, -0.012),  # 0  nose
    (0.487, 0.113, -0.025),  # 1  left eye inner
    (0.477, 0.110, -0.030),  # 2  left eye
    (0.466, 0.113, -0.022),  # 3  left eye outer
    (0.513, 0.113, -0.025),  # 4  right eye inner
    (0.523, 0.110, -0.030),  # 5  right eye
    (0.534, 0.113, -0.022),  # 6  right eye outer
    (0.455, 0.125, 0.008),   # 7  left ear
    (0.545, 0.125, 0.008),   # 8  right ear
    (0.492, 0.152, -0.010),  # 9  mouth left
    (0.508, 0.152, -0.010),  # 10 mouth right
    (0.390, 0.270, 0.000),   # 11 left shoulder
    (0.610, 0.270, 0.000),   # 12 right shoulder
    (0.345, 0.415, 0.008),   # 13 left elbow
    (0.655, 0.415, 0.008),   # 14 right elbow
    (0.350, 0.548, 0.004),   # 15 left wrist
    (0.650, 0.548, 0.004),   # 16 right wrist
    (0.344, 0.564, 0.009),   # 17 left pinky
    (0.656, 0.564, 0.009),   # 18 right pinky
    (0.356, 0.564, -0.004),  # 19 left index
    (0.644, 0.564, -0.004),  # 20 right index
    (0.362, 0.556, -0.014),  # 21 left thumb
    (0.638, 0.556, -0.014),  # 22 right thumb
    (0.450, 0.570, 0.000),   # 23 left hip
    (0.550, 0.570, 0.000),   # 24 right hip
    (0.450, 0.725, 0.004),   # 25 left knee
    (0.550, 0.725, 0.004),   # 26 right knee
    (0.450, 0.878, 0.000),   # 27 left ankle
    (0.550, 0.878, 0.000),   # 28 right ankle
    (0.445, 0.895, 0.014),   # 29 left heel
    (0.555, 0.895, 0.014),   # 30 right heel
    (0.456, 0.900, -0.014),  # 31 left foot index
    (0.544, 0.900, -0.014),  # 32 right foot index
]

# ── Hand shapes (21 landmarks relative to wrist, in image-normalised units) ──
# All coords are in the same [0,1] normalised space as pose landmarks

# Open palm — fingers spread, facing forward
_OPEN_PALM = [
    ( 0.000,  0.000,  0.000),  # 0  wrist
    ( 0.018, -0.008, -0.018),  # 1  thumb CMC
    ( 0.032, -0.020, -0.026),  # 2  thumb MCP
    ( 0.042, -0.034, -0.020),  # 3  thumb IP
    ( 0.048, -0.048, -0.016),  # 4  thumb TIP
    ( 0.014, -0.038, -0.007),  # 5  index MCP
    ( 0.015, -0.060, -0.007),  # 6  index PIP
    ( 0.015, -0.076, -0.007),  # 7  index DIP
    ( 0.015, -0.090, -0.007),  # 8  index TIP
    ( 0.000, -0.040,  0.000),  # 9  middle MCP
    ( 0.000, -0.063,  0.000),  # 10 middle PIP
    ( 0.000, -0.078,  0.000),  # 11 middle DIP
    ( 0.000, -0.092,  0.000),  # 12 middle TIP
    (-0.013, -0.038,  0.007),  # 13 ring MCP
    (-0.013, -0.058,  0.007),  # 14 ring PIP
    (-0.013, -0.072,  0.007),  # 15 ring DIP
    (-0.013, -0.084,  0.007),  # 16 ring TIP
    (-0.024, -0.032,  0.015),  # 17 pinky MCP
    (-0.024, -0.047,  0.015),  # 18 pinky PIP
    (-0.024, -0.059,  0.015),  # 19 pinky DIP
    (-0.024, -0.069,  0.015),  # 20 pinky TIP
]

# Index finger pointing up — other fingers curled
_POINT_UP = [
    ( 0.000,  0.000,  0.000),  # 0  wrist
    ( 0.018, -0.008, -0.018),  # 1  thumb CMC
    ( 0.028, -0.018, -0.015),  # 2  thumb MCP
    ( 0.030, -0.026, -0.012),  # 3  thumb IP
    ( 0.030, -0.032, -0.010),  # 4  thumb TIP (tucked)
    ( 0.014, -0.038, -0.007),  # 5  index MCP
    ( 0.014, -0.060, -0.007),  # 6  index PIP  (extended)
    ( 0.013, -0.078, -0.006),  # 7  index DIP
    ( 0.012, -0.094, -0.005),  # 8  index TIP
    ( 0.000, -0.040,  0.000),  # 9  middle MCP
    (-0.002, -0.044,  0.012),  # 10 middle PIP (curled)
    (-0.004, -0.042,  0.018),  # 11 middle DIP
    (-0.004, -0.038,  0.020),  # 12 middle TIP
    (-0.013, -0.038,  0.007),  # 13 ring MCP
    (-0.015, -0.042,  0.016),  # 14 ring PIP   (curled)
    (-0.016, -0.040,  0.022),  # 15 ring DIP
    (-0.016, -0.036,  0.024),  # 16 ring TIP
    (-0.024, -0.032,  0.015),  # 17 pinky MCP
    (-0.025, -0.036,  0.022),  # 18 pinky PIP  (curled)
    (-0.025, -0.034,  0.026),  # 19 pinky DIP
    (-0.025, -0.030,  0.028),  # 20 pinky TIP
]

# Closed fist
_FIST = [
    ( 0.000,  0.000,  0.000),  # 0  wrist
    ( 0.016, -0.008, -0.015),  # 1  thumb CMC
    ( 0.022, -0.015, -0.012),  # 2  thumb MCP
    ( 0.022, -0.020, -0.010),  # 3  thumb IP
    ( 0.020, -0.024, -0.010),  # 4  thumb TIP
    ( 0.013, -0.036, -0.006),  # 5  index MCP
    ( 0.006, -0.040,  0.010),  # 6  index PIP  (curled)
    ( 0.002, -0.038,  0.016),  # 7  index DIP
    ( 0.001, -0.034,  0.018),  # 8  index TIP
    ( 0.000, -0.038,  0.000),  # 9  middle MCP
    (-0.002, -0.042,  0.012),  # 10 middle PIP (curled)
    (-0.004, -0.040,  0.018),  # 11 middle DIP
    (-0.004, -0.036,  0.020),  # 12 middle TIP
    (-0.012, -0.036,  0.007),  # 13 ring MCP
    (-0.014, -0.040,  0.016),  # 14 ring PIP   (curled)
    (-0.015, -0.038,  0.022),  # 15 ring DIP
    (-0.015, -0.034,  0.024),  # 16 ring TIP
    (-0.022, -0.030,  0.014),  # 17 pinky MCP
    (-0.023, -0.034,  0.020),  # 18 pinky PIP  (curled)
    (-0.023, -0.032,  0.025),  # 19 pinky DIP
    (-0.022, -0.028,  0.027),  # 20 pinky TIP
]

# "V" / victory / peace sign — index + middle extended
_VICTORY = [
    ( 0.000,  0.000,  0.000),  # 0  wrist
    ( 0.018, -0.008, -0.018),  # 1  thumb CMC
    ( 0.028, -0.018, -0.015),  # 2  thumb MCP
    ( 0.030, -0.024, -0.012),  # 3  thumb IP
    ( 0.030, -0.030, -0.010),  # 4  thumb TIP
    ( 0.014, -0.038, -0.007),  # 5  index MCP
    ( 0.016, -0.060, -0.008),  # 6  index PIP  (extended)
    ( 0.016, -0.078, -0.007),  # 7  index DIP
    ( 0.015, -0.094, -0.006),  # 8  index TIP
    ( 0.004, -0.040, -0.002),  # 9  middle MCP
    ( 0.005, -0.062, -0.003),  # 10 middle PIP (extended, slightly spread)
    ( 0.005, -0.080, -0.002),  # 11 middle DIP
    ( 0.004, -0.096, -0.001),  # 12 middle TIP
    (-0.013, -0.038,  0.007),  # 13 ring MCP
    (-0.015, -0.042,  0.016),  # 14 ring PIP   (curled)
    (-0.016, -0.040,  0.022),  # 15 ring DIP
    (-0.016, -0.036,  0.024),  # 16 ring TIP
    (-0.024, -0.032,  0.015),  # 17 pinky MCP
    (-0.025, -0.036,  0.022),  # 18 pinky PIP  (curled)
    (-0.025, -0.034,  0.026),  # 19 pinky DIP
    (-0.025, -0.030,  0.028),  # 20 pinky TIP
]


def _interp_hand(shape_a: list, shape_b: list, t: float) -> list:
    """Linearly interpolate between two hand shapes."""
    return [
        (
            _lerp(shape_a[i][0], shape_b[i][0], t),
            _lerp(shape_a[i][1], shape_b[i][1], t),
            _lerp(shape_a[i][2], shape_b[i][2], t),
        )
        for i in range(21)
    ]


def _hand_from_shape(wrist: tuple, shape: list, flip_x: bool = False) -> list:
    """Build 21 hand landmarks from a shape + wrist anchor."""
    sx = -1.0 if flip_x else 1.0
    return [
        {
            "x": round(wrist[0] + shape[i][0] * sx, 6),
            "y": round(wrist[1] + shape[i][1],       6),
            "z": round(wrist[2] + shape[i][2],        6),
        }
        for i in range(21)
    ]


def generate_demo_data() -> dict:
    """
    240-frame (8 s @ 30 fps) ISL-inspired animation with 4 signs:

    Sign 1  (0-60):    HELLO  — right open palm raised & waved side-to-side
    Sign 2  (60-120):  POINT  — right index finger pointing up & tracing circle
    Sign 3  (120-180): VICTORY — both hands raised, V-sign
    Sign 4  (180-240): REST   — hands return to neutral, hold
    """
    FPS = 30
    N   = 240

    # ── Arm keyframes (indices into _BASE_POSE) ──────────────────────
    # Right arm positions
    R_REST   = (0.650, 0.548, 0.004)   # right wrist rest (same as base)
    R_ELBOW_REST = (0.655, 0.415, 0.008)
    R_HIGH   = (0.530, 0.090, -0.045)  # wrist raised to head-height
    R_ELBOW_HIGH = (0.580, 0.210, -0.030)
    R_MID    = (0.530, 0.220, -0.020)  # mid-height (pointing)
    R_ELBOW_MID  = (0.595, 0.320, 0.005)

    # Left arm positions
    L_REST   = (0.350, 0.548, 0.004)
    L_ELBOW_REST = (0.345, 0.415, 0.008)
    L_HIGH   = (0.470, 0.090, -0.045)
    L_ELBOW_HIGH = (0.420, 0.210, -0.030)

    frames = []

    for i in range(N):
        t = i / N
        fi = i  # frame index

        pose = [list(p) for p in _BASE_POSE]
        right_shape = _OPEN_PALM
        left_shape  = _FIST

        # ═══ Sign 1: HELLO wave  (frames 0–60) ════════════════════════
        if fi < 60:
            s = fi / 60.0

            if s < 0.25:
                # Raise right arm
                p = _ease_in_out(s / 0.25)
                pose[14] = list(_lerp3(R_ELBOW_REST, R_ELBOW_HIGH, p))
                pose[16] = list(_lerp3(R_REST, R_HIGH, p))
                right_shape = _interp_hand(_FIST, _OPEN_PALM, p)
            else:
                # Wave: side-to-side oscillation
                w_t  = (s - 0.25) / 0.75
                wave = math.sin(w_t * math.pi * 5) * 0.060
                pose[14] = list(R_ELBOW_HIGH)
                pose[14][0] += wave * 0.25
                pose[16] = list(R_HIGH)
                pose[16][0] += wave
                pose[16][1] += abs(wave) * 0.12
                right_shape = _OPEN_PALM

        # ═══ Sign 2: POINT finger circle  (frames 60–120) ════════════
        elif fi < 120:
            s = (fi - 60) / 60.0

            if s < 0.15:
                # Transition: high → mid
                p = _ease_in_out(s / 0.15)
                pose[14] = list(_lerp3(R_ELBOW_HIGH, R_ELBOW_MID, p))
                pose[16] = list(_lerp3(R_HIGH, R_MID, p))
                right_shape = _interp_hand(_OPEN_PALM, _POINT_UP, p)
            else:
                # Trace a small clockwise circle with the pointing finger
                angle = (s - 0.15) / 0.85 * math.pi * 3  # 1.5 rotations
                cx = 0.530 + math.cos(angle) * 0.040
                cy = 0.220 + math.sin(angle) * 0.028
                pose[14] = list(R_ELBOW_MID)
                pose[14][0] += math.cos(angle) * 0.015
                pose[16] = [cx, cy, -0.020]
                right_shape = _POINT_UP

        # ═══ Sign 3: VICTORY — both hands  (frames 120–180) ══════════
        elif fi < 180:
            s = (fi - 120) / 60.0

            if s < 0.2:
                # Both arms rise together
                p = _ease_in_out(s / 0.2)
                pose[14] = list(_lerp3(R_ELBOW_MID, R_ELBOW_HIGH, p))
                pose[16] = list(_lerp3(R_MID, R_HIGH, p))
                pose[13] = list(_lerp3(L_ELBOW_REST, L_ELBOW_HIGH, p))
                pose[15] = list(_lerp3(L_REST, L_HIGH, p))
                right_shape = _interp_hand(_POINT_UP, _VICTORY, p)
                left_shape  = _interp_hand(_FIST, _VICTORY, p)
            else:
                # Hold V-sign — subtle pulsing
                pulse = math.sin((s - 0.2) / 0.8 * math.pi * 4) * 0.010
                pose[14] = list(R_ELBOW_HIGH); pose[14][1] += pulse
                pose[16] = list(R_HIGH);       pose[16][1] += pulse
                pose[13] = list(L_ELBOW_HIGH); pose[13][1] += pulse
                pose[15] = list(L_HIGH);       pose[15][1] += pulse
                right_shape = _VICTORY
                left_shape  = _VICTORY

        # ═══ Sign 4: REST — return to neutral  (frames 180–240) ══════
        else:
            s = (fi - 180) / 60.0
            p = _ease_out(min(s / 0.6, 1.0))

            pose[14] = list(_lerp3(R_ELBOW_HIGH, R_ELBOW_REST, p))
            pose[16] = list(_lerp3(R_HIGH, R_REST, p))
            pose[13] = list(_lerp3(L_ELBOW_HIGH, L_ELBOW_REST, p))
            pose[15] = list(_lerp3(L_HIGH, L_REST, p))
            right_shape = _interp_hand(_VICTORY, _FIST, min(p * 1.4, 1.0))
            left_shape  = _interp_hand(_VICTORY, _FIST, min(p * 1.4, 1.0))

        # Keep secondary markers (pinky/index/thumb tips) near wrist
        wrist_r = pose[16]
        wrist_l = pose[15]
        for idx, base_w_idx in [(18, 16), (20, 16), (22, 16)]:
            pose[idx] = [
                wrist_r[0] + (_BASE_POSE[idx][0] - _BASE_POSE[base_w_idx][0]),
                wrist_r[1] + (_BASE_POSE[idx][1] - _BASE_POSE[base_w_idx][1]),
                wrist_r[2] + (_BASE_POSE[idx][2] - _BASE_POSE[base_w_idx][2]),
            ]
        for idx, base_w_idx in [(17, 15), (19, 15), (21, 15)]:
            pose[idx] = [
                wrist_l[0] + (_BASE_POSE[idx][0] - _BASE_POSE[base_w_idx][0]),
                wrist_l[1] + (_BASE_POSE[idx][1] - _BASE_POSE[base_w_idx][1]),
                wrist_l[2] + (_BASE_POSE[idx][2] - _BASE_POSE[base_w_idx][2]),
            ]

        # Gentle breathing / body sway
        breathe = math.sin(t * math.pi * 4) * 0.003
        sway    = math.sin(t * math.pi * 2.5) * 0.003
        for idx in range(len(pose)):
            pose[idx][0] += sway
            pose[idx][1] += breathe if idx <= 12 else 0.0

        pose_lm = [
            {"x": round(p[0], 6), "y": round(p[1], 6),
             "z": round(p[2], 6), "v": 0.99}
            for p in pose
        ]

        frame = {
            "pose": pose_lm,
            "left_hand":  _hand_from_shape(tuple(wrist_l), left_shape,  flip_x=True),
            "right_hand": _hand_from_shape(tuple(wrist_r), right_shape, flip_x=False),
        }
        frames.append(frame)

    return {
        "fps": FPS,
        "original_fps": FPS,
        "width": 1280,
        "height": 720,
        "total_frames": N,
        "original_total_frames": N,
        "frames": frames,
        "is_demo": True,
    }


@app.get("/api/demo-data")
async def demo_data():
    """Return synthetic landmark animation for frontend testing (no video needed)."""
    return JSONResponse(content=generate_demo_data())


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"Serving frontend from {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_DIR}")


# ---------------------------------------------------------------------------
# Entry-point (python app.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
