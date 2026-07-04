# SignAvatar — ISL Sign Language to 3D Avatar

Convert Indian Sign Language (ISL) videos into animated 3D sign-language avatars.

## Architecture

```
┌─────────────────┐         POST /api/process-video         ┌───────────────────┐
│                 │  ────────────────────────────────────▶  │                   │
│   Frontend      │                                         │   Backend         │
│   (HTML/CSS/JS) │  ◀────────────────────────────────────  │   (FastAPI)       │
│   + Three.js    │         JSON landmark data              │   + MediaPipe     │
│   3D Avatar     │                                         │   + OpenCV        │
└─────────────────┘                                         └───────────────────┘
```

**Input** → Sign language video (MP4, WebM, AVI, MOV)  
**Processing** → MediaPipe Holistic extracts 33 body + 42 hand landmarks per frame  
**Output** → Animated 3D humanoid avatar mirroring the signer's movements  

## Quick Start

```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open browser
# http://localhost:8000
```

## Features

- 🎥 Drag-and-drop video upload
- 🤖 Real-time 3D avatar with body, hands, and head
- ✋ Detailed hand rendering (21 landmarks per hand with fingers)
- 🎮 Playback controls: play/pause, seek, speed (0.5×–2×), mirror
- 🌗 Dark-themed glassmorphism UI with bloom effects
- 📊 Processing stats: frames, FPS, duration, landmarks
- 🔄 Demo mode for testing without a video

## Tech Stack

| Layer    | Technology                        |
|----------|-----------------------------------|
| Backend  | Python, FastAPI, MediaPipe, OpenCV |
| Frontend | HTML5, CSS3, JavaScript (ES Modules) |
| 3D       | Three.js r160, UnrealBloomPass     |
| Fonts    | Inter, JetBrains Mono (Google Fonts) |

## Project Structure

```
web_application/
├── backend/
│   ├── app.py                  # FastAPI server + demo data generator
│   ├── requirements.txt        # Python dependencies
│   └── processing/
│       ├── __init__.py
│       └── pose_extractor.py   # MediaPipe Holistic landmark extraction
├── frontend/
│   ├── index.html              # SPA entry point
│   ├── css/
│   │   └── style.css           # Design system & dark theme
│   └── js/
│       ├── app.js              # Application controller
│       └── avatar.js           # Three.js 3D avatar renderer
└── README.md
```
