/**
 * SignAvatar – Main Application Controller
 *
 * Orchestrates:  Upload → API call → Avatar rendering → Playback controls
 */

import { SignAvatar, setVideoAspect } from "./avatar.js";

// ── DOM References ───────────────────────────────────────────────
const $uploadSection   = document.getElementById("section-upload");
const $processSection  = document.getElementById("section-processing");
const $resultSection   = document.getElementById("section-result");

const $uploadZone      = document.getElementById("upload-zone");
const $fileInput       = document.getElementById("file-input");
const $btnDemo         = document.getElementById("btn-demo");

const $processingTitle = document.getElementById("processing-title");
const $processingSub   = document.getElementById("processing-subtitle");
const $progressFill    = document.getElementById("progress-fill");
const $processingDetail= document.getElementById("processing-detail");

const $video           = document.getElementById("original-video");
const $videoPlaceholder= document.getElementById("video-placeholder");
const $avatarCanvas    = document.getElementById("avatar-canvas");

const $btnPlay         = document.getElementById("btn-play");
const $iconPlay        = document.getElementById("icon-play");
const $iconPause       = document.getElementById("icon-pause");
const $btnRestart      = document.getElementById("btn-restart");
const $timelineSlider  = document.getElementById("timeline-slider");
const $timeCurrent     = document.getElementById("time-current");
const $timeTotal       = document.getElementById("time-total");
const $btnMirror       = document.getElementById("btn-mirror");
const $btnNew          = document.getElementById("btn-new");

const $statFrames      = document.getElementById("stat-frames");
const $statFps         = document.getElementById("stat-fps");
const $statDuration    = document.getElementById("stat-duration");
const $statLandmarks   = document.getElementById("stat-landmarks");

const $bgCanvas        = document.getElementById("bg-canvas");

// ── State ────────────────────────────────────────────────────────
let avatar = null;
let videoFile = null;
let isPlaying = false;
let isMirrored = false;
let seekLock = false;

// ── Helpers ──────────────────────────────────────────────────────

function showSection(section) {
    [$uploadSection, $processSection, $resultSection].forEach(s => s.classList.remove("active"));
    section.classList.add("active");
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
}

function toast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
        el.classList.add("removing");
        setTimeout(() => el.remove(), 300);
    }, 4000);
}

// ── Background Particles ─────────────────────────────────────────

function initBgParticles() {
    const ctx = $bgCanvas.getContext("2d");
    let w, h;
    const particles = [];
    const COUNT = 70;

    function resize() {
        w = $bgCanvas.width  = window.innerWidth;
        h = $bgCanvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    for (let i = 0; i < COUNT; i++) {
        particles.push({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.random() * 1.5 + 0.5,
            vx: (Math.random() - 0.5) * 0.25,
            vy: (Math.random() - 0.5) * 0.25,
            alpha: Math.random() * 0.25 + 0.05,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        for (const p of particles) {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0,229,255,${p.alpha})`;
            ctx.fill();
        }
        requestAnimationFrame(draw);
    }
    draw();
}

// ── Upload Handling ──────────────────────────────────────────────

$uploadZone.addEventListener("click", () => $fileInput.click());
$uploadZone.addEventListener("dragover", e => { e.preventDefault(); $uploadZone.classList.add("drag-over"); });
$uploadZone.addEventListener("dragleave", () => $uploadZone.classList.remove("drag-over"));
$uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    $uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length && files[0].type.startsWith("video/")) {
        handleVideoFile(files[0]);
    } else {
        toast("Please drop a video file", "error");
    }
});
$fileInput.addEventListener("change", () => {
    if ($fileInput.files.length) handleVideoFile($fileInput.files[0]);
});

async function handleVideoFile(file) {
    videoFile = file;
    showSection($processSection);
    $processingTitle.textContent = "Processing Video…";
    $processingSub.textContent = "Extracting pose & hand landmarks with MediaPipe";
    $processingDetail.textContent = `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)…`;
    $progressFill.style.width = "15%";

    try {
        const formData = new FormData();
        formData.append("video", file);

        $progressFill.style.width = "40%";
        $processingDetail.textContent = "Sending to server for landmark extraction…";

        const res = await fetch("/api/process-video", { method: "POST", body: formData });

        $progressFill.style.width = "80%";
        $processingDetail.textContent = "Parsing landmark data…";

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || "Server error");
        }

        const data = await res.json();
        $progressFill.style.width = "100%";
        $processingDetail.textContent = `Done — ${data.total_frames} frames extracted`;

        // Brief pause so user sees 100%
        await new Promise(r => setTimeout(r, 600));

        showResult(data, file);
    } catch (err) {
        console.error(err);
        toast(`Error: ${err.message}`, "error");
        showSection($uploadSection);
    }
}

// ── Demo Mode ────────────────────────────────────────────────────

$btnDemo.addEventListener("click", async () => {
    showSection($processSection);
    $processingTitle.textContent = "Loading Demo…";
    $processingSub.textContent = "Generating synthetic sign-language animation";
    $processingDetail.textContent = "Fetching demo landmark data…";
    $progressFill.style.width = "50%";

    try {
        const res = await fetch("/api/demo-data");
        if (!res.ok) throw new Error("Failed to load demo data");
        const data = await res.json();

        $progressFill.style.width = "100%";
        $processingDetail.textContent = "Ready!";
        await new Promise(r => setTimeout(r, 400));

        showResult(data, null);
    } catch (err) {
        console.error(err);
        toast(`Demo error: ${err.message}`, "error");
        showSection($uploadSection);
    }
});

// ── Result Display ───────────────────────────────────────────────

function showResult(data, file) {
    showSection($resultSection);

    // Video panel
    if (file) {
        const url = URL.createObjectURL(file);
        $video.src = url;
        $video.classList.add("visible");
        $videoPlaceholder.classList.remove("visible");
        $video.loop = true;
        $video.load(); // force browser to start buffering

        $video.addEventListener("loadeddata", function onLoaded() {
            $video.removeEventListener("loadeddata", onLoaded);
            // Seek to first frame so it renders (not black)
            $video.currentTime = 0.001;
            // Autoplay (works because video is muted)
            $video.play().catch(() => { /* user can click play manually */ });
        });
    } else {
        $video.classList.remove("visible");
        $videoPlaceholder.classList.add("visible");
    }

    // Set aspect ratio for correct coordinate mapping
    if (data.width && data.height) setVideoAspect(data.width, data.height);

    // Create or reset avatar
    if (avatar) avatar.dispose();
    avatar = new SignAvatar($avatarCanvas);
    avatar.loadData(data);
    avatar.play();
    isPlaying = true;
    updatePlayIcon();

    // Stats
    const duration = data.total_frames / (data.fps || 30);
    $statFrames.textContent = data.total_frames;
    $statFps.textContent = Math.round(data.fps || 30);
    $statDuration.textContent = formatTime(duration);

    // Count average landmarks per frame
    let avgLM = 0;
    let counted = 0;
    for (const f of data.frames) {
        let c = 0;
        if (f.pose) c += f.pose.length;
        if (f.left_hand) c += f.left_hand.length;
        if (f.right_hand) c += f.right_hand.length;
        avgLM += c;
        counted++;
    }
    $statLandmarks.textContent = counted > 0 ? Math.round(avgLM / counted) : "—";

    // Timeline
    $timelineSlider.max = data.total_frames - 1;
    $timeTotal.textContent = formatTime(duration);

    // Frame-change listener
    $avatarCanvas.addEventListener("framechange", onFrameChange);

    toast("Avatar loaded — animation playing", "success");
}

function onFrameChange(e) {
    if (seekLock) return;
    const { frame } = e.detail;
    $timelineSlider.value = frame;
    const elapsed = frame / (avatar.fps || 30);
    $timeCurrent.textContent = formatTime(elapsed);
    // NOTE: video plays independently via its own native controls
}

// ── Playback Controls ────────────────────────────────────────────

function updatePlayIcon() {
    $iconPlay.style.display  = isPlaying ? "none" : "block";
    $iconPause.style.display = isPlaying ? "block" : "none";
}

$btnPlay.addEventListener("click", () => {
    if (!avatar) return;
    isPlaying = !isPlaying;
    isPlaying ? avatar.play() : avatar.pause();
    updatePlayIcon();
});

$btnRestart.addEventListener("click", () => {
    if (!avatar) return;
    avatar.seekFrame(0);
    avatar.play();
    isPlaying = true;
    updatePlayIcon();
});

$timelineSlider.addEventListener("input", () => {
    if (!avatar) return;
    seekLock = true;
    const frame = parseInt($timelineSlider.value, 10);
    avatar.seekFrame(frame);
    const elapsed = frame / (avatar.fps || 30);
    $timeCurrent.textContent = formatTime(elapsed);
});

$timelineSlider.addEventListener("change", () => { seekLock = false; });

// Speed buttons
document.querySelectorAll(".btn-speed").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const speed = parseFloat(btn.dataset.speed);
        if (avatar) avatar.setSpeed(speed);
    });
});

// Mirror
$btnMirror.addEventListener("click", () => {
    isMirrored = !isMirrored;
    $btnMirror.classList.toggle("active", isMirrored);
    if (avatar) avatar.setMirror(isMirrored);
});

// New upload
$btnNew.addEventListener("click", () => {
    if (avatar) { avatar.pause(); avatar.dispose(); avatar = null; }
    if ($video.src) { URL.revokeObjectURL($video.src); $video.removeAttribute("src"); }
    $video.classList.remove("visible");
    $videoPlaceholder.classList.remove("visible");
    $fileInput.value = "";
    videoFile = null;
    isPlaying = false;
    isMirrored = false;
    $btnMirror.classList.remove("active");
    showSection($uploadSection);
});

// ── Boot ─────────────────────────────────────────────────────────

initBgParticles();
showSection($uploadSection);
