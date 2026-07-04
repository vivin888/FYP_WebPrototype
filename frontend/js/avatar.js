/**
 * SignAvatar – Three.js 3D Avatar Renderer  (v2 — precision rewrite)
 *
 * Key improvements over v1:
 *  • Sub-frame linear interpolation → silky smooth motion between keyframes
 *  • Temporal smoothing (EMA) → removes jitter / landmark noise
 *  • Aspect-ratio-corrected coordinate mapping → accurate proportions
 *  • Reduced Z scale → correct depth perception
 *  • Hand landmarks re-anchored to pose wrist for perfect fit
 *  • Dedicated spine bone (hip-mid → shoulder-mid)
 *  • Head sphere raised between ears (not just nose)
 *  • Tapered bone cylinders (wider at root, narrower at tip)
 *  • Visibility-gated rendering (ignores low-confidence landmarks)
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

// ── MediaPipe landmark indices ───────────────────────────────────
const POSE = {
    NOSE: 0,
    L_EYE_INNER: 1, L_EYE: 2, L_EYE_OUTER: 3,
    R_EYE_INNER: 4, R_EYE: 5, R_EYE_OUTER: 6,
    L_EAR: 7,  R_EAR: 8,
    MOUTH_L: 9, MOUTH_R: 10,
    L_SHOULDER: 11, R_SHOULDER: 12,
    L_ELBOW: 13,    R_ELBOW: 14,
    L_WRIST: 15,    R_WRIST: 16,
    L_PINKY: 17,    R_PINKY: 18,
    L_INDEX: 19,    R_INDEX: 20,
    L_THUMB: 21,    R_THUMB: 22,
    L_HIP: 23,      R_HIP: 24,
    L_KNEE: 25,     R_KNEE: 26,
    L_ANKLE: 27,    R_ANKLE: 28,
    L_HEEL: 29,     R_HEEL: 30,
    L_FOOT: 31,     R_FOOT: 32,
};

// ── Bone topology  [fromIdx, toIdx, topRadius, bottomRadius] ─────
//   radii control the taper: shoulder end is wider, wrist end narrower
const BODY_BONES = [
    // Torso cross-bars
    [POSE.L_SHOULDER, POSE.R_SHOULDER, 0.028, 0.028],
    [POSE.L_HIP,      POSE.R_HIP,      0.028, 0.028],
    // Spine (derived below — hip-mid → shoulder-mid)
    // Left side body
    [POSE.L_SHOULDER, POSE.L_HIP,      0.025, 0.022],
    [POSE.R_SHOULDER, POSE.R_HIP,      0.025, 0.022],
    // Left arm
    [POSE.L_SHOULDER, POSE.L_ELBOW,    0.026, 0.020],
    [POSE.L_ELBOW,    POSE.L_WRIST,    0.020, 0.014],
    // Right arm
    [POSE.R_SHOULDER, POSE.R_ELBOW,    0.026, 0.020],
    [POSE.R_ELBOW,    POSE.R_WRIST,    0.020, 0.014],
    // Left leg
    [POSE.L_HIP,      POSE.L_KNEE,     0.030, 0.024],
    [POSE.L_KNEE,     POSE.L_ANKLE,    0.024, 0.016],
    [POSE.L_ANKLE,    POSE.L_HEEL,     0.012, 0.010],
    [POSE.L_ANKLE,    POSE.L_FOOT,     0.014, 0.010],
    // Right leg
    [POSE.R_HIP,      POSE.R_KNEE,     0.030, 0.024],
    [POSE.R_KNEE,     POSE.R_ANKLE,    0.024, 0.016],
    [POSE.R_ANKLE,    POSE.R_HEEL,     0.012, 0.010],
    [POSE.R_ANKLE,    POSE.R_FOOT,     0.014, 0.010],
];

// Face connections (thin wires)
const FACE_BONES = [
    [POSE.L_EAR, POSE.L_EYE],
    [POSE.R_EAR, POSE.R_EYE],
    [POSE.L_EYE, POSE.NOSE],
    [POSE.R_EYE, POSE.NOSE],
    [POSE.MOUTH_L, POSE.MOUTH_R],
];

// Finger chains for 21-landmark hands
const FINGER_CHAINS = [
    [0, 1, 2, 3, 4],     // thumb
    [0, 5, 6, 7, 8],     // index
    [0, 9, 10, 11, 12],  // middle
    [0, 13, 14, 15, 16], // ring
    [0, 17, 18, 19, 20], // pinky
];
const PALM_LINKS = [[5, 9], [9, 13], [13, 17], [0, 5], [0, 17]];

// Finger bone taper: wider at base, thinner at tip
const FINGER_RADII = [
    [0.010, 0.008], // segment 0→1
    [0.008, 0.006], // segment 1→2
    [0.006, 0.005], // segment 2→3
    [0.005, 0.004], // segment 3→4
];

// ── Coordinate mapping ───────────────────────────────────────────
//
// MediaPipe normalises to [0,1]×[0,1] (origin = top-left).
// We map to a world space centred at origin, scaling X by aspect
// ratio so the skeleton isn't squished for non-square videos.
//
const WORLD_HEIGHT = 5.0;   // how tall the full body is in world units
const Z_SCALE      = 1.8;   // depth scale (< WORLD_HEIGHT to avoid distortion)

let _aspectRatio = 4 / 3;  // updated from video metadata

export function setVideoAspect(w, h) {
    _aspectRatio = (w && h && h > 0) ? w / h : 4 / 3;
}

function toWorld(lm, mirror = false) {
    const xs = mirror ? -1 : 1;
    return new THREE.Vector3(
        (lm.x - 0.5) * WORLD_HEIGHT * _aspectRatio * xs,
        -(lm.y - 0.5) * WORLD_HEIGHT,
        -(lm.z ?? 0) * Z_SCALE
    );
}

// ── Linear interpolation helpers ────────────────────────────────
function lerpLm(a, b, t) {
    return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: (a.z ?? 0) + ((b.z ?? 0) - (a.z ?? 0)) * t,
        v: (a.v ?? 1),
    };
}

function lerpFrame(f0, f1, t) {
    if (!f0) return f1;
    if (!f1) return f0;
    const out = { pose: null, left_hand: null, right_hand: null };

    if (f0.pose && f1.pose)
        out.pose = f0.pose.map((lm, i) => lerpLm(lm, f1.pose[i], t));
    else
        out.pose = f0.pose || f1.pose;

    if (f0.left_hand && f1.left_hand)
        out.left_hand = f0.left_hand.map((lm, i) => lerpLm(lm, f1.left_hand[i], t));
    else
        out.left_hand = f0.left_hand || f1.left_hand;

    if (f0.right_hand && f1.right_hand)
        out.right_hand = f0.right_hand.map((lm, i) => lerpLm(lm, f1.right_hand[i], t));
    else
        out.right_hand = f0.right_hand || f1.right_hand;

    return out;
}

// ── Temporal smoother (exponential moving average) ───────────────
const EMA_ALPHA = 0.55;   // higher = more responsive, lower = smoother

function smoothArray(prev, next, alpha) {
    if (!prev || !next) return next;
    return next.map((lm, i) => ({
        x: prev[i].x + alpha * (lm.x - prev[i].x),
        y: prev[i].y + alpha * (lm.y - prev[i].y),
        z: (prev[i].z ?? 0) + alpha * ((lm.z ?? 0) - (prev[i].z ?? 0)),
        v: lm.v ?? 1,
    }));
}

// ── SignAvatar Class ─────────────────────────────────────────────
export class SignAvatar {
    constructor(canvas) {
        this.canvas = canvas;
        this.mirror  = false;
        this.disposed = false;

        // Playback
        this.landmarkData  = null;
        this.currentFrame  = 0;
        this.playing       = false;
        this.speed         = 1.0;
        this.fps           = 30;
        this._subT         = 0;     // sub-frame interpolation t ∈ [0,1)
        this._lastTime     = 0;
        this._smoothPose   = null;  // EMA state
        this._smoothLH     = null;
        this._smoothRH     = null;

        // Mesh pools
        this._boneMeshes  = [];   // { mesh, a, b } or { mesh } for derived
        this._faceMeshes  = [];
        this._spineMesh   = null;
        this._neckMesh    = null;
        this._headSphere  = null;
        this._jointMeshes = [];   // body joints 0..32
        this._lJoints = []; this._lBones = [];
        this._rJoints = []; this._rBones = [];

        this._initScene();
        this._buildAvatar();
        this._animate = this._animate.bind(this);
        this._onResize = this._onResize.bind(this);
        window.addEventListener("resize", this._onResize);
        this._animate(0);
    }

    // ── Scene ─────────────────────────────────────────────────────
    _initScene() {
        const w = this.canvas.parentElement.clientWidth  || 600;
        const h = this.canvas.parentElement.clientHeight || 480;

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(w, h);
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.15;

        this.scene  = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(48, w / h, 0.05, 100);
        this.camera.position.set(0, 0.5, 4.2);

        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping  = true;
        this.controls.dampingFactor  = 0.07;
        this.controls.target.set(0, 0.2, 0);
        this.controls.minDistance = 1.0;
        this.controls.maxDistance = 12;

        // Bloom
        this.composer = new EffectComposer(this.renderer);
        this.composer.addPass(new RenderPass(this.scene, this.camera));
        this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.38, 0.88));

        // Lighting
        this.scene.add(new THREE.AmbientLight(0x3344aa, 0.75));
        const key = new THREE.DirectionalLight(0xb0d0ff, 1.1);
        key.position.set(2, 5, 4); this.scene.add(key);
        const fill = new THREE.DirectionalLight(0x7744cc, 0.35);
        fill.position.set(-3, 1, -2); this.scene.add(fill);
        const rim = new THREE.DirectionalLight(0x00ffff, 0.2);
        rim.position.set(0, -2, 3); this.scene.add(rim);

        // Floor
        const grid = new THREE.GridHelper(10, 32, 0x111133, 0x0a0a22);
        grid.position.y = -2.6; this.scene.add(grid);

        const pg = new THREE.PlaneGeometry(12, 12);
        const pm = new THREE.MeshStandardMaterial({ color: 0x06060f, transparent: true, opacity: 0.55, metalness: 0.85, roughness: 0.7 });
        const plane = new THREE.Mesh(pg, pm);
        plane.rotation.x = -Math.PI / 2; plane.position.y = -2.62;
        this.scene.add(plane);

        this._createParticles();
    }

    _createParticles() {
        const count = 250;
        const pos   = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            pos[i * 3]     = (Math.random() - 0.5) * 14;
            pos[i * 3 + 1] = (Math.random() - 0.5) * 10;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 10;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        this._particles = new THREE.Points(geo,
            new THREE.PointsMaterial({ color: 0x00e5ff, size: 0.018, transparent: true, opacity: 0.28, sizeAttenuation: true }));
        this.scene.add(this._particles);
    }

    // ── Materials ─────────────────────────────────────────────────
    _jMat(color, ei = 0.65) {
        return new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: ei, metalness: 0.15, roughness: 0.3 });
    }
    _bMat(color, opacity = 0.9) {
        return new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.22, metalness: 0.1, roughness: 0.5, transparent: opacity < 1, opacity });
    }

    // ── Avatar Construction ────────────────────────────────────────
    _makeTaperedBone(rTop, rBot) {
        return new THREE.CylinderGeometry(rTop, rBot, 1, 8, 1);
    }

    _buildAvatar() {
        const BODY_J  = 0x00e5ff;
        const BODY_B  = 0x007a8a;
        const HAND_J  = 0xffcc44;
        const HAND_B  = 0xe08c00;
        const HEAD_C  = 0x22eeff;
        const FACE_C  = 0x005566;
        const SPINE_C = 0x0088aa;

        // 33 body joint spheres
        const jGeo = new THREE.IcosahedronGeometry(0.038, 2);
        const jMat = this._jMat(BODY_J);
        for (let i = 0; i < 33; i++) {
            const m = new THREE.Mesh(jGeo, jMat);
            m.visible = false;
            this.scene.add(m);
            this._jointMeshes.push(m);
        }

        // Head sphere — centred between ears, slightly raised
        this._headSphere = new THREE.Mesh(
            new THREE.IcosahedronGeometry(0.18, 3),
            this._jMat(HEAD_C, 0.35)
        );
        this._headSphere.visible = false;
        this.scene.add(this._headSphere);

        // Spine bone (hip-mid → shoulder-mid)  — single special mesh
        this._spineMesh = new THREE.Mesh(this._makeTaperedBone(0.030, 0.026), this._bMat(SPINE_C));
        this._spineMesh.visible = false;
        this.scene.add(this._spineMesh);

        // Neck bone (shoulder-mid → nose)
        this._neckMesh = new THREE.Mesh(this._makeTaperedBone(0.022, 0.016), this._bMat(SPINE_C));
        this._neckMesh.visible = false;
        this.scene.add(this._neckMesh);

        // Body bones (tapered)
        for (const [, , rt, rb] of BODY_BONES) {
            const m = new THREE.Mesh(this._makeTaperedBone(rt, rb), this._bMat(BODY_B));
            m.visible = false;
            this.scene.add(m);
            this._boneMeshes.push(m);
        }

        // Face bones (very thin)
        for (let i = 0; i < FACE_BONES.length; i++) {
            const m = new THREE.Mesh(this._makeTaperedBone(0.007, 0.007), this._bMat(FACE_C, 0.7));
            m.visible = false;
            this.scene.add(m);
            this._faceMeshes.push(m);
        }

        // Hands
        this._buildHand(this._lJoints, this._lBones, HAND_J, HAND_B);
        this._buildHand(this._rJoints, this._rBones, HAND_J, HAND_B);
    }

    _buildHand(joints, bones, jColor, bColor) {
        const jGeo = new THREE.IcosahedronGeometry(0.016, 1);
        const jMat = this._jMat(jColor, 0.75);

        for (let i = 0; i < 21; i++) {
            const m = new THREE.Mesh(jGeo, jMat);
            m.visible = false;
            this.scene.add(m);
            joints.push(m);
        }

        // Finger segments
        for (const chain of FINGER_CHAINS) {
            for (let s = 0; s < chain.length - 1; s++) {
                const [rt, rb] = FINGER_RADII[Math.min(s, FINGER_RADII.length - 1)];
                const m = new THREE.Mesh(this._makeTaperedBone(rt, rb), this._bMat(bColor, 0.82));
                m.visible = false;
                this.scene.add(m);
                bones.push({ mesh: m, a: chain[s], b: chain[s + 1] });
            }
        }
        // Palm links
        for (const [a, b] of PALM_LINKS) {
            const m = new THREE.Mesh(this._makeTaperedBone(0.007, 0.007), this._bMat(bColor, 0.65));
            m.visible = false;
            this.scene.add(m);
            bones.push({ mesh: m, a, b });
        }
    }

    // ── Bone placement ────────────────────────────────────────────
    _placeBone(mesh, pA, pB) {
        const dir = new THREE.Vector3().subVectors(pB, pA);
        const len = dir.length();
        if (len < 0.001) { mesh.visible = false; return; }

        mesh.position.copy(pA).addScaledVector(dir, 0.5);
        mesh.scale.set(1, len, 1);
        mesh.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            dir.normalize()
        );
        mesh.visible = true;
    }

    // ── Frame rendering ───────────────────────────────────────────
    _renderFrame(frame) {
        if (!frame) return;
        const { pose, left_hand, right_hand } = frame;

        // ── Body ──
        if (!pose) {
            this._jointMeshes.forEach(m => m.visible = false);
            this._boneMeshes.forEach(m => m.visible = false);
            this._faceMeshes.forEach(m => m.visible = false);
            this._spineMesh.visible = false;
            this._neckMesh.visible  = false;
            this._headSphere.visible = false;
        } else {
            const P = pose.map(lm => toWorld(lm, this.mirror));

            // Joints — hide low-visibility face landmarks to reduce clutter
            for (let i = 0; i < 33; i++) {
                const vis = pose[i].v ?? 1;
                const isFace = i <= 10;
                this._jointMeshes[i].position.copy(P[i]);
                this._jointMeshes[i].visible = !isFace && vis > 0.35;
            }

            // Head sphere — midpoint between ears, slightly raised
            const headPos = new THREE.Vector3()
                .addVectors(P[POSE.L_EAR], P[POSE.R_EAR])
                .multiplyScalar(0.5);
            headPos.y += 0.06; // raise slightly above ear line
            this._headSphere.position.copy(headPos);
            this._headSphere.visible = true;

            // Spine: hip-mid → shoulder-mid
            const hipMid = new THREE.Vector3()
                .addVectors(P[POSE.L_HIP], P[POSE.R_HIP]).multiplyScalar(0.5);
            const shoulderMid = new THREE.Vector3()
                .addVectors(P[POSE.L_SHOULDER], P[POSE.R_SHOULDER]).multiplyScalar(0.5);
            this._placeBone(this._spineMesh, hipMid, shoulderMid);

            // Neck: shoulder-mid → nose
            this._placeBone(this._neckMesh, shoulderMid, P[POSE.NOSE]);

            // Body bones
            for (let i = 0; i < BODY_BONES.length; i++) {
                const [aIdx, bIdx] = BODY_BONES[i];
                this._placeBone(this._boneMeshes[i], P[aIdx], P[bIdx]);
            }

            // Face bones
            for (let i = 0; i < FACE_BONES.length; i++) {
                const [aIdx, bIdx] = FACE_BONES[i];
                const visA = pose[aIdx].v ?? 1;
                const visB = pose[bIdx].v ?? 1;
                if (visA > 0.3 && visB > 0.3) {
                    this._placeBone(this._faceMeshes[i], P[aIdx], P[bIdx]);
                } else {
                    this._faceMeshes[i].visible = false;
                }
            }

            // Re-anchor hands to pose wrists for perfect fit
            this._renderHand(left_hand,  P[POSE.L_WRIST], this._lJoints, this._lBones);
            this._renderHand(right_hand, P[POSE.R_WRIST], this._rJoints, this._rBones);
        }
    }

    /**
     * Render a hand and re-anchor it to the pose wrist landmark.
     * MediaPipe hand landmarks are in image-space themselves, but
     * aligning wrist [0] to the pose wrist closes any gap that arises
     * from independent detection models.
     */
    _renderHand(landmarks, poseWrist, joints, bones) {
        if (!landmarks) {
            joints.forEach(m => m.visible = false);
            bones.forEach(b => b.mesh.visible = false);
            return;
        }

        // Convert all 21 landmarks
        const P = landmarks.map(lm => toWorld(lm, this.mirror));

        // Compute offset between hand's own wrist (P[0]) and pose wrist
        const offset = poseWrist
            ? new THREE.Vector3().subVectors(poseWrist, P[0])
            : new THREE.Vector3();

        // Apply offset to all landmarks
        const PA = P.map(p => p.clone().add(offset));

        for (let i = 0; i < 21; i++) {
            joints[i].position.copy(PA[i]);
            joints[i].visible = true;
        }
        for (const bone of bones) {
            this._placeBone(bone.mesh, PA[bone.a], PA[bone.b]);
        }
    }

    // ── EMA smoothing ─────────────────────────────────────────────
    _applySmoothing(frame) {
        if (!frame) return frame;

        this._smoothPose = smoothArray(this._smoothPose, frame.pose,       EMA_ALPHA);
        this._smoothLH   = smoothArray(this._smoothLH,   frame.left_hand,  EMA_ALPHA);
        this._smoothRH   = smoothArray(this._smoothRH,   frame.right_hand, EMA_ALPHA);

        return {
            pose:       this._smoothPose,
            left_hand:  this._smoothLH,
            right_hand: this._smoothRH,
        };
    }

    // ── Animation loop ────────────────────────────────────────────
    _animate(time) {
        if (this.disposed) return;
        requestAnimationFrame(this._animate);

        const rawDt = (time - this._lastTime) / 1000;
        const dt = Math.min(rawDt, 0.1); // clamp to avoid spiral of death
        this._lastTime = time;

        if (this._particles) this._particles.rotation.y += dt * 0.018;

        if (this.playing && this.landmarkData) {
            const frameDur = 1.0 / (this.fps * this.speed);
            this._subT += dt / frameDur;

            while (this._subT >= 1) {
                this._subT -= 1;
                this.currentFrame++;
                if (this.currentFrame >= this.landmarkData.frames.length) {
                    this.currentFrame = 0;
                    this._smoothPose = null; // reset smoother on loop
                    this._smoothLH   = null;
                    this._smoothRH   = null;
                }
            }

            // Sub-frame interpolation between currentFrame and next
            const nextFrame = (this.currentFrame + 1) % this.landmarkData.frames.length;
            const blended   = lerpFrame(
                this.landmarkData.frames[this.currentFrame],
                this.landmarkData.frames[nextFrame],
                this._subT
            );
            const smoothed = this._applySmoothing(blended);
            this._renderFrame(smoothed);

            this.canvas.dispatchEvent(new CustomEvent("framechange", {
                detail: { frame: this.currentFrame, total: this.landmarkData.frames.length }
            }));
        }

        this.controls.update();
        this.composer.render();
    }

    // ── Public API ────────────────────────────────────────────────
    loadData(data) {
        this.landmarkData = data;
        this.fps          = data.fps || 30;
        this.currentFrame = 0;
        this._subT        = 0;
        this._smoothPose  = null;
        this._smoothLH    = null;
        this._smoothRH    = null;

        // Set aspect ratio for correct coordinate mapping
        if (data.width && data.height) setVideoAspect(data.width, data.height);

        if (data.frames.length > 0) {
            this._renderFrame(data.frames[0]);
        }
    }

    play()  { this.playing = true;  this._subT = 0; }
    pause() { this.playing = false; }

    seekFrame(idx) {
        if (!this.landmarkData) return;
        this.currentFrame = Math.max(0, Math.min(idx, this.landmarkData.frames.length - 1));
        this._subT = 0;
        this._smoothPose = null;
        this._smoothLH   = null;
        this._smoothRH   = null;
        this._renderFrame(this.landmarkData.frames[this.currentFrame]);
    }

    setSpeed(s)  { this.speed  = s; }
    setMirror(m) {
        this.mirror = m;
        if (this.landmarkData?.frames.length > 0)
            this._renderFrame(this.landmarkData.frames[this.currentFrame]);
    }

    _onResize() {
        if (this.disposed) return;
        const w = this.canvas.parentElement.clientWidth  || 600;
        const h = this.canvas.parentElement.clientHeight || 480;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        this.composer.setSize(w, h);
    }

    dispose() {
        this.disposed = true;
        window.removeEventListener("resize", this._onResize);
        this.controls.dispose();
        this.renderer.dispose();
    }
}
