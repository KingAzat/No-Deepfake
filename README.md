# 🛡️ No-DeepFake Forensic Lab — v3.0

> **AI-generated media detection through multi-vector forensic analysis.**  
> Upload an image and receive a detailed forensic report with per-marker scores, combined scoring rules, and a final verdict.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **ELA** | Error Level Analysis — detects JPEG re-compression inconsistencies (auto-weighted by format reliability) |
| **FFT** | 2D Fourier Transform — identifies periodic GAN artifacts and 1/f spectral deviations |
| **Color Distribution** | RGB histogram entropy, peak sharpness, and saturation analysis |
| **Noise Structure** | Laplacian residual noise — Gaussian test (kurtosis) + spatial autocorrelation |
| **Watermark / Signature** | EXIF metadata scan, LSB steganography patterns, and localized high-frequency zones |
| **Lighting Physics** | Light direction coherence, vignetting ratio, and illumination symmetry |
| **Structural Integrity** | Horizontal symmetry, texture patch similarity, sharpness variance, and block entropy |
| **Repetition / Clone Detection** | Block cosine-similarity matrix + 2D autocorrelation for tiled textures |
| **Multi-Scale Coherence** | Full image vs. quadrants vs. 50% downscale — cross-scale consistency check |

**Weighted Scoring Engine** — each marker contributes with empirically-calibrated weights.  
**Combined Rules** — bonus/malus adjustments when signals converge or diverge (e.g., FFT + Noise → GAN signature confirmed).

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python app.py
```

Open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

## 📁 Project Structure

```
no-deepfake/
├── app.py               # Flask server — routing, upload, orchestration, scoring
├── detector_image.py    # 9 forensic image analysis modules (v3.0)
├── index.html           # Frontend UI (served via Flask templates/)
├── templates/
│   └── index.html       # Template served by Flask (symlink or copy of index.html)
├── uploads/             # Temporary file storage (auto-cleaned after each analysis)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🔌 API

### `POST /analyze`

Upload a media file for forensic analysis.

**Request:** `multipart/form-data` with field `file`

**Supported formats:**
- Image: `jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`
- Audio: `mp3`, `wav`, `flac`, `ogg`, `m4a` *(module coming soon)*
- Video: `mp4`, `mov`, `avi`, `mkv`, `webm` *(module coming soon)*

**Response (JSON):**

```json
{
  "analysis_id": "#NDF-A1B2C3",
  "filename": "photo.jpg",
  "media_type": "image",
  "analysis_time_s": 1.234,
  "timestamp": "2026-05-06 13:00 UTC",
  "final_score": 0.7821,
  "verdict": {
    "label": "LIKELY AI-GENERATED",
    "confidence": "HIGH",
    "color": "danger",
    "probability": 78.2,
    "description": "..."
  },
  "details": { ... },
  "scores_summary": { "ela": 0.12, "fft": 0.85, ... },
  "combined_rules": {
    "adjustment": 0.18,
    "rules_applied": ["+0.10 — FFT et bruit convergent (signaux GAN)", ...]
  }
}
```

### `GET /health`

Returns server status.

```json
{ "status": "operational", "version": "3.0" }
```

---

## 🎯 Verdict Thresholds

| Score Range | Label | Confidence |
|-------------|-------|------------|
| `< 0.25` | ✅ AUTHENTIC | HIGH |
| `0.25 – 0.50` | ⚠️ SUSPICIOUS | MEDIUM |
| `0.50 – 0.75` | 🚨 LIKELY AI-GENERATED | HIGH |
| `> 0.75` | ❌ AI-GENERATED | VERY HIGH |

---

## ⚖️ Marker Weights (Image)

| Marker | Weight | Rationale |
|--------|--------|-----------|
| Lighting | 20% | Physical coherence — highly reliable |
| Structural | 18% | Symmetry + structure |
| Repetition | 15% | Clone patterns — typical of GAN/diffusion |
| FFT | 12% | Spectral artifact of diffusion models |
| Noise | 10% | Useful but less discriminating |
| Watermark | 10% | Decisive when present |
| Multi-Scale | 8% | Global cross-scale coherence |
| ELA | 5% | JPEG-dependent, lower reliability |
| Color Hist | 2% | AI models have gotten very good here |

---

## 📦 Dependencies

```
flask>=3.0.0
Pillow>=10.0.0
numpy>=1.24.0
werkzeug>=3.0.0
```

> Audio/video modules require additional packages — see comments in `requirements.txt`.

---

## 🔒 Security Notes

- Uploaded files are saved with a UUID prefix and **deleted immediately** after analysis.
- Max upload size: **200 MB**.
- The app binds to `0.0.0.0` for local use. Do not expose to the internet without a reverse proxy and authentication.

---

## 📜 License

MIT — free to use, modify, and distribute.
