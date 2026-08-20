# 🛡️ No-DeepFake Forensic Lab — v3.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-stable-green.svg)]()

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
- (Optional but recommended) `venv` or `virtualenv`

### 2. Clone & Setup

```bash
# Clone the repository (if applicable)
# git clone https://github.com/yourusername/no-deepfake.git
# cd no-deepfake

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

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

> ⚠️ **Note:** These thresholds are calibrated on current-generation models. Results should be interpreted as probabilistic indicators, not absolute proof.

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

## 💡 Usage Examples

### Via Web Interface
1. Navigate to `http://localhost:5000`
2. Drag & drop or select an image file
3. Click "Analyze" and wait for the forensic report

### Via cURL (API)
```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@/path/to/your/image.jpg"
```

### Via Python Script
```python
import requests

files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/analyze', files=files)
print(response.json())
```

---

## ⚠️ Limitations & Known Issues

- **JPEG Compression:** Heavily compressed images may produce false positives on ELA.
- **Resolution:** Very low-resolution images (< 100x100px) may yield unreliable results.
- **Artistic Filters:** Images with heavy artistic filters or post-processing may trigger false alarms.
- **New Models:** Rapidly evolving AI generation techniques may temporarily bypass detection until recalibration.
- **Audio/Video:** These modules are not yet implemented (coming in v4.0).

---

## ❓ FAQ / Troubleshooting

**Q: Why is my analysis taking so long?**  
A: Large images (> 4000px) require more processing time. Consider resizing before upload.

**Q: Can I trust a "AUTHENTIC" verdict?**  
A: While our system is robust, no detector is infallible. Use results as one piece of evidence among others.

**Q: The server won't start — port 5000 already in use?**  
A: Change the port in `app.py`: `app.run(host='0.0.0.0', port=8080)`

**Q: How often should I update the model weights?**  
A: We recommend recalibrating every 3–6 months as new generative models emerge.

---

## 🔒 Security Notes

- Uploaded files are saved with a UUID prefix and **deleted immediately** after analysis.
- Max upload size: **200 MB**.
- The app binds to `0.0.0.0` for local use. Do not expose to the internet without a reverse proxy and authentication.
- For production deployment, consider adding rate limiting, HTTPS, and user authentication.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) (if available) before contributing.

---

## 📚 Resources & Further Reading

- [DeepFake Detection Challenge (Facebook)](https://ai.facebook.com/datasets/dfdc/)
- [Google Jigsaw DeepFake Dataset](https://github.com/google/deepfake-detection)
- [Forensic Analysis of Digital Images (Academic Paper)](https://ieeexplore.ieee.org/document/XXXXXXX)
- [Understanding GAN Artifacts in Frequency Domain](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📜 License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- Your Name — *Initial work* — [YourGitHub](https://github.com/yourusername)

See also the list of [contributors](https://github.com/yourusername/no-deepfake/contributors) who participated in this project.

---

## 🙏 Acknowledgments

- Thanks to the forensic imaging research community
- Inspired by work from [Organization/Person Name]
- Special thanks to all beta testers and contributors
