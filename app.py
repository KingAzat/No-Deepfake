"""
No-DeepFake Forensic Lab — app.py
Flask application principale : routing, upload, orchestration des modules de détection,
système de scoring pondéré et rendu du rapport forensique.
"""

import os
import uuid
import json
import time
import mimetypes
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# ─── Imports des modules de détection ─────────────────────────────────────────
from detector_image import analyze_image
# from detector_audio import analyze_audio   # Décommenter quand implémenté
# from detector_video import analyze_video   # Décommenter quand implémenté

# ─── Configuration ────────────────────────────────────────────────────────────

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    "image": {"jpg", "jpeg", "png", "webp", "bmp", "tiff"},
    "audio": {"mp3", "wav", "flac", "ogg", "m4a"},
    "video": {"mp4", "mov", "avi", "mkv", "webm"},
}

MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

# Poids par défaut du scoring global (configurables via WEIGHTS_CONFIG)
WEIGHTS_CONFIG = {
    "image": {
        "lighting":     0.20,   # 🔥 physique = très fiable
        "structural":   0.18,   # 🔥 symétrie + structure
        "repetition":   0.15,   # 🔥 motifs clonés
        
        "fft":          0.12,   # bon complément (artefacts diffusion)
        "noise":        0.10,   # utile mais moins discriminant
        "watermark":    0.10,   # utile seulement si présent
        
        "multiscale":   0.08,   # cohérence globale
        
        "ela":          0.05,   # ❌ faible (JPEG dépendant)
        "color_hist":   0.02,   # ❌ très faible (IA très bonnes ici)
    },
    "audio": {
        "spectral":     0.30,
        "background":   0.25,
        "prosody":      0.25,
        "phase":        0.20,
    },
    "video": {
        "temporal":     0.35,
        "face_track":   0.35,
        "lighting":     0.30,
    },
}

# ─── Application Flask ─────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.urandom(24)


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def detect_media_type(filename: str) -> str | None:
    """
    Détermine le type de média (image/audio/video) à partir de l'extension.
    Retourne None si le type n'est pas supporté.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    for media_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return media_type
    return None


def compute_weighted_score(scores: dict, weights: dict, combined_rules: dict = None) -> float:
    """
    Calcule le score final pondéré à partir des scores individuels,
    puis applique les ajustements des règles combinées.

    formula: final = Σ(score_i × weight_i) / Σ(weight_i) + adjustment
    Normalise par la somme des poids disponibles (gère les analyses manquantes).
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for key, weight in weights.items():
        if key in scores and scores[key] is not None:
            weighted_sum += scores[key] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    base_score = weighted_sum / total_weight

    # Appliquer l'ajustement des règles combinées
    adjustment = 0.0
    if combined_rules and "adjustment" in combined_rules:
        adjustment = combined_rules["adjustment"]

    final = max(0.0, min(1.0, base_score + adjustment))
    return round(final, 4)


def build_verdict(final_score: float) -> dict:
    """
    Traduit le score final en verdict forensique structuré.

    Seuils calibrés empiriquement :
      < 0.25  → Authentic        (très probable réel)
      0.25-0.50 → Suspicious     (éléments douteux)
      0.50-0.75 → Likely AI      (forte probabilité générée)
      > 0.75  → AI-Generated     (quasi-certitude)
    """
    pct = round(final_score * 100, 1)

    if final_score < 0.25:
        return {
            "label": "AUTHENTIC",
            "confidence": "HIGH",
            "color": "authentic",
            "probability": pct,
            "description": (
                "Aucun marqueur forensique significatif détecté. "
                "Le contenu présente des caractéristiques cohérentes avec une origine humaine."
            ),
        }
    elif final_score < 0.50:
        return {
            "label": "SUSPICIOUS",
            "confidence": "MEDIUM",
            "color": "warning",
            "probability": pct,
            "description": (
                "Certains indicateurs atypiques ont été détectés. "
                "Une vérification manuelle complémentaire est recommandée."
            ),
        }
    elif final_score < 0.75:
        return {
            "label": "LIKELY AI-GENERATED",
            "confidence": "HIGH",
            "color": "danger",
            "probability": pct,
            "description": (
                "Plusieurs marqueurs forensiques caractéristiques des modèles génératifs "
                "ont été identifiés. Le contenu présente une forte probabilité d'origine synthétique."
            ),
        }
    else:
        return {
            "label": "AI-GENERATED",
            "confidence": "VERY HIGH",
            "color": "danger",
            "probability": pct,
            "description": (
                "Signatures forensiques irréfutables d'une génération par IA. "
                "Artefacts de modèle diffusion/GAN/vocodeur identifiés avec haute confiance."
            ),
        }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint principal d'analyse.
    Reçoit un fichier multipart, le sauvegarde temporairement,
    orchestre les détecteurs appropriés et retourne le rapport JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400

    filename = secure_filename(file.filename)
    media_type = detect_media_type(filename)

    if media_type is None:
        return jsonify({"error": f"Type de fichier non supporté : {filename}"}), 415

    # Sauvegarde temporaire avec UUID pour éviter les collisions
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = UPLOAD_FOLDER / unique_name

    try:
        file.save(str(file_path))
        t_start = time.monotonic()

        # ── Dispatch vers le bon module ────────────────────────────────────
        if media_type == "image":
            analysis_results = analyze_image(str(file_path))
            weights = WEIGHTS_CONFIG["image"]

        elif media_type == "audio":
            # analysis_results = analyze_audio(str(file_path))
            return jsonify({"error": "Module audio en cours de développement"}), 501

        elif media_type == "video":
            # analysis_results = analyze_video(str(file_path))
            return jsonify({"error": "Module vidéo en cours de développement"}), 501

        t_elapsed = round(time.monotonic() - t_start, 3)

        # Extraction des scores numériques depuis les résultats
        scores = {
            key: result.get("score")
            for key, result in analysis_results.items()
            if isinstance(result, dict) and "score" in result
        }

        final_score = compute_weighted_score(scores, weights, 
                                               analysis_results.get("combined_rules"))
        verdict = build_verdict(final_score)

        # Extraire les règles combinées si présentes
        combined_rules = analysis_results.get("combined_rules", {})

        # ── Construction du rapport final ──────────────────────────────────
        report = {
            "analysis_id": f"#NDF-{uuid.uuid4().hex[:6].upper()}",
            "filename": filename,
            "media_type": media_type,
            "analysis_time_s": t_elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "final_score": final_score,
            "verdict": verdict,
            "details": analysis_results,
            "scores_summary": scores,
            "combined_rules": combined_rules,
        }

        return jsonify(report)

    except Exception as e:
        app.logger.error(f"Erreur d'analyse : {e}", exc_info=True)
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

    finally:
        # Nettoyage du fichier temporaire
        if file_path.exists():
            file_path.unlink()


@app.route("/health")
def health():
    """Endpoint de santé pour vérifier que le serveur tourne."""
    return jsonify({"status": "operational", "version": "3.0"})


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  No-DeepFake Forensic Lab — v3.0")
    print("  Scoring intelligent + règles combinées")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
