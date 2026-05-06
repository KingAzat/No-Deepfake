"""
No-DeepFake Forensic Lab — detector_image.py  (v3.0)
Module de détection d'images générées par IA.

Implémente 9 analyses forensiques indépendantes :
  1. ELA        — Error Level Analysis (cohérence JPEG, fiabilité réduite si non-JPEG)
  2. FFT        — Analyse fréquentielle (motifs GAN périodiques)
  3. COLOR      — Distribution histogrammique RGB
  4. NOISE      — Uniformité et artificialité du bruit
  5. WMARK      — Détection heuristique de watermarks/signatures
  6. LIGHT      — Cohérence physique lumière/ombres
  7. STRUCT     — Anomalies structurelles (symétrie, fréquences locales)
  8. REPETITION — Détection de motifs clonés / textures répétitives (IA typique)
  9. MULTISCALE — Analyse multi-échelle (image complète, quadrants, réduite)

Chaque analyse retourne un dict :
  { "score": float[0,1], "details": dict, "flag": bool }

score → 0.0 = très probablement réel | 1.0 = très probablement IA

Scoring final : pondération intelligente + règles combinées (bonus/malus).
"""

import io
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageChops, ExifTags

# Suppression des avertissements PIL sur les images tronquées
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES INTERNES
# ══════════════════════════════════════════════════════════════════════════════

def _load_image(path: str) -> Image.Image:
    """Charge l'image et la convertit en RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _to_array(img: Image.Image) -> np.ndarray:
    """PIL Image → numpy float32 [0,1]."""
    return np.asarray(img, dtype=np.float32) / 255.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _result(score: float, flag: bool = False, **details) -> dict:
    """Construit un résultat normalisé."""
    return {
        "score": round(_clamp(score), 4),
        "flag": flag,
        "details": details,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1. ANALYSE ELA (Error Level Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ela(img: Image.Image, quality: int = 90) -> dict:
    """
    Error Level Analysis : compare l'image originale à sa version re-compressée
    en JPEG à qualité contrôlée.

    v3.0 : Détecte le format source. Si non-JPEG, la fiabilité est réduite
    automatiquement (×0.4) car l'ELA est conçue pour les artefacts JPEG.

    Métriques extraites :
      - mean_ela      : niveau d'erreur moyen (élevé → altération possible)
      - std_ela       : écart-type (faible → trop homogène → suspect IA)
      - max_ela       : valeur maximale (pics d'artefacts)
      - uniformity    : 1 - (std/mean), ≈1 → bruit trop uniforme
      - reliability   : fiabilité de l'analyse (1.0 pour JPEG, 0.4 sinon)
    """
    # ── Détection du format source ─────────────────────────────────────────
    src_format = (getattr(img, 'format', '') or '').upper()
    is_jpeg = src_format in ('JPEG', 'JPG', 'MPO')
    reliability = 1.0 if is_jpeg else 0.4

    # Re-compression à qualité contrôlée
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    recompressed.load()

    # Différence pixel-à-pixel
    diff = ImageChops.difference(img, recompressed)
    diff_arr = np.asarray(diff, dtype=np.float32)

    mean_ela = float(np.mean(diff_arr))
    std_ela = float(np.std(diff_arr))
    max_ela = float(np.max(diff_arr))

    # Un std très bas par rapport à la moyenne indique une uniformité suspecte
    uniformity = 1.0 - (std_ela / (mean_ela + 1e-6))
    uniformity = _clamp(uniformity)

    # Score composite :
    mean_norm = _clamp(mean_ela / 30.0)
    uniform_weight = 0.5 if uniformity > 0.7 else 0.2

    raw_score = mean_norm * 0.5 + uniformity * uniform_weight + _clamp(max_ela / 80.0) * 0.2
    # Atténuation pour les formats non-JPEG
    score = _clamp(raw_score * reliability)

    # Seuil dynamique basé sur la fiabilité
    flag_threshold = 0.45 if is_jpeg else 0.65

    return _result(
        score,
        flag=(score > flag_threshold),
        mean_ela=round(mean_ela, 3),
        std_ela=round(std_ela, 3),
        max_ela=round(max_ela, 3),
        uniformity=round(uniformity, 3),
        reliability=round(reliability, 2),
        source_format=src_format or 'unknown',
        interpretation=(
            ("⚠️ Analyse ELA moins fiable (format non-JPEG). " if not is_jpeg else "")
            + (
                "Niveau d'erreur très homogène (signature de génération synthétique)"
                if uniformity > 0.7
                else "Erreurs de compression dans la plage normale"
            )
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  2. ANALYSE FFT (Transformée de Fourier)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_fft(img: Image.Image) -> dict:
    """
    Analyse fréquentielle par FFT 2D.

    Les modèles GAN et de diffusion introduisent des artefacts périodiques
    dans le spectre de fréquences :
      - Pics ponctuels dans le spectre → grille de convolution GAN
      - Énergie anormale dans les hautes fréquences → artefacts de déconvolution
      - Spectre trop "propre" → manque du bruit naturel 1/f

    Analyse :
      1. FFT 2D du canal de luminance
      2. Calcul du spectre log-magnitude centré
      3. Détection de pics anormaux (ratio crête/fond)
      4. Vérification de la loi 1/f (spectre naturel ∝ 1/f^α, α ≈ 2)
    """
    # Conversion en niveaux de gris
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float64)

    # FFT 2D + centrage
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)

    h, w = log_mag.shape
    cx, cy = h // 2, w // 2

    # ── Détection de pics anormaux ─────────────────────────────────────────
    # Exclut le DC central (5% de la zone centrale)
    mask = np.ones((h, w), dtype=bool)
    r_dc = int(min(h, w) * 0.05)
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - cx) ** 2 + (xx - cy) ** 2 < r_dc ** 2] = False

    flat = log_mag[mask]
    median_mag = float(np.median(flat))
    p99 = float(np.percentile(flat, 99))

    # Pics : valeurs > median + 3σ hors du disque central
    std_mag = float(np.std(flat))
    threshold_peaks = median_mag + 3.5 * std_mag
    n_peaks = int(np.sum(flat > threshold_peaks))
    peak_ratio = _clamp(n_peaks / (flat.size + 1e-6) * 200)  # normalisé

    # ── Loi 1/f (Bruit rose naturel) ───────────────────────────────────────
    # Calcule le spectre radial moyen et ajuste α = -slope en log-log
    max_r = min(cx, cy)
    radial_profile = []
    for r in range(1, max_r):
        ring_mask = (
            (np.abs(np.sqrt((yy - cx) ** 2 + (xx - cy) ** 2) - r) < 0.5)
        )
        vals = magnitude[ring_mask]
        if vals.size > 0:
            radial_profile.append(float(np.mean(vals)))

    if len(radial_profile) > 10:
        freqs = np.arange(1, len(radial_profile) + 1, dtype=float)
        log_f = np.log(freqs)
        log_p = np.log(np.array(radial_profile) + 1e-8)
        # Régression linéaire en log-log
        coeffs = np.polyfit(log_f, log_p, 1)
        alpha = float(-coeffs[0])  # exposant spectral (naturel ≈ 2.0)
        # Déviation par rapport au naturel
        alpha_deviation = abs(alpha - 2.0) / 2.0
    else:
        alpha = 0.0
        alpha_deviation = 0.5

    # ── Score FFT ──────────────────────────────────────────────────────────
    score = (
        peak_ratio * 0.50
        + _clamp(alpha_deviation) * 0.35
        + _clamp((p99 - median_mag) / (std_mag + 1e-6) / 5.0) * 0.15
    )
    score = _clamp(score)

    # Seuil dynamique : basé sur la force des pics détectés
    flag_threshold = 0.40 if n_peaks > 10 else 0.55

    return _result(
        score,
        flag=(score > flag_threshold or n_peaks > 20),
        n_anomalous_peaks=n_peaks,
        peak_ratio=round(peak_ratio, 4),
        spectral_alpha=round(alpha, 3),
        alpha_deviation=round(alpha_deviation, 3),
        median_magnitude=round(median_mag, 3),
        interpretation=(
            f"Exposant spectral α={alpha:.2f} (attendu ≈2.0). "
            f"{n_peaks} pic(s) périodique(s) détecté(s). "
            + ("ARTEFACTS GAN SUSPECTS." if n_peaks > 20 else "Spectre dans la normale.")
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  3. ANALYSE DISTRIBUTION DES COULEURS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_color_histogram(img: Image.Image) -> dict:
    """
    Analyse statistique des histogrammes RGB.

    Les images IA présentent souvent :
      - Des pics artificiels dans les histogrammes (saturation de certaines valeurs)
      - Une distribution trop uniforme/lisse (interpolation GAN)
      - Une saturation excessive (modèles de diffusion surentraînés)
      - Un déséquilibre inhabituel entre canaux
      - Un manque de valeurs extrêmes (0 et 255) → gamme compressée

    Métriques :
      - entropy_rgb    : entropie de chaque canal (faible = trop lisse)
      - peak_sharpness : acuité des pics (élevé = pics artificiels)
      - saturation_mean: saturation HSV moyenne
      - balance_score  : équilibre inter-canaux
    """
    arr = _to_array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    scores_per_channel = []
    entropies = {}
    peak_sharpnesses = {}

    for name, channel in [("R", r), ("G", g), ("B", b)]:
        hist, _ = np.histogram(channel, bins=256, range=(0, 1))
        hist_norm = hist / (hist.sum() + 1e-8)

        # Entropie de Shannon
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))
        entropies[name] = round(entropy, 3)

        # Acuité des pics : variance de la dérivée seconde de l'histogramme
        d2 = np.diff(hist_norm.astype(float), n=2)
        peak_sharpness = float(np.var(d2) * 1e6)
        peak_sharpnesses[name] = round(peak_sharpness, 3)

        # Un canal trop lisse → entropie anormalement haute et uniforme
        # Un canal avec pics → score élevé
        channel_score = (
            _clamp(peak_sharpness / 50.0) * 0.5
            + _clamp((8.0 - entropy) / 4.0) * 0.5  # entropie < 4 → suspect
        )
        scores_per_channel.append(channel_score)

    # ── Saturation HSV ─────────────────────────────────────────────────────
    # PIL ne supporte pas HSV nativement — calcul manuel depuis RGB
    r_ch = (arr[:, :, 0]).astype(np.float32)
    g_ch = (arr[:, :, 1]).astype(np.float32)
    b_ch = (arr[:, :, 2]).astype(np.float32)
    cmax = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    cmin = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    delta = cmax - cmin
    sat_arr = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0).astype(np.float32)
    sat_mean = float(np.mean(sat_arr))
    sat_std = float(np.std(sat_arr))

    # Saturation trop élevée ou trop uniforme → suspect
    sat_score = _clamp(sat_mean * 0.7 + (1.0 - sat_std) * 0.3)

    # ── Équilibre inter-canaux ─────────────────────────────────────────────
    means = [float(np.mean(r)), float(np.mean(g)), float(np.mean(b))]
    balance = float(np.std(means)) / (float(np.mean(means)) + 1e-8)
    # Déséquilibre très élevé → possible artefact de génération
    balance_score = _clamp(balance * 3.0)

    # ── Score final couleur ────────────────────────────────────────────────
    color_score = (
        float(np.mean(scores_per_channel)) * 0.50
        + sat_score * 0.30
        + balance_score * 0.20
    )

    # Seuil dynamique : si saturation très élevée, seuil + bas
    flag_threshold = 0.45 if sat_mean > 0.6 else 0.55

    return _result(
        _clamp(color_score),
        flag=(color_score > flag_threshold),
        entropy_rgb=entropies,
        peak_sharpness_rgb=peak_sharpnesses,
        saturation_mean=round(sat_mean, 3),
        saturation_std=round(sat_std, 3),
        channel_means={"R": round(means[0], 3), "G": round(means[1], 3), "B": round(means[2], 3)},
        interpretation=(
            "Distribution chromatique atypique. Pics artificiels ou saturation excessive."
            if color_score > flag_threshold
            else "Distribution couleur dans la plage normale."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  4. ANALYSE DU BRUIT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_noise(img: Image.Image) -> dict:
    """
    Analyse de la structure du bruit résiduel.

    Les vraies photographies contiennent du bruit de capteur (bruit thermique,
    grenaillage photonique) qui est spatialement incohérent et suit des
    distributions statistiques précises (Poisson/Gaussien).

    Les images générées par IA présentent souvent :
      - Absence de bruit haute fréquence → image trop lisse
      - Bruit trop régulier spatialement (pattern GAN)
      - Distribution du bruit non-Gaussienne
      - Autocorrélation spatiale anormale

    Méthode :
      1. Extraction du bruit résiduel via filtre passe-haut (Laplacien)
      2. Analyse statistique de la distribution
      3. Test d'autocorrélation spatiale
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # ── Extraction bruit par filtre Laplacien ──────────────────────────────
    img_lap = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
        scale=1, offset=128
    ))
    noise = np.asarray(img_lap, dtype=np.float32) - 128.0
    noise_flat = noise.flatten()

    noise_std = float(np.std(noise_flat))
    noise_mean = float(np.mean(noise_flat))

    # ── Test de Gaussianité (Excess Kurtosis) ─────────────────────────────
    if noise_std > 0:
        normalized = (noise_flat - noise_mean) / noise_std
        kurtosis = float(np.mean(normalized ** 4)) - 3.0
        # Bruit gaussien → kurtosis ≈ 0
        # Bruit IA → kurtosis anormal (très élevé ou négatif)
        kurtosis_score = _clamp(abs(kurtosis) / 10.0)
    else:
        kurtosis = 0.0
        kurtosis_score = 1.0  # Absence totale de bruit → suspect

    # ── Autocorrélation spatiale ───────────────────────────────────────────
    # Un bruit naturel a une autocorrélation ≈ 0 pour lag > 0
    h, w = noise.shape
    sub = noise[:min(h, 256), :min(w, 256)]  # sous-échantillon pour la vitesse
    autocorr_h = float(np.corrcoef(sub[:-1, :].flatten(), sub[1:, :].flatten())[0, 1])
    autocorr_v = float(np.corrcoef(sub[:, :-1].flatten(), sub[:, 1:].flatten())[0, 1])
    spatial_autocorr = float(abs(autocorr_h) + abs(autocorr_v)) / 2.0
    # Autocorrélation élevée → bruit structuré → suspect IA
    autocorr_score = _clamp(spatial_autocorr * 2.5)

    # ── Niveau de bruit global ─────────────────────────────────────────────
    # Trop peu de bruit → image trop lisse → IA probable
    # noise_std élevé → plus naturel (sauf si pattern)
    if noise_std < 2.0:
        smoothness_score = 0.8  # très lisse → suspect
    elif noise_std < 5.0:
        smoothness_score = 0.4
    else:
        smoothness_score = 0.1  # beaucoup de bruit → naturel

    # ── Score bruit ────────────────────────────────────────────────────────
    noise_score = (
        kurtosis_score * 0.35
        + autocorr_score * 0.35
        + smoothness_score * 0.30
    )

    # Seuil dynamique : si autocorrélation très forte, seuil + bas
    flag_threshold = 0.40 if spatial_autocorr > 0.3 else 0.50

    return _result(
        _clamp(noise_score),
        flag=(noise_score > flag_threshold),
        noise_std=round(noise_std, 3),
        excess_kurtosis=round(kurtosis, 3),
        spatial_autocorrelation=round(spatial_autocorr, 4),
        interpretation=(
            "Bruit résiduel artificiel ou absent. Signature d'image synthétique."
            if noise_score > flag_threshold
            else "Structure du bruit compatible avec une origine photographique."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  5. DÉTECTION DE WATERMARKS / SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

def analyze_watermark(img: Image.Image) -> dict:
    """
    Détection heuristique de watermarks et signatures de modèles génératifs.

    Approche multi-couches :
      A. Analyse des métadonnées EXIF → logiciels de génération connus
      B. Détection de motifs périodiques stéganographiques (LSB analysis)
      C. Recherche de zones à haute énergie localisée (logos, textes)
      D. Analyse des coins et bordures (zones communes pour les watermarks)

    Note : Sans OCR externe, la détection de texte est heuristique
    (zones de haute fréquence localisée + contraste anormal).
    """
    arr = _to_array(img)
    flags_found = []
    evidence = {}

    # ── A. Métadonnées EXIF ────────────────────────────────────────────────
    ai_software_keywords = [
        "stable diffusion", "midjourney", "dall-e", "dalle", "firefly",
        "imagen", "deepfake", "faceswap", "reface", "artbreeder",
        "runwayml", "runway", "pika", "sora", "emu", "adobe firefly",
        "diffusers", "controlnet", "automatic1111", "comfyui", "invoke",
        "novelai", "tensor.art", "civitai", "flux", "sdxl",
    ]
    exif_data = {}
    software_detected = None

    try:
        raw_exif = img._getexif() if hasattr(img, "_getexif") else None
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_data[tag_name] = str(value)[:100]

            # Vérifie les champs logiciels
            soft_fields = ["Software", "ProcessingSoftware", "Artist", "ImageDescription"]
            for field in soft_fields:
                if field in exif_data:
                    val_lower = exif_data[field].lower()
                    for kw in ai_software_keywords:
                        if kw in val_lower:
                            software_detected = f"{field}: {exif_data[field][:50]}"
                            flags_found.append(f"EXIF_SOFTWARE: {software_detected}")
                            break
    except Exception:
        pass

    evidence["exif_fields"] = list(exif_data.keys())
    evidence["software_detected"] = software_detected

    # ── B. Analyse LSB (Least Significant Bits) ───────────────────────────
    # Stéganographie : information cachée dans les bits de poids faible
    arr_uint8 = np.asarray(img)
    lsb = arr_uint8 & 1  # extraire le bit le moins significatif
    lsb_mean = float(np.mean(lsb))
    # Un LSB parfaitement équilibré (≈0.5) peut indiquer une stéganographie
    lsb_bias = abs(lsb_mean - 0.5) * 2  # 0 = parfait équilibre (suspect), 1 = tout 0 ou 1
    lsb_suspect = lsb_bias < 0.05  # trop équilibré
    if lsb_suspect:
        flags_found.append("LSB_STEGANOGRAPHY_PATTERN")
    evidence["lsb_balance"] = round(lsb_mean, 4)
    evidence["lsb_suspect"] = lsb_suspect

    # ── C. Zones de haute fréquence localisée ─────────────────────────────
    # Détecte des régions inhabituellement actives dans les hautes fréquences
    gray = img.convert("L")
    gray_arr = np.asarray(gray, dtype=np.float32)
    # Filtre passe-haut pour détecter les contours/textes
    hp = gray.filter(ImageFilter.FIND_EDGES)
    hp_arr = np.asarray(hp, dtype=np.float32)

    h, w = hp_arr.shape
    # Divise en 16 zones (4x4) et compare les niveaux d'énergie
    zones = []
    for i in range(4):
        for j in range(4):
            zone = hp_arr[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            zones.append(float(np.mean(zone)))

    zone_mean = float(np.mean(zones))
    zone_std = float(np.std(zones))
    # Un seul quadrant avec énergie très élevée → watermark localisé
    n_outlier_zones = sum(1 for z in zones if z > zone_mean + 2.5 * zone_std)
    if n_outlier_zones >= 1:
        flags_found.append(f"LOCALIZED_HIGH_FREQ_ZONES: {n_outlier_zones}")
    evidence["n_outlier_zones"] = n_outlier_zones
    evidence["zone_energy_std"] = round(zone_std, 3)

    # ── D. Analyse des bordures ────────────────────────────────────────────
    # Watermarks souvent placés dans les coins (10% de la bordure)
    border_w = max(1, w // 10)
    border_h = max(1, h // 10)
    border_regions = [
        hp_arr[:border_h, :],         # haut
        hp_arr[-border_h:, :],        # bas
        hp_arr[:, :border_w],         # gauche
        hp_arr[:, -border_w:],        # droite
    ]
    center_region = hp_arr[border_h:-border_h, border_w:-border_w]
    border_mean = float(np.mean([np.mean(r) for r in border_regions]))
    center_mean = float(np.mean(center_region)) if center_region.size > 0 else 0.0

    border_ratio = border_mean / (center_mean + 1e-6)
    border_suspect = border_ratio > 1.5
    if border_suspect:
        flags_found.append(f"BORDER_ACTIVITY_RATIO: {border_ratio:.2f}")
    evidence["border_vs_center_ratio"] = round(border_ratio, 3)

    # ── Score watermark ────────────────────────────────────────────────────
    n_flags = len(flags_found)
    if software_detected:
        wm_score = 0.95  # Preuve directe
    elif n_flags >= 3:
        wm_score = 0.80
    elif n_flags == 2:
        wm_score = 0.60
    elif n_flags == 1:
        wm_score = 0.35
    else:
        wm_score = 0.05

    return _result(
        wm_score,
        flag=(n_flags >= 2 or software_detected is not None),
        flags_found=flags_found,
        **evidence,
        interpretation=(
            f"Watermark/signature détecté(e) : {flags_found}"
            if flags_found
            else "Aucune signature de générateur IA identifiée."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  6. ANALYSE PHYSIQUE (Lumière & Ombres)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_lighting(img: Image.Image) -> dict:
    """
    Détection d'incohérences physiques dans la lumière et les ombres.

    Les images IA échouent souvent à respecter la physique de la lumière :
      - Direction lumière incohérente entre objets différents
      - Ombres manquantes ou incorrectes
      - Reflets non-physiques (lumière dans les yeux ne correspond pas)
      - Gradient de lumière non-uniforme (vignettage artificiel vs naturel)

    Approche heuristique (sans modèle 3D) :
      1. Analyse du gradient de luminosité global
      2. Détection de sources lumineuses multiples incohérentes
      3. Analyse de la symétrie de l'illumination
      4. Vignettage : comparaison bords/centre
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = arr.shape

    # ── 1. Gradient de luminosité global ──────────────────────────────────
    # Calcul des gradients sobel manuels
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Direction dominante de la lumière (vote de Hough simplifié)
    # Angle du gradient pondéré par magnitude
    angles = np.arctan2(gy, gx + 1e-8)
    weights = grad_mag
    
    # Histogramme des angles pondéré
    angle_hist, _ = np.histogram(
        angles.flatten(), bins=36, range=(-math.pi, math.pi),
        weights=weights.flatten()
    )
    angle_hist_norm = angle_hist / (angle_hist.sum() + 1e-8)
    dominant_fraction = float(np.max(angle_hist_norm))

    # Si la fraction dominante est faible → lumière incohérente (vient de partout)
    # Faible dominance → suspects
    dominance_score = _clamp(1.0 - dominant_fraction * 4.0)

    # ── 2. Analyse vignettage ──────────────────────────────────────────────
    # Compare l'intensité moyenne du centre vs les bords
    margin = min(h, w) // 6
    center = arr[margin:-margin, margin:-margin]
    border_vals = []
    if margin > 0:
        border_vals.extend(arr[:margin, :].flatten())
        border_vals.extend(arr[-margin:, :].flatten())
        border_vals.extend(arr[margin:-margin, :margin].flatten())
        border_vals.extend(arr[margin:-margin, -margin:].flatten())

    center_lum = float(np.mean(center)) if center.size > 0 else 0.5
    border_lum = float(np.mean(border_vals)) if border_vals else 0.5

    # Rapport centre/bord (vignettage naturel → bords plus sombres)
    vignette_ratio = center_lum / (border_lum + 1e-6)
    # Un vignettage trop prononcé ou absent → suspect
    if vignette_ratio > 2.5:
        vignette_score = 0.6  # Trop fort → artificiel
    elif vignette_ratio < 0.8:
        vignette_score = 0.3  # Bords plus clairs → insolite
    else:
        vignette_score = 0.1  # Normal

    # ── 3. Symétrie de l'illumination ─────────────────────────────────────
    # Compare les moitiés gauche/droite et haut/bas
    left_mean = float(np.mean(arr[:, :w//2]))
    right_mean = float(np.mean(arr[:, w//2:]))
    top_mean = float(np.mean(arr[:h//2, :]))
    bot_mean = float(np.mean(arr[h//2:, :]))

    lr_diff = abs(left_mean - right_mean)
    tb_diff = abs(top_mean - bot_mean)

    # Différence modérée → naturel. Parfaite symétrie → généré. Extrême → suspect.
    symmetry_score = _clamp(1.0 - (lr_diff + tb_diff) * 3.0)
    # (faible différence → score élevé car trop parfait)

    # ── Score lumière ──────────────────────────────────────────────────────
    light_score = (
        dominance_score * 0.40
        + vignette_score * 0.30
        + symmetry_score * 0.30
    )

    return _result(
        _clamp(light_score),
        flag=(light_score > 0.50),
        dominant_light_fraction=round(dominant_fraction, 4),
        center_luminosity=round(center_lum, 3),
        border_luminosity=round(border_lum, 3),
        vignette_ratio=round(vignette_ratio, 3),
        lr_luminosity_diff=round(lr_diff, 3),
        tb_luminosity_diff=round(tb_diff, 3),
        interpretation=(
            "Incohérences physiques dans la distribution lumineuse détectées."
            if light_score > 0.50
            else "Distribution lumineuse globalement cohérente."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  7. ANALYSE STRUCTURELLE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_structural(img: Image.Image) -> dict:
    """
    Détection d'anomalies structurelles caractéristiques des images IA.

    Axes d'analyse :
      1. Symétrie globale trop parfaite (signe de génération conditionnée)
      2. Répétition de textures (tiling GAN)
      3. Analyse de la netteté locale (zones floues incohérentes)
      4. Régularité statistique sur des sous-patchs (trop uniforme → IA)
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = arr.shape

    # ── 1. Symétrie horizontale (flip) ─────────────────────────────────────
    flipped_h = np.fliplr(arr)
    sym_h_diff = float(np.mean(np.abs(arr - flipped_h)))
    # Très faible différence → trop symétrique → suspect
    sym_h_score = _clamp(1.0 - sym_h_diff * 10.0)

    # ── 2. Répétition de textures (auto-similarité) ────────────────────────
    # Divise en patchs de 32×32 et compare les statistiques
    patch_size = 32
    patch_stats = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = arr[i:i+patch_size, j:j+patch_size]
            patch_stats.append((float(np.mean(patch)), float(np.std(patch))))

    if len(patch_stats) > 4:
        means = [s[0] for s in patch_stats]
        stds = [s[1] for s in patch_stats]
        # Variance faible des statistiques → textures trop répétitives
        mean_var = float(np.var(means))
        std_var = float(np.var(stds))
        texture_uniformity = _clamp(1.0 - (mean_var + std_var) * 15.0)
    else:
        texture_uniformity = 0.5

    # ── 3. Carte de netteté locale ─────────────────────────────────────────
    # Laplacien local pour estimer la netteté par zone
    lap = gray.filter(ImageFilter.FIND_EDGES)
    lap_arr = np.asarray(lap, dtype=np.float32)

    # Sharpness par quadrant
    quadrants = [
        lap_arr[:h//2, :w//2], lap_arr[:h//2, w//2:],
        lap_arr[h//2:, :w//2], lap_arr[h//2:, w//2:],
    ]
    q_means = [float(np.mean(q)) for q in quadrants]
    sharpness_variation = float(np.std(q_means)) / (float(np.mean(q_means)) + 1e-6)

    # Très faible variation → toute l'image est uniformément nette → suspect
    # Trop forte variation → bokeh artificiel → suspect
    if sharpness_variation < 0.1:
        sharpness_score = 0.6  # Uniformément net → peut être IA
    elif sharpness_variation > 1.5:
        sharpness_score = 0.4  # Très inégal → peut être bokeh IA
    else:
        sharpness_score = 0.1  # Plage normale

    # ── 4. Régularité statistique globale (complexité visuelle) ───────────
    # L'entropie par blocs : images trop régulières → complexité faible
    block_size = 16
    block_entropies = []
    for i in range(0, h - block_size, block_size * 2):
        for j in range(0, w - block_size, block_size * 2):
            block = arr[i:i+block_size, j:j+block_size]
            h_b, _ = np.histogram(block, bins=16, range=(0, 1))
            h_n = h_b / (h_b.sum() + 1e-8)
            ent = float(-np.sum(h_n * np.log2(h_n + 1e-10)))
            block_entropies.append(ent)

    if block_entropies:
        mean_block_ent = float(np.mean(block_entropies))
        std_block_ent = float(np.std(block_entropies))
        # Entropie par bloc très uniforme → complexité IA artificielle
        regularity_score = _clamp(1.0 - std_block_ent / (mean_block_ent + 1e-6))
    else:
        regularity_score = 0.5

    # ── Score structural ────────────────────────────────────────────────────
    struct_score = (
        sym_h_score * 0.25
        + texture_uniformity * 0.30
        + sharpness_score * 0.20
        + regularity_score * 0.25
    )

    # Seuil dynamique : si symétrie très forte, seuil + bas
    flag_threshold = 0.40 if sym_h_diff < 0.02 else 0.50

    return _result(
        _clamp(struct_score),
        flag=(struct_score > flag_threshold),
        horizontal_symmetry=round(sym_h_diff, 4),
        texture_uniformity=round(texture_uniformity, 4),
        sharpness_variation=round(sharpness_variation, 4),
        block_entropy_std=round(std_block_ent if block_entropies else 0, 4),
        interpretation=(
            "Anomalies structurelles : symétrie excessive, textures répétitives ou netteté artificielle."
            if struct_score > flag_threshold
            else "Structure visuelle dans les limites normales."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  8. DÉTECTION DE RÉPÉTITIONS (IA typique)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_repetition(img: Image.Image) -> dict:
    """
    Détection de motifs répétitifs typiques des images générées par IA.

    Les GAN et modèles de diffusion produisent souvent :
      - Des textures clonées (copy-paste de features)
      - Des motifs répétitifs à intervalles réguliers
      - Des micro-structures dupliquées invisibles à l'oeil nu

    Méthode :
      1. Découpage en blocs et calcul de descripteurs (mean, std, gradient)
      2. Comparaison inter-blocs par similarité (cosine similarity)
      3. Détection d'autocorrélation 2D (motifs périodiques spatiaux)
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = arr.shape

    # ── 1. Comparaison de blocs (block similarity) ─────────────────────────
    block_size = 48
    stride = 24  # overlap pour capturer plus de paires
    descriptors = []
    positions = []

    for i in range(0, h - block_size, stride):
        for j in range(0, w - block_size, stride):
            block = arr[i:i + block_size, j:j + block_size]
            # Descripteur : mean, std, gradient moyen, énergie haute fréquence
            gx = float(np.mean(np.abs(np.diff(block, axis=1))))
            gy = float(np.mean(np.abs(np.diff(block, axis=0))))
            desc = np.array([
                float(np.mean(block)),
                float(np.std(block)),
                gx, gy,
                float(np.mean(block ** 2)),  # énergie
            ])
            descriptors.append(desc)
            positions.append((i, j))

    n_blocks = len(descriptors)
    if n_blocks < 4:
        return _result(0.0, flag=False, n_blocks=n_blocks,
                       interpretation="Image trop petite pour l'analyse de répétition.")

    descs = np.array(descriptors)
    # Normalisation pour cosine similarity
    norms = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
    descs_norm = descs / norms

    # Matrice de similarité (n_blocks × n_blocks)
    # On limite à 400 blocs max pour la performance
    max_blocks = min(n_blocks, 400)
    descs_sub = descs_norm[:max_blocks]
    sim_matrix = descs_sub @ descs_sub.T

    # Masquer la diagonale et les voisins proches (±2 positions)
    for i in range(max_blocks):
        for j in range(max(0, i - 2), min(max_blocks, i + 3)):
            sim_matrix[i, j] = 0.0

    # Paires très similaires (seuil > 0.98 = quasi-identiques)
    n_clone_pairs = int(np.sum(sim_matrix > 0.98)) // 2
    n_similar_pairs = int(np.sum(sim_matrix > 0.95)) // 2

    clone_ratio = _clamp(n_clone_pairs / (max_blocks + 1) * 5.0)
    similar_ratio = _clamp(n_similar_pairs / (max_blocks + 1) * 2.0)

    # ── 2. Autocorrélation 2D (motifs périodiques spatiaux) ────────────────
    # Utilise un patch central pour détecter la périodicité
    patch_h = min(h, 256)
    patch_w = min(w, 256)
    center_patch = arr[
        (h - patch_h) // 2:(h + patch_h) // 2,
        (w - patch_w) // 2:(w + patch_w) // 2
    ]

    # Autocorrélation via FFT
    f = np.fft.fft2(center_patch - np.mean(center_patch))
    autocorr = np.fft.ifft2(f * np.conj(f)).real
    autocorr /= (autocorr[0, 0] + 1e-8)  # normaliser le pic central à 1

    # Exclure le pic central (zone 5%)
    ac_h, ac_w = autocorr.shape
    mask = np.ones_like(autocorr, dtype=bool)
    r_excl = int(min(ac_h, ac_w) * 0.05)
    yy, xx = np.ogrid[:ac_h, :ac_w]
    mask[yy ** 2 + xx ** 2 < r_excl ** 2] = False

    # Pics de périodicité : valeurs élevées hors du centre
    ac_masked = autocorr[mask]
    ac_max = float(np.max(ac_masked)) if ac_masked.size > 0 else 0.0
    periodicity_score = _clamp(ac_max * 3.0)  # fort → motif périodique

    # ── Score répétition ───────────────────────────────────────────────────
    rep_score = (
        clone_ratio * 0.40
        + similar_ratio * 0.30
        + periodicity_score * 0.30
    )

    flag_threshold = 0.40 if n_clone_pairs > 5 else 0.55

    return _result(
        _clamp(rep_score),
        flag=(rep_score > flag_threshold),
        n_clone_pairs=n_clone_pairs,
        n_similar_pairs=n_similar_pairs,
        periodicity_peak=round(ac_max, 4),
        n_blocks_analyzed=max_blocks,
        interpretation=(
            f"{n_clone_pairs} paire(s) de blocs clonés, {n_similar_pairs} paire(s) similaires. "
            + ("Motifs répétitifs détectés — signature IA probable."
               if rep_score > flag_threshold
               else "Pas de répétition anormale détectée.")
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  9. ANALYSE MULTI-ÉCHELLE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_multiscale(img: Image.Image) -> dict:
    """
    Analyse multi-échelle : exécute des sous-analyses à trois niveaux
    de résolution et mesure la cohérence inter-échelles.

    Les images réelles montrent des caractéristiques cohérentes à toutes
    les échelles. Les images IA présentent souvent des incohérences
    entre l'image pleine résolution et ses sous-régions.

    Niveaux :
      1. Image complète (score de référence)
      2. 4 quadrants analysés séparément
      3. Image réduite à 50%
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = arr.shape

    def _quick_noise_score(patch: np.ndarray) -> float:
        """Score de bruit rapide via Laplacien."""
        ph, pw = patch.shape
        if ph < 4 or pw < 4:
            return 0.5
        # Convolution manuelle simplifiée (centre uniquement)
        sub = patch[1:-1, 1:-1]
        neighbors = (
            patch[:-2, :-2] + patch[:-2, 1:-1] + patch[:-2, 2:] +
            patch[1:-1, :-2]                   + patch[1:-1, 2:] +
            patch[2:, :-2]  + patch[2:, 1:-1]  + patch[2:, 2:]
        )
        noise = 8 * sub - neighbors
        noise_std = float(np.std(noise))
        if noise_std < 0.02:
            return 0.8  # très lisse
        elif noise_std < 0.05:
            return 0.4
        else:
            return 0.1

    def _quick_fft_score(patch: np.ndarray) -> float:
        """Score FFT rapide."""
        ph, pw = patch.shape
        if ph < 16 or pw < 16:
            return 0.5
        f = np.fft.fft2(patch)
        mag = np.abs(np.fft.fftshift(f))
        log_mag = np.log1p(mag)
        flat = log_mag.flatten()
        med = float(np.median(flat))
        std = float(np.std(flat))
        threshold = med + 3.5 * std
        n_peaks = int(np.sum(flat > threshold))
        return _clamp(n_peaks / (flat.size + 1e-6) * 200)

    # ── 1. Score de référence (image complète) ────────────────────────────
    full_noise = _quick_noise_score(arr)
    full_fft = _quick_fft_score(arr)
    full_score = (full_noise + full_fft) / 2.0

    # ── 2. Scores par quadrant ────────────────────────────────────────────
    quadrant_scores = []
    quadrants = [
        arr[:h // 2, :w // 2],
        arr[:h // 2, w // 2:],
        arr[h // 2:, :w // 2],
        arr[h // 2:, w // 2:],
    ]
    for q in quadrants:
        qs = (_quick_noise_score(q) + _quick_fft_score(q)) / 2.0
        quadrant_scores.append(qs)

    quadrant_mean = float(np.mean(quadrant_scores))
    quadrant_std = float(np.std(quadrant_scores))

    # ── 3. Score image réduite (50%) ──────────────────────────────────────
    small = img.resize((max(w // 2, 16), max(h // 2, 16)), Image.LANCZOS).convert("L")
    small_arr = np.asarray(small, dtype=np.float32) / 255.0
    small_noise = _quick_noise_score(small_arr)
    small_fft = _quick_fft_score(small_arr)
    small_score = (small_noise + small_fft) / 2.0

    # ── Mesure d'incohérence inter-échelles ───────────────────────────────
    # Grande variance entre les échelles = incohérence suspecte
    all_scores = [full_score] + quadrant_scores + [small_score]
    cross_scale_std = float(np.std(all_scores))
    cross_scale_mean = float(np.mean(all_scores))

    # Score multiscale : forte moyenne + faible variance → IA cohérente
    # Forte moyenne + forte variance → incohérence → IA probable aussi
    ms_score = cross_scale_mean * 0.60 + _clamp(cross_scale_std * 3.0) * 0.40

    flag_threshold = 0.45

    return _result(
        _clamp(ms_score),
        flag=(ms_score > flag_threshold),
        full_scale_score=round(full_score, 4),
        quadrant_scores=[round(s, 4) for s in quadrant_scores],
        quadrant_std=round(quadrant_std, 4),
        half_scale_score=round(small_score, 4),
        cross_scale_std=round(cross_scale_std, 4),
        interpretation=(
            f"Incohérence multi-échelle détectée (σ={cross_scale_std:.3f}). "
            + ("Signature d'image synthétique à plusieurs niveaux de résolution."
               if ms_score > flag_threshold
               else "Cohérence inter-échelle normale.")
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATEUR PRINCIPAL (v3.0 — scoring intelligent)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_combined_rules(scores: dict) -> tuple[float, list[str]]:
    """
    Applique les règles combinées (bonus/malus) basées sur la convergence
    ou divergence des scores individuels.

    Returns:
        (adjustment, explanations) — float bonus/malus et liste des raisons.
    """
    adjustment = 0.0
    explanations = []

    def _s(key: str) -> float:
        return scores.get(key, 0.0) or 0.0

    # ── Bonus : convergence de signaux forts ───────────────────────────────
    if _s('fft') > 0.65 and _s('noise') > 0.65:
        adjustment += 0.10
        explanations.append("+0.10 — FFT et bruit convergent (signaux GAN)")

    if _s('ela') > 0.55 and _s('structural') > 0.55:
        adjustment += 0.08
        explanations.append("+0.08 — ELA et structure convergent")

    if _s('repetition') > 0.60 and _s('fft') > 0.50:
        adjustment += 0.08
        explanations.append("+0.08 — Répétitions + artefacts fréquentiels")

    if _s('multiscale') > 0.55 and _s('noise') > 0.50:
        adjustment += 0.06
        explanations.append("+0.06 — Incohérence multi-échelle + bruit artificiel")

    if _s('watermark') > 0.80:
        adjustment += 0.12
        explanations.append("+0.12 — Signature de générateur IA confirmée")

    # ── Malus : faux positifs probables ────────────────────────────────────
    valid_scores = [v for v in scores.values() if v is not None]
    if valid_scores:
        high_count = sum(1 for s in valid_scores if s > 0.6)
        mean_score = float(np.mean(valid_scores))

        if high_count == 1 and mean_score < 0.30:
            adjustment -= 0.10
            explanations.append("-0.10 — Signal isolé, probable faux positif")

        if high_count == 0 and mean_score < 0.20:
            adjustment -= 0.05
            explanations.append("-0.05 — Tous les signaux faibles, probable authentique")

    return adjustment, explanations


def analyze_image(file_path: str) -> dict[str, Any]:
    """
    Orchestre toutes les analyses forensiques sur une image.
    v3.0 : inclut analyze_repetition, analyze_multiscale,
    et scoring intelligent avec règles combinées.

    Args:
        file_path: Chemin absolu vers le fichier image.

    Returns:
        Dictionnaire contenant les résultats de chaque sous-analyse
        + un champ 'combined_rules' avec les ajustements appliqués.
    """
    results: dict[str, Any] = {}

    try:
        img = _load_image(file_path)
    except Exception as e:
        logger.error(f"Impossible de charger l'image : {e}")
        return {"error": {"score": None, "details": {"message": str(e)}, "flag": False}}

    # Préserver le format source pour l'ELA
    src_format = None
    try:
        src_img = Image.open(file_path)
        src_format = src_img.format
        img.format = src_format  # propager le format
    except Exception:
        pass

    # ── Informations de base ───────────────────────────────────────────────
    results["metadata"] = {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": src_format or "unknown",
        "aspect_ratio": round(img.width / img.height, 3) if img.height else 0,
    }

    # ── Analyses individuelles (chacune isolée pour ne pas bloquer les autres) ──
    analyses = [
        ("ela",          analyze_ela,               img),
        ("fft",          analyze_fft,               img),
        ("color_hist",   analyze_color_histogram,   img),
        ("noise",        analyze_noise,             img),
        ("watermark",    analyze_watermark,         img),
        ("lighting",     analyze_lighting,          img),
        ("structural",   analyze_structural,        img),
        ("repetition",   analyze_repetition,        img),
        ("multiscale",   analyze_multiscale,        img),
    ]

    for name, func, image in analyses:
        try:
            results[name] = func(image)
            logger.debug(f"[{name}] score={results[name]['score']:.3f}")
        except Exception as e:
            logger.warning(f"Analyse [{name}] échouée : {e}", exc_info=True)
            results[name] = {
                "score": None,
                "flag": False,
                "details": {"error": str(e)},
            }

    # ── Règles combinées ──────────────────────────────────────────────────
    scores = {
        key: result.get("score")
        for key, result in results.items()
        if isinstance(result, dict) and "score" in result and result["score"] is not None
    }
    adjustment, rule_explanations = _apply_combined_rules(scores)
    results["combined_rules"] = {
        "adjustment": round(adjustment, 3),
        "rules_applied": rule_explanations,
    }

    return results
