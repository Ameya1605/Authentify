"""
Stamp / Watermark Verification Module
- Identifies altered, missing, or tampered stamps and official seals
- Detects circular/elliptical seal patterns
- Analyzes color distribution in stamp regions
- Validates stamp placement and consistency
"""

import cv2
import numpy as np
from typing import Dict, List, Any


def detect_circular_stamps(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect circular/elliptical seals using Hough Circle Transform
    and contour analysis.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We WANT to capture the main watermark, so we do NOT use the harsh threshold.
    # Instead, we just blur the grayscale image.
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using HoughCircles.
    # To prevent "too many circles" from intersecting repeating background patterns,
    # we enforce a massive minDist (200 pixels) so only 1 circle can exist per area
    # and we capture the large central watermark.
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=200, param1=60, param2=40,
        minRadius=40, maxRadius=500
    )
    
    detected_stamps = []
    if circles is not None:
        circles = circles[0]
        # Absolutely prevent circle spam by taking ONLY the top 3 strongest circles globally
        for circle in circles[:3]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            detected_stamps.append({
                "type": "circular",
                "center_x": x,
                "center_y": y,
                "radius": r,
                "bbox": {
                    "x": max(0, x - r),
                    "y": max(0, y - r),
                    "width": r * 2,
                    "height": r * 2,
                },
            })
    
    return detected_stamps


def detect_colored_regions(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect colored stamp/seal regions (typically red, blue, or purple).
    Stamps often have distinct colors different from the document text.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # We want ONLY distinct official stamp inks: Red, Blue, Purple, Green
    # We require decent saturation (S > 50) so we don't catch slightly tinted white paper or noisy backgrounds.
    color_ranges = [
        # Stamp Red / Pinkish Red
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
        (np.array([160, 50, 50]), np.array([180, 255, 255])),
        # Stamp Blue / Indigo
        (np.array([90, 50, 50]), np.array([130, 255, 255])),
        # Stamp Purple
        (np.array([130, 40, 50]), np.array([160, 255, 255])),
        # Stamp Green
        (np.array([35, 50, 50]), np.array([85, 255, 255])),
    ]
    
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        color_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)
        
    # Morphological clean up to cluster stamp ink components together firmly
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colored_regions = []
    
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Reject tiny specks, AND reject massive blocks like portrait photos (e.g. > 5% of document)
        if 800 < area < (img_area * 0.05):
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            
            # Official seals/logos are highly compact (squares, circles).
            # A line of colored text will have a wide aspect ratio. We restrict to [0.5, 2.0]
            if 0.5 < aspect < 2.0:
                # Reject solid saturated background artifacts (high fill ratio)
                fill_ratio = area / float(w * h)
                if fill_ratio > 0.85:
                    continue
                    
                # Ensure it's somewhat compact/symmetric like a logo, not a random jagged shape
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                circle_area = np.pi * (radius ** 2)
                compactness = area / max(circle_area, 1.0)
                
                # A square has compactness ~0.63, a circle is 1.0. 
                # Text/Jagged noise will have very low compactness.
                if compactness > 0.35:
                    circularity = 4 * np.pi * area / max(cv2.arcLength(cnt, True) ** 2, 1)
                    colored_regions.append({
                        "color": "stamp_ink",
                        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "area": int(area),
                        "circularity": round(float(circularity), 3),
                    })
    
    return colored_regions


def analyze_stamp_integrity(img: np.ndarray, stamp_region: Dict) -> Dict[str, Any]:
    """
    Analyze a detected stamp region for signs of tampering:
    - Edge consistency
    - Color uniformity
    - Noise patterns
    """
    bbox = stamp_region["bbox"]
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    
    # Extract stamp ROI
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return {"integrity_score": 0.5, "issue": "Empty region"}
    
    # Edge analysis
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.mean(edges) / 255.0
    
    # Color uniformity in HSV
    if len(roi.shape) == 3:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv_roi[:, :, 0])
        s_std = np.std(hsv_roi[:, :, 1])
        v_std = np.std(hsv_roi[:, :, 2])
        color_uniformity = 1.0 - min(1.0, (h_std + s_std + v_std) / 300.0)
    else:
        color_uniformity = 1.0 - min(1.0, np.std(gray_roi) / 128.0)
    
    # Noise analysis using Laplacian
    laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
    noise_level = float(np.std(laplacian))
    noise_score = 1.0 - min(1.0, noise_level / 100.0)
    
    integrity = (edge_density * 0.3 + color_uniformity * 0.4 + noise_score * 0.3)
    
    return {
        "integrity_score": round(float(integrity), 3),
        "edge_density": round(float(edge_density), 4),
        "color_uniformity": round(float(color_uniformity), 4),
        "noise_score": round(float(noise_score), 4),
    }


def detect_watermark(img: np.ndarray) -> Dict[str, Any]:
    """
    Detect watermark presence using frequency domain analysis.
    Watermarks typically show up as specific patterns in the FFT.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # DFT analysis
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    
    # Normalize magnitude spectrum
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Analyze central region (watermarks create peaks)
    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    central = mag_norm[cy-30:cy+30, cx-30:cx+30]
    
    # Check for periodic patterns (watermark indicators)
    central_mean = float(np.mean(central))
    overall_mean = float(np.mean(mag_norm))
    
    # High central-to-overall ratio suggests structured patterns
    ratio = central_mean / max(overall_mean, 1.0)
    watermark_detected = ratio > 2.5
    
    # Analyze for semi-transparent overlays
    # Watermarks often reduce local contrast
    local_contrasts = []
    block_size = 32
    for r in range(0, h - block_size, block_size):
        for c in range(0, w - block_size, block_size):
            block = gray[r:r+block_size, c:c+block_size]
            local_contrasts.append(float(np.std(block)))
    
    contrast_variance = float(np.std(local_contrasts)) if local_contrasts else 0
    
    return {
        "watermark_detected": watermark_detected,
        "frequency_ratio": round(ratio, 3),
        "contrast_variance": round(contrast_variance, 3),
        "spectrum_magnitude": mag_norm,
    }


def generate_stamp_heatmap(img: np.ndarray, stamps: List[Dict], colored_regions: List[Dict]) -> np.ndarray:
    """Generate heatmap highlighting stamp/seal regions cleanly along the ink."""
    if len(img.shape) == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()
        
    overlay = base.copy()
    h, w = base.shape[:2]
    
    # We want to find ink in these regions. Since stamps can be red/blue/purple,
    # we'll look for pixels that differ from the white background
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    _, ink_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Highlight circular stamps
    for stamp in stamps:
        cx, cy, r = stamp["center_x"], stamp["center_y"], stamp["radius"]
        integrity = stamp.get("integrity", {}).get("integrity_score", 0.5)
        color = (255, 0, 255) if integrity > 0.6 else (0, 165, 255) # Magenta for good, Orange for bad
        
        # Create a circular mask for this stamp
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)
        
        # Intersection of ink and circle
        stamp_ink = cv2.bitwise_and(ink_mask, ink_mask, mask=circle_mask)
        
        # Dilate slightly for visibility
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_ink = cv2.dilate(stamp_ink, kernel)
        
        # Apply color overlay where ink is
        mask_idx = dilated_ink > 0
        overlay[mask_idx] = cv2.addWeighted(base[mask_idx], 0.3, np.full_like(base[mask_idx], color), 0.7, 0)
        
        # Draw clean border
        cv2.circle(overlay, (cx, cy), r, color, 2)
    
    # 2. Highlight colored regions
    for region in colored_regions:
        bbox = region["bbox"]
        x, y, rw, rh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        integrity = region.get("integrity", {}).get("integrity_score", 0.5)
        color = (255, 0, 255) if integrity > 0.6 else (0, 0, 255) # Magenta for good, Red for bad
        
        # Get ROI ink
        roi_ink = ink_mask[y:y+rh, x:x+rw]
        
        # Apply color overlay on the ink only
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_ink = cv2.dilate(roi_ink, kernel)
        
        mask_idx = dilated_ink > 0
        sub_rect = overlay[y:y+rh, x:x+rw]
        
        # Only overwrite if we haven't already highlighted this as a circular stamp
        # (Very simple approach just blending over)
        blended = cv2.addWeighted(sub_rect, 0.4, np.full_like(sub_rect, color), 0.6, 0)
        sub_rect[mask_idx] = blended[mask_idx]
        
        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x+rw, y+rh), color, 1)
        
    return overlay


def run_stamp_analysis(img_normalized: np.ndarray, img_gray: np.ndarray) -> Dict[str, Any]:
    """Full stamp/watermark verification pipeline."""
    
    # Detect circular stamps
    circular_stamps = detect_circular_stamps(img_gray)
    
    # Detect colored regions
    colored_regions = detect_colored_regions(img_normalized)
    
    # Analyze integrity of each detected region
    integrity_scores = []
    for stamp in circular_stamps:
        integrity = analyze_stamp_integrity(img_normalized, stamp)
        stamp["integrity"] = integrity
        integrity_scores.append(integrity["integrity_score"])
    
    for region in colored_regions:
        integrity = analyze_stamp_integrity(img_normalized, region)
        region["integrity"] = integrity
        integrity_scores.append(integrity["integrity_score"])
    
    # Detect watermark
    watermark = detect_watermark(img_gray)
    
    # Generate heatmap
    heatmap = generate_stamp_heatmap(img_normalized, circular_stamps, colored_regions)
    
    # Compute overall score
    total_detections = len(circular_stamps) + len(colored_regions)
    
    if total_detections == 0:
        stamp_score = 0.3  # No stamps found is suspicious
        presence_comment = "No stamps or seals detected"
    else:
        avg_integrity = np.mean(integrity_scores) if integrity_scores else 0.5
        stamp_score = avg_integrity * 0.7 + 0.3  # Base 0.3 for stamp presence
        presence_comment = f"{total_detections} stamp/seal region(s) detected"
    
    # Watermark bonus
    if watermark["watermark_detected"]:
        stamp_score = min(1.0, stamp_score + 0.1)
    
    return {
        "module": "Stamp/Watermark Verification",
        "score": round(float(stamp_score), 3),
        "circular_stamps": circular_stamps,
        "colored_regions": [
            {k: v for k, v in r.items() if k != "integrity"}
            | {"integrity_score": r.get("integrity", {}).get("integrity_score", 0)}
            for r in colored_regions[:10]
        ],
        "watermark": {
            "detected": watermark["watermark_detected"],
            "frequency_ratio": watermark["frequency_ratio"],
        },
        "heatmap": heatmap,
        "details": {
            "circular_stamps_count": len(circular_stamps),
            "colored_regions_count": len(colored_regions),
            "watermark_present": watermark["watermark_detected"],
            "avg_integrity": round(float(np.mean(integrity_scores)), 3) if integrity_scores else 0,
        },
        "findings": _generate_stamp_findings(
            circular_stamps, colored_regions, watermark, integrity_scores
        ),
    }


def _generate_stamp_findings(
    stamps: List, colored: List, watermark: Dict, scores: List
) -> List[str]:
    """Generate human-readable findings."""
    findings = []
    
    total = len(stamps) + len(colored)
    if total == 0:
        findings.append("🔴 No official stamps or seals detected - document may lack institutional markings")
    elif total == 1:
        findings.append("🟡 Only 1 stamp/seal region found - typically certificates have multiple")
    else:
        findings.append(f"✅ {total} stamp/seal regions detected")
    
    if scores:
        avg = np.mean(scores)
        if avg < 0.4:
            findings.append("🔴 Stamp integrity is very low - possible manipulation detected")
        elif avg < 0.6:
            findings.append("🟡 Stamp integrity is moderate - some regions show irregularities")
        else:
            findings.append("✅ Stamp integrity appears consistent")
    
    if watermark["watermark_detected"]:
        findings.append("✅ Watermark pattern detected in frequency domain")
    else:
        findings.append("🟡 No clear watermark pattern detected")
    
    return findings
