"""
Signature Verification Module
- Compares signatures with reference patterns
- Analyzes shape, stroke pattern, position
- Detects forged or copy-pasted signatures
- Generates signature region heatmap
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional


def detect_signature_regions(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect potential signature regions in the document.
    Signatures typically appear in the lower portion and have
    specific stroke characteristics.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # We must scan the ENTIRE document for signatures, not just the bottom half,
    # as human written signatures can appear anywhere!
    # No longer cropping `gray[h // 2:, :]`
    
    # Use global Otsu thresholding (better for pen pressure variation)
    blurred_for_thresh = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_for_thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to connect signature strokes in any direction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(dilated, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    signature_candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = cw / max(ch, 1)
        
        # Signature heuristics: highly permissive to capture any pen strokes
        if (0.5 < aspect < 15 and
            area > 300 and
            cw > 30 and ch > 10 and
            cw < w * 0.9):
            
            # Use actual y coordinate directly since we scan the full image
            actual_y = y
            
            roi = gray[actual_y:actual_y+ch, x:x+cw]
            if roi.size == 0:
                continue
                
            # Compute Otsu to get stroke pixels
            _, sig_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Stroke fill ratio (ink vs bounding box)
            fill_ratio = cv2.countNonZero(sig_thresh) / float(roi.size)
            
            # Text paragraphs have very high fill ratios compared to sprawling signatures
            # 0.70 is our high cutoff (solid black blocks)
            if fill_ratio > 0.70:
                continue
            
            stroke_density = fill_ratio
            
            signature_candidates.append({
                "bbox": {
                    "x": int(x), "y": int(actual_y),
                    "width": int(cw), "height": int(ch),
                },
                "area": int(area),
                "aspect_ratio": round(float(aspect), 2),
                "stroke_density": round(stroke_density, 4),
                "confidence": round(min(1.0, stroke_density * 3 + 0.3), 3),
            })
    
    # Sort by confidence and area
    signature_candidates.sort(key=lambda s: s["confidence"] * s["area"], reverse=True)
    return signature_candidates[:5]  # Return top 5 candidates


def analyze_signature_strokes(img: np.ndarray, sig_bbox: Dict) -> Dict[str, Any]:
    """
    Analyze signature stroke characteristics:
    - Stroke width distribution
    - Continuity
    - Pressure variation
    """
    x, y, w, h = sig_bbox["x"], sig_bbox["y"], sig_bbox["width"], sig_bbox["height"]
    
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y+h, x:x+w]
    
    if roi.size == 0:
        return {"stroke_score": 0.5, "analysis": "Empty ROI"}
    
    # Binarize
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Distance transform for stroke width
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Stroke width statistics
    stroke_pixels = dist[dist > 0]
    if len(stroke_pixels) == 0:
        return {"stroke_score": 0.5, "analysis": "No strokes detected"}
    
    mean_width = float(np.mean(stroke_pixels))
    std_width = float(np.std(stroke_pixels))
    max_width = float(np.max(stroke_pixels))
    
    # Width uniformity (printed signatures have very uniform width)
    width_cv = std_width / max(mean_width, 0.01)  # Coefficient of variation
    
    # Natural signatures have some variation but not too much
    if width_cv < 0.15:
        naturalness = 0.4  # Too uniform = possibly printed/stamped
        naturalness_note = "Very uniform strokes - possibly machine-generated"
    elif width_cv < 0.5:
        naturalness = 0.9  # Good variation = likely handwritten
        naturalness_note = "Natural stroke variation detected"
    else:
        naturalness = 0.5  # Too much variation = possibly assembled
        naturalness_note = "Excessive stroke variation"
    
    # Continuity analysis using connected components
    n_components, labels = cv2.connectedComponents(binary)
    
    # Natural signatures typically have fewer connected components
    if n_components < 5:
        continuity = 0.9
    elif n_components < 15:
        continuity = 0.7
    else:
        continuity = 0.4
    
    stroke_score = naturalness * 0.5 + continuity * 0.5
    
    return {
        "stroke_score": round(float(stroke_score), 3),
        "mean_width": round(mean_width, 2),
        "std_width": round(std_width, 2),
        "max_width": round(max_width, 2),
        "width_cv": round(float(width_cv), 3),
        "connected_components": int(n_components),
        "naturalness": round(float(naturalness), 3),
        "naturalness_note": naturalness_note,
        "continuity": round(float(continuity), 3),
    }


def check_signature_placement(img_shape: tuple, sig_bbox: Dict) -> Dict[str, Any]:
    """
    Verify signature placement is reasonable.
    Checks position relative to document layout.
    """
    h, w = img_shape[:2]
    sx, sy = sig_bbox["x"], sig_bbox["y"]
    sw, sh = sig_bbox["width"], sig_bbox["height"]
    
    # Normalized position
    rel_x = sx / w
    rel_y = sy / h
    
    # Signatures typically in lower 40% and not at extreme edges
    position_natural = True
    position_notes = []
    
    if rel_y < 0.4:
        position_natural = False
        position_notes.append("Signature unusually high on document")
    
    if rel_x < 0.05 or (sx + sw) / w > 0.95:
        position_natural = False
        position_notes.append("Signature at extreme edge - possibly overlaid")
    
    # Size check
    size_ratio = (sw * sh) / (w * h)
    if size_ratio > 0.15:
        position_natural = False
        position_notes.append("Signature region unusually large")
    elif size_ratio < 0.002:
        position_natural = False
        position_notes.append("Signature region unusually small")
    
    if not position_notes:
        position_notes.append("Signature placement appears normal")
    
    placement_score = 0.9 if position_natural else 0.4
    
    return {
        "placement_score": round(float(placement_score), 3),
        "relative_position": {"x": round(rel_x, 3), "y": round(rel_y, 3)},
        "size_ratio": round(float(size_ratio), 5),
        "position_natural": position_natural,
        "notes": position_notes,
    }


def detect_copy_paste_signature(img: np.ndarray, sig_bbox: Dict) -> Dict[str, Any]:
    """
    Check if signature might be copy-pasted by analyzing:
    - Edge artifacts around the signature boundary
    - Background consistency
    - Compression artifacts
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = sig_bbox["x"], sig_bbox["y"], sig_bbox["width"], sig_bbox["height"]
    
    # Get a slightly larger region around the signature
    margin = 15
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray.shape[1], x + w + margin)
    y2 = min(gray.shape[0], y + h + margin)
    
    extended_roi = gray[y1:y2, x1:x2]
    if extended_roi.size == 0:
        return {"paste_score": 0.5}
    
    # Analyze boundary region (the margin area)
    # If pasted, boundary often shows artifacts
    laplacian = cv2.Laplacian(extended_roi, cv2.CV_64F)
    
    # Inner vs outer Laplacian comparison
    inner_lap = laplacian[margin:margin+h, margin:margin+w]
    
    # Create border mask
    border_mask = np.ones_like(laplacian, dtype=bool)
    border_mask[margin:margin+h, margin:margin+w] = False
    outer_lap = laplacian[border_mask]
    
    if len(outer_lap) == 0 or inner_lap.size == 0:
        return {"paste_score": 0.5}
    
    inner_std = float(np.std(inner_lap))
    outer_std = float(np.std(outer_lap))
    
    # Sharp transition at boundary suggests copy-paste
    boundary_ratio = abs(inner_std - outer_std) / max(outer_std, 1.0)
    
    if boundary_ratio > 3.0:
        paste_likelihood = 0.8
        note = "Sharp boundary discontinuity detected - possible paste artifact"
    elif boundary_ratio > 1.5:
        paste_likelihood = 0.5
        note = "Moderate boundary difference - inconclusive"
    else:
        paste_likelihood = 0.15
        note = "Boundary appears natural"
    
    return {
        "paste_likelihood": round(float(paste_likelihood), 3),
        "boundary_ratio": round(float(boundary_ratio), 3),
        "inner_edge_std": round(inner_std, 2),
        "outer_edge_std": round(outer_std, 2),
        "note": note,
    }


def generate_signature_heatmap(img: np.ndarray, signatures: List[Dict]) -> np.ndarray:
    """Generate heatmap highlighting ONLY the signature strokes cleanly."""
    if len(img.shape) == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()
    
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    overlay = base.copy()
    
    for sig in signatures:
        bbox = sig["bbox"]
        x, y, sw, sh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        roi_gray = gray[y:y+sh, x:x+sw]
        if roi_gray.size == 0:
            continue
            
        # Get signature strokes
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Determine color based on confidence/score
        conf = sig.get("combined_score", sig.get("confidence", 0.5))
        if conf > 0.6:
            color = (0, 200, 0) # Green (authentic)
        elif conf > 0.4:
            color = (0, 165, 255) # Orange (warning)
        else:
            color = (0, 0, 255) # Red (forged/low confidence)
            
        # Create a colored version of the stroke mask
        colored_strokes = np.zeros((sh, sw, 3), dtype=np.uint8)
        colored_strokes[binary > 0] = color
        
        # Dilate strokes slightly for a nice glow effect without blocking content
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_strokes = cv2.dilate(colored_strokes, kernel, iterations=1)
        
        # Add to main image
        sub_rect = overlay[y:y+sh, x:x+sw]
        mask = np.any(dilated_strokes > 0, axis=-1)
        
        # Blend the glow onto the original strokes
        blended = cv2.addWeighted(sub_rect, 0.3, dilated_strokes, 0.7, 0)
        sub_rect[mask] = blended[mask]
        
        # Draw a clean outline box to show the bounds
        cv2.rectangle(overlay, (x, y), (x+sw, y+sh), color, 1)
        
    return overlay


def run_signature_analysis(img_normalized: np.ndarray, img_gray: np.ndarray) -> Dict[str, Any]:
    """Full signature verification pipeline."""
    
    # Detect signature regions
    signatures = detect_signature_regions(img_gray)
    
    # Analyze each signature
    analysis_results = []
    for sig in signatures:
        stroke_analysis = analyze_signature_strokes(img_gray, sig["bbox"])
        placement = check_signature_placement(img_normalized.shape, sig["bbox"])
        paste_check = detect_copy_paste_signature(img_gray, sig["bbox"])
        
        sig_result = {
            "bbox": sig["bbox"],
            "confidence": sig["confidence"],
            "stroke_analysis": stroke_analysis,
            "placement": placement,
            "paste_check": paste_check,
            "combined_score": round(
                stroke_analysis["stroke_score"] * 0.4 +
                placement["placement_score"] * 0.3 +
                (1.0 - paste_check.get("paste_likelihood", 0.5)) * 0.3,
                3
            ),
        }
        analysis_results.append(sig_result)
    
    # Generate heatmap
    heatmap = generate_signature_heatmap(img_normalized, signatures)
    
    # Overall score
    if not analysis_results:
        sig_score = 0.3  # No signature is suspicious
    else:
        scores = [r["combined_score"] for r in analysis_results]
        sig_score = float(np.mean(scores))
    
    return {
        "module": "Signature Verification",
        "score": round(sig_score, 3),
        "signatures_found": len(signatures),
        "signatures": [
            {
                "bbox": r["bbox"],
                "confidence": r["confidence"],
                "combined_score": r["combined_score"],
                "stroke_naturalness": r["stroke_analysis"]["naturalness"],
                "placement_ok": r["placement"]["position_natural"],
                "paste_likelihood": r["paste_check"].get("paste_likelihood", 0),
            }
            for r in analysis_results
        ],
        "heatmap": heatmap,
        "details": {
            "signatures_detected": len(signatures),
            "avg_naturalness": round(
                float(np.mean([r["stroke_analysis"]["naturalness"] for r in analysis_results])), 3
            ) if analysis_results else 0,
            "paste_risk": round(
                float(np.max([r["paste_check"].get("paste_likelihood", 0) for r in analysis_results])), 3
            ) if analysis_results else 0,
        },
        "findings": _generate_sig_findings(analysis_results),
    }


def _generate_sig_findings(results: List[Dict]) -> List[str]:
    """Generate human-readable signature findings."""
    findings = []
    
    if not results:
        findings.append("🔴 No signatures detected in the document")
        return findings
    
    findings.append(f"{'✅' if len(results) >= 1 else '🟡'} {len(results)} signature region(s) detected")
    
    for i, r in enumerate(results):
        prefix = f"Signature #{i+1}"
        
        if r["stroke_analysis"]["naturalness"] > 0.7:
            findings.append(f"✅ {prefix}: Natural handwritten stroke pattern")
        elif r["stroke_analysis"]["naturalness"] < 0.5:
            findings.append(f"🔴 {prefix}: {r['stroke_analysis']['naturalness_note']}")
        
        if r["paste_check"].get("paste_likelihood", 0) > 0.6:
            findings.append(f"🔴 {prefix}: {r['paste_check'].get('note', 'Paste artifact detected')}")
        
        if not r["placement"]["position_natural"]:
            for note in r["placement"]["notes"]:
                findings.append(f"🟡 {prefix}: {note}")
    
    return findings
