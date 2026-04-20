"""
OCR Text Extraction Module
- Extracts text fields (name, ID, date, grades) from documents
- Identifies inconsistencies in text regions
- Generates confidence scores for extracted text
- Produces heatmap of text confidence levels
"""

import cv2
import numpy as np
import re
import os
import pytesseract
from pytesseract import Output

# Set tesseract path for Windows
if os.name == 'nt':
    _tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path

from typing import Dict, List, Any


# Common certificate field patterns
FIELD_PATTERNS = {
    "name": [
        r"(?i)(?:name|student|candidate|presented to)\s*[:;]?\s*([A-Z\s]{3,})",
        r"(?i)(?:certify that)\s+([A-Z\s]{3,})",
        r"(?i)(?:mr\.|ms\.|mrs\.)\s+([A-Z\s]{3,})",
    ],
    "roll_number": [
        r"(?i)(?:roll\s*(?:no|number|#)?|id|reg(?:istration)?|serial)\s*[:;]?\s*(\d+\w*)",
    ],
    "date": [
        r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(?i)((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})",
        r"(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})",
    ],
    "degree": [
        r"(?i)(?:degree|diploma|certificate|bachelor|master|doctor)\s+(?:of\s+)?([A-Z\w\s,]{5,})",
    ],
}


def extract_text_regions(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect text regions using MSER and contour analysis.
    Returns list of bounding boxes with confidence.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use morphological operations to find text lines
    # 1. Edge detection (vertical edges)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 2. Dilation to connect letters into words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)
        area = w * h
        # Typical text lines are wider than they are tall, but not massive blocks
        if aspect_ratio >= 1.0 and h > 8 and h < 50 and area > 100:
            text_regions.append({
                "x": int(x), "y": int(y),
                "width": int(w), "height": int(h),
                "confidence": min(1.0, w / 200.0), # simple pseudo-confidence
            })
    
    # Merge overlapping regions
    merged = _merge_overlapping_boxes(text_regions)
    return merged


def _merge_overlapping_boxes(boxes: List[Dict], overlap_thresh: float = 0.3) -> List[Dict]:
    """Merge overlapping bounding boxes using NMS-like approach."""
    if not boxes:
        return []
    
    coords = np.array([[b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]] for b in boxes])
    areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    idxs = np.argsort(areas)[::-1]
    
    picked = []
    while len(idxs) > 0:
        i = idxs[0]
        picked.append(i)
        
        xx1 = np.maximum(coords[i, 0], coords[idxs[1:], 0])
        yy1 = np.maximum(coords[i, 1], coords[idxs[1:], 1])
        xx2 = np.minimum(coords[i, 2], coords[idxs[1:], 2])
        yy2 = np.minimum(coords[i, 3], coords[idxs[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs[1:]]
        
        remaining = np.where(overlap < overlap_thresh)[0]
        idxs = idxs[remaining + 1]
    
    return [boxes[i] for i in picked[:50]]  # Limit to top 50 regions


def analyze_text_consistency(img: np.ndarray) -> Dict[str, Any]:
    """
    Analyze text regions for consistency issues:
    - Font uniformity (variance in stroke width)
    - Alignment analysis
    - Spacing regularity
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Edge detection for stroke analysis
    edges = cv2.Canny(gray, 50, 150)
    
    # Divide image into grid for regional analysis
    grid_rows, grid_cols = 8, 6
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    density_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    variance_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell = edges[y1:y2, x1:x2]
            cell_gray = gray[y1:y2, x1:x2]
            density_map[r, c] = np.mean(cell) / 255.0
            variance_map[r, c] = np.std(cell_gray.astype(np.float32)) / 255.0
    
    # Detect anomalous regions (high variance = potential tampering)
    mean_var = np.mean(variance_map)
    std_var = np.std(variance_map)
    anomaly_map = np.abs(variance_map - mean_var) / max(std_var, 1e-6)
    
    # Score: lower anomaly = more consistent = more authentic
    max_anomaly = float(np.max(anomaly_map))
    consistency_score = max(0.0, min(1.0, 1.0 - (max_anomaly / 6.0)))
    
    return {
        "consistency_score": round(consistency_score, 3),
        "density_map": density_map.tolist(),
        "variance_map": variance_map.tolist(),
        "anomaly_map": anomaly_map.tolist(),
        "max_anomaly_zscore": round(max_anomaly, 3),
    }


def extract_fields_from_text(raw_text: str) -> Dict[str, str]:
    """Smart field extraction logic."""
    clean_text = raw_text.replace('\n', ' ')
    extracted = {}
    
    for field_name, patterns in FIELD_PATTERNS.items():
        found_val = "Not Detected"
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                val = match.group(1).strip()
                # Simple cleanup
                val = re.sub(r'[\s]{2,}', ' ', val)
                val = re.sub(r'^[:;\s]+', '', val)
                if len(val) > 2:
                    found_val = val
                    break
        extracted[field_name] = found_val
    return extracted


def generate_text_heatmap(img: np.ndarray, text_regions: List[Dict], anomaly_map: List[List[float]]) -> np.ndarray:
    """Generate a high-fidelity word-level heatmap overlay."""
    h, w = img.shape[:2]
    base = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    
    grid_rows = len(anomaly_map)
    grid_cols = len(anomaly_map[0]) if grid_rows > 0 else 0
    cell_h, cell_w = h // max(grid_rows, 1), w // max(grid_cols, 1)
    
    for region in text_regions:
        rx, ry, rw, rh = region["x"], region["y"], region["width"], region["height"]
        conf = region.get("confidence", 0.5)
        
        # Calculate grid anomaly for this word's location
        cx, cy = rx + rw // 2, ry + rh // 2
        r_idx = min(grid_rows - 1, max(0, cy // cell_h))
        c_idx = min(grid_cols - 1, max(0, cx // cell_w))
        anomaly = anomaly_map[r_idx][c_idx] if grid_rows > 0 else 0
        
        # Severity selection
        # Red if OCR is uncertain OR grid anomaly is high
        if conf < 0.6 or anomaly > 3.0:
            color = (0, 0, 255) # Red
            alpha = 0.5
        elif conf < 0.85 or anomaly > 1.8:
            color = (0, 165, 255) # Orange
            alpha = 0.3
        else:
            color = (0, 255, 0) # Green
            alpha = 0.1
            
        # Draw transparent box
        sub_rect = overlay[ry:ry+rh, rx:rx+rw]
        if sub_rect.size > 0:
            fill = np.full_like(sub_rect, color)
            cv2.addWeighted(sub_rect, 1.0 - alpha, fill, alpha, 0, sub_rect)
            cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), color, 1)
        
    return overlay


def run_ocr_analysis(img_normalized: np.ndarray, img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Full OCR analysis pipeline using Tesseract for precise word-level results.
    """
    try:
        # 1. OCR Data Extraction
        # Use Tesseract to get bounding boxes, text, and confidence
        custom_config = r'--oem 3 --psm 3'
        d = pytesseract.image_to_data(img_normalized, output_type=Output.DICT, config=custom_config)
        
        n_boxes = len(d['text'])
        text_regions = []
        raw_full_text = []
        
        for i in range(n_boxes):
            text = d['text'][i].strip()
            conf = float(d['conf'][i])
            
            if text and conf > -1:
                text_regions.append({
                    "x": int(d['left'][i]),
                    "y": int(d['top'][i]),
                    "width": int(d['width'][i]),
                    "height": int(d['height'][i]),
                    "text": text,
                    "confidence": conf / 100.0
                })
                raw_full_text.append(text)
        
        full_text = " ".join(raw_full_text)
        
        # 2. Analyze Text Consistency (Using our custom grid analysis + OCR confidence)
        consistency = analyze_text_consistency(img_gray)
        
        # Adjust consistency score using Tesseract's mean confidence
        mean_conf = np.mean([r["confidence"] for r in text_regions]) if text_regions else 0.5
        ocr_score = (consistency["consistency_score"] * 0.4 + mean_conf * 0.6)
        
        # 3. Generate Heatmap
        # High opacity overlay for visualizations
        heatmap = generate_text_heatmap(img_normalized, text_regions, consistency["anomaly_map"])
        
        # 4. Extract Structured Fields
        extracted_fields = extract_fields_from_text(full_text)
        
        return {
            "module": "OCR Text Extraction",
            "score": round(float(ocr_score), 3),
            "text_regions_count": len(text_regions),
            "extracted_fields": extracted_fields,
            "text_regions": text_regions[:50],  # Return more for verification
            "consistency": consistency,
            "heatmap": heatmap,
            "details": {
                "mean_ocr_confidence": round(float(mean_conf), 3),
                "consistency_score": consistency["consistency_score"],
                "anomaly_zones": int(np.sum(np.array(consistency["anomaly_map"]) > 2.5)),
            },
            "findings": _generate_ocr_findings(consistency, len(text_regions), mean_conf),
        }
    except Exception as e:
        # Fallback to basic if Tesseract fails
        return {
            "module": "OCR Text Extraction",
            "score": 0.5,
            "error": str(e),
            "heatmap": img_normalized,
            "findings": ["⚠️ OCR Engine error: Analysis limited to basic structure."]
        }


def _generate_ocr_findings(consistency: Dict, num_regions: int, mean_conf: float) -> List[str]:
    """Generate human-readable findings based on Tesseract data."""
    findings = []
    
    if mean_conf < 0.6:
        findings.append("🔴 Low text clarity detected - document might be a low-quality photocopy or digitally manipulated")
    elif mean_conf < 0.8:
        findings.append("🟡 Moderate text confidence - some regions are blurry or inconsistent")
    
    if consistency["consistency_score"] < 0.5:
        findings.append("🔴 High font inconsistency - likely font substitution or text replacement across fields")
    
    anomaly_zones = int(np.sum(np.array(consistency["anomaly_map"]) > 2.5))
    if anomaly_zones > 2:
        findings.append(f"🔴 {anomaly_zones} regions show suspicious structural variance")
    
    findings.append(f"✅ Extracted {num_regions} distinct text elements for verification")
    
    return findings
