"""
AUTHENTIFY - Fake Document Detection Backend
Main FastAPI server integrating all verification modules.

Phase 4 — Integration & Result Fusion
- Integrates all modules into a single processing pipeline
- Merges results using a weighted scoring mechanism
- Highlights tampered regions via bounding boxes
- Generates a single Authenticity Score
"""

import os
import io
import uuid
import base64
import time
from typing import Dict, Any

import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF for PDF extraction
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from modules.preprocessing import preprocess_pipeline
from modules.ocr_extraction import run_ocr_analysis
from modules.stamp_verification import run_stamp_analysis
from modules.signature_verification import run_signature_analysis
from modules.clone_detection import run_clone_detection
from modules.metadata_analysis import run_metadata_analysis
from modules.report_generator import generate_pdf_report
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

app = FastAPI(
    title="Authentify API",
    description="AI-powered Fake Document Detection System",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module weights for final score
MODULE_WEIGHTS = {
    "OCR Text Extraction": 0.20,
    "Stamp/Watermark Verification": 0.20,
    "Signature Verification": 0.20,
    "Clone Region Detection": 0.25,
    "Metadata Analysis": 0.15,
}


def extract_image_bytes(file_bytes: bytes) -> bytes:
    """Attempt to extract or convert the uploaded file to an image."""
    # Check for PDF magic bytes
    if file_bytes[:4] == b'%PDF':
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=150)
                return pix.tobytes("png")
        except Exception as e:
            raise ValueError(f"Could not extract image from PDF: {str(e)}")
        raise ValueError("PDF document is empty.")
    
    return file_bytes


def encode_image_to_base64(img: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV image to base64 JPEG string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', img, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def deskew_document(image_np: np.ndarray) -> np.ndarray:
    """
    Detects document boundaries and performs a perspective transform 
    to flatten the image for accurate forensic analysis.
    """
    # 1. Edge detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # 2. Find the document contour (largest 4-sided polygon)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screen_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (image_np.shape[0] * image_np.shape[1] * 0.2):
            screen_cnt = approx
            break
            
    if screen_cnt is None:
        return image_np # Fallback to original if detection fails
    
    # 3. Perspective Transform
    pts = screen_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_np, M, (max_width, max_height))
    
    return warped


def compute_authenticity_score(module_results: list) -> Dict[str, Any]:
    """
    Compute weighted authenticity score from all modules.
    Returns score, verdict, and confidence level.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    
    for result in module_results:
        module_name = result["module"]
        weight = MODULE_WEIGHTS.get(module_name, 0.15)
        weighted_sum += result["score"] * weight
        total_weight += weight
    
    if total_weight == 0:
        final_score = 0.5
    else:
        final_score = weighted_sum / total_weight
    
    # Determine verdict
    if final_score >= 0.80:
        verdict = "AUTHENTIC"
        verdict_description = "The document appears to be genuine with high confidence."
        risk_level = "LOW"
    elif final_score >= 0.60:
        verdict = "LIKELY AUTHENTIC"
        verdict_description = "The document appears mostly genuine but has some minor concerns."
        risk_level = "MODERATE"
    elif final_score >= 0.40:
        verdict = "SUSPICIOUS"
        verdict_description = "The document shows signs of potential manipulation. Manual review recommended."
        risk_level = "HIGH"
    elif final_score >= 0.20:
        verdict = "LIKELY FORGED"
        verdict_description = "Multiple indicators suggest this document has been tampered with."
        risk_level = "VERY HIGH"
    else:
        verdict = "FORGED"
        verdict_description = "Strong evidence of document forgery detected across multiple modules."
        risk_level = "CRITICAL"
    
    # Confidence based on agreement between modules
    scores = [r["score"] for r in module_results]
    score_std = float(np.std(scores)) if scores else 0.5
    confidence = max(0.3, 1.0 - score_std)
    
    return {
        "score": round(float(final_score), 3),
        "percentage": round(float(final_score) * 100, 1),
        "verdict": verdict,
        "verdict_description": verdict_description,
        "risk_level": risk_level,
        "confidence": round(float(confidence), 3),
    }


def generate_annotated_image(img: np.ndarray, module_results: list) -> np.ndarray:
    """
    Generate annotated image with bounding boxes from all modules.
    """
    annotated = img.copy()
    
    for result in module_results:
        if result["module"] == "Signature Verification":
            for sig in result.get("signatures", []):
                bbox = sig["bbox"]
                color = (0, 255, 0) if sig.get("combined_score", 0) > 0.6 else (0, 165, 255) if sig.get("combined_score", 0) > 0.4 else (0, 0, 255)
                # Draw bounding box
                cv2.rectangle(annotated,
                    (bbox["x"], bbox["y"]),
                    (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                    color, 2
                )
                label = f"Sig: {sig.get('combined_score', 0):.0%}"
                cv2.putText(annotated, label,
                    (bbox["x"], bbox["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                
        elif result["module"] == "Stamp/Watermark Verification":
            # Draw circular stamps
            for stamp in result.get("circular_stamps", []):
                cx, cy, r = stamp["center_x"], stamp["center_y"], stamp["radius"]
                integrity = stamp.get("integrity_score", stamp.get("integrity", {}).get("integrity_score", 0))
                color = (255, 0, 255) if integrity > 0.6 else (0, 165, 255)
                cv2.circle(annotated, (cx, cy), r, color, 3)
                label = f"Seal: {integrity:.0%}"
                cv2.putText(annotated, label,
                    (cx - 30, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                
            # Draw colored regions
            for region in result.get("colored_regions", []):
                bbox = region["bbox"]
                integrity = region.get("integrity_score", region.get("integrity", {}).get("integrity_score", 0))
                color = (255, 0, 255) if integrity > 0.6 else (0, 0, 255)
                cv2.rectangle(annotated,
                    (bbox["x"], bbox["y"]),
                    (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                    color, 2
                )
                label = f"Stamp: {integrity:.0%}"
                cv2.putText(annotated, label,
                    (bbox["x"], bbox["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
        
        elif result["module"] == "Clone Region Detection":
            for region in result.get("clone_regions", []):
                for key in ["source_region", "target_region"]:
                    bbox = region[key]
                    color = (0, 0, 255) if region["severity"] == "high" else (0, 165, 255)
                    cv2.rectangle(annotated,
                        (bbox["x"], bbox["y"]),
                        (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                        color, 2
                    )
                    label = f"Clone ({region['severity']})"
                    cv2.putText(annotated, label,
                        (bbox["x"], bbox["y"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
    
    return annotated


@app.get("/")
async def root():
    return {"status": "ok", "service": "Authentify API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "modules": list(MODULE_WEIGHTS.keys())}


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Main document analysis endpoint.
    Accepts an upload of any file and tries to extract an image for verification modules.
    """
    # Read file
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    if len(file_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
    
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        # Convert to image if it's a PDF or other supported format
        image_bytes = extract_image_bytes(file_bytes)
        
        # New Step: Auto-Deskew / Perspective Correction
        # Convert bytes to OpenCV format first
        nparr = np.frombuffer(image_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        deskewed_img = deskew_document(raw_img)
        
        # Convert back to bytes for the existing preprocess_pipeline
        _, deskewed_bytes_arr = cv2.imencode('.png', deskewed_img)
        deskewed_bytes = deskewed_bytes_arr.tobytes()
        
        # Phase 2: Preprocessing
        preprocessed = preprocess_pipeline(deskewed_bytes)
        img_normalized = preprocessed["normalized"]
        img_gray = preprocessed["grayscale"]
        
        # Phase 3: Run all detection modules
        module_results = []
        
        # Module 1: OCR Text Extraction
        ocr_result = run_ocr_analysis(img_normalized, img_gray)
        ocr_heatmap = encode_image_to_base64(ocr_result.pop("heatmap"))
        module_results.append(ocr_result)
        
        # Module 2: Stamp/Watermark Verification
        stamp_result = run_stamp_analysis(img_normalized, img_gray)
        stamp_heatmap = encode_image_to_base64(stamp_result.pop("heatmap"))
        module_results.append(stamp_result)
        
        # Module 3: Signature Verification
        sig_result = run_signature_analysis(img_normalized, img_gray)
        sig_heatmap = encode_image_to_base64(sig_result.pop("heatmap"))
        module_results.append(sig_result)
        
        # Module 4: Clone Region Detection
        clone_result = run_clone_detection(img_normalized, img_gray)
        clone_heatmap = encode_image_to_base64(clone_result.pop("heatmap"))
        ela_map = encode_image_to_base64(clone_result.pop("ela_map"))
        module_results.append(clone_result)
        
        # Module 5: Metadata Analysis
        meta_result = run_metadata_analysis(file_bytes, img_normalized)
        module_results.append(meta_result)
        
        # Phase 4: Result Fusion
        authenticity = compute_authenticity_score(module_results)
        
        # Generate annotated image
        annotated = generate_annotated_image(img_normalized, module_results)
        annotated_b64 = encode_image_to_base64(annotated)
        original_b64 = encode_image_to_base64(img_normalized)
        
        processing_time = round(time.time() - start_time, 2)
        
        # Collect all findings
        all_findings = []
        for result in module_results:
            findings = result.pop("findings", [])
            all_findings.extend(findings)
        
        # Build response
        response = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "processing_time_seconds": processing_time,
            "authenticity": authenticity,
            "modules": [
                {
                    "module": r["module"],
                    "score": r["score"],
                    "details": r.get("details", {}),
                }
                for r in module_results
            ],
            "findings": all_findings,
            "heatmaps": {
                "ocr": ocr_heatmap,
                "stamp": stamp_heatmap,
                "signature": sig_heatmap,
                "clone": clone_heatmap,
                "ela": ela_map,
            },
            "images": {
                "original": original_b64,
                "annotated": annotated_b64,
            },
            "metadata": next(
                (r for r in module_results if r["module"] == "Metadata Analysis"),
                {}
            ),
        }
        
        return JSONResponse(content=response)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/report/pdf")
async def get_pdf_report(data: Dict[str, Any]):
    """
    Generates a PDF report from the provided analysis data.
    """
    try:
        pdf_bytes = generate_pdf_report(data)
        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=authentify-report-{data.get('analysis_id', 'result')[:8]}.pdf"
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
