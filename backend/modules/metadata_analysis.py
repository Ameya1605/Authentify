"""
Metadata Analysis Module
- Examines hidden metadata for editing history
- Detects software traces indicating tampering
- Identifies date/version inconsistencies
- Analyzes EXIF data, file properties
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import struct
from typing import Dict, List, Any, Optional
from datetime import datetime


def extract_exif_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract EXIF metadata from the image file.
    """
    metadata = {}
    
    try:
        img = Image.open(io.BytesIO(file_bytes))
        
        # Basic image info
        metadata["format"] = img.format
        metadata["mode"] = img.mode
        metadata["size"] = {"width": img.size[0], "height": img.size[1]}
        
        # EXIF data
        exif_data = img._getexif()
        if exif_data:
            exif_parsed = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                # Convert bytes to string for JSON serialization
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='replace')
                    except:
                        value = str(value)
                elif isinstance(value, tuple):
                    value = str(value)
                exif_parsed[tag_name] = value
            metadata["exif"] = exif_parsed
        else:
            metadata["exif"] = None
        
        # ICC Profile
        if "icc_profile" in img.info:
            metadata["has_icc_profile"] = True
        else:
            metadata["has_icc_profile"] = False
        
        # DPI
        if "dpi" in img.info:
            metadata["dpi"] = img.info["dpi"]
        
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata


def analyze_file_signature(file_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze file magic bytes and structure for integrity.
    """
    file_info = {
        "size_bytes": len(file_bytes),
        "detected_format": "unknown",
        "format_consistent": True,
    }
    
    # Check magic bytes
    if file_bytes[:2] == b'\xff\xd8':
        file_info["detected_format"] = "JPEG"
        # Check for proper JPEG ending
        if file_bytes[-2:] != b'\xff\xd9':
            file_info["format_consistent"] = False
            file_info["format_note"] = "JPEG file does not end with proper markers"
    elif file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        file_info["detected_format"] = "PNG"
    elif file_bytes[:4] == b'%PDF':
        file_info["detected_format"] = "PDF"
    elif file_bytes[:4] in (b'II\x2a\x00', b'MM\x00\x2a'):
        file_info["detected_format"] = "TIFF"
    elif file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WEBP':
        file_info["detected_format"] = "WEBP"
    elif file_bytes[:3] == b'BM\x00' or file_bytes[:2] == b'BM':
        file_info["detected_format"] = "BMP"
    
    return file_info


def detect_editing_software(metadata: Dict) -> Dict[str, Any]:
    """
    Detect traces of editing software in metadata.
    """
    software_traces = {
        "detected_software": [],
        "suspicious_software": [],
        "editing_detected": False,
    }
    
    exif = metadata.get("exif", {})
    if not exif:
        software_traces["note"] = "No EXIF data available for software analysis"
        return software_traces
    
    # Common editing software indicators
    editing_tools = [
        "photoshop", "gimp", "paint", "canva", "illustrator",
        "corel", "affinity", "pixlr", "fotor", "snapseed",
        "lightroom", "capture one", "rawtherapee", "darktable",
        "inkscape", "sketch", "figma",
    ]
    
    scanning_tools = [
        "scanner", "scan", "epson", "canon", "hp",
        "brother", "xerox", "ricoh", "konica",
    ]
    
    # Check Software tag
    software = str(exif.get("Software", "")).lower()
    make = str(exif.get("Make", "")).lower()
    model = str(exif.get("Model", "")).lower()
    
    for tool in editing_tools:
        if tool in software or tool in make:
            software_traces["detected_software"].append(software or make)
            software_traces["suspicious_software"].append(tool)
            software_traces["editing_detected"] = True
    
    for tool in scanning_tools:
        if tool in software or tool in make or tool in model:
            software_traces["detected_software"].append(f"Scanner: {software or make or model}")
    
    # Check processing software
    processing = str(exif.get("ProcessingSoftware", "")).lower()
    if processing:
        software_traces["detected_software"].append(f"Processing: {processing}")
        for tool in editing_tools:
            if tool in processing:
                software_traces["suspicious_software"].append(tool)
                software_traces["editing_detected"] = True
    
    return software_traces


def analyze_timestamps(metadata: Dict) -> Dict[str, Any]:
    """
    Analyze date/time metadata for inconsistencies.
    """
    timestamp_info = {
        "timestamps": {},
        "inconsistencies": [],
        "timestamp_score": 1.0,
    }
    
    exif = metadata.get("exif", {})
    if not exif:
        timestamp_info["note"] = "No EXIF timestamps available"
        timestamp_info["timestamp_score"] = 0.5
        return timestamp_info
    
    # Extract all timestamp fields
    date_fields = {
        "DateTime": exif.get("DateTime"),
        "DateTimeOriginal": exif.get("DateTimeOriginal"),
        "DateTimeDigitized": exif.get("DateTimeDigitized"),
    }
    
    parsed_dates = {}
    for field, value in date_fields.items():
        if value:
            timestamp_info["timestamps"][field] = str(value)
            try:
                dt = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                parsed_dates[field] = dt
            except:
                try:
                    dt = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
                    parsed_dates[field] = dt
                except:
                    pass
    
    # Check for inconsistencies
    if len(parsed_dates) >= 2:
        dates = list(parsed_dates.values())
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                diff = abs((dates[i] - dates[j]).total_seconds())
                if diff > 86400:  # More than 1 day difference
                    timestamp_info["inconsistencies"].append(
                        f"Date mismatch: {list(parsed_dates.keys())[i]} vs {list(parsed_dates.keys())[j]} "
                        f"({int(diff/3600)} hours apart)"
                    )
                    timestamp_info["timestamp_score"] -= 0.3
    
    # Check if dates are in the future
    now = datetime.now()
    for field, dt in parsed_dates.items():
        if dt > now:
            timestamp_info["inconsistencies"].append(f"Future date in {field}: {dt}")
            timestamp_info["timestamp_score"] -= 0.2
    
    timestamp_info["timestamp_score"] = max(0.0, timestamp_info["timestamp_score"])
    
    return timestamp_info


def analyze_compression_consistency(file_bytes: bytes, img: np.ndarray) -> Dict[str, Any]:
    """
    Analyze JPEG quantization tables and compression consistency.
    Multiple save operations leave traces.
    """
    compression_info = {
        "estimated_quality": 0,
        "double_compression": False,
        "compression_score": 1.0,
    }
    
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        
        if pil_img.format == "JPEG":
            # Estimate JPEG quality
            if hasattr(pil_img, 'quantization') and pil_img.quantization:
                qtable = list(pil_img.quantization.values())[0]
                if isinstance(qtable, (list, tuple)):
                    avg_q = sum(qtable) / len(qtable)
                    # Rough quality estimation
                    estimated_quality = max(1, min(100, int(100 - avg_q / 2)))
                    compression_info["estimated_quality"] = estimated_quality
                    
                    if estimated_quality < 50:
                        compression_info["compression_score"] -= 0.2
                        compression_info["note"] = "Low quality suggests multiple re-compressions"
            
            # Double compression detection using blocking artifacts
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            h, w = gray.shape
            
            # Analyze 8x8 block boundaries (JPEG artifacts)
            block_boundary_diffs = []
            for y in range(7, h - 1, 8):
                for x in range(0, w):
                    diff = abs(int(gray[y, x]) - int(gray[y + 1, x]))
                    block_boundary_diffs.append(diff)
            
            non_boundary_diffs = []
            for y in range(4, h - 1, 8):
                for x in range(0, w):
                    diff = abs(int(gray[y, x]) - int(gray[y + 1, x]))
                    non_boundary_diffs.append(diff)
            
            if block_boundary_diffs and non_boundary_diffs:
                boundary_mean = np.mean(block_boundary_diffs)
                non_boundary_mean = np.mean(non_boundary_diffs)
                
                blocking_ratio = boundary_mean / max(non_boundary_mean, 0.01)
                
                if blocking_ratio > 1.8:
                    compression_info["double_compression"] = True
                    compression_info["compression_score"] -= 0.3
                    compression_info["blocking_ratio"] = round(float(blocking_ratio), 3)
        
        elif pil_img.format == "PNG":
            compression_info["note"] = "PNG format - lossless compression"
            compression_info["estimated_quality"] = 100
        
    except Exception as e:
        compression_info["error"] = str(e)
    
    compression_info["compression_score"] = max(0.0, compression_info["compression_score"])
    return compression_info


def run_metadata_analysis(file_bytes: bytes, img_normalized: np.ndarray) -> Dict[str, Any]:
    """Full metadata analysis pipeline."""
    
    # Extract metadata
    metadata = extract_exif_metadata(file_bytes)
    
    # File signature analysis
    file_sig = analyze_file_signature(file_bytes)
    
    # Software detection
    software = detect_editing_software(metadata)
    
    # Timestamp analysis
    timestamps = analyze_timestamps(metadata)
    
    # Compression analysis
    compression = analyze_compression_consistency(file_bytes, img_normalized)
    
    # Overall score
    scores = [
        timestamps["timestamp_score"],
        compression["compression_score"],
        0.3 if software["editing_detected"] else 0.9,
        0.7 if file_sig["format_consistent"] else 0.3,
    ]
    
    meta_score = float(np.mean(scores))
    
    return {
        "module": "Metadata Analysis",
        "score": round(meta_score, 3),
        "file_info": {
            "size_bytes": file_sig["size_bytes"],
            "size_kb": round(file_sig["size_bytes"] / 1024, 1),
            "format": file_sig["detected_format"],
            "format_consistent": file_sig["format_consistent"],
        },
        "image_info": {
            "format": metadata.get("format"),
            "mode": metadata.get("mode"),
            "dimensions": metadata.get("size"),
            "dpi": str(metadata.get("dpi", "Not available")),
        },
        "software_analysis": {
            "editing_detected": software["editing_detected"],
            "detected_software": software["detected_software"],
            "suspicious_tools": software["suspicious_software"],
        },
        "timestamps": timestamps["timestamps"],
        "timestamp_issues": timestamps["inconsistencies"],
        "compression": {
            "estimated_quality": compression["estimated_quality"],
            "double_compression": compression["double_compression"],
        },
        "exif_fields": _sanitize_exif(metadata.get("exif")),
        "details": {
            "timestamp_score": timestamps["timestamp_score"],
            "compression_score": compression["compression_score"],
            "software_risk": software["editing_detected"],
            "format_valid": file_sig["format_consistent"],
        },
        "findings": _generate_metadata_findings(
            software, timestamps, compression, file_sig, metadata
        ),
    }


def _sanitize_exif(exif: Optional[Dict]) -> Dict:
    """Sanitize EXIF data for JSON serialization."""
    if not exif:
        return {}
    
    sanitized = {}
    for k, v in exif.items():
        try:
            # Test if value is JSON serializable
            import json
            json.dumps(v)
            sanitized[str(k)] = v
        except (TypeError, ValueError):
            sanitized[str(k)] = str(v)
    
    return sanitized


def _generate_metadata_findings(
    software: Dict, timestamps: Dict, compression: Dict,
    file_sig: Dict, metadata: Dict
) -> List[str]:
    """Generate human-readable metadata findings."""
    findings = []
    
    # Software findings
    if software["editing_detected"]:
        tools = ", ".join(software["suspicious_software"])
        findings.append(f"🔴 Editing software detected: {tools}")
    else:
        if software["detected_software"]:
            findings.append(f"✅ Software: {', '.join(software['detected_software'][:3])}")
        else:
            findings.append("🟡 No software metadata found (may have been stripped)")
    
    # Timestamp findings
    if timestamps["inconsistencies"]:
        for issue in timestamps["inconsistencies"]:
            findings.append(f"🔴 {issue}")
    elif timestamps["timestamps"]:
        findings.append("✅ Timestamps appear consistent")
    else:
        findings.append("🟡 No timestamp information available")
    
    # Compression
    if compression["double_compression"]:
        findings.append("🔴 Double JPEG compression detected - document may have been re-saved after editing")
    
    quality = compression.get("estimated_quality", 0)
    if quality > 0 and quality < 60:
        findings.append(f"🟡 Low compression quality ({quality}%) - possible multiple re-compressions")
    elif quality >= 80:
        findings.append(f"✅ Good image quality ({quality}%)")
    
    # File integrity
    if not file_sig["format_consistent"]:
        findings.append(f"🔴 File format inconsistency: {file_sig.get('format_note', 'Unknown issue')}")
    else:
        findings.append(f"✅ File format ({file_sig['detected_format']}) appears valid")
    
    # Metadata stripping check
    if not metadata.get("exif"):
        findings.append("🟡 EXIF metadata is missing - may have been intentionally stripped")
    
    return findings
