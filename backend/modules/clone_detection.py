"""
Clone Region Detection Module
- Detects duplicated or copy-pasted regions within the document
- Uses block-based matching for copy-move forgery detection
- Identifies reused signatures, seals, text blocks
- Generates clone detection heatmap
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def extract_blocks(img_gray: np.ndarray, block_size: int = 16, step: int = 4) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract overlapping blocks from the image for comparison.
    Returns feature vectors and their positions.
    """
    h, w = img_gray.shape[:2]
    blocks = []
    positions = []
    
    for y in range(0, h - block_size, step):
        for x in range(0, w - block_size, step):
            block = img_gray[y:y+block_size, x:x+block_size]
            
            # Skip blocks with no detail (e.g., solid white/black background)
            if np.std(block) < 15.0:
                continue
                
            # Use DCT coefficients as features (more robust than raw pixels)
            dct = cv2.dct(block.astype(np.float32))
            # Take top-left corner of DCT (most significant coefficients)
            feature = dct[:8, :8].flatten()
            blocks.append(feature)
            positions.append((x, y))
    
    if len(blocks) == 0:
        return np.array([]), []
        
    return np.array(blocks), positions


def find_clone_matches(features: np.ndarray, positions: List[Tuple[int, int]], 
                       threshold: float = 0.95, min_distance: int = 30) -> List[Dict[str, Any]]:
    """
    Find matching blocks that are far apart (indicating copy-move).
    Uses lexicographic sorting for efficient matching.
    """
    if len(features) == 0:
        return []
    
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = features / norms
    
    # Sort by feature similarity using lexicographic sort
    indices = np.lexsort(normalized.T)
    sorted_features = normalized[indices]
    sorted_positions = [positions[i] for i in indices]
    
    matches = []
    seen_regions = set()
    
    for i in range(len(sorted_features) - 1):
        # Compare adjacent sorted features
        similarity = np.dot(sorted_features[i], sorted_features[i + 1])
        
        if similarity > threshold:
            pos1 = sorted_positions[i]
            pos2 = sorted_positions[i + 1]
            
            # Check minimum distance (nearby blocks are naturally similar)
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            if dist > min_distance:
                # Avoid duplicate regions
                region_key = (
                    pos1[0] // 20, pos1[1] // 20,
                    pos2[0] // 20, pos2[1] // 20,
                )
                if region_key not in seen_regions:
                    seen_regions.add(region_key)
                    matches.append({
                        "source": {"x": pos1[0], "y": pos1[1]},
                        "target": {"x": pos2[0], "y": pos2[1]},
                        "similarity": round(float(similarity), 4),
                        "distance": round(float(dist), 1),
                    })
    
    # Sort by similarity (most suspicious first)
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches[:30]  # Return top 30 matches


def cluster_clone_regions(matches: List[Dict], block_size: int = 16, 
                          cluster_radius: int = 40) -> List[Dict[str, Any]]:
    """
    Cluster individual block matches into larger forged regions.
    """
    if not matches:
        return []
    
    # Collect all match points
    source_points = []
    target_points = []
    
    for m in matches:
        source_points.append((m["source"]["x"], m["source"]["y"]))
        target_points.append((m["target"]["x"], m["target"]["y"]))
    
    # Simple distance-based clustering
    def cluster_points(points: List[Tuple[int, int]], radius: int) -> List[Dict]:
        if not points:
            return []
        
        clusters = []
        used = set()
        
        for i, p1 in enumerate(points):
            if i in used:
                continue
            cluster = [p1]
            used.add(i)
            
            for j, p2 in enumerate(points):
                if j in used:
                    continue
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist < radius:
                    cluster.append(p2)
                    used.add(j)
            
            if len(cluster) >= 2:
                xs = [p[0] for p in cluster]
                ys = [p[1] for p in cluster]
                clusters.append({
                    "bbox": {
                        "x": min(xs),
                        "y": min(ys),
                        "width": max(xs) - min(xs) + block_size,
                        "height": max(ys) - min(ys) + block_size,
                    },
                    "match_count": len(cluster),
                })
        
        return clusters
    
    source_clusters = cluster_points(source_points, cluster_radius)
    target_clusters = cluster_points(target_points, cluster_radius)
    
    # Pair source and target clusters
    clone_pairs = []
    for i, sc in enumerate(source_clusters):
        if i < len(target_clusters):
            tc = target_clusters[i]
            clone_pairs.append({
                "source_region": sc["bbox"],
                "target_region": tc["bbox"],
                "match_density": sc["match_count"],
                "severity": "high" if sc["match_count"] > 5 else "medium" if sc["match_count"] > 2 else "low",
            })
    
    return clone_pairs


def error_level_analysis(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    Error Level Analysis (ELA) - detects regions with different 
    compression levels, indicating potential manipulation.
    """
    # Encode and decode at specified quality to get re-compressed version
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', img, encode_params)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # Compute difference (error levels)
    if img.shape != decoded.shape:
        decoded = cv2.resize(decoded, (img.shape[1], img.shape[0]))
    
    ela = cv2.absdiff(img, decoded)
    
    # Scale for visibility
    ela_scaled = cv2.convertScaleAbs(ela, alpha=15.0)
    
    return ela_scaled


def generate_clone_heatmap(img: np.ndarray, matches: List[Dict], 
                           clone_regions: List[Dict], block_size: int = 16) -> np.ndarray:
    """Generate heatmap showing clone/copy-paste detections."""
    if len(img.shape) == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()
    
    h, w = base.shape[:2]
    heatmap_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Mark clone regions with semi-transparent boxes
    for region in clone_regions:
        for key in ["source_region", "target_region"]:
            bbox = region[key]
            x, y, rw, rh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            color = (0, 0, 255) if region["severity"] == "high" else (0, 165, 255)
            # Create a localized highlight
            sub_rect = heatmap_overlay[y:y+rh, x:x+rw]
            # Darken existing slightly, then blend box color
            overlay_box = np.full_like(sub_rect, color)
            cv2.addWeighted(sub_rect, 0.5, overlay_box, 0.5, 0, sub_rect)
            
            # Draw strong border
            cv2.rectangle(heatmap_overlay, (x, y), (x+rw, y+rh), color, 3)
            
    # Apply the colored overlay to the base image only where we drew
    mask = np.any(heatmap_overlay > 0, axis=-1)
    overlay = base.copy()
    overlay[mask] = cv2.addWeighted(base[mask], 0.4, heatmap_overlay[mask], 0.6, 0)
    
    # Draw connection lines between paired regions
    for region in clone_regions:
        src = region["source_region"]
        tgt = region["target_region"]
        src_center = (src["x"] + src["width"]//2, src["y"] + src["height"]//2)
        tgt_center = (tgt["x"] + tgt["width"]//2, tgt["y"] + tgt["height"]//2)
        
        color = (0, 0, 255) if region["severity"] == "high" else (0, 165, 255)
        # Draw dashed or thick line connecting them
        cv2.line(overlay, src_center, tgt_center, color, 2, cv2.LINE_AA)
        # Draw dots at centers
        cv2.circle(overlay, src_center, 4, (255, 255, 255), -1)
        cv2.circle(overlay, tgt_center, 4, (255, 255, 255), -1)
        cv2.circle(overlay, src_center, 6, color, 2)
        cv2.circle(overlay, tgt_center, 6, color, 2)
    
    return overlay


def run_clone_detection(img_normalized: np.ndarray, img_gray: np.ndarray) -> Dict[str, Any]:
    """Full clone/copy-move detection pipeline."""
    
    # Extract blocks and find matches
    block_size = 16
    features, positions = extract_blocks(img_gray, block_size=block_size, step=6)
    matches = find_clone_matches(features, positions, threshold=0.97, min_distance=40)
    
    # Cluster into regions
    clone_regions = cluster_clone_regions(matches, block_size=block_size)
    
    # Error Level Analysis
    ela_map = error_level_analysis(img_normalized)
    
    # Combine ELA with clone detection for overall score
    ela_intensity = float(np.mean(ela_map)) / 255.0
    
    # Generate heatmap
    heatmap = generate_clone_heatmap(img_normalized, matches, clone_regions, block_size)
    
    # Score calculation
    n_matches = len(matches)
    n_regions = len(clone_regions)
    high_severity = sum(1 for r in clone_regions if r["severity"] == "high")
    
    if n_matches == 0:
        clone_score = 0.95  # No clones found = likely authentic
    else:
        match_penalty = min(0.5, n_matches * 0.02)
        region_penalty = min(0.3, high_severity * 0.15)
        clone_score = max(0.1, 0.95 - match_penalty - region_penalty)
    
    # Adjust with ELA
    if ela_intensity > 0.3:
        clone_score = max(0.1, clone_score - 0.15)
    
    return {
        "module": "Clone Region Detection",
        "score": round(float(clone_score), 3),
        "matches_found": n_matches,
        "clone_regions": clone_regions,
        "ela_intensity": round(ela_intensity, 4),
        "heatmap": heatmap,
        "ela_map": ela_map,
        "details": {
            "total_matches": n_matches,
            "clone_regions": n_regions,
            "high_severity_regions": high_severity,
            "ela_mean_intensity": round(ela_intensity, 4),
        },
        "findings": _generate_clone_findings(n_matches, clone_regions, ela_intensity),
    }


def _generate_clone_findings(n_matches: int, regions: List, ela_intensity: float) -> List[str]:
    """Generate human-readable clone detection findings."""
    findings = []
    
    if n_matches == 0:
        findings.append("✅ No copy-move duplications detected")
    elif n_matches < 5:
        findings.append(f"🟡 {n_matches} potential block matches found - minor concern")
    else:
        findings.append(f"🔴 {n_matches} block matches detected - significant copy-move indicators")
    
    high = sum(1 for r in regions if r["severity"] == "high")
    if high > 0:
        findings.append(f"🔴 {high} high-severity cloned region(s) identified")
    
    medium = sum(1 for r in regions if r["severity"] == "medium")
    if medium > 0:
        findings.append(f"🟡 {medium} moderate clone region(s) detected")
    
    if ela_intensity > 0.3:
        findings.append("🔴 High ELA intensity - document may have been re-compressed after editing")
    elif ela_intensity > 0.15:
        findings.append("🟡 Moderate ELA intensity - some regions show compression differences")
    else:
        findings.append("✅ ELA analysis shows consistent compression levels")
    
    return findings
