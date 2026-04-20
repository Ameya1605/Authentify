# Authentify - Master Application Prompt

You can use the following comprehensive prompt to generate the entire Authentify web application. It includes all the structural requirements, visual indicator specs, and module boundaries described in your synopsis.

***

**System Role / Context:**
You are an expert full-stack developer and AI/Computer Vision engineer. I want you to build "Authentify," a fully automated, AI-based web application designed to detect forged and tampered educational certificates. The system provides automated verification, integrating OCR, image processing, and machine learning.

Please generate the complete codebase for this full-stack application (e.g., FastAPI/Python for the backend, React/Vite/Next.js for the frontend). Ensure that the UI/UX feels extremely premium, highly interactive, and visually stunning.

## System Architecture & Requirements

### 1. Image Preprocessing (Phase 2)
Implement a robust image preprocessing module that takes an uploaded document and performs:
- Image normalization and resizing for consistent processing.
- Grayscale conversion for uniform pixel analysis.
- Noise removal (e.g., Gaussian Blur, Bilateral Filtering) to improve clarity.
- Adaptive thresholding for binarization to aid text and pattern extraction.

### 2. OCR Text Extraction Module
Extract key certificate fields (Name, Roll Number/ID, Grades, Date of Issue, Institution).
- Identify inconsistencies in text regions by analyzing font uniformity, spacing, and alignment.
- Implement stroke variance or character density anomaly detection.
- **Visual Indicator:** Generate a "Text Consistency Heatmap" overlaid on the original document that highlights anomalous text zones indicating potential text manipulation. Provide a confidence score for text extraction.

### 3. Stamp / Watermark Verification Module
Identify altered, missing, or tampered official seals and watermarks.
- Detect circular or elliptical seal patterns (e.g., using Hough Circle Transform).
- Analyze color uniformity within the stamp area (common seal colors like red, blue, purple).
- Check stamp integrity using edge density and noise pattern analysis.
- Identify hidden watermarks using frequency domain analysis (FFT/DFT).
- **Visual Indicator:** Create a heatmap or bounding box overlay isolating the detected stamps and highlighting their integrity scores.

### 4. Signature Verification Module
Examine the authenticity of signatures in the document.
- Localize signature regions, generally near the lower document boundary.
- **Stroke Analysis:** Process stroke width distribution, continuity, and pressure variance (naturalness vs. generated).
- **Placement Logic:** Validate if the signature is positioned correctly and naturally.
- **Copy-Paste Check:** Search for paste artifacts, sharp boundary discontinuities, and compression mismatches specifically around the signature.
- **Visual Indicator:** Present a "Signature Heatmap" or targeted viewing zone outlining signatures with a bounding box and stroke naturalness percentage.

### 5. Clone Region Detection (Copy-Move Forgery)
Locate copy-pasted or duplicated regions indicating text, signature, or stamp reuse.
- Implement overlap block extraction (e.g., via DCT coefficients or robust feature matching).
- Cluster matches into primary cloned regions based on vector proximity and spatial distance (matching blocks that are too far apart indicate a copy-paste duplication).
- Execute Error Level Analysis (ELA) to spotlight segments manipulated or resaved with varied compression qualities.
- **Visual Indicator:** Draw an ELA map and a clone detection heatmap highlighting directly copied areas connected by visual linking lines.

### 6. Metadata Analysis Module
Examine the hidden layers of the document file for manipulation traces.
- Extract EXIF metadata, ICC profiles, and resolution data.
- Hunt for software tags suggesting usage of editing software (e.g., Photoshop, Gimp, Canva).
- Validate date/timestamp consistency, seeking mismatches between creation and modification dates.
- Review file signature (magic bytes) to ensure consistent format markers.

### 7. Results Fusion & Scoring Engine (Phase 4)
Integrate all independent module results using a weighted scoring mechanism to output a single reliable decision.
- Determine individual weights for OCR, Stamp Verification, Signature Verification, Clone Detection, and Metadata Analysis.
- Consolidate output into a singular **Authenticity Score**.
- Output specific, actionable verdicts (e.g., Authentic, Likely Authentic, Suspicious, Forged).

### 8. Frontend UI/UX Design & User Experience
The User Interface must be modern, highly presentable, and look premium. 
- **Dynamic Interactions:** Implement a smooth drag-and-drop file upload zone with micro-animations.
- **Theming:** Use a very modern, eye-catching aesthetic like dark-mode glassmorphism, accent gradients, and precise typography.
- **Visual Indicators (Crucial):** Instead of just plain text output, display a rich visual console. Combine the heatmaps from the detection modules into interactive tabs where the user can toggle between the "Original", "Annotated Bounding Boxes", "OCR Heatmap", "Stamp Heatmap", "Signature Heatmap", and "ELA Map" views.
- **Reporting:** Display a comprehensive, structured "Findings" listing panel and a Metadata summary, culminating in an option to download a structured Verification Report.

## Deliverables
- Fully functional backend processing pipeline in Python.
- Beautiful, high-tech React-based frontend dashboard.
- Clear REST API routes linking the frontend to the forensic modules.
- Visually clear, highly accurate heatmap feedback guiding the end-user on exact forgery localization.
