# Authentify: AI-Powered Fake Document Detection

![Authentify Banner](frontend/src/assets/hero.png)

**Authentify** is a professional-grade forensic engine designed to detect forged certificates, manipulated documents, and digital tampering. By leveraging advanced Computer Vision (OpenCV), Multi-modal AI (HuggingFace/Tesseract), and Forensic Analysis (ELA), Authentify provides a comprehensive "Authenticity Score" and detailed heatmaps of suspicious regions.

---

## 🛡️ Key Features

### 🔍 Multimodal Forensic Analysis
Authentify doesn't just look at text; it analyzes the structural integrity of the entire document:
- **OCR Text Extraction**: Precision extraction of names, IDs, and dates with word-level confidence heatmaps.
- **Stamp & Watermark Verification**: Detects official seals and analyzes their integrity/positioning.
- **Signature Analysis**: Verifies handwritten stroke naturalness and detects "cut-and-paste" signature artifacts.
- **Clone Region Detection**: Identifies duplicated pixels often used to cover up original text.
- **Error Level Analysis (ELA)**: Detects digital manipulation by highlighting differences in compression levels.
- **Metadata Forensics**: Reveals hidden software traces (Photoshop, GIMP) and hardware signatures.

### 📊 Advanced Visualizations
- **Detailed Heatmaps**: Visual overlays showing exactly where the AI is suspicious.
- **Forensic PDF Reports**: Professional, high-fidelity reports containing all analysis findings and annotated images.
- **Confidence Scoring**: A weighted authenticity score derived from all forensic modules.

---

## 🛠️ Tech Stack

### Backend (The Forensic Engine)
- **Framework**: FastAPI (Python 3.13+)
- **Computer Vision**: OpenCV, Scikit-Image
- **OCR/ML**: Tesseract-OCR, python-doctr (Deep Learning), NumPy 2.x
- **Document Handling**: PyMuPDF (Fitz), Pillow, fpdf2
- **Forensics**: Custom ELA and stroke variance algorithms

### Frontend (The Intelligence Dashboard)
- **Framework**: React 19 + Vite
- **Styling**: Vanilla CSS with a high-end **Glassmorphic Cyber-Intelligence** aesthetic
- **Interactions**: Framer-style transitions and dynamic SVG score gauges

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.13+**
- **Node.js (v18+)**
- **Tesseract-OCR**: [Download and install](https://github.com/UB-Mannheim/tesseract/wiki) for your OS.

### Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
*Backend runs on `http://localhost:8000`.*

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Frontend runs on `http://localhost:5173`.*

---

## 📖 Usage
1. **Upload**: Drag and drop any certificate (PDF or Image) into the dashboard.
2. **Analyze**: The system will run the forensic pipeline (Preprocessing -> OCR -> Stamps -> Signatures -> Clones -> Metadata).
3. **Inspect**: Use the tabs to toggle between various heatmaps and structural findings.
4. **Export**: Download a professional PDF report for legal or administrative verification.

---

## 🏛️ Project Information
- **Developed by**: [Ameya1605](https://github.com/Ameya1605)
- **Institution**: G.C.O.E., Nagpur
- **Purpose**: Academic Prototype for Document Forensics & Security

---

## ⚖️ Disclaimer
*Authentify is a forensic tool intended for aid in manual document verification. While its AI models are highly accurate, results should be cross-verified by human experts for high-stakes decisions.*
