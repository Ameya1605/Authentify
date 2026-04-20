import { useState, useCallback, useRef, useEffect } from 'react'
import './index.css'

const API_URL = 'http://localhost:8000'

const MODULE_ICONS = {
  'OCR Text Extraction': '📝',
  'Stamp/Watermark Verification': '🔏',
  'Signature Verification': '✍️',
  'Clone Region Detection': '🔍',
  'Metadata Analysis': '📊',
}

const HEATMAP_KEYS = {
  'OCR Text Extraction': 'ocr',
  'Stamp/Watermark Verification': 'stamp',
  'Signature Verification': 'signature',
  'Clone Region Detection': 'clone',
}

function getScoreColor(score) {
  if (score >= 0.8) return 'var(--status-authentic)'
  if (score >= 0.6) return 'var(--status-info)'
  if (score >= 0.4) return 'var(--status-warning)'
  return 'var(--status-danger)'
}

function getVerdictClass(verdict) {
  if (verdict === 'AUTHENTIC' || verdict === 'LIKELY AUTHENTIC') return 'verdict-authentic'
  if (verdict === 'SUSPICIOUS') return 'verdict-warning'
  return 'verdict-danger'
}

// ===== Score Gauge Component =====
function ScoreGauge({ score, color }) {
  const circumference = 2 * Math.PI * 85
  const [offset, setOffset] = useState(circumference)
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setOffset(circumference - (score * circumference))
    }, 300)
    return () => clearTimeout(timer)
  }, [score, circumference])

  return (
    <div className="score-gauge">
      <svg viewBox="0 0 200 200">
        <circle className="score-gauge-bg" cx="100" cy="100" r="85" />
        <circle
          className="score-gauge-fill"
          cx="100" cy="100" r="85"
          style={{
            stroke: color,
            strokeDasharray: circumference,
            strokeDashoffset: offset,
          }}
        />
      </svg>
      <div className="score-gauge-center">
        <span className="score-value" style={{ color }}>
          {Math.round(score * 100)}
        </span>
        <span className="score-label">Authenticity</span>
      </div>
    </div>
  )
}

// ===== Processing Overlay =====
function ProcessingOverlay({ step, progress }) {
  return (
    <div className="processing-overlay">
      <div className="processing-card">
        <div className="processing-spinner" />
        <h3 className="processing-title">Analyzing Document</h3>
        <p className="processing-step">{step}</p>
        <div className="processing-progress">
          <div
            className="processing-progress-bar"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  )
}

// ===== Main App =====
export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState('')
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [activeModule, setActiveModule] = useState(0)
  const [activeTab, setActiveTab] = useState('original')
  const fileInputRef = useRef(null)

  const handleFileSelect = useCallback((selectedFile) => {
    if (!selectedFile) return
    if (selectedFile.size > 20 * 1024 * 1024) {
      setError('File too large. Maximum size is 20MB.')
      return
    }
    setFile(selectedFile)
    setError(null)
    setResults(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(selectedFile)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragActive(false)
    const droppedFile = e.dataTransfer.files[0]
    handleFileSelect(droppedFile)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragActive(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setDragActive(false)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResults(null)

    const steps = [
      { text: 'Preprocessing image...', progress: 10 },
      { text: 'Running OCR text extraction...', progress: 25 },
      { text: 'Verifying stamps & watermarks...', progress: 40 },
      { text: 'Analyzing signatures...', progress: 55 },
      { text: 'Detecting cloned regions...', progress: 70 },
      { text: 'Extracting metadata...', progress: 85 },
      { text: 'Computing authenticity score...', progress: 95 },
    ]

    let stepIndex = 0
    const stepInterval = setInterval(() => {
      if (stepIndex < steps.length) {
        setLoadingStep(steps[stepIndex].text)
        setLoadingProgress(steps[stepIndex].progress)
        stepIndex++
      }
    }, 800)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error: ${response.status}`)
      }

      const data = await response.json()
      setResults(data)
      setActiveModule(0)
      setActiveTab('original')
    } catch (err) {
      setError(err.message || 'Failed to analyze document. Please ensure the backend is running.')
    } finally {
      clearInterval(stepInterval)
      setLoading(false)
      setLoadingProgress(0)
    }
  }, [file])

  const removeFile = useCallback(() => {
    setFile(null)
    setPreview(null)
    setResults(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  const downloadReport = useCallback(() => {
    if (!results) return
    const report = {
      analysis_id: results.analysis_id,
      filename: results.filename,
      timestamp: new Date().toISOString(),
      authenticity: results.authenticity,
      modules: results.modules,
      findings: results.findings,
      metadata: results.metadata,
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `authentify-report-${results.analysis_id.slice(0, 8)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [results])

  const downloadPdfReport = useCallback(async () => {
    if (!results) return
    try {
      const response = await fetch(`${API_URL}/api/report/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(results),
      })
      if (!response.ok) throw new Error('Failed to generate PDF')
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `authentify-report-${results.analysis_id.slice(0, 8)}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError('Could not download PDF report. ' + err.message)
    }
  }, [results])

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const tabs = [
    { key: 'original', label: 'Original', icon: '🖼️' },
    { key: 'annotated', label: 'Annotated', icon: '🏷️' },
    { key: 'ocr', label: 'OCR Heatmap', icon: '📝' },
    { key: 'stamp', label: 'Stamp Heatmap', icon: '🔏' },
    { key: 'signature', label: 'Signature Heatmap', icon: '✍️' },
    { key: 'clone', label: 'Clone Heatmap', icon: '🔍' },
    { key: 'ela', label: 'ELA Map', icon: '⚡' },
  ]

  const getActiveImage = () => {
    if (!results) return null
    if (activeTab === 'original') return results.images?.original
    if (activeTab === 'annotated') return results.images?.annotated
    return results.heatmaps?.[activeTab]
  }

  const getTabLabel = () => {
    const tab = tabs.find(t => t.key === activeTab)
    return tab ? `${tab.icon} ${tab.label}` : ''
  }

  return (
    <>
      <div className="app-bg" />

      {/* Navbar */}
      <nav className="navbar" id="navbar">
        <div className="navbar-inner">
          <div className="navbar-brand">
            <div className="brand-icon">🛡️</div>
            <span className="brand-text">AUTHENTIFY</span>
            <span className="brand-badge">AI POWERED</span>
          </div>
          <div className="navbar-status">
            <span className="status-dot" />
            <span>System Online</span>
          </div>
        </div>
      </nav>

      <main className="main-container" id="main-content">
        {/* Hero */}
        <section className="hero-section" id="hero">
          <h1 className="hero-title">
            Detect <span className="hero-title-gradient">Forged Documents</span> Instantly
          </h1>
          <p className="hero-subtitle">
            Advanced AI-powered certificate verification system. Upload any document and get
            comprehensive authenticity analysis with heatmap visualizations, signature verification,
            and tampering detection.
          </p>
          <div className="hero-badges">
            <div className="hero-badge">
              <span className="hero-badge-icon">📝</span> OCR Extraction
            </div>
            <div className="hero-badge">
              <span className="hero-badge-icon">🔏</span> Stamp Verification
            </div>
            <div className="hero-badge">
              <span className="hero-badge-icon">✍️</span> Signature Analysis
            </div>
            <div className="hero-badge">
              <span className="hero-badge-icon">🔍</span> Clone Detection
            </div>
            <div className="hero-badge">
              <span className="hero-badge-icon">📊</span> Metadata Forensics
            </div>
          </div>
        </section>

        {/* Error Banner */}
        {error && (
          <div className="error-banner" id="error-banner">
            <span className="error-icon">⚠️</span>
            <span className="error-text">{error}</span>
            <button className="error-dismiss" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}

        {/* Upload Section */}
        <section className="upload-section" id="upload-section">
          <div
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            id="upload-zone"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <div className="upload-zone-content">
              <div className="upload-icon">
                {dragActive ? '📥' : '📄'}
              </div>
              <h3 className="upload-title">
                {dragActive ? 'Drop your document here' : 'Upload Document for Analysis'}
              </h3>
              <p className="upload-subtitle">
                Drag & drop a certificate or click to browse
              </p>
              <div className="upload-formats">
                {['PDF', 'JPEG', 'PNG', 'DOCX', 'Any Format'].map(f => (
                  <span key={f} className="format-tag">{f}</span>
                ))}
              </div>
            </div>
            <input
              ref={fileInputRef}
              className="upload-input"
              type="file"
              onChange={(e) => handleFileSelect(e.target.files[0])}
              id="file-input"
            />
          </div>

          {/* File Preview */}
          {file && preview && (
            <div className="file-preview" id="file-preview">
              {file.type.startsWith('image/') ? (
                <img src={preview} alt="Preview" className="file-preview-thumb" />
              ) : (
                <div className="file-preview-thumb" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '32px', background: 'var(--bg-primary)' }}>
                  📄
                </div>
              )}
              <div className="file-preview-info">
                <div className="file-preview-name">{file.name}</div>
                <div className="file-preview-size">
                  {formatFileSize(file.size)} · {file.type}
                </div>
              </div>
              <button
                className="file-preview-remove"
                onClick={(e) => { e.stopPropagation(); removeFile() }}
                title="Remove file"
                id="remove-file-btn"
              >
                ✕
              </button>
            </div>
          )}

          {/* Analyze Button */}
          {file && !results && (
            <div style={{ textAlign: 'center' }}>
              <button
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={loading}
                id="analyze-btn"
              >
                <span className="analyze-btn-icon">🔬</span>
                {loading ? 'Analyzing...' : 'Analyze Document'}
              </button>
            </div>
          )}
        </section>

        {/* Processing Overlay */}
        {loading && (
          <ProcessingOverlay step={loadingStep} progress={loadingProgress} />
        )}

        {/* Results */}
        {results && (
          <section className="results-section" id="results-section">
            {/* Action Bar */}
            <div className="action-bar">
              <button className="action-btn action-btn-primary" onClick={downloadReport} id="download-report-btn">
                📊 Download JSON
              </button>
              <button className="action-btn action-btn-primary" onClick={downloadPdfReport} id="download-pdf-btn" style={{ background: 'var(--accent-gradient-alt)', marginLeft: '10px' }}>
                📄 Download PDF Report
              </button>
              <button className="action-btn" onClick={removeFile} id="new-analysis-btn">
                🔄 New Analysis
              </button>
              <button className="action-btn" style={{ marginLeft: 'auto' }}>
                ⏱️ {results.processing_time_seconds}s
              </button>
            </div>

            {/* Score Overview */}
            <div className="score-overview" id="score-overview">
              <div className="score-gauge-container">
                <ScoreGauge
                  score={results.authenticity.score}
                  color={getScoreColor(results.authenticity.score)}
                />
              </div>
              <div className="score-details">
                <div
                  className={`verdict-badge ${getVerdictClass(results.authenticity.verdict)}`}
                  id="verdict-badge"
                >
                  {results.authenticity.verdict === 'AUTHENTIC' || results.authenticity.verdict === 'LIKELY AUTHENTIC'
                    ? '✅' : results.authenticity.verdict === 'SUSPICIOUS' ? '⚠️' : '🚫'}
                  {' '}{results.authenticity.verdict}
                </div>
                <p className="verdict-description">
                  {results.authenticity.verdict_description}
                </p>
                <div className="score-meta-grid">
                  <div className="score-meta-item">
                    <div className="score-meta-value" style={{ color: getScoreColor(results.authenticity.score) }}>
                      {results.authenticity.percentage}%
                    </div>
                    <div className="score-meta-label">Score</div>
                  </div>
                  <div className="score-meta-item">
                    <div className="score-meta-value" style={{ color: getScoreColor(results.authenticity.confidence) }}>
                      {Math.round(results.authenticity.confidence * 100)}%
                    </div>
                    <div className="score-meta-label">Confidence</div>
                  </div>
                  <div className="score-meta-item">
                    <div className="score-meta-value" style={{
                      color: results.authenticity.risk_level === 'LOW'
                        ? 'var(--status-authentic)'
                        : results.authenticity.risk_level === 'MODERATE'
                          ? 'var(--status-warning)'
                          : 'var(--status-danger)',
                    }}>
                      {results.authenticity.risk_level}
                    </div>
                    <div className="score-meta-label">Risk Level</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Module Scores */}
            <div className="modules-bar" id="modules-bar">
              {results.modules.map((mod, idx) => (
                <div
                  key={mod.module}
                  className={`module-score-card ${activeModule === idx ? 'active' : ''}`}
                  onClick={() => {
                    setActiveModule(idx)
                    const heatmapKey = HEATMAP_KEYS[mod.module]
                    if (heatmapKey) setActiveTab(heatmapKey)
                  }}
                  id={`module-card-${idx}`}
                >
                  <div className="module-score-icon">
                    {MODULE_ICONS[mod.module] || '📋'}
                  </div>
                  <div className="module-score-value" style={{ color: getScoreColor(mod.score) }}>
                    {Math.round(mod.score * 100)}%
                  </div>
                  <div className="module-score-name">
                    {mod.module.split('/')[0]}
                  </div>
                  <div className="module-score-bar">
                    <div
                      className="module-score-bar-fill"
                      style={{
                        width: `${mod.score * 100}%`,
                        background: getScoreColor(mod.score),
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Visual Analysis Tabs */}
            <div className="analysis-panel" id="analysis-panel">
              <div className="analysis-tabs">
                {tabs.map(tab => (
                  <button
                    key={tab.key}
                    className={`analysis-tab ${activeTab === tab.key ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab.key)}
                    id={`tab-${tab.key}`}
                  >
                    <span className="analysis-tab-icon">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </div>
              <div className="analysis-content">
                <div className="analysis-image-container">
                  {getActiveImage() && (
                    <>
                      <img
                        className="analysis-image"
                        src={`data:image/jpeg;base64,${getActiveImage()}`}
                        alt={getTabLabel()}
                        id="analysis-image"
                      />
                      <div className="analysis-image-label">{getTabLabel()}</div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Findings */}
            <div className="findings-section" id="findings-section">
              <div className="findings-header">
                <span style={{ fontSize: '1.3rem' }}>🔎</span>
                <h3 className="findings-title">Analysis Findings</h3>
                <span className="findings-count">{results.findings.length} items</span>
              </div>
              {results.findings.map((finding, idx) => {
                const icon = finding.startsWith('✅') ? '✅'
                  : finding.startsWith('🔴') ? '🔴'
                  : finding.startsWith('🟡') ? '🟡'
                  : finding.startsWith('⚠️') ? '⚠️' : '📌'
                const text = finding.replace(/^[✅🔴🟡⚠️📌]\s*/, '')
                return (
                  <div key={idx} className="finding-item" id={`finding-${idx}`}>
                    <span className="finding-icon">{icon}</span>
                    <span className="finding-text">{text}</span>
                  </div>
                )
              })}
            </div>

            {/* Metadata */}
            {results.metadata && (
              <div className="findings-section" id="metadata-section">
                <div className="findings-header">
                  <span style={{ fontSize: '1.3rem' }}>📊</span>
                  <h3 className="findings-title">Document Metadata</h3>
                </div>
                <div className="metadata-grid">
                  {results.metadata.file_info && (
                    <div className="metadata-card">
                      <div className="metadata-card-title">File Information</div>
                      <div className="metadata-card-value">
                        {results.metadata.file_info.format}
                      </div>
                      <div className="metadata-card-detail">
                        {results.metadata.file_info.size_kb} KB
                        {results.metadata.file_info.format_consistent
                          ? ' · Format Valid ✅'
                          : ' · Format Issue ⚠️'}
                      </div>
                    </div>
                  )}
                  {results.metadata.image_info && (
                    <div className="metadata-card">
                      <div className="metadata-card-title">Image Properties</div>
                      <div className="metadata-card-value">
                        {results.metadata.image_info.dimensions?.width} × {results.metadata.image_info.dimensions?.height}
                      </div>
                      <div className="metadata-card-detail">
                        {results.metadata.image_info.mode} · DPI: {results.metadata.image_info.dpi}
                      </div>
                    </div>
                  )}
                  {results.metadata.software_analysis && (
                    <div className="metadata-card">
                      <div className="metadata-card-title">Software Analysis</div>
                      <div className="metadata-card-value" style={{
                        color: results.metadata.software_analysis.editing_detected
                          ? 'var(--status-danger)' : 'var(--status-authentic)',
                      }}>
                        {results.metadata.software_analysis.editing_detected
                          ? '⚠️ Editing Detected' : '✅ No Editing Tools'}
                      </div>
                      <div className="metadata-card-detail">
                        {results.metadata.software_analysis.detected_software?.join(', ') || 'No software traces'}
                      </div>
                    </div>
                  )}
                  {results.metadata.compression && (
                    <div className="metadata-card">
                      <div className="metadata-card-title">Compression Analysis</div>
                      <div className="metadata-card-value">
                        Quality: {results.metadata.compression.estimated_quality}%
                      </div>
                      <div className="metadata-card-detail">
                        {results.metadata.compression.double_compression
                          ? '⚠️ Double compression detected'
                          : '✅ Normal compression'}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="footer" id="footer">
        <p>
          <strong>AUTHENTIFY</strong> — AI-Powered Fake Document Detection System
        </p>
        <p style={{ marginTop: '0.5rem' }}>
          G.C.O.E., Nagpur · Built with Computer Vision & Machine Learning
        </p>
      </footer>
    </>
  )
}
