"""
Modernized Web Studio UI for chunkmirage dynamic virtual Zarr pipelines.
Pre-configured with Human Organ Atlas (HOA) / UCL HiP-CT Heart datasets,
camera coordinates, multiscale OME-Zarr pipelines, and live telemetry.
Includes a dedicated Passthrough mode for zero-overhead connectivity testing.
"""

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Organ Atlas - Virtual Zarr Compute Studio (chunkmirage)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: #111827;
            --bg-card: rgba(17, 24, 39, 0.9);
            --bg-input: #1e293b;
            --border: rgba(255, 255, 255, 0.08);
            --border-focus: #6366f1;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --accent: #06b6d4;
            --accent-glow: rgba(6, 182, 212, 0.35);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --radius-lg: 14px;
            --radius-md: 8px;
            --radius-sm: 5px;
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-display: 'Outfit', var(--font-sans);
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html, body {
            width: 100%;
            overflow-x: hidden;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-sans);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 75%, rgba(6, 182, 212, 0.1) 0%, transparent 45%);
            background-attachment: fixed;
            line-height: 1.5;
        }

        header {
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(16px);
            background: rgba(11, 15, 25, 0.85);
            position: sticky;
            top: 0;
            z-index: 50;
            padding: 0.85rem 1.75rem;
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }

        .brand {
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 0.75rem;
            min-width: 0;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            flex-shrink: 0;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 16px var(--primary-glow);
        }

        .brand-icon svg { width: 20px; height: 20px; fill: white; }

        .brand-text { min-width: 0; }

        .brand-text h1 {
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #fff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .brand-text p {
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 500;
        }

        .nav-links {
            display: flex;
            flex-direction: row;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .nav-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.4rem 0.8rem;
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.35rem;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .nav-btn:hover {
            color: #fff;
            border-color: var(--accent);
            background: rgba(6, 182, 212, 0.1);
        }

        .main-container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 1.5rem;
            width: 100%;
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 1.5rem;
            flex: 1;
            min-width: 0;
        }

        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.35rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            gap: 1.15rem;
            flex: 1 1 540px;
            min-width: 0;
            max-width: 100%;
        }

        .panel-title {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.75rem;
        }

        .badge {
            font-size: 0.7rem;
            font-family: var(--font-mono);
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            font-weight: 500;
            background: rgba(99, 102, 241, 0.15);
            color: var(--primary);
            border: 1px solid rgba(99, 102, 241, 0.3);
            white-space: nowrap;
        }

        .badge.accent {
            background: rgba(6, 182, 212, 0.15);
            color: var(--accent);
            border-color: rgba(6, 182, 212, 0.3);
        }

        .badge.success {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border-color: rgba(16, 185, 129, 0.3);
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            width: 100%;
            min-width: 0;
        }

        .flex-row {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 0.75rem;
            width: 100%;
            min-width: 0;
        }

        .flex-col {
            flex: 1 1 160px;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        label {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-muted);
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        input[type="text"], input[type="number"], select {
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.55rem 0.75rem;
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            font-family: inherit;
            outline: none;
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        .presets {
            display: flex;
            flex-direction: row;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.25rem;
            width: 100%;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.72rem;
            padding: 0.25rem 0.55rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }

        .preset-btn:hover {
            background: rgba(99, 102, 241, 0.18);
            color: #fff;
            border-color: var(--primary);
        }

        .preset-btn.active {
            background: rgba(6, 182, 212, 0.2);
            color: #fff;
            border-color: var(--accent);
        }

        /* Pipeline Steps Container */
        .pipeline-container {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            border: 1px dashed var(--border);
            padding: 0.85rem;
            border-radius: var(--radius-md);
            background: rgba(0, 0, 0, 0.2);
            width: 100%;
            min-width: 0;
        }

        .step-card {
            background: #162032;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.85rem;
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            width: 100%;
            min-width: 0;
        }

        .step-header {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .step-num {
            font-size: 0.75rem;
            font-family: var(--font-mono);
            font-weight: 600;
            color: var(--accent);
        }

        .step-remove {
            background: transparent;
            border: none;
            color: var(--danger);
            cursor: pointer;
            font-size: 0.82rem;
            opacity: 0.75;
            transition: opacity 0.2s;
        }

        .step-remove:hover { opacity: 1; }

        .step-grid {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 0.6rem;
            width: 100%;
            min-width: 0;
        }

        .step-grid .form-group {
            flex: 1 1 130px;
            min-width: 0;
        }

        .add-step-btn {
            background: rgba(99, 102, 241, 0.1);
            border: 1px dashed var(--primary);
            color: var(--primary);
            padding: 0.55rem;
            border-radius: var(--radius-md);
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            width: 100%;
        }

        .add-step-btn:hover {
            background: rgba(99, 102, 241, 0.2);
            color: #fff;
        }

        .btn-launch {
            background: linear-gradient(135deg, var(--primary), #4f46e5);
            color: #fff;
            border: none;
            padding: 0.85rem 1.25rem;
            border-radius: var(--radius-md);
            font-family: var(--font-display);
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            text-decoration: none;
            box-shadow: 0 4px 20px var(--primary-glow);
            transition: transform 0.15s, box-shadow 0.15s;
            width: 100%;
        }

        .btn-launch:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 24px var(--primary-glow);
        }

        .output-box {
            background: #0d121f;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            width: 100%;
            min-width: 0;
        }

        .output-header {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            color: var(--text-muted);
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        .output-code {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: var(--accent);
            word-break: break-all;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
            user-select: all;
            background: rgba(0, 0, 0, 0.3);
            padding: 0.5rem;
            border-radius: var(--radius-sm);
            max-height: 85px;
            overflow-y: auto;
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
        }

        .copy-btn {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.2rem 0.5rem;
            border-radius: var(--radius-sm);
            font-size: 0.72rem;
            cursor: pointer;
            transition: all 0.15s;
        }

        .copy-btn:hover {
            color: #fff;
            border-color: var(--accent);
        }

        .stats-grid {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 0.6rem;
            width: 100%;
            min-width: 0;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.65rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1 1 100px;
            min-width: 0;
        }

        .stat-val {
            font-family: var(--font-mono);
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
        }

        .stat-lbl {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
            text-align: center;
        }

        #toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #1e293b;
            color: #fff;
            padding: 0.75rem 1.25rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--accent);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            font-size: 0.85rem;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.3s;
            pointer-events: none;
            z-index: 100;
        }

        #toast.show {
            opacity: 1;
            transform: translateY(0);
        }

        /* Tooltip Infrastructure */
        .has-tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            cursor: help;
        }

        .has-tooltip .tooltip-text {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            z-index: 9999;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%) translateY(4px);
            background: #131b2e;
            color: #f1f5f9;
            font-size: 0.74rem;
            font-weight: 400;
            line-height: 1.45;
            padding: 0.6rem 0.85rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-focus);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.75);
            width: max-content;
            max-width: 320px;
            white-space: normal;
            pointer-events: none;
            transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s;
        }

        .has-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }

        .has-tooltip.tooltip-bottom .tooltip-text {
            bottom: auto;
            top: calc(100% + 8px);
            transform: translateX(-50%) translateY(-4px);
        }

        .has-tooltip.tooltip-bottom:hover .tooltip-text {
            transform: translateX(-50%) translateY(0);
        }

        .has-tooltip.tooltip-left .tooltip-text {
            left: auto;
            right: 0;
            transform: translateY(4px);
        }

        .has-tooltip.tooltip-left:hover .tooltip-text {
            transform: translateY(0);
        }

        .has-tooltip.tooltip-bottom.tooltip-left .tooltip-text {
            transform: translateY(-4px);
        }

        .has-tooltip.tooltip-bottom.tooltip-left:hover .tooltip-text {
            transform: translateY(0);
        }

        .has-tooltip.tooltip-right .tooltip-text {
            left: 0;
            transform: translateY(4px);
        }

        .has-tooltip.tooltip-right:hover .tooltip-text {
            transform: translateY(0);
        }

        .info-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: rgba(99, 102, 241, 0.25);
            color: var(--accent);
            font-size: 0.65rem;
            font-family: var(--font-mono);
            font-weight: 700;
            margin-left: 6px;
            vertical-align: middle;
            font-style: normal;
        }

        pre {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: #e2e8f0;
            background: #090d16;
            padding: 0.75rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            overflow-x: auto;
            max-height: 200px;
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
            word-break: break-all;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>
            </div>
            <div class="brand-text">
                <h1>Human Organ Atlas Virtual Compute Studio</h1>
                <p>Powered by <span class="has-tooltip tooltip-bottom" style="color: var(--accent); border-bottom: 1px dotted var(--accent); font-weight: 500;">chunkmirage<span class="tooltip-text"><strong>chunkmirage:</strong> Framework for on-demand virtual Zarr computation. Transforms arbitrarily large remote Zarr datasets on the fly without materializing intermediate volumes to disk.</span></span> &bull; Multi-Stage Volumetric Pipelines</p>
            </div>
        </div>
        <div class="nav-links">
            <div class="has-tooltip tooltip-bottom tooltip-left">
                <a href="/ui" target="_blank" class="nav-btn">
                    <span>⚡ Chunkmirage Control</span>
                </a>
                <div class="tooltip-text">
                    <strong style="color: var(--accent);">Chunkmirage Live Dashboard (/ui):</strong><br>
                    Built-in control dashboard from chunkmirage.<br>
                    &bull; <strong>Dataset Registry:</strong> Lists all active virtual pipelines on this server.<br>
                    &bull; <strong>Cryptographic Digests (@{digest}):</strong> Displays content hashes for deterministic caching.<br>
                    &bull; <strong>Live Event Stream:</strong> Uses Server-Sent Events (SSE) to update in real time when any client or worker registers a pipeline.
                </div>
            </div>
            <div class="has-tooltip tooltip-bottom tooltip-left">
                <a href="/_stats" target="_blank" class="nav-btn">
                    <span>📊 Metrics</span>
                </a>
                <div class="tooltip-text">
                    <strong style="color: var(--accent);">Live Telemetry Endpoint (/_stats):</strong><br>
                    Real-time JSON endpoint reporting server internals:<br>
                    &bull; <strong>Memory Budget:</strong> Bytes currently held in LRU chunk cache.<br>
                    &bull; <strong>Cache Performance:</strong> Hit/miss ratio and total chunk count.<br>
                    &bull; <strong>Engines:</strong> Active compute backends (Local, Dask, Workstation).
                </div>
            </div>
        </div>
    </header>

    <div class="main-container">
        <!-- Configuration Left Panel -->
        <div class="panel">
            <div class="panel-title">
                <span>Dataset & Pipeline Configuration</span>
                <span class="badge has-tooltip tooltip-left" id="halo-total-badge">
                    Cumulative Halo: 0 px
                    <span class="tooltip-text">
                        <strong>Automated Halo Calculation:</strong><br>
                        Chunkmirage automatically calculates spatial padding required for each filter (e.g. ⌈3σ⌉ for Hessian filters) and propagates them. Chunks are retrieved with this halo and cropped after compute to ensure continuous boundaries with zero edge artifacts.
                    </span>
                </span>
            </div>

            <!-- Source Dataset Input with HOA Presets -->
            <div class="form-group">
                <label>
                    <span class="has-tooltip tooltip-right">
                        <span>Source Volume URL (Human Organ Atlas / HiP-CT / S3)</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Virtual Chunked Source:</strong><br>
                            Streamed lazily via chunkmirage & worldarray. Supports HTTPS, Google Cloud Storage (gs://), AWS S3 (s3://), or local files. Chunks are retrieved on demand without downloading the full dataset.
                        </span>
                    </span>
                    <span id="source-format-hint" style="color: var(--accent);">Zarr v3 / OME-Zarr</span>
                </label>
                <input type="text" id="src_url" value="https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr">
                <div class="presets">
                    <button class="preset-btn active" onclick="setPreset('hoa_heart_https')">❤️ UCL HiP-CT Heart (HTTPS)</button>
                    <button class="preset-btn" onclick="setPreset('hoa_heart_gs')">UCL HiP-CT Heart (gs://)</button>
                    <button class="preset-btn" onclick="setPreset('flyem')">Janelia FlyEM (S3)</button>
                    <button class="preset-btn" onclick="setPreset('synthetic')">Procedural 3D Blobs (Synthetic)</button>
                </div>
            </div>

            <!-- Pipeline Mode Quick Toggles -->
            <div class="form-group">
                <label>
                    <span>Quick Pipeline Presets</span>
                    <span style="font-size: 0.72rem; color: var(--accent);">One-click pipeline templates</span>
                </label>
                <div class="presets">
                    <button class="preset-btn active" id="btn-mode-sato" onclick="setPipelineMode('sato')">🌟 Sato Tubeness (Default)</button>
                    <button class="preset-btn" id="btn-mode-frangi" onclick="setPipelineMode('frangi')">🩸 Frangi Vesselness</button>
                    <button class="preset-btn" id="btn-mode-segment" onclick="setPipelineMode('segment')">🔬 Segmentation (Sato + Li)</button>
                    <button class="preset-btn" id="btn-mode-sobel" onclick="setPipelineMode('sobel')">📐 Sobel 3D Boundaries</button>
                    <button class="preset-btn" id="btn-mode-denoised-sato" onclick="setPipelineMode('denoised-sato')">✨ Denoised (Median + Sato)</button>
                    <button class="preset-btn" id="btn-mode-tophat" onclick="setPipelineMode('tophat')">🔍 White Top-Hat (Micro-Vessels)</button>
                    <button class="preset-btn" id="btn-mode-unsharp" onclick="setPipelineMode('unsharp')">⚡ Unsharp Mask (Sharpening)</button>
                    <button class="preset-btn" id="btn-mode-ctwindow" onclick="setPipelineMode('ctwindow')">🎛️ CT Window (0-255 uint8)</button>
                    <button class="preset-btn" id="btn-mode-passthrough" onclick="setPipelineMode('passthrough')">🚀 Passthrough (Raw CT Test)</button>
                </div>
            </div>

            <!-- Pipeline Operations Stack -->
            <div class="form-group">
                <label>
                    <span class="has-tooltip tooltip-right">
                        <span>Processing Pipeline Stages (Fused Execution)</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Fused Pipeline Engine:</strong><br>
                            Chunkmirage dynamically chains operations in memory. Chunks flow through stages without saving intermediate arrays to disk, enabling instant visual exploration of terabyte-scale volumes.
                        </span>
                    </span>
                    <span style="font-size: 0.72rem; color: var(--text-muted);">Shared per-stage LRU caching</span>
                </label>

                <div class="pipeline-container" id="pipeline-steps">
                    <!-- Dynamic Step Cards inserted here -->
                </div>

                <button class="add-step-btn" onclick="addStep()">
                    <span>+ Add Processing Operation (e.g. Sato, Frangi, Sobel, Median, Li, Otsu)</span>
                </button>
            </div>

            <!-- Intensity Windowing (CT Window for HiP-CT / Vessel Response) -->
            <div class="flex-row">
                <div class="flex-col">
                    <label>
                        <span>Display Range Min (vmin)</span>
                        <span style="font-size: 0.7rem; color: var(--text-muted);">Shader min</span>
                    </label>
                    <input type="number" id="val_vmin" value="0" onchange="updateUI()">
                </div>
                <div class="flex-col">
                    <label>
                        <span>Display Range Max (vmax)</span>
                        <span style="font-size: 0.7rem; color: var(--text-muted);">Shader max</span>
                    </label>
                    <input type="number" id="val_vmax" value="500" onchange="updateUI()">
                </div>
            </div>

            <!-- Output Settings -->
            <div class="flex-row">
                <div class="flex-col">
                    <label class="has-tooltip tooltip-right">
                        <span>Output Format</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Multi-Protocol Frontends:</strong><br>
                            Chunkmirage translates virtual compute pipelines into OME-Zarr v3 (zarr3://), Neuroglancer Precomputed, N5, or Zarr v2 on-the-fly, allowing any viewer or client to connect seamlessly.
                        </span>
                    </label>
                    <select id="output_format" onchange="updateUI()">
                        <option value="zarr3" selected>OME-Zarr v3 (zarr3://)</option>
                        <option value="precomputed">Neuroglancer Precomputed</option>
                        <option value="n5">N5 Format</option>
                        <option value="zarr">Zarr v2</option>
                    </select>
                </div>
                <div class="flex-col">
                    <label class="has-tooltip">
                        <span>Compute Backend</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Server Compute Layer:</strong><br>
                            Switch between Local (multi-threaded NumPy/SciPy in-process), Dask (distributed chunk graph execution), and Workstation (remote execution) backends.
                        </span>
                    </label>
                    <select id="compute_backend" onchange="updateUI()">
                        <option value="local" selected>Local (NumPy/SciPy)</option>
                        <option value="dask">Dask (Threaded/Distributed)</option>
                        <option value="workstation">Workstation (Remote)</option>
                    </select>
                </div>
                <div class="flex-col">
                    <label class="has-tooltip tooltip-left">
                        <span>Target Data Type</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>On-the-Fly Type Casting:</strong><br>
                            Converts voxel output data type during compression. Selecting uint8 reduces network payload size by 4x for high-throughput 3D volume navigation.
                        </span>
                    </label>
                    <select id="target_dtype" onchange="updateUI()">
                        <option value="float32" selected>float32 (Native Filter Output)</option>
                        <option value="original">Original (e.g. uint16 for Raw Heart)</option>
                        <option value="uint8">uint8 (0 - 255)</option>
                        <option value="uint16">uint16 (0 - 65535)</option>
                    </select>
                </div>
            </div>

            <!-- Host & Client Options -->
            <div class="flex-row">
                <div class="flex-col">
                    <label>Proxy Server Address</label>
                    <input type="text" id="proxy_host" value="" oninput="updateUI()" placeholder="Auto-detected origin">
                </div>
                <div class="flex-col">
                    <label>Neuroglancer Client</label>
                    <input type="text" id="ng_client" value="" oninput="updateUI()" placeholder="Auto-detected /viewer">
                    <div class="presets">
                        <button class="preset-btn" onclick="setNgClient('self')">Self-Hosted (/viewer)</button>
                        <button class="preset-btn" onclick="setNgClient('https://neuroglancer-demo.appspot.com')">Official Demo</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Launch & Preview Right Panel -->
        <div class="panel">
            <div class="panel-title">
                <span>Interactive Viewer Launch</span>
                <span class="badge success has-tooltip tooltip-left" id="engine-status">
                    Zero Disk Writes
                    <span class="tooltip-text">
                        <strong>Pure Virtual Compute Proxy:</strong><br>
                        All chunks are processed on the fly in memory and streamed directly to the browser with zero disk footprint.
                    </span>
                </span>
            </div>

            <!-- 1-Click Launch Button -->
            <a id="btn-launch-ng" href="#" target="_blank" class="btn-launch">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
                <span id="btn-launch-label">Open in Neuroglancer (Heart 4-Panel 3D)</span>
            </a>

            <!-- Virtual Zarr Source URL Box -->
            <div class="output-box">
                <div class="output-header">
                    <span class="has-tooltip tooltip-right">
                        <span>Virtual Layer Source URL (Neuroglancer format)</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Virtual Zarr Endpoint:</strong><br>
                            The streaming URL containing the format and pipeline identifier. Paste directly into any Neuroglancer instance or Zarr v3 reader.
                        </span>
                    </span>
                    <button class="copy-btn" onclick="copyText('out-source-url')">Copy URL</button>
                </div>
                <div class="output-code" id="out-source-url">Generating...</div>
            </div>

            <!-- Pipeline Payload Box -->
            <div class="output-box">
                <div class="output-header">
                    <span class="has-tooltip tooltip-right">
                        <span>Encoded Spec Payload (Stateless & Cache-Busted)</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Stateless Deflate Payload:</strong><br>
                            Compresses the entire pipeline JSON into a URL-safe token (Deflate-Raw + Base64URL). No database or server state needed. Chunkmirage uses content hashing (@{digest}) to bust browser cache whenever parameters change.
                        </span>
                    </span>
                    <button class="copy-btn" onclick="copyText('out-payload')">Copy Payload</button>
                </div>
                <div class="output-code" id="out-payload">...</div>
            </div>

            <!-- Live Cache Telemetry -->
            <div class="form-group">
                <label>
                    <span class="has-tooltip tooltip-right">
                        <span>Shared Cache Telemetry</span>
                        <span class="info-icon">i</span>
                        <span class="tooltip-text">
                            <strong>Hierarchical Chunk Cache:</strong><br>
                            Chunkmirage memoizes chunks at each pipeline stage. Modifying a downstream step (like a threshold or shader) reuses upstream filtered chunks from cache without recomputing them.
                        </span>
                    </span>
                    <button class="preset-btn" onclick="clearCache()" style="padding: 0.1rem 0.4rem;">Flush Cache</button>
                </label>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-val" id="stat-mem">0 MB</div>
                        <div class="stat-lbl">Memory Used</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val" id="stat-chunks">0</div>
                        <div class="stat-lbl">Cached Chunks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val" id="stat-ratio">0.0%</div>
                        <div class="stat-lbl">Cache Hit Ratio</div>
                    </div>
                </div>
            </div>

            <!-- Live Metadata Preview -->
            <div class="form-group">
                <label>
                    <span>Generated OME-Zarr 0.5 Metadata Preview</span>
                    <button class="preset-btn" onclick="fetchLiveMetadata()">Inspect zarr.json</button>
                </label>
                <pre id="meta-preview">Click "Inspect zarr.json" to preview metadata generated on-the-fly by chunkmirage...</pre>
            </div>
        </div>
    </div>

    <div id="toast">Copied to clipboard!</div>

    <script>
        // Start in Sato Tubeness mode by default (fast, robust 3D vessel enhancement)
        let steps = [
            { op: 'sato', sigmas: [1.0, 2.0], black_ridges: false }
        ];

        let currentPayload = "";
        let currentSourceUrl = "";

        const PRESETS = {
            hoa_heart_https: "https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr",
            hoa_heart_gs: "gs://ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr",
            flyem: "s3://neuroglancer-janelia-flyem-hemibrain/v1.2/groundtruth.zarr",
            synthetic: "synthetic://blobs?shape=256,256,256"
        };

        function setPipelineMode(mode) {
            document.querySelectorAll('.presets button[id^="btn-mode-"]').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.getElementById(`btn-mode-${mode}`);
            if (activeBtn) activeBtn.classList.add('active');

            if (mode === 'sato') {
                steps = [{ op: 'sato', sigmas: [1.0, 2.0], black_ridges: false }];
                document.getElementById('target_dtype').value = 'float32';
                document.getElementById('val_vmin').value = "0";
                document.getElementById('val_vmax').value = "500";
            } else if (mode === 'frangi') {
                steps = [{ op: 'frangi', sigmas: [1.0, 2.0], alpha: 0.5, beta: 0.5, black_ridges: false }];
                document.getElementById('target_dtype').value = 'float32';
                document.getElementById('val_vmin').value = "0";
                document.getElementById('val_vmax').value = "0.2";
            } else if (mode === 'segment') {
                steps = [
                    { op: 'sato', sigmas: [1.0, 2.0], black_ridges: false },
                    { op: 'li', value: 1 }
                ];
                document.getElementById('target_dtype').value = 'uint8';
                document.getElementById('val_vmin').value = "";
                document.getElementById('val_vmax').value = "";
            } else if (mode === 'sobel') {
                steps = [
                    { op: 'gaussian', sigma: 1.0 },
                    { op: 'sobel' }
                ];
                document.getElementById('target_dtype').value = 'float32';
                document.getElementById('val_vmin').value = "0";
                document.getElementById('val_vmax').value = "3000";
            } else if (mode === 'denoised-sato') {
                steps = [
                    { op: 'median', size: 3 },
                    { op: 'sato', sigmas: [1.5, 3.0], black_ridges: false }
                ];
                document.getElementById('target_dtype').value = 'float32';
                document.getElementById('val_vmin').value = "0";
                document.getElementById('val_vmax').value = "400";
            } else if (mode === 'tophat') {
                steps = [{ op: 'white_tophat', radius: 2 }];
                document.getElementById('target_dtype').value = 'original';
                document.getElementById('val_vmin').value = "0";
                document.getElementById('val_vmax').value = "1500";
            } else if (mode === 'unsharp') {
                steps = [{ op: 'unsharp_mask', radius: 1.5, amount: 2.0 }];
                document.getElementById('target_dtype').value = 'original';
                document.getElementById('val_vmin').value = "18000";
                document.getElementById('val_vmax').value = "26000";
            } else if (mode === 'ctwindow') {
                steps = [{ op: 'ct_window', vmin: 18000, vmax: 26000, target_dtype: "uint8" }];
                document.getElementById('target_dtype').value = 'original';
                document.getElementById('val_vmin').value = "0";
                document.getElementById('val_vmax').value = "255";
            } else if (mode === 'passthrough') {
                steps = [{ op: 'passthrough' }];
                document.getElementById('target_dtype').value = 'original';
                document.getElementById('val_vmin').value = "18000";
                document.getElementById('val_vmax').value = "26000";
            }
            renderSteps();
            updateUI();
        }

        function setPreset(key) {
            const url = PRESETS[key] || key;
            document.getElementById('src_url').value = url;

            document.querySelectorAll('.presets .preset-btn').forEach(btn => {
                if (btn.id.startsWith('btn-mode-')) return;
                btn.classList.remove('active');
            });
            if (event && event.target) event.target.classList.add('active');

            if (url.includes('heart') || url.includes('ucl-hip-ct')) {
                const firstOp = steps.length > 0 ? steps[0].op : 'passthrough';
                if (firstOp === 'passthrough' || firstOp === 'unsharp_mask') {
                    document.getElementById('val_vmin').value = "18000";
                    document.getElementById('val_vmax').value = "26000";
                }
            }
            updateUI();
        }

        function setNgClient(url) {
            if (url === 'self') {
                document.getElementById('ng_client').value = `${window.location.origin}/viewer`;
            } else {
                document.getElementById('ng_client').value = url;
            }
            updateUI();
        }

        function renderSteps() {
            const container = document.getElementById('pipeline-steps');
            container.innerHTML = "";

            let totalHalo = 0;

            steps.forEach((step, idx) => {
                const card = document.createElement('div');
                card.className = 'step-card';

                let stepHalo = 0;
                if (step.op === 'sato' || step.op === 'frangi' || step.op === 'meijering') {
                    const sigs = Array.isArray(step.sigmas) ? step.sigmas : [2.0];
                    const maxS = Math.max(...sigs);
                    stepHalo = Math.ceil(3.0 * maxS);
                } else if (step.op === 'gaussian') {
                    stepHalo = Math.ceil(3.0 * (step.sigma || 1.0));
                } else if (step.op === 'unsharp_mask') {
                    stepHalo = Math.ceil(3.0 * (step.radius || 1.5));
                } else if (step.op === 'sobel' || step.op === 'scharr' || step.op === 'laplace') {
                    stepHalo = 1;
                } else if (step.op === 'median') {
                    stepHalo = Math.floor((step.size || 3) / 2);
                } else if (step.op === 'erode' || step.op === 'dilate' || step.op === 'white_tophat') {
                    stepHalo = step.radius || 1;
                } else if (step.op === 'otsu' || step.op === 'li') {
                    stepHalo = 4;
                }
                totalHalo += stepHalo;

                let fieldsHtml = "";
                if (step.op === 'passthrough') {
                    fieldsHtml = `
                        <div class="form-group" style="flex: 2 1 200px;">
                            <label>Status</label>
                            <span style="font-size: 0.8rem; color: var(--success); padding: 0.4rem 0;">Direct unmodified chunk streaming (0 halo)</span>
                        </div>
                    `;
                } else if (step.op === 'sato') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Scales (&sigma; voxels)</label>
                            <input type="text" value="${(step.sigmas || [1.0, 2.0]).join(', ')}" onchange="updateStep(${idx}, 'sigmas', this.value.split(',').map(Number))">
                        </div>
                        <div class="form-group">
                            <label>Ridge Mode</label>
                            <select onchange="updateStep(${idx}, 'black_ridges', this.value === 'true')">
                                <option value="false" ${!step.black_ridges ? 'selected' : ''}>Bright Ridges (Vessels)</option>
                                <option value="true" ${step.black_ridges ? 'selected' : ''}>Dark Ridges</option>
                            </select>
                        </div>
                    `;
                } else if (step.op === 'frangi') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Scales (&sigma; voxels)</label>
                            <input type="text" value="${(step.sigmas || [1.0, 2.0]).join(', ')}" onchange="updateStep(${idx}, 'sigmas', this.value.split(',').map(Number))">
                        </div>
                        <div class="form-group">
                            <label>Ridge Mode</label>
                            <select onchange="updateStep(${idx}, 'black_ridges', this.value === 'true')">
                                <option value="false" ${!step.black_ridges ? 'selected' : ''}>Bright Ridges</option>
                                <option value="true" ${step.black_ridges ? 'selected' : ''}>Dark Ridges</option>
                            </select>
                        </div>
                    `;
                } else if (step.op === 'meijering') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Scales (&sigma; voxels)</label>
                            <input type="text" value="${(step.sigmas || [1.0, 2.0]).join(', ')}" onchange="updateStep(${idx}, 'sigmas', this.value.split(',').map(Number))">
                        </div>
                    `;
                } else if (step.op === 'sobel' || step.op === 'scharr' || step.op === 'laplace') {
                    fieldsHtml = `
                        <div class="form-group" style="flex: 2 1 200px;">
                            <label>Method</label>
                            <span style="font-size: 0.8rem; color: var(--accent); padding: 0.4rem 0;">3D spatial derivative (halo: 1 px)</span>
                        </div>
                    `;
                } else if (step.op === 'unsharp_mask') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Blur Radius (&sigma; voxels)</label>
                            <input type="number" step="0.5" min="0.5" value="${step.radius || 1.5}" onchange="updateStep(${idx}, 'radius', parseFloat(this.value))">
                        </div>
                        <div class="form-group">
                            <label>Sharpen Amount</label>
                            <input type="number" step="0.5" min="0.1" value="${step.amount || 1.5}" onchange="updateStep(${idx}, 'amount', parseFloat(this.value))">
                        </div>
                    `;
                } else if (step.op === 'median') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Cube Size (odd: 3, 5, 7)</label>
                            <input type="number" step="2" min="3" max="15" value="${step.size || 3}" onchange="updateStep(${idx}, 'size', parseInt(this.value))">
                        </div>
                    `;
                } else if (step.op === 'erode' || step.op === 'dilate' || step.op === 'white_tophat') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Radius (voxels)</label>
                            <input type="number" min="1" max="10" value="${step.radius || (step.op === 'white_tophat' ? 2 : 1)}" onchange="updateStep(${idx}, 'radius', parseInt(this.value))">
                        </div>
                    `;
                } else if (step.op === 'ct_window') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Window Min (vmin)</label>
                            <input type="number" value="${step.vmin ?? 18000}" onchange="updateStep(${idx}, 'vmin', parseFloat(this.value))">
                        </div>
                        <div class="form-group">
                            <label>Window Max (vmax)</label>
                            <input type="number" value="${step.vmax ?? 26000}" onchange="updateStep(${idx}, 'vmax', parseFloat(this.value))">
                        </div>
                    `;
                } else if (step.op === 'dynamic_range') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Percentile Min</label>
                            <input type="number" step="0.5" min="0" max="50" value="${step.pmin ?? 0.0}" onchange="updateStep(${idx}, 'pmin', parseFloat(this.value))">
                        </div>
                        <div class="form-group">
                            <label>Percentile Max</label>
                            <input type="number" step="0.5" min="50" max="100" value="${step.pmax ?? 99.5}" onchange="updateStep(${idx}, 'pmax', parseFloat(this.value))">
                        </div>
                    `;
                } else if (step.op === 'gamma') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Gamma Exponent</label>
                            <input type="number" step="0.1" min="0.1" max="5.0" value="${step.gamma ?? 0.5}" onchange="updateStep(${idx}, 'gamma', parseFloat(this.value))">
                        </div>
                    `;
                } else if (step.op === 'gaussian') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Sigma (&sigma; voxels)</label>
                            <input type="number" step="0.5" min="0.5" value="${step.sigma || 1.0}" onchange="updateStep(${idx}, 'sigma', parseFloat(this.value))">
                        </div>
                    `;
                } else if (step.op === 'otsu' || step.op === 'li') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Foreground Label</label>
                            <input type="number" min="1" max="255" value="${step.value || 1}" onchange="updateStep(${idx}, 'value', parseInt(this.value))">
                        </div>
                    `;
                } else if (step.op === 'threshold') {
                    fieldsHtml = `
                        <div class="form-group">
                            <label>Threshold Low</label>
                            <input type="number" step="0.1" value="${step.low ?? 0.2}" onchange="updateStep(${idx}, 'low', parseFloat(this.value))">
                        </div>
                    `;
                }

                card.innerHTML = `
                    <div class="step-header">
                        <span class="step-num">Stage ${idx + 1}: ${step.op.toUpperCase()} &bull; Halo: ${stepHalo} px</span>
                        ${steps.length > 1 ? `<button class="step-remove" onclick="removeStep(${idx})">&times; Remove</button>` : ''}
                    </div>
                    <div class="step-grid">
                        <div class="form-group">
                            <label>Operation</label>
                            <select onchange="changeStepOp(${idx}, this.value)">
                                <optgroup label="Vasculature & Ridges">
                                    <option value="sato" ${step.op === 'sato' ? 'selected' : ''}>Sato Tubeness (Default)</option>
                                    <option value="frangi" ${step.op === 'frangi' ? 'selected' : ''}>Frangi Vesselness</option>
                                    <option value="meijering" ${step.op === 'meijering' ? 'selected' : ''}>Meijering Ridge</option>
                                </optgroup>
                                <optgroup label="Gradients & Sharpening">
                                    <option value="sobel" ${step.op === 'sobel' ? 'selected' : ''}>Sobel 3D Magnitude</option>
                                    <option value="scharr" ${step.op === 'scharr' ? 'selected' : ''}>Scharr 3D Magnitude</option>
                                    <option value="laplace" ${step.op === 'laplace' ? 'selected' : ''}>Laplace 3D Filter</option>
                                    <option value="unsharp_mask" ${step.op === 'unsharp_mask' ? 'selected' : ''}>Unsharp Mask</option>
                                    <option value="gaussian" ${step.op === 'gaussian' ? 'selected' : ''}>Gaussian Blur</option>
                                </optgroup>
                                <optgroup label="Morphology & Denoising">
                                    <option value="median" ${step.op === 'median' ? 'selected' : ''}>Median 3D Filter</option>
                                    <option value="white_tophat" ${step.op === 'white_tophat' ? 'selected' : ''}>White Top-Hat</option>
                                    <option value="erode" ${step.op === 'erode' ? 'selected' : ''}>Grayscale Erosion</option>
                                    <option value="dilate" ${step.op === 'dilate' ? 'selected' : ''}>Grayscale Dilation</option>
                                </optgroup>
                                <optgroup label="Intensity & Contrast">
                                    <option value="ct_window" ${step.op === 'ct_window' ? 'selected' : ''}>CT Window / Level</option>
                                    <option value="dynamic_range" ${step.op === 'dynamic_range' ? 'selected' : ''}>Dynamic Range (Percentiles)</option>
                                    <option value="gamma" ${step.op === 'gamma' ? 'selected' : ''}>Gamma Contrast</option>
                                </optgroup>
                                <optgroup label="Segmentation">
                                    <option value="li" ${step.op === 'li' ? 'selected' : ''}>Li Cross-Entropy (Sparse)</option>
                                    <option value="otsu" ${step.op === 'otsu' ? 'selected' : ''}>Otsu Segmentation</option>
                                    <option value="threshold" ${step.op === 'threshold' ? 'selected' : ''}>Threshold</option>
                                </optgroup>
                                <optgroup label="Testing">
                                    <option value="passthrough" ${step.op === 'passthrough' ? 'selected' : ''}>Passthrough (Raw Data)</option>
                                </optgroup>
                            </select>
                        </div>
                        ${fieldsHtml}
                    </div>
                `;
                container.appendChild(card);
            });

            document.getElementById('halo-total-badge').innerText = `Cumulative Halo: ${totalHalo} px`;
        }

        function addStep() {
            steps.push({ op: 'sato', sigmas: [1.0, 2.0] });
            renderSteps();
            updateUI();
        }

        function removeStep(idx) {
            steps.splice(idx, 1);
            renderSteps();
            updateUI();
        }

        function changeStepOp(idx, newOp) {
            if (newOp === 'passthrough') steps[idx] = { op: 'passthrough' };
            else if (newOp === 'sato') steps[idx] = { op: 'sato', sigmas: [1.0, 2.0], black_ridges: false };
            else if (newOp === 'frangi') steps[idx] = { op: 'frangi', sigmas: [1.0, 2.0], black_ridges: false };
            else if (newOp === 'meijering') steps[idx] = { op: 'meijering', sigmas: [1.0, 2.0], black_ridges: false };
            else if (newOp === 'sobel') steps[idx] = { op: 'sobel' };
            else if (newOp === 'scharr') steps[idx] = { op: 'scharr' };
            else if (newOp === 'laplace') steps[idx] = { op: 'laplace' };
            else if (newOp === 'unsharp_mask') steps[idx] = { op: 'unsharp_mask', radius: 1.5, amount: 2.0 };
            else if (newOp === 'gaussian') steps[idx] = { op: 'gaussian', sigma: 1.0 };
            else if (newOp === 'median') steps[idx] = { op: 'median', size: 3 };
            else if (newOp === 'erode') steps[idx] = { op: 'erode', radius: 1 };
            else if (newOp === 'dilate') steps[idx] = { op: 'dilate', radius: 1 };
            else if (newOp === 'white_tophat') steps[idx] = { op: 'white_tophat', radius: 2 };
            else if (newOp === 'ct_window') steps[idx] = { op: 'ct_window', vmin: 18000, vmax: 26000, target_dtype: "uint8" };
            else if (newOp === 'dynamic_range') steps[idx] = { op: 'dynamic_range', pmin: 0.0, pmax: 99.5, target_dtype: "uint8" };
            else if (newOp === 'gamma') steps[idx] = { op: 'gamma', gamma: 0.5 };
            else if (newOp === 'otsu') steps[idx] = { op: 'otsu', value: 1 };
            else if (newOp === 'li') steps[idx] = { op: 'li', value: 1 };
            else if (newOp === 'threshold') steps[idx] = { op: 'threshold', low: 0.2 };
            renderSteps();
            updateUI();
        }

        function updateStep(idx, field, value) {
            steps[idx][field] = value;
            renderSteps();
            updateUI();
        }

        function base64UrlEncode(str) {
            return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
        }

        function buildPipelineSpec() {
            const srcUrl = document.getElementById('src_url').value.trim();
            const targetDtype = document.getElementById('target_dtype').value;
            const backend = document.getElementById('compute_backend').value;

            const opsList = JSON.parse(JSON.stringify(steps));
            if (targetDtype !== 'original' && targetDtype !== 'float32') {
                const lastOp = opsList.length > 0 ? opsList[opsList.length - 1].op : '';
                if (lastOp === 'frangi' && targetDtype === 'uint8') {
                    opsList.push({ op: 'scale', factor: 255.0 });
                }
                opsList.push({ op: 'cast', dtype: targetDtype });
            }

            return {
                source: srcUrl,
                backend: backend,
                ops: opsList
            };
        }

        function updateUI() {
            const spec = buildPipelineSpec();
            const specStr = JSON.stringify(spec);
            currentPayload = base64UrlEncode(specStr);

            let proxyHost = document.getElementById('proxy_host').value.trim();
            if (!proxyHost) {
                proxyHost = window.location.origin;
            }
            proxyHost = proxyHost.replace(/\/+$/, '');
            const fmt = document.getElementById('output_format').value;

            let scheme = 'zarr3';
            if (fmt === 'precomputed') scheme = 'precomputed';
            else if (fmt === 'n5') scheme = 'n5';
            else if (fmt === 'zarr') scheme = 'zarr';

            currentSourceUrl = `${scheme}://${proxyHost}/${currentPayload}/${fmt}`;

            document.getElementById('out-payload').innerText = currentPayload;
            document.getElementById('out-source-url').innerText = currentSourceUrl;

            let ngHost = document.getElementById('ng_client').value.trim();
            if (!ngHost) {
                ngHost = `${window.location.origin}/viewer`;
            }
            ngHost = ngHost.replace(/\/+$/, '');

            const isMask = steps.some(s => s.op === 'otsu' || s.op === 'li' || s.op === 'threshold');
            const targetDtype = document.getElementById('target_dtype').value;
            const vminInput = document.getElementById('val_vmin').value.trim();
            const vmaxInput = document.getElementById('val_vmax').value.trim();

            const isHeart = spec.source.includes('heart') || spec.source.includes('bm18');
            let vmin = vminInput ? parseFloat(vminInput) : 0;
            let vmax = vmaxInput ? parseFloat(vmaxInput) : 255;

            // If inputs are left blank, calculate intelligent defaults based on the first/last op
            if (!vminInput && !vmaxInput) {
                const firstOp = steps.length > 0 ? steps[0].op : 'sato';
                if (firstOp === 'passthrough' && isHeart) {
                    vmin = 18000; vmax = 26000;
                } else if (firstOp === 'unsharp_mask' && isHeart) {
                    vmin = 18000; vmax = 26000;
                } else if (firstOp === 'sato') {
                    vmin = 0; vmax = 500;
                } else if (firstOp === 'frangi') {
                    vmin = 0; vmax = 0.2;
                } else if (firstOp === 'sobel' || firstOp === 'scharr') {
                    vmin = 0; vmax = 3000;
                } else if (firstOp === 'white_tophat') {
                    vmin = 0; vmax = 1500;
                } else {
                    vmin = 0; vmax = 255;
                }
            }

            const ngLayer = {
                type: isMask ? "segmentation" : "image",
                source: currentSourceUrl,
                tab: "source",
                opacity: 1,
                volumeRendering: "max",
                volumeRenderingDepthSamples: 256,
                name: isHeart ? "hipct-heart" : "chunkmirage-layer"
            };

            if (!isMask) {
                ngLayer.shaderControls = {
                    normalized: {
                        range: [vmin, vmax]
                    }
                };
            }

            const ngState = {
                dimensions: {
                    x: [0.00000801, "m"],
                    y: [0.00000801, "m"],
                    z: [0.00000801, "m"]
                },
                position: isHeart ? [5928, 6807.5, 9495.5] : [0, 0, 0],
                crossSectionScale: 10,
                projectionOrientation: [0.3094263970851898, 0.20628425478935242, 0, -0.9282791018486023],
                projectionScale: 4096,
                layers: [ngLayer],
                showSlices: false,
                selectedLayer: {
                    visible: true,
                    layer: isHeart ? "hipct-heart" : "chunkmirage-layer"
                },
                layout: "4panel"
            };

            const hashPart = encodeURIComponent(JSON.stringify(ngState));
            document.getElementById('btn-launch-ng').href = `${ngHost}/#!${hashPart}`;
            document.getElementById('btn-launch-label').innerText = isHeart ? "Open in Neuroglancer (Heart 4-Panel 3D)" : "Open in Neuroglancer";
        }

        async function fetchLiveMetadata() {
            const pre = document.getElementById('meta-preview');
            pre.innerText = "Fetching live OME-Zarr metadata from chunkmirage...";
            const fmt = document.getElementById('output_format').value;

            const url = `/${currentPayload}/${fmt}/zarr.json`;
            try {
                const resp = await fetch(url);
                if (resp.ok) {
                    const data = await resp.json();
                    pre.innerText = JSON.stringify(data, null, 2);
                } else {
                    pre.innerText = `HTTP ${resp.status}: ${await resp.text()}`;
                }
            } catch (err) {
                pre.innerText = `Error: ${err.message}`;
            }
        }

        async function refreshStats() {
            try {
                const resp = await fetch('/_stats');
                if (resp.ok) {
                    const data = await resp.json();
                    const cache = data.cache || {};
                    const mb = ((cache.bytes || 0) / (1024 * 1024)).toFixed(1);
                    document.getElementById('stat-mem').innerText = `${mb} MB`;
                    document.getElementById('stat-chunks').innerText = cache.entries || 0;

                    const hits = cache.hits || 0;
                    const misses = cache.misses || 0;
                    const total = hits + misses;
                    const ratio = total > 0 ? ((hits / total) * 100).toFixed(1) : '0.0';
                    document.getElementById('stat-ratio').innerText = `${ratio}%`;
                }
            } catch (e) {}
        }

        async function clearCache() {
            try {
                await fetch('/api/cache', { method: 'DELETE' });
                refreshStats();
                showToast("Server chunk cache cleared!");
            } catch (e) {}
        }

        function copyText(elemId) {
            const text = document.getElementById(elemId).innerText;
            navigator.clipboard.writeText(text).then(() => {
                showToast("Copied to clipboard!");
            });
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }

        // Initialize UI on load
        renderSteps();
        updateUI();
        setInterval(refreshStats, 3000);
        refreshStats();
    </script>
</body>
</html>
"""
