"""
Web UI for interactive Neuroglancer Zarr v3 link generation.
"""

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zarr v3 Neuroglancer Filter Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0e17;
            --bg-surface: #111827;
            --bg-card: rgba(17, 24, 39, 0.75);
            --bg-input: #1e293b;
            --border: rgba(255, 255, 255, 0.08);
            --border-focus: #6366f1;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --accent: #06b6d4;
            --accent-glow: rgba(6, 182, 212, 0.3);
            --success: #10b981;
            --warning: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --radius-md: 10px;
            --radius-sm: 6px;
            --font-sans: 'Inter', system-ui, sans-serif;
            --font-display: 'Outfit', var(--font-sans);
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-sans);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 85% 75%, rgba(6, 182, 212, 0.12) 0%, transparent 45%);
            background-attachment: fixed;
            line-height: 1.5;
        }

        /* Header */
        header {
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            background: rgba(10, 14, 23, 0.8);
            position: sticky;
            top: 0;
            z-index: 50;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px var(--primary-glow);
        }

        .brand-icon svg {
            width: 22px;
            height: 22px;
            fill: white;
        }

        .brand-text h1 {
            font-family: var(--font-display);
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text span {
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Container */
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            width: 100%;
            flex: 1;
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 2rem;
        }

        @media (max-width: 960px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        /* Card Styles */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.75rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            margin-bottom: 1.5rem;
            transition: border-color 0.2s;
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.15);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.75rem;
        }

        .card-header h2 {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Form elements */
        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .form-label .tag {
            font-size: 0.7rem;
            color: var(--accent);
            background: rgba(6, 182, 212, 0.1);
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
        }

        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-family: var(--font-sans);
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s;
        }

        input[type="text"]:focus,
        input[type="number"]:focus,
        select:focus {
            border-color: var(--border-focus);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .input-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .presets-row {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }

        .preset-btn {
            font-size: 0.75rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.25rem 0.65rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all 0.2s;
        }

        .preset-btn:hover {
            background: rgba(99, 102, 241, 0.15);
            color: #fff;
            border-color: var(--primary);
        }

        /* Info Buttons & Tooltip Popovers */
        .info-label-group {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .info-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 17px;
            height: 17px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #94a3b8;
            font-size: 0.68rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            line-height: 1;
            padding: 0;
            flex-shrink: 0;
        }

        .info-btn:hover, .info-btn.active {
            background: var(--primary);
            color: #fff;
            border-color: var(--primary);
            box-shadow: 0 0 10px var(--primary-glow);
            transform: scale(1.1);
        }

        /* Dynamic Method Insight Card */
        .method-insight-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            margin-top: 0.65rem;
            backdrop-filter: blur(8px);
            transition: all 0.25s ease;
        }

        .method-insight-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.4rem;
        }

        .insight-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #fff;
        }

        .insight-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
            margin-bottom: 0.5rem;
        }

        .insight-footer {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            font-size: 0.76rem;
            padding-top: 0.45rem;
            border-top: 1px solid rgba(255, 255, 255, 0.07);
        }

        .insight-row {
            display: flex;
            align-items: flex-start;
            gap: 0.4rem;
            line-height: 1.35;
        }

        .insight-label {
            font-weight: 600;
            color: var(--accent);
            white-space: nowrap;
        }

        .insight-text {
            color: #cbd5e1;
        }

        /* Floating Tooltip Popover */
        .tooltip-popover {
            position: fixed;
            z-index: 9999;
            width: 330px;
            max-width: calc(100vw - 2rem);
            background: rgba(15, 23, 42, 0.96);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(99, 102, 241, 0.45);
            border-radius: var(--radius-md);
            padding: 1rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.75), 0 0 20px var(--primary-glow);
            font-size: 0.82rem;
            line-height: 1.45;
            color: var(--text-main);
            display: none;
            opacity: 0;
            transform: translateY(6px);
            transition: opacity 0.15s ease, transform 0.15s ease;
            pointer-events: auto;
        }

        .tooltip-popover.show {
            display: block;
            opacity: 1;
            transform: translateY(0);
        }

        .tooltip-popover-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .tooltip-popover-title {
            font-weight: 700;
            font-size: 0.88rem;
            color: #fff;
        }

        .tooltip-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 1.1rem;
            line-height: 1;
            padding: 0.1rem 0.3rem;
            border-radius: 4px;
        }
        .tooltip-close:hover { color: #fff; background: rgba(255, 255, 255, 0.1); }

        .compute-badge {
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }

        .compute-fast {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .compute-mod {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .compute-heavy {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .tooltip-popover-desc {
            font-size: 0.8rem;
            color: #cbd5e1;
            margin-bottom: 0.6rem;
        }

        .tooltip-block {
            background: rgba(0, 0, 0, 0.28);
            border-radius: var(--radius-sm);
            padding: 0.5rem 0.65rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid var(--accent);
        }

        .tooltip-block-title {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent);
            margin-bottom: 0.2rem;
        }

        .tooltip-block-text {
            font-size: 0.77rem;
            color: #94a3b8;
            line-height: 1.35;
        }

        .tooltip-tips {
            font-size: 0.76rem;
            color: #a5b4fc;
            background: rgba(99, 102, 241, 0.1);
            border-radius: var(--radius-sm);
            padding: 0.45rem 0.65rem;
            border: 1px solid rgba(99, 102, 241, 0.25);
            line-height: 1.35;
        }

        /* Toggle Switches */
        .toggle-wrap {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0;
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: var(--bg-input);
            border: 1px solid var(--border);
            transition: .3s;
            border-radius: 24px;
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 3px;
            bottom: 3px;
            background-color: var(--text-muted);
            transition: .3s;
            border-radius: 50%;
        }

        input:checked + .toggle-slider {
            background-color: var(--primary);
            border-color: var(--primary);
        }

        input:checked + .toggle-slider:before {
            transform: translateX(20px);
            background-color: white;
        }

        /* Action Panel / Output Preview */
        .output-card {
            position: sticky;
            top: 5rem;
        }

        .btn-launch {
            width: 100%;
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: 1rem 1.5rem;
            font-size: 1.05rem;
            font-weight: 700;
            font-family: var(--font-display);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 25px var(--primary-glow);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            text-decoration: none;
            margin-bottom: 1.5rem;
        }

        .btn-launch:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px var(--primary-glow), 0 0 20px var(--accent-glow);
        }

        .btn-launch:active {
            transform: translateY(0);
        }

        .btn-launch svg {
            width: 20px;
            height: 20px;
            fill: white;
        }

        .code-box {
            position: relative;
            background: #070a10;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
        }

        .code-box pre {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: #cbd5e1;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 110px;
            overflow-y: auto;
        }

        .copy-btn {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-muted);
            padding: 0.25rem 0.5rem;
            font-size: 0.72rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }

        .copy-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .btn-preview-meta {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.65rem 1rem;
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-preview-meta:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent);
        }

        #meta-preview-wrap {
            display: none;
            margin-top: 1rem;
            border-top: 1px solid var(--border);
            padding-top: 1rem;
        }

        /* Stats Bar */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .stat-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.75rem;
        }

        .stat-label {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-value {
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 700;
            color: #fff;
            margin-top: 0.2rem;
        }

        /* Toast notification */
        #toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #10b981;
            color: white;
            padding: 0.75rem 1.25rem;
            border-radius: var(--radius-md);
            font-size: 0.88rem;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        #toast.show {
            transform: translateY(0);
            opacity: 1;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="brand-text">
                <h1>Zarr v3 Neuroglancer Studio</h1>
                <span>On-The-Fly Volumetric Filter Proxy</span>
            </div>
        </div>
        <div id="proxy-status" class="status-badge">
            <div class="status-dot"></div>
            <span id="status-text">Proxy Active</span>
        </div>
    </header>

    <main class="container">
        <!-- Configuration Form -->
        <section class="config-section">
            <div class="card">
                <div class="card-header">
                    <h2>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                            <line x1="12" y1="22.08" x2="12" y2="12"/>
                        </svg>
                        1. Dataset Source
                    </h2>
                </div>
                <div class="form-group">
                    <label class="form-label" for="src_url">
                        <span class="info-label-group">
                            Zarr v3 / OME-Zarr 0.5 Dataset URL
                            <button type="button" class="info-btn" onclick="toggleParamTooltip('src_url', event)" onmouseenter="showParamTooltip('src_url', event)" onmouseleave="hideParamTooltip()" title="Dataset storage and remote fetch impact">ℹ</button>
                        </span>
                        <span class="tag">S3, HTTPS, or Local</span>
                    </label>
                    <input type="text" id="src_url" value="https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr" placeholder="s3://bucket/path/to/dataset.zarr" />
                    <div class="presets-row">
                        <button type="button" class="preset-btn" onclick="setPreset('https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr')">UCL HiP-CT Heart (HTTPS)</button>
                        <button type="button" class="preset-btn" onclick="setPreset('gs://ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr')">UCL HiP-CT Heart (gs://)</button>
                    </div>

                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="3"/>
                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                        </svg>
                        2. 3D Volumetric Filter
                    </h2>
                </div>
                
                <div class="form-group">
                    <label class="form-label" for="filter_method">
                        <span class="info-label-group">
                            Algorithm
                            <button type="button" class="info-btn" onclick="toggleParamTooltip('filter_method', event)" onmouseenter="showParamTooltip('filter_method', event)" onmouseleave="hideParamTooltip()" title="Algorithm details & compute complexity">ℹ</button>
                        </span>
                    </label>
                    <select id="filter_method" onchange="onMethodChange()">
                        <option value="frangi" selected>Frangi Vesselness / Tubeness (3D)</option>
                        <option value="sato">Sato Tubeness (3D)</option>
                        <option value="gaussian">Gaussian Smoothing (3D)</option>
                        <option value="threshold_otsu">Otsu Threshold Binarization</option>
                        <option value="constant">Constant Fill (Benchmark)</option>
                        <option value="identity">Identity (Passthrough)</option>
                    </select>

                    <!-- Dynamic Algorithm Insight Banner -->
                    <div id="method-insight-card" class="method-insight-card">
                        <div class="method-insight-top">
                            <span id="insight-title" class="insight-title">Frangi Vesselness Filter (3D)</span>
                            <span id="insight-badge" class="compute-badge compute-heavy">🔴 High Compute</span>
                        </div>
                        <div id="insight-desc" class="insight-desc">Computes the 3D Hessian tensor across scales to detect multi-scale tubular structures.</div>
                        <div class="insight-footer">
                            <div class="insight-row">
                                <span class="insight-label">⚡ Compute Impact:</span>
                                <span id="insight-compute" class="insight-text">6 3D convolutions + 3×3 eigenvalue decomposition per voxel for each scale.</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">💡 Recommendation:</span>
                                <span id="insight-tips" class="insight-text">Use 1 or 2 sigmas (e.g. 1.5 or 1.0, 2.0) for responsive navigation.</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Parameters dynamic block -->
                <div id="params-frangi-sato">
                    <div class="form-group">
                        <label class="form-label" for="sigmas">
                            <span class="info-label-group">
                                Scale Sigmas (σ)
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('sigmas', event)" onmouseenter="showParamTooltip('sigmas', event)" onmouseleave="hideParamTooltip()" title="Sigmas compute impact">ℹ</button>
                            </span>
                            <span class="tag">Comma-separated</span>
                        </label>
                        <input type="text" id="sigmas" value="1.0, 2.0" placeholder="e.g. 1.0, 2.0, 3.0" />
                    </div>
                    <div class="input-row">
                        <div class="form-group">
                            <label class="form-label" for="param_alpha">
                                <span class="info-label-group">
                                    Alpha (Plate sensitivity)
                                    <button type="button" class="info-btn" onclick="toggleParamTooltip('param_alpha', event)" onmouseenter="showParamTooltip('param_alpha', event)" onmouseleave="hideParamTooltip()" title="Alpha compute details">ℹ</button>
                                </span>
                            </label>
                            <input type="number" id="param_alpha" value="0.5" step="0.1" min="0.1" max="2.0" />
                        </div>
                        <div class="form-group">
                            <label class="form-label" for="param_beta">
                                <span class="info-label-group">
                                    Beta (Blob sensitivity)
                                    <button type="button" class="info-btn" onclick="toggleParamTooltip('param_beta', event)" onmouseenter="showParamTooltip('param_beta', event)" onmouseleave="hideParamTooltip()" title="Beta compute details">ℹ</button>
                                </span>
                            </label>
                            <input type="number" id="param_beta" value="0.5" step="0.1" min="0.1" max="2.0" />
                        </div>
                    </div>
                    <div class="toggle-wrap">
                        <div>
                            <div style="font-weight: 600; font-size: 0.88rem;">
                                <span class="info-label-group">
                                    <span>Black Ridges</span>
                                    <button type="button" class="info-btn" onclick="toggleParamTooltip('black_ridges', event)" onmouseenter="showParamTooltip('black_ridges', event)" onmouseleave="hideParamTooltip()" title="Black ridges compute details">ℹ</button>
                                </span>
                            </div>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">Detect dark structures on light background</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="black_ridges">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>

                <div id="params-gaussian" style="display: none;">
                    <div class="form-group">
                        <label class="form-label" for="gauss_sigma">
                            <span class="info-label-group">
                                Gaussian Sigma (σ)
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('gauss_sigma', event)" onmouseenter="showParamTooltip('gauss_sigma', event)" onmouseleave="hideParamTooltip()" title="Gaussian compute details">ℹ</button>
                            </span>
                        </label>
                        <input type="number" id="gauss_sigma" value="1.5" step="0.5" min="0.1" max="10.0" />
                    </div>
                </div>

                <div id="params-constant" style="display: none;">
                    <div class="form-group">
                        <label class="form-label" for="const_val">
                            <span class="info-label-group">
                                Constant Value
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('const_val', event)" onmouseenter="showParamTooltip('const_val', event)" onmouseleave="hideParamTooltip()" title="Constant benchmark details">ℹ</button>
                            </span>
                        </label>
                        <input type="number" id="const_val" value="128" min="0" max="255" />
                    </div>
                </div>

                <div id="params-otsu" style="display: none;">
                    <div style="font-size: 0.82rem; color: #cbd5e1; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: var(--radius-md); padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;">
                        <span><strong>Automatic Thresholding:</strong> Dynamically calculates the optimal foreground/background binarization threshold via voxel histogram variance. Parameter-free.</span>
                        <button type="button" class="info-btn" onclick="toggleParamTooltip('otsu_info', event)" onmouseenter="showParamTooltip('otsu_info', event)" onmouseleave="hideParamTooltip()" title="Otsu binarization details">ℹ</button>
                    </div>
                </div>

                <div id="params-identity" style="display: none;">
                    <div style="font-size: 0.82rem; color: #cbd5e1; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: var(--radius-md); padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;">
                        <span><strong>Passthrough Mode:</strong> Zero volumetric filtering applied. Delivers raw subvolume slices with optional window clipping and normalization. Ideal for baseline comparison.</span>
                        <button type="button" class="info-btn" onclick="toggleParamTooltip('identity_info', event)" onmouseenter="showParamTooltip('identity_info', event)" onmouseleave="hideParamTooltip()" title="Passthrough details">ℹ</button>
                    </div>
                </div>

                <div class="form-group" style="margin-top: 1rem;">
                    <label class="form-label" for="halo_padding">
                        <span class="info-label-group">
                            Boundary Halo Padding (voxels)
                            <button type="button" class="info-btn" onclick="toggleParamTooltip('halo_padding', event)" onmouseenter="showParamTooltip('halo_padding', event)" onmouseleave="hideParamTooltip()" title="Halo compute & RAM impact">ℹ</button>
                        </span>
                        <span class="tag" id="halo-auto-tag">Auto: 6 voxels</span>
                    </label>
                    <input type="number" id="halo_padding" placeholder="Leave empty for auto-calculation" min="0" max="64" />
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
                            <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
                            <line x1="6" y1="6" x2="6.01" y2="6"/>
                            <line x1="6" y1="18" x2="6.01" y2="18"/>
                        </svg>
                        3. Encoding & Output Codecs
                    </h2>
                </div>
                <div class="input-row">
                    <div class="form-group">
                        <label class="form-label" for="target_dtype">
                            <span class="info-label-group">
                                Target Data Type
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('target_dtype', event)" onmouseenter="showParamTooltip('target_dtype', event)" onmouseleave="hideParamTooltip()" title="Data type bandwidth & VRAM impact">ℹ</button>
                            </span>
                        </label>
                        <select id="target_dtype">
                            <option value="uint8" selected>uint8 (8-bit grayscale, 0-255)</option>
                            <option value="uint16">uint16 (16-bit, 0-65535)</option>
                            <option value="float32">float32 (raw floating point)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="compression">
                            <span class="info-label-group">
                                Chunk Compression
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('compression', event)" onmouseenter="showParamTooltip('compression', event)" onmouseleave="hideParamTooltip()" title="Compression codec speed">ℹ</button>
                            </span>
                        </label>
                        <select id="compression">
                            <option value="blosc" selected>Blosc (Fast Zstd - Recommended)</option>
                            <option value="gzip">GZip</option>
                            <option value="none">Uncompressed Raw</option>
                        </select>
                    </div>
                </div>
                <div class="input-row" style="margin-top: 0.5rem;">
                    <div class="form-group">
                        <label class="form-label" for="val_vmin">
                            <span class="info-label-group">
                                Window Min (vmin)
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('val_vmin_vmax', event)" onmouseenter="showParamTooltip('val_vmin_vmax', event)" onmouseleave="hideParamTooltip()" title="CT windowing details">ℹ</button>
                            </span>
                            <span class="tag">Air / Background</span>
                        </label>
                        <input type="number" id="val_vmin" value="18000" placeholder="e.g. 18000" />
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="val_vmax">
                            <span class="info-label-group">
                                Window Max (vmax)
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('val_vmin_vmax', event)" onmouseenter="showParamTooltip('val_vmin_vmax', event)" onmouseleave="hideParamTooltip()" title="CT windowing details">ℹ</button>
                            </span>
                            <span class="tag">Dense Structures</span>
                        </label>
                        <input type="number" id="val_vmax" value="26000" placeholder="e.g. 26000" />
                    </div>
                </div>
                <div class="presets-row" style="margin-bottom: 0.75rem;">
                    <button type="button" class="preset-btn" onclick="setWindow(18000, 26000)">HGIA CT Window (18k–26k)</button>
                    <button type="button" class="preset-btn" onclick="setWindow('', '')">Auto Min/Max</button>
                </div>
                <div class="toggle-wrap">
                    <div>
                        <div style="font-weight: 600; font-size: 0.88rem;">
                            <span class="info-label-group">
                                <span>Dynamic Range Normalization</span>
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('normalize', event)" onmouseenter="showParamTooltip('normalize', event)" onmouseleave="hideParamTooltip()" title="Normalization details">ℹ</button>
                            </span>
                        </div>
                        <div style="font-size: 0.75rem; color: var(--text-muted);">Scale filter/windowed response across target data type range</div>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="normalize" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                            <circle cx="12" cy="12" r="3"/>
                        </svg>
                        4. Host & Neuroglancer Viewer Configuration
                    </h2>
                </div>
                <div class="input-row">
                    <div class="form-group">
                        <label class="form-label" for="proxy_host">
                            <span class="info-label-group">
                                Proxy Server Address
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('proxy_host', event)" onmouseenter="showParamTooltip('proxy_host', event)" onmouseleave="hideParamTooltip()" title="Proxy server address & network binding">ℹ</button>
                            </span>
                        </label>
                        <input type="text" id="proxy_host" value="http://128.40.89.2:8000" placeholder="http://128.40.89.2:8000" />
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="ng_client">
                            <span class="info-label-group">
                                Neuroglancer Client
                                <button type="button" class="info-btn" onclick="toggleParamTooltip('ng_client', event)" onmouseenter="showParamTooltip('ng_client', event)" onmouseleave="hideParamTooltip()" title="Neuroglancer client hosting & WebWorker decoders">ℹ</button>
                            </span>
                            <span class="tag" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">Self-Hosted</span>
                        </label>
                        <input type="text" id="ng_client" value="http://128.40.89.2:8000/viewer" placeholder="http://128.40.89.2:8000/viewer" />
                        <div class="presets-row">
                            <button type="button" class="preset-btn" onclick="setNgClient('http://128.40.89.2:8000/viewer')">Self-Hosted (Zero CORS)</button>
                            <button type="button" class="preset-btn" onclick="setNgClient('https://neuroglancer-demo.appspot.com')">Official Demo</button>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label" for="layer_name">
                        <span class="info-label-group">
                            Layer Name
                            <button type="button" class="info-btn" onclick="toggleParamTooltip('layer_name', event)" onmouseenter="showParamTooltip('layer_name', event)" onmouseleave="hideParamTooltip()" title="Neuroglancer layer identifier">ℹ</button>
                        </span>
                    </label>
                    <input type="text" id="layer_name" value="filtered-tubeness" placeholder="Layer name in viewer" />
                </div>
            </div>
        </section>

        <!-- Live Link Output & Actions -->
        <aside class="action-section">
            <div class="card output-card">
                <div class="card-header">
                    <h2>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                        </svg>
                        Generated Links
                    </h2>
                    <span class="tag" style="background: rgba(99, 102, 241, 0.15); color: #818cf8; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">LIVE</span>
                </div>

                <!-- 1-Click Launch Button -->
                <a id="btn-open-neuroglancer" href="#" target="_blank" class="btn-launch">
                    <svg viewBox="0 0 24 24">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/>
                    </svg>
                    Open in Neuroglancer
                </a>

                <div class="form-group">
                    <label class="form-label">
                        Neuroglancer Source URL (zarr3://)
                    </label>
                    <div class="code-box">
                        <button type="button" class="copy-btn" onclick="copyText('ng_source_url', 'Source URL copied!')">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                            Copy
                        </button>
                        <pre><code id="ng_source_url">Generating...</code></pre>
                    </div>
                </div>

                <div class="form-group">
                    <label class="form-label">
                        Base64 Config Payload
                    </label>
                    <div class="code-box">
                        <button type="button" class="copy-btn" onclick="copyText('config_payload_str', 'Payload copied!')">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                            Copy
                        </button>
                        <pre><code id="config_payload_str">Generating...</code></pre>
                    </div>
                </div>

                <button type="button" class="btn-preview-meta" onclick="previewMetadata()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    Fetch Falsified zarr.json Preview
                </button>

                <div id="meta-preview-wrap">
                    <label class="form-label">Proxy Falsified Metadata:</label>
                    <div class="code-box">
                        <pre><code id="meta-preview-code">Loading...</code></pre>
                    </div>
                </div>

                <!-- Live Cache Metrics Dashboard -->
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">Raw Chunk Cache</div>
                        <div class="stat-value" id="stat-raw-mb">0.0 MB</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Processed Cache</div>
                        <div class="stat-value" id="stat-proc-mb">0.0 MB</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Cache Hits</div>
                        <div class="stat-value" id="stat-hits">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Hit Ratio</div>
                        <div class="stat-value" id="stat-ratio">0.0%</div>
                    </div>
                </div>
            </div>
        </aside>
    </main>

    <div id="toast">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
            <polyline points="20 6 9 17 4 12"/>
        </svg>
        <span id="toast-text">Copied to clipboard</span>
    </div>

    <!-- Floating Tooltip Popover Card -->
    <div id="tooltip-popover" class="tooltip-popover" onmouseenter="clearTimeout(tooltipTimeout)" onmouseleave="hideParamTooltip()">
        <div class="tooltip-popover-header">
            <span id="pop-title" class="tooltip-popover-title">Parameter Details</span>
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span id="pop-badge" class="compute-badge compute-mod">Compute Cost</span>
                <button type="button" class="tooltip-close" onclick="closeTooltip()">×</button>
            </div>
        </div>
        <div id="pop-desc" class="tooltip-popover-desc">Parameter description...</div>
        <div class="tooltip-block">
            <div class="tooltip-block-title">⚡ Computational Impact</div>
            <div id="pop-impact" class="tooltip-block-text">Mathematical operations and latency...</div>
        </div>
        <div id="pop-tips" class="tooltip-tips">💡 Recommendation...</div>
    </div>

    <script>
        let currentPayload = "";
        let currentSourceUrl = "";
        let activeTooltipParam = null;
        let tooltipPinned = false;
        let tooltipTimeout = null;

        const methodInsights = {
            frangi: {
                name: "Frangi Vesselness Filter (3D)",
                tier: "🔴 High Compute (~50–180ms)",
                tierClass: "compute-heavy",
                desc: "Calculates the 3D Hessian tensor across scales and decomposes 3 eigenvalues (λ₁, λ₂, λ₃) at every voxel to quantify vesselness vs platiness/blobbiness.",
                computeNotes: "Scales linearly with number of sigmas. Each sigma runs 6 3D Gaussian derivatives + full 3×3 eigenvalue decomposition at every voxel in the chunk.",
                tips: "Use 1 or 2 sigmas (e.g. 1.0, 2.0 or 1.5) for responsive navigation. Avoid > 3 sigmas during live interaction."
            },
            sato: {
                name: "Sato Tubeness Filter (3D)",
                tier: "🟡 Moderate Compute (~40–90ms)",
                tierClass: "compute-mod",
                desc: "Optimized specifically for continuous 3D tubular structures (coronary blood vessels, bronchi). Directly evaluates sorted eigenvalues without sheet/blob exponential ratios.",
                computeNotes: "Performs 3D Hessian + eigenvalue sorting, but response formula is faster and mathematically cleaner than Frangi.",
                tips: "Ideal for vascular micro-CT datasets. Sigmas [1.0, 2.0] highlights small to medium coronary trees smoothly."
            },
            gaussian: {
                name: "Gaussian Smoothing (3D)",
                tier: "⚡ Ultra-Fast (~2–5ms)",
                tierClass: "compute-fast",
                desc: "Performs isotropic 3D Gaussian filtering. Suppresses micro-CT Poisson speckle while preserving bulk tissue boundaries.",
                computeNotes: "Separable 1D kernels (X → Y → Z). Complexity is strictly O(N), executing in under 5ms per chunk.",
                tips: "Sigma 1.0–1.5 produces instant, clean denoising for real-time 60fps panning."
            },
            threshold_otsu: {
                name: "Otsu Automatic Binarization",
                tier: "⚡ Ultra-Fast (~1–2ms)",
                tierClass: "compute-fast",
                desc: "Calculates an optimal binarization threshold that minimizes intra-class variance of voxels, segmenting foreground tissue from background.",
                computeNotes: "Histogram-based calculation. Instantaneous (sub-2ms) per chunk.",
                tips: "Great for segmenting myocardium and bone from background air and fluid cavities."
            },
            constant: {
                name: "Constant Value Fill",
                tier: "⚡ Instant (~0.5ms)",
                tierClass: "compute-fast",
                desc: "Fills the chunk with a uniform value. Used for network transmission and codec decompression benchmarking.",
                computeNotes: "Zero convolution math. Pure memory write.",
                tips: "Use to verify raw network throughput and WebGL chunk rendering performance."
            },
            identity: {
                name: "Identity (Passthrough)",
                tier: "⚡ Instant (~1ms)",
                tierClass: "compute-fast",
                desc: "Passes through the raw data slice with optional CT window clipping and dynamic range normalization.",
                computeNotes: "Zero filter computation. Performs only subvolume slicing and optional byte normalization.",
                tips: "Baseline for comparing raw anatomical image data against filtered versions."
            }
        };

        const paramDetails = {
            src_url: {
                title: "Dataset Source URL",
                tier: "🌐 Storage / IO",
                tierClass: "compute-mod",
                desc: "Remote or local URI pointing to the root of the Zarr v3 or OME-Zarr 0.5 pyramid hierarchy.",
                impact: "Network latency depends on storage tier. Direct HTTPS / Google Cloud Storage ranges from 20–80ms per remote chunk. The proxy automatically caches raw subvolumes in memory (RawChunkCache) so subsequent overlapping requests are served in < 1ms.",
                tips: "For HGIA data, both 'zarr://gs://...' and HTTPS URLs are natively supported."
            },
            filter_method: {
                title: "Algorithm Selection",
                tier: "⚡ Select Method",
                tierClass: "compute-mod",
                desc: "Chooses the 3D volumetric image processing kernel applied on-the-fly to requested chunks.",
                impact: "Gaussian, Otsu, and Identity are sub-5ms. Frangi and Sato perform 3D Hessian convolutions and eigenvalue decompositions.",
                tips: "Start with Gaussian for instant denoising, or Sato [1.0, 2.0] for coronary vessel extraction."
            },
            sigmas: {
                title: "Scale Sigmas (σ)",
                tier: "🔴 Major Compute Driver",
                tierClass: "compute-heavy",
                desc: "Determines the spatial radius of structures to enhance (effective vessel radius ≈ √2·σ voxels).",
                impact: "Direct multiplier on 3D convolutions and eigenvalue solves. Each sigma adds 6 3D convolutions + 3×3 eigenvalue decomposition per voxel. Also determines boundary halo: h = ⌈3·σ_max⌉. Higher max sigma forces larger halo context.",
                tips: "Keep to 1 or 2 values (e.g. '1.5' or '1.0, 2.0') for live interactive viewing. Avoid > 3 sigmas."
            },
            param_alpha: {
                title: "Alpha (Plate Sensitivity)",
                tier: "⚡ Zero Compute Cost",
                tierClass: "compute-fast",
                desc: "Controls Frangi sensitivity for suppressing planar/plate-like structures (e.g. organ surfaces, muscle fascia). Default: 0.5.",
                impact: "Point-wise eigenvalue arithmetic (λ₂/λ₃ ratio). Does not trigger any extra convolutions or memory operations.",
                tips: "Lower values (0.2–0.3) penalize plates more strictly."
            },
            param_beta: {
                title: "Beta (Blob Sensitivity)",
                tier: "⚡ Zero Compute Cost",
                tierClass: "compute-fast",
                desc: "Controls Frangi sensitivity for penalizing spherical/blob-like structures (nodules, noise specks). Default: 0.5.",
                impact: "Point-wise eigenvalue arithmetic (λ₁/√(λ₂·λ₃)). Zero impact on convolution overhead.",
                tips: "Lower values (0.2–0.4) reject blobs and beads more aggressively."
            },
            black_ridges: {
                title: "Black Ridges",
                tier: "⚡ Zero Compute Cost",
                tierClass: "compute-fast",
                desc: "Inverts structure detection polarity. When checked, detects dark tubes on light background (e.g. air-filled bronchioles). When unchecked, detects bright vessels on dark background.",
                impact: "Simple sign negation during eigenvalue sorting. Zero computational difference.",
                tips: "Keep unchecked for contrast-enhanced CT vessels; check for air airways."
            },
            gauss_sigma: {
                title: "Gaussian Sigma (σ)",
                tier: "⚡ Ultra-Fast (< 5ms)",
                tierClass: "compute-fast",
                desc: "Standard deviation of the 3D Gaussian smoothing kernel in voxel units.",
                impact: "Very low. Uses separable 1D Gaussian convolutions along X, then Y, then Z. Complexity is linear O(N) regardless of sigma.",
                tips: "1.0 to 1.5 is ideal for micro-CT grain smoothing without blurring anatomical boundaries."
            },
            const_val: {
                title: "Constant Fill Value",
                tier: "⚡ Instant (< 1ms)",
                tierClass: "compute-fast",
                desc: "Uniform scalar assigned to all voxels in the chunk.",
                impact: "Memory fill operation only. Useful for benchmarking transfer latency.",
                tips: "Default 128 gives mid-gray in uint8."
            },
            otsu_info: {
                title: "Otsu Binarization",
                tier: "⚡ Ultra-Fast (1–2ms)",
                tierClass: "compute-fast",
                desc: "Computes the global intra-class variance minimizer across 256 intensity histogram bins per chunk.",
                impact: "O(N) histogram accumulation followed by 256 threshold evaluations. Zero floating-point convolutions.",
                tips: "Pairs well with pre-filtering or windowing for instant organ contour segmentation."
            },
            identity_info: {
                title: "Identity Passthrough",
                tier: "⚡ Instant (< 1ms)",
                tierClass: "compute-fast",
                desc: "Bypasses all volumetric filtering and passes raw subvolume chunks with optional clipping/normalization.",
                impact: "Zero arithmetic computation beyond memory slicing and byte casting.",
                tips: "Provides the maximum achievable frame rate and serves as the ground-truth control."
            },
            halo_padding: {
                title: "Boundary Halo Padding",
                tier: "🟡 Memory & Fetch Driver",
                tierClass: "compute-mod",
                desc: "Additional border voxels fetched around the 64³ chunk to prevent boundary seam artifacts from kernel convolution.",
                impact: "Increases volume size by (64 + 2h)³. For h=6, chunk expands from 262k to 438k voxels (+67%). Overly large halos slow down S3 fetches.",
                tips: "Leave blank for optimal auto-calculation (⌈3·σ⌉). Manual override only needed for specific custom filters."
            },
            target_dtype: {
                title: "Target Data Type",
                tier: "🟡 Network & VRAM Impact",
                tierClass: "compute-mod",
                desc: "Bit-depth of the processed chunks sent over HTTP to Neuroglancer.",
                impact: "uint8 = 1 byte/voxel; uint16 = 2 bytes/voxel; float32 = 4 bytes/voxel. uint8 yields 2×–4× faster network transfers and uses 2×–4× less GPU VRAM in the browser.",
                tips: "Use uint8 for maximum viewer smoothness and fastest loading. Use uint16 if raw 16-bit CT precision is needed."
            },
            compression: {
                title: "Compression Codec",
                tier: "⚡ Fast (Multi-Threaded)",
                tierClass: "compute-fast",
                desc: "Compression algorithm applied to chunks before HTTP transmission.",
                impact: "Blosc (Zstandard) compresses in 1–2ms on server and decompresses via WebAssembly in browser. Shrinks chunks by 3×–8×.",
                tips: "Blosc is strongly recommended. Gzip is slightly slower. None increases network bandwidth usage."
            },
            val_vmin_vmax: {
                title: "CT Window Min / Max (vmin, vmax)",
                tier: "⚡ Sub-Millisecond",
                tierClass: "compute-fast",
                desc: "Hounsfield attenuation clipping window applied to raw voxel values before filtering or normalization.",
                impact: "Single vectorised numpy.clip call (< 0.5ms). Zero impact on convolution overhead.",
                tips: "For HGIA heart, use 18000 (air/cavity threshold) to 26000 (dense tissue) to maximize tissue contrast."
            },
            normalize: {
                title: "Dynamic Range Normalization",
                tier: "⚡ Sub-Millisecond",
                tierClass: "compute-fast",
                desc: "Linearly maps filtered values across the full range (0–255 for uint8) so standard Neuroglancer shaders show optimal contrast.",
                impact: "Simple min-max linear scaling (< 0.5ms).",
                tips: "Keep enabled for uint8 so cross sections and 3D volume renderings display bright structures immediately."
            },
            proxy_host: {
                title: "Proxy Server Address",
                tier: "🌐 Network Config",
                tierClass: "compute-fast",
                desc: "The HTTP address where this proxy server is listening. Must be reachable from your web browser.",
                impact: "Direct local network or VPN connection. When accessing from remote machines, use the server's public IP (e.g. 128.40.89.2:8000) rather than localhost.",
                tips: "Preset defaults to the verified remote IP: http://128.40.89.2:8000."
            },
            ng_client: {
                title: "Neuroglancer Client",
                tier: "🌐 Viewer Frontend",
                tierClass: "compute-fast",
                desc: "The web application hosting the Neuroglancer interface and WebGL 3D rendering pipeline.",
                impact: "Self-hosted (/viewer) includes all local WebAssembly decoders and WebWorkers (zero CORS, zero mixed-content blocks). The official demo requires public HTTPS certificates.",
                tips: "Always use http://128.40.89.2:8000/viewer for reliable LAN/remote performance."
            },
            layer_name: {
                title: "Layer Name",
                tier: "⚡ Zero Compute Cost",
                tierClass: "compute-fast",
                desc: "Human-readable label assigned to this volume layer inside Neuroglancer's left sidebar.",
                impact: "Metadata only. Zero performance impact.",
                tips: "Give descriptive names (e.g. 'heart-sato-vessels' or 'passthrough-raw') when layering multiple views."
            }
        };

        function openTooltip(paramKey, targetEl) {
            const popover = document.getElementById('tooltip-popover');
            const info = paramDetails[paramKey];
            if (!info) return;

            activeTooltipParam = paramKey;

            document.getElementById('pop-title').innerText = info.title;
            const badge = document.getElementById('pop-badge');
            badge.innerText = info.tier;
            badge.className = 'compute-badge ' + (info.tierClass || 'compute-mod');

            document.getElementById('pop-desc').innerText = info.desc;
            document.getElementById('pop-impact').innerText = info.impact;
            document.getElementById('pop-tips').innerText = '💡 ' + info.tips;

            // Highlight button
            document.querySelectorAll('.info-btn').forEach(b => b.classList.remove('active'));
            if (targetEl) targetEl.classList.add('active');

            popover.classList.add('show');

            if (targetEl) {
                const rect = targetEl.getBoundingClientRect();
                let left = rect.right + 12;
                let top = rect.top - 10;

                if (left + 350 > window.innerWidth) {
                    left = rect.left - 350;
                }
                if (left < 10) left = 10;

                const popHeight = popover.offsetHeight || 260;
                if (top + popHeight > window.innerHeight - 10) {
                    top = window.innerHeight - popHeight - 10;
                }
                if (top < 10) top = 10;

                popover.style.left = left + 'px';
                popover.style.top = top + 'px';
            }
        }

        function showParamTooltip(paramKey, event) {
            if (tooltipPinned) return;
            clearTimeout(tooltipTimeout);
            openTooltip(paramKey, event.currentTarget);
        }

        function hideParamTooltip() {
            if (tooltipPinned) return;
            tooltipTimeout = setTimeout(() => {
                closeTooltip();
            }, 250);
        }

        function toggleParamTooltip(paramKey, event) {
            if (event) {
                event.preventDefault();
                event.stopPropagation();
            }
            clearTimeout(tooltipTimeout);
            const popover = document.getElementById('tooltip-popover');
            if (activeTooltipParam === paramKey && popover.classList.contains('show') && tooltipPinned) {
                tooltipPinned = false;
                closeTooltip();
            } else {
                tooltipPinned = true;
                openTooltip(paramKey, event.currentTarget);
            }
        }

        function closeTooltip() {
            tooltipPinned = false;
            const popover = document.getElementById('tooltip-popover');
            popover.classList.remove('show');
            document.querySelectorAll('.info-btn').forEach(b => b.classList.remove('active'));
            activeTooltipParam = null;
        }

        document.addEventListener('click', (e) => {
            const popover = document.getElementById('tooltip-popover');
            if (popover && popover.classList.contains('show') && !popover.contains(e.target) && !e.target.classList.contains('info-btn')) {
                closeTooltip();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeTooltip();
        });

        function updateMethodInsight() {
            const m = document.getElementById('filter_method').value;
            const info = methodInsights[m];
            if (!info) return;

            document.getElementById('insight-title').innerText = info.name;
            const badge = document.getElementById('insight-badge');
            badge.innerText = info.tier;
            badge.className = 'compute-badge ' + info.tierClass;

            document.getElementById('insight-desc').innerText = info.desc;
            document.getElementById('insight-compute').innerText = info.computeNotes;
            document.getElementById('insight-tips').innerText = info.tips;
        }

        function onMethodChange() {
            const m = document.getElementById('filter_method').value;
            document.getElementById('params-frangi-sato').style.display = (m === 'frangi' || m === 'sato') ? 'block' : 'none';
            document.getElementById('params-gaussian').style.display = (m === 'gaussian') ? 'block' : 'none';
            document.getElementById('params-constant').style.display = (m === 'constant') ? 'block' : 'none';
            document.getElementById('params-otsu').style.display = (m === 'threshold_otsu') ? 'block' : 'none';
            document.getElementById('params-identity').style.display = (m === 'identity') ? 'block' : 'none';
            updateMethodInsight();
            updateConfig();
        }

        function setWindow(vmin, vmax) {
            document.getElementById('val_vmin').value = vmin;
            document.getElementById('val_vmax').value = vmax;
            updateConfig();
        }

        function setPreset(url) {
            document.getElementById('src_url').value = url;
            if (url.includes('ucl-hip-ct')) {
                setWindow(18000, 26000);
            }
            updateConfig();
        }

        function setNgClient(url) {
            document.getElementById('ng_client').value = url;
            updateConfig();
        }

        function buildConfigObject() {
            let src_url = document.getElementById('src_url').value.trim();
            if (src_url.startsWith('zarr://')) src_url = src_url.substring(7).trim();
            if (src_url.startsWith('zarr3://')) src_url = src_url.substring(8).trim();
            if (src_url.startsWith('gs://')) {
                src_url = 'https://storage.googleapis.com/' + src_url.substring(5).replace(/^\/+/, '');
            } else if (src_url.startsWith('gcs://')) {
                src_url = 'https://storage.googleapis.com/' + src_url.substring(6).replace(/^\/+/, '');
            }
            src_url = src_url.replace(/\/+$/, '');

            const method = document.getElementById('filter_method').value;
            const target_dtype = document.getElementById('target_dtype').value;
            const compression = document.getElementById('compression').value;
            const normalize = document.getElementById('normalize').checked;
            
            const params = {};
            
            const vminInput = document.getElementById('val_vmin').value.trim();
            if (vminInput) params['vmin'] = parseFloat(vminInput);
            const vmaxInput = document.getElementById('val_vmax').value.trim();
            if (vmaxInput) params['vmax'] = parseFloat(vmaxInput);

            if (method === 'frangi' || method === 'sato') {
                const sigmasStr = document.getElementById('sigmas').value;
                const sigmas = sigmasStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
                params['sigmas'] = sigmas.length > 0 ? sigmas : [1.0, 2.0];
                if (method === 'frangi') {
                    params['alpha'] = parseFloat(document.getElementById('param_alpha').value) || 0.5;
                    params['beta'] = parseFloat(document.getElementById('param_beta').value) || 0.5;
                }
                params['black_ridges'] = document.getElementById('black_ridges').checked;
            } else if (method === 'gaussian') {
                params['sigma'] = parseFloat(document.getElementById('gauss_sigma').value) || 1.5;
            } else if (method === 'constant') {
                params['value'] = parseFloat(document.getElementById('const_val').value) || 128;
            }

            const haloInput = document.getElementById('halo_padding').value.trim();
            const halo = haloInput ? parseInt(haloInput, 10) : null;

            return {
                src_url: src_url,
                method: method,
                params: params,
                halo: halo,
                target_dtype: target_dtype,
                normalize: normalize,
                compression: compression,
                cname: "zstd",
                clevel: 5
            };
        }

        function base64UrlEncode(str) {
            return btoa(str)
                .replace(/\+/g, '-')
                .replace(/\//g, '_')
                .replace(/=+$/, '');
        }

        function updateConfig() {
            const configObj = buildConfigObject();
            
            // Auto halo badge update
            if (configObj.method === 'frangi' || configObj.method === 'sato') {
                const maxSig = Math.max(...(configObj.params.sigmas || [2.0]));
                document.getElementById('halo-auto-tag').innerText = 'Auto: ' + Math.ceil(3 * maxSig) + ' voxels';
            } else if (configObj.method === 'gaussian') {
                document.getElementById('halo-auto-tag').innerText = 'Auto: ' + Math.ceil(3 * (configObj.params.sigma || 1.5)) + ' voxels';
            } else {
                document.getElementById('halo-auto-tag').innerText = 'Auto: 0 voxels';
            }

            const jsonStr = JSON.stringify(configObj);
            currentPayload = base64UrlEncode(jsonStr);

            let proxyHost = document.getElementById('proxy_host').value.trim();
            if (!proxyHost) {
                proxyHost = window.location.origin.includes('localhost') ? 'http://128.40.89.2:8000' : window.location.origin;
                document.getElementById('proxy_host').value = proxyHost;
            }
            proxyHost = proxyHost.replace(/\/+$/, '');

            currentSourceUrl = `zarr3://${proxyHost}/${currentPayload}`;

            document.getElementById('config_payload_str').innerText = currentPayload;
            document.getElementById('ng_source_url').innerText = currentSourceUrl;

            // Generate full Neuroglancer launch link
            let ngHost = document.getElementById('ng_client').value.trim() || 'http://128.40.89.2:8000/viewer';
            ngHost = ngHost.replace(/\/+$/, '');
            const layerName = document.getElementById('layer_name').value.trim() || 'filtered-zarr3';

            const shaderRange = configObj.target_dtype === 'uint16' ? [18192, 29950] : [0, 255];

            const ngLayer = {
                type: "image",
                source: currentSourceUrl,
                tab: "source",
                opacity: 1,
                shaderControls: {
                    normalized: {
                        range: shaderRange
                    }
                },
                volumeRendering: "max",
                volumeRenderingDepthSamples: 256,
                name: layerName
            };

            const isHeart = configObj.src_url.includes('heart') || configObj.src_url.includes('bm18');
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
                    layer: layerName
                },
                layout: "4panel"
            };

            const hashPart = encodeURIComponent(JSON.stringify(ngState));
            const fullNgUrl = `${ngHost}/#!${hashPart}`;

            document.getElementById('btn-open-neuroglancer').href = fullNgUrl;
        }

        function copyText(elementId, message) {
            const text = document.getElementById(elementId).innerText;
            navigator.clipboard.writeText(text).then(() => {
                showToast(message || 'Copied to clipboard!');
            });
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            document.getElementById('toast-text').innerText = msg;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 2500);
        }

        async function previewMetadata() {
            const wrap = document.getElementById('meta-preview-wrap');
            const codeEl = document.getElementById('meta-preview-code');
            wrap.style.display = 'block';
            codeEl.innerText = "Querying proxy for falsified zarr.json...";

            try {
                const resp = await fetch(`/${currentPayload}/zarr.json`);
                if (!resp.ok) {
                    codeEl.innerText = `HTTP ${resp.status}: ${await resp.text()}`;
                    return;
                }
                const data = await resp.json();
                codeEl.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                codeEl.innerText = "Fetch error: " + err.message;
            }
        }

        async function refreshStats() {
            try {
                const resp = await fetch('/_stats');
                if (resp.ok) {
                    const data = await resp.json();
                    document.getElementById('stat-raw-mb').innerText = data.raw_chunk_cache.currsize_mb + ' MB';
                    document.getElementById('stat-proc-mb').innerText = data.processed_chunk_cache.currsize_mb + ' MB';
                    document.getElementById('stat-hits').innerText = data.processed_chunk_cache.hits + data.raw_chunk_cache.hits;
                    const totHits = data.processed_chunk_cache.hits + data.raw_chunk_cache.hits;
                    const totMisses = data.processed_chunk_cache.misses + data.raw_chunk_cache.misses;
                    const ratio = (totHits + totMisses) > 0 ? ((totHits / (totHits + totMisses)) * 100).toFixed(1) : '0.0';
                    document.getElementById('stat-ratio').innerText = ratio + '%';
                }
            } catch (e) {
                // Ignore background polling errors
            }
        }

        // Attach event listeners for real-time live preview
        document.querySelectorAll('input, select').forEach(el => {
            el.addEventListener('input', updateConfig);
            el.addEventListener('change', updateConfig);
        });

        // Initialize on load
        updateConfig();
        setInterval(refreshStats, 3000);
        refreshStats();
    </script>
</body>
</html>
"""
