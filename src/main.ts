import './style.css';

const WHEEL_ZOOM_FACTOR = 1.08;
const MIN_ZOOM = 1e-3;
// Double-single (hi/lo float32) arithmetic stays pixel-accurate to roughly
// 2^-48 relative precision; beyond this zoom the image would degrade.
const MAX_ZOOM = 1e10;
// Below this zoom plain float32 iteration is visually identical to
// double-single and several times faster.
const FLOAT_PRECISION_MAX_ZOOM = 1e4;
const MAX_ITERATIONS = 4096;
const HUD_UPDATE_INTERVAL_MS = 200;

const vertexSrc = `#version 300 es
    in vec2 aPosition;
    out vec2 vPosition;

    void main() {
        vPosition = aPosition;
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`;

const fragmentSrc = `#version 300 es
    precision highp float;

    in vec2 vPosition;
    out vec4 finalColor;

    uniform vec2 uResolution;
    uniform vec2 uCenterX;
    uniform vec2 uCenterY;
    uniform vec2 uInvZoom;
    uniform int uMaxIterations;
    uniform int uColorIterations;
    uniform int uPrecisionMode;
    uniform float uTime;
    uniform vec2 uC;
    uniform vec2 uZ0;
    uniform float uModeBlend;
    uniform float uExponent;

    vec2 complexPow(vec2 z, float p) {
        float r2 = z.x * z.x + z.y * z.y;
        if (r2 < 1e-20) return vec2(0.0, 0.0);
        float r = sqrt(r2);
        float theta = atan(z.y, z.x);
        float rp = pow(r, p);
        float pTheta = p * theta;
        return vec2(rp * cos(pTheta), rp * sin(pTheta));
    }

    vec2 twoSum(float a, float b) {
        float s = a + b;
        float bb = s - a;
        float err = (a - (s - bb)) + (b - bb);
        return vec2(s, err);
    }

    vec2 dsAdd(vec2 a, vec2 b) {
        vec2 s = twoSum(a.x, b.x);
        float e = a.y + b.y + s.y;
        return twoSum(s.x, e);
    }

    vec2 twoProd(float a, float b) {
        float p = a * b;
        const float split = 4097.0;
        float aSplit = a * split;
        float aHi = aSplit - (aSplit - a);
        float aLo = a - aHi;
        float bSplit = b * split;
        float bHi = bSplit - (bSplit - b);
        float bLo = b - bHi;
        float err = ((aHi * bHi - p) + aHi * bLo + aLo * bHi) + aLo * bLo;
        return vec2(p, err);
    }

    vec2 dsMulFloat(vec2 a, float b) {
        vec2 p = twoProd(a.x, b);
        float e = a.y * b + p.y;
        return twoSum(p.x, e);
    }

    vec3 gradient(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 black = vec3(0.0, 0.0, 0.0);
        float fade = smoothstep(0.0, 0.18, t);
        float tc = (t - 0.05) / 0.95;
        tc = clamp(tc, 0.0, 1.0);
        float phase = uTime * 0.1;
        vec3 a = vec3(0.5, 0.5, 0.5);
        vec3 b = vec3(0.5, 0.5, 0.5);
        vec3 c = vec3(1.0, 1.0, 1.0);
        vec3 d = vec3(0.0, 0.33, 0.67) + phase;
        vec3 color = a + b * cos(6.28318 * (c * tc + d));
        return mix(black, color, fade);
    }

    vec4 pixelToComplexDS(vec2 pixelPos) {
        float dx = pixelPos.x - uResolution.x * 0.5;
        float dy = uResolution.y * 0.5 - pixelPos.y;
        vec2 cx = dsAdd(uCenterX, dsMulFloat(uInvZoom, dx));
        vec2 cy = dsAdd(uCenterY, dsMulFloat(uInvZoom, dy));
        return vec4(cx.x, cx.y, cy.x, cy.y);
    }

    float iterSmooth(vec2 z0, vec2 c, int limit) {
        vec2 z = z0;
        float iter = 0.0;

        for (int i = 0; i < ${MAX_ITERATIONS}; i++) {
            if (i >= limit) break;
            vec2 zp = complexPow(z, uExponent);
            float x = zp.x + c.x;
            float y = zp.y + c.y;
            float mag2 = x * x + y * y;
            if (mag2 > 4.0) {
                float safeMag2 = max(mag2, 4.000001);
                return iter + 1.0 - log2(log2(safeMag2));
            }
            z = vec2(x, y);
            iter += 1.0;
        }
        return -1.0;
    }

    vec2 dsComplexPow(vec2 zx, vec2 zy, float p) {
        float x = zx.x + zx.y;
        float y = zy.x + zy.y;
        return complexPow(vec2(x, y), p);
    }

    float iterSmoothDS(vec4 z0Full, vec4 cFull, int limit) {
        vec2 zx = z0Full.xy;
        vec2 zy = z0Full.zw;
        vec2 cx = cFull.xy;
        vec2 cy = cFull.zw;
        float iter = 0.0;

        for (int i = 0; i < ${MAX_ITERATIONS}; i++) {
            if (i >= limit) break;
            vec2 zp = dsComplexPow(zx, zy, uExponent);
            vec2 nzx = dsAdd(vec2(zp.x, 0.0), cx);
            vec2 nzy = dsAdd(vec2(zp.y, 0.0), cy);
            float x = nzx.x + nzx.y;
            float y = nzy.x + nzy.y;
            float mag2 = x * x + y * y;
            if (mag2 > 4.0) {
                float safeMag2 = max(mag2, 4.000001);
                return iter + 1.0 - log2(log2(safeMag2));
            }
            zx = nzx;
            zy = nzy;
            iter += 1.0;
        }
        return -1.0;
    }

    vec3 colorFromSmooth(float smoothIter) {
        if (smoothIter < 0.0) return vec3(0.0);
        float colorIter = max(1.0, float(uColorIterations));
        float t = pow(clamp(smoothIter / colorIter, 0.0, 1.0), 0.6);
        return gradient(t);
    }

    void main() {
        vec2 screenPos = vPosition * 0.5 + 0.5;
        screenPos.y = 1.0 - screenPos.y;
        vec2 pixelPos = screenPos * uResolution;
        int limit = clamp(uMaxIterations, 1, ${MAX_ITERATIONS});

        vec4 pixelPosDS = pixelToComplexDS(pixelPos);
        vec2 pixelC = vec2(pixelPosDS.x + pixelPosDS.y, pixelPosDS.z + pixelPosDS.w);

        float s;
        if (uModeBlend <= 0.001) {
            // Mandelbrot: z0 fixed, c varies per pixel.
            s = uPrecisionMode == 0
                ? iterSmooth(uZ0, pixelC, limit)
                : iterSmoothDS(vec4(uZ0.x, 0.0, uZ0.y, 0.0), pixelPosDS, limit);
        } else if (uModeBlend >= 0.999) {
            // Julia: z0 varies per pixel, c fixed.
            s = uPrecisionMode == 0
                ? iterSmooth(pixelC, uC, limit)
                : iterSmoothDS(pixelPosDS, vec4(uC.x, 0.0, uC.y, 0.0), limit);
        } else {
            vec2 z0 = uModeBlend * pixelC;
            vec2 c = (1.0 - uModeBlend) * pixelC + uModeBlend * uC;
            if (uPrecisionMode == 0) {
                s = iterSmooth(z0, c, limit);
            } else {
                vec2 z0x = dsMulFloat(pixelPosDS.xy, uModeBlend);
                vec2 z0y = dsMulFloat(pixelPosDS.zw, uModeBlend);
                vec2 cx = dsAdd(dsMulFloat(pixelPosDS.xy, 1.0 - uModeBlend), vec2(uModeBlend * uC.x, 0.0));
                vec2 cy = dsAdd(dsMulFloat(pixelPosDS.zw, 1.0 - uModeBlend), vec2(uModeBlend * uC.y, 0.0));
                s = iterSmoothDS(vec4(z0x, z0y), vec4(cx, cy), limit);
            }
        }

        finalColor = vec4(colorFromSmooth(s), 1.0);
    }
`;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Shader compile error: ${log}`);
    }
    return shader;
}

function createProgram(gl: WebGL2RenderingContext): WebGLProgram {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(`Program link error: ${log}`);
    }
    return program;
}

const loader = document.getElementById('loader');

function hideLoader() {
    if (!loader) return;
    loader.classList.add('loader-hidden');
    loader.addEventListener('transitionend', () => loader.remove(), { once: true });
}

function showLoaderError(message: string) {
    if (!loader) return;
    loader.innerHTML = `<div class="loader-error">${message}</div>`;
}

function init() {
    type DoubleDouble = [number, number];

    const splitDouble = (value: number): [number, number] => {
        const hi = Math.fround(value);
        return [hi, value - hi];
    };

    const twoSum = (a: number, b: number): DoubleDouble => {
        const s = a + b;
        const bb = s - a;
        const err = (a - (s - bb)) + (b - bb);
        return [s, err];
    };

    const ddAdd = (a: DoubleDouble, b: DoubleDouble): DoubleDouble => {
        const s = twoSum(a[0], b[0]);
        const e = a[1] + b[1] + s[1];
        return twoSum(s[0], e);
    };

    const ddFromNumber = (value: number): DoubleDouble => splitDouble(value);

    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    canvas.style.touchAction = 'none';
    document.getElementById('app')!.appendChild(canvas);

    const gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: false,
        powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not supported by this browser');

    const program = createProgram(gl);

    const uniformNames = [
        'uResolution', 'uCenterX', 'uCenterY', 'uInvZoom',
        'uMaxIterations', 'uColorIterations', 'uPrecisionMode',
        'uTime', 'uC', 'uZ0', 'uModeBlend', 'uExponent',
    ] as const;
    const loc: Record<(typeof uniformNames)[number], WebGLUniformLocation | null> = Object.fromEntries(
        uniformNames.map((name) => [name, gl.getUniformLocation(program, name)]),
    ) as Record<(typeof uniformNames)[number], WebGLUniformLocation | null>;

    const positionLoc = gl.getAttribLocation(program, 'aPosition');
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const positions = new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]);
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    gl.useProgram(program);

    const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        const w = Math.floor(window.innerWidth * dpr);
        const h = Math.floor(window.innerHeight * dpr);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
            canvas.style.width = `${window.innerWidth}px`;
            canvas.style.height = `${window.innerHeight}px`;
            gl.viewport(0, 0, w, h);
        }
    };
    resize();

    const baseZoom = Math.min(canvas.width, canvas.height) / 3.5;

    const controlsPanel = document.createElement('details');
    controlsPanel.className = 'hud-panel controls-panel';
    controlsPanel.open = true;
    controlsPanel.innerHTML = `
        <summary>Controls</summary>
        <div class="controls-body">
            <p class="controls-group-label">Desktop</p>
            <ul class="controls-list">
                <li><span class="control-key">Scroll</span><span class="control-desc">Zoom in/out</span></li>
                <li><span class="control-key">Middle drag</span><span class="control-desc">Pan camera</span></li>
                <li><span class="control-key">Left drag</span><span class="control-desc">Adjust active parameter (Julia C / Mandelbrot z0)</span></li>
                <li><span class="control-key">Right drag</span><span class="control-desc">Blend mode (X) and exponent (Y)</span></li>
                <li><span class="control-key">Double-click</span><span class="control-desc">Release manual C / z0 control</span></li>
                <li><span class="control-key">Space</span><span class="control-desc">Pause or resume animation</span></li>
            </ul>
            <p class="controls-group-label">Mobile</p>
            <ul class="controls-list">
                <li><span class="control-key">1 finger</span><span class="control-desc">Same as left drag</span></li>
                <li><span class="control-key">2 fingers</span><span class="control-desc">Same as right drag</span></li>
                <li><span class="control-key">Pinch</span><span class="control-desc">Zoom</span></li>
            </ul>
        </div>
    `;
    document.body.appendChild(controlsPanel);

    const paramsPanel = document.createElement('details');
    paramsPanel.className = 'hud-panel params-panel';
    paramsPanel.open = true;
    const paramsTitle = document.createElement('summary');
    paramsTitle.textContent = 'Live Parameters';
    paramsPanel.appendChild(paramsTitle);
    const paramsBody = document.createElement('div');
    paramsBody.className = 'params-body';
    paramsPanel.appendChild(paramsBody);
    document.body.appendChild(paramsPanel);

    const createParamRow = (label: string) => {
        const row = document.createElement('div');
        row.className = 'param-row';
        const labelEl = document.createElement('span');
        labelEl.className = 'param-label';
        labelEl.textContent = label;
        const bar = document.createElement('div');
        bar.className = 'param-bar';
        const fill = document.createElement('div');
        fill.className = 'param-bar-fill';
        bar.appendChild(fill);
        const valueEl = document.createElement('span');
        valueEl.className = 'param-value';
        valueEl.textContent = '0.00';
        row.appendChild(labelEl);
        row.appendChild(bar);
        row.appendChild(valueEl);
        paramsBody.appendChild(row);
        return {
            set(value01: number, text: string) {
                const clamped = Math.max(0, Math.min(1, value01));
                fill.style.width = `${(clamped * 100).toFixed(1)}%`;
                valueEl.textContent = text;
            },
        };
    };

    const paramRows = {
        modeBlend: createParamRow('Mode blend'),
        exponent: createParamRow('Exponent'),
        cReal: createParamRow('C real'),
        cImag: createParamRow('C imag'),
        z0Real: createParamRow('z0 real'),
        z0Imag: createParamRow('z0 imag'),
        zoom: createParamRow('Zoom'),
        iterations: createParamRow('Iterations'),
        fps: createParamRow('FPS'),
    };

    const normalize = (value: number, min: number, max: number) => {
        if (max <= min) return 0;
        return (value - min) / (max - min);
    };

    let cameraCenterXDD: DoubleDouble = ddFromNumber(-0.5);
    let cameraCenterYDD: DoubleDouble = ddFromNumber(0);
    let cameraZoom = baseZoom;

    const uniforms = {
        uResolution: new Float32Array([window.innerWidth, window.innerHeight]),
        uCenterX: new Float32Array([cameraCenterXDD[0], cameraCenterXDD[1]]),
        uCenterY: new Float32Array([cameraCenterYDD[0], cameraCenterYDD[1]]),
        uInvZoom: new Float32Array(splitDouble(1 / cameraZoom)),
        uMaxIterations: 200,
        uColorIterations: 320,
        uPrecisionMode: 0,
        uTime: 0,
        uC: new Float32Array([0.285, 0.01]),
        uZ0: new Float32Array([0, 0]),
        uModeBlend: 0,
        uExponent: 2,
    };

    const getCanvasCoords = (clientX: number, clientY: number) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    };

    const isMobileDevice = window.matchMedia('(pointer: coarse)').matches
        || navigator.maxTouchPoints > 0
        || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const getTouchDistance = (a: Touch, b: Touch) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const getTouchMidpoint = (a: Touch, b: Touch) => ({
        x: (a.clientX + b.clientX) * 0.5,
        y: (a.clientY + b.clientY) * 0.5,
    });

    let mouseControlC: { real: number; imag: number } | null = null;
    let mouseControlZ0: { real: number; imag: number } | null = null;
    const paramSensitivity = 0.0008;
    const cLimits = { realMin: -2, realMax: 1, imagMin: -1.5, imagMax: 1.5 };
    const z0Limits = { realMin: -2, realMax: 2, imagMin: -2, imagMax: 2 };
    let dragStart: {
        x: number;
        y: number;
        real: number;
        imag: number;
        isJulia: boolean;
    } | null = null;
    let panStart: { x: number; y: number; centerX: DoubleDouble; centerY: DoubleDouble } | null = null;
    let blendDragStart: { x: number; y: number; blend: number; exponent: number } | null = null;
    let modeBlend = 0;
    let exponent = 2;
    let animationPaused = true;
    let phaseOffsetCRe = 0;
    let phaseOffsetCIm = 0;
    let phaseOffsetZ0Re = 0;
    let phaseOffsetZ0Im = 0;
    let phaseOffsetBlend = 0;
    let phaseOffsetExp = 0;
    let lastFrameUsedFormulaC = false;
    let lastFrameUsedFormulaZ0 = false;
    let lastFrameUsedFormulaBlend = false;
    const blendSensitivity = 0.0015;
    const exponentSensitivity = 0.003;

    let pinchStart: { distance: number; zoom: number } | null = null;
    let touchTwoFingerStart: { x: number; y: number; blend: number; exponent: number } | null = null;

    const onPointerDown = (e: PointerEvent) => {
        if (e.pointerType === 'touch') return;
        const { x, y } = getCanvasCoords(e.clientX, e.clientY);
        if (e.button === 0) {
            if (e.detail === 2) {
                mouseControlC = null;
                mouseControlZ0 = null;
                dragStart = null;
            } else {
                const isJulia = modeBlend > 0.5;
                const lim = isJulia ? cLimits : z0Limits;
                const rawReal = isJulia
                    ? (mouseControlC?.real ?? uniforms.uC[0])
                    : (mouseControlZ0?.real ?? uniforms.uZ0[0]);
                const rawImag = isJulia
                    ? (mouseControlC?.imag ?? uniforms.uC[1])
                    : (mouseControlZ0?.imag ?? uniforms.uZ0[1]);
                const real = Math.max(lim.realMin, Math.min(lim.realMax, rawReal));
                const imag = Math.max(lim.imagMin, Math.min(lim.imagMax, rawImag));
                dragStart = { x, y, real, imag, isJulia };
                if (isJulia) {
                    mouseControlC = { real, imag };
                } else {
                    mouseControlZ0 = { real, imag };
                }
                animationPaused = true;
            }
        } else if (e.button === 1) {
            panStart = {
                x,
                y,
                centerX: [cameraCenterXDD[0], cameraCenterXDD[1]],
                centerY: [cameraCenterYDD[0], cameraCenterYDD[1]],
            };
            animationPaused = true;
        } else if (e.button === 2) {
            blendDragStart = { x, y, blend: modeBlend, exponent };
            animationPaused = true;
        }
    };
    const onPointerMove = (e: PointerEvent) => {
        if (e.pointerType === 'touch') return;
        const { x, y } = getCanvasCoords(e.clientX, e.clientY);
        if (e.buttons & 1 && dragStart) {
            const dx = x - dragStart.x;
            const dy = y - dragStart.y;
            const rawReal = dragStart.real + paramSensitivity * dx;
            const rawImag = dragStart.imag - paramSensitivity * dy;
            if (dragStart.isJulia) {
                mouseControlC = {
                    real: Math.max(cLimits.realMin, Math.min(cLimits.realMax, rawReal)),
                    imag: Math.max(cLimits.imagMin, Math.min(cLimits.imagMax, rawImag)),
                };
            } else {
                mouseControlZ0 = {
                    real: Math.max(z0Limits.realMin, Math.min(z0Limits.realMax, rawReal)),
                    imag: Math.max(z0Limits.imagMin, Math.min(z0Limits.imagMax, rawImag)),
                };
            }
        } else if ((e.buttons & 4) && panStart) {
            const dx = x - panStart.x;
            const dy = y - panStart.y;
            cameraCenterXDD = ddAdd(panStart.centerX, ddFromNumber(-dx / cameraZoom));
            cameraCenterYDD = ddAdd(panStart.centerY, ddFromNumber(dy / cameraZoom));
        } else if ((e.buttons & 2) && blendDragStart) {
            const dx = x - blendDragStart.x;
            const dy = y - blendDragStart.y;
            modeBlend = Math.max(0, Math.min(1, blendDragStart.blend + blendSensitivity * dx));
            exponent = Math.max(1.01, Math.min(8, blendDragStart.exponent - exponentSensitivity * dy));
        }
    };
    const onPointerUp = (e: PointerEvent) => {
        if (e.pointerType === 'touch') return;
        if (e.button === 0) dragStart = null;
        if (e.button === 1) panStart = null;
        if (e.button === 2) blendDragStart = null;
    };
    const onPointerLeave = () => {
        blendDragStart = null;
        panStart = null;
    };

    const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const { x, y } = getCanvasCoords(e.clientX, e.clientY);
        const dx = x - canvas.width * 0.5;
        const dy = canvas.height * 0.5 - y;
        const zoomFactor = e.deltaY > 0 ? 1 / WHEEL_ZOOM_FACTOR : WHEEL_ZOOM_FACTOR;
        const k = Math.pow(zoomFactor, Math.min(3, Math.abs(e.deltaY) / 50));
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, cameraZoom * k));
        const invZoom = 1 / cameraZoom;
        const invNewZoom = 1 / newZoom;
        cameraCenterXDD = ddAdd(cameraCenterXDD, ddFromNumber(dx * (invZoom - invNewZoom)));
        cameraCenterYDD = ddAdd(cameraCenterYDD, ddFromNumber(dy * (invZoom - invNewZoom)));
        cameraZoom = newZoom;
    };

    const onTouchStart = (e: TouchEvent) => {
        if (!isMobileDevice) return;
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
            const isJulia = modeBlend > 0.5;
            const lim = isJulia ? cLimits : z0Limits;
            const rawReal = isJulia
                ? (mouseControlC?.real ?? uniforms.uC[0])
                : (mouseControlZ0?.real ?? uniforms.uZ0[0]);
            const rawImag = isJulia
                ? (mouseControlC?.imag ?? uniforms.uC[1])
                : (mouseControlZ0?.imag ?? uniforms.uZ0[1]);
            const real = Math.max(lim.realMin, Math.min(lim.realMax, rawReal));
            const imag = Math.max(lim.imagMin, Math.min(lim.imagMax, rawImag));
            dragStart = { x, y, real, imag, isJulia };
            if (isJulia) {
                mouseControlC = { real, imag };
            } else {
                mouseControlZ0 = { real, imag };
            }
            blendDragStart = null;
            touchTwoFingerStart = null;
            pinchStart = null;
            animationPaused = true;
        } else if (e.touches.length >= 2) {
            const a = e.touches[0];
            const b = e.touches[1];
            const midpoint = getTouchMidpoint(a, b);
            const { x, y } = getCanvasCoords(midpoint.x, midpoint.y);
            touchTwoFingerStart = { x, y, blend: modeBlend, exponent };
            blendDragStart = { x, y, blend: modeBlend, exponent };
            pinchStart = { distance: Math.max(1, getTouchDistance(a, b)), zoom: cameraZoom };
            dragStart = null;
            animationPaused = true;
        }
        e.preventDefault();
    };

    const onTouchMove = (e: TouchEvent) => {
        if (!isMobileDevice) return;
        if (e.touches.length === 1 && dragStart) {
            const touch = e.touches[0];
            const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
            const dx = x - dragStart.x;
            const dy = y - dragStart.y;
            const rawReal = dragStart.real + paramSensitivity * dx;
            const rawImag = dragStart.imag - paramSensitivity * dy;
            if (dragStart.isJulia) {
                mouseControlC = {
                    real: Math.max(cLimits.realMin, Math.min(cLimits.realMax, rawReal)),
                    imag: Math.max(cLimits.imagMin, Math.min(cLimits.imagMax, rawImag)),
                };
            } else {
                mouseControlZ0 = {
                    real: Math.max(z0Limits.realMin, Math.min(z0Limits.realMax, rawReal)),
                    imag: Math.max(z0Limits.imagMin, Math.min(z0Limits.imagMax, rawImag)),
                };
            }
        } else if (e.touches.length >= 2) {
            const a = e.touches[0];
            const b = e.touches[1];

            if (!touchTwoFingerStart) {
                const midpoint = getTouchMidpoint(a, b);
                const { x, y } = getCanvasCoords(midpoint.x, midpoint.y);
                touchTwoFingerStart = { x, y, blend: modeBlend, exponent };
                blendDragStart = { x, y, blend: modeBlend, exponent };
            }
            if (!pinchStart) {
                pinchStart = { distance: Math.max(1, getTouchDistance(a, b)), zoom: cameraZoom };
            }

            const midpoint = getTouchMidpoint(a, b);
            const { x, y } = getCanvasCoords(midpoint.x, midpoint.y);
            const dx = x - touchTwoFingerStart.x;
            const dy = y - touchTwoFingerStart.y;
            modeBlend = Math.max(0, Math.min(1, touchTwoFingerStart.blend + blendSensitivity * dx));
            exponent = Math.max(1.01, Math.min(8, touchTwoFingerStart.exponent - exponentSensitivity * dy));

            const distance = Math.max(1, getTouchDistance(a, b));
            const zoomScale = distance / pinchStart.distance;
            cameraZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, pinchStart.zoom * zoomScale));
        }
        e.preventDefault();
    };

    const onTouchEnd = (e: TouchEvent) => {
        if (!isMobileDevice) return;
        if (e.touches.length === 0) {
            dragStart = null;
            blendDragStart = null;
            touchTwoFingerStart = null;
            pinchStart = null;
        } else if (e.touches.length === 1) {
            blendDragStart = null;
            touchTwoFingerStart = null;
            pinchStart = null;
            const touch = e.touches[0];
            const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
            const isJulia = modeBlend > 0.5;
            const lim = isJulia ? cLimits : z0Limits;
            const rawReal = isJulia
                ? (mouseControlC?.real ?? uniforms.uC[0])
                : (mouseControlZ0?.real ?? uniforms.uZ0[0]);
            const rawImag = isJulia
                ? (mouseControlC?.imag ?? uniforms.uC[1])
                : (mouseControlZ0?.imag ?? uniforms.uZ0[1]);
            const real = Math.max(lim.realMin, Math.min(lim.realMax, rawReal));
            const imag = Math.max(lim.imagMin, Math.min(lim.imagMax, rawImag));
            dragStart = { x, y, real, imag, isJulia };
        }
        e.preventDefault();
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('touchstart', onTouchStart, { passive: false });
    canvas.addEventListener('touchmove', onTouchMove, { passive: false });
    canvas.addEventListener('touchend', onTouchEnd, { passive: false });
    canvas.addEventListener('touchcancel', onTouchEnd, { passive: false });

    window.addEventListener('keydown', (e: KeyboardEvent) => {
        if (e.code === 'Space') {
            e.preventDefault();
            animationPaused = !animationPaused;
        }
    });

    const setUniforms = () => {
        gl.uniform2fv(loc.uResolution, uniforms.uResolution);
        gl.uniform2fv(loc.uCenterX, uniforms.uCenterX);
        gl.uniform2fv(loc.uCenterY, uniforms.uCenterY);
        gl.uniform2fv(loc.uInvZoom, uniforms.uInvZoom);
        gl.uniform1i(loc.uMaxIterations, uniforms.uMaxIterations);
        gl.uniform1i(loc.uColorIterations, uniforms.uColorIterations);
        gl.uniform1i(loc.uPrecisionMode, uniforms.uPrecisionMode);
        gl.uniform1f(loc.uTime, uniforms.uTime);
        gl.uniform2fv(loc.uC, uniforms.uC);
        gl.uniform2fv(loc.uZ0, uniforms.uZ0);
        gl.uniform1f(loc.uModeBlend, uniforms.uModeBlend);
        gl.uniform1f(loc.uExponent, uniforms.uExponent);
    };

    let smoothedFrameMs = 1000 / 60;
    let lastFrameTime = performance.now();
    let lastHudUpdateAt = 0;
    let firstFrameDrawn = false;

    const tick = () => {
        if (document.hidden) {
            requestAnimationFrame(tick);
            return;
        }
        const nowMs = performance.now();
        const frameMs = nowMs - lastFrameTime;
        lastFrameTime = nowMs;
        smoothedFrameMs = smoothedFrameMs * 0.95 + frameMs * 0.05;

        uniforms.uResolution[0] = canvas.width;
        uniforms.uResolution[1] = canvas.height;
        uniforms.uCenterX[0] = cameraCenterXDD[0];
        uniforms.uCenterX[1] = cameraCenterXDD[1];
        uniforms.uCenterY[0] = cameraCenterYDD[0];
        uniforms.uCenterY[1] = cameraCenterYDD[1];
        const [invZoomHi, invZoomLo] = splitDouble(1 / cameraZoom);
        uniforms.uInvZoom[0] = invZoomHi;
        uniforms.uInvZoom[1] = invZoomLo;
        const t = nowMs * 0.001;
        uniforms.uTime = t;

        const wobblePeriod = 90.0;
        const wobbleOmega = (2 * Math.PI) / wobblePeriod;
        const wobbleAmp = 0.4;

        if (mouseControlC !== null) {
            uniforms.uC[0] = Math.max(cLimits.realMin, Math.min(cLimits.realMax, mouseControlC.real));
            uniforms.uC[1] = Math.max(cLimits.imagMin, Math.min(cLimits.imagMax, mouseControlC.imag));
        } else if (!animationPaused) {
            const periodRe = 45.0;
            const periodIm = 30.0;
            const omegaRe = (2 * Math.PI) / periodRe;
            const omegaIm = (2 * Math.PI) / periodIm;
            const cAmp = 0.06;
            const cBase = 0.285;
            const basePhaseRe = omegaRe * t + wobbleAmp * Math.sin(wobbleOmega * t);
            const basePhaseIm = omegaIm * t + wobbleAmp * Math.sin(wobbleOmega * t * 1.3);
            if (!lastFrameUsedFormulaC) {
                const cRe = uniforms.uC[0];
                const cIm = uniforms.uC[1];
                const cosVal = Math.max(-1, Math.min(1, (cRe - cBase) / cAmp));
                const sinVal = Math.max(-1, Math.min(1, cIm / cAmp));
                phaseOffsetCRe = Math.acos(cosVal) - basePhaseRe;
                phaseOffsetCIm = Math.asin(sinVal) - basePhaseIm;
            }
            const phaseRe = basePhaseRe + phaseOffsetCRe;
            const phaseIm = basePhaseIm + phaseOffsetCIm;
            const cRe = cBase + cAmp * Math.cos(phaseRe);
            const cIm = cAmp * Math.sin(phaseIm);
            uniforms.uC[0] = Math.max(cLimits.realMin, Math.min(cLimits.realMax, cRe));
            uniforms.uC[1] = Math.max(cLimits.imagMin, Math.min(cLimits.imagMax, cIm));
        }

        if (mouseControlZ0 !== null) {
            uniforms.uZ0[0] = Math.max(z0Limits.realMin, Math.min(z0Limits.realMax, mouseControlZ0.real));
            uniforms.uZ0[1] = Math.max(z0Limits.imagMin, Math.min(z0Limits.imagMax, mouseControlZ0.imag));
        } else if (!animationPaused) {
            const z0PeriodRe = 55.0;
            const z0PeriodIm = 38.0;
            const z0OmegaRe = (2 * Math.PI) / z0PeriodRe;
            const z0OmegaIm = (2 * Math.PI) / z0PeriodIm;
            const z0Amp = 0.15;
            const basePhaseZ0Re = z0OmegaRe * t + wobbleAmp * Math.sin(wobbleOmega * t * 0.8);
            const basePhaseZ0Im = z0OmegaIm * t + wobbleAmp * Math.sin(wobbleOmega * t * 1.1);
            if (!lastFrameUsedFormulaZ0) {
                const z0Re = uniforms.uZ0[0];
                const z0Im = uniforms.uZ0[1];
                const cosVal = Math.max(-1, Math.min(1, z0Re / z0Amp));
                const sinVal = Math.max(-1, Math.min(1, z0Im / z0Amp));
                phaseOffsetZ0Re = Math.acos(cosVal) - basePhaseZ0Re;
                phaseOffsetZ0Im = Math.asin(sinVal) - basePhaseZ0Im;
            }
            const phaseZ0Re = basePhaseZ0Re + phaseOffsetZ0Re;
            const phaseZ0Im = basePhaseZ0Im + phaseOffsetZ0Im;
            const z0Re = z0Amp * Math.cos(phaseZ0Re);
            const z0Im = z0Amp * Math.sin(phaseZ0Im);
            uniforms.uZ0[0] = Math.max(z0Limits.realMin, Math.min(z0Limits.realMax, z0Re));
            uniforms.uZ0[1] = Math.max(z0Limits.imagMin, Math.min(z0Limits.imagMax, z0Im));
        }

        if (!animationPaused && blendDragStart === null) {
            const blendPeriod = 120.0;
            const expPeriod = 90.0;
            const blendPhase = t * (2 * Math.PI) / blendPeriod;
            const expPhase = t * (2 * Math.PI) / expPeriod + 0.5;
            if (!lastFrameUsedFormulaBlend) {
                const blendSinVal = Math.max(-1, Math.min(1, 2 * modeBlend - 1));
                const expInner = (exponent - 2) / 1.5 - 0.5;
                const expSinVal = Math.max(-1, Math.min(1, 2 * expInner));
                phaseOffsetBlend = Math.asin(blendSinVal) - blendPhase;
                phaseOffsetExp = Math.asin(expSinVal) - expPhase;
            }
            modeBlend = Math.max(0, Math.min(1, 0.5 + 0.5 * Math.sin(blendPhase + phaseOffsetBlend)));
            exponent = Math.max(1.01, Math.min(8, 2 + 1.5 * (0.5 + 0.5 * Math.sin(expPhase + phaseOffsetExp))));
        }
        uniforms.uModeBlend = modeBlend;
        uniforms.uExponent = exponent;
        lastFrameUsedFormulaC = mouseControlC === null && !animationPaused;
        lastFrameUsedFormulaZ0 = mouseControlZ0 === null && !animationPaused;
        lastFrameUsedFormulaBlend = !animationPaused && blendDragStart === null;

        // Deterministic quality: same zoom always renders identically.
        const zoom = cameraZoom;
        const iterBase = Math.min(MAX_ITERATIONS, Math.floor(50 + Math.log2(Math.max(1, zoom)) * 60));
        uniforms.uMaxIterations = iterBase;
        uniforms.uColorIterations = iterBase;
        uniforms.uPrecisionMode = zoom > FLOAT_PRECISION_MAX_ZOOM ? 1 : 0;

        if (nowMs - lastHudUpdateAt >= HUD_UPDATE_INTERVAL_MS) {
            lastHudUpdateAt = nowMs;
            const cReal = uniforms.uC[0];
            const cImag = uniforms.uC[1];
            const z0Real = uniforms.uZ0[0];
            const z0Imag = uniforms.uZ0[1];
            paramRows.modeBlend.set(modeBlend, modeBlend.toFixed(2));
            paramRows.exponent.set(normalize(exponent, 1.01, 8), exponent.toFixed(2));
            paramRows.cReal.set(normalize(cReal, cLimits.realMin, cLimits.realMax), cReal.toFixed(3));
            paramRows.cImag.set(normalize(cImag, cLimits.imagMin, cLimits.imagMax), cImag.toFixed(3));
            paramRows.z0Real.set(normalize(z0Real, z0Limits.realMin, z0Limits.realMax), z0Real.toFixed(3));
            paramRows.z0Imag.set(normalize(z0Imag, z0Limits.imagMin, z0Limits.imagMax), z0Imag.toFixed(3));
            const logZoom = Math.log10(Math.max(1, zoom));
            paramRows.zoom.set(Math.min(1, logZoom / Math.log10(MAX_ZOOM)), zoom.toExponential(2));
            paramRows.iterations.set(iterBase / MAX_ITERATIONS, iterBase.toString());
            const fps = 1000 / Math.max(0.0001, smoothedFrameMs);
            paramRows.fps.set(Math.min(1, fps / 120), fps.toFixed(1));
        }

        setUniforms();
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

        if (!firstFrameDrawn) {
            firstFrameDrawn = true;
            hideLoader();
        }

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);

    window.addEventListener('resize', () => {
        const oldMin = Math.min(canvas.width, canvas.height);
        resize();
        const newMin = Math.min(canvas.width, canvas.height);
        if (oldMin > 0 && newMin > 0 && oldMin !== newMin) {
            cameraZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, cameraZoom * (newMin / oldMin)));
        }
    });
}

try {
    init();
} catch (error) {
    console.error(error);
    showLoaderError(error instanceof Error ? error.message : 'Failed to start the fractal renderer');
}
