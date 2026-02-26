import { Application, Geometry, Mesh, Shader, GlProgram, UniformGroup } from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import './style.css';

const ORBIT_CAPACITY = 256;
const ORBIT_CACHE_LIMIT = 12;
const TARGET_FPS = 58;
const QUALITY_MIN = 0.35;
const QUALITY_MAX = 2.2;
const INTERACTION_RECALC_DEBOUNCE_MS = 48;

const vertexSrc = `
    in vec2 aPosition;
    out vec2 vPosition;

    void main() {
        vPosition = aPosition;
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`;

const fragmentSrc = `
    precision highp float;

    in vec2 vPosition;
    out vec4 finalColor;

    uniform vec2 uResolution;
    uniform vec2 uCenterX;
    uniform vec2 uCenterY;
    uniform vec2 uInvZoom;
    uniform int uMaxIterations;
    uniform int uColorIterations;
    uniform int uLod;
    uniform int uPrecisionMode;
    uniform int uUsePerturb;
    uniform float uPerturbBlend;
    uniform int uRefLength;
    uniform vec2 uRefCenterX;
    uniform vec2 uRefCenterY;
    uniform vec2 uRefOrbit[256];
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

    vec2 dsAdd(vec2 a, vec2 b) {
        vec2 s = twoSum(a.x, b.x);
        float e = a.y + b.y + s.y;
        vec2 r = twoSum(s.x, e);
        return vec2(r.x, r.y);
    }

    vec2 dsMulFloat(vec2 a, float b) {
        vec2 p = twoProd(a.x, b);
        float e = a.y * b + p.y;
        vec2 r = twoSum(p.x, e);
        return vec2(r.x, r.y);
    }

    vec2 dsSub(vec2 a, vec2 b) {
        return dsAdd(a, vec2(-b.x, -b.y));
    }

    vec2 dsMul(vec2 a, vec2 b) {
        vec2 p = twoProd(a.x, b.x);
        float e = a.x * b.y + a.y * b.x + p.y;
        vec2 r = twoSum(p.x, e);
        return vec2(r.x, r.y);
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

    vec2 pixelToComplex(vec2 pixelPos) {
        float dx = pixelPos.x - uResolution.x * 0.5;
        float dy = uResolution.y * 0.5 - pixelPos.y;
        vec2 cxDs = dsAdd(uCenterX, dsMulFloat(uInvZoom, dx));
        vec2 cyDs = dsAdd(uCenterY, dsMulFloat(uInvZoom, dy));
        return vec2(cxDs.x + cxDs.y, cyDs.x + cyDs.y);
    }

    vec4 pixelToComplexDS(vec2 pixelPos) {
        float dx = pixelPos.x - uResolution.x * 0.5;
        float dy = uResolution.y * 0.5 - pixelPos.y;
        vec2 cx = dsAdd(uCenterX, dsMulFloat(uInvZoom, dx));
        vec2 cy = dsAdd(uCenterY, dsMulFloat(uInvZoom, dy));
        return vec4(cx.x, cx.y, cy.x, cy.y);
    }

    float iterSmooth(vec2 z0, vec2 c, float maxIter) {
        vec2 z = z0;
        float iter = 0.0;
        int limit = int(maxIter);
        if (limit < 1) limit = 1;
        if (limit > uMaxIterations) limit = uMaxIterations;

        for (int i = 0; i < 4096; i++) {
            if (i >= limit) break;
            vec2 zp = complexPow(z, uExponent);
            float x = zp.x + c.x;
            float y = zp.y + c.y;
            if ((x * x + y * y) > 4.0) {
                float mag2 = max(x * x + y * y, 4.000001);
                return iter + 1.0 - log2(log2(mag2));
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

    float iterSmoothDS(vec4 z0Full, vec4 cFull, float maxIter) {
        vec2 zx = z0Full.xy;
        vec2 zy = z0Full.zw;
        vec2 cx = cFull.xy;
        vec2 cy = cFull.zw;
        float iter = 0.0;
        int limit = int(maxIter);
        if (limit < 1) limit = 1;
        if (limit > uMaxIterations) limit = uMaxIterations;

        for (int i = 0; i < 4096; i++) {
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

    float mandelbrotSmooth(vec2 c, float maxIter) {
        return iterSmooth(vec2(0.0, 0.0), c, maxIter);
    }

    float juliaSmooth(vec2 z0, float maxIter) {
        vec2 z = z0;
        vec2 c = uC;
        float iter = 0.0;
        int limit = int(maxIter);
        if (limit < 1) limit = 1;
        if (limit > uMaxIterations) limit = uMaxIterations;

        for (int i = 0; i < 4096; i++) {
            if (i >= limit) break;
            vec2 zp = complexPow(z, uExponent);
            float x = zp.x + c.x;
            float y = zp.y + c.y;
            if ((x * x + y * y) > 4.0) {
                float mag2 = max(x * x + y * y, 4.000001);
                return iter + 1.0 - log2(log2(mag2));
            }
            z = vec2(x, y);
            iter += 1.0;
        }
        return -1.0;
    }

    float mandelbrotSmoothDS(vec4 cFull, float maxIter) {
        vec2 zx = vec2(0.0, 0.0);
        vec2 zy = vec2(0.0, 0.0);
        vec2 cx = cFull.xy;
        vec2 cy = cFull.zw;
        float iter = 0.0;
        int limit = int(maxIter);
        if (limit < 1) limit = 1;
        if (limit > uMaxIterations) limit = uMaxIterations;

        for (int i = 0; i < 4096; i++) {
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

    float juliaSmoothDS(vec4 z0, float maxIter) {
        vec2 zx = z0.xy;
        vec2 zy = z0.zw;
        vec2 cx = vec2(uC.x, 0.0);
        vec2 cy = vec2(uC.y, 0.0);
        float iter = 0.0;
        int limit = int(maxIter);
        if (limit < 1) limit = 1;
        if (limit > uMaxIterations) limit = uMaxIterations;

        for (int i = 0; i < 4096; i++) {
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

    vec2 complexMul(vec2 a, vec2 b) {
        return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    float mandelbrotSmoothPerturb(vec2 dc, vec4 cFull, float maxIter) {
        int maxLimit = int(maxIter);
        if (maxLimit < 1) maxLimit = 1;
        if (maxLimit > uMaxIterations) maxLimit = uMaxIterations;
        int limit = uRefLength;
        if (maxLimit < limit) limit = maxLimit;
        if (limit > 256) limit = 256;
        vec2 dz = vec2(0.0);
        float iter = 0.0;
        vec2 z = vec2(0.0);

        for (int i = 0; i < 256; i++) {
            if (i >= limit) break;
            vec2 Z = uRefOrbit[i];
            z = Z + dz;
            float mag2 = dot(z, z);
            if (mag2 > 4.0) {
                float safeMag2 = max(mag2, 4.000001);
                return iter + 1.0 - log2(log2(safeMag2));
            }
            vec2 twoZdz = 2.0 * complexMul(Z, dz);
            vec2 dz2 = complexMul(dz, dz);
            dz = twoZdz + dz2 + dc;
            iter += 1.0;
        }

        vec2 zx = vec2(z.x, 0.0);
        vec2 zy = vec2(z.y, 0.0);
        vec2 cx = cFull.xy;
        vec2 cy = cFull.zw;

        for (int i = 0; i < 4096; i++) {
            if (i >= maxLimit - limit) break;
            vec2 zx2 = dsMul(zx, zx);
            vec2 zy2 = dsMul(zy, zy);
            vec2 zxy = dsMul(zx, zy);
            vec2 nx = dsAdd(dsSub(zx2, zy2), cx);
            vec2 ny = dsAdd(dsMulFloat(zxy, 2.0), cy);
            float x = nx.x + nx.y;
            float y = ny.x + ny.y;
            float mag2 = x * x + y * y;
            if (mag2 > 4.0) {
                float safeMag2 = max(mag2, 4.000001);
                return iter + 1.0 - log2(log2(safeMag2));
            }
            zx = nx;
            zy = ny;
            iter += 1.0;
        }
        return -1.0;
    }

    vec3 sampleColorAtPixel(vec2 pixelPos, float maxIter) {
        vec4 pixelPosDS = pixelToComplexDS(pixelPos);
        vec2 pixelC = vec2(pixelPosDS.x + pixelPosDS.y, pixelPosDS.z + pixelPosDS.w);
        float sDirect;
        if (uModeBlend <= 0.001) {
            sDirect = uPrecisionMode == 0
                ? iterSmooth(uZ0, pixelC, maxIter)
                : iterSmoothDS(vec4(uZ0.x, 0.0, uZ0.y, 0.0), pixelPosDS, maxIter);
        } else if (uModeBlend >= 0.999) {
            sDirect = uPrecisionMode == 0
                ? juliaSmooth(pixelC, maxIter)
                : juliaSmoothDS(pixelPosDS, maxIter);
        } else {
            vec2 z0 = uModeBlend * pixelC;
            vec2 c = (1.0 - uModeBlend) * pixelC + uModeBlend * uC;
            sDirect = iterSmooth(z0, c, maxIter);
        }

        float s = sDirect;
        if (uModeBlend <= 0.001 && uUsePerturb == 1 && uRefLength > 0) {
            vec2 dcx = dsSub(pixelPosDS.xy, uRefCenterX);
            vec2 dcy = dsSub(pixelPosDS.zw, uRefCenterY);
            vec2 dc = vec2(dcx.x + dcx.y, dcy.x + dcy.y);
            float sPerturb = mandelbrotSmoothPerturb(dc, pixelPosDS, maxIter);
            s = mix(sDirect, sPerturb, uPerturbBlend);
        }
        return colorFromSmooth(s);
    }

    void main() {
        vec2 screenPos = vPosition * 0.5 + 0.5;
        screenPos.y = 1.0 - screenPos.y;
        vec2 pixelPos = screenPos * uResolution;
        float maxIter = float(uMaxIterations);
        vec3 color = sampleColorAtPixel(pixelPos, maxIter);

        if (uLod >= 1) {
            float j = 0.5;
            vec3 a = sampleColorAtPixel(pixelPos + vec2( j,  j), maxIter);
            vec3 b = sampleColorAtPixel(pixelPos + vec2(-j,  j), maxIter);
            vec3 c = sampleColorAtPixel(pixelPos + vec2( j, -j), maxIter);
            vec3 d = sampleColorAtPixel(pixelPos + vec2(-j, -j), maxIter);
            color = (a + b + c + d) * 0.25;
        }
        if (uLod >= 2 && uPrecisionMode == 0) {
            float j2 = 0.6;
            vec3 d = sampleColorAtPixel(pixelPos + vec2(-j2, -j2), maxIter);
            vec3 e = sampleColorAtPixel(pixelPos + vec2( 0.0, -j2), maxIter);
            vec3 f = sampleColorAtPixel(pixelPos + vec2( j2,  0.0), maxIter);
            vec3 g = sampleColorAtPixel(pixelPos + vec2( 0.0,  j2), maxIter);
            vec3 h = sampleColorAtPixel(pixelPos + vec2(-j2,  0.0), maxIter);
            color = (color * 4.0 + d + e + f + g + h) / 9.0;
        }
        finalColor = vec4(color, 1.0);
    }
`;

async function init() {
    const splitDouble = (value: number): [number, number] => {
        const hi = Math.fround(value);
        return [hi, value - hi];
    };

    const app = new Application();
    await app.init({
        resizeTo: window,
        autoDensity: true,
        resolution: window.devicePixelRatio || 1,
        preference: 'webgl',
        backgroundColor: 0x000000,
    });

    document.getElementById('app')!.appendChild(app.canvas);

    const controlsPanel = document.createElement('details');
    controlsPanel.className = 'hud-panel controls-panel';
    controlsPanel.open = true;
    controlsPanel.innerHTML = `
        <summary>Controls</summary>
        <ul class="controls-list">
            <li><strong>Scroll:</strong> zoom in/out</li>
            <li><strong>Middle drag:</strong> pan camera</li>
            <li><strong>Left drag:</strong> adjust active parameter (Julia C / Mandelbrot z0)</li>
            <li><strong>Right drag:</strong> blend mode (X) and exponent (Y)</li>
            <li><strong>Mobile:</strong> 1 finger = left drag, 2 fingers = right drag, pinch = zoom</li>
            <li><strong>Double left click:</strong> release manual C / z0 control</li>
            <li><strong>Space:</strong> pause/resume animation</li>
        </ul>
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
        quality: createParamRow('Capacity'),
    };

    const normalize = (value: number, min: number, max: number) => {
        if (max <= min) return 0;
        return (value - min) / (max - min);
    };

    const viewport = new Viewport({
        screenWidth: window.innerWidth,
        screenHeight: window.innerHeight,
        events: app.renderer.events,
    });

    app.stage.addChild(viewport);
    viewport.wheel().decelerate();

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
    let panStart: { x: number; y: number; centerX: number; centerY: number } | null = null;
    let blendDragStart: { x: number; y: number; blend: number; exponent: number } | null = null;
    let modeBlend = 1;
    let exponent = 2;
    let animationPaused = true;
    const blendSensitivity = 0.0015;
    const exponentSensitivity = 0.003;

    const getCanvasCoords = (clientX: number, clientY: number) => {
        const rect = app.canvas.getBoundingClientRect();
        return { x: clientX - rect.left, y: clientY - rect.top };
    };

    const isMobileDevice = window.matchMedia('(pointer: coarse)').matches
        || navigator.maxTouchPoints > 0
        || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const getTouchDistance = (a: Touch, b: Touch) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const getTouchMidpoint = (a: Touch, b: Touch) => ({
        x: (a.clientX + b.clientX) * 0.5,
        y: (a.clientY + b.clientY) * 0.5,
    });

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
                    ? (mouseControlC?.real ?? uniforms.uniforms.uC[0])
                    : (mouseControlZ0?.real ?? uniforms.uniforms.uZ0[0]);
                const rawImag = isJulia
                    ? (mouseControlC?.imag ?? uniforms.uniforms.uC[1])
                    : (mouseControlZ0?.imag ?? uniforms.uniforms.uZ0[1]);
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
            panStart = { x, y, centerX: cameraCenterX, centerY: cameraCenterY };
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
            lastInteractionAt = performance.now();
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
            cameraCenterX = panStart.centerX - dx / cameraZoom;
            cameraCenterY = panStart.centerY + dy / cameraZoom;
            lastInteractionAt = performance.now();
            cameraVersion += 1;
            activeOrbitVersion = -1;
        } else if ((e.buttons & 2) && blendDragStart) {
            const dx = x - blendDragStart.x;
            const dy = y - blendDragStart.y;
            modeBlend = Math.max(0, Math.min(1, blendDragStart.blend + blendSensitivity * dx));
            exponent = Math.max(1.01, Math.min(8, blendDragStart.exponent - exponentSensitivity * dy));
            lastInteractionAt = performance.now();
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

    const onTouchStart = (e: TouchEvent) => {
        if (!isMobileDevice) return;
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
            const isJulia = modeBlend > 0.5;
            const lim = isJulia ? cLimits : z0Limits;
            const rawReal = isJulia
                ? (mouseControlC?.real ?? uniforms.uniforms.uC[0])
                : (mouseControlZ0?.real ?? uniforms.uniforms.uZ0[0]);
            const rawImag = isJulia
                ? (mouseControlC?.imag ?? uniforms.uniforms.uC[1])
                : (mouseControlZ0?.imag ?? uniforms.uniforms.uZ0[1]);
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
            lastInteractionAt = performance.now();
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
            cameraZoom = Math.max(1e-3, Math.min(Number.MAX_VALUE, pinchStart.zoom * zoomScale));
            cameraVersion += 1;
            activeOrbitVersion = -1;
            lastInteractionAt = performance.now();
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
                ? (mouseControlC?.real ?? uniforms.uniforms.uC[0])
                : (mouseControlZ0?.real ?? uniforms.uniforms.uZ0[0]);
            const rawImag = isJulia
                ? (mouseControlC?.imag ?? uniforms.uniforms.uC[1])
                : (mouseControlZ0?.imag ?? uniforms.uniforms.uZ0[1]);
            const real = Math.max(lim.realMin, Math.min(lim.realMax, rawReal));
            const imag = Math.max(lim.imagMin, Math.min(lim.imagMax, rawImag));
            dragStart = { x, y, real, imag, isJulia };
        }
        e.preventDefault();
    };

    app.canvas.addEventListener('pointerdown', onPointerDown);
    app.canvas.addEventListener('pointermove', onPointerMove);
    app.canvas.addEventListener('pointerup', onPointerUp);
    app.canvas.addEventListener('pointerleave', onPointerLeave);
    app.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    app.canvas.addEventListener('touchstart', onTouchStart, { passive: false });
    app.canvas.addEventListener('touchmove', onTouchMove, { passive: false });
    app.canvas.addEventListener('touchend', onTouchEnd, { passive: false });
    app.canvas.addEventListener('touchcancel', onTouchEnd, { passive: false });

    window.addEventListener('keydown', (e: KeyboardEvent) => {
        if (e.code === 'Space') {
            e.preventDefault();
            animationPaused = !animationPaused;
        }
    });

    const geometry = new Geometry({
        attributes: { aPosition: [-1, -1, 1, -1, 1, 1, -1, 1] },
        indexBuffer: [0, 1, 2, 0, 2, 3],
    });

    const baseZoom = Math.min(window.innerWidth, window.innerHeight) / 3.5;
    viewport.moveCenter(0, 0);
    viewport.scaled = 1;

    let cameraCenterX = -0.5;
    let cameraCenterY = 0;
    let cameraZoom = baseZoom;
    const perturbWorker = new Worker(new URL('./perturbationWorker.ts', import.meta.url), { type: 'module' });
    const refOrbit = new Float32Array(ORBIT_CAPACITY * 2);
    let refLength = 0;
    let refRequestId = 0;
    let latestAppliedOrbitId = 0;
    let pendingOrbitRequest = false;
    let lastOrbitCenterX = cameraCenterX;
    let lastOrbitCenterY = cameraCenterY;
    let lastOrbitZoom = cameraZoom;
    let cameraVersion = 0;
    let activeOrbitVersion = -1;
    const requestVersionById = new Map<number, number>();
    const orbitCache: Array<{
        centerX: number;
        centerY: number;
        zoom: number;
        length: number;
        orbit: Float32Array;
        usedAt: number;
    }> = [];

    const applyOrbit = (
        orbit: Float32Array,
        length: number,
        centerX: number,
        centerY: number,
        zoom: number,
        version: number,
    ) => {
        refLength = Math.min(length, ORBIT_CAPACITY);
        refOrbit.fill(0);
        refOrbit.set(orbit.subarray(0, refLength * 2), 0);
        uniforms.uniforms.uRefLength = refLength;
        const [refXHi, refXLo] = splitDouble(centerX);
        const [refYHi, refYLo] = splitDouble(centerY);
        uniforms.uniforms.uRefCenterX[0] = refXHi;
        uniforms.uniforms.uRefCenterX[1] = refXLo;
        uniforms.uniforms.uRefCenterY[0] = refYHi;
        uniforms.uniforms.uRefCenterY[1] = refYLo;
        lastOrbitCenterX = centerX;
        lastOrbitCenterY = centerY;
        lastOrbitZoom = zoom;
        activeOrbitVersion = version;
    };

    const uniforms = new UniformGroup({
        uResolution: { value: new Float32Array([window.innerWidth, window.innerHeight]), type: 'vec2<f32>' },
        uCenterX: { value: new Float32Array(splitDouble(-0.5)), type: 'vec2<f32>' },
        uCenterY: { value: new Float32Array(splitDouble(0)), type: 'vec2<f32>' },
        uInvZoom: { value: new Float32Array(splitDouble(1 / cameraZoom)), type: 'vec2<f32>' },
        uMaxIterations: { value: 200, type: 'i32' },
        uColorIterations: { value: 320, type: 'i32' },
        uLod: { value: 0, type: 'i32' },
        uPrecisionMode: { value: 0, type: 'i32' },
        uUsePerturb: { value: 0, type: 'i32' },
        uPerturbBlend: { value: 0, type: 'f32' },
        uRefLength: { value: 0, type: 'i32' },
        uRefCenterX: { value: new Float32Array(splitDouble(cameraCenterX)), type: 'vec2<f32>' },
        uRefCenterY: { value: new Float32Array(splitDouble(cameraCenterY)), type: 'vec2<f32>' },
        uRefOrbit: { value: refOrbit, type: 'vec2<f32>', size: ORBIT_CAPACITY },
        uTime: { value: 0, type: 'f32' },
        uC: { value: new Float32Array([0.285, 0.01]), type: 'vec2<f32>' },
        uZ0: { value: new Float32Array([0, 0]), type: 'vec2<f32>' },
        uModeBlend: { value: 1, type: 'f32' },
        uExponent: { value: 2, type: 'f32' },
    });

    const shader = new Shader({
        glProgram: GlProgram.from({ vertex: vertexSrc, fragment: fragmentSrc }),
        resources: { mandelbrotUniforms: uniforms },
    });

    const mesh = new Mesh({ geometry, shader });
    app.stage.addChildAt(mesh, 0);

    perturbWorker.onmessage = (event: MessageEvent<{ id: number; length: number; orbit: Float32Array }>) => {
        pendingOrbitRequest = false;
        const { id, length, orbit } = event.data;
        if (id < latestAppliedOrbitId) return;
        const version = requestVersionById.get(id);
        requestVersionById.delete(id);
        if (version === undefined || version !== cameraVersion) return;

        latestAppliedOrbitId = id;
        applyOrbit(orbit, length, lastOrbitCenterX, lastOrbitCenterY, lastOrbitZoom, version);
        orbitUpdatedSinceLastFrame = true;

        orbitCache.push({
            centerX: lastOrbitCenterX,
            centerY: lastOrbitCenterY,
            zoom: lastOrbitZoom,
            length: Math.min(length, ORBIT_CAPACITY),
            orbit: new Float32Array(orbit),
            usedAt: performance.now(),
        });
        if (orbitCache.length > ORBIT_CACHE_LIMIT) {
            orbitCache.sort((a, b) => b.usedAt - a.usedAt);
            orbitCache.length = ORBIT_CACHE_LIMIT;
        }
    };

    let lastInteractionAt = performance.now();
    let currentIterations = 220;
    let qualityScale = 1.0;
    let smoothedFrameMs = 1000 / 60;
    let discoveredCapacity = currentIterations;
    let colorIterations = 320;
    let lodState = 1;
    let orbitUpdatedSinceLastFrame = false;
    let lastAdaptiveUpdateAt = 0;

    app.ticker.add(() => {
        const frameMs = app.ticker.deltaMS;
        const deltaCenter = viewport.center;
        const deltaScale = viewport.scaled;
        const hadInteraction = Math.abs(deltaCenter.x) > 1e-7
            || Math.abs(deltaCenter.y) > 1e-7
            || Math.abs(deltaScale - 1) > 1e-7;
        const activeInput = dragStart !== null
            || panStart !== null
            || blendDragStart !== null
            || touchTwoFingerStart !== null
            || pinchStart !== null;

        if (hadInteraction) {
            const zoomBefore = cameraZoom;
            cameraCenterX += deltaCenter.x / zoomBefore;
            cameraCenterY += -deltaCenter.y / zoomBefore;
            cameraZoom = Math.max(1e-3, Math.min(Number.MAX_VALUE, cameraZoom * deltaScale));
            viewport.moveCenter(0, 0);
            viewport.scaled = 1;
            lastInteractionAt = performance.now();
            cameraVersion += 1;
            activeOrbitVersion = -1;
        }

        const nowMs = performance.now();
        const zoom = cameraZoom;
        const res = app.renderer.width / app.renderer.resolution;
        const resY = app.renderer.height / app.renderer.resolution;
        const idleMs = nowMs - lastInteractionAt;
        const activelyInteracting = activeInput || hadInteraction || idleMs < 180;
        const [centerXHi, centerXLo] = splitDouble(cameraCenterX);
        const [centerYHi, centerYLo] = splitDouble(cameraCenterY);
        const [invZoomHi, invZoomLo] = splitDouble(1 / zoom);

        uniforms.uniforms.uResolution[0] = res;
        uniforms.uniforms.uResolution[1] = resY;
        uniforms.uniforms.uCenterX[0] = centerXHi;
        uniforms.uniforms.uCenterX[1] = centerXLo;
        uniforms.uniforms.uCenterY[0] = centerYHi;
        uniforms.uniforms.uCenterY[1] = centerYLo;
        uniforms.uniforms.uInvZoom[0] = invZoomHi;
        uniforms.uniforms.uInvZoom[1] = invZoomLo;
        const t = performance.now() * 0.001;
        if (!animationPaused) {
            uniforms.uniforms.uTime = t;
        }

        const wobblePeriod = 90.0;
        const wobbleOmega = (2 * Math.PI) / wobblePeriod;
        const wobbleAmp = 0.4;

        if (mouseControlC !== null) {
            uniforms.uniforms.uC[0] = Math.max(cLimits.realMin, Math.min(cLimits.realMax, mouseControlC.real));
            uniforms.uniforms.uC[1] = Math.max(cLimits.imagMin, Math.min(cLimits.imagMax, mouseControlC.imag));
        } else if (!animationPaused) {
            const periodRe = 45.0;
            const periodIm = 30.0;
            const omegaRe = (2 * Math.PI) / periodRe;
            const omegaIm = (2 * Math.PI) / periodIm;
            const phaseRe = omegaRe * t + wobbleAmp * Math.sin(wobbleOmega * t);
            const phaseIm = omegaIm * t + wobbleAmp * Math.sin(wobbleOmega * t * 1.3);
            const cAmp = 0.06;
            const cBase = 0.285;
            const cRe = cBase + cAmp * Math.cos(phaseRe);
            const cIm = cAmp * Math.sin(phaseIm);
            uniforms.uniforms.uC[0] = Math.max(cLimits.realMin, Math.min(cLimits.realMax, cRe));
            uniforms.uniforms.uC[1] = Math.max(cLimits.imagMin, Math.min(cLimits.imagMax, cIm));
        }

        if (mouseControlZ0 !== null) {
            uniforms.uniforms.uZ0[0] = Math.max(z0Limits.realMin, Math.min(z0Limits.realMax, mouseControlZ0.real));
            uniforms.uniforms.uZ0[1] = Math.max(z0Limits.imagMin, Math.min(z0Limits.imagMax, mouseControlZ0.imag));
        } else if (!animationPaused) {
            const z0PeriodRe = 55.0;
            const z0PeriodIm = 38.0;
            const z0OmegaRe = (2 * Math.PI) / z0PeriodRe;
            const z0OmegaIm = (2 * Math.PI) / z0PeriodIm;
            const z0Amp = 0.15;
            const phaseZ0Re = z0OmegaRe * t + wobbleAmp * Math.sin(wobbleOmega * t * 0.8);
            const phaseZ0Im = z0OmegaIm * t + wobbleAmp * Math.sin(wobbleOmega * t * 1.1);
            const z0Re = z0Amp * Math.cos(phaseZ0Re);
            const z0Im = z0Amp * Math.sin(phaseZ0Im);
            uniforms.uniforms.uZ0[0] = Math.max(z0Limits.realMin, Math.min(z0Limits.realMax, z0Re));
            uniforms.uniforms.uZ0[1] = Math.max(z0Limits.imagMin, Math.min(z0Limits.imagMax, z0Im));
        }

        if (!animationPaused && blendDragStart === null) {
            const blendPeriod = 120.0;
            const expPeriod = 90.0;
            modeBlend = Math.max(0, Math.min(1, 0.5 + 0.5 * Math.sin(t * (2 * Math.PI) / blendPeriod)));
            exponent = Math.max(1.01, Math.min(8, 2 + 1.5 * (0.5 + 0.5 * Math.sin(t * (2 * Math.PI) / expPeriod + 0.5))));
        }
        uniforms.uniforms.uModeBlend = modeBlend;
        uniforms.uniforms.uExponent = exponent;
        const hadOrbitUpdate = orbitUpdatedSinceLastFrame;
        const dynamicUpdate = activelyInteracting || !animationPaused || hadOrbitUpdate;
        const interactionDebounceReady = !activelyInteracting
            || (nowMs - lastAdaptiveUpdateAt) >= INTERACTION_RECALC_DEBOUNCE_MS;
        const adaptiveUpdate = dynamicUpdate && (
            hadOrbitUpdate
            || !activelyInteracting
            || !animationPaused
            || interactionDebounceReady
        );
        if (adaptiveUpdate) {
            lastAdaptiveUpdateAt = nowMs;
        }
        orbitUpdatedSinceLastFrame = false;

        const targetFrameMs = 1000 / TARGET_FPS;
        if (adaptiveUpdate) {
            smoothedFrameMs = smoothedFrameMs * 0.9 + frameMs * 0.1;
            if (smoothedFrameMs > targetFrameMs * 1.16) {
                qualityScale *= 0.94;
            } else if (smoothedFrameMs < targetFrameMs * 0.9) {
                qualityScale *= 1.015;
            }
            if (frameMs > targetFrameMs * 1.7) {
                qualityScale *= 0.86;
            }
            qualityScale = Math.max(QUALITY_MIN, Math.min(QUALITY_MAX, qualityScale));
        }
        const effectiveQuality = activelyInteracting ? Math.max(QUALITY_MIN, qualityScale * 0.62) : qualityScale;

        if (adaptiveUpdate) {
            if (lodState <= 0) {
                if (effectiveQuality > 0.94) lodState = 1;
            } else if (lodState === 1) {
                if (effectiveQuality < 0.72) lodState = 0;
                else if (effectiveQuality > 1.86 && !activelyInteracting) lodState = 2;
            } else {
                if (effectiveQuality < 1.5 || activelyInteracting) lodState = 1;
            }
        }
        uniforms.uniforms.uLod = lodState;
        uniforms.uniforms.uPrecisionMode = 0;

        const orbitDriftPxRaw = Math.hypot(
            (cameraCenterX - lastOrbitCenterX) * zoom,
            (cameraCenterY - lastOrbitCenterY) * zoom,
        );
        const orbitZoomDeltaRaw = Math.abs(Math.log2(Math.max(1e-12, zoom / lastOrbitZoom)));
        const orbitStaleRaw = orbitDriftPxRaw > 48 || orbitZoomDeltaRaw > 0.3;

        if ((orbitStaleRaw || refLength === 0) && activeOrbitVersion !== cameraVersion) {
            let best: (typeof orbitCache)[number] | null = null;
            let bestScore = Number.POSITIVE_INFINITY;
            for (const entry of orbitCache) {
                const driftPx = Math.hypot(
                    (cameraCenterX - entry.centerX) * zoom,
                    (cameraCenterY - entry.centerY) * zoom,
                );
                const zoomDelta = Math.abs(Math.log2(Math.max(1e-12, zoom / entry.zoom)));
                if (driftPx > 72 || zoomDelta > 0.45) continue;
                const score = driftPx + zoomDelta * 120;
                if (score < bestScore) {
                    best = entry;
                    bestScore = score;
                }
            }
            if (best) {
                best.usedAt = performance.now();
                applyOrbit(best.orbit, best.length, best.centerX, best.centerY, best.zoom, cameraVersion);
            }
        }

        const orbitDriftPx = Math.hypot(
            (cameraCenterX - lastOrbitCenterX) * zoom,
            (cameraCenterY - lastOrbitCenterY) * zoom,
        );
        const orbitZoomDelta = Math.abs(Math.log2(Math.max(1e-12, zoom / lastOrbitZoom)));
        const orbitStale = orbitDriftPx > 48 || orbitZoomDelta > 0.3;

        uniforms.uniforms.uUsePerturb = 0;
        uniforms.uniforms.uPerturbBlend = 0;

        const shouldRequestOrbit = zoom > 2e5
            && !pendingOrbitRequest
            && (activeOrbitVersion !== cameraVersion || orbitStale || idleMs > 800 || refLength === 0);
        if (shouldRequestOrbit) {
            pendingOrbitRequest = true;
            refRequestId += 1;
            requestVersionById.set(refRequestId, cameraVersion);
            lastOrbitCenterX = cameraCenterX;
            lastOrbitCenterY = cameraCenterY;
            lastOrbitZoom = zoom;
            const [reqXHi, reqXLo] = splitDouble(cameraCenterX);
            const [reqYHi, reqYLo] = splitDouble(cameraCenterY);
            const precisionDigits = Math.min(220, Math.max(64, Math.floor(Math.log10(Math.max(10, zoom))) + 48));
            perturbWorker.postMessage({
                id: refRequestId,
                centerXHi: reqXHi,
                centerXLo: reqXLo,
                centerYHi: reqYHi,
                centerYLo: reqYLo,
                maxIterations: ORBIT_CAPACITY,
                precisionDigits,
            });
        }

        if (adaptiveUpdate) {
            const iterBase = 240 + Math.log2(Math.max(1, zoom)) * 58;
            const iterTarget = Math.min(4096, Math.floor(iterBase * effectiveQuality));
            const maxStep = 200;
            if (currentIterations < iterTarget) {
                currentIterations = Math.min(iterTarget, currentIterations + maxStep);
            } else {
                currentIterations = Math.max(iterTarget, currentIterations - maxStep * 2);
            }
            uniforms.uniforms.uMaxIterations = Math.floor(currentIterations);

            const colorTarget = Math.min(4096, Math.floor(iterBase));
            if (colorIterations < colorTarget) {
                colorIterations = Math.min(colorTarget, colorIterations + 24);
            } else {
                colorIterations = Math.max(colorTarget, colorIterations - 4);
            }
            uniforms.uniforms.uColorIterations = colorIterations;
            if (!activelyInteracting && smoothedFrameMs <= targetFrameMs * 1.04) {
                discoveredCapacity = Math.max(discoveredCapacity, uniforms.uniforms.uMaxIterations);
            }
        }

        const cReal = uniforms.uniforms.uC[0];
        const cImag = uniforms.uniforms.uC[1];
        const z0Real = uniforms.uniforms.uZ0[0];
        const z0Imag = uniforms.uniforms.uZ0[1];

        paramRows.modeBlend.set(modeBlend, modeBlend.toFixed(2));
        paramRows.exponent.set(normalize(exponent, 1.01, 8), exponent.toFixed(2));
        paramRows.cReal.set(normalize(cReal, cLimits.realMin, cLimits.realMax), cReal.toFixed(3));
        paramRows.cImag.set(normalize(cImag, cLimits.imagMin, cLimits.imagMax), cImag.toFixed(3));
        paramRows.z0Real.set(normalize(z0Real, z0Limits.realMin, z0Limits.realMax), z0Real.toFixed(3));
        paramRows.z0Imag.set(normalize(z0Imag, z0Limits.imagMin, z0Limits.imagMax), z0Imag.toFixed(3));

        const logZoom = Math.log10(Math.max(1, zoom));
        paramRows.zoom.set(Math.min(1, logZoom / 12), zoom.toExponential(2));
        const maxIterations = uniforms.uniforms.uMaxIterations;
        paramRows.iterations.set(Math.min(1, maxIterations / 4096), maxIterations.toString());
        const fps = 1000 / Math.max(0.0001, smoothedFrameMs);
        paramRows.fps.set(Math.min(1, fps / 120), fps.toFixed(1));
        paramRows.quality.set(
            Math.min(1, discoveredCapacity / 4096),
            `${Math.round(qualityScale * 100)}% (${discoveredCapacity} it)`,
        );
    });

    window.addEventListener('resize', () => {
        viewport.resize(window.innerWidth, window.innerHeight);
        cameraVersion += 1;
        activeOrbitVersion = -1;
    });
}

init().catch(console.error);
