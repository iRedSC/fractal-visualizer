import './style.css';

const ORBIT_CAPACITY = 1024;
const ORBIT_FLOATS_PER_POINT = 4;
const ORBIT_CACHE_LIMIT = 12;
const TARGET_FPS = 58;
const QUALITY_MIN = 0.35;
const QUALITY_MAX = 2.2;
const INTERACTION_RECALC_DEBOUNCE_MS = 48;
const WHEEL_ZOOM_FACTOR = 1.08;

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
    uniform int uLod;
    uniform int uPrecisionMode;
    uniform int uUsePerturb;
    uniform float uPerturbBlend;
    uniform int uRefLength;
    uniform vec2 uRefCenterX;
    uniform vec2 uRefCenterY;
    uniform sampler2D uRefOrbitTex;
    uniform vec2 uCenterRefOffsetX;
    uniform vec2 uCenterRefOffsetY;
    uniform float uPerturbDcCoeff;
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

    vec4 dsComplexAdd(vec4 a, vec4 b) {
        vec2 re = dsAdd(a.xy, b.xy);
        vec2 im = dsAdd(a.zw, b.zw);
        return vec4(re, im);
    }

    vec4 dsComplexMul(vec4 a, vec4 b) {
        vec2 re = dsSub(dsMul(a.xy, b.xy), dsMul(a.zw, b.zw));
        vec2 im = dsAdd(dsMul(a.xy, b.zw), dsMul(a.zw, b.xy));
        return vec4(re, im);
    }

    vec4 dsComplexScale(vec4 z, float s) {
        vec2 re = dsMulFloat(z.xy, s);
        vec2 im = dsMulFloat(z.zw, s);
        return vec4(re, im);
    }

    float dsComplexMag2(vec4 z) {
        vec2 re2 = dsMul(z.xy, z.xy);
        vec2 im2 = dsMul(z.zw, z.zw);
        vec2 mag2 = dsAdd(re2, im2);
        return mag2.x + mag2.y;
    }

    float mandelbrotSmoothPerturb(vec4 dcFull, vec4 cFull, float maxIter) {
        int maxLimit = int(maxIter);
        if (maxLimit < 1) maxLimit = 1;
        if (maxLimit > uMaxIterations) maxLimit = uMaxIterations;
        int limit = uRefLength;
        if (maxLimit < limit) limit = maxLimit;
        if (limit > 1024) limit = 1024;
        vec4 dz = vec4(0.0);
        float iter = 0.0;
        vec4 z = vec4(0.0);
        bool needFullPrecision = false;

        for (int i = 0; i < 1024; i++) {
            if (i >= limit) break;
            vec4 Zi = texelFetch(uRefOrbitTex, ivec2(i, 0), 0);
            z = dsComplexAdd(Zi, dz);
            float mag2 = dsComplexMag2(z);
            if (mag2 > 4.0) {
                float safeMag2 = max(mag2, 4.000001);
                return iter + 1.0 - log2(log2(safeMag2));
            }
            float dzMag2 = dsComplexMag2(dz);
            float ZMag2 = dsComplexMag2(Zi);
            if (dzMag2 > ZMag2) {
                needFullPrecision = true;
                break;
            }
            float dzMag = sqrt(dzMag2);
            if (dzMag > 1e-30 && mag2 < dzMag2 * 1e-20) {
                needFullPrecision = true;
                break;
            }
            vec4 twoZdz = dsComplexScale(dsComplexMul(Zi, dz), 2.0);
            vec4 dz2 = dsComplexMul(dz, dz);
            vec4 dcScaled = dsComplexScale(dcFull, uPerturbDcCoeff);
            dz = dsComplexAdd(dsComplexAdd(twoZdz, dz2), dcScaled);
            iter += 1.0;
        }

        if (needFullPrecision) {
            vec2 zx = z.xy;
            vec2 zy = z.zw;
            vec2 cx = cFull.xy;
            vec2 cy = cFull.zw;
            int remaining = maxLimit - int(iter);
            for (int i = 0; i < 4096; i++) {
                if (i >= remaining) break;
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

        vec2 zx = z.xy;
        vec2 zy = z.zw;
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
        vec4 cFull;
        vec4 z0Full;
        float sDirect;
        if (uModeBlend <= 0.001) {
            z0Full = vec4(uZ0.x, 0.0, uZ0.y, 0.0);
            cFull = pixelPosDS;
            sDirect = uPrecisionMode == 0
                ? iterSmooth(uZ0, pixelC, maxIter)
                : iterSmoothDS(z0Full, cFull, maxIter);
        } else if (uModeBlend >= 0.999) {
            z0Full = pixelPosDS;
            cFull = vec4(uC.x, 0.0, uC.y, 0.0);
            sDirect = uPrecisionMode == 0
                ? juliaSmooth(pixelC, maxIter)
                : juliaSmoothDS(z0Full, maxIter);
        } else {
            vec2 z0 = uModeBlend * pixelC;
            vec2 c = (1.0 - uModeBlend) * pixelC + uModeBlend * uC;
            vec2 z0x = dsMulFloat(pixelPosDS.xy, uModeBlend);
            vec2 z0y = dsMulFloat(pixelPosDS.zw, uModeBlend);
            vec2 cMixX = dsMulFloat(pixelPosDS.xy, 1.0 - uModeBlend);
            vec2 cMixY = dsMulFloat(pixelPosDS.zw, 1.0 - uModeBlend);
            vec2 cDSX = dsAdd(cMixX, vec2(uModeBlend * uC.x, 0.0));
            vec2 cDSY = dsAdd(cMixY, vec2(uModeBlend * uC.y, 0.0));
            z0Full = vec4(z0x.x, z0x.y, z0y.x, z0y.y);
            cFull = vec4(cDSX.x, cDSX.y, cDSY.x, cDSY.y);
            sDirect = uPrecisionMode == 0
                ? iterSmooth(z0, c, maxIter)
                : iterSmoothDS(z0Full, cFull, maxIter);
        }

        float s = sDirect;
        if (uUsePerturb == 1 && uRefLength > 0) {
            float dx = pixelPos.x - uResolution.x * 0.5;
            float dy = uResolution.y * 0.5 - pixelPos.y;
            vec2 dcx = dsAdd(uCenterRefOffsetX, dsMulFloat(uInvZoom, dx));
            vec2 dcy = dsAdd(uCenterRefOffsetY, dsMulFloat(uInvZoom, dy));
            vec4 dc = vec4(dcx.x, dcx.y, dcy.x, dcy.y);
            float sPerturb = mandelbrotSmoothPerturb(dc, pixelPosDS, maxIter);
            s = mix(sDirect, sPerturb, uPerturbBlend);
        }
        return colorFromSmooth(s);
    }

    vec2 clampToViewport(vec2 pixelPos) {
        vec2 maxPixel = max(vec2(0.0), uResolution - vec2(1.0));
        return clamp(pixelPos, vec2(0.0), maxPixel);
    }

    void main() {
        vec2 screenPos = vPosition * 0.5 + 0.5;
        screenPos.y = 1.0 - screenPos.y;
        vec2 pixelPos = clampToViewport(screenPos * uResolution);
        float maxIter = float(uMaxIterations);
        vec3 color = sampleColorAtPixel(pixelPos, maxIter);

        if (uLod >= 1) {
            float j = 0.5;
            vec3 a = sampleColorAtPixel(clampToViewport(pixelPos + vec2( j,  j)), maxIter);
            vec3 b = sampleColorAtPixel(clampToViewport(pixelPos + vec2(-j,  j)), maxIter);
            vec3 c = sampleColorAtPixel(clampToViewport(pixelPos + vec2( j, -j)), maxIter);
            vec3 d = sampleColorAtPixel(clampToViewport(pixelPos + vec2(-j, -j)), maxIter);
            color = (a + b + c + d) * 0.25;
        }
        if (uLod >= 2 && uPrecisionMode == 0) {
            float j2 = 0.6;
            vec3 d = sampleColorAtPixel(clampToViewport(pixelPos + vec2(-j2, -j2)), maxIter);
            vec3 e = sampleColorAtPixel(clampToViewport(pixelPos + vec2( 0.0, -j2)), maxIter);
            vec3 f = sampleColorAtPixel(clampToViewport(pixelPos + vec2( j2,  0.0)), maxIter);
            vec3 g = sampleColorAtPixel(clampToViewport(pixelPos + vec2( 0.0,  j2)), maxIter);
            vec3 h = sampleColorAtPixel(clampToViewport(pixelPos + vec2(-j2,  0.0)), maxIter);
            color = (color * 4.0 + d + e + f + g + h) / 9.0;
        }
        finalColor = vec4(color, 1.0);
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

async function init() {
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

    const ddSub = (a: DoubleDouble, b: DoubleDouble): DoubleDouble => ddAdd(a, [-b[0], -b[1]]);

    const ddFromNumber = (value: number): DoubleDouble => splitDouble(value);
    const ddToNumber = (value: DoubleDouble): number => value[0] + value[1];

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
    if (!gl) throw new Error('WebGL2 not supported');

    const program = createProgram(gl);
    const positionLoc = gl.getAttribLocation(program, 'aPosition');

    const orbitTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, orbitTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, ORBIT_CAPACITY, 1, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const positions = new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]);
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

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

    const baseZoom = Math.min(window.innerWidth, window.innerHeight) / 3.5;

    let cameraCenterXDD: DoubleDouble = ddFromNumber(-0.5);
    let cameraCenterYDD: DoubleDouble = ddFromNumber(0);
    let cameraZoom = baseZoom;
    const perturbWorker = new Worker(new URL('./perturbationWorker.ts', import.meta.url), { type: 'module' });
    let refLength = 0;
    let refRequestId = 0;
    let latestAppliedOrbitId = 0;
    let pendingOrbitRequest = false;
    let lastOrbitCenterXDD: DoubleDouble = [cameraCenterXDD[0], cameraCenterXDD[1]];
    let lastOrbitCenterYDD: DoubleDouble = [cameraCenterYDD[0], cameraCenterYDD[1]];
    let lastOrbitZoom = cameraZoom;
    let lastOrbitModeBlend = 0;
    let requestedOrbitCenterXDD: DoubleDouble = [cameraCenterXDD[0], cameraCenterXDD[1]];
    let requestedOrbitCenterYDD: DoubleDouble = [cameraCenterYDD[0], cameraCenterYDD[1]];
    let requestedOrbitZoom = cameraZoom;
    let requestedOrbitModeBlend = 0;
    let cameraVersion = 0;
    let activeOrbitVersion = -1;
    const requestVersionById = new Map<number, number>();
    const orbitCache: Array<{
        centerX: number;
        centerY: number;
        zoom: number;
        modeBlend: number;
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
        gl.bindTexture(gl.TEXTURE_2D, orbitTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, refLength, 1, gl.RGBA, gl.FLOAT, orbit);
        uniforms.uRefLength = refLength;
        const [refXHi, refXLo] = splitDouble(centerX);
        const [refYHi, refYLo] = splitDouble(centerY);
        uniforms.uRefCenterX[0] = refXHi;
        uniforms.uRefCenterX[1] = refXLo;
        uniforms.uRefCenterY[0] = refYHi;
        uniforms.uRefCenterY[1] = refYLo;
        lastOrbitCenterXDD = ddFromNumber(centerX);
        lastOrbitCenterYDD = ddFromNumber(centerY);
        lastOrbitZoom = zoom;
        lastOrbitModeBlend = modeBlend;
        activeOrbitVersion = version;
    };

    const uniforms = {
        uResolution: new Float32Array([window.innerWidth, window.innerHeight]),
        uCenterX: new Float32Array([cameraCenterXDD[0], cameraCenterXDD[1]]),
        uCenterY: new Float32Array([cameraCenterYDD[0], cameraCenterYDD[1]]),
        uInvZoom: new Float32Array(splitDouble(1 / cameraZoom)),
        uMaxIterations: 200,
        uColorIterations: 320,
        uLod: 0,
        uPrecisionMode: 0,
        uUsePerturb: 0,
        uPerturbBlend: 0,
        uRefLength: 0,
        uRefCenterX: new Float32Array([cameraCenterXDD[0], cameraCenterXDD[1]]),
        uRefCenterY: new Float32Array([cameraCenterYDD[0], cameraCenterYDD[1]]),
        uCenterRefOffsetX: new Float32Array(splitDouble(0)),
        uCenterRefOffsetY: new Float32Array(splitDouble(0)),
        uPerturbDcCoeff: 1,
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
            cameraCenterXDD = ddAdd(panStart.centerX, ddFromNumber(-dx / cameraZoom));
            cameraCenterYDD = ddAdd(panStart.centerY, ddFromNumber(dy / cameraZoom));
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

    const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const { x, y } = getCanvasCoords(e.clientX, e.clientY);
        const resX = canvas.width;
        const resY = canvas.height;
        const dx = x - resX * 0.5;
        const dy = resY * 0.5 - y;
        const zoomFactor = e.deltaY > 0 ? 1 / WHEEL_ZOOM_FACTOR : WHEEL_ZOOM_FACTOR;
        const k = Math.pow(zoomFactor, Math.min(3, Math.abs(e.deltaY) / 50));
        const newZoom = Math.max(1e-3, Math.min(Number.MAX_VALUE, cameraZoom * k));
        const invZoom = 1 / cameraZoom;
        const invNewZoom = 1 / newZoom;
        cameraCenterXDD = ddAdd(cameraCenterXDD, ddFromNumber(dx * (invZoom - invNewZoom)));
        cameraCenterYDD = ddAdd(cameraCenterYDD, ddFromNumber(dy * (invZoom - invNewZoom)));
        cameraZoom = newZoom;
        lastInteractionAt = performance.now();
        cameraVersion += 1;
        activeOrbitVersion = -1;
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

    perturbWorker.onmessage = (event: MessageEvent<{ id: number; length: number; orbit: Float32Array }>) => {
        pendingOrbitRequest = false;
        const { id, length, orbit } = event.data;
        if (id < latestAppliedOrbitId) return;
        const version = requestVersionById.get(id);
        requestVersionById.delete(id);
        if (version === undefined || version !== cameraVersion) return;

        latestAppliedOrbitId = id;
        const requestedOrbitCenterX = ddToNumber(requestedOrbitCenterXDD);
        const requestedOrbitCenterY = ddToNumber(requestedOrbitCenterYDD);
        applyOrbit(orbit, length, requestedOrbitCenterX, requestedOrbitCenterY, requestedOrbitZoom, version);
        orbitUpdatedSinceLastFrame = true;

        orbitCache.push({
            centerX: requestedOrbitCenterX,
            centerY: requestedOrbitCenterY,
            zoom: requestedOrbitZoom,
            modeBlend: requestedOrbitModeBlend,
            length: Math.min(length, ORBIT_CAPACITY),
            orbit: new Float32Array(orbit.subarray(0, length * ORBIT_FLOATS_PER_POINT)),
            usedAt: performance.now(),
        });
        if (orbitCache.length > ORBIT_CACHE_LIMIT) {
            orbitCache.sort((a, b) => b.usedAt - a.usedAt);
            orbitCache.length = ORBIT_CACHE_LIMIT;
        }
    };

    let lastInteractionAt = performance.now();
    let smoothedPerturbBlend = 0;
    let currentIterations = 220;
    let qualityScale = 1.0;
    let smoothedFrameMs = 1000 / 60;
    let discoveredCapacity = currentIterations;
    let colorIterations = 320;
    let lodState = 1;
    let orbitUpdatedSinceLastFrame = false;
    let lastAdaptiveUpdateAt = 0;
    let lastFrameTime = performance.now();

    const uniformLocations: Record<string, WebGLUniformLocation | null> = {};
    const getLoc = (name: string) => {
        if (!(name in uniformLocations)) {
            uniformLocations[name] = gl.getUniformLocation(program, name);
        }
        return uniformLocations[name];
    };

    const setUniforms = () => {
        gl.uniform2fv(getLoc('uResolution'), uniforms.uResolution);
        gl.uniform2fv(getLoc('uCenterX'), uniforms.uCenterX);
        gl.uniform2fv(getLoc('uCenterY'), uniforms.uCenterY);
        gl.uniform2fv(getLoc('uInvZoom'), uniforms.uInvZoom);
        gl.uniform1i(getLoc('uMaxIterations'), uniforms.uMaxIterations);
        gl.uniform1i(getLoc('uColorIterations'), uniforms.uColorIterations);
        gl.uniform1i(getLoc('uLod'), uniforms.uLod);
        gl.uniform1i(getLoc('uPrecisionMode'), uniforms.uPrecisionMode);
        gl.uniform1i(getLoc('uUsePerturb'), uniforms.uUsePerturb);
        gl.uniform1f(getLoc('uPerturbBlend'), uniforms.uPerturbBlend);
        gl.uniform1i(getLoc('uRefLength'), uniforms.uRefLength);
        gl.uniform2fv(getLoc('uRefCenterX'), uniforms.uRefCenterX);
        gl.uniform2fv(getLoc('uRefCenterY'), uniforms.uRefCenterY);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, orbitTex);
        gl.uniform1i(getLoc('uRefOrbitTex'), 0);
        gl.uniform2fv(getLoc('uCenterRefOffsetX'), uniforms.uCenterRefOffsetX);
        gl.uniform2fv(getLoc('uCenterRefOffsetY'), uniforms.uCenterRefOffsetY);
        gl.uniform1f(getLoc('uPerturbDcCoeff'), uniforms.uPerturbDcCoeff);
        gl.uniform1f(getLoc('uTime'), uniforms.uTime);
        gl.uniform2fv(getLoc('uC'), uniforms.uC);
        gl.uniform2fv(getLoc('uZ0'), uniforms.uZ0);
        gl.uniform1f(getLoc('uModeBlend'), uniforms.uModeBlend);
        gl.uniform1f(getLoc('uExponent'), uniforms.uExponent);
    };

    const tick = () => {
        if (document.hidden) {
            requestAnimationFrame(tick);
            return;
        }
        const nowMs = performance.now();
        const frameMs = nowMs - lastFrameTime;
        lastFrameTime = nowMs;

        const res = canvas.width;
        const resY = canvas.height;
        const idleMs = nowMs - lastInteractionAt;
        const activeInput = dragStart !== null
            || panStart !== null
            || blendDragStart !== null
            || touchTwoFingerStart !== null
            || pinchStart !== null;
        const activelyInteracting = activeInput || idleMs < 180;
        const centerXHi = cameraCenterXDD[0];
        const centerXLo = cameraCenterXDD[1];
        const centerYHi = cameraCenterYDD[0];
        const centerYLo = cameraCenterYDD[1];
        const [invZoomHi, invZoomLo] = splitDouble(1 / cameraZoom);

        uniforms.uResolution[0] = res;
        uniforms.uResolution[1] = resY;
        uniforms.uCenterX[0] = centerXHi;
        uniforms.uCenterX[1] = centerXLo;
        uniforms.uCenterY[0] = centerYHi;
        uniforms.uCenterY[1] = centerYLo;
        uniforms.uInvZoom[0] = invZoomHi;
        uniforms.uInvZoom[1] = invZoomLo;
        const t = performance.now() * 0.001;
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
        const zoom = cameraZoom;
        if (adaptiveUpdate) {
            smoothedFrameMs = smoothedFrameMs * 0.95 + frameMs * 0.05;
            if (smoothedFrameMs > targetFrameMs * 1.2) {
                qualityScale *= 0.995;
            } else if (smoothedFrameMs < targetFrameMs * 0.9) {
                qualityScale *= 1.005;
            }
            if (frameMs > targetFrameMs * 2.0) {
                qualityScale *= 0.98;
            }
            qualityScale = Math.max(QUALITY_MIN, Math.min(QUALITY_MAX, qualityScale));
        }
        const effectiveQuality = activelyInteracting ? Math.max(QUALITY_MIN, qualityScale * 0.85) : qualityScale;

        const logZoom = Math.log10(Math.max(1, zoom));
        const lodFromZoom = logZoom < 2 ? 0 : (logZoom < 4 ? 1 : 2);
        if (adaptiveUpdate) {
            if (lodState <= 0) {
                if (lodFromZoom >= 1 && effectiveQuality > 0.7) lodState = 1;
            } else if (lodState === 1) {
                if (lodFromZoom < 1 || effectiveQuality < 0.5) lodState = 0;
                else if (lodFromZoom >= 2 && effectiveQuality > 1.2 && !activelyInteracting) lodState = 2;
            } else {
                if (lodFromZoom < 2 || effectiveQuality < 0.9) lodState = 1;
            }
        }
        uniforms.uLod = lodState;
        uniforms.uPrecisionMode = 1;

        const cameraCenterX = ddToNumber(cameraCenterXDD);
        const cameraCenterY = ddToNumber(cameraCenterYDD);
        const lastOrbitCenterX = ddToNumber(lastOrbitCenterXDD);
        const lastOrbitCenterY = ddToNumber(lastOrbitCenterYDD);

        const orbitDriftPxRaw = Math.hypot(
            (cameraCenterX - lastOrbitCenterX) * zoom,
            (cameraCenterY - lastOrbitCenterY) * zoom,
        );
        const orbitZoomDeltaRaw = Math.abs(Math.log2(Math.max(1e-12, zoom / lastOrbitZoom)));
        const orbitModeBlendDelta = Math.abs(modeBlend - lastOrbitModeBlend);
        const orbitStaleRaw = orbitDriftPxRaw > 48 || orbitZoomDeltaRaw > 0.3 || orbitModeBlendDelta > 0.02;

        if ((orbitStaleRaw || refLength === 0) && activeOrbitVersion !== cameraVersion) {
            let best: (typeof orbitCache)[number] | null = null;
            let bestScore = Number.POSITIVE_INFINITY;
            for (const entry of orbitCache) {
                if (Math.abs(entry.modeBlend - modeBlend) > 0.02) continue;
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
                lastOrbitModeBlend = best.modeBlend;
                applyOrbit(best.orbit, best.length, best.centerX, best.centerY, best.zoom, cameraVersion);
            }
        }

        const orbitDriftPx = Math.hypot(
            (cameraCenterX - lastOrbitCenterX) * zoom,
            (cameraCenterY - lastOrbitCenterY) * zoom,
        );
        const orbitZoomDelta = Math.abs(Math.log2(Math.max(1e-12, zoom / lastOrbitZoom)));
        const orbitStale = orbitDriftPx > 48 || orbitZoomDelta > 0.3 || Math.abs(modeBlend - lastOrbitModeBlend) > 0.02;

        const usePerturb = refLength > 0 && zoom > 2e5;
        const perturbTarget = usePerturb
            ? (orbitStale ? (activelyInteracting ? 0.6 : 0.15) : 1.0)
            : 0.0;
        const perturbSmoothFactor = activelyInteracting ? 0.12 : 0.25;
        smoothedPerturbBlend += (perturbTarget - smoothedPerturbBlend) * perturbSmoothFactor;
        uniforms.uUsePerturb = usePerturb ? 1 : 0;
        uniforms.uPerturbBlend = smoothedPerturbBlend;
        uniforms.uPerturbDcCoeff = 1 - modeBlend;
        if (usePerturb) {
            const [offXHi, offXLo] = ddSub(cameraCenterXDD, lastOrbitCenterXDD);
            const [offYHi, offYLo] = ddSub(cameraCenterYDD, lastOrbitCenterYDD);
            uniforms.uCenterRefOffsetX[0] = offXHi;
            uniforms.uCenterRefOffsetX[1] = offXLo;
            uniforms.uCenterRefOffsetY[0] = offYHi;
            uniforms.uCenterRefOffsetY[1] = offYLo;
        }

        const shouldRequestOrbit = zoom > 2e5
            && !pendingOrbitRequest
            && (activeOrbitVersion !== cameraVersion || orbitStale || idleMs > 800 || refLength === 0);
        if (shouldRequestOrbit) {
            pendingOrbitRequest = true;
            refRequestId += 1;
            requestVersionById.set(refRequestId, cameraVersion);
            requestedOrbitCenterXDD = [cameraCenterXDD[0], cameraCenterXDD[1]];
            requestedOrbitCenterYDD = [cameraCenterYDD[0], cameraCenterYDD[1]];
            requestedOrbitZoom = zoom;
            requestedOrbitModeBlend = modeBlend;
            const reqXHi = cameraCenterXDD[0];
            const reqXLo = cameraCenterXDD[1];
            const reqYHi = cameraCenterYDD[0];
            const reqYLo = cameraCenterYDD[1];
            // Keep extra decimal margin so reference orbit remains stable at deep zoom.
            const logZoom = Math.log10(Math.max(1, zoom));
            const iterHint = Math.max(ORBIT_CAPACITY, uniforms.uMaxIterations);
            const precisionDigits = Math.min(
                512,
                Math.max(64, Math.ceil(logZoom) + 28, Math.ceil(Math.log2(iterHint + 1)) * 14),
            );
            const orbitMode = modeBlend <= 0.001 ? 'mandelbrot' : (modeBlend >= 0.999 ? 'julia' : 'blended');
            perturbWorker.postMessage({
                id: refRequestId,
                centerXHi: reqXHi,
                centerXLo: reqXLo,
                centerYHi: reqYHi,
                centerYLo: reqYLo,
                maxIterations: ORBIT_CAPACITY,
                precisionDigits,
                mode: orbitMode,
                modeBlend: modeBlend,
                cReal: uniforms.uC[0],
                cImag: uniforms.uC[1],
            });
        }

        if (adaptiveUpdate) {
            const iterBase = 50 + Math.log2(Math.max(1, zoom)) * 60;
            const iterTarget = Math.min(4096, Math.floor(iterBase * effectiveQuality));
            const maxStepUp = 40;
            const maxStepDown = 2;
            if (currentIterations < iterTarget) {
                currentIterations = Math.min(iterTarget, currentIterations + maxStepUp);
            } else {
                currentIterations = Math.max(iterTarget, currentIterations - maxStepDown);
            }
            uniforms.uMaxIterations = Math.floor(currentIterations);

            const colorTarget = Math.min(4096, Math.floor(iterBase));
            if (colorIterations < colorTarget) {
                colorIterations = Math.min(colorTarget, colorIterations + 12);
            } else {
                colorIterations = Math.max(colorTarget, colorIterations - 1);
            }
            uniforms.uColorIterations = colorIterations;
            if (!activelyInteracting && smoothedFrameMs <= targetFrameMs * 1.04) {
                discoveredCapacity = Math.max(discoveredCapacity, uniforms.uMaxIterations);
            }
        }

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

        paramRows.zoom.set(Math.min(1, logZoom / 12), zoom.toExponential(2));
        const maxIterations = uniforms.uMaxIterations;
        paramRows.iterations.set(Math.min(1, maxIterations / 4096), maxIterations.toString());
        const fps = 1000 / Math.max(0.0001, smoothedFrameMs);
        paramRows.fps.set(Math.min(1, fps / 120), fps.toFixed(1));
        paramRows.quality.set(
            Math.min(1, discoveredCapacity / 4096),
            `${Math.round(qualityScale * 100)}% (${discoveredCapacity} it)`,
        );

        setUniforms();
        gl.useProgram(program);
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);

    window.addEventListener('resize', () => {
        resize();
        cameraVersion += 1;
        activeOrbitVersion = -1;
    });
}

init().catch(console.error);
