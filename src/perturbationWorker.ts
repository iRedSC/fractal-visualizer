import Decimal from 'decimal.js';

type OrbitRequest = {
    id: number;
    centerXHi: number;
    centerXLo: number;
    centerYHi: number;
    centerYLo: number;
    maxIterations: number;
    precisionDigits: number;
    mode: 'mandelbrot' | 'julia' | 'blended';
    modeBlend?: number;
    cReal?: number;
    cImag?: number;
};

type OrbitResponse = {
    id: number;
    length: number;
    orbit: Float32Array;
};

/** Split a number into hi+lo double-double for GPU precision */
function splitDouble(value: number): [number, number] {
    const hi = Math.fround(value);
    return [hi, value - hi];
}

self.onmessage = (event: MessageEvent<OrbitRequest>) => {
    const req = event.data;
    const minPrecisionFromIterations = Math.ceil(Math.log2(Math.max(2, req.maxIterations + 1))) * 14;
    const precision = Math.min(512, Math.max(req.precisionDigits, minPrecisionFromIterations));
    Decimal.set({ precision, rounding: Decimal.ROUND_HALF_EVEN });

    const centerX = new Decimal(req.centerXHi).plus(req.centerXLo);
    const centerY = new Decimal(req.centerYHi).plus(req.centerYLo);
    const mode = req.mode ?? 'mandelbrot';
    const blend = req.modeBlend ?? 0;
    const cRe = req.cReal ?? 0;
    const cIm = req.cImag ?? 0;

    let zr: Decimal;
    let zi: Decimal;
    let cx: Decimal;
    let cy: Decimal;
    if (mode === 'mandelbrot') {
        zr = new Decimal(0);
        zi = new Decimal(0);
        cx = centerX;
        cy = centerY;
    } else if (mode === 'julia') {
        zr = centerX;
        zi = centerY;
        cx = new Decimal(cRe);
        cy = new Decimal(cIm);
    } else {
        zr = centerX.mul(blend);
        zi = centerY.mul(blend);
        cx = centerX.mul(1 - blend).plus(new Decimal(cRe).mul(blend));
        cy = centerY.mul(1 - blend).plus(new Decimal(cIm).mul(blend));
    }

    const out = new Float32Array(req.maxIterations * 4);
    let length = 0;

    for (let i = 0; i < req.maxIterations; i += 1) {
        const re = zr.toNumber();
        const im = zi.toNumber();
        const [reHi, reLo] = splitDouble(re);
        const [imHi, imLo] = splitDouble(im);
        out[i * 4] = reHi;
        out[i * 4 + 1] = reLo;
        out[i * 4 + 2] = imHi;
        out[i * 4 + 3] = imLo;
        length = i + 1;

        const zr2 = zr.mul(zr);
        const zi2 = zi.mul(zi);
        const nextZr = zr2.minus(zi2).plus(cx);
        const nextZi = zr.mul(zi).mul(2).plus(cy);

        zr = nextZr;
        zi = nextZi;

        if (zr2.plus(zi2).greaterThan(16)) break;
    }

    const response: OrbitResponse = {
        id: req.id,
        length,
        orbit: out.subarray(0, length * 4),
    };

    postMessage(response);
};
