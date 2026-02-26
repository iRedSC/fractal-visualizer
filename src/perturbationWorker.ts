import Decimal from 'decimal.js';

type OrbitRequest = {
    id: number;
    centerXHi: number;
    centerXLo: number;
    centerYHi: number;
    centerYLo: number;
    maxIterations: number;
    precisionDigits: number;
};

type OrbitResponse = {
    id: number;
    length: number;
    orbit: Float32Array;
};

self.onmessage = (event: MessageEvent<OrbitRequest>) => {
    const req = event.data;
    Decimal.set({ precision: req.precisionDigits, rounding: Decimal.ROUND_HALF_EVEN });

    const cx = new Decimal(req.centerXHi).plus(req.centerXLo);
    const cy = new Decimal(req.centerYHi).plus(req.centerYLo);

    const out = new Float32Array(req.maxIterations * 2);
    let zr = new Decimal(0);
    let zi = new Decimal(0);
    let length = 0;

    for (let i = 0; i < req.maxIterations; i += 1) {
        out[i * 2] = zr.toNumber();
        out[i * 2 + 1] = zi.toNumber();
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
        orbit: out.subarray(0, length * 2),
    };

    postMessage(response);
};
