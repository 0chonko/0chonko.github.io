#!/usr/bin/env python3
"""Regenerate the Open Kernel Challenge blog figures from local judge profiles."""
import glob
import json
import math
import os

ROOT = os.path.expanduser('~/Documents/Projects/PMMP/Larger-Deliverables/Open-Kernel-Challenge')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets/blog/open-kernel-challenge')
BW, PEAK = 4.8e12, 989e12
FONT = 'system-ui, -apple-system, Segoe UI, sans-serif'
GAIN, LOSS, TIP, INK, MUTED = '#3b6ea5', '#c0392b', '#1f4473', '#1a1a1a', '#555'

# Chronological. Job id prefix -> label.
BF16 = [
    ('e3287aa5', 'SGLang path mirror'),
    ('8d304dbc', 'SGLang path mirror, rerun'),
    ('f2a7ffcd', 'Inline SGLang Hopper GEMV'),
    ('e77df0f8', 'GEMV + dispatch retune'),
    ('a3b39365', 'Custom thin-m kernel'),
    ('513efba3', 'G stack, pass 1'),
    ('f9c2d360', 'G stack, pass 2'),
    ('39e4f815', 'G stack, pass 3'),
    ('83e25962', 'H: split-K resume'),
    ('7a7da607', 'J stack'),
    ('46df7e39', 'J stack, rerun'),
    ('100d2b88', 'L: scalar FP32 router gate'),
    ('232541c8', 'L: FlashInfer thin-m'),
    ('5b4c3822', 'L: FlashInfer, no-split tile'),
    ('da1d5df0', 'L: hi/lo router + FlashInfer'),
    ('0b63c534', 'L: WGMMA-wide Stream-K'),
    ('5b09583c', 'L: router to m=16'),
    ('59c589ad', 'M: SM90 GEMM for wide m'),
    ('db81cd3c', 'M: SM90 GEMM, retile'),
    ('82ac6460', 'M: SM90 GEMM, o layer'),
    ('11b8f6c2', 'FlashInfer large-m allow-list'),
    ('1b9767dc', 'Piece: small-m 12-bit GEMV'),
    ('0a312869', 'Piece: m=32 WGMMA GEMV'),
    ('7a175ac1', 'Piece: router hi/lo WGMMA'),
    ('8a37bd81', 'Piece: qkv tiles'),
    ('3838811e', 'Piece: wide2 + o layer'),
    ('fbcb20f5', 'Piece: cuBLASLt m32w'),
    ('40567e96', 'Assemble 1: small-m'),
    ('cc64f6c3', 'Assemble 2: + m=32'),
    ('4914922c', 'Assemble 3: + router hi/lo'),
    ('6384a0f5', 'Assemble 4: + qkv'),
    ('37d94bb2', 'Assemble 5: + wide2 / o'),
    ('4c52de7c', 'Assemble 6: + cuBLASLt m32w'),
    ('00c3a102', 'Tip submit'),
]
ATTN = [
    ('fa7dd57d', 'Custom Triton attend'),
    ('b5f2814b', 'FA3 + fused prep / epilogue'),
    ('aa490ace', 'FlashInfer verify layout'),
    ('06d20d2d', 'FA4-style pipeline'),
    ('21805ead', 'SageAttention3 FP8 Q'),
    ('d4ed918f', 'Hybrid FA3 fuse + Sage'),
    ('47300abf', 'ThunderKittens 16-row tiles'),
    ('037127fe', 'ThunderKittens, rerun'),
    ('1905de54', 'FuTransformer zip-fusion'),
    ('9212d0c3', 'Merged fuse stack'),
    ('dd060979', 'AttnFuse GQA'),
    ('95925ab7', 'Fold + FlashMask SWA'),
    ('3a30c353', 'DScale Hopper verify'),
    ('1328d929', 'Strided epilogue loads'),
    ('dcabef58', 'Single-pass full-context GQA'),
    ('94b9f054', 'Tip: gate copy folded'),
]
TIPS = {'00c3a102', '94b9f054'}
LOADS = ['decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32']
BF16_LAYERS = {'qkv_gate': (2304, 8192, 2), 'o': (8192, 1024, 2), 'pooler': (4096, 8192, 2),
               'unpooler': (8192, 4096, 2), 'router': (256, 8192, 4)}


def bf16_work(c, _):
    m = c['m']
    n, k, w = BF16_LAYERS[c['layer']]
    if c['layer'] == 'unpooler' and m <= 8:
        n = 1024
    return 2 * m * n * k, n * k * w + m * k * 2 + m * n * w


def attn_work(c, sol_ns):
    # Memory-bound at SOL, so unique KV tokens follow from the floor (Hq=16, Hkv=2, D=64).
    q, b = c.get('q_len', 1), c['batch']
    q_bytes = b * q * (2 * 16 + 2 * 2) * 64 * 2 * 2
    tokens = (sol_ns * 1e-9 * BW - q_bytes) / 256
    return 4 * tokens * q * 16 * 64, tokens * 256 + q_bytes


def load_runs(spec, work):
    profiles = {}
    for f in glob.glob(f'{ROOT}/*/muse_*/profiles/*.json') + glob.glob(f'{ROOT}/*/muse_*/profiler/progressed/*.json'):
        d = json.load(open(f))
        profiles[d['job_id'][:8]] = d
    runs = []
    for jid, label in spec:
        d = profiles[jid]
        cases = {}
        for b in d['benchmarks']:
            flops, nbytes = work(b['case'], b['speed_of_light_ns'])
            cases[(b['case']['layer'], b['case']['load'])] = dict(
                flops=flops, ai=flops / nbytes, ns=b['median_ns'], base_ns=b['baseline_ns'], speedup=b['speedup'])
        s = d['summary']
        runs.append(dict(id=jid, label=label, tip=jid in TIPS, speedup=s['speedup'], ci=s['speedup_ci'], cases=cases))
    return runs


def mix(a, b, t):
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02x%02x%02x' % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def color(s):
    t = max(-1.0, min(1.0, math.log(s) / math.log(1.5)))
    return mix('#f2f2f2', GAIN if t > 0 else LOSS, abs(t))


def pct(s):
    return f'{(s - 1) * 100:+.1f}%'.replace('-', '−')


def svg(w, h, label, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}" font-size="12" '
            f'role="img" aria-label="{label}">\n<rect width="{w}" height="{h}" fill="#ffffff"/>\n' + '\n'.join(body) + '\n</svg>\n')


def header(title, sub):
    return [f'<text x="20" y="24" font-weight="600" font-size="14" fill="{INK}">{title}</text>',
            f'<text x="20" y="42" fill="#5a5a5a">{sub}</text>']


def row_label(x, y, r):
    weight = ' font-weight="600"' if r['tip'] else ''
    return f'<text x="{x}" y="{y}" text-anchor="end"{weight} fill="{INK}">{r["label"]}</text>'


def ladder(runs, title, sub, name):
    L, R, rh, top, lo, hi = 250, 690, 18, 70, -40, 50
    X = lambda p: L + (min(max(p, lo), hi) - lo) / (hi - lo) * (R - L)
    h = top + rh * len(runs) + 46
    body = header(title, sub)
    for p in range(lo, hi + 1, 10):
        stroke = INK if p == 0 else '#e6e6e6'
        body.append(f'<line x1="{X(p):.1f}" y1="{top - 6}" x2="{X(p):.1f}" y2="{h - 40}" stroke="{stroke}"/>')
        tick = f'{p:+d}%'.replace('+0%', '0%').replace('-', '−')
        body.append(f'<text x="{X(p):.1f}" y="{h - 26}" text-anchor="middle" fill="{MUTED}" font-size="11">{tick}</text>')
    for i, r in enumerate(runs):
        y = top + i * rh
        p, (c0, c1) = (r['speedup'] - 1) * 100, r['ci']
        fill = TIP if r['tip'] else GAIN if p >= 0 else LOSS
        x0, x1 = sorted((X(0), X(p)))
        body.append(row_label(L - 8, y + 9, r))
        body.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 1):.1f}" height="12" fill="{fill}"/>')
        if p < lo:
            body.append(f'<path d="M{L + 6} {y - 1} l-4 14 M{L + 10} {y - 1} l-4 14" stroke="#fff" stroke-width="2"/>')
        else:
            body.append(f'<line x1="{X((c0 - 1) * 100):.1f}" y1="{y + 6}" x2="{X((c1 - 1) * 100):.1f}" y2="{y + 6}" stroke="{INK}"/>')
        weight = ' font-weight="600"' if r['tip'] else ''
        body.append(f'<text x="752" y="{y + 10}" text-anchor="end"{weight} fill="{INK}">{pct(r["speedup"])}</text>')
    body.append(f'<text x="{(L + R) / 2}" y="{h - 10}" text-anchor="middle" fill="{MUTED}" font-size="11">'
                'geo-mean change vs SGLang over all loads; whisker = judge 95% CI; bars clipped at −40%</text>')
    write(name, svg(760, h, title, body))


def heatmap(runs, layers, title, sub, foot, name):
    L, R, rh, top, gap = 250, 750, 14, 92, 4
    cw = (R - L - gap * (len(layers) - 1)) / (len(layers) * len(LOADS))
    h = top + rh * len(runs) + 64
    body = header(title, sub)
    for li, (layer, short) in enumerate(layers):
        gx = L + li * (len(LOADS) * cw + gap)
        body.append(f'<text x="{gx + len(LOADS) * cw / 2:.1f}" y="{top - 22}" text-anchor="middle" font-weight="600" fill="{INK}">{short}</text>')
        for j, load in enumerate(LOADS):
            x = gx + j * cw
            body.append(f'<text x="{x + cw / 2:.1f}" y="{top - 6}" text-anchor="middle" fill="{MUTED}" font-size="9">{load[0]}{load.split()[1]}</text>')
            for i, r in enumerate(runs):
                s = r['cases'][(layer, load)]['speedup']
                body.append(f'<rect x="{x:.1f}" y="{top + i * rh}" width="{cw - 1:.1f}" height="{rh - 1}" fill="{color(s)}"><title>{r["label"]}, {layer} {load}: {pct(s)}</title></rect>')
    for i, r in enumerate(runs):
        body.append(row_label(L - 8, top + i * rh + 10, r))
        if r['tip']:
            body.append(f'<rect x="{L - 1}" y="{top + i * rh - 1}" width="{R - L + 1}" height="{rh + 1}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    legend(body, L, h - 44)
    body.append(f'<text x="{L + 230}" y="{h - 34}" fill="{MUTED}" font-size="11">{foot}</text>')
    write(name, svg(760, h, title, body))


def legend(body, x, y, w=200):
    body.append('<defs><linearGradient id="g"><stop offset="0" stop-color="%s"/><stop offset=".5" stop-color="#f2f2f2"/>'
                '<stop offset="1" stop-color="%s"/></linearGradient></defs>' % (LOSS, GAIN))
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="10" fill="url(#g)" stroke="#e6e6e6"/>')
    for t, s in ((0, '≤ −33%'), (0.5, '0%'), (1, '≥ +50%')):
        body.append(f'<text x="{x + t * w}" y="{y + 24}" text-anchor="middle" fill="{MUTED}" font-size="11">{s}</text>')
    body.append(f'<text x="{x + w / 2}" y="{y - 4}" text-anchor="middle" fill="{MUTED}" font-size="11">per-case change vs SGLang</text>')


def roofline(boards, name):
    L, R, T, B = 70, 690, 60, 380
    xlo, xhi, ylo, yhi = 0.25, 512, 1e-4, 10 ** 3.5
    X = lambda v: L + math.log(v / xlo) / math.log(xhi / xlo) * (R - L)
    Y = lambda v: B - math.log10(max(v, ylo) / ylo) / math.log10(yhi / ylo) * (B - T)
    title = 'H200 roofline: every benched case from every run, both boards'
    body = header(title, 'Memory roof 4.8 TB/s, compute peak 989 TFLOPS. Color is the per-case change vs SGLang.')
    body.append(f'<rect x="{L}" y="{T}" width="{R - L}" height="{B - T}" fill="#fafafa" stroke="#e6e6e6"/>')
    xt, yt = [1, 4, 16, 64, 256], [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    for v in xt:
        body.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{B}" stroke="#e6e6e6"/>')
        body.append(f'<text x="{X(v):.1f}" y="{B + 16}" text-anchor="middle" fill="{MUTED}" font-size="11">{v}</text>')
    for v in yt:
        body.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}" stroke="#e6e6e6"/>')
        body.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="11">{"1k" if v == 1000 else f"{v:g}"}</text>')
    ridge = PEAK / BW
    body.append(f'<path d="M{L} {Y(BW * xlo / 1e12):.1f} L{X(ridge):.1f} {Y(PEAK / 1e12):.1f} L{R} {Y(PEAK / 1e12):.1f}" fill="none" stroke="{INK}" stroke-width="2"/>')
    body.append(f'<line x1="{X(ridge):.1f}" y1="{T}" x2="{X(ridge):.1f}" y2="{B}" stroke="#b5b5b5" stroke-dasharray="4 3"/>')
    body.append(f'<text x="{X(ridge) - 4:.1f}" y="{B - 8}" text-anchor="end" fill="{MUTED}" font-size="11">ridge ≈ {ridge:.0f}</text>')

    def mark(shape, x, y, fill, stroke, sw, r=3.2):
        if shape == 'circle':
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        return f'<path d="M{x:.1f} {y - r * 1.15:.1f} l{r:.1f} {r * 1.8:.1f} h{-2 * r:.1f}z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    tips, rings = [], []
    for shape, runs in boards:
        for r in runs:
            for c in r['cases'].values():
                m = mark(shape, X(c['ai']), Y(c['flops'] / c['ns'] / 1e3), color(c['speedup']), INK if r['tip'] else 'none', 1.2)
                (tips if r['tip'] else body).append(m)
        for key, c in runs[0]['cases'].items():
            base = sorted(r['cases'][key]['base_ns'] for r in runs)[len(runs) // 2]
            rings.append(mark(shape, X(c['ai']), Y(c['flops'] / base / 1e3), 'none', INK, 1, 5))
    body += rings + tips
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{INK}"/><line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="{INK}"/>')
    body.append(f'<text x="{(L + R) / 2}" y="{B + 34}" text-anchor="middle" fill="{MUTED}" font-size="11">'
                'arithmetic intensity (FLOP/byte): ≈ m for projections, ≈ 16 decode / 128 verify for attention</text>')
    body.append(f'<text x="18" y="{(T + B) / 2}" fill="{MUTED}" font-size="11" transform="rotate(-90 18 {(T + B) / 2})">achieved TFLOPS</text>')
    y = B + 62
    body.append(mark('circle', 30, y, '#bbb', 'none', 0))
    body.append(f'<text x="42" y="{y + 4}" fill="{INK}">BF16 projections</text>')
    body.append(mark('triangle', 30, y + 22, '#bbb', 'none', 0))
    body.append(f'<text x="42" y="{y + 26}" fill="{INK}">Attention decode</text>')
    body.append(mark('circle', 180, y, 'none', INK, 1, 5))
    body.append(f'<text x="192" y="{y + 4}" fill="{INK}">SGLang (median)</text>')
    body.append(mark('circle', 180, y + 22, '#bbb', INK, 1.2))
    body.append(f'<text x="192" y="{y + 26}" fill="{INK}">tip submit</text>')
    legend(body, 470, y - 4)
    write(name, svg(760, B + 112, title, body))


def write(name, text):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(text)


def main():
    bf16, attn = load_runs(BF16, bf16_work), load_runs(ATTN, attn_work)
    ladder(bf16, 'muse_bf16_proj: every run vs SGLang', f'{len(bf16)} judge runs in order. Tip +42.6% (CI +40% to +45%).', 'bf16-experiment-ladder.svg')
    ladder(attn, 'muse_attention_decode: every run vs SGLang', f'{len(attn)} judge runs in order. Tip +14.9% (CI +13% to +17%).', 'attn-experiment-ladder.svg')
    heatmap(bf16, [(k, k) for k in BF16_LAYERS], 'muse_bf16_proj: per-case change vs SGLang',
            'Rows are runs (same order as the ladder); columns are layer × load. Each layer runs 62× per step.',
            'verify 1 / 16 / 32 = m 8 / 128 / 256', 'bf16-case-heatmap.svg')
    heatmap(attn, [('sliding', 'sliding (×46)'), ('full', 'full (×16)')], 'muse_attention_decode: per-case change vs SGLang',
            'Rows are runs (same order as the ladder); columns are layer × load (batch 1 / 16 / 32).',
            'verify = 8-token block', 'attn-case-heatmap.svg')
    roofline([('circle', bf16), ('triangle', attn)], 'all-experiments-roofline.svg')


if __name__ == '__main__':
    main()
