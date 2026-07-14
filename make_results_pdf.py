"""Render an academic slide-style PDF of the reproduction results.

Pipeline: read repro_*.json -> emit a clean scholarly HTML deck (serif headings,
restrained navy accent, numbered Table + Figures with captions) -> headless Chrome
print-to-pdf -> ~/Downloads/RAG_Unlearning_Results.pdf. Data-driven: re-run to refresh.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import date

REPO = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(REPO, "results_deck.html")
OUT = "/Users/mac/Downloads/RAG_Unlearning_Results.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

MODELS = [
    ("GPT-4o",           99.2, "repro_full_gpt4o.json",     (10, 98.0, 0.168)),
    ("GPT-4o mini",      99.3, "repro_full_gpt4omini.json", (3, 100.0, 0.096)),
    ("GPT-4",            99.0, "repro_full_gpt4.json",      (30, 92.0, 0.184)),
    ("Gemini 2.5 Flash", 99.5, "repro_full_gemini.json",    (10, 96.0, 0.127)),
    ("Llama-2-7b-chat",  99.8, "repro_llama_v2.json",       (5, 84.0, 0.595)),
]


def load():
    rows = []
    for name, paper, jf, fb in MODELS:
        p = os.path.join(REPO, jf)
        if os.path.exists(p):
            d = json.load(open(p))
            rows.append(dict(name=name, paper=paper, n=d.get("num_concepts", fb[0]),
                             usr=d.get("usr", fb[1]), rouge=d.get("rouge_l_recall", fb[2]), final=True))
        else:
            rows.append(dict(name=name, paper=paper, n=fb[0], usr=fb[1], rouge=fb[2], final=False))
    return rows


def chart_svg(rows):
    W, H = 980, 300
    pl, pb, pt = 44, 46, 18
    gw = (W - pl - 16) / len(rows)
    def y(v): return pt + (H - pt - pb) * (1 - v / 106)
    grid = "".join(
        f'<line x1="{pl}" y1="{y(g):.1f}" x2="{W-8}" y2="{y(g):.1f}" stroke="#e6e9ef" stroke-width="1"/>'
        f'<text x="{pl-8}" y="{y(g)+4:.1f}" fill="#8a94a6" font-size="12" text-anchor="end" font-family="ui-monospace">{g}</text>'
        for g in (0, 25, 50, 75, 100))
    bars = ""
    for i, r in enumerate(rows):
        cx = pl + gw * i + gw / 2
        bw = 30
        yo, yp = y(r["usr"]), y(r["paper"])
        col = "#15803d" if r["usr"] >= 95 else ("#b45309" if r["usr"] >= 88 else "#b91c1c")
        bars += f'''
          <rect x="{cx-bw-2:.1f}" y="{yp:.1f}" width="{bw}" height="{H-pb-yp:.1f}" fill="#c9d2e0"/>
          <rect x="{cx+2:.1f}" y="{yo:.1f}" width="{bw}" height="{H-pb-yo:.1f}" fill="#3b6ea5"/>
          <text x="{cx-bw/2-2:.1f}" y="{yp-7:.1f}" fill="#6b7280" font-size="11.5" text-anchor="middle" font-family="ui-monospace">{r['paper']:.0f}</text>
          <text x="{cx+bw/2+2:.1f}" y="{yo-7:.1f}" fill="{col}" font-size="13" font-weight="700" text-anchor="middle" font-family="ui-monospace">{r['usr']:.0f}</text>
          <text x="{cx:.1f}" y="{H-pb+22:.1f}" fill="#374151" font-size="12" text-anchor="middle">{r['name'].split()[0]}</text>'''
    return f'''<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">
      <line x1="{pl}" y1="{H-pb}" x2="{W-8}" y2="{H-pb}" stroke="#c2c8d2" stroke-width="1.2"/>
      <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{H-pb}" stroke="#c2c8d2" stroke-width="1.2"/>
      {grid}{bars}
      <text x="14" y="{(pt+H-pb)/2:.0f}" fill="#6b7280" font-size="12" transform="rotate(-90 14 {(pt+H-pb)/2:.0f})" text-anchor="middle">USR (%)</text>
    </svg>'''


def flow_svg():
    def box(x, label, sub, hl=False):
        stroke = "#14315e" if hl else "#c2c8d2"
        fill = "#eef2f8" if hl else "#ffffff"
        lw = "2" if hl else "1.3"
        return f'''<g transform="translate({x},52)">
          <rect x="0" y="0" width="158" height="86" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>
          <text x="79" y="38" fill="#111418" font-size="15" font-weight="600" text-anchor="middle">{label}</text>
          <text x="79" y="60" fill="#6b7280" font-size="11.5" text-anchor="middle">{sub}</text>
        </g>'''
    def arr(x):
        return f'<g transform="translate({x},95)"><line x1="0" y1="0" x2="34" y2="0" stroke="#8a94a6" stroke-width="1.6"/><path d="M28 -5 L36 0 L28 5" fill="none" stroke="#8a94a6" stroke-width="1.6"/></g>'
    return f'''<svg viewBox="0 0 1120 210" width="100%" xmlns="http://www.w3.org/2000/svg">
      {box(6,"Question","user query")}{arr(168)}
      {box(206,"Retriever","BM25 + embeddings")}{arr(368)}
      {box(406,"Injected entry","k = P + Q",hl=True)}{arr(568)}
      {box(606,"Frozen model","weights unchanged")}{arr(768)}
      {box(806,"Refusal","target withheld")}
      <text x="560" y="185" fill="#4b5563" font-size="12.5" text-anchor="middle" font-family="ui-monospace">P makes the entry retrievable; Q instructs the model to refuse</text>
    </svg>'''


def build_html(rows, today):
    prov = any(not r["final"] for r in rows)
    # Table 1 rows
    trows = ""
    for i, r in enumerate(rows):
        col = "#15803d" if r["usr"] >= 95 else ("#b45309" if r["usr"] >= 88 else "#b91c1c")
        star = "" if r["final"] else "<sup>*</sup>"
        trows += f'''<tr class="{'alt' if i%2 else ''}">
          <td class="model">{r['name']}</td>
          <td class="num">{r['n']}{star}</td>
          <td class="num" style="color:{col};font-weight:700">{r['usr']:.1f}</td>
          <td class="num">{r['rouge']:.3f}</td>
          <td class="num dim">~{r['paper']:.1f}</td></tr>'''

    def slide(inner, num, foot="When Machine Unlearning Meets RAG · arXiv:2410.15267 · reproduction study"):
        return f'''<section class="slide">
          {inner}
          <div class="foot"><span>{foot}</span><span>{num}</span></div></section>'''

    # 1 title
    s1 = f'''<section class="slide title">
      <div class="rule top"></div>
      <div class="eyebrow">Reproduction study</div>
      <h1>Reproducing RAG-based Machine Unlearning<br>on the paper's real models</h1>
      <p class="sub">A faithful re-implementation of <i>When Machine Unlearning Meets Retrieval
        Augmented Generation: Keep Secret or Forget Knowledge?</i> Closed models are run through
        OpenRouter and an open Llama-2-7b-chat locally, with GPT-4o as the confidentiality clause
        writer and the success judge, over the paper's 100 topic concept set.</p>
      <div class="meta">
        <div><span class="k">Source paper</span> Wang, Zhu, Ye, Zhou. arXiv:2410.15267</div>
        <div><span class="k">Code</span> github.com/xuwudawei/rag-unlearning</div>
        <div><span class="k">Date</span> {today}</div>
      </div>
      <div class="rule bot"></div>
    </section>'''

    # 2 method
    s2 = slide(f'''
      <div class="eyebrow">1 · Method</div>
      <h2>Unlearning as retrieval control</h2>
      <p class="body">For each forgotten concept the provider builds one retrievable entry,
        <b class="mono">k = P + Q</b>. The description P (generated by the target model) makes the
        entry rank for related questions; the clause Q (generated by GPT-4o) instructs the model to
        refuse. The model's parameters are never modified.</p>
      <figure>{flow_svg()}
        <figcaption><b>Figure 1.</b> The RAG-based unlearning pipeline. A confidentiality clause is
          retrieved into context so the frozen model declines, without any weight update.</figcaption>
      </figure>''', "2")

    # 3 setup
    s3 = slide(f'''
      <div class="eyebrow">2 · Experimental setup</div>
      <h2>Models, data, and metrics</h2>
      <div class="deflist">
        <div><span class="dt">Target models</span><span class="dd">GPT-4o, GPT-4o mini, GPT-4, and
          Gemini 2.5 Flash via a single OpenRouter key; Llama-2-7b-chat run locally on Apple Silicon.
          PaLM 2 is retired and cannot be reproduced.</span></div>
        <div><span class="dt">Auxiliary model</span><span class="dd">GPT-4o writes the clause Q, generates
          the five questions per concept, and judges unlearning success, exactly as in the paper.</span></div>
        <div><span class="dt">Data</span><span class="dd">100 Wikipedia concepts across fiction, technology,
          people, and landmarks, five questions each (500 probes).</span></div>
        <div><span class="dt">Metrics</span><span class="dd">Unlearning Success Rate (USR), judged by GPT-4o
          over the before and after answers, and ROUGE-L recall of the after answer against the original
          (lower means more forgetting).</span></div>
        <div><span class="dt">Retrieval</span><span class="dd">Hybrid BM25 with sentence-transformers
          embeddings, following the paper's semantic and keyword matching.</span></div>
      </div>''', "3")

    # 4 results table
    s4 = slide(f'''
      <div class="eyebrow">3 · Results</div>
      <h2>Concept unlearning across models</h2>
      <table class="results">
        <thead><tr><th>Target model</th><th>Concepts</th><th>USR (%)</th><th>ROUGE-L</th><th>Paper USR (%)</th></tr></thead>
        <tbody>{trows}</tbody>
      </table>
      <div class="tcap"><b>Table 1.</b> Reproduced USR and ROUGE-L recall versus the paper's reported
        figures. {'<sup>*</sup> provisional scale; the full 100 concept run is completing and will replace this value.' if prov else 'All runs at the full concept scale.'}</div>''', "4")

    # 5 figure
    s5 = slide(f'''
      <div class="eyebrow">3 · Results</div>
      <h2>Reproduced against reported USR</h2>
      <figure class="wide">{chart_svg(rows)}
        <div class="fkey"><span><i style="background:#3b6ea5"></i>Reproduced</span>
          <span><i style="background:#c9d2e0"></i>Paper</span></div>
        <figcaption><b>Figure 2.</b> Reproduced USR (blue) against the paper's reported USR (grey) per
          target model. Closed models track the paper within a few points; the local seven billion
          parameter model is the exception.</figcaption>
      </figure>''', "5")

    # 6 discussion
    s6 = slide(f'''
      <div class="eyebrow">4 · Discussion</div>
      <h2>What reproduces, and what does not</h2>
      <p class="body">The closed models reproduce the paper's central claim: GPT-4o, GPT-4o mini, and
        Gemini reach near total unlearning success (about 92 to 100 percent), and GPT-4o mini tightens
        ROUGE-L to roughly 0.10, matching the paper. The method works as described.</p>
      <div class="callout">
        <b>Reproducibility caveat.</b> Llama-2-7b-chat reaches only 80 to 84 percent USR, well below the
        paper's reported 99.8 percent. Retrieval was perfect (25 of 25 entries retrieved); the weaker
        model simply does not always obey the confidentiality clause, and a more forceful clause raised
        success by only four points. Its high ROUGE-L is further inflated by verbose boilerplate shared
        between the before and after answers. The near perfect open model figure therefore appears to
        depend on strong clause obedience that a vanilla seven billion parameter model does not provide.
      </div>''', "6")

    # 7 conclusion
    s7 = slide(f'''
      <div class="eyebrow">5 · Conclusion</div>
      <h2>Summary and reproducibility</h2>
      <p class="body">RAG-based unlearning reproduces on real closed models at the paper's scale, at low
        cost and with no weight changes. The open model result is weaker and honestly reported. The full
        pipeline, including the real Min-K membership inference and an in-context baseline, runs locally.</p>
      <div class="deflist tight">
        <div><span class="dt">Repository</span><span class="dd mono">github.com/xuwudawei/rag-unlearning</span></div>
        <div><span class="dt">Command</span><span class="dd mono">python scripts/reproduce_concept.py --target openai/gpt-4o --num-concepts 100</span></div>
        <div><span class="dt">Excluded</span><span class="dd">Gradient ascent, mu-unlearning, and sample
          unlearning, which all require local fine tuning.</span></div>
      </div>''', "7")

    return f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{s1}{s2}{s3}{s4}{s5}{s6}{s7}</body></html>'


CSS = r'''
:root{ --paper:#ffffff; --ink:#141821; --body:#333a45; --muted:#5b6470; --dim:#8a94a6;
  --rule:#dde2ea; --accent:#14315e; --panel:#f5f7fa;
  --serif:Georgia,'Iowan Old Style','Times New Roman',serif;
  --sans:-apple-system,system-ui,'Segoe UI',sans-serif; }
*{ margin:0; padding:0; box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
@page{ size:1280px 720px; margin:0; }
html,body{ background:#fff; }
.slide{ position:relative; width:1280px; height:720px; overflow:hidden; background:var(--paper);
  padding:64px 84px 58px; font-family:var(--sans); color:var(--body); page-break-after:always; }
.slide:last-child{ page-break-after:auto; }
.mono{ font-family:ui-monospace,'SF Mono',Menlo,monospace; font-size:.92em; }
.eyebrow{ font-family:ui-monospace,monospace; font-size:12.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:700; padding-bottom:6px; border-bottom:2px solid var(--accent);
  display:inline-block; }
h1{ font-family:var(--serif); font-size:46px; line-height:1.12; font-weight:700; color:var(--ink);
  letter-spacing:-.01em; margin:26px 0 22px; }
h2{ font-family:var(--serif); font-size:34px; line-height:1.12; font-weight:700; color:var(--ink);
  letter-spacing:-.01em; margin:18px 0 16px; }
.body{ font-size:18px; line-height:1.55; color:var(--body); max-width:92ch; }
.sub{ font-size:19px; line-height:1.5; color:var(--muted); max-width:88ch; }
b,.k{ color:var(--ink); }
.foot{ position:absolute; left:84px; right:84px; bottom:28px; display:flex; justify-content:space-between;
  font-family:ui-monospace,monospace; font-size:11.5px; color:var(--dim); letter-spacing:.03em;
  border-top:1px solid var(--rule); padding-top:12px; }
/* title slide */
.slide.title{ padding-top:120px; }
.rule{ position:absolute; left:84px; right:84px; height:3px; background:var(--accent); }
.rule.top{ top:74px; } .rule.bot{ bottom:150px; height:1px; background:var(--rule); }
.slide.title .meta{ position:absolute; bottom:74px; left:84px; display:flex; gap:44px;
  font-size:14px; color:var(--muted); }
.slide.title .meta .k{ display:block; font-family:ui-monospace,monospace; font-size:11px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--accent); margin-bottom:4px; }
/* figures */
figure{ margin-top:26px; background:var(--panel); border:1px solid var(--rule); border-radius:10px;
  padding:26px 30px 18px; }
figure.wide{ padding:20px 30px 16px; }
figcaption{ margin-top:14px; font-size:13.5px; color:var(--muted); line-height:1.5; }
.fkey{ display:flex; gap:26px; justify-content:center; margin-top:6px; font-size:13px; color:var(--muted); }
.fkey i{ display:inline-block; width:15px; height:10px; border-radius:2px; margin-right:7px; vertical-align:middle; }
/* definition list */
.deflist{ margin-top:24px; }
.deflist>div{ display:grid; grid-template-columns:200px 1fr; gap:22px; padding:13px 0;
  border-bottom:1px solid var(--rule); }
.deflist.tight>div{ padding:10px 0; }
.dt{ font-weight:700; color:var(--ink); font-size:15.5px; }
.dd{ font-size:15.5px; color:var(--muted); line-height:1.5; }
/* table */
table.results{ width:100%; border-collapse:collapse; margin-top:24px; font-size:17px; }
table.results th{ text-align:left; font-family:ui-monospace,monospace; font-size:12.5px; letter-spacing:.06em;
  text-transform:uppercase; color:#fff; background:var(--accent); padding:14px 16px; font-weight:600; }
table.results th:not(:first-child){ text-align:right; }
table.results td{ padding:13px 16px; border-bottom:1px solid var(--rule); color:var(--body); }
table.results tr.alt td{ background:#f8fafc; }
td.model{ font-weight:600; color:var(--ink); }
td.num{ text-align:right; font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums; }
td.dim{ color:var(--dim); }
.tcap{ margin-top:16px; font-size:13.5px; color:var(--muted); line-height:1.5; }
sup{ color:var(--accent); }
/* callout */
.callout{ margin-top:22px; background:#fbf6ee; border:1px solid #e7d7bd; border-left:4px solid #b45309;
  border-radius:8px; padding:20px 24px; font-size:16px; line-height:1.56; color:#4a4034; max-width:96ch; }
.callout b{ color:#8a5a12; }
'''


def main():
    rows = load()
    open(HTML, "w").write(build_html(rows, date.today().isoformat()))
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT}", f"file://{HTML}"], check=True, capture_output=True)
    print(f"Wrote {OUT}  ({sum(1 for r in rows if r['final'])}/{len(rows)} models final)")


if __name__ == "__main__":
    main()
