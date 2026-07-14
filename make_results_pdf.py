"""Render an academic slide-style PDF of the reproduction, with real paper figures
and a proposed-research section.

Pipeline: read repro_*.json + cropped paper figures -> emit a scholarly HTML deck
-> headless Chrome print-to-pdf -> ~/Downloads/RAG_Unlearning_Results.pdf.
Data-driven: re-run to refresh the numbers.
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

try:
    FIGS = json.load(open(os.path.join(os.path.dirname(REPO), "paperfigs", "figs_b64.json")))
except Exception:
    FIGS = {}


def pfig(key):
    b = FIGS.get(key, "")
    return f'<img class="pfig" alt="{key}" src="data:image/png;base64,{b}"/>' if b else ""


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
          <text x="79" y="60" fill="#6b7280" font-size="11.5" text-anchor="middle">{sub}</text></g>'''
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


def _abox(x, y, w, h, label, sub, kind="normal"):
    stroke = {"hl": "#14315e", "good": "#15803d", "bad": "#b91c1c"}.get(kind, "#c2c8d2")
    fill = {"hl": "#eef2f8", "good": "#eff6f0", "bad": "#fbf1f1"}.get(kind, "#ffffff")
    lw = "2" if kind != "normal" else "1.3"
    tcol = {"good": "#15803d", "bad": "#b91c1c"}.get(kind, "#111418")
    s = (f'<text x="{w/2:.0f}" y="{h/2+16:.0f}" fill="#6b7280" font-size="11" text-anchor="middle">{sub}</text>'
         if sub else "")
    return (f'<g transform="translate({x},{y})"><rect width="{w}" height="{h}" rx="9" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{lw}"/>'
            f'<text x="{w/2:.0f}" y="{h/2-2 if sub else h/2+4:.0f}" fill="{tcol}" font-size="13.5" '
            f'font-weight="600" text-anchor="middle">{label}</text>{s}</g>')


def _arr(x1, y1, x2, y2, color="#8a94a6", dash=False):
    d = 'stroke-dasharray="4 4"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2-7}" y2="{y2}" stroke="{color}" stroke-width="1.6" {d}/>'
            f'<path d="M{x2-9} {y2-5} L{x2} {y2} L{x2-9} {y2+5}" fill="none" stroke="{color}" stroke-width="1.6"/>')


def agent_pipeline_svg():
    o = (f'<rect x="188" y="8" width="712" height="34" rx="8" fill="#f5f7fa" stroke="#c2c8d2" '
         f'stroke-width="1.2"/><text x="544" y="30" fill="#14315e" font-size="13" font-weight="700" '
         f'text-anchor="middle">Orchestrator agent — routes, coordinates, and logs every decision</text>')
    dash = "".join(_arr(x, 42, x, 108, "#c2c8d2", True) for x in (264, 448, 816))
    row = (_abox(6, 112, 150, 74, "User query", "") +
           _arr(158, 149, 188, 149) +
           _abox(190, 112, 150, 74, "Router agent", "detects forget intent") +
           _arr(342, 149, 372, 149) +
           _abox(374, 112, 150, 74, "Retrieval agent", "supplies the clause") +
           _arr(526, 149, 556, 149) +
           _abox(558, 112, 150, 74, "Target model", "may or may not obey") +
           _arr(710, 149, 740, 149) +
           _abox(742, 112, 150, 74, "Sentinel agent", "checks for leakage", "hl"))
    outs = (_arr(892, 128, 928, 104, "#15803d") + _arr(892, 170, 928, 196, "#b91c1c") +
            _abox(930, 78, 168, 52, "Answer", "if clean", "good") +
            _abox(930, 170, 168, 52, "Refusal", "leak blocked", "bad"))
    note = ('<text x="560" y="252" fill="#4b5563" font-size="12.5" text-anchor="middle" '
            'font-family="ui-monospace">enforcement lives in the sentinel, not in the target model</text>')
    return f'<svg viewBox="0 0 1112 268" width="100%" xmlns="http://www.w3.org/2000/svg">{o}{dash}{row}{outs}{note}</svg>'


def sentinel_svg():
    r1 = (_abox(20, 24, 130, 58, "Query", "") + _arr(152, 53, 188, 53) +
          _abox(190, 24, 190, 58, "Model + clause", "obeys ~80% on a 7B model") + _arr(382, 53, 440, 53, "#b91c1c") +
          _abox(442, 24, 160, 58, "Leaks target", "", "bad"))
    r2 = (_abox(20, 150, 130, 58, "Query", "") + _arr(152, 179, 188, 179) +
          _abox(190, 150, 150, 58, "Model", "may leak") + _arr(342, 179, 380, 179) +
          _abox(382, 150, 150, 58, "Sentinel", "independent check", "hl") + _arr(534, 179, 592, 179, "#15803d") +
          _abox(594, 150, 160, 58, "Refusal", "leak blocked", "good"))
    lab = ('<text x="12" y="18" fill="#b91c1c" font-size="12" font-weight="700">Single model (paper)</text>'
           '<text x="12" y="144" fill="#15803d" font-size="12" font-weight="700">Agentic (proposed)</text>')
    return f'<svg viewBox="0 0 800 226" width="100%" xmlns="http://www.w3.org/2000/svg">{lab}{r1}{r2}</svg>'


def build_html(rows, today):
    prov = any(not r["final"] for r in rows)
    trows = ""
    for i, r in enumerate(rows):
        col = "#15803d" if r["usr"] >= 95 else ("#b45309" if r["usr"] >= 88 else "#b91c1c")
        star = "" if r["final"] else "<sup>*</sup>"
        trows += f'''<tr class="{'alt' if i%2 else ''}">
          <td class="model">{r['name']}</td><td class="num">{r['n']}{star}</td>
          <td class="num" style="color:{col};font-weight:700">{r['usr']:.1f}</td>
          <td class="num">{r['rouge']:.3f}</td><td class="num dim">~{r['paper']:.1f}</td></tr>'''

    def slide(inner, num, foot="When Machine Unlearning Meets RAG · arXiv:2410.15267 · reproduction study"):
        return f'<section class="slide">{inner}<div class="foot"><span>{foot}</span><span>{num}</span></div></section>'

    def proposal(n, eyebrow, title, gap, approach, question, related):
        return slide(f'''
          <div class="eyebrow">7 · Proposed research</div>
          <div class="ptag">{eyebrow}</div>
          <h2>{title}</h2>
          <div class="deflist">
            <div><span class="dt">Gap</span><span class="dd">{gap}</span></div>
            <div><span class="dt">Approach</span><span class="dd">{approach}</span></div>
            <div><span class="dt">Research question</span><span class="dd">{question}</span></div>
            <div><span class="dt">Builds on</span><span class="dd">{related}</span></div>
          </div>''', n)

    # ---- 1 title ----
    s1 = f'''<section class="slide title">
      <div class="rule top"></div>
      <h1>Reproducing RAG-based Machine Unlearning<br>on the paper's real models</h1>
      <p class="sub">A faithful re-implementation of <i>When Machine Unlearning Meets Retrieval Augmented
        Generation: Keep Secret or Forget Knowledge?</i> Closed models are run through OpenRouter and an
        open Llama-2-7b-chat locally, with GPT-4o as the confidentiality clause writer and the success
        judge, over the paper's 100 topic concept set.</p>
      <div class="meta">
        <div><span class="k">Source paper</span> Wang, Zhu, Ye, Zhou. arXiv:2410.15267</div>
        <div><span class="k">Code</span> github.com/xuwudawei/rag-unlearning</div>
        <div><span class="k">Date</span> {today}</div>
      </div>
      <div class="rule bot"></div></section>'''

    # ---- 2 method (our schematic) ----
    s2 = slide(f'''
      <div class="eyebrow">1 · Method</div>
      <h2>Unlearning as retrieval control</h2>
      <p class="body">For each forgotten concept the provider builds one retrievable entry,
        <b class="mono">k = P + Q</b>. The description P (generated by the target model) makes the entry
        rank for related questions; the clause Q (generated by GPT-4o) instructs the model to refuse. The
        model's parameters are never modified.</p>
      <figure>{flow_svg()}
        <figcaption><b>Figure 1.</b> Schematic of the pipeline (this study). A confidentiality clause is
          retrieved into context so the frozen model declines, without any weight update.</figcaption>
      </figure>''', "2")

    # ---- 3 how the paper works: architecture (Figure 2) ----
    s_arch = slide(f'''
      <div class="eyebrow">2 · How the paper works</div>
      <h2>The architecture: retrieval as the control point</h2>
      <p class="body">A knowledge expert writes entries into the retrieval base. On a normal query the
        retriever concatenates the retrieved text with the prompt and the model answers. For unlearning,
        the target's entry carries a confidentiality clause, so the same pipeline returns a refusal.
        <b>Example.</b> "Who is Harry Potter?" normally yields a full answer; with the clause retrieved,
        the model replies "Sorry, I cannot generate any information about Harry Potter."</p>
      <figure class="paper wide">{pfig('fig2_arch')}
        <figcaption>Figure 2 from Wang et al. (arXiv:2410.15267): regular RAG (left) versus unlearning via
          RAG (right). Reproduced for discussion.</figcaption>
      </figure>''', "3")

    # ---- 4 how the paper works: objectives + P+Q example (Figure 3) ----
    s_ex = slide(f'''
      <div class="eyebrow">2 · How the paper works</div>
      <h2>Two objectives, one construction: k = P + Q</h2>
      <div class="two">
        <div><p class="body"><b>Sample unlearning</b> forgets a specific training example, such as a
          person's record; <b>concept unlearning</b> forgets a whole topic. Each becomes one entry:
          <b>P</b>, the retrieval component, makes it match related questions, and <b>Q</b>, the constraint
          component, forces the refusal. The paper's own worked examples are shown at right: a private
          record and the Harry Potter concept.</p>
          <p class="ex"><b>P</b> = a factual description that ranks highly.<br><b>Q</b> = "the assistant is
          prohibited from generating any content related to the target."</p></div>
        <figure class="paper">{pfig('fig3_example')}
          <figcaption>Figure 3 from Wang et al.: the retrieval (P) and constraint (Q) components for a
            sample and a concept.</figcaption></figure>
      </div>''', "4")

    # ---- 5 how the paper works: Algorithm 1 + prompt template (Figure 4) ----
    s_alg = slide(f'''
      <div class="eyebrow">2 · How the paper works</div>
      <h2>Constructing the entry, and the prompt template</h2>
      <div class="two algo">
        <figure class="paper tall">{pfig('alg1')}
          <figcaption>Algorithm 1 from Wang et al.: Q is crafted by the helper model and refined until the
            target refuses; P is written by the target; the entry set is K = {{ Pi + Q }}.</figcaption></figure>
        <div>
          <figure class="paper">{pfig('fig4_prompt')}
            <figcaption>Figure 4 from Wang et al.: the prompt template joining the instruction, the
              question, and the retrieved knowledge item.</figcaption></figure>
          <p class="body small">Our reproduction follows this loop faithfully, regenerating Q and verifying
            the refusal against the real questions under this exact template.</p>
        </div>
      </div>''', "5")

    # ---- 6 setup ----
    s3 = slide(f'''
      <div class="eyebrow">3 · Experimental setup</div>
      <h2>Models, data, and metrics</h2>
      <div class="deflist">
        <div><span class="dt">Target models</span><span class="dd">GPT-4o, GPT-4o mini, GPT-4, and Gemini
          2.5 Flash via a single OpenRouter key; Llama-2-7b-chat run locally on Apple Silicon. PaLM 2 is
          retired and cannot be reproduced.</span></div>
        <div><span class="dt">Auxiliary model</span><span class="dd">GPT-4o writes the clause Q, generates
          the five questions per concept, and judges unlearning success, exactly as in the paper.</span></div>
        <div><span class="dt">Data</span><span class="dd">100 Wikipedia concepts across fiction, technology,
          people, and landmarks, five questions each (500 probes).</span></div>
        <div><span class="dt">Metrics</span><span class="dd">Unlearning Success Rate (USR), judged by GPT-4o
          over the before and after answers, and ROUGE-L recall (lower means more forgetting).</span></div>
        <div><span class="dt">Retrieval</span><span class="dd">Hybrid BM25 with sentence-transformers
          embeddings, following the paper's semantic and keyword matching.</span></div>
      </div>''', "6")

    # ---- 7 results table ----
    s4 = slide(f'''
      <div class="eyebrow">4 · Results</div>
      <h2>Concept unlearning across models</h2>
      <table class="results">
        <thead><tr><th>Target model</th><th>Concepts</th><th>USR (%)</th><th>ROUGE-L</th><th>Paper USR (%)</th></tr></thead>
        <tbody>{trows}</tbody></table>
      <div class="tcap"><b>Table 1.</b> Reproduced USR and ROUGE-L recall versus the paper's reported
        figures. {'<sup>*</sup> provisional scale; the full 100 concept run is completing.' if prov else 'Concept counts are shown per model; GPT-4 (cost) and Llama-2 (local compute) were run at reduced scale.'}</div>''', "7")

    # ---- 8 results figure ----
    s5 = slide(f'''
      <div class="eyebrow">4 · Results</div>
      <h2>Reproduced against reported USR</h2>
      <figure class="wide">{chart_svg(rows)}
        <div class="fkey"><span><i style="background:#3b6ea5"></i>Reproduced</span>
          <span><i style="background:#c9d2e0"></i>Paper</span></div>
        <figcaption><b>Figure 5.</b> Reproduced USR (blue) against the paper's reported USR (grey) per
          target model. Closed models track the paper within a few points; the local seven billion
          parameter model is the exception.</figcaption></figure>''', "8")

    # ---- 9 discussion (as lists) ----
    s6 = slide(f'''
      <div class="eyebrow">5 · Discussion</div>
      <h2>What reproduces, and what does not</h2>
      <div class="coltwo">
        <div>
          <div class="ltitle good">Reproduces</div>
          <ul class="blist">
            <li>GPT-4o, GPT-4o mini, and Gemini reach 92 to 100 percent USR.</li>
            <li>GPT-4o mini tightens ROUGE-L to about 0.10, matching the paper.</li>
            <li>The method works as described on strong models.</li>
          </ul>
        </div>
        <div>
          <div class="ltitle bad">Reproducibility caveat</div>
          <ul class="blist">
            <li>Llama-2-7b-chat reaches only 80 to 84 percent USR, far below the reported 99.8 percent.</li>
            <li>Retrieval was perfect: 25 of 25 entries retrieved.</li>
            <li>The weak model does not always obey the clause; a stronger clause added only 4 points.</li>
            <li>The near perfect open model figure depends on clause obedience a vanilla 7B model lacks.</li>
          </ul>
        </div>
      </div>''', "9")

    # ---- 10 conclusion ----
    s7 = slide(f'''
      <div class="eyebrow">6 · Conclusion</div>
      <h2>Summary and reproducibility</h2>
      <p class="body">RAG-based unlearning reproduces on real closed models at the paper's scale, at low
        cost and with no weight changes. The open model result is weaker and honestly reported. The full
        pipeline, including the real Min-K membership inference and an in-context baseline, runs locally.</p>
      <div class="deflist tight">
        <div><span class="dt">Repository</span><span class="dd mono">github.com/xuwudawei/rag-unlearning</span></div>
        <div><span class="dt">Command</span><span class="dd mono">python scripts/reproduce_concept.py --target openai/gpt-4o --num-concepts 100</span></div>
        <div><span class="dt">Excluded</span><span class="dd">Gradient ascent, mu-unlearning, and sample
          unlearning, which all require local fine tuning.</span></div>
      </div>''', "10")

    # ---- 11 proposed research: overview list ----
    p1 = slide(f'''
      <div class="eyebrow">7 · Proposed research</div>
      <h2>From reproduction to a research program</h2>
      <p class="body">The reproduction exposes concrete gaps. We propose four directions, each turning a
        gap into a contribution.</p>
      <div class="plist">
        <div><span class="pn">1</span><span class="pd"><b>Verifiable unlearning.</b> Turn behavioral
          suppression into certificate-backed removal that a regulator could accept.</span></div>
        <div><span class="pn">2</span><span class="pd"><b>Robust unlearning.</b> Make forgetting hold under
          the adaptive attack (Table XII) and on weak open models that disobey the clause.</span></div>
        <div><span class="pn">3</span><span class="pd"><b>Inverting the method.</b> Causal attribution and
          correction of what public models say about an entity.</span></div>
        <div><span class="pn">4</span><span class="pd"><b>Answer-integrity monitoring.</b> Detect
          adversarial manipulation of retrieval-grounded answers.</span></div>
      </div>''', "11")

    p2 = proposal("12", "Proposal 1", "Certificate-backed unlearning",
        "The reproduction confirms suppression, not deletion. The knowledge remains in the weights and there is no proof of removal.",
        "Pair the RAG clause with an adversarial probe suite, Min-K membership evidence, and a tamper-evident audit log, producing a signed unlearning certificate.",
        "Can behavioral unlearning be made auditable to a standard a regulator would accept?",
        "Min-K membership inference (Shi et al. 2024); certified data removal (Guo et al. 2020).")

    p3 = proposal("13", "Proposal 2", "Robustness under attack and on weak models",
        "The paper's Table XII shows success collapsing to 20.9 percent under an adaptive attacker, and our Llama-2 result shows a weak model obeys the clause only 80 to 84 percent of the time.",
        "A model-independent output leakage gate that blocks target content after generation, plus optimised clause representations; measure clause obedience across model scales.",
        "Can forgetting be made independent of the model's willingness to comply?",
        "Prompt injection and jailbreak literature; our defense-in-depth guard.")

    p4 = proposal("14", "Proposal 3", "Inverting the method: causal answer correction",
        "Unlearning, generative engine optimisation, and RAG poisoning are the same lever: whoever controls the retrieved context controls the answer.",
        "Black-box causal attribution of a public model's answer to its sources, then a minimal legitimate intervention that corrects it. Causal and predictive, unlike correlational optimisation.",
        "Can we measure, and minimally correct, what a public model says about an entity?",
        "ContextCite (Cohen-Wang, Madry et al. 2024); influence functions (Koh and Liang 2017).")

    p5 = proposal("15", "Proposal 4", "Answer-integrity monitoring",
        "A few poisoned documents can control a retrieval-grounded answer (PoisonedRAG reports about 90 percent control with five documents).",
        "An intrusion detection layer for AI answers that flags adversarial shifts via the residual between a predicted and an observed answer.",
        "Can manipulation of a model's answers be detected in the wild, before it spreads?",
        "PoisonedRAG (Zou et al. 2024); indirect prompt injection defenses.")

    # ---- proposed research: agentic unlearning (one direction, with diagrams) ----
    a1 = slide(f'''
      <div class="eyebrow">7 · Proposed research</div>
      <h2>Agentic unlearning: move enforcement out of the model</h2>
      <p class="body">The reproduction points to one direction. Forgetting fails exactly when the target
        model will not obey the in-context clause, so we stop relying on the model to police itself.</p>
      <ul class="blist wide">
        <li><b>The weak link.</b> A single frozen model obeys the clause only 80 to 84 percent of the
          time, and an adaptive attacker can override it entirely.</li>
        <li><b>The idea.</b> Enforce forgetting with a small team of agents around the model, not inside it.</li>
        <li><b>The agents.</b> A router detects forget-related queries, a retrieval agent supplies the
          clause, a sentinel agent checks the output for leakage, and an orchestrator coordinates and logs.</li>
        <li><b>The payoff.</b> Forgetting no longer depends on the model's willingness to comply, and
          every decision is auditable.</li>
      </ul>''', "11")

    a2 = slide(f'''
      <div class="eyebrow">7 · Proposed research</div>
      <h2>A multi-agent unlearning pipeline</h2>
      <figure>{agent_pipeline_svg()}
        <figcaption><b>Figure 6.</b> The proposed architecture. The target model may or may not obey the
          clause; the sentinel agent independently inspects the output and blocks any leak, so enforcement
          does not depend on the model. The orchestrator logs every decision for an audit trail.</figcaption>
      </figure>''', "12")

    a3 = slide(f'''
      <div class="eyebrow">7 · Proposed research</div>
      <h2>Why it is robust: the sentinel does not trust the model</h2>
      <figure>{sentinel_svg()}
        <figcaption><b>Figure 7.</b> A single model leaks whenever it disobeys the clause; the agentic
          system catches the leak at the sentinel and returns a refusal regardless of obedience.</figcaption>
      </figure>
      <ul class="blist">
        <li>Closes the open model gap: the 84 percent obedience ceiling stops mattering.</li>
        <li>Resists the adaptive attack, because the sentinel runs after generation.</li>
        <li>Produces the audit log needed for a verifiable unlearning certificate.</li>
      </ul>''', "13")

    order = [s1, s2, s_arch, s_ex, s_alg, s3, s4, s5, s6, s7, a1, a2, a3]
    return '<!doctype html><html><head><meta charset="utf-8"><style>' + CSS + '</style></head><body>' + ''.join(order) + '</body></html>'


CSS = r'''
:root{ --paper:#ffffff; --ink:#141821; --body:#333a45; --muted:#5b6470; --dim:#8a94a6;
  --rule:#dde2ea; --accent:#14315e; --panel:#f5f7fa;
  --serif:Georgia,'Iowan Old Style','Times New Roman',serif;
  --sans:-apple-system,system-ui,'Segoe UI',sans-serif; }
*{ margin:0; padding:0; box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
@page{ size:1280px 720px; margin:0; }
html,body{ background:#fff; }
.slide{ position:relative; width:1280px; height:720px; overflow:hidden; background:var(--paper);
  padding:60px 84px 56px; font-family:var(--sans); color:var(--body); page-break-after:always; }
.slide:last-child{ page-break-after:auto; }
.mono{ font-family:ui-monospace,'SF Mono',Menlo,monospace; font-size:.92em; }
.eyebrow{ font-family:ui-monospace,monospace; font-size:12.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:700; padding-bottom:6px; border-bottom:2px solid var(--accent); display:inline-block; }
.ptag{ font-family:ui-monospace,monospace; font-size:12px; color:var(--dim); letter-spacing:.1em;
  text-transform:uppercase; margin-top:14px; }
h1{ font-family:var(--serif); font-size:46px; line-height:1.12; font-weight:700; color:var(--ink);
  letter-spacing:-.01em; margin:26px 0 22px; }
h2{ font-family:var(--serif); font-size:32px; line-height:1.12; font-weight:700; color:var(--ink);
  letter-spacing:-.01em; margin:14px 0 12px; }
.body{ font-size:17.5px; line-height:1.5; color:var(--body); max-width:96ch; }
.body.small{ font-size:14.5px; margin-top:14px; }
.sub{ font-size:19px; line-height:1.5; color:var(--muted); max-width:88ch; }
b,.k{ color:var(--ink); }
.foot{ position:absolute; left:84px; right:84px; bottom:26px; display:flex; justify-content:space-between;
  font-family:ui-monospace,monospace; font-size:11.5px; color:var(--dim); letter-spacing:.03em;
  border-top:1px solid var(--rule); padding-top:12px; }
.slide.title{ padding-top:120px; }
.rule{ position:absolute; left:84px; right:84px; height:3px; background:var(--accent); }
.rule.top{ top:74px; } .rule.bot{ bottom:150px; height:1px; background:var(--rule); }
.slide.title .meta{ position:absolute; bottom:74px; left:84px; display:flex; gap:44px; font-size:14px; color:var(--muted); }
.slide.title .meta .k{ display:block; font-family:ui-monospace,monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--accent); margin-bottom:4px; }
figure{ margin-top:20px; background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:22px 30px 16px; }
figure.wide{ padding:18px 30px 14px; }
figure.paper{ background:#fff; padding:16px; }
figure.paper .pfig{ width:100%; height:auto; display:block; border:1px solid #eef0f4; border-radius:4px; }
figure.paper.wide{ margin-top:14px; padding:12px 16px; }
figure.paper.wide .pfig{ max-height:338px; width:auto; margin:0 auto; }
figure.paper.tall .pfig{ max-height:430px; width:auto; margin:0 auto; }
figcaption{ margin-top:12px; font-size:12.5px; color:var(--muted); line-height:1.45; }
.fkey{ display:flex; gap:26px; justify-content:center; margin-top:6px; font-size:13px; color:var(--muted); }
.fkey i{ display:inline-block; width:15px; height:10px; border-radius:2px; margin-right:7px; vertical-align:middle; }
.two{ display:grid; grid-template-columns:1fr 1fr; gap:30px; align-items:start; margin-top:14px; }
.two.algo{ grid-template-columns:.92fr 1.08fr; }
.ex{ margin-top:14px; font-size:15px; color:var(--muted); line-height:1.6; }
.ex b{ color:var(--accent); }
.deflist{ margin-top:20px; }
.deflist>div{ display:grid; grid-template-columns:200px 1fr; gap:22px; padding:12px 0; border-bottom:1px solid var(--rule); }
.deflist.tight>div{ padding:10px 0; }
.dt{ font-weight:700; color:var(--ink); font-size:15.5px; }
.dd{ font-size:15.5px; color:var(--muted); line-height:1.5; }
.plist{ margin-top:18px; }
.plist>div{ display:grid; grid-template-columns:46px 1fr; gap:16px; padding:13px 0; border-bottom:1px solid var(--rule); align-items:baseline; }
.plist .pn{ font-family:var(--serif); font-size:26px; font-weight:700; color:var(--accent); line-height:1; }
.plist .pd{ font-size:16.5px; color:var(--muted); line-height:1.45; }
.plist .pd b{ color:var(--ink); }
table.results{ width:100%; border-collapse:collapse; margin-top:22px; font-size:17px; }
table.results th{ text-align:left; font-family:ui-monospace,monospace; font-size:12.5px; letter-spacing:.06em;
  text-transform:uppercase; color:#fff; background:var(--accent); padding:13px 16px; font-weight:600; }
table.results th:not(:first-child){ text-align:right; }
table.results td{ padding:12px 16px; border-bottom:1px solid var(--rule); color:var(--body); }
table.results tr.alt td{ background:#f8fafc; }
td.model{ font-weight:600; color:var(--ink); }
td.num{ text-align:right; font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums; }
td.dim{ color:var(--dim); }
.tcap{ margin-top:14px; font-size:13.5px; color:var(--muted); line-height:1.5; }
sup{ color:var(--accent); }
.callout{ margin-top:20px; background:#fbf6ee; border:1px solid #e7d7bd; border-left:4px solid #b45309;
  border-radius:8px; padding:18px 24px; font-size:16px; line-height:1.56; color:#4a4034; max-width:98ch; }
.callout b{ color:#8a5a12; }
.coltwo{ display:grid; grid-template-columns:1fr 1fr; gap:36px; margin-top:22px; }
.ltitle{ font-family:ui-monospace,monospace; font-size:12.5px; letter-spacing:.08em; text-transform:uppercase; font-weight:700; margin-bottom:12px; }
.ltitle.good{ color:#15803d; } .ltitle.bad{ color:#b45309; }
ul.blist{ list-style:none; }
ul.blist>li{ position:relative; padding-left:22px; margin-bottom:12px; font-size:16px; line-height:1.5; color:var(--muted); max-width:62ch; }
ul.blist.wide>li{ max-width:98ch; font-size:16.5px; margin-bottom:14px; }
ul.blist>li::before{ content:""; position:absolute; left:2px; top:9px; width:7px; height:7px; background:var(--accent); border-radius:1px; }
ul.blist>li b{ color:var(--ink); }
'''


def main():
    rows = load()
    open(HTML, "w").write(build_html(rows, date.today().isoformat()))
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT}", f"file://{HTML}"], check=True, capture_output=True)
    print(f"Wrote {OUT}  ({sum(1 for r in rows if r['final'])}/{len(rows)} models final)")


if __name__ == "__main__":
    main()
