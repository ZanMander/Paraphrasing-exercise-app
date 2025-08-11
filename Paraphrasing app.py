# app.py — Paraphrasing Practice (Streamlit) — full version with Style Delta & target guidance
# -------------------------------------------------------------------------------------------
# - Similarity: Jaccard, Sørensen–Dice, Cosine (+ stopword ignore)
# - Style difference: Burrows’s Delta (friendlier label + bands)
# - Highlights (red=reused, green=unique)
# - Readability & quality: Flesch, FK Grade, Fog, SMOG, ARI, avg sent len, % complex, TTR, lexical density
# - Target bands + delta arrows (paraphrase vs original)
# - On‑screen metrics: interactive table or compact Markdown (toggle)
# - Downloads: .txt summary + one‑click PDF (with highlights, metrics, targets/deltas, meta fields)
# -------------------------------------------------------------------------------------------

import re
import math
from collections import Counter
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st

# -------------------------------
# Tokenization & helpers
# -------------------------------
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

STOPWORDS = {
    "a","an","the","and","or","but","if","while","of","to","in","on","at","for","from",
    "with","without","by","as","is","am","are","was","were","be","been","being","that",
    "this","these","those","it","its","they","them","their","you","your","i","we","our",
    "he","she","his","her","not","no","so","such","than","then","there","here","which",
    "who","whom","what","when","where","why","how","into","over","under","again","also"
}

def tokenize(text: str):
    return WORD_RE.findall(text.lower()) if text else []

def filter_stop(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip()) if text else []
    return [p for p in parts if p]

# -------------------------------
# Syllables & counts (heuristics)
# -------------------------------
def syllables_in_word(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    syl = max(1, len(groups))
    if w.endswith("e") and syl > 1 and not w.endswith(("le","ue")):
        syl -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        syl += 1
    return max(1, syl)

def count_syllables(tokens):
    return sum(syllables_in_word(t) for t in tokens)

# -------------------------------
# Similarity metrics
# -------------------------------
def jaccard_similarity(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

def dice_similarity(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return (2 * len(a & b)) / (len(a) + len(b))

def cosine_similarity(a_tokens, b_tokens):
    a_tf, b_tf = Counter(a_tokens), Counter(b_tokens)
    if not a_tf and not b_tf: return 1.0
    if not a_tf or not b_tf:  return 0.0
    dot = sum(a_tf[t] * b_tf.get(t, 0) for t in a_tf)
    a_norm = math.sqrt(sum(v*v for v in a_tf.values()))
    b_norm = math.sqrt(sum(v*v for v in b_tf.values()))
    if a_norm == 0 or b_norm == 0: return 0.0
    return dot / (a_norm * b_norm)

# -------------------------------
# Burrows’s Delta (style difference)
# -------------------------------
def burrows_delta(orig_text: str, para_text: str, top_n: int = 150):
    A = tokenize(orig_text); B = tokenize(para_text)
    if not A or not B: return 0.0

    def rel_freq(tokens):
        c = Counter(tokens); n = len(tokens)
        return {w: c[w]/n for w in c}

    rfA, rfB = rel_freq(A), rel_freq(B)

    # focus on frequent function words seen in either text
    fw = [w for w in STOPWORDS if w in rfA or w in rfB]
    fw.sort(key=lambda w: (rfA.get(w,0)+rfB.get(w,0)), reverse=True)
    vocab = fw[:top_n] if fw else list({*rfA.keys(), *rfB.keys()})[:top_n]

    zsA, zsB = [], []
    for w in vocab:
        a, b = rfA.get(w, 0.0), rfB.get(w, 0.0)
        mu = (a + b) / 2.0
        sd = math.sqrt(((a - mu)**2 + (b - mu)**2) / 2.0)
        sd = sd if sd > 1e-9 else 1e-9
        zsA.append((a - mu) / sd)
        zsB.append((b - mu) / sd)

    delta = sum(abs(x - y) for x, y in zip(zsA, zsB)) / max(1, len(vocab))
    return delta

def interpret_delta(delta: float):
    if delta < 1.0:   return "Close style",   "🟢"
    if delta < 2.0:   return "Moderate diff", "🟡"
    return "Far style", "🔴"

# -------------------------------
# Highlighting (HTML)
# -------------------------------
CSS = """
<style>
  .hl-box{border:1px solid #e5e7eb;border-radius:8px;padding:10px;background:#fff}
  .hl-legend{font-size:0.9rem;color:#555;margin-top:6px}
  .common{background:#ffdada;border-radius:4px;padding:0 2px}
  .unique{background:#d6ffd6;border-radius:4px;padding:0 2px}
  .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace}
  .pill{display:inline-block;border-radius:999px;padding:2px 8px;border:1px solid #e5e7eb;margin-left:6px}
  .tiny{font-size:12px;color:#6b7280}
</style>
"""

def highlight_paraphrase(original_tokens, paraphrase_text):
    orig_set = set(original_tokens)
    parts = re.findall(r"\b[\w'-]+\b|\s+|[^\w\s]", paraphrase_text)
    out = []
    for p in parts:
        if re.match(r"\s+|[^\w\s]", p):
            out.append(p)
        else:
            cls = "common" if p.lower() in orig_set else "unique"
            out.append(f"<span class='{cls}'>{p}</span>")
    return f"<div class='hl-box mono'>{''.join(out)}</div>"

# -------------------------------
# Readability & quality metrics
# -------------------------------
def readability_metrics(text: str):
    sentences = split_sentences(text)
    tokens = tokenize(text)
    words = [t for t in tokens if re.search(r"[a-zA-Z]", t)]
    n_s = max(1, len(sentences))
    n_w = max(1, len(words))
    n_chars = sum(len(w) for w in words)
    syllables = count_syllables(words)
    polysyll = sum(1 for w in words if syllables_in_word(w) >= 3)
    complex_ratio = polysyll / n_w

    flesch = 206.835 - 1.015*(n_w/n_s) - 84.6*(syllables/n_w)
    fk_grade = 0.39*(n_w/n_s) + 11.8*(syllables/n_w) - 15.59
    fog = 0.4 * ((n_w/n_s) + 100*complex_ratio)
    smog = 1.0430 * math.sqrt(30 * (polysyll/n_s)) + 3.1291
    ari = 4.71*(n_chars/n_w) + 0.5*(n_w/n_s) - 21.43

    avg_sent_len = n_w / n_s
    ttr = len(set(words)) / n_w if n_w else 0.0
    content = [w for w in words if w not in STOPWORDS]
    lex_density = len(content) / n_w if n_w else 0.0

    return {
        "sentences": len(sentences),
        "words": n_w if n_w != 1 else (0 if not words else 1),
        "avg_sentence_len": avg_sent_len,
        "pct_complex_words": complex_ratio*100,
        "flesch": flesch,
        "fk_grade": fk_grade,
        "fog": fog,
        "smog": smog,
        "ari": ari,
        "ttr": ttr*100,
        "lex_density": lex_density*100,
    }

def flesch_band(score):
    if score >= 90: return "Very easy (Grade 5)"
    if score >= 80: return "Easy (Grade 6)"
    if score >= 70: return "Fairly easy (Grade 7)"
    if score >= 60: return "Plain English (Grades 8–9)"
    if score >= 50: return "Fairly difficult (Grade 10–12)"
    if score >= 30: return "Difficult (College)"
    return "Very difficult (College graduate)"

# -------------------------------
# Targets & delta logic
# -------------------------------
TARGETS = {
    "flesch": (50, 70, "range"),
    "fk_grade": (8, 12, "range"),
    "fog": (10, 15, "range"),
    "smog": (8, 12, "range"),
    "ari": (8, 12, "range"),
    "avg_sentence_len": (15, 22, "range"),
    "pct_complex_words": (10, 15, "lower"),
    "ttr": (40, 60, "range"),
    "lex_density": (45, 55, "range"),
}

DISPLAY_NAMES = {
    "flesch": "Flesch Reading Ease",
    "fk_grade": "Flesch–Kincaid Grade",
    "fog": "Gunning Fog",
    "smog": "SMOG",
    "ari": "ARI",
    "avg_sentence_len": "Avg sentence length (words)",
    "pct_complex_words": "% complex words (≥3 syllables)",
    "ttr": "Type–Token Ratio (TTR %)",
    "lex_density": "Lexical density (%)",
}

def delta_eval(name, orig, para):
    lo, hi, mode = TARGETS[name]
    if mode == "higher":
        diff = para - orig
        verdict = "better" if diff > 0 else ("worse" if diff < 0 else "flat")
    elif mode == "lower":
        diff = para - orig
        verdict = "better" if diff < 0 else ("worse" if diff > 0 else "flat")
    else:
        mid = (lo + hi) / 2
        d0 = abs(orig - mid)
        d1 = abs(para - mid)
        verdict = "better" if d1 < d0 else ("worse" if d1 > d0 else "flat")
        diff = para - orig
    return diff, verdict

def arrow(verdict):
    return "▲" if verdict == "better" else ("▼" if verdict == "worse" else "🟰")

# -------------------------------
# Explanations
# -------------------------------
def explain_metrics(jacc, dice, cos):
    pct = lambda x: f"{round(x*100)}%"
    return (
        f"**What these scores mean**  \n"
        f"**Jaccard {pct(jacc)}** – overlap of **unique words** across both texts.  \n"
        f"**Sørensen–Dice {pct(dice)}** – weighted overlap; often a bit higher than Jaccard.  \n"
        f"**Cosine {pct(cos)}** – overlap of **word‑frequency distributions**; independent of length."
    )

TARGET_GUIDE = """
**What the targets mean (and how to hit them)**

- **Flesch Reading Ease (50–70)** → Clear academic prose. *Raise it* by shortening sentences and using plainer wording.
- **FK Grade / Fog / SMOG / ARI (≈8–12)** → Typical tertiary audience. *Nudge toward band* by trimming long sentences and heavy nominalisations.
- **Avg sentence length (15–22 words)** → Mix short and medium sentences. *Fix* run-ons and break complex ideas into steps.
- **% complex words (10–15%)** → Keep jargon purposeful. *Reduce* multi‑syllable words when simpler synonyms exist.
- **TTR (40–60%)** → Balance variety and cohesion. *Improve* by avoiding repetition while keeping key terms consistent.
- **Lexical density (45–55%)** → Balance content and function words. *Adjust* with signposting and transitions if density is too high.
"""

# -------------------------------
# PDF report (ReportLab)
# -------------------------------
def make_paraphrase_markup_for_pdf(original_tokens, paraphrase_text):
    orig_set = set(original_tokens)
    parts = re.findall(r"\b[\w'-]+\b|\s+|[^\w\s]", paraphrase_text)
    out = []
    for p in parts:
        if re.match(r"\s+|[^\w\s]", p):
            out.append(p.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        else:
            color = "#c40000" if p.lower() in orig_set else "#0a7c00"
            safe = p.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            out.append(f"<font color='{color}'>{safe}</font>")
    return "".join(out)

def build_pdf_bytes(orig_text, para_text, jacc, dice, cos, delta_val, orig_read, para_read, original_tokens,
                    student_name, module, lecturer):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception as e:
        return None, e

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    H1 = styles['Heading1']; H1.fontSize = 16; H1.spaceAfter = 6
    H2 = styles['Heading2']; H2.fontSize = 13; H2.spaceAfter = 4
    Body = styles['BodyText']; Body.leading = 14
    Mono = ParagraphStyle('Mono', parent=Body, fontName='Courier', leading=14)

    story = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("Paraphrasing Report", H1))

    meta = f"Generated: {timestamp}"
    who = []
    if student_name: who.append(f"Student: <b>{student_name}</b>")
    if module: who.append(f"Module: <b>{module}</b>")
    if lecturer: who.append(f"Lecturer: <b>{lecturer}</b>")
    if who: meta += " | " + " | ".join(who)
    story.append(Paragraph(meta, Body))
    story.append(Spacer(1, 6))

    delta_label, _ = interpret_delta(delta_val)
    sim_row = [[
        Paragraph(f"<b>Jaccard</b>: {round(jacc*100)}%", Body),
        Paragraph(f"<b>Sørensen–Dice</b>: {round(dice*100)}%", Body),
        Paragraph(f"<b>Cosine</b>: {round(cos*100)}%", Body),
        Paragraph(f"<b>Style difference (Burrows’s Δ)</b>: {delta_val:.2f} — {delta_label}", Body)
    ]]
    story.append(Table(sim_row, colWidths=[120, 140, 120, 180]))
    story.append(Spacer(1, 8))

    esc = lambda s: (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    story.append(Paragraph("Original text", H2))
    story.append(Paragraph(esc(orig_text), Mono))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Paraphrased text (colour shows reuse vs unique words)", H2))
    story.append(Paragraph(make_paraphrase_markup_for_pdf(original_tokens, para_text), Body))
    story.append(Paragraph("Legend: <font color='#c40000'>red</font> = also appears in original; "
                           "<font color='#0a7c00'>green</font> = unique to paraphrase.", Body))
    story.append(Spacer(1, 8))

    expl = explain_metrics(jacc, dice, cos).replace("**","").replace("  \n","<br/>")
    story.append(Paragraph("How to read the similarity scores", H2))
    story.append(Paragraph(expl, Body))
    story.append(Spacer(1, 8))

    # Readability tables with targets & deltas
    def row_for(name):
        disp = DISPLAY_NAMES[name]
        o = orig_read[name]; p = para_read[name]
        diff, v = delta_eval(name, o, p)
        lo, hi, _ = TARGETS[name]
        return [disp, f"{o:.1f}", f"{p:.1f}", f"{diff:+.1f} {('▲' if v=='better' else '▼' if v=='worse' else '🟰')}", f"{lo:.0f}–{hi:.0f}"]

    left_names = ["flesch","fk_grade","avg_sentence_len","pct_complex_words"]
    right_names = ["fog","smog","ari","ttr","lex_density"]

    def table_for(names):
        header = [["Metric","Original","Paraphrase","Δ","Target"]]
        rows = [row_for(n) for n in names]
        data = header + rows
        tbl = Table(data, colWidths=[180, 70, 85, 70, 90])
        tbl.setStyle(TableStyle([
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BACKGROUND',(0,0),(-1,0), colors.HexColor("#eef2ff")),
            ('GRID',(0,0),(-1,-1),0.25, colors.HexColor("#c7d2fe")),
            ('ALIGN',(1,1),(4,-1),'CENTER'),
        ]))
        return tbl

    story.append(Paragraph("Readability & quality (targets and changes)", H2))
    story.append(table_for(left_names)); story.append(Spacer(1, 6))
    story.append(table_for(right_names)); story.append(Spacer(1, 6))

    story.append(Paragraph("Targets & how to reach them", H2))
    story.append(Paragraph(TARGET_GUIDE.replace("**","").replace("—","-").replace("–","-").replace("  \n","<br/>"), Body))

    doc.build(story)
    return buf.getvalue(), None

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Paraphrasing Practice (Streamlit)", layout="wide")
st.title("Paraphrasing Practice – Streamlit Edition")
st.markdown(
    "Paste the original text on the left and your paraphrase on the right. "
    "Toggle **Ignore stopwords** to focus similarity on content words. "
    "Evaluate to see similarity, style difference, highlights, readability, targets, deltas, and download a PDF."
)
st.markdown(CSS, unsafe_allow_html=True)

# Student / module / lecturer meta
m1, m2, m3 = st.columns(3)
with m1: student_name = st.text_input("Student name", value="")
with m2: module = st.text_input("Module", value="")
with m3: lecturer = st.text_input("Lecturer", value="")

# Text inputs
colA, colB = st.columns(2)
with colA: orig_text = st.text_area("Original text", height=220, key="orig")
with colB: para_text = st.text_area("Your paraphrase", height=220, key="para")

# Options
opt1, opt2, _ = st.columns([2,3,3])
with opt1: view_df = st.checkbox("View metrics as interactive table", value=True)
with opt2: ignore_sw = st.checkbox("Ignore stopwords for similarity", value=True)

# Evaluate
if st.button("Evaluate", type="primary"):
    if not orig_text.strip() or not para_text.strip():
        st.warning("Please enter both the original text and your paraphrase.")
        st.stop()

    # Similarity tokens
    A_all, B_all = tokenize(orig_text), tokenize(para_text)
    A = filter_stop(A_all) if ignore_sw else A_all
    B = filter_stop(B_all) if ignore_sw else B_all

    # Similarities
    jacc = jaccard_similarity(A, B)
    dice = dice_similarity(A, B)
    cos  = cosine_similarity(A, B)

    c1, c2, c3 = st.columns(3)
    c1.metric("Jaccard (unique word overlap)", f"{round(jacc*100)}%")
    c2.metric("Sørensen–Dice (weighted overlap)", f"{round(dice*100)}%")
    c3.metric("Cosine (term‑frequency overlap)", f"{round(cos*100)}%")

    # Style difference (Burrows’s Delta)
    delta_val = burrows_delta(orig_text, para_text)
    delta_label, delta_icon = interpret_delta(delta_val)
    st.markdown(
        f"**Writing style difference (Burrows’s Δ):** {delta_icon} **{delta_val:.2f}** "
        f"<span class='tiny'>(lower = more similar style)</span> — {delta_label}",
        unsafe_allow_html=True
    )
    st.caption("This compares function‑word usage patterns. 🟢 close (<1.0), 🟡 moderate (1.0–2.0), 🔴 far (>2.0).")

    # Band feedback (based on Jaccard)
    if jacc < 0.20:
        st.success("Low lexical overlap — good paraphrasing! Check meaning is preserved.")
    elif jacc < 0.40:
        st.info("Moderate overlap — consider changing more wording and sentence structure.")
    else:
        st.warning("High overlap — rephrase further and adjust the structure to reduce reuse.")

    # Explanation
    st.markdown(explain_metrics(jacc, dice, cos))

    # Highlights
    st.subheader("Paraphrase with highlights")
    st.caption("Words in red also appear in the original; words in green are unique to your paraphrase.")
    highlighted_html = highlight_paraphrase(tokenize(orig_text), para_text)
    st.markdown(highlighted_html, unsafe_allow_html=True)

    # Readability & Quality
    st.subheader("Readability & quality signals")
    orig = readability_metrics(orig_text)
    para = readability_metrics(para_text)

    metrics_order = [
        "flesch","fk_grade","fog","smog","ari",
        "avg_sentence_len","pct_complex_words","ttr","lex_density"
    ]

    rows = []
    for name in metrics_order:
        o = float(orig[name]); p = float(para[name])
        diff, verdict = delta_eval(name, o, p)
        lo, hi, _ = TARGETS[name]
        rows.append({
            "Metric": DISPLAY_NAMES[name],
            "Original": round(o, 1),
            "Paraphrase": round(p, 1),
            "Δ (para−orig)": f"{diff:+.1f} {arrow(verdict)}",
            "Verdict": verdict,
            "Target": f"{lo:.0f}–{hi:.0f}",
            "In target?": "Yes" if (lo <= p <= hi) else "No",
        })

    df = pd.DataFrame(rows, columns=[
        "Metric","Original","Paraphrase","Δ (para−orig)","Verdict","Target","In target?"
    ])

    if view_df:
        st.dataframe(df, use_container_width=True)
    else:
        md = []
        md.append("| Metric | Original | Paraphrase | Δ | Verdict | Target | In target? |")
        md.append("|---|---:|---:|:--:|:--:|:--:|:--:|")
        for _, r in df.iterrows():
            md.append(
                f"| {r['Metric']} | {r['Original']:.1f} | {r['Paraphrase']:.1f} | "
                f"{r['Δ (para−orig)']} | {r['Verdict']} | {r['Target']} | {r['In target?']} |"
            )
        st.markdown("\n".join(md))
    st.caption("Legend: ▲ better, ▼ worse, 🟰 no change. Targets are heuristic bands for undergrad academic prose; always favour clarity and your discipline’s conventions.")

    # Target guidance (what/why/how)
    st.subheader("Targets: what they mean & how to achieve them")
    st.markdown(TARGET_GUIDE)

    # ---------------------------
    # Downloads
    # ---------------------------
    st.subheader("Download your attempt")

    # Plain text summary
    txt = StringIO()
    txt.write("Paraphrasing Attempt\n")
    if student_name: txt.write(f"Student: {student_name}\n")
    if module: txt.write(f"Module: {module}\n")
    if lecturer: txt.write(f"Lecturer: {lecturer}\n")
    txt.write(f"Ignore stopwords: {ignore_sw}\n")
    txt.write(f"Jaccard: {round(jacc*100)}%\nSørensen–Dice: {round(dice*100)}%\nCosine: {round(cos*100)}%\n")
    txt.write(f"Style difference (Burrows's Delta): {delta_val:.2f} ({delta_label})\n\n")
    txt.write("Original:\n" + orig_text + "\n\nParaphrase:\n" + para_text + "\n")
    st.download_button("Download as .txt", data=txt.getvalue().encode("utf-8"),
                       file_name="paraphrase_attempt.txt", mime="text/plain")

    # PDF report
    pdf_bytes, pdf_err = build_pdf_bytes(
        orig_text, para_text, jacc, dice, cos, delta_val, orig, para, tokenize(orig_text),
        student_name, module, lecturer
    )
    if pdf_bytes:
        st.download_button("Download PDF report (.pdf)", data=pdf_bytes,
                           file_name="paraphrase_report.pdf", mime="application/pdf")
    else:
        st.error("PDF engine not available. Install ReportLab first:\n\n`pip install reportlab`")

else:
    st.caption("Enter your texts, add optional student/module/lecturer, and click **Evaluate**.")

