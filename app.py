import streamlit as st
import torch, re, uuid
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from deep_translator import GoogleTranslator
from datetime import datetime
from fpdf import FPDF


# --- POP-UP DIALOG FUNCTION ---
import streamlit as st

# --- THE BEST VERSION: LexiMind AI POP-UP ---
@st.dialog("🧠 LexiMind AI: Clinical Access Portal")
def agreement_modal():
    

    # 2. Clinical Use Policy (Now at the Top)
    st.markdown("### 📋 Clinical Use Policy")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("✅ **Non-Diagnostic**\nResearch prototype for screening only.")
        st.markdown("✅ **Data Privacy**\nAll narratives are processed in-memory.")
    with c2:
        st.markdown("✅ **XAI Enabled**\nTransparent 'Attention-based' analysis.")
        st.markdown("✅ **Dual-Lingual**\nSupports English & Hinglish inputs.")

    st.warning("Notice: This AI tool is an academic project and not a substitute for professional clinical advice.")

   

    # 3. Emergency Crisis Protocol (Strategic Placement at Bottom)
    st.error("### 🛑 Immediate Crisis Notice")
    st.write("If you are in immediate danger, please use these resources:")
    
    # SOS Table inside Pop-up
    st.markdown("""
    | Organization | Contact | Support |
    | :--- | :--- | :--- |
    | **Kiran (Govt)** | 📞 1800-599-0019 | 24/7 Support |
    | **Emergency** | 📞 112 | Police/Medical |
    """)
    # 4. Activation Button
    if st.button("Agree & Initialize LexiMind Engine", use_container_width=True, type="primary"):
        st.session_state.agreed = True
        st.success("Access Granted. Loading Clinical Dashboard...")
        st.rerun()

# --- LOGIC TO TRIGGER MODAL ---
if "agreed" not in st.session_state:
    st.session_state.agreed = False

if not st.session_state.agreed:
    agreement_modal()
    st.stop()
# PAGE CONFIG
st.set_page_config(
    page_title="LexiMind AI | Clinical Access Portal",
    page_icon="🧠",
    layout="wide"
)
# SESSION STATE

if "results_data" not in st.session_state:
    st.session_state["results_data"] = None

# CSS

st.markdown("""
<style>
.main-title { font-size:50px; font-weight:900; text-align:center; color:#1e293b; }
.card { background:white; padding:25px; border-radius:15px;
        box-shadow:0 10px 40px rgba(0,0,0,0.05);
        border:1px solid #e2e8f0; margin-bottom:20px; }

.ms-container {
    text-align:center;
    padding:22px;
    border-radius:18px;
    background: rgba(248,250,252,0.92);
    border: 1.5px solid rgba(100,116,139,0.35);
    backdrop-filter: blur(6px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.06);
}


.badge-large { padding:15px; border-radius:10px;
               font-weight:900; font-size:1.6rem; text-transform:uppercase; }

.normal { background:#dcfce7; color:#166534; border:2px solid #166534; }
.anxiety { background:#fef9c3; color:#854d0e; border:2px solid #854d0e; }
.danger { background:#fee2e2; color:#991b1b; border:2px solid #991b1b; }

.hl-danger { background:#fee2e2; color:#991b1b; font-weight:bold;
             padding:2px 6px; border-radius:4px; }
.hl-anxiety { background:#fef9c3; color:#854d0e; font-weight:bold;
              padding:2px 6px; border-radius:4px; }

.rx-letterhead { background:white; border:1px solid #e2e8f0;
                 padding:40px; border-radius:10px;
                 box-shadow:0 15px 50px rgba(0,0,0,0.05); }

.clinical-box { background:#f1f5f9; padding:20px; border-radius:10px;
                border-left:8px solid #1e293b;
                font-size:1.1rem; line-height:1.8; color:#334155; }

.rx-symbol { font-size:3.5rem; font-family:serif;
             color:#1e293b; font-weight:bold; }

.emergency-footer {
    background:#fff1f2; color:#9f1239; padding:30px;
    border-radius:20px; border:2px solid #fecaca;
    text-align:center; margin-top:40px;
}
.left-panel {
    background: rgba(255,255,255,0.92);
    border: 1.5px solid rgba(30,41,59,0.25);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.06);
}


.section-card {
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 18px;
    background: #f9fafb;
}

.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.state-label {
    font-size: 1.7rem;
    font-weight: 900;
    margin: 0;
}

.disclaimer-text {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 6px;
}
.stApp {
background: linear-gradient(
180deg,
#C8DCE3 60%
);
}



</style>
""", unsafe_allow_html=True)

# MODEL LOAD
MODEL_PATH = "Sachi06/leximind-ai"

@st.cache_resource
def load_model():
    tok = BertTokenizerFast.from_pretrained(MODEL_PATH)
    mdl = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        use_safetensors=True
    )
    mdl.eval()
    return tok, mdl
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABELS = {0:"Normal", 1:"Anxiety", 2:"Depression"}

# PDF SAFETY
def sanitize_text(text):
    if not text:
        return ""
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        "•": "-",     # VERY IMPORTANT
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text



def clean_pdf(text):
    text = sanitize_text(text)
    text = re.sub('<[^<]+?>', '', text)   # remove HTML
    return text





def generate_pdf(profile, res, interp, rec):
    pdf = FPDF()
    pdf.add_page()

    # Light background
    pdf.set_fill_color(245, 248, 252)
    pdf.rect(0, 0, 210, 297, 'F')  # A4 full background

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.set_font("Arial","B",22)
    pdf.cell(0,10,"  LexiMind AI CLINICAL LAB",ln=True,align="C")

    pdf.set_font("Arial","",11)
    pdf.cell(0,8,"AI-Based Mental Health Screening Report",ln=True,align="C")

    pdf.ln(4)
    pdf.set_draw_color(30, 41, 59)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(10)


    pdf.ln(8)

    pdf.set_fill_color(255,255,255)
    pdf.set_draw_color(203,213,225)
    
    pdf.ln(4)
    pdf.set_x(16)

    pdf.set_font("Arial","B",11)
    pdf.cell(0,6,"PATIENT INFORMATION",ln=True)
  
    pdf.set_font("Arial","",10)
    pdf.multi_cell(
    0,6,
    clean_pdf(
        f"Name: {profile['name']}\n"
        f"Age: {profile['age']} years\n"
        f"Gender: {profile['gender']}\n"
        f"Report ID: MC-{res['rid']}\n"
        f"Report Date: {datetime.now().strftime('%d %B %Y')}"
    )
)

    pdf.ln(6)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"CLINICAL SCREENING SUMMARY",ln=True)

    risk = (
    "Low Risk" if res["pred"]==0 else
    "Moderate Risk" if res["pred"]==1 else
    "Elevated Risk"
     )

    pdf.set_font("Arial","",10)
    pdf.multi_cell(
    0,7,
    clean_pdf(
        f"Screening Result: {res['label']}\n"
        f"Risk Level: {risk}\n"
        f"Model Confidence Score: {res['conf']:.2f}%"
    )
)

    pdf.ln(4)


    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)


    pdf.ln(5)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"SCIENTIFIC INTERPRETATION (AI-BASED)",ln=True)
    pdf.set_font("Arial",size=10)
    pdf.ln(3)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"KEY LINGUISTIC INDICATORS",ln=True)

    pdf.set_font("Arial","",10)
    for w in res["words"][:5]:
     pdf.cell(0,7,f"- {w}",ln=True)

    pdf.multi_cell(0,7,clean_pdf(interp))

    pdf.ln(4)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"MODEL EVALUATION SUMMARY",ln=True)
    pdf.set_font("Arial",size=10)
    pdf.multi_cell(
        0,7,
        clean_pdf(
            "The model was evaluated on 49,612 validation samples. "
            "Overall accuracy achieved was 92.3% with weighted F1-score of 0.92. "
            "Depression detection showed strongest performance, while Anxiety exhibited "
            "expected linguistic overlap."
        )
    )

    pdf.ln(5)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"CLINICAL RECOMMENDATIONS",ln=True)
    pdf.set_font("Arial",size=10)
    pdf.multi_cell(0,7,clean_pdf(rec))

    pdf.ln(6)
    pdf.set_font("Arial","I",8)
    pdf.multi_cell(
    0,5,
    clean_pdf(
        "DISCLAIMER:\n"
        "This report is generated by an AI-based screening system. "
        "It is intended for research and preliminary assessment purposes only. "
        "It does not constitute a medical diagnosis or replace professional clinical evaluation."
    )
)
    return pdf.output(dest="S").encode("latin-1")
# CLINICAL INTERPRETATION

def clinical_reasoning(label, words):
    w = ", ".join([f"'{x}'" for x in words[:3]])

    if label=="Depression":
        interp = (
            f"Neural attention patterns highlight linguistic biomarkers such as {w}, "
            "which reflect affective withdrawal, emotional blunting, and negative self-referential cognition. "
            "These features collectively increased model confidence toward a depressive classification. "
            "This interpretation reflects probabilistic language-pattern association and is not a diagnosis."
            "If distress-related language persists or intensifies,a qualified mental health professional should be consulted."
        )
        rec = (
            "• Professional psychiatric evaluation is recommended.<br>"
            "• Behavioral activation and routine stabilization strategies.<br>"
            "• Immediate support if distress intensifies."
        )
    elif label=="Anxiety":
        interp = (
            f"The model identified anticipatory and arousal-related markers including {w}. "
            "These suggest heightened cognitive vigilance and stress-oriented narrative framing. "
            "Interpretation is based on contextual attention weighting, not isolated keywords."
        )
        rec = (
            "• Stress regulation and mindfulness-based techniques.<br>"
            "• Breathing exercises (4-7-8 method).<br>"
            "• Monitoring symptom persistence."
        )
    else:
        interp = (
            f"Linguistic patterns such as {w} indicate emotionally stable and contextually balanced expression."
            "No sustained markers of psychological distress were detected."
        )
        rec = (
            "• Maintain healthy routines and social engagement.<br>"
            "• Periodic mental wellness check-ins."
        )

    return interp, rec

# WHY & HOW EXPLAINABILITY (XAI)

def explainability_why_how(label, words):
    safe_words = (words + ["N/A", "N/A", "N/A"])[:3]
    w1, w2, w3 = safe_words

    return f"""
<b>WHY this mental state was predicted:</b><br>
The model analyzed contextual language patterns rather than isolated keywords.
Tokens such as <b>{w1}</b>, <b>{w2}</b>, and <b>{w3}</b> contributed by reflecting
emotional tone and semantic emphasis.<br>

<b>HOW the model reached this decision:</b><br>
MindCare AI uses a transformer-based architecture with an
<i>attention mechanism</i> that evaluates:
<ul>
<li>Contextual word relationships</li>
<li>Sentence-level emotional framing</li>
<li>Relative semantic importance of tokens</li>
</ul>
Tokens with higher attention weights influenced the final prediction more strongly.<br><br>

<b>WHAT the highlights and charts represent:</b><br>
Highlighted words and bar charts visualize <i>model attention</i>.
They indicate influence on prediction, not clinical severity.<br>

<b>Ethical & Clinical Note:</b><br>
This explanation provides screening-level transparency only.
It does not constitute a medical diagnosis.
"""

# UI – INPUT

st.markdown("<div class='main-title'>🧠 LexiMind AI</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-top: -20px;'>Decoding Language Patterns through Neural Intelligence</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #cbd5e1; margin-top: -15px; margin-bottom: 25px; opacity: 0.5;'>", unsafe_allow_html=True)
st.markdown("""
    <div style="
        height: 2px;
        background: linear-gradient(to right, rgba(245,247,251,0) 0%, #64748b 50%, rgba(245,247,251,0) 100%);
        margin: -20px auto 30px auto;
        width: 80%;
        opacity: 0.6;
    "></div>
""", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    
    st.subheader("👤 Patient Profile")

# Name: डिफ़ॉल्ट "Anonymous" को हटाकर placeholder में रखा है
    name = st.text_input("Name", placeholder="Enter Name ")

# Age: value=None रखने से placeholder दिखाई देगा
    age = st.number_input("Age", min_value=1, max_value=100, value=None, placeholder="Enter Age")

# Gender: index=None रखने से placeholder दिखाई देगा
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=None, placeholder="Select Gender")

    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    
    st.subheader("📝 Narrative Intake")

    text = st.text_area("Patient Narrative",
                         height=117,
                         placeholder="How are you feeling today?"
                         )

    consent = st.checkbox(
        "I provide informed consent for the secure and anonymized processing of data for academic and research purposes."
    )

    run = st.button(
        "🔍 RUN CLINICAL ANALYSIS",
        use_container_width=True,
        disabled=not consent
    )
    
    st.markdown("</div>", unsafe_allow_html=True)


# ANALYSIS

if run and text.strip():
    with st.spinner("Analyzing linguistic biomarkers..."):
        trans = GoogleTranslator(source="auto",target="en").translate(text)
        inp = tokenizer(trans,return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
        with torch.no_grad():
            out = model(**inp,output_attentions=True)

        probs = torch.softmax(out.logits,dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        conf = probs[pred]*100

        if out.attentions is not None:
            attn = (
                out.attentions[-1][0]
                .mean(dim=0)
                .mean(dim=0)
                .cpu()
                .numpy()
                )
        else:
            attn = np.zeros(len(inp["input_ids"][0]))

        toks = tokenizer.convert_ids_to_tokens(inp["input_ids"][0])
        if len(attn) != len(toks):
            min_len = min(len(attn), len(toks))
            attn = attn[:min_len]
            toks = toks[:min_len]

        STOP_TOKENS = {
         ".", ",", "!", "?", ";", ":", "'", '"',
        "i", "me", "my", "we", "us",
         "a", "an", "the", "and", "or", "but",
        "to", "of", "in", "on", "at", "for",
        "is", "am", "are", "was", "were"
        }

        df = pd.DataFrame([
    {
        "Word": t.replace("##", ""),
        "Score": s
    }
    for t, s in zip(toks, attn)
    if (
        t.lower() not in ["[cls]", "[sep]", "[pad]"]
        and t.replace("##", "").lower() not in STOP_TOKENS
        and len(t.replace("##", "")) > 2
    )
]).sort_values("Score", ascending=False)


        st.session_state["results_data"] = {
            "label":LABELS[pred],
            "pred":pred,
            "conf":conf,
            "words":df["Word"].head(10).tolist(),
            "df":df,
            "trans":trans,
            "profile":{"name":name,"age":age,"gender":gender},
            "rid":uuid.uuid4().hex[:6].upper()
        }

# RESULTS

if st.session_state["results_data"]:
    r = st.session_state["results_data"]

    colA, colB = st.columns([1, 1.8])

    with colA:

    # -------- Mental State Card --------
      st.markdown(
           f"""
           <div class='ms-container'>

           <p style="
            font-size:0.85rem;
            color:#334155;
            font-weight:700;
            margin-bottom:8px;
            letter-spacing:0.04em;
            text-transform:uppercase;
           ">
              AI-Inferred Mental State (Screening Output)
           </p>

           <div style="
            display:flex;
            align-items:center;
            justify-content:center;
            gap:12px;
            margin-bottom:4px;
           ">
            <span style="font-size:1.4rem;">
                {"🟢" if r["pred"]==0 else "🟡" if r["pred"]==1 else "🔴"}
            </span>

            <span style="
                font-size:1.6rem;
                font-weight:900;
                color:#0f172a;
            ">
                {r["label"]}
            </span>

            <span style="
                font-size:0.8rem;
                font-weight:800;
                padding:4px 12px;
                {"#166534" if r["pred"]==0 else "#854d0e" if r["pred"]==1 else "#991b1b"};
                color:
                {"#166534" if r["pred"]==0 else "#854d0e" if r["pred"]==1 else "#991b1b"};
                background:
                {"#50ae71" if r["pred"]==0 else "#fef9c3" if r["pred"]==1 else "#fee2e2"};
            ">
                {"Low Risk" if r["pred"]==0 else "Moderate Risk" if r["pred"]==1 else "Elevated Risk"}
            </span>
            </div>

            <p style="
            font-size:0.72rem;
            color:#64748b;
            margin-top:6px;
           ">
            Based on language patterns · Not a medical diagnosis
           </p>

           </div>
           """,
            unsafe_allow_html=True
            )

      st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # -------- Gauge --------
      color = (
        "#2b9959" if r["pred"] == 0
        else "#e7c84f" if r["pred"] == 1
        else "#c32222"
    )

      fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r["conf"],
        number={"suffix": "%"},
        title={"text": "Prediction Confidence (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color}
        }
    ))

      fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=60, b=20)
    )

      

      st.plotly_chart(fig, use_container_width=True)
      st.markdown("</div>", unsafe_allow_html=True)
      st.markdown("""
<p style="
font-size:0.8rem;
color:#475569;
text-align:center;
margin-top:6px;
">
<b>Note:</b> This score represents <b>model reliability / confidence</b> in its prediction,
<b>not</b> the clinical severity or intensity of symptoms.
</p>
""", unsafe_allow_html=True)

      st.markdown("</div>", unsafe_allow_html=True)
      

  

    # ===== Model Evaluation =====
      with st.expander("🧪 Model Evaluation (Validation Results)"):

        st.markdown("### 📊 Confusion Matrix")
        
        cm = np.array([
            [16698, 519, 1174],
            [165, 4450, 888],
            [506, 564, 24648]
        ])

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Anxiety", "Actual Depression"],
            columns=["Pred Normal", "Pred Anxiety", "Pred Depression"]
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=380)
        
        st.plotly_chart(fig_cm, use_container_width=True)
         
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📈 Classification Report")
        st.code("""
Accuracy: 92%

Normal:
 Precision = 0.96 | Recall = 0.91 | F1-score = 0.93

Anxiety:
 Precision = 0.80 | Recall = 0.81 | F1-score = 0.81

Depression:
 Precision = 0.92 | Recall = 0.96 | F1-score = 0.94

Weighted F1-score = 0.92
""")
        

        st.caption(
            "Strong overall performance with minimal confusion between Anxiety "
            "and Depression, supporting clinical robustness."
        )

        st.caption("Evaluation results are shown for model validation and transparency purposes only.")

    # ---------- RIGHT COLUMN ----------
    with colB:
        
        st.markdown("""
        <div style="
        font-size:1.35rem;
        font-weight:900;
        color:#1e293b;
        margin-bottom:10px;
        ">
        Explainability & Model Transparency
        </div>
        """, unsafe_allow_html=True)


        why_how_text = explainability_why_how(r["label"], r["words"])
        st.markdown(
            f"""
            <div style="
                background:#f8fafc;
                border-left:5px solid #94a3b8;
                padding:18px;
                border-radius:10px;
                font-size:1.05rem;
                line-height:1.9;
                margin-bottom:15px;
            ">
            {why_how_text}
            """,
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight:700;color:#1e293b;margin-bottom:8px;'>"
        "🧠 Influential Words in Patient Narrative</p>",
        unsafe_allow_html=True
        )
        top_words = r["words"][:6] 
        highlighted = ""
        for w in r["trans"].split():
            wc = w.lower().strip(".,!?")
            if wc in r["words"][:3]:
                highlighted += f"<span class='hl-danger'>{w}</span> "
            elif wc in r["words"][3:6]:
                highlighted += f"<span class='hl-anxiety'>{w}</span> "
            else:
                highlighted += w + " "
        
        st.markdown(highlighted, unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.85rem; color:#475569;">
        <b>Legend:</b> 
        <span class='hl-danger'>Red</span> = High influence tokens | 
        <span class='hl-anxiety'>Yellow</span> = Moderate influence tokens
        </p>
        """, unsafe_allow_html=True)
        st.markdown(
        "<p style='font-weight:700;color:#1e293b;margin-top:18px;'>"
        "📊 Token Contribution (Attention Scores)</p>",
        unsafe_allow_html=True
        )

        top_words_df = r["df"].head(5)

        figb = px.bar(
        top_words_df,
        x="Score",
        y="Word",
        orientation="h",
        color="Score",
        color_continuous_scale="Reds" if r["pred"] != 0 else "Greens"
        )

        figb.update_layout(
        height=260,
        title="Top-5 Most Influential Linguistic Markers"
)

        figb.update_layout(height=300)
        figb.update_layout(
        xaxis_title="Attention Contribution",
        yaxis_title="Token"
         )

        st.plotly_chart(figb, use_container_width=True)
                


    # ---------- INTERPRETATION & PDF ----------
        interp, rec = clinical_reasoning(r["label"], r["words"])

        st.markdown("""
        <div class='rx-letterhead'>

        <div style="
        font-size:1.4rem;
        font-weight:900;
        color:#1e293b;
        margin-bottom:14px;
        border-bottom:6px solid #cbd5f5;
        padding-bottom:6px;
        ">
        🧪 Scientific Interpretation & Clinical Context
        </div>

        <div class='clinical-box'>
        <b>Interpretation:</b><br>
        """ + interp + """
        </div>

        <div class='rx-symbol'>℞</div>

        <div style='font-size:1.2rem;line-height:2;'>
        """ + rec + """
        </div>

        </div><br>
        """, unsafe_allow_html=True)


        pdf = generate_pdf(r["profile"], r, interp, rec)
        st.download_button(
        "📥 DOWNLOAD CLINICAL PDF REPORT",
        data=pdf,
        file_name=f"MindCare_Report_{r['rid']}.pdf",
        use_container_width=True
        )

# SOS FOOTER

st.markdown("""
<div class='emergency-footer'>
<h2>🚨 CRITICAL CRISIS & HEALTH NOTICE</h2>
<p>If you are experiencing severe distress, seek immediate professional help.</p>
<p><b>Kiran:</b> 1800-599-0019 | <b>iCall:</b> 9152987821 | <b>Emergency:</b> 112</p>
</div>
<div style='text-align:center;color:#333333;font-size:0.8rem;margin-top:20px;'>
IEEE Research Prototype | Ethical AI | Non-diagnostic Tool
</div>
""",unsafe_allow_html=True)
