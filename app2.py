import json
import base64
import time
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from groq import Groq

# ==================================================================
#                  CONFIGURACIÓN DE PÁGINA
# ==================================================================
st.set_page_config(page_title="AgroDetect · UNAH-CURC", page_icon="🌱", layout="centered")


def html(s: str):
    """Aplana el HTML: Streamlit convierte en bloque de código cualquier línea
    que empiece con 4+ espacios. Quitando toda la indentación se evita eso."""
    plano = "".join(linea.strip() for linea in s.strip().splitlines())
    st.markdown(plano, unsafe_allow_html=True)


# ==================================================================
#                  ESTADO
# ==================================================================
if "tema" not in st.session_state:
    st.session_state.tema = "oscuro"
if "historial" not in st.session_state:
    st.session_state.historial = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

ES_OSCURO = st.session_state.tema == "oscuro"

# ==================================================================
#                  MODELO Y METADATA
# ==================================================================
@st.cache_resource
def cargar_modelo():
    model = tf.keras.models.load_model("modelo/agrodetect_mobilenetv2.keras")
    with open("modelo/config_app.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


model, meta = cargar_modelo()
CLASES = meta["clases"]
IMG_SIZE = tuple(meta["img_size"])
UMBRAL_ALTO = meta["umbral_decision"]
UMBRAL_DIFERENCIAL = 0.15
ARQUITECTURA = meta["arquitectura_seleccionada"]

if ARQUITECTURA == "MobileNetV2":
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
elif ARQUITECTURA == "EfficientNetB0":
    from tensorflow.keras.applications.efficientnet import preprocess_input
else:
    from tensorflow.keras.applications.resnet50 import preprocess_input

NOMBRES_DISPLAY = {
    "sana": "Hoja sana", "roya": "Roya", "cercospora": "Cercospora",
    "phoma": "Phoma", "arana_roja": "Araña roja", "minador": "Minador de la hoja",
}
ICONOS_CLASE = {
    "sana": "🍃", "roya": "🟠", "cercospora": "⚫",
    "phoma": "🟤", "arana_roja": "🕷️", "minador": "🐛",
}
TINTA_OSCURO = {
    "sana": "#5CE1A0", "roya": "#FF9F45", "cercospora": "#C9B8FF",
    "phoma": "#E8A87C", "arana_roja": "#FF7A85", "minador": "#D6E85C",
}
TINTA_CLARO = {
    "sana": "#0F8A5F", "roya": "#C4621A", "cercospora": "#5A44B8",
    "phoma": "#9B5A2B", "arana_roja": "#C2313F", "minador": "#6E7B12",
}
TINTA_CLASE = TINTA_OSCURO if ES_OSCURO else TINTA_CLARO

# ==================================================================
#                  GROQ
# ==================================================================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def obtener_recomendacion_unica(clase_detectada):
    nombre_legible = NOMBRES_DISPLAY.get(clase_detectada, clase_detectada)
    prompt = f"""Eres un asistente agronómico para caficultores de Honduras.
Se detectó la siguiente condición en una hoja de café: {nombre_legible}.
Da una recomendación práctica y breve (máximo 4 puntos) sobre manejo agronómico,
en español sencillo, orientada a un pequeño productor."""
    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return respuesta.choices[0].message.content


def obtener_recomendacion_diferencial(candidatas):
    lista_texto = ", ".join(candidatas)
    prompt = f"""Eres un asistente agronómico para caficultores de Honduras.
El sistema no logró determinar con certeza una única condición, pero identificó
estas posibles condiciones en una hoja de café (de mayor a menor probabilidad):
{lista_texto}.
Da una recomendación breve (máximo 5 puntos) en español sencillo que:
1) mencione brevemente cómo diferenciar estas condiciones a simple vista,
2) dé una recomendación general de manejo agronómico preventivo mientras se confirma,
3) recomiende consultar a un técnico del IHCAFE para confirmar el diagnóstico exacto."""
    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return respuesta.choices[0].message.content


def imagen_a_base64(imagen_pil):
    buffered = BytesIO()
    imagen_pil.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def obtener_segunda_opinion_visual(imagen_pil, nombres_candidatas):
    img_b64 = imagen_a_base64(imagen_pil)
    lista_texto = ", ".join(nombres_candidatas)
    prompt_texto = f"""Eres un asistente agrícola apoyando el diagnóstico de hojas de café.
Nuestro modelo de clasificación no logró determinar con certeza una única condición,
pero considera estas posibles opciones: {lista_texto}.
Observa la imagen y responde ÚNICAMENTE con un JSON, sin texto adicional, sin backticks,
con este formato exacto:
{{"eleccion": "<una de las opciones tal cual está escrita>", "justificacion": "<máximo 2 oraciones>"}}"""
    respuesta = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt_texto},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
    )
    texto = respuesta.choices[0].message.content.strip()
    return json.loads(texto.replace("```json", "").replace("```", "").strip())


def diagnosticar(imagen_pil):
    img = imagen_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")
    x = preprocess_input(np.expand_dims(x, 0))
    probs = model.predict(x, verbose=0)[0]

    idx_ordenados = np.argsort(probs)[::-1]
    tabla_probs = {CLASES[i]: float(probs[i]) for i in idx_ordenados}
    idx_top = idx_ordenados[0]
    confianza_top = float(probs[idx_top])

    if confianza_top >= UMBRAL_ALTO:
        return "unico", [(CLASES[idx_top], confianza_top)], tabla_probs

    candidatas, acumulado = [], 0.0
    for i in idx_ordenados[:3]:
        p = float(probs[i])
        if p < UMBRAL_DIFERENCIAL:
            break
        candidatas.append((CLASES[i], p))
        acumulado += p

    if len(candidatas) >= 2 and acumulado >= 0.60:
        return "diferencial", candidatas, tabla_probs

    return "incierto", [(CLASES[idx_top], confianza_top)], tabla_probs


# ==================================================================
#                  TEMA
# ==================================================================
OSCURO = """
--bg:#0B0E0C; --bg-2:#0F1411;
--glow-1:rgba(92,225,160,0.14); --glow-2:rgba(255,159,69,0.10);
--panel:rgba(255,255,255,0.035); --panel-2:rgba(255,255,255,0.06);
--linea:rgba(255,255,255,0.09); --linea-fuerte:rgba(255,255,255,0.16);
--txt:#F2F5F1; --txt-2:#A9B3A9; --txt-3:#6B776C;
--acento:#5CE1A0; --acento-2:#FF9F45; --track:rgba(255,255,255,0.07);
--sombra:0 24px 60px -30px rgba(0,0,0,0.9);
"""

CLARO = """
--bg:#F4F6F2; --bg-2:#FFFFFF;
--glow-1:rgba(15,138,95,0.10); --glow-2:rgba(196,98,26,0.08);
--panel:rgba(255,255,255,0.9); --panel-2:rgba(15,25,18,0.035);
--linea:rgba(15,25,18,0.10); --linea-fuerte:rgba(15,25,18,0.18);
--txt:#101711; --txt-2:#4C5850; --txt-3:#84908A;
--acento:#0F8A5F; --acento-2:#C4621A; --track:rgba(15,25,18,0.08);
--sombra:0 20px 50px -32px rgba(16,23,17,0.45);
"""

html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{ {OSCURO if ES_OSCURO else CLARO} }}
html, body, [class*="css"] {{ font-family:'Space Grotesk', sans-serif; }}
.stApp {{
background: radial-gradient(900px 420px at 12% -8%, var(--glow-1), transparent 60%),
radial-gradient(700px 380px at 92% 4%, var(--glow-2), transparent 62%),
linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
background-attachment: fixed; color: var(--txt);
}}
.block-container {{ padding-top:1rem; padding-bottom:3rem; max-width:780px; }}
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] span {{ color: var(--txt-2); }}
[data-testid="stHeader"] {{ background: transparent; height: 0rem; min-height: 0rem; pointer-events: none; }}
code, pre {{ display:none !important; }}
.mono {{ font-family:'JetBrains Mono', monospace; }}

.ad-nav {{ display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .1rem 0; }}
.ad-dot {{ width:9px; height:9px; border-radius:50%; background:var(--acento);
box-shadow:0 0 0 4px color-mix(in srgb, var(--acento) 18%, transparent); flex-shrink:0; }}
.ad-nav-t {{ font-family:'JetBrains Mono',monospace; font-size:.63rem; letter-spacing:2.4px;
text-transform:uppercase; color:var(--txt-3); }}

.ad-hero {{ padding:2.4rem 0 1.6rem 0; border-bottom:1px solid var(--linea); }}
.ad-tag {{ font-family:'JetBrains Mono',monospace; font-size:.6rem; letter-spacing:2.6px;
text-transform:uppercase; color:var(--acento); }}
.ad-h1 {{ font-family:'Instrument Serif', serif; font-weight:400; font-size:4.4rem; line-height:.94;
letter-spacing:-2px; color:var(--txt); margin:.55rem 0 0 0; }}
.ad-h1 i {{ font-style:italic; color:var(--acento); }}
.ad-lede {{ font-size:1rem; color:var(--txt-2); max-width:46ch; margin-top:.85rem; line-height:1.5; }}
.ad-meta {{ display:flex; flex-wrap:wrap; gap:1.4rem; margin-top:1.5rem;
font-family:'JetBrains Mono',monospace; font-size:.62rem; letter-spacing:1.6px;
text-transform:uppercase; color:var(--txt-3); }}
.ad-meta b {{ display:block; color:var(--txt); font-weight:500; letter-spacing:.6px;
font-family:'Space Grotesk',sans-serif; font-size:.8rem; text-transform:none; margin-top:.2rem; }}

.ad-flow {{ display:flex; align-items:center; gap:.55rem; flex-wrap:wrap; margin:1.4rem 0 1.2rem 0; }}
.ad-flow-i {{ display:flex; align-items:center; gap:.45rem; font-family:'JetBrains Mono',monospace;
font-size:.62rem; letter-spacing:1.6px; text-transform:uppercase; color:var(--txt-3); }}
.ad-flow-n {{ width:20px; height:20px; border-radius:6px; background:var(--panel-2);
border:1px solid var(--linea); display:flex; align-items:center; justify-content:center;
font-size:.58rem; color:var(--txt-2); }}
.ad-flow-s {{ flex:1; height:1px; background:var(--linea); min-width:14px; }}

.ad-tip {{ display:flex; gap:.8rem; padding:.9rem 1rem; border-radius:14px;
background:var(--panel); border:1px solid var(--linea); font-size:.87rem; color:var(--txt-2);
backdrop-filter: blur(10px); margin-bottom:1.3rem; }}
.ad-tip b {{ color:var(--txt); }}

[data-testid="stFileUploader"] {{ background:var(--panel); border:1px dashed var(--linea-fuerte);
border-radius:16px; padding:.45rem .8rem; backdrop-filter: blur(10px); }}
[data-testid="stFileUploader"] section {{ background:transparent; }}
[data-testid="stFileUploader"] * {{ color:var(--txt-2) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size:.82rem !important; }}

div[data-testid="stButton"] button {{ background:var(--panel-2) !important; color:var(--txt) !important;
border:1px solid var(--linea) !important; border-radius:12px !important; font-size:.8rem !important;
font-weight:500 !important; padding:.5rem .85rem !important; box-shadow:none !important;
transition: all .18s ease; }}
div[data-testid="stButton"] button:hover {{ border-color:var(--linea-fuerte) !important; transform:translateY(-1px); }}
div[data-testid="stButton"] button[kind="primary"] {{ background:var(--acento) !important;
color:var(--bg) !important; border:none !important; font-weight:600 !important; }}
div[data-testid="stButton"] button:disabled {{ opacity:.35; transform:none; }}

.ad-run {{ display:flex; align-items:center; gap:.6rem; margin:1.8rem 0 .9rem 0; }}
.ad-run img {{ width:34px; height:34px; border-radius:10px; object-fit:cover; border:1px solid var(--linea); }}
.ad-run-t {{ font-family:'JetBrains Mono',monospace; font-size:.6rem; letter-spacing:2px;
text-transform:uppercase; color:var(--txt-3); }}
.ad-run-l {{ flex:1; height:1px; background:var(--linea); }}

.ad-panel {{ background:var(--panel); border:1px solid var(--linea); border-radius:20px;
padding:1.35rem 1.4rem; margin:.8rem 0; backdrop-filter: blur(12px); box-shadow:var(--sombra); }}
.ad-lbl {{ font-family:'JetBrains Mono',monospace; font-size:.6rem; letter-spacing:2.2px;
text-transform:uppercase; color:var(--txt-3); margin-bottom:1rem; }}

.ad-row {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:.35rem .8rem;
align-items:baseline; padding:.55rem 0; border-top:1px solid var(--linea); }}
.ad-row:first-of-type {{ border-top:none; }}
.ad-row-n {{ font-size:.88rem; color:var(--txt); min-width:0; }}
.ad-row-p {{ font-family:'JetBrains Mono',monospace; font-size:.8rem; color:var(--txt); }}
.ad-track {{ grid-column:1/-1; height:3px; border-radius:99px; background:var(--track); overflow:hidden; }}
.ad-fill {{ height:100%; border-radius:99px; animation: crecer .7s cubic-bezier(.2,.9,.3,1) both; }}
@keyframes crecer {{ from {{ transform:scaleX(0); transform-origin:left; }} to {{ transform:scaleX(1); }} }}

.ad-verdict {{ position:relative; overflow:hidden; display:grid;
grid-template-columns:auto minmax(0,1fr); gap:1.1rem; align-items:center;
border:1px solid var(--linea); border-radius:22px; padding:1.4rem 1.5rem; margin:1rem 0;
background: linear-gradient(135deg, color-mix(in srgb, var(--ink) 14%, transparent), var(--panel) 65%);
backdrop-filter: blur(12px); box-shadow:var(--sombra);
animation: entrar .5s cubic-bezier(.2,.9,.3,1) both; }}
@keyframes entrar {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:none; }} }}
.ad-gauge {{ width:82px; height:82px; border-radius:50%; flex-shrink:0; display:grid; place-items:center;
background: conic-gradient(var(--ink) calc(var(--pct) * 1%), var(--track) 0); }}
.ad-gauge-in {{ width:66px; height:66px; border-radius:50%; background:var(--bg-2); display:grid;
place-items:center; font-size:1.5rem; }}
.ad-verdict-n {{ font-family:'Instrument Serif',serif; font-size:1.85rem; line-height:1.05;
color:var(--ink); letter-spacing:-.5px; }}
.ad-verdict-m {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:1.8px;
text-transform:uppercase; color:var(--txt-3); margin-top:.4rem; }}
.ad-pill {{ display:inline-block; margin-top:.55rem; font-family:'JetBrains Mono',monospace;
font-size:.58rem; letter-spacing:1.6px; text-transform:uppercase; padding:.26rem .6rem;
border-radius:99px; color:var(--ink); border:1px solid color-mix(in srgb, var(--ink) 40%, transparent);
background: color-mix(in srgb, var(--ink) 12%, transparent); }}

.ad-note-t {{ font-size:.95rem; font-weight:600; color:var(--txt); margin-bottom:.55rem;
display:flex; align-items:center; gap:.45rem; }}
.ad-note-b {{ font-size:.9rem; color:var(--txt-2); line-height:1.65; white-space:pre-wrap; }}
.ad-cap {{ font-family:'JetBrains Mono',monospace; font-size:.62rem; color:var(--txt-3);
margin-top:.8rem; line-height:1.5; }}
.ad-agree {{ border-left:2px solid var(--acento); padding-left:.75rem; margin:.2rem 0 .7rem 0;
font-size:.88rem; color:var(--txt-2); }}
.ad-dis {{ border-left:2px solid var(--acento-2); padding-left:.75rem; margin:.2rem 0 .7rem 0;
font-size:.88rem; color:var(--txt-2); }}

.stProgress > div > div > div > div {{ background:var(--acento) !important; }}
.ad-prog {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:1.6px;
text-transform:uppercase; color:var(--txt-3); text-align:center; margin:-.2rem 0 .6rem 0; }}

.ad-cred {{ border-top:1px solid var(--linea); margin-top:2.6rem; padding-top:1.3rem; }}
.ad-cred-g {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.9rem; margin-top:.9rem; }}
.ad-cred-i {{ font-size:.85rem; color:var(--txt); line-height:1.35; }}
.ad-cred-i span {{ display:block; font-family:'JetBrains Mono',monospace; font-size:.56rem;
letter-spacing:1.6px; text-transform:uppercase; color:var(--txt-3); margin-bottom:.2rem; }}
.ad-foot {{ display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;
font-family:'JetBrains Mono',monospace; font-size:.58rem; letter-spacing:1.6px;
text-transform:uppercase; color:var(--txt-3); margin-top:1.6rem; padding-bottom:1rem; }}

.ad-list {{ list-style:none; margin:.4rem 0 0 0; padding:0; }}
.ad-li {{ display:grid; grid-template-columns:auto minmax(0,1fr); gap:.75rem;
padding:.7rem 0; border-top:1px solid var(--linea); align-items:start; }}
.ad-li:first-child {{ border-top:none; padding-top:.2rem; }}
.ad-li-n {{ font-family:'JetBrains Mono',monospace; font-size:.6rem; letter-spacing:1px;
color:var(--acento); background:color-mix(in srgb, var(--acento) 12%, transparent);
border:1px solid color-mix(in srgb, var(--acento) 30%, transparent);
border-radius:8px; padding:.22rem .38rem; margin-top:.12rem; }}
.ad-li-t {{ font-size:.92rem; font-weight:600; color:var(--txt); line-height:1.35; }}
.ad-li-b {{ font-size:.87rem; color:var(--txt-2); line-height:1.6; margin-top:.2rem; }}
.ad-li-b b, .ad-li-t b {{ color:var(--txt); }}

/* --- modo claro / oscuro forzado sobre el tema base de Streamlit --- */
:root, html, body, .stApp {{ color-scheme: {"dark" if ES_OSCURO else "light"}; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottomBlockContainer"], section.main {{ background: transparent !important; }}
.stApp, [data-testid="stAppViewContainer"] {{ color: var(--txt) !important; }}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; pointer-events: none !important; }}
[data-testid="stToolbar"] * {{ pointer-events: auto !important; }}
h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] strong {{ color: var(--txt) !important; }}
[data-testid="stAlert"] {{ background: var(--panel) !important; border:1px solid var(--linea) !important;
border-radius:14px !important; color: var(--txt-2) !important; }}
[data-testid="stAlert"] * {{ color: var(--txt-2) !important; }}
[data-testid="stFileUploaderDropzone"], [data-baseweb="file-uploader"] {{ background: transparent !important; }}
[data-testid="stSpinner"] * {{ color: var(--txt-2) !important; }}
hr {{ border-color: var(--linea) !important; }}

@media (max-width:640px) {{
.ad-h1 {{ font-size:3rem; }}
.ad-cred-g {{ grid-template-columns:1fr; }}
.ad-verdict {{ grid-template-columns:1fr; }}
}}
</style>
""")

# ==================================================================
#                  NAV + TEMA
# ==================================================================
col_nav, col_tema = st.columns([5, 1.1])
with col_nav:
    html("""
    <div class="ad-nav">
    <div class="ad-dot"></div>
    <div class="ad-nav-t">AgroDetect &nbsp;/&nbsp; Grupo 4 &nbsp;/&nbsp; UNAH-CURC Comayagua</div>
    </div>
    """)
with col_tema:
    if st.button("☀️ Claro" if ES_OSCURO else "🌙 Oscuro", key="btn_tema", use_container_width=True):
        st.session_state.tema = "claro" if ES_OSCURO else "oscuro"
        st.rerun()

# ==================================================================
#                  HERO
# ==================================================================
html("""
<div class="ad-hero">
<div class="ad-tag">Visión por computadora · Café · Honduras</div>
<h1 class="ad-h1">Agro<i>Detect</i></h1>
<p class="ad-lede">Primera línea de apoyo al diagnóstico fitosanitario del cafeto. Sube una hoja y el modelo estima la condición en segundos.</p>
<div class="ad-meta">
<div>Modelo<b>MobileNetV2</b></div>
<div>Clases<b>6 condiciones</b></div>
<div>Clase<b>Inteligencia Artificial</b></div>
<div>Carrera<b>Ing. en Sistemas</b></div>
</div>
</div>
<div class="ad-flow">
<div class="ad-flow-i"><div class="ad-flow-n">01</div> Captura</div>
<div class="ad-flow-s"></div>
<div class="ad-flow-i"><div class="ad-flow-n">02</div> Análisis</div>
<div class="ad-flow-s"></div>
<div class="ad-flow-i"><div class="ad-flow-n">03</div> Diagnóstico</div>
</div>
<div class="ad-tip"><div>📸</div><div><b>Para mejores resultados:</b> una sola hoja, de cerca, con luz natural y sin objetos de fondo.</div></div>
""")


# ==================================================================
#                  LÓGICA
# ==================================================================
def generar_miniatura_b64(imagen_pil):
    thumb = imagen_pil.copy()
    thumb.thumbnail((80, 80))
    buf = BytesIO()
    thumb.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analizar_y_generar_entrada(imagen_pil):
    modo, candidatas, tabla_probs = diagnosticar(imagen_pil)
    entrada = {
        "thumb_b64": generar_miniatura_b64(imagen_pil),
        "modo": modo, "candidatas": candidatas, "tabla_probs": tabla_probs,
        "recomendacion": None, "opinion": None,
    }

    if modo == "unico":
        clase, _ = candidatas[0]
        try:
            entrada["recomendacion"] = obtener_recomendacion_unica(clase)
        except Exception:
            pass
    elif modo == "diferencial":
        nombres = [NOMBRES_DISPLAY.get(c, c) for c, _ in candidatas]
        try:
            entrada["recomendacion"] = obtener_recomendacion_diferencial(nombres)
        except Exception:
            pass
        try:
            entrada["opinion"] = obtener_segunda_opinion_visual(imagen_pil, nombres)
        except Exception:
            pass
    else:
        try:
            entrada["opinion"] = obtener_segunda_opinion_visual(
                imagen_pil, list(NOMBRES_DISPLAY.values()))
        except Exception:
            pass
    return entrada



import re as _re


def _inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = _re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    return t.strip()


def formatear_recomendacion(texto):
    """Convierte la respuesta del modelo en una lista ordenada y legible:
    cada punto (lo que viene entre ** **) ocupa su propio renglon."""
    if not texto:
        return ""
    t = texto.replace("\r", "")
    # separa puntos aunque el modelo los devuelva todos en un solo parrafo
    t = _re.sub(r"\s*\*+\s*(?=\*\*)", "\n", t)
    t = _re.sub(r"(?<!\n)\s+(?=\d+\.\s*\*\*)", "\n", t)
    lineas = [l.strip(" \t") for l in t.split("\n")]
    lineas = [l for l in lineas if l.strip()]

    intro, items = [], []
    for l in lineas:
        limpio = _re.sub(r"^(?:[-•]+\s*|\*(?!\*)\s*|\d+[\.\)]\s*)", "", l).strip()
        m = _re.match(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$", limpio, _re.S)
        if m:
            items.append((_inline(m.group(1)), _inline(m.group(2))))
        elif items:
            titulo, cuerpo = items[-1]
            items[-1] = (titulo, (cuerpo + " " + _inline(limpio)).strip())
        elif limpio:
            intro.append(_inline(limpio))

    partes = ""
    if intro:
        partes += f'<div class="ad-note-b">{" ".join(intro)}</div>'
    if items:
        li = ""
        for i, (titulo, cuerpo) in enumerate(items, 1):
            li += (f'<li class="ad-li"><span class="ad-li-n">{i:02d}</span>'
                   f'<div><div class="ad-li-t">{titulo}</div>'
                   + (f'<div class="ad-li-b">{cuerpo}</div>' if cuerpo else "")
                   + '</div></li>')
        partes += f'<ol class="ad-list">{li}</ol>'
    if not partes:
        partes = f'<div class="ad-note-b">{_inline(texto)}</div>'
    return partes


def veredicto(icono, nombre, meta_txt, pill, tinta, pct):
    return (
        f'<div class="ad-verdict" style="--ink:{tinta}; --pct:{pct:.0f}">'
        f'<div class="ad-gauge"><div class="ad-gauge-in">{icono}</div></div>'
        f'<div><div class="ad-verdict-n">{nombre}</div>'
        f'<div class="ad-verdict-m">{meta_txt}</div>'
        f'<div class="ad-pill">{pill}</div></div></div>'
    )


def panel(titulo, cuerpo_html):
    return f'<div class="ad-panel"><div class="ad-note-t">{titulo}</div>{cuerpo_html}</div>'


def renderizar_entrada(entrada):
    modo, candidatas, tabla = entrada["modo"], entrada["candidatas"], entrada["tabla_probs"]

    html(
        '<div class="ad-run">'
        f'<img src="data:image/png;base64,{entrada["thumb_b64"]}" alt="muestra"/>'
        '<div class="ad-run-t">Muestra analizada</div><div class="ad-run-l"></div></div>'
    )

    filas = ""
    for clase, p in tabla.items():
        color = TINTA_CLASE.get(clase, "var(--acento)")
        filas += (
            '<div class="ad-row">'
            f'<div class="ad-row-n">{ICONOS_CLASE.get(clase, "🌱")} {NOMBRES_DISPLAY.get(clase, clase)}</div>'
            f'<div class="ad-row-p">{p:.0%}</div>'
            f'<div class="ad-track"><div class="ad-fill" style="width:{p * 100:.1f}%;background:{color}"></div></div>'
            '</div>'
        )
    html(f'<div class="ad-panel"><div class="ad-lbl">Distribución de probabilidad</div>{filas}</div>')

    if modo == "unico":
        clase, conf = candidatas[0]
        html(veredicto(ICONOS_CLASE.get(clase, "🌱"), NOMBRES_DISPLAY.get(clase, clase),
                       f"Confianza {conf:.0%}", "Diagnóstico certificado",
                       TINTA_CLASE.get(clase, "var(--acento)"), conf * 100))
        if entrada["recomendacion"]:
            html(panel("💡 Recomendación agronómica",
                       formatear_recomendacion(entrada["recomendacion"])))
        else:
            st.info("No se pudo generar la recomendación automática. Consulte a un técnico del IHCAFE.")

    elif modo == "diferencial":
        clase, conf = candidatas[0]
        nombre_top = NOMBRES_DISPLAY.get(clase, clase)
        html(veredicto(ICONOS_CLASE.get(clase, "🌱"), nombre_top,
                       f"Confianza {conf:.0%} · no concluyente", "Requiere confirmación",
                       TINTA_CLASE.get(clase, "var(--acento)"), conf * 100))
        if entrada["recomendacion"]:
            html(panel("💡 Orientación y manejo preventivo",
                       formatear_recomendacion(entrada["recomendacion"])))
        else:
            st.info("No se pudo generar la recomendación automática. Consulte a un técnico del IHCAFE.")
        if entrada["opinion"]:
            eleccion = entrada["opinion"].get("eleccion", "")
            just = entrada["opinion"].get("justificacion", "")
            if eleccion.strip().lower() == nombre_top.strip().lower():
                cuerpo = f'<div class="ad-agree">✅ <b>Coincidencia:</b> el modelo de visión también identificó {eleccion}.</div>'
            else:
                cuerpo = f'<div class="ad-dis">⚖️ <b>Sin coincidencia:</b> nuestro modelo sugiere {nombre_top}; el modelo de visión se inclina por {eleccion}. Consulte a un técnico del IHCAFE.</div>'
            html(panel("👁️ Segunda opinión visual",
                       cuerpo
                       + f'<div class="ad-note-b"><em>Justificación:</em> {_inline(just)}</div>'
                       + '<div class="ad-cap">Orientación complementaria de un modelo general; no reemplaza al modelo entrenado ni a un técnico.</div>'))

    else:
        clase, conf = candidatas[0]
        html(veredicto("❓", "Sin determinar", f"Máximo {conf:.0%}",
                       "No concluyente", "var(--txt-3)", conf * 100))
        st.info("Se recomienda consultar a un técnico del IHCAFE.")
        if entrada["opinion"]:
            html(panel("👁️ Segunda opinión visual",
                       f'<div class="ad-note-b"><b>Elección del modelo de visión:</b> {entrada["opinion"].get("eleccion", "")}</div>'
                       + f'<div class="ad-note-b"><em>Justificación:</em> {entrada["opinion"].get("justificacion", "")}</div>'
                       + '<div class="ad-cap">Orientación complementaria; no reemplaza el diagnóstico del modelo entrenado.</div>'))


# ==================================================================
#                  HISTORIAL
# ==================================================================
for entrada in st.session_state.historial:
    renderizar_entrada(entrada)

# ==================================================================
#                  COMPOSER
# ==================================================================
html('<div style="height:1.6rem"></div>')
col_up, col_btn = st.columns([5, 1.3])
with col_up:
    archivo = st.file_uploader("Adjuntar foto", type=["jpg", "jpeg", "png"],
                               label_visibility="collapsed",
                               key=f"uploader_{st.session_state.uploader_key}")
with col_btn:
    st.write("")
    enviar = st.button("Analizar", key="btn_enviar", type="primary",
                       disabled=(archivo is None), use_container_width=True)

if archivo is not None and enviar:
    imagen = Image.open(archivo)
    barra = st.progress(0)
    txt = st.empty()
    for pct, msg in [(20, "Leyendo imagen"), (50, "Extrayendo características"),
                     (80, "Clasificando condición"), (100, "Análisis completo")]:
        txt.markdown(f'<div class="ad-prog">{msg}</div>', unsafe_allow_html=True)
        barra.progress(pct)
        time.sleep(0.3)
    st.session_state.historial.append(analizar_y_generar_entrada(imagen))
    st.session_state.uploader_key += 1
    barra.empty()
    txt.empty()
    st.rerun()

if st.session_state.historial:
    if st.button("Limpiar sesión", key="btn_limpiar"):
        st.session_state.historial = []
        st.session_state.uploader_key += 1
        st.rerun()

# ==================================================================
#                  CRÉDITOS
# ==================================================================
html("""
<div class="ad-cred">
<div class="ad-lbl">Grupo 4 · Proyecto de Inteligencia Artificial</div>
<div class="ad-cred-g">
<div class="ad-cred-i"><span>Integrante</span>Jeimy Jazmín Palma Santos</div>
<div class="ad-cred-i"><span>Integrante</span>Ángeles Izamar Euceda Herrera</div>
<div class="ad-cred-i"><span>Integrante</span>Kilver Said Nolasco Parada</div>
</div>
<div class="ad-foot"><div>UNAH — Campus Comayagua</div><div>Ing. en Sistemas Computacionales</div><div>2026</div></div>
</div>
""")
