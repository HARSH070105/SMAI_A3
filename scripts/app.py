"""
app.py — Monument Explorer Streamlit App
Two modes:
  1. Direct Classification  – upload an image, get monument prediction + info
  2. Scavenger Hunt         – city dropdown → progressive clues → upload photo → verify
"""

import json
import random
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Monument Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── global ── */
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Playfair Display', serif; }

  .stApp { background: #0f0e17; color: #fffffe; }

  /* ── sidebar ── */
  section[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid #2d2d44;
  }
  section[data-testid="stSidebar"] * { color: #fffffe !important; }

  /* ── hero title ── */
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 900;
    background: linear-gradient(135deg, #ff8906, #f25f4c, #e53170);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
  }
  .hero-sub {
    color: #a7a9be;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── mode cards ── */
  .mode-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s, transform 0.2s;
    cursor: pointer;
    min-height: 180px;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
  }
  .mode-card:hover { border-color: #ff8906; transform: translateY(-4px); }
  .mode-card .icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
  .mode-card h3 { font-family: 'Playfair Display', serif; color: #ff8906; margin: 0 0 0.5rem; }
  .mode-card p { color: #a7a9be; font-size: 0.85rem; margin: 0; }

  /* ── result card ── */
  .result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #ff8906;
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 1rem;
  }
  .result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #ff8906;
    margin: 0 0 0.25rem;
  }
  .result-location { color: #a7a9be; font-size: 0.9rem; margin-bottom: 1rem; }

  /* ── confidence bar ── */
  .conf-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
  .conf-label { color: #fffffe; font-size: 0.85rem; width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conf-bar-bg { flex: 1; background: #2d2d44; border-radius: 99px; height: 8px; }
  .conf-bar-fill { height: 8px; border-radius: 99px; background: linear-gradient(90deg, #ff8906, #f25f4c); }
  .conf-pct { color: #ff8906; font-size: 0.85rem; font-weight: 500; width: 44px; text-align: right; }

  /* ── info pills ── */
  .pill {
    display: inline-block;
    background: #2d2d44;
    color: #ff8906;
    border-radius: 99px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
    font-weight: 500;
  }

  /* ── fun facts ── */
  .fact-item {
    display: flex; gap: 0.75rem; align-items: flex-start;
    background: #16213e; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    border-left: 3px solid #ff8906;
  }
  .fact-num { color: #ff8906; font-weight: 700; min-width: 20px; }

  /* ── clue card ── */
  .clue-card {
    background: linear-gradient(135deg, #16213e, #0f3460);
    border: 1px solid #e53170;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
  }
  .clue-badge {
    position: absolute; top: -10px; left: 16px;
    background: #e53170; color: #fff;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 0.2rem 0.75rem; border-radius: 99px; text-transform: uppercase;
  }
  .clue-text { color: #fffffe; font-size: 1rem; line-height: 1.6; margin-top: 0.5rem; }

  /* ── success / fail banners ── */
  .success-banner {
    background: linear-gradient(135deg, #1a3a1a, #0f3d0f);
    border: 1px solid #4caf50; border-radius: 14px; padding: 1.5rem;
    text-align: center; color: #81c784;
    font-family: 'Playfair Display', serif; font-size: 1.4rem;
  }
  .fail-banner {
    background: linear-gradient(135deg, #3a1a1a, #3d0f0f);
    border: 1px solid #e53170; border-radius: 14px; padding: 1.5rem;
    text-align: center; color: #f48fb1;
    font-family: 'Playfair Display', serif; font-size: 1.3rem;
  }

  /* ── divider ── */
  .fancy-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, #ff8906, transparent);
    margin: 1.5rem 0;
  }

  /* ── streamlit overrides ── */
  .stButton > button {
    background: linear-gradient(135deg, #ff8906, #f25f4c) !important;
    color: #0f0e17 !important; border: none !important;
    font-weight: 700 !important; font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important; padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  div[data-testid="stFileUploader"] {
    background: #1a1a2e; border: 1px dashed #2d2d44;
    border-radius: 12px; padding: 1rem;
  }

  .stSelectbox > div > div { background: #1a1a2e !important; color: #fffffe !important; }
  .stProgress > div > div > div { background: linear-gradient(90deg, #ff8906, #f25f4c) !important; }

  /* ── map link button ── */
  .map-btn {
    display: inline-block;
    background: #16213e; border: 1px solid #ff8906;
    color: #ff8906 !important; border-radius: 8px;
    padding: 0.5rem 1.2rem; font-size: 0.85rem;
    text-decoration: none; font-weight: 600;
    transition: background 0.2s;
  }
  .map-btn:hover { background: #ff8906; color: #0f0e17 !important; }

  /* ── section header ── */
  .section-hdr {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; color: #a7a9be;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_metadata(path: str = "monuments_metadata.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Metadata file not found: {path}")
        return {}


@st.cache_resource
def load_clip_model():
    """Load model, processor and precompute text features.
    
    Class names are derived from the local metadata JSON — the HF dataset
    is NOT downloaded here, so no image data is pulled into memory.
    """
    from model_utils import load_model, precompute_text_features
    from dataset_utils import build_enriched_prompts

    METADATA_JSON = "monuments_metadata.json"
    LORA_PATH     = st.secrets.get("LORA_PATH", None)

    # ── class names come from metadata keys, zero dataset download ──
    metadata    = load_metadata(METADATA_JSON)   # already cached via @st.cache_data
    class_names = list(metadata.keys())          # e.g. ["Ajanta Caves", "Charar-E-Sharif", ...]

    enriched_prompts = build_enriched_prompts(class_names, metadata)

    model, processor, device = load_model(lora_weights_path=LORA_PATH)
    text_features = precompute_text_features(
        model, processor, class_names, device,
        enriched_prompts=enriched_prompts,
    )
    return model, processor, text_features, class_names, device


def classify_image(pil_image: Image.Image):
    model, processor, text_features, class_names, device = load_clip_model()
    from model_utils import run_inference
    return run_inference(pil_image, model, processor, text_features, class_names, device)


def get_meta(name: str, metadata: dict) -> dict | None:
    """Fuzzy-ish lookup: try exact key, then replace underscores."""
    return metadata.get(name) or metadata.get(name.replace("_", " "))


def cities_from_metadata(metadata: dict) -> dict[str, list[str]]:
    """Return {city_state: [monument_name, ...]} grouped by location."""
    city_map: dict[str, list[str]] = {}
    for name, info in metadata.items():
        loc = info.get("location", "Unknown")
        city_map.setdefault(loc, []).append(name)
    return dict(sorted(city_map.items()))


# ── Clue generation ─────────────────────────────────────────────────────────

def build_clues(meta: dict) -> list[str]:
    clues = []
    if meta.get("state"):
        clues.append(f"🗺️ This monument is somewhere in **{meta['state']}**.")
    if meta.get("category"):
        clues.append(f"🏷️ It belongs to the category: **{meta['category']}**.")
    if meta.get("opening_hours"):
        clues.append(f"🕐 You can visit between **{meta['opening_hours']}**.")
    if meta.get("ticket_price"):
        clues.append(f"🎟️ Entry fee: **{meta['ticket_price']}**.")
    facts = meta.get("fun_facts", [])
    for fact in facts[:3]:
        clues.append(f"💡 {fact}")
    hist = meta.get("history", "")
    if hist:
        snippet = hist[:180].rstrip()
        clues.append(f"📜 *\"{snippet}…\"*")
    if meta.get("display_name"):
        clues.append(f"🏛️ You're looking for: **{meta['display_name']}**!")
    return clues


# ── Render helpers ───────────────────────────────────────────────────────────

def render_confidence_bars(probs: dict, top_n: int = 5):
    st.markdown('<div class="section-hdr">Confidence</div>', unsafe_allow_html=True)
    items = list(probs.items())[:top_n]
    for label, prob in items:
        pct = prob * 100
        display = label.replace("_", " ")
        st.markdown(f"""
        <div class="conf-row">
          <span class="conf-label" title="{display}">{display}</span>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct:.1f}%"></div>
          </div>
          <span class="conf-pct">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_monument_info(name: str, meta: dict, show_location_link: bool = True):
    display = meta.get("display_name", name.replace("_", " "))
    location = meta.get("location", "")
    category = meta.get("category", "")
    hours    = meta.get("opening_hours", "")
    price    = meta.get("ticket_price", "")
    maps_url = meta.get("google_maps", "")
    wiki_url = meta.get("wikipedia_url", "")

    st.markdown(f"""
    <div class="result-card">
      <div class="result-title">{display}</div>
      <div class="result-location">📍 {location}</div>
      {"<span class='pill'>"+category+"</span>" if category else ""}
      {"<span class='pill'>🕐 "+hours+"</span>" if hours else ""}
      {"<span class='pill'>🎟️ "+price+"</span>" if price else ""}
    </div>
    """, unsafe_allow_html=True)

    history = meta.get("history", "")
    if history:
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-hdr">History</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#c9c9d3;line-height:1.7;">{history}</p>', unsafe_allow_html=True)

    facts = meta.get("fun_facts", [])
    if facts:
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-hdr">Fun Facts</div>', unsafe_allow_html=True)
        for i, fact in enumerate(facts, 1):
            st.markdown(f"""
            <div class="fact-item">
              <span class="fact-num">{i}</span>
              <span style="color:#fffffe;">{fact}</span>
            </div>""", unsafe_allow_html=True)

    if show_location_link and (maps_url or wiki_url):
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        links_html = ""
        if maps_url:
            links_html += f'<a class="map-btn" href="{maps_url}" target="_blank">🗺️ Google Maps</a>&nbsp;&nbsp;'
        if wiki_url:
            links_html += f'<a class="map-btn" href="{wiki_url}" target="_blank">📖 Wikipedia</a>'
        st.markdown(links_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# App layout
# ════════════════════════════════════════════════════════════════════════════

metadata = load_metadata()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
      <div style="font-size:2.5rem;">🏛️</div>
      <div class="hero-title" style="font-size:1.6rem;">Monument<br>Explorer</div>
      <div class="hero-sub" style="font-size:0.75rem; margin-top:0.4rem;">India's Heritage · AI Powered</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio(
        "Choose Mode",
        ["🔍 Direct Classification", "🗺️ Scavenger Hunt"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        '<div style="color:#a7a9be; font-size:0.78rem; line-height:1.6;">'
        'Upload a photo of any Indian monument and let the AI identify it, '
        'or play the Scavenger Hunt to explore heritage sites in real life!'
        '</div>',
        unsafe_allow_html=True
    )

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem;">
  <div class="hero-title">Discover India's<br>Living Heritage</div>
  <div class="hero-sub">Powered by CLIP · Zero-shot monument recognition</div>
</div>
<hr class="fancy-divider">
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — Direct Classification
# ════════════════════════════════════════════════════════════════════════════

if mode == "🔍 Direct Classification":
    st.markdown("## 🔍 Identify a Monument")
    st.markdown(
        '<p style="color:#a7a9be;">Upload any photo and the model will identify the monument, '
        'show confidence scores, and share historical information.</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(image, caption="Your uploaded image", use_container_width=True)

        with col_res:
            with st.spinner("Analysing monument…"):
                predicted_class, probs = classify_image(image)

            meta = get_meta(predicted_class, metadata)
            display_name = (
                meta.get("display_name", predicted_class.replace("_", " "))
                if meta else predicted_class.replace("_", " ")
            )

            top_prob = list(probs.values())[0] * 100
            confidence_label = (
                "Very High" if top_prob > 80
                else "High" if top_prob > 60
                else "Moderate" if top_prob > 40
                else "Low"
            )
            conf_color = (
                "#4caf50" if top_prob > 80
                else "#ff8906" if top_prob > 60
                else "#f25f4c"
            )

            st.markdown(f"""
            <div style="margin-bottom:1rem;">
              <div style="color:#a7a9be; font-size:0.8rem; text-transform:uppercase; letter-spacing:.1em;">Prediction</div>
              <div style="font-family:'Playfair Display',serif; font-size:2rem; color:#ff8906; font-weight:700;">{display_name}</div>
              <div style="color:{conf_color}; font-size:0.9rem; font-weight:600;">
                {confidence_label} confidence · {top_prob:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

            render_confidence_bars(probs, top_n=5)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        if meta:
            render_monument_info(predicted_class, meta)
        else:
            st.info("No detailed metadata found for this monument. Check your `monuments_metadata.json`.")

        # Alternatives section
        alt_items = list(probs.items())[1:4]
        if alt_items:
            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-hdr">Could also be…</div>', unsafe_allow_html=True)
            alt_cols = st.columns(len(alt_items))
            for col, (alt_name, alt_prob) in zip(alt_cols, alt_items):
                alt_meta = get_meta(alt_name, metadata)
                alt_display = (
                    alt_meta.get("display_name", alt_name.replace("_", " "))
                    if alt_meta else alt_name.replace("_", " ")
                )
                alt_loc = alt_meta.get("location", "") if alt_meta else ""
                with col:
                    st.markdown(f"""
                    <div style="background:#1a1a2e; border:1px solid #2d2d44; border-radius:12px; padding:1rem; text-align:center;">
                      <div style="color:#ff8906; font-weight:700; font-family:'Playfair Display',serif; font-size:1rem;">{alt_display}</div>
                      <div style="color:#a7a9be; font-size:0.78rem; margin-top:0.25rem;">{alt_loc}</div>
                      <div style="color:#f25f4c; font-size:0.9rem; font-weight:600; margin-top:0.4rem;">{alt_prob*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — Scavenger Hunt
# ════════════════════════════════════════════════════════════════════════════

else:
    st.markdown("## 🗺️ Scavenger Hunt")
    st.markdown(
        '<p style="color:#a7a9be;">Pick a city, follow the progressive clues, '
        'find the monument in real life, take a photo and upload it to verify!</p>',
        unsafe_allow_html=True
    )

    if not metadata:
        st.error("Metadata not loaded. Cannot run Scavenger Hunt.")
        st.stop()

    city_map = cities_from_metadata(metadata)
    cities   = list(city_map.keys())

    # ── State init ───────────────────────────────────────────────────────────
    for key, default in {
        "hunt_city": None,
        "hunt_target": None,
        "hunt_clues": [],
        "clues_revealed": 0,
        "hunt_solved": False,
        "hunt_failed": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── City & start ─────────────────────────────────────────────────────────
    col_city, col_btn = st.columns([3, 1])
    with col_city:
        selected_city = st.selectbox("Select a city", cities, index=0)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        start = st.button("🎯 Start Hunt")

    if start:
        monuments_in_city = city_map[selected_city]
        target = random.choice(monuments_in_city)
        meta   = get_meta(target, metadata)
        clues  = build_clues(meta) if meta else [f"Find the monument: {target.replace('_',' ')}"]

        st.session_state.hunt_city       = selected_city
        st.session_state.hunt_target     = target
        st.session_state.hunt_clues      = clues
        st.session_state.clues_revealed  = 1
        st.session_state.hunt_solved     = False
        st.session_state.hunt_failed     = False

    # ── Active hunt ───────────────────────────────────────────────────────────
    if st.session_state.hunt_target and not st.session_state.hunt_solved:
        target = st.session_state.hunt_target
        clues  = st.session_state.hunt_clues
        meta   = get_meta(target, metadata)

        st.markdown(f"""
        <div style="background:#16213e; border-radius:12px; padding:1rem 1.5rem; margin:1rem 0;
                    border-left:4px solid #e53170;">
          <span style="color:#a7a9be; font-size:0.8rem; text-transform:uppercase; letter-spacing:.1em;">Hunting in</span><br>
          <span style="font-family:'Playfair Display',serif; font-size:1.3rem; color:#fffffe;">
            📍 {st.session_state.hunt_city}
          </span>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        revealed  = st.session_state.clues_revealed
        total_cls = len(clues)
        progress  = revealed / total_cls
        st.progress(progress)
        st.caption(f"Clue {revealed} of {total_cls} revealed")

        # Show revealed clues
        for i in range(revealed):
            st.markdown(f"""
            <div class="clue-card">
              <span class="clue-badge">Clue {i+1}</span>
              <div class="clue-text">{clues[i]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Reveal next clue button (disabled on last clue)
        col_clue, col_hint = st.columns([2, 1])
        with col_clue:
            if revealed < total_cls:
                if st.button("💡 Next Clue"):
                    st.session_state.clues_revealed += 1
                    st.rerun()
            else:
                st.markdown(
                    '<div style="color:#a7a9be; font-size:0.85rem; padding:0.5rem 0;">'
                    '✅ All clues revealed — time to find it!</div>',
                    unsafe_allow_html=True
                )

        # Google Maps hint (available after 2+ clues)
        if revealed >= 2 and meta and meta.get("google_maps"):
            with col_hint:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<a class="map-btn" href="{meta["google_maps"]}" target="_blank">🗺️ Area Map</a>',
                    unsafe_allow_html=True
                )

        # ── Upload verification ───────────────────────────────────────────────
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown("### 📸 Found it? Upload your photo to verify!")

        verify_img = st.file_uploader(
            "Upload your monument photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="verify_upload",
            label_visibility="collapsed",
        )

        if verify_img:
            v_image = Image.open(verify_img).convert("RGB")
            vcol1, vcol2 = st.columns([1, 1], gap="large")

            with vcol1:
                st.image(v_image, caption="Your photo", use_container_width=True)

            with vcol2:
                with st.spinner("Verifying your location…"):
                    time.sleep(0.5)                          # small UX delay
                    pred_class, probs = classify_image(v_image)

                top_prob  = list(probs.values())[0] * 100
                is_correct = pred_class == target

                if is_correct:
                    st.markdown(f"""
                    <div class="success-banner">
                      🎉 Congratulations!<br>
                      <span style="font-size:1rem; font-family:'DM Sans',sans-serif; color:#a5d6a7;">
                        You found <strong>{(meta or {}).get('display_name', target.replace('_',' '))}</strong>!<br>
                        Confidence: {top_prob:.1f}%
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.hunt_solved = True
                    st.balloons()
                else:
                    pred_meta    = get_meta(pred_class, metadata)
                    pred_display = (
                        pred_meta.get("display_name", pred_class.replace("_", " "))
                        if pred_meta else pred_class.replace("_", " ")
                    )
                    st.markdown(f"""
                    <div class="fail-banner">
                      😕 Not quite…<br>
                      <span style="font-size:0.9rem; font-family:'DM Sans',sans-serif;">
                        The model sees <strong>{pred_display}</strong> ({top_prob:.1f}%).<br>
                        Keep exploring and try again!
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🔄 Try Again"):
                        st.session_state.verify_upload = None
                        st.rerun()

    # ── Solved state ─────────────────────────────────────────────────────────
    if st.session_state.hunt_solved:
        target = st.session_state.hunt_target
        meta   = get_meta(target, metadata)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown("### 🏛️ About this Monument")
        if meta:
            render_monument_info(target, meta)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        if st.button("🎯 Start a New Hunt"):
            for key in ["hunt_city","hunt_target","hunt_clues","clues_revealed","hunt_solved","hunt_failed"]:
                st.session_state[key] = None if "city" in key or "target" in key else ([] if "clues" in key else 0 if "revealed" in key else False)
            st.rerun()
