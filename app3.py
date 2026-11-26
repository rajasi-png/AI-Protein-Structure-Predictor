import os
import time
from datetime import datetime

import streamlit as st
import py3Dmol
from dotenv import load_dotenv

from esm.sdk.api import ESMProtein, GenerationConfig
from esm.sdk.forge import ESM3ForgeInferenceClient

load_dotenv()

st.set_page_config(page_title="AI Protein Structure Explorer", page_icon="🧬", layout="wide")
st.title("🧬 AI Protein Structure Explorer")
st.markdown(
    "**Turn a protein sequence into a predicted 3D structure.** "
    "Proteins are chains of amino acids written as letters. "
    "An AI model predicts the 3D shape they fold into."
)

EXAMPLES = {
    "Small test protein": "MPRTKEINDAGLIVHSPQWFYK",
    "Insulin-like peptide": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "Custom sequence": "",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_URL = os.getenv("ESM_URL", "https://forge.evolutionaryscale.ai")
MODEL_CANDIDATES = [
    m.strip()
    for m in os.getenv(
        "ESM_MODELS", "esm3-open-2024-03,esm3-medium-2024-08,esm3-large-2024-03"
    ).split(",")
    if m.strip()
]


def clean_sequence(raw):
    seq = "".join(raw.split()).upper()
    if not seq:
        st.error("Please enter a protein sequence.")
        st.stop()
    invalid = sorted(set(seq) - VALID_AA)
    if invalid:
        st.error("Invalid characters: " + ", ".join(invalid))
        st.stop()
    if len(seq) < 10:
        st.error("Please enter at least 10 amino acids.")
        st.stop()
    if len(seq) > 400:
        st.error("Please use 400 amino acids or fewer.")
        st.stop()
    return seq


def get_token():
    token = os.getenv("ESM_API_KEY", "").strip().strip('"').strip("'")
    if not token:
        st.error("ESM_API_KEY is missing from your .env file.")
        st.stop()
    return token


def to_pdb_text(protein):
    if hasattr(protein, "to_pdb_string"):
        return protein.to_pdb_string()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.pdb")
        protein.to_pdb(path)
        with open(path) as f:
            return f.read()


def predict_structure(sequence):
    token = get_token()
    protein = ESMProtein(sequence=sequence)
    config = GenerationConfig(track="structure", num_steps=8)
    errors = []

    for model_name in MODEL_CANDIDATES:
        try:
            client = ESM3ForgeInferenceClient(model=model_name, url=DEFAULT_URL, token=token)
            started = time.time()
            result = client.generate(protein, config)
        except Exception as exc:
            errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
            continue

        if not isinstance(result, ESMProtein):
            code = getattr(result, "error_code", "unknown")
            msg = getattr(result, "error_msg", str(result))
            errors.append(f"{model_name}: API error {code}: {msg}")
            continue

        if result.coordinates is None:
            errors.append(f"{model_name}: no coordinates returned")
            continue

        return to_pdb_text(result), round(time.time() - started, 1), model_name

    raise RuntimeError(
        "All models failed (URL: " + DEFAULT_URL + ")\n\n" + "\n".join(errors)
    )


def render_structure(pdb_text, style):
    view = py3Dmol.view(width=850, height=600)
    view.addModel(pdb_text, "pdb")
    if style.startswith("Cartoon"):
        view.setStyle({"cartoon": {"color": "spectrum"}})
    elif style.startswith("Stick"):
        view.setStyle({"stick": {"colorscheme": "cyanCarbon"}})
    else:
        view.setStyle({"sphere": {"scale": 0.35, "color": "spectrum"}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), width=880, height=620)


with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Choose an example", list(EXAMPLES.keys()))
    sequence_input = st.text_area("Protein sequence", value=EXAMPLES[choice], height=180)
    style_choice = st.selectbox(
        "3D display style",
        [
            "Cartoon - overall protein shape",
            "Stick - chemical bonds",
            "Space-filling - atom sizes",
        ],
    )
    predict_button = st.button(" Predict 3D Structure :3", use_container_width=True)
    st.caption(f"API URL: {DEFAULT_URL}")
    st.caption("Models tried: " + ", ".join(MODEL_CANDIDATES))
    st.info("AI predictions are hypotheses; experiments such as X-ray crystallography, cryo-EM and NMR confirm structures.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive 3D Model")
    if predict_button:
        sequence = clean_sequence(sequence_input)
        try:
            with st.spinner("AI is predicting the 3D structure..."):
                pdb_text, elapsed, used_model = predict_structure(sequence)
            st.success(f"Done in {elapsed}s with {used_model}. Length: {len(sequence)} residues.")
            render_structure(pdb_text, style_choice)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download PDB",
                data=pdb_text,
                file_name=f"predicted_protein_{stamp}.pdb",
                mime="chemical/x-pdb",
                use_container_width=True,
            )
            with st.expander("View PDB text"):
                st.code(pdb_text[:3000])
        except Exception as error:
            st.error("Structure prediction failed")
            st.code(str(error))
    else:
        st.info("Pick a sequence and press **Predict 3D Structure**.")

with col2:
    st.subheader("How it works")
    st.markdown(
        "1. **Sequence:** amino-acid letters.\n"
        "2. **AI prediction:** ESM predicts 3D coordinates.\n"
        "3. **PDB file:** coordinates in a standard format.\n"
        "4. **py3Dmol:** interactive 3D view.\n"
        "5. **Chemistry:** shape determines function."
    )

st.divider()
st.caption("Educational demonstration. Predicted structures are computational models.")
