import os
import io
import time
from datetime import datetime

import streamlit as st
import py3Dmol
from dotenv import load_dotenv

from esm.sdk.forge import ESM3ForgeInferenceClient

from esm.sdk.forge import ESM3ForgeInferenceClient

client = ESM3ForgeInferenceClient(model="esm3-open-2024-03", url="https://biohub.ai")

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
)

# Load secret API key
load_dotenv()

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Protein Structure Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 AI Protein Structure Explorer")
st.markdown(
    """
    **Turn a protein sequence into a predicted 3D structure.**

    Proteins are made from amino-acid building blocks represented by letters
    such as A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, and Y.
    This app uses an AI protein-language model to predict a possible 3D shape.
    """
)

# ------------------------------------------------------------
# Example sequences
# ------------------------------------------------------------
EXAMPLES = {
    "Insulin-like short peptide": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "Small test protein": "MPRTKEINDAGLIVHSPQWFYK",
    "Custom sequence": "",
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(raw_sequence: str) -> str:
    sequence = "".join(raw_sequence.split()).upper()

    if not sequence:
        st.error("Please enter a protein sequence.")
        st.stop()

    invalid = sorted(set(sequence) - VALID_AA)

    if invalid:
        st.error(
            "Invalid amino-acid characters found: "
            + ", ".join(invalid)
            + ". Use standard one-letter amino-acid codes."
        )
        st.stop()

    if len(sequence) < 10:
        st.error("Please enter at least 10 amino acids.")
        st.stop()

    if len(sequence) > 400:
        st.error(
            "For a smooth exhibition demo, use sequences of 400 amino acids or fewer."
        )
        st.stop()

    return sequence


def get_esm_client():
    api_key = os.getenv("ESM_API_KEY")

    if not api_key:
        st.error(
            "ESM API key is missing. Add it to your `.env` file as:\n\n"
            "`ESM_API_KEY=your_api_key_here`"
        )
        st.stop()

    return ESM3InferenceClient(
        model="esm3-open",
        api_key=api_key,
    )


def predict_structure(sequence: str):
    client = get_esm_client()

    protein = ESMProtein(sequence=sequence)

    config = GenerationConfig(
        track="structure",
        num_steps=len(sequence),
    )

    started = time.time()

    with st.spinner("AI is predicting the 3D structure..."):
        folded_protein = client.generate(protein, config)

    elapsed = round(time.time() - started, 1)
    pdb_text = folded_protein.to_pdb()

    return pdb_text, elapsed


def render_structure(pdb_text: str, style: str):
    view = py3Dmol.view(width=850, height=600)
    view.addModel(pdb_text, "pdb")

    if style == "Cartoon — shows overall protein shape":
        view.setStyle({"cartoon": {"color": "spectrum"}})

    elif style == "Stick — shows chemical bonds":
        view.setStyle({"stick": {"colorscheme": "cyanCarbon"}})

    elif style == "Space-filling — shows atom sizes":
        view.setStyle({"sphere": {"scale": 0.35, "color": "spectrum"}})

    view.zoomTo()
    html = view._make_html()
    st.components.v1.html(html, width=880, height=620)


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    example_choice = st.selectbox(
        "Choose an example",
        list(EXAMPLES.keys()),
    )

    sequence_input = st.text_area(
        "Protein sequence",
        value=EXAMPLES[example_choice],
        height=180,
        help="Paste a protein sequence using one-letter amino-acid codes.",
    )

    style_choice = st.selectbox(
        "3D display style",
        [
            "Cartoon — shows overall protein shape",
            "Stick — shows chemical bonds",
            "Space-filling — shows atom sizes",
        ],
    )

    predict_button = st.button(
        "🚀 Predict 3D Structure",
        use_container_width=True,
    )

    st.info(
        "AI predictions are scientific hypotheses. "
        "Real laboratory methods such as X-ray crystallography, cryo-EM, "
        "and NMR are needed to confirm a structure experimentally."
    )

# ------------------------------------------------------------
# Main interface
# ------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive 3D Model")

    if predict_button:
        sequence = clean_sequence(sequence_input)

        try:
            pdb_text, elapsed = predict_structure(sequence)

            st.success(
                f"Prediction complete in {elapsed} seconds. "
                f"Sequence length: {len(sequence)} amino acids."
            )

            render_structure(pdb_text, style_choice)

            # Save downloadable PDB
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdb_filename = f"predicted_protein_{timestamp}.pdb"

            st.download_button(
                label="⬇️ Download predicted structure (.pdb)",
                data=pdb_text,
                file_name=pdb_filename,
                mime="chemical/x-pdb",
                use_container_width=True,
            )

            with st.expander("View predicted PDB file"):
                st.code(pdb_text[:3000] + "\n\n... truncated for display ...")

        except Exception as error:
            st.error(f"Structure prediction failed: {error}")

    else:
        st.info(
            "Enter a sequence and press **Predict 3D Structure** to begin."
        )

with col2:
    st.subheader("How it works")

    st.markdown(
        """
        1. **Sequence input**  
        A protein is a chain of amino acids, written as letters.

        2. **AI prediction**  
        ESM reads the sequence and predicts likely 3D coordinates.

        3. **PDB file**  
        The coordinates are converted into a standard PDB structure file.

        4. **3D visualization**  
        py3Dmol renders the model so you can rotate and inspect it.

        5. **Chemistry connection**  
        A protein’s shape helps determine its function—such as binding,
        catalysis, transport, immunity, or structural support.
        """
    )

    st.subheader("Try changing the sequence")

    st.markdown(
        """
        - Change a few amino acids.
        - Observe how the predicted shape may change.
        - Discuss why protein shape matters in medicine and drug design.
        """
    )

st.divider()
st.caption(
    "Educational demonstration. Predicted structures are computational models, "
    "not experimentally confirmed structures."
)