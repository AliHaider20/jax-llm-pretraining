import streamlit as st
import yaml

from data_loader import tokenizer
from model import MiniGPT
from inference import detect_red_team_prompt, load_model_from_checkpoint

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Prompt Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Model Configuration")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

@st.cache_resource
def load_model():
    """Load model from checkpoint using shared inference logic."""
    with st.spinner("Loading model..."):
        try:
            model = load_model_from_checkpoint()
            st.sidebar.success("✅ Model loaded from checkpoint")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load checkpoint: {str(e)}")
            st.stop()  # Stop execution if checkpoint fails

model = load_model()

# Model info
with st.sidebar.expander("📊 Model Info", expanded=False):
    st.write(f"**Max Length:** {config['MAX_LENGTH']}")
    st.write(f"**Vocab Size:** {tokenizer.n_vocab}")
    st.write(f"**Embed Dim:** {config['EMBED_DIM']}")
    st.write(f"**Num Heads:** {config['NUM_HEADS']}")
    st.write(f"**FF Dim:** {config['FEED_FORWARD_DIM']}")
    st.write(f"**Transformer Blocks:** {config['NUM_LAYERS']}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 Prompt Detector")
st.markdown("Generate model responses for your prompts using the trained MiniGPT model.")

# Create two columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📝 Input")
    prompt = st.text_area(
        "Enter your prompt:",
        height=80,
        placeholder="Type or paste your prompt here...",
        key="prompt_input"
    )

with col2:
    st.subheader("⚡ Parameters")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Lower = more focused | Higher = more creative"
    )
    
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=500,
        value=100,
        help="Maximum tokens to generate"
    )
    
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=10000,
        value=42,
        help="Random seed for reproducibility"
    )

# ══════════════════════════════════════════════════════════════════════════════
# RUN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

col1, col2, col3 = st.columns(3)

with col1:
    run_button = st.button("▶️ Run Inference", use_container_width=True)

with col2:
    clear_button = st.button("🔄 Clear", use_container_width=True)

with col3:
    if st.button("📋 Example Prompt", use_container_width=True):
        st.session_state.prompt_input = "What are the benefits of machine learning?"

if clear_button:
    st.session_state.prompt_input = ""
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

if run_button:
    if not prompt.strip():
        st.error("❌ Please enter a prompt before running inference.")
    else:
        try:
            with st.spinner("🔄 Running inference..."):
                output = detect_red_team_prompt(
                    model,
                    raw_prompt=prompt,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    seed=seed,
                )
            
            st.success("✅ Inference complete!")
            
            # Display results
            st.subheader("📤 Output")
            
            col_output, col_stats = st.columns([3, 1])
            
            with col_output:
                st.text_area(
                    "Generated Output:",
                    value=output,
                    height=200,
                    disabled=True,
                    key="output_display"
                )
            
            with col_stats:
                st.metric("Temperature", f"{temperature:.1f}")
                st.metric("Max Tokens", max_tokens)
                st.metric("Output Length", len(output.split()))
            
            # Download button
            st.download_button(
                label="⬇️ Download Output",
                data=output,
                file_name="output.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px;">
        🔍 Prompt Detector UI | Powered by MiniGPT + JAX
    </div>
    """,
    unsafe_allow_html=True
)
