import streamlit as st
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# API KEY CONFIGURATION
# - Cloud deploy: set GROQ_API_KEY in Streamlit Cloud "Secrets" settings
# - Local dev: paste key below OR set it in .streamlit/secrets.toml
# ==========================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
# ==========================================
if GROQ_API_KEY and len(GROQ_API_KEY) > 20:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Layout configuration
st.set_page_config(page_title="AMA Law Assistant", page_icon="🏛️", layout="centered")

# Inject Custom CSS for the AI Albania Theme with Modern Animations
st.markdown("""
<style>
    /* CSS Fade & Slide Animations */
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Main Background & Text */
    .stApp {
        background-color: #0d0f12 !important;
        color: #e0e6ed !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar styling */
    h1, h2, h3, p, span {
        color: #f8fafc !important;
        letter-spacing: -0.01em;
    }
    .accent-text {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #cc0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: slideUpFade 0.6s ease-out forwards;
    }
    
    /* Chat Message Bubbles - Apply entrance animations */
    .stChatMessage {
        border-radius: 16px;
        padding: 1.5rem;
        background-color: #171a21 !important;
        border: 1px solid #232730 !important;
        margin-bottom: 20px;
        animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 16px rgba(0,0,0,0.3);
    }
    
    /* User Message Bubble */
    [data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 4px solid #ff4b4b !important;
        background-color: #1a202a !important;
    }

    /* Slick Inputs, Selectboxes & Buttons */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stRadio label {
        color: #ffffff !important;
    }
    .stTextInput>div>div>input {
        background-color: #1a202a !important;
        border: 1px solid #232730 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.4) !important;
    }
    
    /* HTML Details/Citations overrides for sleek looks */
    details.citation {
        display: block;
        margin-top: 10px;
        margin-bottom: 12px;
        background: #1e2430;
        border-left: 3px solid #ff4b4b; /* Solid Red side border like a professional blockquote */
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 0.85em; /* Small readable text */
        line-height: 1.4;
        color: #a0aec0 !important;
        transition: all 0.2s ease;
    }
    details.citation > summary {
        display: inline-block;
        color: #ff4b4b !important;
        font-weight: 600;
        outline: none;
        list-style: none; /* Hide default arrow */
        cursor: pointer;
        padding-top: 4px;
        padding-bottom: 4px;
        transition: color 0.1s ease;
    }
    details.citation > summary:hover, details.citation[open] > summary {
        color: #ffffff !important; /* Turns white when hovered or clicked */
    }
    details.citation > summary::-webkit-details-marker {
        display: none; /* Safari fix */
    }
    details.citation[open] {
        background: #11151c;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        padding-bottom: 10px;
    }
    
    /* Buttons with Glow */
    .stButton>button {
        background-color: #1a202a !important;
        color: #ffffff !important;
        border: 1px solid #232730 !important;
        border-radius: 10px !important;
        font-weight: 600;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stButton>button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.15);
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
        box-shadow: none;
    }
    
    /* Clean Popover content backgrounds */
    [data-testid="stPopoverBody"] {
        background-color: #171a21 !important;
        border: 1px solid #232730 !important;
        border-radius: 12px;
        padding: 16px;
    }
    /* Make popover trigger button match regular buttons in height */
    [data-testid="stPopover"] button {
        background-color: #1a202a !important;
        color: #ffffff !important;
        border: 1px solid #232730 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        height: 38px !important;
        padding: 0 0.75rem !important;
        white-space: nowrap !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    [data-testid="stPopover"] button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.15);
    }
    .stButton > button {
        height: 38px !important;
        padding: 0 0.75rem !important;
        font-size: 0.875rem !important;
    }
    
    /* Hide branding and stray iframe component icons */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    iframe[height="0"] { display: none !important; }
    /* Hide Streamlit's auto-added anchor links on headings */
    h2 a, h3 a, h4 a { display: none !important; }
    /* Hide Streamlit Cloud 'Created by' badge and deploy button */
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stDeployButton"] {display: none !important;}
    /* Footer label below the chat input bar */
    .ama-footer {
        position: fixed;
        bottom: 4px;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 0.78rem;
        color: #4a5568 !important;
        letter-spacing: 0.02em;
        pointer-events: none;
        z-index: 999;
    }
    .ama-footer span { color: #ff4b4b !important; }
</style>
""", unsafe_allow_html=True)

# Initialization of UI State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "selected_model_state" not in st.session_state:
    # Llama 3 70B is best for generating strictly perfect foreign languages and obeying formatting instructions
    st.session_state.selected_model_state = "llama-3.3-70b-versatile"

# Floating Help Widget (pure HTML/CSS, no JS - fixes Streamlit script stripping)
st.markdown("""
<details class="help-fab-wrap">
  <summary class="help-fab-btn">?</summary>
  <div class="help-panel">
    <div class="help-lang-row">
      <label class="help-lang-label">
        <input type="radio" name="helplang" value="en" checked hidden>
        <span class="help-lang-chip">EN</span>
      </label>
      <label class="help-lang-label">
        <input type="radio" name="helplang" value="al" hidden>
        <span class="help-lang-chip">AL</span>
      </label>
    </div>
    <!-- EN Content -->
    <div class="help-content-en">
      <strong>📖 What is AMA Law Assistant?</strong><br>
      A RAG AI assistant built on <em>Law No. 97/2013 on Audiovisual Media in Albania</em>.<br><br>
      <strong>How it works:</strong><br>
      • Legal question → retrieves exact passages from the law + cited source.<br>
      • General question → answers from its own knowledge, no citation.<br><br>
      <strong>⚠️ Limitations:</strong><br>
      • Covers only Law No. 97/2013, not amendments.<br>
      • Based on the English translation of the document.<br>
      • Always verify AI answers with the original text.
    </div>
    <!-- AL Content -->
    <div class="help-content-al">
      <strong>📖 Çfarë është ky AMA Law Assistant?</strong><br>
      Asistent AI i llojit RAG mbi <em>Ligjin Nr. 97/2013 për Median Audiovizive në Shqipëri</em>.<br><br>
      <strong>Si funksionon:</strong><br>
      • Pyetje ligjore → gjeneron pasazhet përshtatshme bazuar në ligj + burim të marra nga ligji.<br>
      • Pyetje e përgjithshme → përgjigjet nga njohuritë fillestare, pa citime.<br><br>
      <strong>⚠️ Kufizimet:</strong><br>
      • Mbulon vetëm Ligjin Nr. 97/2013, jo ndryshimet.<br>
      • Bazuar në versionin anglisht të dokumentit.<br>
      • Verifikojini gjithmonë përgjigjet me tekstin origjinal.
    </div>
  </div>
</details>
<style>
  /* Wrapper sits fixed bottom-right */
  .help-fab-wrap {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 10000;
    list-style: none;
  }
  /* The ? circle button */
  .help-fab-btn {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: #1e2430;
    border: 2px solid #ff4b4b;
    color: #ff4b4b;
    font-size: 1.3rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    list-style: none;
    transition: background 0.2s, color 0.2s, transform 0.2s;
    box-shadow: 0 4px 18px rgba(255,75,75,0.25);
    user-select: none;
  }
  .help-fab-btn:hover {
    background: #ff4b4b;
    color: #ffffff;
    transform: scale(1.1);
  }
  .help-fab-wrap summary::-webkit-details-marker { display: none; }
  /* The panel that opens above the button */
  .help-panel {
    position: absolute;
    bottom: 56px;
    right: 0;
    width: 310px;
    background: rgba(17, 21, 28, 0.97);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,75,75,0.35);
    border-radius: 14px;
    padding: 16px 18px 14px;
    color: #d0d8e8;
    font-size: 0.83rem;
    line-height: 1.65;
    box-shadow: 0 16px 48px rgba(0,0,0,0.7);
  }
  /* Language chips row */
  .help-lang-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }
  .help-lang-label {
    flex: 1;
    display: flex;
  }
  .help-lang-chip {
    display: block;
    width: 100%;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    border: 1px solid #2d3748;
    color: #6b7a99;
    font-weight: 700;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  input[name=helplang]:checked + .help-lang-chip {
    border-color: #ff4b4b;
    color: #ff4b4b;
  }
  /* Show/hide bilingual content using CSS :has() */
  .help-content-al { display: none; }
  .help-lang-row:has(input[value=al]:checked) ~ .help-content-en { display: none; }
  .help-lang-row:has(input[value=al]:checked) ~ .help-content-al { display: block; }
</style>
""", unsafe_allow_html=True)
# Inject click-outside-to-close JS via components (bypasses Streamlit markdown script sanitizer)
import streamlit.components.v1 as components
components.html("""
<script>
  // Close the help <details> when user clicks anywhere outside it
  var parentDoc = window.parent.document;
  parentDoc.addEventListener('click', function(e) {
    var details = parentDoc.querySelector('.help-fab-wrap');
    if (details && details.open && !details.contains(e.target)) {
      details.removeAttribute('open');
    }
  }, true);
</script>
""", height=0)


# Main App Header — 3 columns: wide title + Clear + Settings (equal button widths)
col1, col2, col3 = st.columns([5.2, 1.4, 1.4])
with col1:
    st.markdown(
        "<h2 style='margin-top: -10px; margin-bottom: 0; white-space: nowrap;'>"
        "🏛️ <span class='accent-text'>AMA</span> Law Assistant</h2>",
        unsafe_allow_html=True
    )

# Clear Chat Button — hover pops via CSS, click clears directly
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Settings Popover
with col3:
    with st.popover("⚙️ Settings", use_container_width=True):
        st.markdown("**Model Configuration**")
        if GROQ_API_KEY and len(GROQ_API_KEY) > 20:
            selected_model = st.radio(
                "🎯 Active AI Model",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                index=0,
                help="These open-source models are hosted completely free globally."
            )
            st.session_state.selected_model_state = selected_model
        else:
            st.error("Invalid Groq API Key!")

st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_and_process_pdf(file_path):
    with st.spinner("Analyzing AMA Law knowledge base..."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store

# Relative path: works both locally and on Streamlit Cloud
pdf_path = os.path.join(os.path.dirname(__file__), "Law-no.-97-2013-on-the-Audiovisual-Media-in-the-Republic-of-Albania.pdf")

if not GROQ_API_KEY or len(GROQ_API_KEY) < 20:
    st.error("Please open `app.py` in your code editor and paste a valid **Groq API Key** into the `GROQ_API_KEY` variable at the top of the file! Get one for free at [console.groq.com/keys](https://console.groq.com/keys).")
else:
    if os.path.exists(pdf_path):
        if st.session_state.vector_store is None:
            try:
                st.session_state.vector_store = load_and_process_pdf(pdf_path)
            except Exception as e:
                st.error(f"Failed to process the PDF correctly: {str(e)}")
                st.stop()
        
        # Render historical chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # Footer label
        st.markdown(
            "<div class='ama-footer'>Made by AI Albania <span>❤️</span> for safe and responsible journalism</div>",
            unsafe_allow_html=True
        )

        # Create new chat prompt
        if prompt := st.chat_input("Ask a question about audiovisual laws (English or Albanian)..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Answer Generator
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Processing inquiry intelligently..."):
                    llm = ChatGroq(model=st.session_state.selected_model_state, temperature=0.2)
                    
                    system_prompt = (
                        "You are an expert AI assistant specializing in the Audiovisual Media Authority (AMA) in Albania. "
                        "LANGUAGE CRITICAL OVERRIDE: YOU MUST DETECT THE EXACT LANGUAGE THE USER TYPES IN. "
                        "IF THEY TYPE IN ALBANIAN, GENERATE 100% OF YOUR RESPONSE IN FLAWLESS STANDARD ALBANIAN (GJUHA STANDARDE SHQIPE). "
                        "IF THEY TYPE IN ENGLISH, GENERATE 100% OF YOUR RESPONSE IN FLAWLESS ENGLISH.\n\n"
                        "CRITICAL CITATION BYPASS RULES:\n"
                        "1) If the user asks who you are, asks about your identity, says hello, or asks a general question unrelated to the document, DO NOT USE THE CONTEXT AND DO NOT GENERATE ANY HTML TAGS.\n"
                        "2) Simply answer them normally and completely ignore all citation formatting!\n\n"
                        "If the question explicitly pertains to Albanian broadcast laws, you MUST use the retrieved Context provided below to answer accurately.\n"
                        "CITATION RULE FOR LEGAL QUESTIONS ONLY:\n"
                        "You MUST merge all your citations into EXACTLY ONE single tag at the very bottom of your entire response. DO NOT create multiple buttons!\n"
                        "Use this EXACT HTML details tag structure strictly ONCE at the end:\n"
                        "<br><details class='citation'><summary>📌 Source (Pages X)</summary>LITERAL ENGLISH QUOTE 1<br><br>LITERAL ENGLISH QUOTE 2</details>\n"
                        "Replace 'Pages X' with the exact page numbers from the context metadata.\n"
                        "Do NOT translate the English quotes! You MUST literally copy-paste the exact 1-to-1 English wording from the Context snippets right into the details tag! Keep it to exactly one badge beneath the final text.\n"
                        "MATHEMATICAL BAN: If you do not have any exact English quotes from the AMA document to put inside the details tag, YOU MUST NEVER GENERATE THE `<details>` TAG! Empty citation boxes are strictly banned!\n\n"
                        "Context snippets from the AMA Law:\n{context}"
                    )
                    
                    try:
                        docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                        context_texts = []
                        for doc in docs:
                            # PyPDFLoader usually stores the exact page int in doc.metadata['page']. PyPDF pages are 0-indexed so we add 1.
                            page_num = doc.metadata.get('page', 0) + 1
                            context_texts.append(f"[Snippet from Page {page_num}]:\n{doc.page_content}")
                            
                        context_text = "\n\n".join(context_texts)
                        
                        formatted_system = system_prompt.format(context=context_text)
                        
                        response = llm.invoke([
                            ("system", formatted_system),
                            ("human", prompt)
                        ])
                        answer = response.content
                        
                        placeholder.markdown(answer, unsafe_allow_html=True)
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        placeholder.error(f"Failed to fetch an answer: {e}")
    else:
        st.error(f"Could not find the document at `{pdf_path}`. Ensure it exists.")
