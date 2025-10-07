import streamlit as st
from src.rag_pipeline import RagPipeline

def run_app():
    st.set_page_config(page_title="CodeGen Agent", page_icon="💻", layout="wide")

    st.markdown(
        """
        <style>
        .stApp { border: 10px solid #014BB6; padding: 10px; margin: 15px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo = "helpers/cellula.jpg"
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image(logo, width=500)
        st.markdown("<h1>💻 Code Generation Agent 💻</h1>", unsafe_allow_html=True)

    st.markdown("---")

    query = st.text_area("✍️ Enter function signature and docstring:", height=250)

    if st.button("🚀 Generate Code"):
        if query.strip():
            with st.spinner("Generating code... Please wait."):
                rag = RagPipeline()
                generated_code, retrieved_examples = rag.generate_code(query)

            st.success("✅ Code generated successfully!")

            st.markdown("### 📚 Retrieved Examples (FAISS)")
            for idx, (p, s) in enumerate(retrieved_examples, 1):
                with st.expander(f"Example {idx}"):
                    st.markdown("**Prompt:**")
                    st.code(p, language="python")
                    st.markdown("**Solution:**")
                    st.code(s, language="python")

            st.markdown("---")
            st.markdown("### 🧑‍💻 Generated Code")
            if "```" in generated_code:
                parts = generated_code.split("```")
                before = parts[0].strip()
                code_block = parts[1].replace("python", "", 1).strip()
                after = parts[2].strip() if len(parts) > 2 else ""
                if before: st.write(before)
                if code_block: st.code(code_block, language="python")
                if after: st.write(after)
            else:
                st.code(generated_code, language="python")

if __name__ == "__main__":
    run_app()
