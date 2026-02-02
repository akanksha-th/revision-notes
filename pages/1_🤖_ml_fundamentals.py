import streamlit as st

st.title("🤖 Machine Learning Basics")

with st.sidebar:
    st.header("🧩 ML Concepts")
    with st.expander("📌 Types of ML", expanded=True):
        st.markdown("""
        - Supervised Learning  
        - Unsupervised Learning  
        - Reinforcement Learning  
        """)

    with st.expander("🧠 Bias–Variance Tradeoff"):
        st.markdown("- Underfitting\n- Overfitting")

    with st.expander("📏 Model Evaluation"):
        st.markdown("- Accuracy\n- Precision\n- Recall\n- F1 Score")

st.success("🚀 Clean concepts = strong ML foundations")
