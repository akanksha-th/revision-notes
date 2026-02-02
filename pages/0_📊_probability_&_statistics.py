import streamlit as st

st.title("📊 Probability & Statistics")

with st.sidebar:
    st.header("Content")
    with st.expander("🎲 Random Variables", expanded=True):
        # st.page_link("pages/0_📊_probability_&_statistics.py", label="Definition")
        st.markdown("- Discrete vs Continuous\n- PMF / PDF")

    with st.expander("📐 Expectation & Variance"):
        st.markdown("- Mean\n- Variance\n- Standard Deviation")

    with st.expander("📈 Distributions"):
        st.markdown("- Bernoulli\n- Binomial\n- Normal\n- Poisson")

    with st.expander("🧠 Bayes’ Theorem"):
        st.markdown("- Prior\n- Likelihood\n- Posterior")

    with st.expander("🧪 Hypothesis Testing"):
        st.markdown("- Null Hypothesis\n- p-value\n- Type I & II errors")


st.divider()
st.markdown("""
**Always prefer reasoning over formula.**
            
---
            
##### Absolute Foundations

**Probability**: it is actually the long-run frequency of the event.   
Think, `P(heads) = 0.5`   
It actually means, if we repeat an _experiment_ many times, ~50% of the outcomes will be heads.  
            
**Sample Space**: all possible outcomes of an experiment.   
Example: if the event is to toss a coin once, `sample space = {H, T}`   
Toss twice, `sample space = {HH, HT, TH, TT}`    
            
**Event**: an event is a subset of the sample space.   
Example:   
        - `Event A = at least one head`   
        - `Event B = exactly one head`   
            
**Mutually Exclusive vs Independent**: 
        - **Mutually Exclusive**: two events cannot happen together.   
            - Example: let' say a die is rolled, `Event A = die shows 1` and `Event B = die shows 5`. Clearly, if A happens, B cannot. These are mutually exclusive events.   
        - **Independent Events**: two events do NOT affect each other - they can occur simultaneously.   
            - Example: let's say a die is rolled and a coin is tossed simultaneously, `Event A = getting heads`, `Event B = die shows a 2`. Coin result doesn’t change die outcome.   
Mutually exclusive events are NOT independent.
            
---
            
##### Basic Statistics
            
**
""")