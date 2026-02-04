import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Probability Theory Workbook",
    page_icon="🎲",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("📚 Probability Workbook")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Jump to Section:",
    [
        "🏠 Home",
        "🎯 Sample Spaces & Events",
        "📊 Probability Measures",
        "🧮 Counting & Combinatorics",
        "🔀 Conditional Probability",
        "🎲 Independence",
        "📈 Discrete Random Variables",
        "📉 Continuous Random Variables",
        "🔗 Joint Distributions",
        "💻 Interactive Practice",
        "📋 Quick Reference"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Each section includes intuition, examples, and practice problems!")

# Main Content
if section == "🏠 Home":
    st.title("🎲 PROBABILITY THEORY WORKBOOK")
    st.subheader("Complete Guide: Foundations to Advanced Applications")
    
    st.markdown("""
    ### Welcome! 👋
    
    This comprehensive workbook covers probability theory from the ground up, with emphasis on:
    - **Clear Intuitions**: Understand the "why" before the "how"
    - **Extensive Examples**: Real-world applications and detailed walkthroughs
    - **Practice Problems**: Test your understanding with progressive exercises
    - **Interactive Tools**: Visualize concepts and run calculations
    
    ---
    
    #### 📚 What's Covered:
    
    **Part 1: Foundations**
    - Sample spaces and events
    - Probability axioms and measures
    - Counting techniques and combinatorics
    
    **Part 2: Advanced Concepts**
    - Conditional probability and Bayes' Rule
    - Independence of events
    - Random variables (discrete & continuous)
    
    **Part 3: Distributions**
    - Common discrete distributions (Binomial, Poisson, Geometric, etc.)
    - Common continuous distributions (Normal, Exponential, Gamma, etc.)
    - Joint and conditional distributions
    
    **Part 4: Applications**
    - Expected value and variance
    - Moment-generating functions
    - Central Limit Theorem
    - Sampling distributions
    
    ---
    
    ### 🎯 Learning Approach
    
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📖 Read**
        
        Start with clear explanations and intuitive examples
        """)
    
    with col2:
        st.success("""
        **✍️ Practice**
        
        Work through examples and solve practice problems
        """)
    
    with col3:
        st.warning("""
        **🔬 Experiment**
        
        Use interactive tools to visualize and explore
        """)
    
    st.markdown("---")
    st.markdown("👈 **Select a section from the sidebar to begin your journey!**")

elif section == "🎯 Sample Spaces & Events":
    st.title("🎯 Sample Spaces and Events")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Theory", "💡 Examples", "✍️ Practice", "🎮 Interactive"])
    
    with tab1:
        st.header("What is Probability?")
        
        st.info("""
        **INTUITION**: Probability is a mathematical framework for quantifying uncertainty and randomness.
        
        Think of probability as a way to measure how likely something is to happen, using numbers between 0 and 1:
        - **0** = Impossible (will never happen)
        - **1** = Certain (will definitely happen)
        - **0.5** = Equally likely to happen or not happen
        """)
        
        st.markdown("""
        ### Sample Space (Ω)
        
        **Definition**: The set of ALL possible outcomes of a random experiment.
        
        **Intuition**: Before you do an experiment, the sample space is like listing everything that *could* happen.
        
        **Notation**: We use Ω (omega) to denote the sample space.
        
        #### Examples of Sample Spaces:
        
        1. **Coin Flip**: Ω = {H, T}
        2. **Die Roll**: Ω = {1, 2, 3, 4, 5, 6}
        3. **Two Coin Flips**: Ω = {HH, HT, TH, TT}
        4. **Waiting Time**: Ω = [0, ∞) (all non-negative real numbers)
        
        ---
        
        ### Events
        
        **Definition**: An event is a subset of the sample space - a collection of outcomes we're interested in.
        
        **Intuition**: An event is like asking a question about the experiment: "Did this specific thing happen?"
        
        #### Example Events:
        
        For rolling a die (Ω = {1, 2, 3, 4, 5, 6}):
        - Event A = "roll an even number" = {2, 4, 6}
        - Event B = "roll greater than 4" = {5, 6}
        - Event C = "roll a 3" = {3}
        
        ---
        
        ### Set Operations on Events
        
        Events are sets, so we can combine them using set operations:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Union (A ∪ B)**
            - "A OR B happens"
            - All outcomes in A, in B, or in both
            - Example: A = {2, 4, 6}, B = {5, 6}
            - A ∪ B = {2, 4, 5, 6}
            
            **Complement (A^c)**
            - "A does NOT happen"
            - All outcomes NOT in A
            - Example: A = {2, 4, 6}
            - A^c = {1, 3, 5}
            """)
        
        with col2:
            st.markdown("""
            **Intersection (A ∩ B)**
            - "BOTH A AND B happen"
            - Only outcomes in both A and B
            - Example: A = {2, 4, 6}, B = {5, 6}
            - A ∩ B = {6}
            
            **Empty Set (∅)**
            - The impossible event
            - Contains no outcomes
            - Example: A = {1, 2}, B = {5, 6}
            - A ∩ B = ∅
            """)
        
        st.success("""
        **Key Insight**: 
        - **Disjoint (Mutually Exclusive) Events**: A ∩ B = ∅
        - This means A and B cannot both happen at the same time
        - Example: Getting "heads" and "tails" on a single coin flip
        """)
    
    with tab2:
        st.header("💡 Detailed Examples")
        
        st.subheader("Example 1: Two Coin Flips")
        
        st.markdown("""
        **Experiment**: Flip a fair coin twice and record the sequence.
        
        **Sample Space**: 
        ```
        Ω = {HH, HT, TH, TT}
        ```
        
        **Define Events**:
        - A = "at least one heads" = {HH, HT, TH}
        - B = "exactly one heads" = {HT, TH}
        - C = "no heads" = {TT}
        
        **Set Operations**:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code("""
A ∪ B = {HH, HT, TH}
(at least one heads OR exactly one heads)

A ∩ B = {HT, TH}
(at least one heads AND exactly one heads)
            """)
        
        with col2:
            st.code("""
A^c = {TT}
(NOT at least one heads = no heads)

A ∩ C = ∅
(can't have at least one heads AND no heads)
            """)
        
        st.markdown("---")
        
        st.subheader("Example 2: Rolling Two Dice")
        
        st.markdown("""
        **Experiment**: Roll two fair six-sided dice and record the outcomes.
        
        **Sample Space**: 
        ```
        Ω = {(1,1), (1,2), ..., (6,6)}  [36 total outcomes]
        ```
        
        **Define Events**:
        - A = "sum is 7" = {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)}
        - B = "first die is 3" = {(3,1), (3,2), (3,3), (3,4), (3,5), (3,6)}
        - C = "both dice show same number" = {(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)}
        
        **Analysis**:
        """)
        
        st.code("""
A ∩ B = {(3,4)}  (sum is 7 AND first die is 3)

A ∩ C = ∅  (sum is 7 AND both dice same - impossible!)

B ∩ C = {(3,3)}  (first die is 3 AND both dice same)

|A| = 6 outcomes
|B| = 6 outcomes
|C| = 6 outcomes
        """)
        
        st.markdown("---")
        
        st.subheader("Example 3: Continuous Sample Space")
        
        st.markdown("""
        **Experiment**: Measure the time (in minutes) a car waits at a red light.
        
        **Sample Space**: 
        ```
        Ω = [0, ∞)  (all non-negative real numbers)
        ```
        
        **Why infinite?** In theory, there's no upper limit on how long you might wait!
        
        **Define Events**:
        - A = "wait less than 2 minutes" = [0, 2)
        - B = "wait between 1 and 3 minutes" = [1, 3]
        - C = "wait more than 2.5 minutes" = (2.5, ∞)
        
        **Set Operations**:
        """)
        
        st.code("""
A ∩ B = [1, 2)  (overlap between [0,2) and [1,3])

B ∩ C = (2.5, 3]  (overlap between [1,3] and (2.5,∞))

A ∪ C = [0, 2) ∪ (2.5, ∞)  (less than 2 OR more than 2.5)
        """)
    
    with tab3:
        st.header("✍️ Practice Problems")
        
        st.subheader("Problem Set 1: Basic Concepts")
        
        st.markdown("""
        **Problem 1**: For the experiment of tossing a coin three times:
        
        a) Write out the complete sample space Ω.
        
        b) Define event A = "get exactly 2 heads". List all outcomes in A.
        
        c) Define event B = "first toss is heads". List all outcomes in B.
        
        d) Find A ∩ B and interpret what this event means.
        
        e) Find A ∪ B and interpret what this event means.
        
        f) Find A^c and interpret what this event means.
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
a) Ω = {HHH, HHT, HTH, HTT, THH, THT, TTH, TTT}
   Total: 8 outcomes

b) A = {HHT, HTH, THH}
   (exactly 2 heads means 2 H's and 1 T)

c) B = {HHH, HHT, HTH, HTT}
   (first toss is H, second and third can be anything)

d) A ∩ B = {HHT, HTH}
   Interpretation: "Exactly 2 heads AND first toss is heads"
   This means: first is H, and exactly one more H in positions 2 or 3

e) A ∪ B = {HHH, HHT, HTH, HTT, THH}
   Interpretation: "Exactly 2 heads OR first toss is heads"

f) A^c = {HHH, HTT, THT, TTH, TTT}
   Interpretation: "NOT exactly 2 heads"
   This means: 0, 1, or 3 heads
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 2**: A standard deck has 52 cards (13 ranks × 4 suits).
        
        You draw one card. Define:
        - A = "card is a heart"
        - B = "card is a face card (J, Q, K)"
        - C = "card is an ace"
        
        a) How many outcomes are in A? In B? In C?
        
        b) Find A ∩ B and describe this event.
        
        c) Find A ∪ C and describe this event.
        
        d) Are events B and C disjoint? Why or why not?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
a) |A| = 13 (13 hearts in deck)
   |B| = 12 (3 face cards × 4 suits)
   |C| = 4 (4 aces in deck)

b) A ∩ B = {Jack of Hearts, Queen of Hearts, King of Hearts}
   "Card is a heart AND a face card"
   |A ∩ B| = 3

c) A ∪ C = All hearts plus all aces
   |A ∪ C| = 13 + 3 = 16
   (13 hearts + 3 non-heart aces, since ace of hearts already counted)

d) Yes, B and C are disjoint!
   B ∩ C = ∅
   Reason: A card cannot be both a face card (J,Q,K) and an ace at the same time
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 3**: Consider the experiment of selecting a random number from the interval [0, 10].
        
        Define:
        - A = [0, 5]
        - B = (3, 8)
        - C = [6, 10]
        
        a) Find A ∩ B
        
        b) Find A ∪ C
        
        c) Find B^c (relative to [0, 10])
        
        d) Are A and C disjoint?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
a) A ∩ B = (3, 5]
   (overlap between [0,5] and (3,8))

b) A ∪ C = [0, 5] ∪ [6, 10] = [0, 10] except (5, 6)
   More formally: A ∪ C = [0, 5] ∪ [6, 10]

c) B^c = [0, 3] ∪ [8, 10]
   (everything in [0,10] that's NOT in (3,8))

d) No, A and C are NOT disjoint
   They don't overlap because 5 < 6
   A ∩ C = ∅
            """)
    
    with tab4:
        st.header("🎮 Interactive Exploration")
        
        st.subheader("Visualize Sample Spaces and Events")
        
        experiment = st.selectbox(
            "Choose an experiment:",
            ["Two Coin Flips", "Roll Two Dice", "Draw from Deck"]
        )
        
        if experiment == "Two Coin Flips":
            st.markdown("""
            **Sample Space**: {HH, HT, TH, TT}
            
            Select events to visualize:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                event_a = st.multiselect(
                    "Event A (blue):",
                    ["HH", "HT", "TH", "TT"],
                    default=["HH", "HT", "TH"]
                )
            
            with col2:
                event_b = st.multiselect(
                    "Event B (red):",
                    ["HH", "HT", "TH", "TT"],
                    default=["HT", "TH"]
                )
            
            st.markdown("---")
            
            # Calculate set operations
            intersection = set(event_a) & set(event_b)
            union = set(event_a) | set(event_b)
            a_only = set(event_a) - set(event_b)
            b_only = set(event_b) - set(event_a)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("A ∩ B (Intersection)", len(intersection))
                st.write(sorted(list(intersection)))
            
            with col2:
                st.metric("A ∪ B (Union)", len(union))
                st.write(sorted(list(union)))
            
            with col3:
                st.metric("A \\ B (A but not B)", len(a_only))
                st.write(sorted(list(a_only)))
            
            if len(intersection) == 0:
                st.success("✅ Events A and B are DISJOINT (mutually exclusive)")
            else:
                st.info("ℹ️ Events A and B are NOT disjoint - they overlap")
        
        elif experiment == "Roll Two Dice":
            st.markdown("**Sample Space**: 36 possible outcomes when rolling two dice")
            
            # Create visualization
            outcomes = [(i, j) for i in range(1, 7) for j in range(1, 7)]
            
            condition = st.selectbox(
                "Highlight outcomes where:",
                ["Sum equals 7", "Sum is even", "Both dice same", "First die is 6", "Sum > 9"]
            )
            
            highlighted = []
            if condition == "Sum equals 7":
                highlighted = [(i, j) for i, j in outcomes if i + j == 7]
            elif condition == "Sum is even":
                highlighted = [(i, j) for i, j in outcomes if (i + j) % 2 == 0]
            elif condition == "Both dice same":
                highlighted = [(i, j) for i, j in outcomes if i == j]
            elif condition == "First die is 6":
                highlighted = [(i, j) for i, j in outcomes if i == 6]
            elif condition == "Sum > 9":
                highlighted = [(i, j) for i, j in outcomes if i + j > 9]
            
            # Create grid visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            
            for i in range(1, 7):
                for j in range(1, 7):
                    if (i, j) in highlighted:
                        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                                   facecolor='lightblue', edgecolor='black'))
                    else:
                        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                                   facecolor='white', edgecolor='gray'))
                    ax.text(i, j, f'({i},{j})', ha='center', va='center', fontsize=8)
            
            ax.set_xlim(0.5, 6.5)
            ax.set_ylim(0.5, 6.5)
            ax.set_xlabel('First Die', fontsize=12)
            ax.set_ylabel('Second Die', fontsize=12)
            ax.set_title(f'Event: {condition}', fontsize=14, fontweight='bold')
            ax.set_xticks(range(1, 7))
            ax.set_yticks(range(1, 7))
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.metric("Number of Favorable Outcomes", len(highlighted))
            st.metric("Total Possible Outcomes", 36)
            st.metric("Fraction", f"{len(highlighted)}/36 = {len(highlighted)/36:.4f}")

elif section == "📊 Probability Measures":
    st.title("📊 Probability Measures")
    
    tab1, tab2, tab3 = st.tabs(["📖 Axioms", "💡 Examples", "✍️ Practice"])
    
    with tab1:
        st.header("The Three Axioms of Probability")
        
        st.info("""
        **INTUITION**: The axioms of probability are the fundamental rules that ALL probabilities must follow.
        Think of them as the "laws of physics" for probability.
        """)
        
        st.markdown("""
        ### Formal Definition
        
        A **probability measure** P is a function that assigns numbers to events such that:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **Axiom 1**
            
            P(Ω) = 1
            
            The probability of the entire sample space is 1.
            
            **Intuition**: SOMETHING must happen!
            """)
        
        with col2:
            st.success("""
            **Axiom 2**
            
            0 ≤ P(A) ≤ 1
            
            For any event A.
            
            **Intuition**: Probabilities are between 0% and 100%
            """)
        
        with col3:
            st.success("""
            **Axiom 3**
            
            If A ∩ B = ∅, then:
            P(A ∪ B) = P(A) + P(B)
            
            **Intuition**: Probabilities of disjoint events add up
            """)
        
        st.markdown("---")
        
        st.header("Important Properties (Derived from Axioms)")
        
        st.markdown("""
        From these three simple axioms, we can prove many useful properties:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **1. Complement Rule**
            ```
            P(A^c) = 1 - P(A)
            ```
            *If prob of rain is 0.3, prob of no rain is 0.7*
            
            **2. Probability of Empty Set**
            ```
            P(∅) = 0
            ```
            *Impossible events have probability 0*
            
            **3. Monotonicity**
            ```
            If A ⊆ B, then P(A) ≤ P(B)
            ```
            *Subset probabilities can't exceed superset*
            """)
        
        with col2:
            st.markdown("""
            **4. Difference Rule**
            ```
            If A ⊆ B, then P(B \\ A) = P(B) - P(A)
            ```
            
            **5. Addition Law (General)**
            ```
            P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
            ```
            *Subtract overlap to avoid double-counting*
            
            **6. Bound**
            ```
            P(A) ≤ 1 for any event A
            ```
            """)
        
        st.markdown("---")
        
        st.header("Equally Likely Outcomes")
        
        st.warning("""
        **Special Case**: When all outcomes in Ω are equally likely, we have a simple formula:
        
        $$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{\# of outcomes in A}}{\text{total \# of outcomes}}$$
        
        This is why counting is so important in probability!
        """)
    
    with tab2:
        st.header("💡 Detailed Examples")
        
        st.subheader("Example 1: Verifying the Axioms")
        
        st.markdown("""
        **Experiment**: Roll a fair six-sided die.
        
        **Sample Space**: Ω = {1, 2, 3, 4, 5, 6}
        
        **Probability Assignment**: Since the die is fair, each outcome is equally likely:
        ```
        P({1}) = P({2}) = P({3}) = P({4}) = P({5}) = P({6}) = 1/6
        ```
        
        **Verify Axiom 1**: P(Ω) = ?
        """)
        
        st.code("""
P(Ω) = P({1, 2, 3, 4, 5, 6})
     = P({1}) + P({2}) + P({3}) + P({4}) + P({5}) + P({6})
     = 1/6 + 1/6 + 1/6 + 1/6 + 1/6 + 1/6
     = 6/6 = 1 ✓
        """)
        
        st.markdown("**Verify Axiom 2**: Are all probabilities between 0 and 1?")
        
        st.code("""
Each outcome has probability 1/6 ≈ 0.167

0 ≤ 1/6 ≤ 1 ✓
        """)
        
        st.markdown("**Verify Axiom 3**: For disjoint events:")
        
        st.code("""
Let A = {2, 4, 6} (even numbers)
Let B = {1, 3, 5} (odd numbers)

Note: A ∩ B = ∅ (disjoint!)

P(A ∪ B) = P({1,2,3,4,5,6}) = 1

P(A) + P(B) = 3/6 + 3/6 = 6/6 = 1 ✓
        """)
        
        st.markdown("---")
        
        st.subheader("Example 2: Using the Addition Law")
        
        st.markdown("""
        **Experiment**: Draw one card from a standard 52-card deck.
        
        Events:
        - A = "card is a heart" → P(A) = 13/52 = 1/4
        - B = "card is a face card (J, Q, K)" → P(B) = 12/52 = 3/13
        
        **Question**: What's P(A ∪ B) = "heart OR face card"?
        
        **Solution**: Use the Addition Law!
        """)
        
        st.code("""
Step 1: Find P(A ∩ B) = "heart AND face card"
        A ∩ B = {J♥, Q♥, K♥}
        P(A ∩ B) = 3/52

Step 2: Apply Addition Law
        P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
                 = 13/52 + 12/52 - 3/52
                 = 22/52
                 = 11/26 ≈ 0.423

**Interpretation**: About 42.3% chance of drawing a heart or face card
        """)
        
        st.warning("""
        **Why subtract P(A ∩ B)?**
        
        If we just added P(A) + P(B), we'd count the 3 heart face cards TWICE:
        - Once in the 13 hearts
        - Once in the 12 face cards
        
        Subtracting P(A ∩ B) corrects for this double-counting!
        """)
        
        st.markdown("---")
        
        st.subheader("Example 3: Complement Rule in Action")
        
        st.markdown("""
        **Problem**: In a game, you roll two dice. You win if the sum is 7 or 11.
        What's the probability of winning?
        
        **Direct Approach**:
        """)
        
        st.code("""
Sum = 7: {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)} → 6 outcomes
Sum = 11: {(5,6), (6,5)} → 2 outcomes

Total favorable: 8 outcomes
Total possible: 36 outcomes

P(win) = 8/36 = 2/9 ≈ 0.222
        """)
        
        st.markdown("**Using Complement** (sometimes easier!):")
        
        st.code("""
P(lose) = P(sum is NOT 7 or 11)
        = 1 - P(win)
        = 1 - 2/9
        = 7/9 ≈ 0.778

So P(win) = 1 - P(lose) = 1 - 7/9 = 2/9
        """)
    
    with tab3:
        st.header("✍️ Practice Problems")
        
        st.markdown("""
        **Problem 1**: A box contains 5 red balls, 3 blue balls, and 2 green balls.
        You randomly select one ball.
        
        a) What is P(red)?
        
        b) What is P(not red)?
        
        c) What is P(red or blue)?
        
        d) Verify that P(red) + P(blue) + P(green) = 1
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Total balls = 5 + 3 + 2 = 10

a) P(red) = 5/10 = 1/2 = 0.5

b) P(not red) = P(red^c) = 1 - P(red) = 1 - 1/2 = 1/2 = 0.5
   OR: P(not red) = (3 + 2)/10 = 5/10 = 1/2

c) P(red or blue) = P(red ∪ blue)
                   = P(red) + P(blue)  [disjoint events!]
                   = 5/10 + 3/10
                   = 8/10 = 4/5 = 0.8

d) P(red) + P(blue) + P(green)
   = 5/10 + 3/10 + 2/10
   = 10/10 = 1 ✓
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 2**: Roll a fair die twice. Find:
        
        a) P(both rolls are even)
        
        b) P(sum of rolls is 8)
        
        c) P(at least one 6)
        
        d) P(rolls are different)
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Total outcomes = 6 × 6 = 36

a) P(both even) = P(first even) × P(second even)
                = (3/6) × (3/6) = 9/36 = 1/4

   OR count: {(2,2), (2,4), (2,6), (4,2), (4,4), (4,6), 
              (6,2), (6,4), (6,6)} = 9 outcomes
   P = 9/36 = 1/4

b) Sum = 8: {(2,6), (3,5), (4,4), (5,3), (6,2)} = 5 outcomes
   P(sum is 8) = 5/36

c) **Using Complement** (easier!):
   P(at least one 6) = 1 - P(no 6's)
                     = 1 - P(first not 6 AND second not 6)
                     = 1 - (5/6) × (5/6)
                     = 1 - 25/36
                     = 11/36

d) P(different) = 1 - P(same)
   P(same) = {(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)} = 6/36
   P(different) = 1 - 6/36 = 30/36 = 5/6
            """)

elif section == "🧮 Counting & Combinatorics":
    st.title("🧮 Counting Techniques & Combinatorics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Theory", "💡 Examples", "✍️ Practice", "🧮 Calculator"])
    
    with tab1:
        st.header("Why Counting Matters in Probability")
        
        st.info("""
        **INTUITION**: When outcomes are equally likely, probability reduces to counting:
        
        $$P(A) = \\frac{\\text{# outcomes in A}}{\\text{# total outcomes}}$$
        
        So we need efficient ways to count large numbers of outcomes!
        """)
        
        st.markdown("---")
        
        st.header("1. Multiplication Principle")
        
        st.success("""
        **Rule**: If you perform k experiments in sequence, where:
        - Experiment 1 has n₁ outcomes
        - Experiment 2 has n₂ outcomes
        - ...
        - Experiment k has nₖ outcomes
        
        Then the total number of outcomes for all k experiments is:
        
        **n₁ × n₂ × ... × nₖ**
        """)
        
        st.markdown("""
        **Example**: Restaurant menu
        - 3 appetizers
        - 5 main courses  
        - 4 desserts
        
        Total meals = 3 × 5 × 4 = **60 different complete meals**
        """)
        
        st.markdown("---")
        
        st.header("2. Permutations (Order Matters)")
        
        st.warning("""
        **Definition**: A permutation is an **ordered** arrangement of objects.
        
        **Intuition**: Think "arrangement where position matters"
        - ABC is different from BAC
        - 1st place ≠ 2nd place ≠ 3rd place
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Formula 1**: Permutations of n distinct objects
            
            $$P(n) = n!$$
            
            where n! = n × (n-1) × (n-2) × ... × 2 × 1
            
            **Example**: Arrange 4 books on a shelf
            - 1st position: 4 choices
            - 2nd position: 3 choices
            - 3rd position: 2 choices
            - 4th position: 1 choice
            - Total: 4! = 4×3×2×1 = **24 ways**
            """)
        
        with col2:
            st.markdown("""
            **Formula 2**: Select and arrange r objects from n distinct objects
            
            $$P(n,r) = \\frac{n!}{(n-r)!}$$
            
            **Example**: How many ways to assign gold, silver, bronze medals among 8 athletes?
            
            $$P(8,3) = \\frac{8!}{5!} = 8 \\times 7 \\times 6 = 336$$
            """)
        
        st.markdown("---")
        
        st.header("3. Combinations (Order Doesn't Matter)")
        
        st.success("""
        **Definition**: A combination is an **unordered** selection of objects.
        
        **Intuition**: Think "choose a group where order doesn't matter"
        - {A, B, C} is the same as {C, A, B}
        - Committee of 3 people - doesn't matter who you picked first
        """)
        
        st.latex(r"\binom{n}{r} = \frac{n!}{r!(n-r)!}")
        
        st.markdown("""
        Read as "n choose r" - the number of ways to choose r objects from n objects.
        
        **Example**: Choose 3 students from a class of 10 for a study group
        
        $$\\binom{10}{3} = \\frac{10!}{3!7!} = \\frac{10 \\times 9 \\times 8}{3 \\times 2 \\times 1} = 120$$
        """)
        
        st.markdown("---")
        
        st.header("Permutations vs Combinations")
        
        comparison_data = {
            'Aspect': ['Order', 'Formula', 'Example', 'Result'],
            'Permutations': [
                'MATTERS',
                'n!/(n-r)!',
                '3 people for President, VP, Treasurer from 10',
                'P(10,3) = 720'
            ],
            'Combinations': [
                'DOESN\'T MATTER',
                'n!/(r!(n-r)!)',
                '3 people for a committee from 10',
                'C(10,3) = 120'
            ]
        }
        
        st.table(pd.DataFrame(comparison_data))
        
        st.info("""
        **Key Relationship**: 
        
        $$\\binom{n}{r} = \\frac{P(n,r)}{r!}$$
        
        Why? Permutations count each combination r! times (all orderings).
        Divide by r! to remove ordering and get combinations.
        """)
    
    with tab2:
        st.header("💡 Detailed Examples")
        
        st.subheader("Example 1: License Plates")
        
        st.markdown("""
        **Problem**: A license plate has:
        - 3 letters (A-Z)
        - 4 digits (0-9)
        
        How many different license plates are possible?
        
        **Solution using Multiplication Principle**:
        """)
        
        st.code("""
Each letter: 26 choices
Each digit: 10 choices

Total = 26 × 26 × 26 × 10 × 10 × 10 × 10
      = 26³ × 10⁴
      = 17,576 × 10,000
      = 175,760,000

Over 175 million possible license plates!
        """)
        
        st.markdown("---")
        
        st.subheader("Example 2: Passwords (Order Matters!)")
        
        st.markdown("""
        **Problem**: Create a 4-character password using the letters {A, B, C, D, E}
        
        a) With repetition allowed
        b) Without repetition
        
        **Solutions**:
        """)
        
        st.code("""
a) WITH repetition:
   - 1st character: 5 choices
   - 2nd character: 5 choices (can reuse)
   - 3rd character: 5 choices
   - 4th character: 5 choices
   
   Total = 5 × 5 × 5 × 5 = 5⁴ = 625 passwords

b) WITHOUT repetition:
   - 1st character: 5 choices
   - 2nd character: 4 choices (can't reuse)
   - 3rd character: 3 choices
   - 4th character: 2 choices
   
   Total = 5 × 4 × 3 × 2 = P(5,4) = 5!/1! = 120 passwords
        """)
        
        st.markdown("---")
        
        st.subheader("Example 3: Poker Hands (Combinations)")
        
        st.markdown("""
        **Problem**: A poker hand is 5 cards from a 52-card deck.
        
        a) How many different poker hands exist?
        b) How many hands have all hearts?
        c) What's P(all hearts)?
        
        **Solutions**:
        """)
        
        st.code("""
a) Total poker hands = C(52, 5)
                      = 52!/(5! × 47!)
                      = (52 × 51 × 50 × 49 × 48)/(5 × 4 × 3 × 2 × 1)
                      = 2,598,960 hands

b) Hands with all hearts = C(13, 5)
   (choosing 5 cards from 13 hearts)
                          = 13!/(5! × 8!)
                          = 1,287 hands

c) P(all hearts) = (# all-heart hands)/(# total hands)
                 = 1,287/2,598,960
                 = 0.000495
                 ≈ 0.05%
                 
Very rare! About 1 in 2000 hands.
        """)
        
        st.markdown("---")
        
        st.subheader("Example 4: Committees (Combinations)")
        
        st.markdown("""
        **Problem**: From 10 people (6 men, 4 women), form a 4-person committee.
        
        a) How many committees are possible?
        b) How many committees have exactly 2 women?
        c) How many committees have at least 1 woman?
        
        **Solutions**:
        """)
        
        st.code("""
a) Total committees = C(10, 4)
                    = 10!/(4! × 6!)
                    = 210 committees

b) Exactly 2 women means 2 women AND 2 men:
   
   # ways to choose 2 women from 4 = C(4, 2) = 6
   # ways to choose 2 men from 6 = C(6, 2) = 15
   
   By Multiplication Principle:
   Total = 6 × 15 = 90 committees

c) At least 1 woman = 1, 2, 3, or 4 women
   
   **Using Complement** (easier!):
   P(at least 1 woman) = 1 - P(no women)
   
   P(no women) = P(all men) = C(6,4)/C(10,4)
   
   # all-men committees = C(6, 4) = 15
   
   So: committees with at least 1 woman = 210 - 15 = 195
        """)
        
        st.markdown("---")
        
        st.subheader("Example 5: Distinguishing Permutations vs Combinations")
        
        st.markdown("""
        **Scenario**: You have 5 friends: Alice, Bob, Carol, David, Eve
        
        Compare these situations:
        """)
        
        comparison = {
            'Question': [
                'Choose 3 friends to invite to dinner',
                'Assign President, VP, Secretary roles to 3 friends',
                'Arrange 3 friends in a line for a photo'
            ],
            'Order Matters?': ['NO', 'YES', 'YES'],
            'Count Method': ['Combination', 'Permutation', 'Permutation'],
            'Formula': ['C(5,3)', 'P(5,3)', 'P(5,3)'],
            'Result': [
                '5!/(3!×2!) = 10',
                '5!/ 2! = 60',
                '5!/2! = 60'
            ]
        }
        
        st.table(pd.DataFrame(comparison))
        
        st.info("""
        **Key Insight**: 
        - Dinner invitation: {Alice, Bob, Carol} = {Carol, Alice, Bob} → Combination
        - Officer roles: (Alice=Pres, Bob=VP, Carol=Sec) ≠ (Bob=Pres, Alice=VP, Carol=Sec) → Permutation
        - Photo lineup: [Alice][Bob][Carol] ≠ [Carol][Bob][Alice] → Permutation
        """)
    
    with tab3:
        st.header("✍️ Practice Problems")
        
        st.markdown("""
        **Problem 1**: A restaurant offers:
        - 4 types of bread
        - 6 types of meat
        - 8 types of toppings
        
        How many different sandwiches can you make if you choose one of each?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Using Multiplication Principle:

Total = (# breads) × (# meats) × (# toppings)
      = 4 × 6 × 8
      = 192 different sandwiches
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 2**: How many ways can you arrange the letters in "MATH"?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
We have 4 distinct letters: M, A, T, H

# permutations = 4! = 4 × 3 × 2 × 1 = 24 ways

Some examples: MATH, MAHT, MTHA, MTAH, MHAT, MAHM, 
               AMTH, AHMT, ATMH, ATHM, AHMT, AHTM, ...
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 3**: From a deck of 52 cards:
        
        a) How many 5-card hands contain exactly 3 aces?
        
        b) What's the probability of getting exactly 3 aces in a 5-card hand?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
a) Exactly 3 aces means:
   - Choose 3 aces from 4 aces: C(4,3) = 4
   - Choose 2 non-aces from 48 non-aces: C(48,2) = 1,128
   
   Total hands with exactly 3 aces = 4 × 1,128 = 4,512

b) P(exactly 3 aces) = 4,512 / C(52,5)
                      = 4,512 / 2,598,960
                      = 0.001736
                      ≈ 0.17%
   
   About 1 in 575 hands
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 4**: 10 students, need to select:
        
        a) 3 students for a presentation (order doesn't matter)
        
        b) President, VP, and Secretary (order matters)
        
        How many ways for each?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
a) Presentation group (unordered):
   C(10,3) = 10!/(3!×7!)
           = (10 × 9 × 8)/(3 × 2 × 1)
           = 120 ways

b) Officer positions (ordered):
   P(10,3) = 10!/ 7!
           = 10 × 9 × 8
           = 720 ways
   
Note: P(10,3) = 3! × C(10,3) = 6 × 120 = 720
(Each combination can be arranged in 3! = 6 ways)
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 5 (Challenge)**: A committee of 5 people must include:
        - At least 2 women (from 6 women available)
        - At least 2 men (from 8 men available)
        
        How many such committees are possible?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Break into cases by # of women:

Case 1: 2 women, 3 men
  C(6,2) × C(8,3) = 15 × 56 = 840

Case 2: 3 women, 2 men
  C(6,3) × C(8,2) = 20 × 28 = 560

Total = 840 + 560 = 1,400 committees

Alternative check:
Total committees = C(14,5) = 2,002
Subtract invalid ones:
  - All women or 4 women: C(6,5) + C(6,4)×C(8,1) = 6 + 120 = 126
  - All men or 4 men: C(8,5) + C(8,4)×C(6,1) = 56 + 420 = 476
  - 1 woman, 4 men: C(6,1)×C(8,4) = 6 × 70 = 420 (included above)
  - 4 women, 1 man: C(6,4)×C(8,1) = 15 × 8 = 120 (included above)
  
Valid = 2,002 - (126 + 476) = 1,400 ✓
            """)
    
    with tab4:
        st.header("🧮 Combinatorics Calculator")
        
        calc_type = st.selectbox(
            "Choose calculation type:",
            ["Factorial (n!)", "Permutations P(n,r)", "Combinations C(n,r)"]
        )
        
        if calc_type == "Factorial (n!)":
            n = st.number_input("Enter n:", min_value=0, max_value=20, value=5)
            
            if st.button("Calculate"):
                import math
                result = math.factorial(n)
                st.success(f"**{n}! = {result:,}**")
                
                st.write("Calculation:")
                if n <= 10:
                    steps = " × ".join([str(i) for i in range(n, 0, -1)])
                    st.write(f"{n}! = {steps} = {result:,}")
        
        elif calc_type == "Permutations P(n,r)":
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("Total objects (n):", min_value=0, max_value=20, value=10)
            with col2:
                r = st.number_input("Select (r):", min_value=0, max_value=20, value=3)
            
            if st.button("Calculate"):
                if r > n:
                    st.error("r cannot be greater than n!")
                else:
                    import math
                    result = math.perm(n, r)
                    st.success(f"**P({n},{r}) = {result:,}**")
                    
                    st.write("Formula:")
                    st.latex(f"P({n},{r}) = \\frac{{{n}!}}{{({n}-{r})!}} = \\frac{{{n}!}}{{{n-r}!}}")
                    
                    st.write("Calculation:")
                    numerator = math.factorial(n)
                    denominator = math.factorial(n-r)
                    st.write(f"= {numerator:,} / {denominator:,} = {result:,}")
        
        else:  # Combinations
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("Total objects (n):", min_value=0, max_value=20, value=10)
            with col2:
                r = st.number_input("Select (r):", min_value=0, max_value=20, value=3)
            
            if st.button("Calculate"):
                if r > n:
                    st.error("r cannot be greater than n!")
                else:
                    result = int(comb(n, r))
                    st.success(f"**C({n},{r}) = {result:,}**")
                    
                    st.write("Formula:")
                    st.latex(f"\\binom{{{n}}}{{{r}}} = \\frac{{{n}!}}{{{r}!({n}-{r})!}}")
                    
                    st.write("Calculation:")
                    import math
                    numerator = math.factorial(n)
                    denominator = math.factorial(r) * math.factorial(n-r)
                    st.write(f"= {numerator:,} / {denominator:,} = {result:,}")

elif section == "🔀 Conditional Probability":
    st.title("🔀 Conditional Probability & Bayes' Rule")
    
    tab1, tab2, tab3 = st.tabs(["📖 Theory", "💡 Examples", "✍️ Practice"])
    
    with tab1:
        st.header("What is Conditional Probability?")
        
        st.info("""
        **INTUITION**: Conditional probability asks: "Given that we know something happened, 
        what's the probability of something else?"
        
        It's about **updating our beliefs** based on **new information**.
        """)
        
        st.markdown("""
        ### Notation and Formula
        
        **P(A|B)** reads as "probability of A **given** B"
        """)
        
        st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
        
        st.markdown("""
        where P(B) > 0
        
        **Intuition behind the formula**:
        - We know B happened, so B becomes our "new universe"
        - We want the fraction of B that is also in A
        - Divide the overlap (A ∩ B) by the total probability of B
        """)
        
        st.markdown("---")
        
        st.header("The Multiplication Law")
        
        st.success("""
        Rearranging the conditional probability formula gives us:
        
        **P(A ∩ B) = P(A|B) × P(B) = P(B|A) × P(A)**
        
        **Intuition**: The probability that both A and B happen equals:
        - The probability B happens, times
        - The probability A happens given B already happened
        """)
        
        st.markdown("---")
        
        st.header("Law of Total Probability")
        
        st.warning("""
        **Setup**: If B₁, B₂, ..., Bₙ partition the sample space (divide it into non-overlapping pieces), then:
        
        $$P(A) = P(A|B_1)P(B_1) + P(A|B_2)P(B_2) + ... + P(A|B_n)P(B_n)$$
        
        **Intuition**: To find P(A), consider all the ways A can happen (through each Bᵢ),
        and add up their probabilities.
        """)
        
        st.markdown("---")
        
        st.header("Bayes' Rule")
        
        st.success("""
        **Bayes' Rule** lets us "reverse" conditional probabilities:
        
        $$P(B|A) = \\frac{P(A|B) \\cdot P(B)}{P(A)}$$
        
        Or with Law of Total Probability:
        
        $$P(B_i|A) = \\frac{P(A|B_i) \\cdot P(B_i)}{\\sum_j P(A|B_j) \\cdot P(B_j)}$$
        """)
        
        st.markdown("""
        **Components**:
        - **P(Bᵢ)** = Prior probability (what we knew before)
        - **P(A|Bᵢ)** = Likelihood (how likely is our observation given Bᵢ)
        - **P(Bᵢ|A)** = Posterior probability (what we know after observing A)
        """)
        
        st.info("""
        **When to use Bayes' Rule**:
        - You know P(A|B) but want P(B|A)
        - Medical testing: Know P(positive test | disease), want P(disease | positive test)
        - Spam filtering: Know P(word | spam), want P(spam | word)
        """)
    
    with tab2:
        st.header("💡 Detailed Examples")
        
        st.subheader("Example 1: Drawing Cards")
        
        st.markdown("""
        **Setup**: Draw one card from a standard deck.
        
        **Events**:
        - A = "card is a King"
        - B = "card is a face card (J, Q, K)"
        - C = "card is a heart"
        
        **Find**: P(King | face card) and P(heart | King)
        """)
        
        st.code("""
Solution for P(King | face card):

P(A|B) = P(A ∩ B) / P(B)

P(B) = P(face card) = 12/52 = 3/13
P(A ∩ B) = P(King and face card) = P(King) = 4/52 = 1/13

P(A|B) = (1/13) / (3/13) = 1/3

Interpretation: If we know it's a face card, there's a 1/3 chance it's a King
(makes sense: face cards are J, Q, K, so 1 out of 3 is King!)

---

Solution for P(heart | King):

P(C|A) = P(C ∩ A) / P(A)

P(A) = P(King) = 4/52
P(C ∩ A) = P(heart and King) = 1/52 (only King of hearts)

P(C|A) = (1/52) / (4/52) = 1/4

Interpretation: If we know it's a King, there's a 1/4 chance it's a heart
(makes sense: 4 Kings total, 1 is a heart!)
        """)
        
        st.markdown("---")
        
        st.subheader("Example 2: Medical Testing (Bayes' Rule)")
        
        st.markdown("""
        **Problem**: A disease affects 1% of the population. A test for the disease:
        - Correctly identifies 90% of people WITH the disease (sensitivity)
        - Correctly identifies 95% of people WITHOUT the disease (specificity)
        
        **Question**: If someone tests positive, what's the probability they have the disease?
        
        **Solution**:
        """)
        
        st.code("""
Define events:
D = person has disease
T+ = test positive

Given information:
P(D) = 0.01 (1% have disease)
P(D^c) = 0.99 (99% don't have disease)
P(T+|D) = 0.90 (sensitivity)
P(T-|D^c) = 0.95, so P(T+|D^c) = 0.05 (false positive rate)

Want: P(D|T+)

Step 1: Use Bayes' Rule
P(D|T+) = P(T+|D) × P(D) / P(T+)

Step 2: Find P(T+) using Law of Total Probability
P(T+) = P(T+|D)×P(D) + P(T+|D^c)×P(D^c)
      = (0.90)(0.01) + (0.05)(0.99)
      = 0.009 + 0.0495
      = 0.0585

Step 3: Apply Bayes' Rule
P(D|T+) = (0.90 × 0.01) / 0.0585
        = 0.009 / 0.0585
        = 0.1538
        ≈ 15.4%

INTERPRETATION: Even with a positive test, only about 15% actually have the disease!

Why so low? The disease is rare (1%), so most positive tests are false positives
from the 99% of healthy people.
        """)
        
        st.warning("""
        **Key Insight**: This is why understanding conditional probability matters in medicine!
        
        - 90% sensitivity sounds good
        - But with a rare disease (1% prevalence)
        - Most positive tests are actually false positives
        - Only 15.4% of positive tests indicate actual disease
        """)
        
        st.markdown("---")
        
        st.subheader("Example 3: Multi-stage Experiment")
        
        st.markdown("""
        **Problem**: Box 1 has 3 red and 2 blue balls. Box 2 has 2 red and 4 blue balls.
        
        Process:
        1. Roll a fair die
        2. If roll ≤ 4, pick from Box 1; if roll > 4, pick from Box 2
        3. Draw one ball
        
        **Questions**:
        a) What's P(red ball)?
        b) If we drew red, what's P(it came from Box 1)?
        """)
        
        st.code("""
Define events:
B1 = select Box 1
B2 = select Box 2
R = draw red ball

Given:
P(B1) = 4/6 = 2/3 (roll 1,2,3,4)
P(B2) = 2/6 = 1/3 (roll 5,6)
P(R|B1) = 3/5 (3 red out of 5 in Box 1)
P(R|B2) = 2/6 = 1/3 (2 red out of 6 in Box 2)

a) Find P(R) using Law of Total Probability:

P(R) = P(R|B1)×P(B1) + P(R|B2)×P(B2)
     = (3/5)(2/3) + (1/3)(1/3)
     = 2/5 + 1/9
     = 18/45 + 5/45
     = 23/45
     ≈ 0.511

b) Find P(B1|R) using Bayes' Rule:

P(B1|R) = P(R|B1)×P(B1) / P(R)
        = (3/5)(2/3) / (23/45)
        = (2/5) / (23/45)
        = (2/5) × (45/23)
        = 90/115
        = 18/23
        ≈ 0.783

If we drew red, there's about 78% chance it came from Box 1.
        """)
    
    with tab3:
        st.header("✍️ Practice Problems")
        
        st.markdown("""
        **Problem 1**: Roll two fair dice. Given that the sum is greater than 7,
        what's the probability that both dice show the same number?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Define events:
A = "both dice same" = {(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)}
B = "sum > 7"

Find outcomes in B:
Sum = 8: (2,6), (3,5), (4,4), (5,3), (6,2) → 5 outcomes
Sum = 9: (3,6), (4,5), (5,4), (6,3) → 4 outcomes
Sum = 10: (4,6), (5,5), (6,4) → 3 outcomes
Sum = 11: (5,6), (6,5) → 2 outcomes
Sum = 12: (6,6) → 1 outcome
Total in B: 15 outcomes

Find A ∩ B:
"Both same AND sum > 7" = {(4,4), (5,5), (6,6)} → 3 outcomes

P(A|B) = P(A ∩ B) / P(B)
       = (3/36) / (15/36)
       = 3/15
       = 1/5 = 0.2

Answer: 20% chance
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 2**: In a class:
        - 60% are women
        - 30% of women have brown eyes
        - 40% of men have brown eyes
        
        a) What fraction of the class has brown eyes?
        b) If a student has brown eyes, what's the probability they're a woman?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Define events:
W = woman
M = man
B = brown eyes

Given:
P(W) = 0.6
P(M) = 0.4
P(B|W) = 0.3
P(B|M) = 0.4

a) Find P(B) using Law of Total Probability:

P(B) = P(B|W)×P(W) + P(B|M)×P(M)
     = (0.3)(0.6) + (0.4)(0.4)
     = 0.18 + 0.16
     = 0.34

34% of class has brown eyes

b) Find P(W|B) using Bayes' Rule:

P(W|B) = P(B|W)×P(W) / P(B)
       = (0.3)(0.6) / 0.34
       = 0.18 / 0.34
       ≈ 0.529

About 52.9% of brown-eyed students are women
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 3**: Three machines produce bolts:
        - Machine A: 25% of production, 5% defective
        - Machine B: 35% of production, 4% defective
        - Machine C: 40% of production, 2% defective
        
        a) What's the probability a randomly selected bolt is defective?
        b) If a bolt is defective, what's the probability it came from Machine C?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Define events:
A, B, C = bolt from Machine A, B, C
D = defective bolt

Given:
P(A) = 0.25, P(D|A) = 0.05
P(B) = 0.35, P(D|B) = 0.04
P(C) = 0.40, P(D|C) = 0.02

a) Find P(D) using Law of Total Probability:

P(D) = P(D|A)×P(A) + P(D|B)×P(B) + P(D|C)×P(C)
     = (0.05)(0.25) + (0.04)(0.35) + (0.02)(0.40)
     = 0.0125 + 0.014 + 0.008
     = 0.0345

3.45% of bolts are defective

b) Find P(C|D) using Bayes' Rule:

P(C|D) = P(D|C)×P(C) / P(D)
       = (0.02)(0.40) / 0.0345
       = 0.008 / 0.0345
       ≈ 0.232

Only about 23.2% of defective bolts come from Machine C

Why so low? Machine C has the lowest defect rate (2%), 
so even though it produces 40% of bolts, it contributes
proportionally fewer defects.
            """)

elif section == "🎲 Independence":
    st.title("🎲 Independent Events")
    
    tab1, tab2, tab3 = st.tabs(["📖 Theory", "💡 Examples", "✍️ Practice"])
    
    with tab1:
        st.header("What Does Independence Mean?")
        
        st.info("""
        **INTUITION**: Events A and B are independent if knowing one happened doesn't change 
        the probability of the other happening.
        
        **Examples**:
        - ✅ **Independent**: Coin flip 1 and coin flip 2
        - ❌ **Dependent**: Drawing cards without replacement
        - ✅ **Independent**: Rain today and coin flip result
        - ❌ **Dependent**: Rain today and rain tomorrow
        """)
        
        st.markdown("""
        ### Mathematical Definition
        
        Events A and B are **independent** if:
        """)
        
        st.latex(r"P(A \cap B) = P(A) \times P(B)")
        
        st.success("""
        **Equivalent definitions** (any one implies the others):
        1. P(A ∩ B) = P(A) × P(B)
        2. P(A|B) = P(A)
        3. P(B|A) = P(B)
        
        All three say the same thing in different ways!
        """)
        
        st.markdown("---")
        
        st.header("Testing for Independence")
        
        st.code("""
To check if A and B are independent:

Step 1: Calculate P(A), P(B), and P(A ∩ B)

Step 2: Check if P(A ∩ B) = P(A) × P(B)

If YES → Independent
If NO  → Dependent
        """)
        
        st.markdown("---")
        
        st.header("Pairwise vs Mutually Independent")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Pairwise Independent**
            
            Every pair of events is independent:
            - A and B independent
            - A and C independent  
            - B and C independent
            
            But all three together might not be!
            """)
        
        with col2:
            st.markdown("""
            **Mutually Independent**
            
            STRONGER condition - requires:
            - Pairwise independence, AND
            - P(A ∩ B ∩ C) = P(A)×P(B)×P(C)
            - Plus all subset conditions
            
            This is what we usually mean by "independent"
            """)
        
        st.warning("""
        **Important**: Pairwise independence does NOT imply mutual independence!
        
        See Example 2 in the Examples tab for a demonstration.
        """)
    
    with tab2:
        st.header("💡 Detailed Examples")
        
        st.subheader("Example 1: Coin Flips (Independent)")
        
        st.markdown("""
        **Problem**: Flip a fair coin twice.
        
        Events:
        - A = "first flip is heads" = {HH, HT}
        - B = "second flip is heads" = {HH, TH}
        
        **Verify independence**:
        """)
        
        st.code("""
Sample space: Ω = {HH, HT, TH, TT}
Each outcome has probability 1/4

Calculate probabilities:
P(A) = P({HH, HT}) = 2/4 = 1/2
P(B) = P({HH, TH}) = 2/4 = 1/2
P(A ∩ B) = P({HH}) = 1/4

Check independence:
P(A) × P(B) = (1/2) × (1/2) = 1/4
P(A ∩ B) = 1/4

Since P(A ∩ B) = P(A) × P(B), events A and B are INDEPENDENT! ✓

Alternative check using conditional probability:
P(A|B) = P(A ∩ B) / P(B) = (1/4) / (1/2) = 1/2 = P(A) ✓

Interpretation: Knowing the second flip is heads doesn't change 
the probability that the first flip was heads (still 1/2).
        """)
        
        st.markdown("---")
        
        st.subheader("Example 2: Pairwise but Not Mutually Independent")
        
        st.markdown("""
        **Problem**: Flip a fair coin twice.
        
        Events:
        - A = "first flip is H" = {HH, HT}
        - B = "second flip is H" = {HH, TH}
        - C = "exactly one H" = {HT, TH}
        
        **Show**: A, B, C are pairwise independent but NOT mutually independent
        """)
        
        st.code("""
Calculate individual probabilities:
P(A) = 2/4 = 1/2
P(B) = 2/4 = 1/2
P(C) = 2/4 = 1/2

Check pairwise independence:

1) A and B:
   P(A ∩ B) = P({HH}) = 1/4
   P(A) × P(B) = (1/2)(1/2) = 1/4 ✓ Independent

2) A and C:
   P(A ∩ C) = P({HT}) = 1/4
   P(A) × P(C) = (1/2)(1/2) = 1/4 ✓ Independent

3) B and C:
   P(B ∩ C) = P({TH}) = 1/4
   P(B) × P(C) = (1/2)(1/2) = 1/4 ✓ Independent

All pairs are independent! ✓

Check mutual independence:
P(A ∩ B ∩ C) = P(first H AND second H AND exactly one H)
              = P(∅) = 0
              (Impossible to have both H AND exactly one H!)

P(A) × P(B) × P(C) = (1/2)(1/2)(1/2) = 1/8

Since 0 ≠ 1/8, events are NOT mutually independent! ✗

CONCLUSION: Pairwise independent but NOT mutually independent!
        """)
        
        st.markdown("---")
        
        st.subheader("Example 3: Drawing Cards (Dependent)")
        
        st.markdown("""
        **Problem**: Draw 2 cards from a deck WITHOUT replacement.
        
        Events:
        - A = "first card is an ace"
        - B = "second card is an ace"
        
        **Show these are dependent**:
        """)
        
        st.code("""
Calculate probabilities:
P(A) = 4/52 = 1/13

For P(B), use Law of Total Probability:
P(B) = P(B|A)×P(A) + P(B|A^c)×P(A^c)

If first is ace: 3 aces left out of 51 cards
P(B|A) = 3/51 = 1/17

If first not ace: 4 aces left out of 51 cards
P(B|A^c) = 4/51

P(B) = (1/17)(1/13) + (4/51)(12/13)
     = 1/221 + 48/663
     = 3/663 + 48/663
     = 51/663 = 1/13

For P(A ∩ B):
P(A ∩ B) = P(A) × P(B|A)
         = (4/52) × (3/51)
         = 12/2652 = 1/221

Check independence:
P(A) × P(B) = (1/13) × (1/13) = 1/169

But P(A ∩ B) = 1/221

Since 1/221 ≠ 1/169, events are DEPENDENT! ✗

Why? Drawing without replacement means the first draw affects 
the composition of the deck for the second draw.
        """)
        
        st.info("""
        **Key Insight**: With replacement → Independent
        
        If we replaced the first card before drawing the second:
        - P(B|A) = 4/52 (deck restored)
        - P(A ∩ B) = (4/52) × (4/52) = P(A) × P(B) ✓
        - Events would be independent!
        """)
    
    with tab3:
        st.header("✍️ Practice Problems")
        
        st.markdown("""
        **Problem 1**: Roll a fair die twice.
        
        Events:
        - A = "first roll is even"
        - B = "second roll is > 4"
        - C = "sum is 7"
        
        a) Are A and B independent?
        b) Are A and C independent?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Sample space has 36 outcomes: (1,1), (1,2), ..., (6,6)

a) Check if A and B are independent:

A = {(2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
     (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
     (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)}
P(A) = 18/36 = 1/2

B = {(1,5), (1,6), (2,5), (2,6), (3,5), (3,6),
     (4,5), (4,6), (5,5), (5,6), (6,5), (6,6)}
P(B) = 12/36 = 1/3

A ∩ B = {(2,5), (2,6), (4,5), (4,6), (6,5), (6,6)}
P(A ∩ B) = 6/36 = 1/6

Check: P(A) × P(B) = (1/2) × (1/3) = 1/6 = P(A ∩ B) ✓

YES, A and B are independent!
(First roll doesn't affect second roll)

b) Check if A and C are independent:

C = "sum is 7" = {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)}
P(C) = 6/36 = 1/6

A ∩ C = {(2,5), (4,3), (6,1)}
P(A ∩ C) = 3/36 = 1/12

Check: P(A) × P(C) = (1/2) × (1/6) = 1/12 = P(A ∩ C) ✓

YES, A and C are independent!
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 2**: In a class, 40% are seniors and 30% are math majors.
        18% are senior math majors.
        
        Are "being a senior" and "being a math major" independent?
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
Define events:
S = senior
M = math major

Given:
P(S) = 0.40
P(M) = 0.30
P(S ∩ M) = 0.18

Check independence:
P(S) × P(M) = (0.40) × (0.30) = 0.12

But P(S ∩ M) = 0.18

Since 0.18 ≠ 0.12, events are DEPENDENT (NOT independent)

Interpretation: Being a senior is NOT independent of being a math major.
In fact, P(S ∩ M) > P(S) × P(M), suggesting seniors are MORE likely 
to be math majors than would be expected if independent.

Alternative check using conditional probability:
P(M|S) = P(S ∩ M) / P(S) = 0.18 / 0.40 = 0.45

Since P(M|S) = 0.45 ≠ 0.30 = P(M), events are dependent.
(45% of seniors are math majors, higher than overall 30%)
            """)

elif section == "📈 Discrete Random Variables":
    st.title("📈 Discrete Random Variables")
    st.write("Content continues... (additional sections follow the same pattern)")
    st.info("This is a preview. The full workbook continues with detailed coverage of discrete and continuous random variables, distributions, and more.")

elif section == "📉 Continuous Random Variables":
    st.title("📉 Continuous Random Variables")
    st.write("Content continues...")
    st.info("Full coverage of continuous distributions, PDFs, CDFs, and applications.")

elif section == "🔗 Joint Distributions":
    st.title("🔗 Joint Distributions")
    st.write("Content continues...")
    st.info("Detailed treatment of joint probability distributions, marginal distributions, and conditional distributions.")

elif section == "💻 Interactive Practice":
    st.title("💻 Interactive Practice")
    
    practice_topic = st.selectbox(
        "Choose a topic to practice:",
        ["Probability Calculations", "Combinatorics", "Bayes' Rule", "Random Variables"]
    )
    
    if practice_topic == "Probability Calculations":
        st.header("Practice: Basic Probability")
        
        st.write("**Scenario**: Roll a fair six-sided die")
        
        event_type = st.radio(
            "What event do you want to calculate?",
            ["Single outcome", "Multiple outcomes", "Complement"]
        )
        
        if event_type == "Single outcome":
            outcome = st.slider("Select outcome:", 1, 6, 3)
            
            st.success(f"**P(roll a {outcome}) = 1/6 ≈ {1/6:.4f}**")
            
        elif event_type == "Multiple outcomes":
            outcomes = st.multiselect("Select outcomes:", [1, 2, 3, 4, 5, 6], default=[2, 4, 6])
            
            if outcomes:
                prob = len(outcomes) / 6
                st.success(f"**P(roll one of {outcomes}) = {len(outcomes)}/6 ≈ {prob:.4f}**")
        
        else:  # Complement
            outcomes = st.multiselect("Select outcomes for event A:", [1, 2, 3, 4, 5, 6], default=[1, 2])
            
            if outcomes:
                prob_a = len(outcomes) / 6
                prob_comp = 1 - prob_a
                complement = [x for x in range(1, 7) if x not in outcomes]
                
                st.success(f"**P(A) = {prob_a:.4f}**")
                st.success(f"**P(A^c) = 1 - P(A) = {prob_comp:.4f}**")
                st.write(f"A^c = {complement}")

elif section == "📋 Quick Reference":
    st.title("📋 Quick Reference Guide")
    
    ref_type = st.selectbox(
        "Choose reference type:",
        ["Formulas", "Distributions", "Key Concepts", "Common Mistakes"]
    )
    
    if ref_type == "Formulas":
        st.header("Essential Probability Formulas")
        
        st.markdown("""
        ### Basic Probability
        
        | Formula | Name | When to Use |
        |---------|------|-------------|
        | P(A^c) = 1 - P(A) | Complement Rule | Find probability of NOT A |
        | P(A ∪ B) = P(A) + P(B) - P(A ∩ B) | Addition Law | Find probability of A OR B |
        | P(A ∩ B) = P(A) × P(B) | Independence | When A and B are independent |
        
        ### Conditional Probability
        
        | Formula | Name | When to Use |
        |---------|------|-------------|
        | P(A\|B) = P(A ∩ B) / P(B) | Conditional Probability | Probability of A given B |
        | P(A ∩ B) = P(A\|B) × P(B) | Multiplication Law | Find intersection |
        | P(A) = Σ P(A\|Bᵢ)P(Bᵢ) | Law of Total Probability | When {Bᵢ} partition Ω |
        | P(Bᵢ\|A) = P(A\|Bᵢ)P(Bᵢ) / P(A) | Bayes' Rule | Reverse conditional probability |
        
        ### Counting
        
        | Formula | Name | When to Use |
        |---------|------|-------------|
        | n! | Factorial | Arrange n distinct objects |
        | P(n,r) = n!/(n-r)! | Permutations | Arrange r objects from n (order matters) |
        | C(n,r) = n!/(r!(n-r)!) | Combinations | Choose r objects from n (order doesn't matter) |
        """)
    
    elif ref_type == "Distributions":
        st.header("Common Probability Distributions")
        
        st.markdown("""
        ### Discrete Distributions
        
        | Distribution | PMF | Mean | Variance | Use Case |
        |--------------|-----|------|----------|----------|
        | Bernoulli(p) | P(X=1)=p, P(X=0)=1-p | p | p(1-p) | Single success/failure trial |
        | Binomial(n,p) | C(n,k)p^k(1-p)^(n-k) | np | np(1-p) | # successes in n trials |
        | Geometric(p) | (1-p)^(k-1)p | 1/p | (1-p)/p² | # trials until first success |
        | Poisson(λ) | e^(-λ)λ^k/k! | λ | λ | # events in fixed interval |
        
        ### Continuous Distributions
        
        | Distribution | PDF | Mean | Variance | Use Case |
        |--------------|-----|------|----------|----------|
        | Uniform(a,b) | 1/(b-a) | (a+b)/2 | (b-a)²/12 | Equally likely values in [a,b] |
        | Exponential(λ) | λe^(-λx) | 1/λ | 1/λ² | Time between events |
        | Normal(μ,σ²) | Complex | μ | σ² | Bell curve, CLT |
        """)

# Footer
st.markdown("---")
st.caption("🎲 Probability Theory Workbook | Comprehensive Guide with Examples & Practice | Built with Streamlit")
