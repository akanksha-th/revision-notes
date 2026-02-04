import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Statistical Tests Workbook",
    page_icon="📊",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("📚 Navigation")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Jump to Section:",
    [
        "🏠 Home",
        "📖 Foundational Concepts",
        "📈 Parametric Tests",
        "📉 Non-Parametric Tests",
        "🔢 Categorical Tests",
        "🔗 Correlation Tests",
        "✓ Normality Tests",
        "🚀 Advanced Tests",
        "🌳 Decision Tree",
        "💻 Interactive Lab",
        "📋 Quick Reference"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use the navigation menu to jump between sections quickly!")

# Main Content
if section == "🏠 Home":
    st.title("📊 STATISTICAL TESTS WORKBOOK")
    st.subheader("Complete Guide: Beginner to Advanced")
    
    st.markdown("""
    ### Welcome! 👋
    
    This interactive workbook covers everything you need to know about statistical tests for data science,
    from basic concepts to advanced techniques used in production ML systems.
    
    #### 📚 What's Inside:
    
    - **Foundational Concepts**: Hypothesis testing, p-values, Type I/II errors
    - **Parametric Tests**: t-tests, ANOVA, Z-tests
    - **Non-Parametric Tests**: Mann-Whitney, Wilcoxon, Kruskal-Wallis
    - **Categorical Tests**: Chi-square, Fisher's Exact
    - **Correlation Tests**: Pearson, Spearman
    - **Advanced Topics**: Sequential testing, multiple testing correction, permutation tests
    - **Interactive Lab**: Run tests on your own data!
    - **Quick Reference**: Cheat sheets and decision trees
    
    #### 🎯 Perfect For:
    
    ✅ Data Science interview preparation  
    ✅ A/B testing in production  
    ✅ Statistical analysis refresher  
    ✅ Academic projects and research  
    
    #### 🚀 Getting Started:
    
    Use the **sidebar navigation** to jump to any section, or scroll through sequentially.
    Each section includes:
    - Clear intuition and real-world examples
    - Assumptions and when to use each test
    - Working code in Python (and R where applicable)
    - Interactive examples you can try yourself
    
    ---
    
    ### 📌 Key Principles
    
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎯 Intuition First**
        
        Understand *why* before *how*. Every test starts with a clear explanation of what it does.
        """)
    
    with col2:
        st.success("""
        **💼 Real Applications**
        
        See how each test is used in actual data science work and A/B testing.
        """)
    
    with col3:
        st.warning("""
        **⚠️ Check Assumptions**
        
        Always verify your data meets the test's requirements before applying it.
        """)
    
    st.markdown("---")
    st.markdown("👈 **Select a section from the sidebar to begin!**")

elif section == "📖 Foundational Concepts":
    st.title("📖 PART 1: Foundational Concepts")
    
    tab1, tab2, tab3 = st.tabs(["Hypothesis Testing", "Type I/II Errors", "One vs Two-Tailed"])
    
    with tab1:
        st.header("1.1 What is Hypothesis Testing?")
        
        st.info("""
        **INTUITION**: You have a sample (small group), but want to make claims about the population (everyone). 
        Hypothesis testing helps you decide if patterns in your sample are real or just random luck.
        """)
        
        st.markdown("""
        **Real-world analogy**: You flip a coin 10 times and get 8 heads. Is the coin biased, or did you just get lucky? 
        Statistics answers this!
        
        #### Key Components:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Null Hypothesis (H₀)**: The "boring" assumption - nothing special is happening (e.g., coin is fair)
            - **Alternative Hypothesis (H₁)**: The "interesting" claim you want to prove (e.g., coin is biased)
            - **p-value**: Probability of seeing your data if H₀ is true. Low p-value → reject H₀
            """)
        
        with col2:
            st.markdown("""
            - **Significance Level (α)**: Threshold for decision (usually 0.05 = 5%)
            - **Test Statistic**: A number calculated from your data (t, z, F, χ²)
            """)
        
        st.success("**Decision Rule**: If p-value < α → Reject H₀ (result is 'statistically significant')")
        
        st.warning("""
        #### ⚠️ Common Misconceptions:
        
        - p-value is NOT the probability that H₀ is true
        - "Significant" doesn't mean "important" - just "unlikely to be random"
        - Failing to reject H₀ doesn't prove it's true (absence of evidence ≠ evidence of absence)
        """)
    
    with tab2:
        st.header("1.2 Type I and Type II Errors")
        
        st.info("**INTUITION**: Statistical tests can make mistakes!")
        
        # Create error matrix
        error_df = pd.DataFrame({
            'H₀ True (no effect)': ['Type I Error (α)\nFalse Positive ❌', 'Correct ✓'],
            'H₀ False (effect exists)': ['Correct ✓', 'Type II Error (β)\nFalse Negative ❌']
        }, index=['Reject H₀', 'Fail to Reject H₀'])
        
        st.table(error_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("""
            **Type I Error (α)**
            
            Launching a feature that doesn't actually improve metrics (wasted resources)
            """)
        
        with col2:
            st.warning("""
            **Type II Error (β)**
            
            Missing a good feature because test didn't detect the improvement
            """)
        
        st.success("**Statistical Power = 1 - β**: Probability of detecting an effect when it exists (aim for 80%+)")
    
    with tab3:
        st.header("1.3 One-tailed vs Two-tailed Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Two-tailed**
            
            Testing if something is *different* (bigger OR smaller)
            
            - H₁: μ ≠ μ₀
            - Use when: You don't know the direction of change
            
            **Example**: "Does the new UI change conversion rate?" (could go up or down)
            """)
        
        with col2:
            st.success("""
            **One-tailed**
            
            Testing if something is *specifically bigger* or *specifically smaller*
            
            - H₁: μ > μ₀ (or μ < μ₀)
            - Use when: You have a directional hypothesis
            
            **Example**: "Does the new UI increase conversion rate?" (only care about increase)
            """)

elif section == "📈 Parametric Tests":
    st.title("📈 PART 2: Parametric Tests")
    st.caption("Tests that assume normal distribution")
    
    test_type = st.selectbox(
        "Select a test to learn about:",
        ["Z-Test (One Sample)", "T-Test (One Sample)", "T-Test (Independent)", 
         "T-Test (Paired)", "ANOVA"]
    )
    
    if test_type == "Z-Test (One Sample)":
        st.header("2.1 Z-Test (One Sample)")
        
        st.info("**INTUITION**: Is my sample mean different from a known population mean?")
        
        st.markdown("""
        #### When to Use:
        - Comparing sample mean to known population mean
        - Sample size > 30 (or population std known)
        - Data is normally distributed or n is large
        
        #### Assumptions:
        1. Random sampling
        2. Normal distribution (or n > 30)
        3. Known population standard deviation (σ)
        
        #### Formula:
        """)
        
        st.latex(r"z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}}")
        
        st.success("""
        **Real DS Pipeline Use**:
        - A/B testing when you have historical baseline metrics
        - Quality control: "Is this batch's average weight 100g as specified?"
        - Checking if campaign performance matches company average
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
import numpy as np
from scipy import stats

# Example: Is average user session time different from 5 minutes?
sample_mean = 5.3  # minutes
pop_mean = 5.0
pop_std = 1.2
n = 100

# Calculate z-statistic
z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Session time is significantly different from 5 min")
else:
    print("Fail to reject H₀: No significant difference")
            """, language="python")
        
        with st.expander("📊 R Code Example"):
            st.code("""
# Manual calculation
sample_mean <- 5.3
pop_mean <- 5.0
pop_std <- 1.2
n <- 100

z_stat <- (sample_mean - pop_mean) / (pop_std / sqrt(n))
p_value <- 2 * pnorm(-abs(z_stat))

cat("Z-statistic:", z_stat, "\\n")
cat("P-value:", p_value, "\\n")
            """, language="r")
    
    elif test_type == "T-Test (One Sample)":
        st.header("2.2 T-Test (One Sample)")
        
        st.info("**INTUITION**: Like Z-test, but when you don't know population std (most real cases!)")
        
        st.markdown("""
        #### When to Use:
        - Sample mean vs hypothesized population mean
        - Population std is unknown
        - Sample size < 30 (for larger samples, t ≈ z)
        
        #### Assumptions:
        1. Random sampling
        2. Normal distribution (important for small n)
        3. Unknown population std
        
        #### Formula:
        """)
        
        st.latex(r"t = \frac{\bar{x} - \mu}{s/\sqrt{n}}")
        st.caption("where s = sample standard deviation")
        
        st.success("""
        **Real DS Pipeline Use**:
        - Testing if average purchase value changed after redesign
        - Checking if model prediction error is centered at zero
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats

# Example: Did average app rating change from 4.0?
ratings = [4.2, 4.5, 3.9, 4.1, 4.3, 4.0, 4.4, 3.8, 4.2, 4.1]
pop_mean = 4.0

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(ratings, pop_mean)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
            """, language="python")
        
        with st.expander("📊 R Code Example"):
            st.code("""
ratings <- c(4.2, 4.5, 3.9, 4.1, 4.3, 4.0, 4.4, 3.8, 4.2, 4.1)
pop_mean <- 4.0

result <- t.test(ratings, mu = pop_mean)
print(result)
            """, language="r")
    
    elif test_type == "T-Test (Independent)":
        st.header("2.3 T-Test (Two Independent Samples)")
        
        st.info("**INTUITION**: Are the means of two different groups significantly different?")
        
        st.markdown("""
        #### When to Use:
        - A/B testing (control vs treatment)
        - Comparing two independent groups
        - Samples are from different populations
        
        #### Assumptions:
        1. Random sampling from both groups
        2. Normal distribution in both groups
        3. Independence between groups
        4. Equal variances (Welch's t-test relaxes this)
        
        #### Formula:
        """)
        
        st.latex(r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}")
        
        st.success("""
        **Real DS Pipeline Use**:
        - **Most common in DS!** A/B testing conversions, revenue, engagement
        - Comparing ML model performance on two datasets
        - Testing if feature values differ between churned vs active users
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats
import numpy as np

# Example: A/B test - does new button color increase clicks?
control_clicks = [23, 25, 21, 24, 26, 22, 25, 23, 24, 25]
treatment_clicks = [28, 30, 27, 29, 31, 28, 30, 29, 28, 32]

# Independent t-test (assuming equal variances)
t_stat, p_value = stats.ttest_ind(treatment_clicks, control_clicks)

# Welch's t-test (unequal variances - safer default)
t_stat_welch, p_value_welch = stats.ttest_ind(treatment_clicks, control_clicks, equal_var=False)

print(f"Standard t-test p-value: {p_value:.4f}")
print(f"Welch's t-test p-value: {p_value_welch:.4f}")

# Effect size (Cohen's d)
mean_diff = np.mean(treatment_clicks) - np.mean(control_clicks)
pooled_std = np.sqrt((np.var(control_clicks) + np.var(treatment_clicks)) / 2)
cohens_d = mean_diff / pooled_std
print(f"Cohen's d (effect size): {cohens_d:.4f}")
            """, language="python")
    
    elif test_type == "T-Test (Paired)":
        st.header("2.4 T-Test (Paired Samples)")
        
        st.info("**INTUITION**: Compare two measurements from the SAME subjects (before/after)")
        
        st.markdown("""
        #### When to Use:
        - Before/after comparisons on same subjects
        - Matched pairs design
        - Repeated measures on same items
        
        #### Key Difference from Independent:
        Accounts for correlation between pairs!
        
        #### Assumptions:
        1. Paired observations
        2. Differences are normally distributed
        3. Random sampling
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Before/after analysis: user behavior before and after feature launch
        - Comparing two ML models on same test set
        - Pre-post intervention studies (e.g., ad campaign effect on same users)
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats
import numpy as np

# Example: User engagement before and after app redesign (same users)
engagement_before = [45, 52, 48, 50, 47, 53, 49, 51, 46, 50]
engagement_after = [50, 55, 52, 54, 51, 57, 53, 55, 50, 54]

# Paired t-test
t_stat, p_value = stats.ttest_rel(engagement_after, engagement_before)

print(f"Paired t-test p-value: {p_value:.4f}")
print(f"Mean difference: {np.mean(np.array(engagement_after) - np.array(engagement_before)):.2f}")
            """, language="python")
    
    elif test_type == "ANOVA":
        st.header("2.5 ANOVA (Analysis of Variance)")
        
        st.info("**INTUITION**: T-test for MORE than 2 groups. Tests if any group means differ.")
        
        st.markdown("""
        #### When to Use:
        - Comparing 3+ groups
        - One categorical variable, one continuous outcome
        
        #### Why not multiple t-tests?
        Increases Type I error (family-wise error rate)
        
        #### Types:
        - **One-way ANOVA**: One categorical variable
        - **Two-way ANOVA**: Two categorical variables + interaction
        
        #### Assumptions:
        1. Normal distribution in each group
        2. Equal variances across groups (homoscedasticity)
        3. Independence of observations
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Comparing conversion rates across multiple marketing channels
        - Testing if different customer segments have different LTV
        - Comparing ML model performance across multiple feature sets
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats

# Example: Testing engagement across 4 app versions
version_A = [45, 48, 46, 47, 49]
version_B = [52, 55, 53, 54, 56]
version_C = [48, 50, 49, 51, 47]
version_D = [60, 62, 61, 63, 59]

# One-way ANOVA
f_stat, p_value = stats.f_oneway(version_A, version_B, version_C, version_D)

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: At least one version has different engagement")
    print("Need post-hoc tests to find which pairs differ!")
            """, language="python")
        
        with st.expander("📊 Post-hoc Tests"):
            st.code("""
from scipy.stats import ttest_ind
from itertools import combinations

# Bonferroni correction
groups = {'A': version_A, 'B': version_B, 'C': version_C, 'D': version_D}
pairs = list(combinations(groups.keys(), 2))
alpha_corrected = 0.05 / len(pairs)  # Bonferroni correction

print(f"\\nPost-hoc pairwise comparisons (α = {alpha_corrected:.4f}):")
for pair in pairs:
    g1, g2 = pair
    _, p = ttest_ind(groups[g1], groups[g2])
    sig = "***" if p < alpha_corrected else ""
    print(f"{g1} vs {g2}: p = {p:.4f} {sig}")
            """, language="python")

elif section == "📉 Non-Parametric Tests":
    st.title("📉 PART 3: Non-Parametric Tests")
    st.caption("Tests that don't assume normal distribution")
    
    st.warning("""
    #### When to Use Non-parametric Tests:
    - Data is not normally distributed
    - Ordinal data (rankings, Likert scales)
    - Small sample sizes
    - Outliers present
    - Violated parametric assumptions
    """)
    
    test_type = st.selectbox(
        "Select a test:",
        ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis Test"]
    )
    
    if test_type == "Mann-Whitney U Test":
        st.header("3.1 Mann-Whitney U Test (Wilcoxon Rank-Sum)")
        
        st.info("**INTUITION**: Non-parametric version of independent t-test. Compares distributions of two groups using ranks instead of means.")
        
        st.markdown("""
        #### When to Use:
        - Two independent groups
        - Can't assume normality
        - Ordinal or continuous data
        
        #### How It Works:
        Ranks all values together, then compares rank sums
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Comparing user satisfaction scores (ordinal: 1-5 stars)
        - A/B testing with skewed metrics (e.g., revenue per user)
        - Comparing pageviews (often heavy-tailed distribution)
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats

# Example: Comparing satisfaction ratings (1-5) between two UX designs
design_A = [3, 4, 3, 5, 3, 4, 3, 4, 5, 3]
design_B = [4, 5, 4, 5, 5, 4, 5, 4, 5, 5]

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(design_B, design_A, alternative='two-sided')

print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")
    
    elif test_type == "Wilcoxon Signed-Rank Test":
        st.header("3.2 Wilcoxon Signed-Rank Test")
        
        st.info("**INTUITION**: Non-parametric version of paired t-test. Compares paired observations using ranks of differences.")
        
        st.markdown("""
        #### When to Use:
        - Paired/matched samples
        - Can't assume normality of differences
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Before/after comparisons with non-normal data
        - Comparing two recommendation algorithms on same users (ordinal ratings)
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats

# Example: User engagement scores before/after feature
before = [3, 4, 3, 5, 2, 4, 3, 5, 4, 3]
after = [4, 5, 4, 5, 3, 5, 4, 5, 5, 4]

w_stat, p_value = stats.wilcoxon(after, before)

print(f"Wilcoxon statistic: {w_stat:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")
    
    elif test_type == "Kruskal-Wallis Test":
        st.header("3.3 Kruskal-Wallis Test")
        
        st.info("**INTUITION**: Non-parametric version of one-way ANOVA. Tests if 3+ groups have different distributions.")
        
        st.markdown("""
        #### When to Use:
        - 3+ independent groups
        - Can't assume normality
        - Ordinal or continuous data
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Comparing user ratings across multiple app versions
        - Testing engagement across different customer segments with skewed data
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy import stats

# Example: Engagement across 3 marketing channels
email = [45, 48, 52, 46, 50, 47, 49]
social = [60, 62, 58, 61, 59, 63, 60]
search = [52, 54, 53, 55, 51, 56, 52]

h_stat, p_value = stats.kruskal(email, social, search)

print(f"H-statistic: {h_stat:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")

elif section == "🔢 Categorical Tests":
    st.title("🔢 PART 4: Categorical Data Tests")
    
    test_type = st.selectbox(
        "Select a test:",
        ["Chi-Square Test", "Fisher's Exact Test"]
    )
    
    if test_type == "Chi-Square Test":
        st.header("4.1 Chi-Square Test for Independence")
        
        st.info("**INTUITION**: Tests if two categorical variables are related (associated) or independent.")
        
        st.markdown("""
        #### When to Use:
        - Both variables are categorical
        - Testing association/relationship
        - Contingency table (cross-tabulation)
        
        #### Assumptions:
        1. Random sampling
        2. Expected frequency ≥ 5 in each cell (if violated, use Fisher's Exact)
        3. Observations are independent
        
        #### Formula:
        """)
        
        st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
        
        st.success("""
        **Real DS Pipeline Use**:
        - **Very common!** Testing if user segment is related to purchase behavior
        - Checking if device type affects conversion (mobile vs desktop vs tablet)
        - Testing if marketing channel is associated with customer churn
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

# Example: Is device type related to conversion?
# Rows: Device, Columns: Converted (No/Yes)
data = pd.DataFrame({
    'No': [150, 200, 100],  # No conversion
    'Yes': [50, 80, 120]    # Converted
}, index=['Mobile', 'Desktop', 'Tablet'])

print("Observed frequencies:")
print(data)

# Chi-square test
chi2, p_value, dof, expected = chi2_contingency(data)

print(f"\\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print("\\nExpected frequencies:")
print(pd.DataFrame(expected, index=data.index, columns=data.columns))

# Cramér's V (effect size for chi-square)
n = data.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(data.shape) - 1)))
print(f"\\nCramér's V (effect size): {cramers_v:.4f}")
            """, language="python")
    
    elif test_type == "Fisher's Exact Test":
        st.header("4.2 Fisher's Exact Test")
        
        st.info("**INTUITION**: Like chi-square, but uses exact probabilities. Better for small samples.")
        
        st.markdown("""
        #### When to Use:
        - 2x2 contingency table
        - Small sample sizes (expected frequency < 5 in any cell)
        - Need exact p-value
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - A/B testing with low conversion rates (small counts)
        - Rare event analysis
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import fisher_exact

# Example: Testing if promotion affects rare event (e.g., premium upgrade)
# Rows: Promotion, Columns: Upgraded (No/Yes)
table = [[95, 5],    # No promotion: 95 didn't upgrade, 5 upgraded
         [88, 12]]   # With promotion: 88 didn't upgrade, 12 upgraded

oddsratio, p_value = fisher_exact(table)

print(f"Odds ratio: {oddsratio:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")

elif section == "🔗 Correlation Tests":
    st.title("🔗 PART 5: Correlation Tests")
    
    test_type = st.selectbox(
        "Select a test:",
        ["Pearson Correlation", "Spearman Correlation"]
    )
    
    if test_type == "Pearson Correlation":
        st.header("5.1 Pearson Correlation")
        
        st.info("**INTUITION**: Measures LINEAR relationship strength between two continuous variables (-1 to +1).")
        
        st.markdown("""
        #### When to Use:
        - Both variables continuous
        - Linear relationship
        - Both normally distributed
        
        #### Interpretation:
        - r = 0: No linear relationship
        - r = +1: Perfect positive linear relationship
        - r = -1: Perfect negative linear relationship
        - |r| > 0.7: Strong
        - 0.3-0.7: Moderate
        - < 0.3: Weak
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Feature selection: Check multicollinearity
        - Understanding feature relationships
        - Validating that related metrics move together
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import pearsonr

# Example: Correlation between ad spend and revenue
ad_spend = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
revenue = [15000, 18000, 22000, 26000, 29000, 33000, 37000]

r, p_value = pearsonr(ad_spend, revenue)

print(f"Pearson correlation: {r:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"R-squared: {r**2:.4f} ({r**2*100:.1f}% of variance explained)")
            """, language="python")
    
    elif test_type == "Spearman Correlation":
        st.header("5.2 Spearman Rank Correlation")
        
        st.info("**INTUITION**: Measures MONOTONIC relationship (not just linear) using ranks. Non-parametric alternative to Pearson.")
        
        st.markdown("""
        #### When to Use:
        - Non-linear but monotonic relationship
        - Ordinal data
        - Presence of outliers
        - Non-normal distributions
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Correlating rankings (e.g., user preferences)
        - When relationship exists but isn't strictly linear
        - Robust to outliers in feature analysis
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import spearmanr

# Example: Correlation between user activity rank and purchase rank
activity_score = [10, 45, 23, 67, 89, 34, 56, 78, 12, 90]
purchase_amount = [15, 120, 50, 200, 350, 80, 180, 290, 25, 400]

rho, p_value = spearmanr(activity_score, purchase_amount)

print(f"Spearman correlation: {rho:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")

elif section == "✓ Normality Tests":
    st.title("✓ PART 6: Normality Tests")
    
    test_type = st.selectbox(
        "Select a test:",
        ["Shapiro-Wilk Test", "Kolmogorov-Smirnov Test", "Levene's Test"]
    )
    
    if test_type == "Shapiro-Wilk Test":
        st.header("6.1 Shapiro-Wilk Test")
        
        st.info("**INTUITION**: Tests if data comes from a normal distribution.")
        
        st.markdown("""
        #### When to Use:
        - Before parametric tests (check assumptions)
        - Sample size < 50 (most powerful for small samples)
        
        #### Null Hypothesis:
        Data is normally distributed
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Checking assumptions before t-tests, ANOVA
        - Deciding between parametric vs non-parametric tests
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import shapiro

# Example: Check if conversion rates are normally distributed
conversion_rates = [0.15, 0.18, 0.16, 0.17, 0.19, 0.14, 0.16, 0.18, 0.15, 0.17]

stat, p_value = shapiro(conversion_rates)

print(f"Shapiro-Wilk statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Data appears normally distributed (fail to reject H₀)")
else:
    print("Data does NOT appear normally distributed (reject H₀)")
            """, language="python")
    
    elif test_type == "Kolmogorov-Smirnov Test":
        st.header("6.2 Kolmogorov-Smirnov Test")
        
        st.info("**INTUITION**: Tests if sample comes from a specific distribution (can test vs normal, uniform, etc.)")
        
        st.markdown("""
        #### When to Use:
        - Larger sample sizes (n > 50)
        - Testing against any theoretical distribution
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import kstest
import numpy as np

data = np.random.normal(0, 1, 100)

# Test against normal distribution
stat, p_value = kstest(data, 'norm')

print(f"KS statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")
    
    elif test_type == "Levene's Test":
        st.header("6.3 Levene's Test (Equal Variance Test)")
        
        st.info("**INTUITION**: Tests if multiple groups have equal variances (homoscedasticity).")
        
        st.markdown("""
        #### When to Use:
        - Before ANOVA (checking assumptions)
        - Before t-test with equal variance assumption
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Validating ANOVA assumptions
        - Deciding between standard vs Welch's t-test
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from scipy.stats import levene

group1 = [23, 25, 21, 24, 26, 22, 25]
group2 = [28, 30, 27, 29, 31, 28, 30]
group3 = [18, 20, 19, 21, 17, 19, 20]

stat, p_value = levene(group1, group2, group3)

print(f"Levene statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Variances are equal (homoscedastic)")
else:
    print("Variances are NOT equal (heteroscedastic)")
            """, language="python")

elif section == "🚀 Advanced Tests":
    st.title("🚀 PART 7: Advanced Tests for Data Science")
    
    test_type = st.selectbox(
        "Select a topic:",
        ["Proportion Z-Test", "Multiple Testing Correction", "Permutation Test", "Bootstrap CI"]
    )
    
    if test_type == "Proportion Z-Test":
        st.header("7.1 Two-Sample Proportion Test (Z-test for Proportions)")
        
        st.info("**INTUITION**: Tests if two groups have different success rates/proportions.")
        
        st.markdown("""
        #### When to Use:
        - A/B testing with binary outcomes (converted/not, clicked/not)
        - Comparing conversion rates, click-through rates
        - Large sample sizes
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - **THE most common test in online A/B testing!**
        - Testing if new design increases conversion rate
        - Comparing click-through rates between ad variants
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Example: A/B test for conversion rates
# Control: 150 conversions out of 1000 users
# Treatment: 180 conversions out of 1000 users

successes = np.array([150, 180])
nobs = np.array([1000, 1000])

z_stat, p_value = proportions_ztest(successes, nobs)

# Calculate proportions and difference
p_control = successes[0] / nobs[0]
p_treatment = successes[1] / nobs[1]
diff = p_treatment - p_control

print(f"Control conversion rate: {p_control:.1%}")
print(f"Treatment conversion rate: {p_treatment:.1%}")
print(f"Absolute difference: {diff:.1%}")
print(f"Relative lift: {(diff/p_control)*100:.1f}%")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
            """, language="python")
    
    elif test_type == "Multiple Testing Correction":
        st.header("7.2 Multiple Testing Correction")
        
        st.info("**INTUITION**: When doing many tests, you increase chance of false positives. Corrections control this.")
        
        st.markdown("""
        #### When to Use:
        - Running multiple A/B tests simultaneously
        - Testing many features for significance
        - Any time you're doing >1 hypothesis test
        
        #### Methods:
        1. **Bonferroni**: Divide α by number of tests (conservative)
        2. **Holm-Bonferroni**: Sequential Bonferroni (less conservative)
        3. **Benjamini-Hochberg (FDR)**: Controls false discovery rate (less conservative)
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Testing multiple variants in multivariate tests
        - Feature selection with many features
        - Analyzing metrics for multiple user segments
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
from statsmodels.stats.multitest import multipletests

# Example: Testing 10 different features
p_values = [0.001, 0.03, 0.04, 0.15, 0.002, 0.06, 0.25, 0.01, 0.35, 0.08]

# Original α = 0.05
alpha = 0.05

# Bonferroni correction
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')

# Benjamini-Hochberg (FDR)
reject_bh, pvals_bh, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

print("Original p-values:", p_values)
print("\\nBonferroni correction:")
print("  Adjusted p-values:", pvals_bonf)
print("  Reject H₀:", reject_bonf)
print(f"  Significant features: {sum(reject_bonf)}")

print("\\nBenjamini-Hochberg (FDR):")
print("  Adjusted p-values:", pvals_bh)
print("  Reject H₀:", reject_bh)
print(f"  Significant features: {sum(reject_bh)}")
            """, language="python")
    
    elif test_type == "Permutation Test":
        st.header("7.3 Permutation Test")
        
        st.info("**INTUITION**: Non-parametric test that works by shuffling labels. No distribution assumptions!")
        
        st.markdown("""
        #### When to Use:
        - Small samples
        - Weird distributions
        - When you don't trust parametric assumptions
        - Complex test statistics
        
        #### How it works:
        1. Calculate test statistic on observed data
        2. Randomly shuffle group labels many times
        3. Calculate test statistic for each shuffle
        4. p-value = proportion of shuffles with statistic as extreme as observed
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - A/B tests with unusual metrics
        - Testing with small samples where normality is questionable
        - Custom test statistics
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
import numpy as np

def permutation_test(group1, group2, n_permutations=10000):
    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Combine data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(combined)
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:]
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        perm_diffs.append(perm_diff)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value, perm_diffs

# Example
control = [23, 25, 21, 24, 26, 22, 25, 23]
treatment = [28, 30, 27, 29, 31, 28, 30, 32]

obs_diff, p_val, perm_dist = permutation_test(control, treatment)

print(f"Observed difference: {obs_diff:.2f}")
print(f"P-value: {p_val:.4f}")
            """, language="python")
    
    elif test_type == "Bootstrap CI":
        st.header("7.4 Bootstrap Confidence Intervals")
        
        st.info("**INTUITION**: Create confidence intervals by resampling your data with replacement.")
        
        st.markdown("""
        #### When to Use:
        - Estimating uncertainty around any statistic
        - Non-normal data
        - Complex statistics (median, ratio, etc.)
        """)
        
        st.success("""
        **Real DS Pipeline Use**:
        - Confidence intervals for conversion rates
        - Uncertainty in model performance metrics
        - Any metric where analytical CI is hard to derive
        """)
        
        with st.expander("📝 Python Code Example"):
            st.code("""
import numpy as np

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=95):
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    lower = (100 - ci) / 2
    upper = 100 - lower
    ci_lower = np.percentile(bootstrap_stats, lower)
    ci_upper = np.percentile(bootstrap_stats, upper)
    
    return ci_lower, ci_upper, bootstrap_stats

# Example
conversion_rates = [0.15, 0.18, 0.16, 0.17, 0.19, 0.14, 0.16, 0.18, 0.15, 0.17]

ci_lower, ci_upper, boot_dist = bootstrap_ci(conversion_rates, statistic=np.median, ci=95)

print(f"Observed median: {np.median(conversion_rates):.4f}")
print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            """, language="python")

elif section == "🌳 Decision Tree":
    st.title("🌳 PART 8: Choosing the Right Test")
    
    st.markdown("""
    ## Decision Tree for Statistical Test Selection
    
    Use this flowchart to choose the appropriate test for your data:
    """)
    
    # Interactive decision tree
    data_type = st.radio("What type of outcome variable do you have?", 
                        ["Categorical", "Continuous"])
    
    if data_type == "Categorical":
        st.info("### Categorical Outcome Selected")
        
        cat_scenario = st.selectbox(
            "What's your scenario?",
            ["1 categorical variable (frequency counts)",
             "2 categorical variables (contingency table)",
             "Comparing 2 proportions"]
        )
        
        if cat_scenario == "1 categorical variable (frequency counts)":
            st.success("✓ Use: **Chi-square goodness of fit test**")
        
        elif cat_scenario == "2 categorical variables (contingency table)":
            sample_size = st.radio("Sample size?", ["Large (expected freq ≥ 5)", "Small OR 2x2 table"])
            if sample_size == "Large (expected freq ≥ 5)":
                st.success("✓ Use: **Chi-square test of independence**")
            else:
                st.success("✓ Use: **Fisher's exact test**")
        
        elif cat_scenario == "Comparing 2 proportions":
            st.success("✓ Use: **Two-proportion Z-test**")
    
    else:  # Continuous
        st.info("### Continuous Outcome Selected")
        
        cont_scenario = st.selectbox(
            "What are you comparing?",
            ["Sample vs fixed value",
             "2 groups",
             "3+ groups",
             "Relationship between 2 variables"]
        )
        
        if cont_scenario == "Sample vs fixed value":
            pop_std = st.radio("Population standard deviation:", ["Known", "Unknown"])
            
            if pop_std == "Known":
                st.success("✓ Use: **One-sample Z-test**")
            else:
                normal = st.radio("Is data normally distributed?", ["Yes", "No/Small sample"])
                if normal == "Yes":
                    st.success("✓ Use: **One-sample t-test**")
                else:
                    st.success("✓ Use: **Wilcoxon signed-rank test**")
        
        elif cont_scenario == "2 groups":
            paired = st.radio("Are the groups:", ["Independent", "Paired/Matched"])
            
            if paired == "Independent":
                normal = st.radio("Normal distribution?", ["Yes", "No"])
                if normal == "Yes":
                    variance = st.radio("Equal variances?", ["Yes", "No"])
                    if variance == "Yes":
                        st.success("✓ Use: **Independent t-test**")
                    else:
                        st.success("✓ Use: **Welch's t-test**")
                else:
                    st.success("✓ Use: **Mann-Whitney U test**")
            else:
                normal = st.radio("Normal distribution of differences?", ["Yes", "No"])
                if normal == "Yes":
                    st.success("✓ Use: **Paired t-test**")
                else:
                    st.success("✓ Use: **Wilcoxon signed-rank test**")
        
        elif cont_scenario == "3+ groups":
            normal = st.radio("Normal distribution + equal variances?", ["Yes", "No"])
            if normal == "Yes":
                st.success("✓ Use: **One-way ANOVA** (+ post-hoc tests)")
            else:
                st.success("✓ Use: **Kruskal-Wallis test**")
        
        elif cont_scenario == "Relationship between 2 variables":
            relationship = st.radio("Type of relationship?", 
                                   ["Linear (both continuous, normal)", 
                                    "Monotonic (ordinal or non-linear)"])
            if relationship == "Linear (both continuous, normal)":
                st.success("✓ Use: **Pearson correlation**")
            else:
                st.success("✓ Use: **Spearman correlation**")

elif section == "💻 Interactive Lab":
    st.title("💻 PART 9: Interactive Statistical Testing Lab")
    st.markdown("Try running statistical tests on your own data or use sample data!")
    
    test_choice = st.selectbox(
        "Which test would you like to try?",
        ["Independent t-test", "Paired t-test", "ANOVA", "Chi-square", "Proportion Z-test"]
    )
    
    if test_choice == "Independent t-test":
        st.header("Interactive Independent t-test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Control Group")
            control_input = st.text_area(
                "Enter values (comma-separated):",
                "23, 25, 21, 24, 26, 22, 25, 23, 24, 25",
                key="control"
            )
        
        with col2:
            st.subheader("Treatment Group")
            treatment_input = st.text_area(
                "Enter values (comma-separated):",
                "28, 30, 27, 29, 31, 28, 30, 29, 28, 32",
                key="treatment"
            )
        
        equal_var = st.checkbox("Assume equal variances", value=False)
        
        if st.button("Run t-test", type="primary"):
            try:
                control = np.array([float(x.strip()) for x in control_input.split(',')])
                treatment = np.array([float(x.strip()) for x in treatment_input.split(',')])
                
                # Run test
                t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=equal_var)
                
                # Calculate effect size
                mean_diff = np.mean(treatment) - np.mean(control)
                pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
                cohens_d = mean_diff / pooled_std
                
                # Display results
                st.success("### Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Control Mean", f"{np.mean(control):.2f}")
                with col2:
                    st.metric("Treatment Mean", f"{np.mean(treatment):.2f}")
                with col3:
                    st.metric("Difference", f"{mean_diff:.2f}")
                with col4:
                    st.metric("Cohen's d", f"{cohens_d:.3f}")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("t-statistic", f"{t_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ **Reject H₀**: Significant difference detected (p < 0.05)")
                else:
                    st.info("ℹ️ **Fail to Reject H₀**: No significant difference (p ≥ 0.05)")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.boxplot([control, treatment], labels=['Control', 'Treatment'])
                ax.set_ylabel('Value')
                ax.set_title('Distribution Comparison')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif test_choice == "Proportion Z-test":
        st.header("Interactive Proportion Z-test")
        st.caption("For A/B testing with binary outcomes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Control Group")
            control_conv = st.number_input("Conversions:", min_value=0, value=150, key="c_conv")
            control_total = st.number_input("Total users:", min_value=1, value=1000, key="c_total")
        
        with col2:
            st.subheader("Treatment Group")
            treatment_conv = st.number_input("Conversions:", min_value=0, value=180, key="t_conv")
            treatment_total = st.number_input("Total users:", min_value=1, value=1000, key="t_total")
        
        if st.button("Run Proportion Test", type="primary"):
            try:
                from statsmodels.stats.proportion import proportions_ztest
                
                successes = np.array([control_conv, treatment_conv])
                nobs = np.array([control_total, treatment_total])
                
                z_stat, p_value = proportions_ztest(successes, nobs)
                
                p_control = control_conv / control_total
                p_treatment = treatment_conv / treatment_total
                diff = p_treatment - p_control
                relative_lift = (diff / p_control) * 100 if p_control > 0 else 0
                
                st.success("### Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Control Rate", f"{p_control:.1%}")
                with col2:
                    st.metric("Treatment Rate", f"{p_treatment:.1%}")
                with col3:
                    st.metric("Absolute Diff", f"{diff:.1%}")
                with col4:
                    st.metric("Relative Lift", f"{relative_lift:.1f}%")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Z-statistic", f"{z_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ **Reject H₀**: Significant difference in conversion rates (p < 0.05)")
                else:
                    st.info("ℹ️ **Fail to Reject H₀**: No significant difference (p ≥ 0.05)")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 5))
                groups = ['Control', 'Treatment']
                rates = [p_control * 100, p_treatment * 100]
                bars = ax.bar(groups, rates, color=['#3498db', '#2ecc71'])
                ax.set_ylabel('Conversion Rate (%)')
                ax.set_title('Conversion Rate Comparison')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif section == "📋 Quick Reference":
    st.title("📋 PART 10: Quick Reference Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Test Selection Table", "Formulas", "Common Mistakes", "Interview Tips"])
    
    with tab1:
        st.header("Statistical Tests Quick Reference")
        
        reference_data = {
            'Test': ['One-sample Z-test', 'One-sample t-test', 'Independent t-test', 
                    'Welch\'s t-test', 'Paired t-test', 'ANOVA', 'Mann-Whitney U',
                    'Wilcoxon Signed-Rank', 'Kruskal-Wallis', 'Chi-square',
                    'Fisher\'s Exact', 'Proportion Z-test', 'Pearson Correlation',
                    'Spearman Correlation', 'Shapiro-Wilk', 'Levene\'s Test'],
            'Use Case': ['Sample mean vs population mean', 'Sample mean vs hypothesized mean',
                        '2 independent groups', '2 independent groups', '2 related/matched groups',
                        '3+ independent groups', '2 independent groups', '2 paired groups',
                        '3+ independent groups', '2 categorical variables', '2 categorical (2x2)',
                        '2 proportions', 'Linear relationship', 'Monotonic relationship',
                        'Test normality', 'Test equal variance'],
            'Data Type': ['Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous',
                         'Continuous', 'Ordinal/Continuous', 'Ordinal/Continuous', 'Ordinal/Continuous',
                         'Categorical', 'Categorical', 'Binary', 'Continuous', 'Ordinal/Continuous',
                         'Continuous', 'Continuous'],
            'Python': ['scipy.stats', 'stats.ttest_1samp()', 'stats.ttest_ind()',
                      'stats.ttest_ind(equal_var=False)', 'stats.ttest_rel()', 'stats.f_oneway()',
                      'stats.mannwhitneyu()', 'stats.wilcoxon()', 'stats.kruskal()',
                      'chi2_contingency()', 'fisher_exact()', 'proportions_ztest()',
                      'stats.pearsonr()', 'stats.spearmanr()', 'stats.shapiro()', 'stats.levene()']
        }
        
        df = pd.DataFrame(reference_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("Key Formulas Cheat Sheet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### T-statistic")
            st.latex(r"t = \frac{\bar{x} - \mu}{s/\sqrt{n}}")
            
            st.markdown("### Cohen's d (Effect Size)")
            st.latex(r"d = \frac{\bar{x}_1 - \bar{x}_2}{\text{pooled\_std}}")
            
            st.markdown("### Pooled Standard Deviation")
            st.latex(r"\text{pooled\_std} = \sqrt{\frac{s_1^2 + s_2^2}{2}}")
        
        with col2:
            st.markdown("### Chi-square")
            st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
            
            st.markdown("### Pearson Correlation")
            st.latex(r"r = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x - \bar{x})^2 \sum(y - \bar{y})^2}}")
            
            st.markdown("### Confidence Interval")
            st.latex(r"CI = \bar{x} \pm (t \times \frac{s}{\sqrt{n}})")
    
    with tab3:
        st.header("Common Mistakes to Avoid")
        
        mistakes = [
            ("Peeking at data repeatedly", "Use sequential testing or Bonferroni correction"),
            ("Using parametric tests on non-normal data", "Check assumptions first!"),
            ("Ignoring effect size", "Statistical significance ≠ practical importance"),
            ("Not correcting for multiple comparisons", "Leads to false discoveries"),
            ("Confusing correlation with causation", "Need controlled experiments"),
            ("Using one-tailed tests inappropriately", "Be honest about directional hypotheses"),
            ("Stopping experiment early without correction", "Inflates Type I error"),
            ("Not checking sample size/power", "Underpowered tests miss real effects"),
            ("Treating p=0.049 and p=0.051 very differently", "p-values are continuous!"),
            ("HARKing (Hypothesizing After Results Known)", "Pre-register hypotheses")
        ]
        
        for i, (mistake, solution) in enumerate(mistakes, 1):
            st.error(f"**{i}. {mistake}**")
            st.success(f"✓ {solution}")
            st.divider()
    
    with tab4:
        st.header("Final Tips for DS Interviews")
        
        tips = [
            "**Always start with exploratory analysis** - visualize before testing",
            "**State your assumptions explicitly** - shows you understand the test",
            "**Check assumptions** - normality, independence, equal variance",
            "**Report effect sizes** - not just p-values",
            "**Use appropriate corrections** - Bonferroni, Benjamini-Hochberg",
            "**Know when to use parametric vs non-parametric** - it's often asked!",
            "**Practice explaining in business terms** - 'Users in treatment group spent $5 more on average, with 95% confidence between $3-$7'",
            "**Understand A/B testing deeply** - most common DS application",
            "**Be comfortable with both R and Python syntax**",
            "**Know the difference between one-tailed and two-tailed tests**"
        ]
        
        for i, tip in enumerate(tips, 1):
            st.info(f"{i}. {tip}")
        
        st.success("""
        ### Remember: The Goal
        
        The goal isn't to memorize every test, but to:
        - ✅ Understand the intuition behind each test
        - ✅ Know when to use which test
        - ✅ Check assumptions before applying
        - ✅ Interpret results correctly
        - ✅ Communicate findings to non-technical stakeholders
        """)

# Footer
st.markdown("---")
st.caption("📊 Statistical Tests Workbook | Built with Streamlit | For Data Science Learning")
