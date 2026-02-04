import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="ML Foundations Workbook",
    page_icon="🤖",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("🤖 ML Foundations")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home",
        "📊 Linear Regression",
        "📈 Logistic Regression",
        "🌳 Decision Trees",
        "🌲 Random Forests",
        "🎯 Support Vector Machines",
        "🧠 Neural Networks Fundamentals",
        "⚡ Activation Functions",
        "📉 Loss Functions",
        "🎢 Gradient Descent Variants",
        "🎛️ Regularization Techniques",
        "✅ Model Evaluation",
        "🔄 Cross-Validation",
        "🎨 Feature Engineering",
        "📊 Dimensionality Reduction",
        "🔍 Hyperparameter Tuning",
        "💻 Interactive Lab",
        "📋 Quick Reference"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Every algorithm includes mathematical derivations, intuitions, and code!")

# Main Content
if section == "🏠 Home":
    st.title("🤖 MACHINE LEARNING FOUNDATIONS")
    st.subheader("Complete Mathematical and Conceptual Guide")
    
    st.markdown("""
    ### Welcome to the Complete ML Workbook! 👋
    
    This comprehensive resource covers **every fundamental concept** in machine learning with:
    - 📐 **Full Mathematical Derivations** - Understand the "why" behind every algorithm
    - 💡 **Clear Intuitions** - Grasp concepts before diving into math
    - 🔬 **Working Code Examples** - See theory in practice
    - 🎯 **Real Applications** - Know when and how to use each technique
    - ✍️ **Practice Problems** - Solidify your understanding
    
    ---
    
    ### 🗺️ Complete Coverage Map
    
    #### 1. Supervised Learning Algorithms
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Regression:**
        - Linear Regression (OLS, Gradient Descent)
        - Polynomial Regression
        - Ridge Regression (L2)
        - Lasso Regression (L1)
        - Elastic Net
        """)
    
    with col2:
        st.markdown("""
        **Classification:**
        - Logistic Regression
        - Decision Trees (CART, ID3, C4.5)
        - Random Forests
        - Support Vector Machines
        - Naive Bayes
        - k-Nearest Neighbors
        """)
    
    st.markdown("""
    #### 2. Neural Networks & Deep Learning Foundations
    
    - Perceptron
    - Multi-Layer Perceptron (MLP)
    - Backpropagation (Full Derivation)
    - Activation Functions (Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, GELU)
    - Weight Initialization Strategies
    - Batch Normalization
    
    #### 3. Optimization & Training
    
    **Gradient Descent Family:**
    - Batch Gradient Descent
    - Stochastic Gradient Descent (SGD)
    - Mini-Batch Gradient Descent
    - Momentum
    - Nesterov Accelerated Gradient
    - AdaGrad
    - RMSprop
    - Adam
    - AdamW
    - Learning Rate Schedules
    
    **Loss Functions:**
    - MSE, MAE, Huber Loss
    - Cross-Entropy (Binary & Categorical)
    - Hinge Loss
    - KL Divergence
    
    #### 4. Regularization & Generalization
    
    - L1 Regularization (Lasso)
    - L2 Regularization (Ridge)
    - Elastic Net
    - Dropout
    - Early Stopping
    - Data Augmentation
    - Batch Normalization
    
    #### 5. Model Evaluation & Selection
    
    - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
    - Confusion Matrix Analysis
    - Cross-Validation (k-fold, Stratified, Leave-One-Out)
    - Bias-Variance Tradeoff
    - Learning Curves
    - Validation Curves
    
    #### 6. Feature Engineering & Preprocessing
    
    - Scaling & Normalization
    - Encoding Categorical Variables
    - Feature Selection Methods
    - Dimensionality Reduction (PCA, t-SNE, LDA)
    - Feature Extraction
    
    #### 7. Advanced Topics
    
    - Ensemble Methods (Bagging, Boosting, Stacking)
    - Hyperparameter Optimization (Grid Search, Random Search, Bayesian)
    - Imbalanced Data Handling
    - Model Interpretability
    
    ---
    
    ### 🎯 How to Use This Workbook
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **📖 1. Learn**
        
        - Read intuitions
        - Study derivations
        - Understand assumptions
        """)
    
    with col2:
        st.info("""
        **💻 2. Practice**
        
        - Run code examples
        - Modify parameters
        - Visualize results
        """)
    
    with col3:
        st.warning("""
        **🔬 3. Experiment**
        
        - Interactive labs
        - Custom datasets
        - Compare algorithms
        """)
    
    st.markdown("---")
    st.markdown("👈 **Select a topic from the sidebar to begin your ML journey!**")

elif section == "📊 Linear Regression":
    st.title("📊 Linear Regression")
    st.markdown("*The foundation of supervised learning*")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Intuition", 
        "📐 Mathematics", 
        "💻 Implementation", 
        "🎯 Applications",
        "✍️ Practice"
    ])
    
    with tab1:
        st.header("What is Linear Regression?")
        
        st.info("""
        **INTUITION**: Linear regression finds the "best-fit" straight line through your data points.
        
        Think of it like finding the trend in data:
        - 📈 House prices vs square footage
        - 📊 Sales vs advertising spend
        - 🌡️ Temperature vs ice cream sales
        
        The goal: Predict a continuous output (y) from input features (x).
        """)
        
        st.markdown("""
        ### The Linear Model
        
        We model the relationship as:
        """)
        
        st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon")
        
        st.markdown("""
        Where:
        - **y** = target variable (what we want to predict)
        - **xᵢ** = features/predictors
        - **βᵢ** = coefficients/weights (what we learn)
        - **β₀** = intercept/bias term
        - **ε** = error term (noise)
        
        **In vector form**:
        """)
        
        st.latex(r"y = \mathbf{X}\boldsymbol{\beta} + \epsilon")
        
        st.markdown("""
        ### Simple Linear Regression (One Feature)
        
        The simplest case with just one feature:
        """)
        
        st.latex(r"y = \beta_0 + \beta_1 x")
        
        st.markdown("""
        - **β₀** = y-intercept (where line crosses y-axis)
        - **β₁** = slope (how much y changes when x increases by 1)
        """)
        
        # Interactive visualization
        st.markdown("---")
        st.subheader("📊 Interactive Visualization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Adjust the line:**")
            intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
            slope = st.slider("Slope (β₁)", -3.0, 3.0, 0.5, 0.1)
        
        with col2:
            # Generate sample data
            np.random.seed(42)
            x = np.linspace(0, 10, 50)
            y_true = 2 + 0.5 * x
            y_data = y_true + np.random.normal(0, 1, 50)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y_data, alpha=0.5, label='Data points')
            ax.plot(x, y_true, 'g--', label='True line (β₀=2, β₁=0.5)', linewidth=2)
            ax.plot(x, intercept + slope * x, 'r-', label=f'Your line (β₀={intercept}, β₁={slope})', linewidth=2)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title('Linear Regression Visualization', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Calculate and show MSE
            y_pred = intercept + slope * x
            mse = np.mean((y_data - y_pred)**2)
            st.metric("Mean Squared Error", f"{mse:.4f}")
        
        st.markdown("---")
        
        st.success("""
        **Key Insight**: The goal of linear regression is to find the values of β₀ and β₁ 
        that minimize the distance between the line and the data points!
        """)
        
        st.markdown("""
        ### Assumptions of Linear Regression
        
        For linear regression to work well, we assume:
        
        1. **Linearity**: The relationship between X and y is linear
        2. **Independence**: Observations are independent of each other
        3. **Homoscedasticity**: Constant variance of errors
        4. **Normality**: Errors are normally distributed
        5. **No multicollinearity**: Features are not highly correlated (for multiple regression)
        """)
    
    with tab2:
        st.header("📐 Mathematical Derivation")
        
        st.markdown("""
        ### The Cost Function
        
        We need a way to measure how "good" our line is. We use **Mean Squared Error (MSE)**:
        """)
        
        st.latex(r"J(\boldsymbol{\beta}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)})^2")
        
        st.markdown("""
        Where:
        - **m** = number of training examples
        - **h_β(x)** = hypothesis function = β₀ + β₁x₁ + ... + βₙxₙ
        - **y⁽ⁱ⁾** = actual value for example i
        - The **(1/2)** is just for mathematical convenience
        
        **In matrix form**:
        """)
        
        st.latex(r"J(\boldsymbol{\beta}) = \frac{1}{2m} (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})")
        
        st.markdown("---")
        
        st.subheader("Method 1: Normal Equation (Closed-Form Solution)")
        
        st.markdown("""
        We can solve for the optimal β analytically by setting the gradient to zero:
        """)
        
        st.latex(r"\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = 0")
        
        st.markdown("""
        **Step-by-step derivation**:
        
        Starting with:
        """)
        
        st.latex(r"J(\boldsymbol{\beta}) = \frac{1}{2m}(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})")
        
        st.markdown("Expand:")
        
        st.latex(r"""J(\boldsymbol{\beta}) = \frac{1}{2m}(\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} 
        - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \mathbf{y}^T\mathbf{y})""")
        
        st.markdown("Take gradient with respect to β:")
        
        st.latex(r"\nabla_{\boldsymbol{\beta}} J = \frac{1}{m}(\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} - \mathbf{X}^T\mathbf{y})")
        
        st.markdown("Set equal to zero and solve:")
        
        st.latex(r"\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}")
        
        st.success("""
        **Normal Equation (Final Result)**:
        """)
        
        st.latex(r"\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}")
        
        st.warning("""
        **Computational Complexity**: O(n³) where n is the number of features
        
        **Pros**:
        - Exact solution (no iterations needed)
        - No learning rate to tune
        
        **Cons**:
        - Slow when n is large (>10,000 features)
        - Requires X^T X to be invertible
        - Doesn't work for non-linear models
        """)
        
        st.markdown("---")
        
        st.subheader("Method 2: Gradient Descent")
        
        st.markdown("""
        Instead of solving analytically, we can iteratively improve our estimate:
        
        **Algorithm**:
        1. Start with random β
        2. Calculate gradient of cost function
        3. Update β in the opposite direction of gradient
        4. Repeat until convergence
        
        **Update Rule**:
        """)
        
        st.latex(r"\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\boldsymbol{\beta})")
        
        st.markdown("Computing the partial derivative:")
        
        st.latex(r"\frac{\partial}{\partial \beta_j} J(\boldsymbol{\beta}) = \frac{1}{m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}")
        
        st.markdown("**Vectorized form**:")
        
        st.latex(r"\boldsymbol{\beta} := \boldsymbol{\beta} - \frac{\alpha}{m} \mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})")
        
        st.markdown("""
        Where:
        - **α** = learning rate (step size)
        - Repeat until convergence (gradient ≈ 0)
        """)
        
        st.info("""
        **Why Gradient Descent?**
        - Works even with millions of features
        - Can be adapted for online learning
        - Foundation for neural networks
        - Can be parallelized
        """)
    
    with tab3:
        st.header("💻 Implementation from Scratch")
        
        st.markdown("### Method 1: Normal Equation")
        
        st.code("""
import numpy as np

class LinearRegressionNormal:
    def __init__(self):
        self.beta = None
    
    def fit(self, X, y):
        \"\"\"
        Fit using normal equation: β = (X^T X)^(-1) X^T y
        
        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Target values
        \"\"\"
        # Add bias term (column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation
        self.beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        return self
    
    def predict(self, X):
        \"\"\"Make predictions\"\"\"
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta
    
    def score(self, X, y):
        \"\"\"Calculate R² score\"\"\"
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegressionNormal()
model.fit(X, y)

print(f"Coefficients: {model.beta}")
print(f"R² Score: {model.score(X, y):.4f}")
        """, language="python")
        
        st.markdown("---")
        
        st.markdown("### Method 2: Gradient Descent")
        
        st.code("""
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        \"\"\"
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence threshold
        \"\"\"
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tol = tolerance
        self.beta = None
        self.cost_history = []
    
    def fit(self, X, y):
        \"\"\"
        Fit using gradient descent
        \"\"\"
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        
        # Initialize parameters randomly
        self.beta = np.random.randn(n)
        
        # Gradient descent
        for iteration in range(self.n_iter):
            # Compute predictions
            y_pred = X_b @ self.beta
            
            # Compute cost (MSE)
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
            
            # Compute gradient
            gradient = (1/m) * X_b.T @ (y_pred - y)
            
            # Update parameters
            self.beta -= self.lr * gradient
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
        
        return self
    
    def predict(self, X):
        \"\"\"Make predictions\"\"\"
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta
    
    def plot_cost_history(self):
        \"\"\"Plot cost over iterations\"\"\"
        import matplotlib.pyplot as plt
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Cost Function over Iterations')
        plt.grid(True)
        plt.show()

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print(f"Coefficients: {model.beta}")
print(f"Final Cost: {model.cost_history[-1]:.6f}")
        """, language="python")
        
        st.markdown("---")
        
        st.markdown("### Comparison with scikit-learn")
        
        st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

# Make predictions
y_pred = sklearn_model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Intercept: {sklearn_model.intercept_:.4f}")
print(f"Coefficients: {sklearn_model.coef_}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
        """, language="python")
        
        st.success("""
        **Pro Tip**: Use the Normal Equation for small datasets (< 10,000 features), 
        and Gradient Descent for larger datasets or when you need online learning.
        """)
    
    with tab4:
        st.header("🎯 Real-World Applications")
        
        st.markdown("""
        ### When to Use Linear Regression
        
        ✅ **Good for**:
        - Predicting continuous values
        - Understanding feature relationships
        - When relationships are approximately linear
        - When interpretability matters
        - Baseline model for comparison
        
        ❌ **Not good for**:
        - Complex non-linear relationships
        - Classification problems (use logistic regression)
        - When assumptions are violated
        - Outliers are present (use robust regression)
        """)
        
        st.markdown("---")
        
        st.subheader("Common Use Cases")
        
        use_cases = {
            "Application": [
                "House Price Prediction",
                "Sales Forecasting",
                "Risk Assessment",
                "Medical Dosage",
                "Energy Consumption",
                "Stock Price Trends"
            ],
            "Input Features": [
                "Square footage, bedrooms, location",
                "Advertising spend, seasonality",
                "Credit score, income, debt",
                "Age, weight, medical history",
                "Temperature, building size",
                "Market indicators, company metrics"
            ],
            "Target": [
                "Price ($)",
                "Revenue ($)",
                "Default probability",
                "Optimal dosage (mg)",
                "kWh consumed",
                "Price change"
            ]
        }
        
        st.table(pd.DataFrame(use_cases))
        
        st.markdown("---")
        
        st.subheader("Feature Engineering Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Polynomial Features**:
            - Add x², x³ for curves
            - Creates non-linear model
            - Example: y = β₀ + β₁x + β₂x²
            """)
            
            st.markdown("""
            **Interaction Terms**:
            - Multiply features together
            - Captures combined effects
            - Example: area × bedrooms
            """)
        
        with col2:
            st.markdown("""
            **Log Transformations**:
            - For skewed data
            - Makes relationship linear
            - Example: log(price) vs area
            """)
            
            st.markdown("""
            **Standardization**:
            - Scale features to mean=0, std=1
            - Helps gradient descent converge
            - Makes coefficients comparable
            """)
    
    with tab5:
        st.header("✍️ Practice Problems")
        
        st.markdown("""
        **Problem 1**: Derive the gradient of the MSE cost function.
        
        Given: $J(\\boldsymbol{\\beta}) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_{\\boldsymbol{\\beta}}(\\mathbf{x}^{(i)}) - y^{(i)})^2$
        
        Find: $\\frac{\\partial J}{\\partial \\beta_j}$
        """)
        
        with st.expander("💡 Show Solution"):
            st.markdown("""
            **Step 1**: Expand the hypothesis function
            
            $h_{\\boldsymbol{\\beta}}(\\mathbf{x}) = \\beta_0 + \\beta_1 x_1 + ... + \\beta_n x_n$
            
            **Step 2**: Apply chain rule
            
            $\\frac{\\partial J}{\\partial \\beta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (h_{\\boldsymbol{\\beta}}(\\mathbf{x}^{(i)}) - y^{(i)}) \\frac{\\partial h_{\\boldsymbol{\\beta}}}{\\partial \\beta_j}$
            
            **Step 3**: Note that $\\frac{\\partial h_{\\boldsymbol{\\beta}}}{\\partial \\beta_j} = x_j^{(i)}$
            
            **Final Result**:
            
            $\\frac{\\partial J}{\\partial \\beta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (h_{\\boldsymbol{\\beta}}(\\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}$
            """)
        
        st.markdown("---")
        
        st.markdown("""
        **Problem 2**: Implement mini-batch gradient descent.
        
        Modify the gradient descent implementation to use mini-batches of size 32.
        """)
        
        with st.expander("💡 Show Solution"):
            st.code("""
import numpy as np

def mini_batch_gradient_descent(X, y, learning_rate=0.01, 
                                n_epochs=100, batch_size=32):
    \"\"\"
    Mini-batch gradient descent for linear regression
    \"\"\"
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias
    beta = np.random.randn(n + 1)
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_b[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute gradient on batch
            gradient = (1/batch_size) * X_batch.T @ (X_batch @ beta - y_batch)
            
            # Update parameters
            beta -= learning_rate * gradient
    
    return beta
            """, language="python")

elif section == "📈 Logistic Regression":
    st.title("📈 Logistic Regression")
    st.markdown("*From regression to classification*")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Intuition",
        "📐 Mathematics", 
        "💻 Implementation",
        "✍️ Practice"
    ])
    
    with tab1:
        st.header("What is Logistic Regression?")
        
        st.info("""
        **INTUITION**: Despite its name, logistic regression is used for **classification**, not regression!
        
        It predicts the **probability** that an instance belongs to a particular class.
        
        **Examples**:
        - Email: Spam or Not Spam? 🔄
        - Medical: Disease or No Disease? 🏥
        - Customer: Will Churn or Stay? 👤
        - Transaction: Fraud or Legitimate? 💳
        """)
        
        st.markdown("""
        ### Why Not Use Linear Regression for Classification?
        
        **Problems with linear regression**:
        1. Outputs can be < 0 or > 1 (not valid probabilities!)
        2. Not robust to outliers
        3. Assumes linear relationship
        
        **Logistic regression solves this** by:
        - Squashing output between 0 and 1 using sigmoid function
        - Outputting probabilities, not raw values
        - Using maximum likelihood estimation
        """)
        
        st.markdown("---")
        
        st.subheader("The Sigmoid Function")
        
        st.markdown("""
        The **sigmoid** (or logistic) function transforms any real number to [0, 1]:
        """)
        
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        
        # Plot sigmoid
        z = np.linspace(-10, 10, 100)
        sigmoid = 1 / (1 + np.exp(-z))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(z, sigmoid, linewidth=3, color='blue')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision boundary (p=0.5)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('z', fontsize=12)
        ax.set_ylabel('σ(z)', fontsize=12)
        ax.set_title('Sigmoid Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("""
        **Properties**:
        - σ(0) = 0.5
        - σ(z) → 1 as z → ∞
        - σ(z) → 0 as z → -∞
        - Derivative: σ'(z) = σ(z)(1 - σ(z))
        """)
        
        st.markdown("---")
        
        st.subheader("The Logistic Regression Model")
        
        st.markdown("""
        **Hypothesis function**:
        """)
        
        st.latex(r"h_{\boldsymbol{\beta}}(\mathbf{x}) = \sigma(\boldsymbol{\beta}^T\mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T\mathbf{x}}}")
        
        st.markdown("""
        This gives us P(y=1|x), the probability that y=1 given features x.
        
        **Decision rule**:
        - If P(y=1|x) ≥ 0.5, predict class 1
        - If P(y=1|x) < 0.5, predict class 0
        
        **Equivalently**: Predict class 1 if β^T x ≥ 0
        """)
    
    with tab2:
        st.header("📐 Mathematical Derivation")
        
        st.markdown("""
        ### Maximum Likelihood Estimation
        
        Unlike linear regression (which uses MSE), we use **maximum likelihood estimation**.
        
        **Likelihood of one example**:
        """)
        
        st.latex(r"P(y^{(i)}|\mathbf{x}^{(i)}; \boldsymbol{\beta}) = h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})^{y^{(i)}} (1-h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}))^{1-y^{(i)}}")
        
        st.markdown("""
        **Likelihood of all examples** (assuming independence):
        """)
        
        st.latex(r"L(\boldsymbol{\beta}) = \prod_{i=1}^{m} h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})^{y^{(i)}} (1-h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}))^{1-y^{(i)}}")
        
        st.markdown("""
        **Log-likelihood** (easier to work with):
        """)
        
        st.latex(r"\ell(\boldsymbol{\beta}) = \sum_{i=1}^{m} y^{(i)} \log(h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})) + (1-y^{(i)}) \log(1-h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}))")
        
        st.markdown("---")
        
        st.subheader("Binary Cross-Entropy Loss")
        
        st.markdown("""
        We want to **maximize** log-likelihood, which is equivalent to **minimizing** the negative log-likelihood:
        """)
        
        st.latex(r"J(\boldsymbol{\beta}) = -\frac{1}{m}\ell(\boldsymbol{\beta}) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})) + (1-y^{(i)}) \log(1-h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}))]")
        
        st.success("""
        This is the **Binary Cross-Entropy Loss** (or Log Loss)!
        """)
        
        st.markdown("---")
        
        st.subheader("Gradient of the Cost Function")
        
        st.markdown("""
        **Derivation**: Let h = h_β(x) = σ(β^T x)
        
        We need: $\\frac{\\partial J}{\\partial \\beta_j}$
        
        **Step 1**: Derivative of sigmoid
        """)
        
        st.latex(r"\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1-\sigma(z))")
        
        st.markdown("**Step 2**: Apply chain rule")
        
        st.latex(r"\frac{\partial}{\partial \beta_j} h_{\boldsymbol{\beta}}(\mathbf{x}) = h_{\boldsymbol{\beta}}(\mathbf{x})(1-h_{\boldsymbol{\beta}}(\mathbf{x})) x_j")
        
        st.markdown("**Step 3**: Compute gradient of cost")
        
        st.markdown("After algebra (details omitted), we get:")
        
        st.latex(r"\frac{\partial J}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}")
        
        st.warning("""
        **Amazing fact**: This has the SAME form as linear regression gradient!
        
        The only difference is that h is now the sigmoid function instead of a linear function.
        """)
        
        st.markdown("---")
        
        st.subheader("Gradient Descent Update")
        
        st.latex(r"\beta_j := \beta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}")
        
        st.markdown("**Vectorized**:")
        
        st.latex(r"\boldsymbol{\beta} := \boldsymbol{\beta} - \frac{\alpha}{m} \mathbf{X}^T(\sigma(\mathbf{X}\boldsymbol{\beta}) - \mathbf{y})")
    
    with tab3:
        st.header("💻 Implementation from Scratch")
        
        st.code("""
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.beta = None
        self.cost_history = []
    
    def sigmoid(self, z):
        \"\"\"Sigmoid activation function\"\"\"
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        \"\"\"
        Train logistic regression model using gradient descent
        
        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Target values (0 or 1)
        \"\"\"
        # Add bias term
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        
        # Initialize parameters
        self.beta = np.zeros(n + 1)
        
        # Gradient descent
        for i in range(self.n_iter):
            # Forward pass
            z = X_b @ self.beta
            h = self.sigmoid(z)
            
            # Compute cost (binary cross-entropy)
            epsilon = 1e-15  # To prevent log(0)
            cost = -np.mean(y * np.log(h + epsilon) + (1-y) * np.log(1-h + epsilon))
            self.cost_history.append(cost)
            
            # Compute gradient
            gradient = (1/m) * X_b.T @ (h - y)
            
            # Update parameters
            self.beta -= self.lr * gradient
        
        return self
    
    def predict_proba(self, X):
        \"\"\"Predict class probabilities\"\"\"
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b @ self.beta)
    
    def predict(self, X, threshold=0.5):
        \"\"\"Predict class labels\"\"\"
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        \"\"\"Calculate accuracy\"\"\"
        return np.mean(self.predict(X) == y)

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                          n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Coefficients: {model.beta}")
        """, language="python")
        
        st.markdown("---")
        
        st.subheader("Multiclass Logistic Regression (One-vs-Rest)")
        
        st.code("""
class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.models = []
        self.classes = None
    
    def fit(self, X, y):
        \"\"\"
        Train K binary classifiers (one per class)
        \"\"\"
        self.classes = np.unique(y)
        
        # Train one classifier per class
        for c in self.classes:
            # Create binary labels (1 if class c, 0 otherwise)
            y_binary = (y == c).astype(int)
            
            # Train binary classifier
            model = LogisticRegression(self.lr, self.n_iter)
            model.fit(X, y_binary)
            
            self.models.append(model)
        
        return self
    
    def predict_proba(self, X):
        \"\"\"Predict probabilities for all classes\"\"\"
        # Get probabilities from each binary classifier
        probas = np.array([model.predict_proba(X) for model in self.models]).T
        
        # Normalize to sum to 1
        return probas / probas.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        \"\"\"Predict class with highest probability\"\"\"
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
        """, language="python")
    
    with tab4:
        st.header("✍️ Practice Problems")
        
        st.markdown("""
        **Problem 1**: Prove that the derivative of the sigmoid function is σ(z)(1 - σ(z)).
        """)
        
        with st.expander("💡 Show Solution"):
            st.markdown("""
            **Given**: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
            
            **Find**: $\\frac{d\\sigma}{dz}$
            
            **Solution**:
            
            Let $u = 1 + e^{-z}$, so $\\sigma = \\frac{1}{u} = u^{-1}$
            
            By chain rule:
            
            $\\frac{d\\sigma}{dz} = \\frac{d\\sigma}{du} \\cdot \\frac{du}{dz}$
            
            $\\frac{d\\sigma}{du} = -u^{-2} = -\\frac{1}{(1 + e^{-z})^2}$
            
            $\\frac{du}{dz} = -e^{-z}$
            
            Therefore:
            
            $\\frac{d\\sigma}{dz} = -\\frac{1}{(1 + e^{-z})^2} \\cdot (-e^{-z}) = \\frac{e^{-z}}{(1 + e^{-z})^2}$
            
            Simplify:
            
            $= \\frac{1}{1 + e^{-z}} \\cdot \\frac{e^{-z}}{1 + e^{-z}}$
            
            $= \\frac{1}{1 + e^{-z}} \\cdot \\frac{1 + e^{-z} - 1}{1 + e^{-z}}$
            
            $= \\frac{1}{1 + e^{-z}} \\cdot (1 - \\frac{1}{1 + e^{-z}})$
            
            $= \\sigma(z)(1 - \\sigma(z))$ ✓
            """)

elif section == "⚡ Activation Functions":
    st.title("⚡ Activation Functions")
    st.markdown("*The non-linearity in neural networks*")
    
    tab1, tab2, tab3 = st.tabs(["📖 Overview", "📐 Mathematics", "🎯 Usage Guide"])
    
    with tab1:
        st.header("Why Activation Functions?")
        
        st.info("""
        **INTUITION**: Activation functions introduce **non-linearity** into neural networks.
        
        Without them:
        - Multiple linear layers = one linear layer
        - Can't learn complex patterns
        - Network is just linear regression!
        
        With activation functions:
        - Can approximate any function (Universal Approximation Theorem)
        - Learn complex, non-linear relationships
        - Create deep hierarchical representations
        """)
        
        st.markdown("---")
        
        st.subheader("Interactive Comparison")
        
        x = np.linspace(-5, 5, 1000)
        
        # Compute all activation functions
        activations = {
            'Sigmoid': 1 / (1 + np.exp(-x)),
            'Tanh': np.tanh(x),
            'ReLU': np.maximum(0, x),
            'Leaky ReLU': np.where(x > 0, x, 0.01 * x),
            'ELU': np.where(x > 0, x, 1.0 * (np.exp(x) - 1)),
            'Swish': x * (1 / (1 + np.exp(-x))),
            'GELU': 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        }
        
        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (name, y) in enumerate(activations.items()):
            axes[idx].plot(x, y, linewidth=2.5, color=f'C{idx}')
            axes[idx].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            axes[idx].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([-5, 5])
        
        # Hide extra subplots
        for idx in range(len(activations), 9):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.header("📐 Complete Mathematical Reference")
        
        # Sigmoid
        st.subheader("1. Sigmoid (Logistic)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function**:")
            st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}}")
            
            st.markdown("**Derivative**:")
            st.latex(r"\sigma'(x) = \sigma(x)(1 - \sigma(x))")
            
            st.markdown("**Range**: (0, 1)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - Smooth gradient
            - Output interpretable as probability
            - Historic significance
            
            **Cons**:
            - ❌ Vanishing gradient problem
            - ❌ Not zero-centered
            - ❌ Computationally expensive (exp)
            
            **Use**: Output layer for binary classification
            """)
        
        st.markdown("---")
        
        # Tanh
        st.subheader("2. Tanh (Hyperbolic Tangent)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function**:")
            st.latex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}")
            
            st.markdown("**Derivative**:")
            st.latex(r"\tanh'(x) = 1 - \tanh^2(x)")
            
            st.markdown("**Range**: (-1, 1)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - ✅ Zero-centered (better than sigmoid)
            - Stronger gradients than sigmoid
            
            **Cons**:
            - ❌ Still has vanishing gradient
            - ❌ Computationally expensive
            
            **Use**: Hidden layers in RNNs, small networks
            
            **Relation to Sigmoid**: tanh(x) = 2σ(2x) - 1
            """)
        
        st.markdown("---")
        
        # ReLU
        st.subheader("3. ReLU (Rectified Linear Unit)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function**:")
            st.latex(r"\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}")
            
            st.markdown("**Derivative**:")
            st.latex(r"\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}")
            
            st.markdown("**Range**: [0, ∞)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - ✅ Simple and fast to compute
            - ✅ No vanishing gradient for x > 0
            - ✅ Sparse activations (many zeros)
            - ✅ Accelerates convergence
            
            **Cons**:
            - ❌ Dying ReLU problem (neurons can die)
            - ❌ Not zero-centered
            
            **Use**: **Default choice** for hidden layers in CNNs, MLPs
            """)
        
        st.markdown("---")
        
        # Leaky ReLU
        st.subheader("4. Leaky ReLU")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function** (α = 0.01 typically):")
            st.latex(r"\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}")
            
            st.markdown("**Derivative**:")
            st.latex(r"\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}")
            
            st.markdown("**Range**: (-∞, ∞)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - ✅ Fixes dying ReLU problem
            - ✅ Small gradient for x < 0
            - ✅ Computationally cheap
            
            **Cons**:
            - Not always better than ReLU
            - Need to tune α
            
            **Use**: When ReLU causes dying neurons
            
            **Variants**: PReLU (learnable α), RReLU (random α)
            """)
        
        st.markdown("---")
        
        # ELU
        st.subheader("5. ELU (Exponential Linear Unit)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function** (α = 1.0 typically):")
            st.latex(r"\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}")
            
            st.markdown("**Derivative**:")
            st.latex(r"\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}")
            
            st.markdown("**Range**: (-α, ∞)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - ✅ No dying ReLU problem
            - ✅ Mean activation closer to zero
            - ✅ Smooth everywhere
            - ✅ Can produce negative outputs
            
            **Cons**:
            - ❌ Computationally expensive (exp)
            - ❌ Slower than ReLU
            
            **Use**: When you need smooth gradients, better than ReLU for some tasks
            """)
        
        st.markdown("---")
        
        # Swish
        st.subheader("6. Swish (Self-Gated)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function**:")
            st.latex(r"\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}")
            
            st.markdown("**Derivative**:")
            st.latex(r"\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))")
            st.latex(r"= \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x))")
            
            st.markdown("**Range**: (-∞, ∞)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - ✅ Smooth, non-monotonic
            - ✅ Self-gating property
            - ✅ Often outperforms ReLU
            - ✅ Unbounded above, bounded below
            
            **Cons**:
            - ❌ Computationally expensive
            - ❌ Not always better than ReLU
            
            **Use**: State-of-the-art networks (discovered by Google)
            """)
        
        st.markdown("---")
        
        # GELU
        st.subheader("7. GELU (Gaussian Error Linear Unit)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function** (exact):")
            st.latex(r"\text{GELU}(x) = x \cdot \Phi(x)")
            st.markdown("where Φ(x) is the Gaussian CDF")
            
            st.markdown("**Approximation**:")
            st.latex(r"\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)")
            
            st.markdown("**Range**: (-∞, ∞)")
        
        with col2:
            st.markdown("""
            **Pros**:
            - ✅ Smooth, probabilistic interpretation
            - ✅ Used in BERT, GPT
            - ✅ State-of-the-art performance
            - ✅ Stochastic regularization
            
            **Cons**:
            - ❌ Computationally expensive
            - ❌ Complex to implement exactly
            
            **Use**: **Transformers**, modern NLP models (BERT, GPT-3)
            """)
        
        st.markdown("---")
        
        # Softmax
        st.subheader("8. Softmax (Multi-class)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function** (for vector z):")
            st.latex(r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}")
            
            st.markdown("**Derivative** (Jacobian):")
            st.latex(r"\frac{\partial \text{Softmax}(z_i)}{\partial z_j} = \begin{cases} \text{Softmax}(z_i)(1 - \text{Softmax}(z_i)) & \text{if } i = j \\ -\text{Softmax}(z_i)\text{Softmax}(z_j) & \text{if } i \neq j \end{cases}")
            
            st.markdown("**Range**: (0, 1), Σ = 1")
        
        with col2:
            st.markdown("""
            **Properties**:
            - Outputs sum to 1 (probability distribution)
            - Exponential emphasizes large values
            - Differentiable
            
            **Use**: 
            - ✅ **Output layer for multi-class classification**
            - Works with cross-entropy loss
            - Temperature parameter for controlling "sharpness"
            
            **Note**: Never use in hidden layers!
            """)
    
    with tab3:
        st.header("🎯 How to Choose Activation Functions")
        
        st.subheader("Decision Guide")
        
        decision_guide = {
            "Layer Type": [
                "Hidden Layers (CNNs)",
                "Hidden Layers (RNNs/LSTMs)",
                "Hidden Layers (Transformers)",
                "Output Layer (Binary Classification)",
                "Output Layer (Multi-class)",
                "Output Layer (Regression)"
            ],
            "Recommended": [
                "ReLU",
                "Tanh",
                "GELU",
                "Sigmoid",
                "Softmax",
                "Linear (None)"
            ],
            "Alternatives": [
                "Leaky ReLU, ELU, Swish",
                "ReLU, GRU activation",
                "Swish, ReLU",
                "None",
                "None",
                "ReLU (if outputs must be positive)"
            ],
            "Why": [
                "Fast, sparse, works well",
                "Zero-centered, smooth gradients",
                "State-of-the-art for NLP",
                "Outputs probabilities [0,1]",
                "Outputs probability distribution",
                "Direct value prediction"
            ]
        }
        
        st.table(pd.DataFrame(decision_guide))
        
        st.markdown("---")
        
        st.subheader("Common Problems & Solutions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Problem**: Vanishing Gradients
            - Symptoms: Training stalls, early layers don't learn
            - Solution: Use ReLU instead of Sigmoid/Tanh
            - Alternative: Batch normalization, skip connections
            
            **Problem**: Dying ReLU
            - Symptoms: Many neurons output 0, no learning
            - Solution: Use Leaky ReLU or ELU
            - Alternative: Lower learning rate, better initialization
            
            **Problem**: Exploding Activations
            - Symptoms: Activations become very large
            - Solution: Batch normalization, gradient clipping
            - Alternative: Lower learning rate
            """)
        
        with col2:
            st.markdown("""
            **Problem**: Slow Convergence
            - Symptoms: Training takes forever
            - Solution: Try Swish or GELU
            - Alternative: Better optimizer (Adam), higher learning rate
            
            **Problem**: Outputs Not in Range
            - Symptoms: Predictions out of valid range
            - Solution: Use appropriate output activation
            - Alternative: Post-processing, clipping
            
            **Problem**: Computational Cost
            - Symptoms: Training too slow
            - Solution: Use ReLU instead of Swish/GELU/ELU
            - Alternative: Reduce model size, use GPUs
            """)
        
        st.markdown("---")
        
        st.subheader("Implementation Tips")
        
        st.code("""
import numpy as np

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, ActivationFunctions.elu(x, alpha) + alpha)
    
    @staticmethod
    def swish(x):
        return x * (1 / (1 + np.exp(-x)))
    
    @staticmethod
    def swish_derivative(x):
        sigmoid = 1 / (1 + np.exp(-x))
        swish = x * sigmoid
        return swish + sigmoid * (1 - swish)
    
    @staticmethod
    def gelu(x):
        # Approximation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x):
        # Numerical stability: subtract max
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        """, language="python")

elif section == "🎢 Gradient Descent Variants":
    st.title("🎢 Gradient Descent Variants")
    st.markdown("*Optimizing the optimization*")
    
    tab1, tab2, tab3 = st.tabs(["📖 Overview", "📐 Mathematics", "💻 Implementation"])
    
    with tab1:
        st.header("The Gradient Descent Family")
        
        st.info("""
        **INTUITION**: Gradient descent is like hiking down a mountain in fog.
        
        You can only see your immediate surroundings (the gradient), and you want to reach 
        the bottom (minimum loss) as efficiently as possible.
        
        Different variants use different strategies for taking steps!
        """)
        
        st.markdown("""
        ### Why So Many Variants?
        
        **Standard gradient descent has problems**:
        - Can be slow to converge
        - Can get stuck in local minima or saddle points
        - Struggles with ravines (different scales in different dimensions)
        - Treats all parameters equally
        
        **Modern variants solve these issues** through:
        - Adaptive learning rates
        - Momentum to escape saddle points
        - Per-parameter learning rates
        - Better convergence guarantees
        """)
        
        st.markdown("---")
        
        st.subheader("Quick Comparison")
        
        comparison = {
            "Algorithm": [
                "Batch GD",
                "SGD",
                "Mini-batch GD",
                "Momentum",
                "Nesterov",
                "AdaGrad",
                "RMSprop",
                "Adam",
                "AdamW"
            ],
            "Speed": [
                "Slow",
                "Fast",
                "Fast",
                "Fast",
                "Fast",
                "Medium",
                "Fast",
                "Fast",
                "Fast"
            ],
            "Memory": [
                "Low",
                "Low",
                "Low",
                "Medium",
                "Medium",
                "High",
                "Medium",
                "High",
                "High"
            ],
            "Hyperparameters": [
                "1 (α)",
                "1 (α)",
                "2 (α, batch)",
                "2 (α, β)",
                "2 (α, β)",
                "1 (α)",
                "2 (α, β)",
                "3 (α, β₁, β₂)",
                "4 (α, β₁, β₂, λ)"
            ],
            "Best For": [
                "Small datasets",
                "Online learning",
                "Most cases",
                "Deep networks",
                "Deep networks",
                "Sparse features",
                "RNNs",
                "Most cases (default)",
                "Transformers, modern NLP"
            ]
        }
        
        st.table(pd.DataFrame(comparison))
    
    with tab2:
        st.header("📐 Mathematical Formulations")
        
        # Batch GD
        st.subheader("1. Batch Gradient Descent")
        
        st.markdown("""
        **Update Rule**:
        """)
        
        st.latex(r"\theta := \theta - \alpha \nabla_\theta J(\theta)")
        
        st.markdown("""
        Where:
        - θ = parameters
        - α = learning rate
        - ∇J(θ) = gradient of cost over **entire dataset**
        
        **Pros**: Stable, deterministic
        **Cons**: Slow for large datasets, can get stuck in local minima
        """)
        
        st.markdown("---")
        
        # SGD
        st.subheader("2. Stochastic Gradient Descent (SGD)")
        
        st.markdown("""
        **Update Rule** (for each example):
        """)
        
        st.latex(r"\theta := \theta - \alpha \nabla_\theta J(\theta; x^{(i)}, y^{(i)})")
        
        st.markdown("""
        Where gradient is computed on **single example** at a time.
        
        **Pros**: Fast, can escape local minima
        **Cons**: Noisy updates, may not converge exactly
        """)
        
        st.markdown("---")
        
        # Mini-batch
        st.subheader("3. Mini-Batch Gradient Descent")
        
        st.markdown("""
        **Update Rule** (for each mini-batch):
        """)
        
        st.latex(r"\theta := \theta - \alpha \nabla_\theta J(\theta; x^{(i:i+b)}, y^{(i:i+b)})")
        
        st.markdown("""
        Where b = batch size (typically 32, 64, 128, or 256).
        
        **Best of both worlds**: Fast and stable
        
        **This is the standard** in practice!
        """)
        
        st.markdown("---")
        
        # Momentum
        st.subheader("4. Momentum")
        
        st.success("""
        **KEY IDEA**: Add inertia to gradient descent, like a ball rolling downhill.
        """)
        
        st.markdown("""
        **Update Rules**:
        """)
        
        st.latex(r"v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta J(\theta)")
        st.latex(r"\theta := \theta - \alpha v_t")
        
        st.markdown("""
        Where:
        - v = velocity (exponentially weighted average of gradients)
        - β = momentum coefficient (typically 0.9)
        
        **Intuition**: Current update is combination of:
        - Current gradient (1-β)
        - Previous velocity (β)
        
        **Benefits**:
        - Accelerates convergence
        - Dampens oscillations
        - Helps escape local minima and saddle points
        """)
        
        st.markdown("---")
        
        # Nesterov
        st.subheader("5. Nesterov Accelerated Gradient (NAG)")
        
        st.markdown("""
        **Update Rules**:
        """)
        
        st.latex(r"v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta J(\theta - \alpha \beta v_{t-1})")
        st.latex(r"\theta := \theta - \alpha v_t")
        
        st.markdown("""
        **Key Difference from Momentum**: 
        - Compute gradient at "lookahead" position (θ - αβv_{t-1})
        - More intelligent momentum
        
        **Intuition**: "Look before you leap"
        - First make a jump based on previous velocity
        - Then compute gradient at that position
        - Correct the velocity based on that gradient
        
        **Benefits**: Often faster convergence than standard momentum
        """)
        
        st.markdown("---")
        
        # AdaGrad
        st.subheader("6. AdaGrad (Adaptive Gradient)")
        
        st.success("""
        **KEY IDEA**: Different learning rates for different parameters.
        """)
        
        st.markdown("""
        **Update Rules**:
        """)
        
        st.latex(r"G_t = G_{t-1} + (\nabla_\theta J(\theta))^2")
        st.latex(r"\theta := \theta - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta)")
        
        st.markdown("""
        Where:
        - G_t = sum of squared gradients up to time t
        - ε = small constant for numerical stability (10^-8)
        
        **Intuition**:
        - Parameters with large gradients get smaller learning rates
        - Parameters with small gradients get larger learning rates
        
        **Benefits**:
        - No need to manually tune learning rate
        - Good for sparse features
        
        **Problem**: Learning rate decays too aggressively (G_t always increases)
        """)
        
        st.markdown("---")
        
        # RMSprop
        st.subheader("7. RMSprop (Root Mean Square Propagation)")
        
        st.markdown("""
        **Update Rules**:
        """)
        
        st.latex(r"E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) (\nabla_\theta J(\theta))^2")
        st.latex(r"\theta := \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta J(\theta)")
        
        st.markdown("""
        Where:
        - E[g²]_t = exponentially weighted moving average of squared gradients
        - β = decay rate (typically 0.9 or 0.99)
        
        **Key Difference from AdaGrad**: Uses moving average instead of sum
        
        **Benefits**:
        - Fixes AdaGrad's aggressive decay
        - Works well for RNNs
        - Adaptive per-parameter learning rates
        """)
        
        st.markdown("---")
        
        # Adam
        st.subheader("8. Adam (Adaptive Moment Estimation)")
        
        st.success("""
        **Most Popular Optimizer!**
        
        Combines the best of Momentum and RMSprop.
        """)
        
        st.markdown("""
        **Update Rules**:
        """)
        
        st.latex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J(\theta)")
        st.latex(r"v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta J(\theta))^2")
        
        st.markdown("**Bias correction** (important for early iterations):")
        
        st.latex(r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}")
        st.latex(r"\hat{v}_t = \frac{v_t}{1-\beta_2^t}")
        
        st.markdown("**Parameter update**:")
        
        st.latex(r"\theta := \theta - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t")
        
        st.markdown("""
        Where:
        - m_t = first moment (mean of gradients) - like Momentum
        - v_t = second moment (variance of gradients) - like RMSprop
        - β₁ = 0.9 (typically)
        - β₂ = 0.999 (typically)
        - α = 0.001 (typically)
        - ε = 10^-8
        
        **Why it works**:
        - m_t provides momentum
        - v_t provides adaptive learning rates
        - Bias correction fixes initialization
        
        **When to use**: **Default choice** for most problems!
        """)
        
        st.markdown("---")
        
        # AdamW
        st.subheader("9. AdamW (Adam with Decoupled Weight Decay)")
        
        st.markdown("""
        **Update Rules**:
        """)
        
        st.latex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J(\theta)")
        st.latex(r"v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta J(\theta))^2")
        st.latex(r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}")
        st.latex(r"\theta := \theta - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta\right)")
        
        st.markdown("""
        **Key Difference from Adam**: 
        - Separates weight decay from gradient update
        - Weight decay term: λθ (not affected by adaptive learning rate)
        
        **Benefits**:
        - Better generalization than Adam
        - Proper L2 regularization
        - **Recommended for Transformers and modern NLP**
        
        **Typical values**:
        - λ = 0.01 (weight decay)
        - Same β₁, β₂ as Adam
        """)
    
    with tab3:
        st.header("💻 Implementation")
        
        st.code("""
import numpy as np

class Optimizers:
    \"\"\"Collection of optimization algorithms\"\"\"
    
    @staticmethod
    def sgd(params, grads, learning_rate=0.01):
        \"\"\"
        Stochastic Gradient Descent
        
        params: current parameters
        grads: gradients
        \"\"\"
        return params - learning_rate * grads
    
    @staticmethod
    def momentum(params, grads, velocity, learning_rate=0.01, beta=0.9):
        \"\"\"
        SGD with Momentum
        
        velocity: momentum vector (initialized to zeros)
        beta: momentum coefficient
        \"\"\"
        velocity = beta * velocity + (1 - beta) * grads
        params = params - learning_rate * velocity
        return params, velocity
    
    @staticmethod
    def nesterov(params, grads_fn, velocity, learning_rate=0.01, beta=0.9):
        \"\"\"
        Nesterov Accelerated Gradient
        
        grads_fn: function to compute gradients at given params
        \"\"\"
        # Look ahead
        params_ahead = params - learning_rate * beta * velocity
        
        # Compute gradient at lookahead position
        grads = grads_fn(params_ahead)
        
        # Update velocity and parameters
        velocity = beta * velocity + (1 - beta) * grads
        params = params - learning_rate * velocity
        
        return params, velocity
    
    @staticmethod
    def adagrad(params, grads, G, learning_rate=0.01, epsilon=1e-8):
        \"\"\"
        AdaGrad
        
        G: sum of squared gradients (initialized to zeros)
        \"\"\"
        G = G + grads ** 2
        params = params - learning_rate * grads / (np.sqrt(G) + epsilon)
        return params, G
    
    @staticmethod
    def rmsprop(params, grads, E_g2, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        \"\"\"
        RMSprop
        
        E_g2: exponentially weighted avg of squared gradients
        \"\"\"
        E_g2 = beta * E_g2 + (1 - beta) * (grads ** 2)
        params = params - learning_rate * grads / (np.sqrt(E_g2) + epsilon)
        return params, E_g2
    
    @staticmethod
    def adam(params, grads, m, v, t, learning_rate=0.001, 
             beta1=0.9, beta2=0.999, epsilon=1e-8):
        \"\"\"
        Adam
        
        m: first moment (mean)
        v: second moment (variance)
        t: timestep
        \"\"\"
        # Update biased moments
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return params, m, v
    
    @staticmethod
    def adamw(params, grads, m, v, t, learning_rate=0.001,
              beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        \"\"\"
        AdamW (Adam with decoupled weight decay)
        \"\"\"
        # Update biased moments
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters with decoupled weight decay
        params = params - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * params)
        
        return params, m, v

# Example usage
class SimpleNN:
    def __init__(self, optimizer='adam'):
        self.W = np.random.randn(10, 5) * 0.01
        self.b = np.zeros((1, 5))
        self.optimizer = optimizer
        
        # Initialize optimizer states
        if optimizer == 'momentum':
            self.v_W = np.zeros_like(self.W)
            self.v_b = np.zeros_like(self.b)
        elif optimizer in ['adam', 'adamw']:
            self.m_W = np.zeros_like(self.W)
            self.v_W = np.zeros_like(self.W)
            self.m_b = np.zeros_like(self.b)
            self.v_b = np.zeros_like(self.b)
            self.t = 0
        elif optimizer == 'rmsprop':
            self.E_W = np.zeros_like(self.W)
            self.E_b = np.zeros_like(self.b)
    
    def update(self, grad_W, grad_b, learning_rate=0.001):
        \"\"\"Update parameters using specified optimizer\"\"\"
        opt = Optimizers()
        
        if self.optimizer == 'sgd':
            self.W = opt.sgd(self.W, grad_W, learning_rate)
            self.b = opt.sgd(self.b, grad_b, learning_rate)
        
        elif self.optimizer == 'momentum':
            self.W, self.v_W = opt.momentum(self.W, grad_W, self.v_W, learning_rate)
            self.b, self.v_b = opt.momentum(self.b, grad_b, self.v_b, learning_rate)
        
        elif self.optimizer == 'adam':
            self.t += 1
            self.W, self.m_W, self.v_W = opt.adam(
                self.W, grad_W, self.m_W, self.v_W, self.t, learning_rate)
            self.b, self.m_b, self.v_b = opt.adam(
                self.b, grad_b, self.m_b, self.v_b, self.t, learning_rate)
        
        elif self.optimizer == 'adamw':
            self.t += 1
            self.W, self.m_W, self.v_W = opt.adamw(
                self.W, grad_W, self.m_W, self.v_W, self.t, learning_rate)
            self.b, self.m_b, self.v_b = opt.adamw(
                self.b, grad_b, self.m_b, self.v_b, self.t, learning_rate)
        
        elif self.optimizer == 'rmsprop':
            self.W, self.E_W = opt.rmsprop(self.W, grad_W, self.E_W, learning_rate)
            self.b, self.E_b = opt.rmsprop(self.b, grad_b, self.E_b, learning_rate)
        """, language="python")

elif section == "📋 Quick Reference":
    st.title("📋 Machine Learning Quick Reference")
    
    ref_type = st.selectbox(
        "Choose reference type:",
        ["Algorithm Comparison", "Loss Functions", "Metrics", "Best Practices"]
    )
    
    if ref_type == "Algorithm Comparison":
        st.header("Algorithm Comparison Matrix")
        
        comparison = {
            "Algorithm": [
                "Linear Regression",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "Neural Network"
            ],
            "Type": [
                "Regression",
                "Classification",
                "Both",
                "Both",
                "Both",
                "Both"
            ],
            "Interpretability": [
                "High",
                "High",
                "High",
                "Low",
                "Medium",
                "Low"
            ],
            "Training Speed": [
                "Fast",
                "Fast",
                "Fast",
                "Medium",
                "Slow",
                "Slow"
            ],
            "Prediction Speed": [
                "Very Fast",
                "Very Fast",
                "Fast",
                "Fast",
                "Medium",
                "Fast"
            ],
            "Handles Non-linearity": [
                "No",
                "No",
                "Yes",
                "Yes",
                "Yes (with kernels)",
                "Yes"
            ],
            "Feature Scaling Needed": [
                "Recommended",
                "Recommended",
                "No",
                "No",
                "Yes",
                "Yes"
            ]
        }
        
        st.table(pd.DataFrame(comparison))

else:
    st.title(f"{section}")
    st.info("This section is under construction. The complete workbook will cover all ML algorithms comprehensively!")
    st.markdown("""
    ### Coming Soon:
    - Complete mathematical derivations
    - Working code implementations
    - Real-world examples
    - Practice problems with solutions
    """)

# Footer
st.markdown("---")
st.caption("🤖 ML Foundations Workbook | Complete Mathematical & Practical Guide | Built with Streamlit")
