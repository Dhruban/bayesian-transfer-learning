import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bayesian Transfer Learning",
    layout="wide"
)

# Set random seed for reproducibility
np.random.seed(42)

# Title and introduction
st.title("Bayesian Transfer Learning: Interactive Demonstration")

st.markdown(r"""
This app demonstrates **positive** and **negative transfer** in a Bayesian framework using a simple Bernoulli model.

### Model Setup
- **Source data**: $Z_s^{(i)} \sim \text{Bernoulli}(\theta_s^*)$
- **Target data**: $Z_t^{(j)} \sim \text{Bernoulli}(\theta_t^*)$
- **Prior**: $\omega(\theta_t | \theta_s) = \text{Uniform}[\theta_s - c, \theta_s + c]$

**Key Concept**: The prior is *proper* when $|\theta_s^* - \theta_t^*| \leq c$, leading to **positive transfer**. 
Otherwise, it's *improper*, causing **negative transfer**.
""")

# Core functions
def generate_data(theta, n):
    """Generate Bernoulli samples."""
    return np.random.binomial(1, theta, n)

def kl_divergence_bernoulli(p, q):
    """KL divergence between two Bernoulli distributions."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def compute_transfer_posterior(theta_grid, D_s, D_t, theta_s_hat, c):
    """
    Compute transfer learning posterior P(theta_t | D_s, D_t).
    
    Uses the conditional prior omega(theta_t | theta_s) = Uniform[theta_s - c, theta_s + c]
    """
    # Compute likelihood from target data
    n_success_t = np.sum(D_t)
    n_t = len(D_t)
    likelihood = theta_grid**n_success_t * (1 - theta_grid)**(n_t - n_success_t)
    
    # Apply conditional prior based on source estimate
    prior = np.zeros_like(theta_grid)
    lower = max(0, theta_s_hat - c)
    upper = min(1, theta_s_hat + c)
    mask = (theta_grid >= lower) & (theta_grid <= upper)
    if np.sum(mask) > 0:
        prior[mask] = 1.0 / (upper - lower)
    
    # Compute posterior
    posterior = likelihood * prior
    if np.sum(posterior) > 0:
        posterior = posterior / (np.sum(posterior) * (theta_grid[1] - theta_grid[0]))
    
    return posterior, lower, upper

def compute_online_regret(theta_true, D_t, theta_predictor):
    """
    Compute cumulative regret for online learning.
    
    Regret at step k = log loss with predictor - log loss with true theta
    """
    n = len(D_t)
    regret = np.zeros(n)
    
    for k in range(n):
        z_k = D_t[k]
        # Predictor loss
        p_pred = np.clip(theta_predictor, 1e-10, 1 - 1e-10)
        loss_pred = -z_k * np.log(p_pred) - (1 - z_k) * np.log(1 - p_pred)
        
        # Optimal loss
        p_true = np.clip(theta_true, 1e-10, 1 - 1e-10)
        loss_opt = -z_k * np.log(p_true) - (1 - z_k) * np.log(1 - p_true)
        
        regret[k] = loss_pred - loss_opt
    
    return np.cumsum(regret)

# Sidebar for parameters
st.sidebar.header("Parameters")

st.sidebar.markdown("### True Parameters")
theta_s_star = st.sidebar.slider(
    "θ_s* (source)", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.7, 
    step=0.05,
    help="True success probability in source domain"
)

theta_t_star = st.sidebar.slider(
    "θ_t* (target)", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.3, 
    step=0.05,
    help="True success probability in target domain"
)

st.sidebar.markdown("### Prior Knowledge")
c = st.sidebar.slider(
    "c (knowledge level)", 
    min_value=0.05, 
    max_value=0.5, 
    value=0.1, 
    step=0.05,
    help="How close we believe target is to source (prior support width)"
)

st.sidebar.markdown("### Sample Sizes")
m = st.sidebar.slider(
    "m (source samples)", 
    min_value=10, 
    max_value=500, 
    value=100, 
    step=10,
    help="Number of observations from source domain"
)

n = st.sidebar.slider(
    "n (target samples)", 
    min_value=10, 
    max_value=200, 
    value=100, 
    step=10,
    help="Number of observations from target domain"
)

# Generate data button
if st.sidebar.button("Generate New Data"):
    st.session_state.seed = np.random.randint(0, 10000)

# Use session state for reproducibility within a run
if 'seed' not in st.session_state:
    st.session_state.seed = 42

np.random.seed(st.session_state.seed)

# Generate data
D_s = generate_data(theta_s_star, m)
D_t = generate_data(theta_t_star, n)

# Estimate source parameter
theta_s_hat = np.mean(D_s) if len(D_s) > 0 else 0.5

# Grid for posterior
theta_grid = np.linspace(0, 1, 1000)

# Compute transfer learning posterior
posterior_transfer, lower_bound, upper_bound = compute_transfer_posterior(
    theta_grid, D_s, D_t, theta_s_hat, c
)

# Compute target-only posterior (Beta distribution)
alpha_post = 1 + np.sum(D_t)
beta_post = 1 + len(D_t) - np.sum(D_t)
posterior_target_only = stats.beta.pdf(theta_grid, alpha_post, beta_post)

# Find MAP estimates
if np.sum(posterior_transfer) > 0:
    map_transfer = theta_grid[np.argmax(posterior_transfer)]
else:
    map_transfer = theta_s_hat

map_target_only = theta_grid[np.argmax(posterior_target_only)]

# Compute regret
regret_transfer = compute_online_regret(theta_t_star, D_t, map_transfer)
regret_target_only = compute_online_regret(theta_t_star, D_t, map_target_only)

# Check if prior is proper
is_proper = abs(theta_s_star - theta_t_star) <= c
transfer_type = "POSITIVE TRANSFER" if is_proper else "NEGATIVE TRANSFER"
color_type = "green" if is_proper else "red"

# Display transfer regime
st.markdown(f"## Transfer Regime: :{color_type}[{transfer_type}]")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Domain Gap", f"{abs(theta_s_star - theta_t_star):.3f}")
with col2:
    st.metric("Knowledge Level", f"{c:.2f}")
with col3:
    st.metric("Prior Status", "Proper" if is_proper else "Improper")
with col4:
    gap_vs_c = "≤" if is_proper else ">"
    st.metric("Gap vs c", f"Gap {gap_vs_c} c")

# Create visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Posterior Distribution
ax1 = axes[0]
ax1.plot(theta_grid, posterior_transfer, 'b-', linewidth=2.5, label='With Transfer', alpha=0.8)
ax1.plot(theta_grid, posterior_target_only, 'gray', linewidth=2.5, label='Target Only', alpha=0.6, linestyle='--')
ax1.axvline(theta_t_star, color='red', linestyle='--', linewidth=2, label=f'True θ_t* = {theta_t_star:.2f}')
ax1.axvline(theta_s_hat, color='orange', linestyle=':', linewidth=2, label=f'Est. θ_s = {theta_s_hat:.2f}')
ax1.axvspan(lower_bound, upper_bound, alpha=0.2, color='cyan', label=f'Prior Support')
ax1.set_xlabel('θ_t', fontsize=13, fontweight='bold')
ax1.set_ylabel('Posterior Density', fontsize=13, fontweight='bold')
ax1.set_title('Posterior Distribution P(θ_t | D_s, D_t)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Plot 2: Cumulative Regret
ax2 = axes[1]
steps = np.arange(1, n + 1)
ax2.plot(steps, regret_transfer, 'b-', linewidth=2.5, label='With Transfer', alpha=0.8)
ax2.plot(steps, regret_target_only, 'gray', linewidth=2.5, label='Target Only', alpha=0.6, linestyle='--')

# Add theoretical curves
if is_proper:
    theoretical = 0.5 * np.log(steps) + 1.5
    ax2.plot(steps, theoretical, 'g:', linewidth=2, alpha=0.6, label='~0.5 log(n) (theory)')
else:
    # Estimate KL divergence
    theta_boundary = lower_bound if theta_t_star < lower_bound else upper_bound
    kl_div = kl_divergence_bernoulli(theta_t_star, theta_boundary)
    theoretical = kl_div * steps
    ax2.plot(steps, theoretical, 'r:', linewidth=2, alpha=0.6, label=f'~{kl_div:.2f}·n (theory)')

ax2.set_xlabel('Target Samples (n)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
ax2.set_title('Online Learning Regret', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Data Visualization
ax3 = axes[2]

# Show source and target data distributions
source_mean = np.mean(D_s)
target_mean = np.mean(D_t)

bars = ax3.bar(['Source\n(observed)', 'Target\n(observed)'], 
               [source_mean, target_mean], 
               color=['orange', 'blue'], 
               alpha=0.6, 
               edgecolor='black',
               linewidth=2)

ax3.axhline(theta_s_star, color='orange', linestyle='--', linewidth=2, label=f'True θ_s* = {theta_s_star:.2f}')
ax3.axhline(theta_t_star, color='red', linestyle='--', linewidth=2, label=f'True θ_t* = {theta_t_star:.2f}')

ax3.set_ylabel('Success Probability', fontsize=13, fontweight='bold')
ax3.set_title('Observed vs. True Parameters', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1)

# Add sample size annotations
ax3.text(0, source_mean + 0.05, f'n={m}', ha='center', fontsize=10, fontweight='bold')
ax3.text(1, target_mean + 0.05, f'n={n}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

# Summary statistics
st.markdown("## Results Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Transfer Learning Performance")
    st.write(f"**MAP Estimate (transfer):** {map_transfer:.3f}")
    st.write(f"**MAP Estimate (target only):** {map_target_only:.3f}")
    st.write(f"**True θ_t*:** {theta_t_star:.2f}")
    st.write(f"**Error (transfer):** {abs(map_transfer - theta_t_star):.3f}")
    st.write(f"**Error (target only):** {abs(map_target_only - theta_t_star):.3f}")

with col2:
    st.markdown("### Cumulative Regret")
    st.write(f"**Final regret (transfer):** {regret_transfer[-1]:.2f}")
    st.write(f"**Final regret (target only):** {regret_target_only[-1]:.2f}")
    st.write(f"**Difference:** {regret_transfer[-1] - regret_target_only[-1]:.2f}")
    
    if regret_transfer[-1] < regret_target_only[-1]:
        st.success("Transfer learning HELPS (lower regret)")
    else:
        st.error("Transfer learning HURTS (higher regret)")

# Interpretation
st.markdown("## Interpretation")

if is_proper:
    st.success(f"""
    **Positive Transfer Scenario**
    
    The domain gap ({abs(theta_s_star - theta_t_star):.3f}) is within the prior tolerance ({c:.2f}), making the prior *proper*.
    
    - The true θ_t* = {theta_t_star:.2f} is within the prior support [{lower_bound:.2f}, {upper_bound:.2f}]
    - The posterior correctly concentrates around the true target parameter
    - Source data reduces uncertainty and improves early predictions
    - Regret grows logarithmically (~0.5 log n), which is optimal for online learning
    - Transfer learning achieves lower cumulative regret than target-only learning
    """)
else:
    st.error(f"""
    **Negative Transfer Scenario**
    
    The domain gap ({abs(theta_s_star - theta_t_star):.3f}) exceeds the prior tolerance ({c:.2f}), making the prior *improper*.
    
    - The true θ_t* = {theta_t_star:.2f} is OUTSIDE the prior support [{lower_bound:.2f}, {upper_bound:.2f}]
    - The posterior concentrates at the wrong location (boundary of prior support)
    - Source data is misleading and actively harms predictions
    - Regret grows linearly (~n), not logarithmically
    - Transfer learning achieves HIGHER cumulative regret than target-only learning
    
    **The source data makes things worse!**
    """)

# Suggested experiments
with st.expander("Suggested Experiments"):
    st.markdown("""
    ### Try These Scenarios:
    
    1. **Positive Transfer (Small Gap)**
       - Set θ_s* = 0.4, θ_t* = 0.35, c = 0.1
       - Observe: Posterior concentrates correctly, regret is lower than target-only
    
    2. **Negative Transfer (Large Gap)**
       - Set θ_s* = 0.8, θ_t* = 0.3, c = 0.1
       - Observe: Posterior stuck at boundary, regret grows linearly
    
    3. **Boundary Case**
       - Set θ_s* = 0.5, θ_t* = 0.4, c = 0.1 (gap = 0.1, exactly at threshold)
       - Observe: On the edge between proper and improper
    
    4. **Effect of Knowledge Level**
       - Fix θ_s* = 0.7, θ_t* = 0.3
       - Try c = 0.1 (negative) vs c = 0.4 (positive)
       - Observe: Larger c is more robust but less informative
    
    5. **Sample Size Effects**
       - Set up negative transfer (e.g., θ_s* = 0.8, θ_t* = 0.3, c = 0.1)
       - Increase m from 50 to 500
       - Observe: More source data amplifies the negative effect
    """)

# Footer
st.markdown("---")
st.markdown("""
**Key Takeaways:**
- Prior properness determines transfer success
- Observable failure: negative transfer shows linear (not logarithmic) regret growth
- Trade-off: larger c is safer but less beneficial when domains are similar
""")
