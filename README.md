# GNNs for Influence Maximization: 6th-Semester Project

An ongoing research project focused on bypassing computationally heavy Monte Carlo simulations for Influence Maximization by using Graph Neural Networks as structural feature aggregators.

### Phase 1: Midsem Work
* **The Graph:** Built a synthetic 200-node BarabÃ¡si-Albert (BA) scale-free network.
* **The Ground Truth:** Simulated Independent Cascade (IC) and Linear Threshold (LT) diffusions (200 Monte Carlo simulations per node).
* **The Model:** Implemented a single static GAT layer + Ridge Regression head.
* **The Result:** Proved that a graph-aware deep learning approach outperformed classical baselines. The GAT model achieved an $R^2$ of 0.708, representing a **~29% improvement** in predictive fit over the best classical baseline (Polynomial Regression, $R^2$ 0.548) and a **~37% improvement** over standard Linear Regression ($R^2$ 0.516).

### Phase 2: Endsem Work
* **The Graph:** Scaled the environment up to a 500-node BA network with planted high-degree hubs.
* **The Models:** Benchmarked four distinct message-passing architectures: GAT, GCN, GIN, and GraphSAGE.
* **The Result:** Reduced Mean Squared Error (MSE) by ~40% over linear baselines. GIN and GraphSAGE showed the most promise in handling extreme degree variance.

### The Post-Endsem Analysis
After reviewing the endsem architecture, I identified four critical flaws in using standard GNNs for diffusion:
1. **Static Features:** Standard GATs ignored edge transmission probabilities ($p_{ij}$). They assigned attention purely based on node similarity, completely missing whether an edge was probabilistically "strong" or "weak".
2. **The Floor Effect:** Massive hub nodes washed out attention weights for peripheral nodes due to standard Softmax math, causing the model to over-predict low-tier influencers.
3. **Volatility on Outliers:** Predicting extreme top-tier viral cascades caused severe gradient instability because standard MSE aggressively penalizes massive outliers.
4. **Feature Distortion:** Using the same learning speed for the entire model caused the early layers to scramble my starting data (centralities: Betweenness, Closeness, Eigenvector) before the network could even figure out how to use it

---

### Phase 3: Post-Endsem (The PGAT-IM Architecture)
To solve these bottlenecks, I abandoned standard library GNNs and engineered **PGAT-IM (Probability-Gated Attention Network)** a custom layer mathematically aligned with the IC/LT diffusion rules.

Here is what I added in PGAT-IM that does not exist in standard architectures:
* **Probability Gate ($g_{ij}$):** Intercepted the pre-softmax attention logit with a learned `tanh` gate. This actively throttles neural attention if the physical edge transmission probability ($p_{ij}$) is weak.
* **Temperature-Scaled Softmax:** Divided the attention logits by $\sqrt{degree_i}$ before softmax. This dynamic temperature scaling prevents massive hub nodes from diluting the attention distribution.
* **Double-Probability Message Passing:** Scaled the actual passed feature vector by both the neural attention ($ lpha_{ij}$) *and* the normalized statistical edge probability.
* **Multi-Task Huber Loss:** Unified IC and LT into a dual-head model. Used Huber loss ($\delta=0.1$) to act as L2 for normal predictions and L1 for extreme outliers, preventing gradient explosions on massive cascades.
* **Layer-wise Learning Rate Decay (LLRD):** Applied tiny learning rates to the input layers to preserve mathematical centrality truths, and larger learning rates only for the final predictive heads.
