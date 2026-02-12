# Kakeya AGI Simulator — Streaming LLM Stick Bundles with Novelty-Gated Adaptive Learning
---

## Overview

The **Kakeya AGI Simulator** is a **first-principles framework** demonstrating how **emergent general intelligence (AGI)** can arise from **deterministic streaming neural computation**.  

- Multiple pre-trained LLMs are represented as **weight stick bundles**.  
- Activations are processed in **Q16.16 fixed-point arithmetic**, mimicking **FPGA/ASIC execution**.  
- Updates are triggered via **local novelty-gated adaptive learning**.  
- **Kakeya-inspired geometric transformations** ensure maximal coverage of high-dimensional representation space.

**Key Features:**

- Deterministic fixed-point execution for hardware reproducibility.  
- Multi-LLM streaming with SVD-derived principal components.  
- Novelty-gated adaptive learning enabling surprise-driven updates.  
- Kakeya geometric coverage ensuring theoretical completeness in representation.  
- Real-time visualization of activations, sparsity, and novelty events.

---

## System Architecture

### 1. LLM Stick Bundles
- Each LLM is decomposed into **principal component sticks** using **SVD (Eckart–Young decomposition)**.  
- **High-Dimensional Geometry:** Sticks encode weight directions, forming the skeleton of the model’s knowledge.  
- **Deterministic Streaming:** Components are streamed sequentially, preserving exact behavior under quantized execution.

### 2. Fixed-Point Deterministic Execution
- Activations are **quantized to Q16.16 fixed-point**, simulating FPGA/ASIC integer arithmetic.  
- **Bit-Exact Computation:** Ensures reproducibility without floating-point drift.  
- **Hardware Realism:** Demonstrates AGI-like logic operating under constrained, integer-only environments.

### 3. Local Novelty-Gated Adaptive Learning
- Activations are modulated by **novelty gates**, triggering updates when angular change exceeds a threshold.  
- **Adaptive Plasticity:** Learning rates scale dynamically with input divergence, simulating a surprise-driven system.  
- Supports emergent coordination across multiple LLM sticks **without centralized control**.

### 4. Kakeya Geometric Coverage
- Stick activations are **rotated and combined across multiple LLMs** to approximate a **Kakeya set**.  
- **Representational Completeness:** Allows mapping any input signal to relevant high-dimensional directions.  
- **Generalization:** Integrates local novelty with global coverage to handle out-of-distribution inputs.

---

## Real-Time Interactive Visualization

The simulator tracks four key metrics of AGI state:

1. **Entropy Dynamics:** Measures information density across streamed sticks.  
2. **Energy Evolution:** Monitors stability of SVD components.  
3. **Novelty Detections:** Identifies surprise-driven activation events.  
4. **Adaptive Gain (α):** Visualizes dynamic sensitivity to novel inputs.

These visualizations enable **instant feedback** on emergent patterns and system stability.
