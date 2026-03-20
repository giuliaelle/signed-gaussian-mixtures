# 🌊 Negative Domains Explorer

An interactive Streamlit application to explore how **negative regions of signed Gaussian mixtures** evolve under **Gaussian smoothing**.

👉 The app is designed as a **visual and intuitive tool** to study how topology changes as the smoothing parameter increases.

---

## 🔍 Core Idea

We consider a signed Gaussian mixture of the form:

\[
f(x, y) = \sum_{k=1}^{K} w_k \, \mathcal{N}((x,y)\mid \mu_k, \Sigma_k)
\]

We then define a smoothed version:

\[
u(x,y,t) = \sum_{k=1}^{K} w_k \, \mathcal{N}((x,y)\mid \mu_k, \Sigma_k + tI)
\]

The goal is to analyse how the **negative regions**

\[
\{(x,y) : u(x,y,t) < \tau\}
\]

evolve as the smoothing parameter \( t \) increases.

---

## 🧠 What the App Shows

For a given mixture and value of \( t \), the app displays:

- **Scalar field** \( u(x,y,t) \)
- **Negative mask** (where the field is below threshold)
- **Connected components** of the negative regions

This allows you to observe phenomena such as:

- shrinking of negative regions  
- merging of domains  
- disappearance of components  
- topological simplification under smoothing  

---

## 🎛️ How to Use

1. **Start from a preset scenario**
2. Move the **smoothing parameter \( t \)** slider
3. Observe how negative regions evolve
4. (Optional) switch to **Advanced mode** to modify:
   - weights
   - means
   - covariances
   - number of components

---

## 🧪 Preset Scenarios

The app includes several curated examples:

- Three negative wells  
- Two symmetric lobes  
- Single persistent negative region  
- Near-merging domains  
- Mixed anisotropic case  

These are designed to highlight different qualitative behaviours.

---

## ⚙️ Features

- Interactive control of:
  - smoothing parameter \( t \)
  - threshold \( \tau \)
  - spatial domain
  - resolution

- Two modes:
  - **Basic** → clean exploration  
  - **Advanced** → full parameter control  

- Visualization:
  - scalar field
  - zero contour
  - connected components

- Export:
  - GIF animation of evolution in \( t \)
  - JSON save/load of configurations

---

## 🧩 Interpretation

As \( t \) increases, the field is progressively smoothed.

In many cases, this leads to:

- fewer negative regions  
- smoother boundaries  
- simpler topology  

This behaviour is related to **diffusion-like effects** and can be interpreted through the lens of **maximum principles**.

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app_2.py
