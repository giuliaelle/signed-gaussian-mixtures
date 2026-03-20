# 🌊 Negative Domains Explorer

An interactive Streamlit application to explore how **negative regions of signed Gaussian mixtures** evolve under **Gaussian smoothing**.

This app is designed as a visual and intuitive tool to study how the topology of negative domains changes as the smoothing parameter increases.

---

## 🔍 Core idea

We consider a signed Gaussian mixture of the form

`f(x, y) = sum_{k=1}^K w_k N((x,y) | mu_k, Sigma_k)`

and its smoothed version

`u(x, y, t) = sum_{k=1}^K w_k N((x,y) | mu_k, Sigma_k + t I)`

The goal is to analyse how the negative regions

`{ (x,y) : u(x,y,t) < tau }`

evolve as the smoothing parameter `t` increases.

---

## 🧠 What the app shows

For a given mixture and value of `t`, the app displays:

- the scalar field `u(x,y,t)`
- the negative mask (where the field is below the chosen threshold)
- the connected components of the negative regions

This allows you to observe phenomena such as:

- shrinking of negative regions
- merging of domains
- disappearance of components
- topological simplification under smoothing

---

## 🎛️ How to use

1. Start from a preset scenario
2. Move the smoothing parameter `t`
3. Observe how the negative regions evolve
4. Optionally switch to **Advanced mode** to modify:
   - weights
   - means
   - covariances
   - number of components

---

## 🧪 Preset scenarios

The app includes several curated examples:

- Three negative wells
- Two symmetric lobes
- Single persistent negative region
- Near-merging domains
- Mixed anisotropic case

These presets are designed to highlight different qualitative behaviours.

---

## ⚙️ Features

- Interactive control of:
  - smoothing parameter `t`
  - threshold `tau`
  - spatial domain
  - resolution

- Two interface modes:
  - **Basic** → simple exploration
  - **Advanced** → full parameter control

- Visualisation of:
  - scalar field
  - zero contour
  - connected negative domains

- Export tools:
  - GIF animation of the evolution in `t`
  - JSON save/load of configurations

---

## 🧩 Interpretation

As `t` increases, the field is progressively smoothed.

In many cases, this leads to:

- fewer negative regions
- smoother boundaries
- simpler topology

This behaviour is consistent with diffusion-like smoothing and can be used as an intuitive way to study the geometry of signed mixtures.

---

## 🚀 Run locally

Install dependencies:

```bash
pip install -r requirements.txt
