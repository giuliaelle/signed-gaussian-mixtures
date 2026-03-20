import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import ndimage


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Negative Domains Explorer",
    layout="wide"
)

plt.rcParams.update({
    "figure.figsize": (16, 6),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 11
})


# ============================================================
# DEFAULTS
# ============================================================

CONFIG_VERSION = 3

DEFAULT_GLOBALS = {
    "mode": "Basic",
    "preset_name": "Three negative wells",
    "t_value": 0.0,
    "threshold": 0.0,
    "xlim": 6.0,
    "ylim": 6.0,
    "resolution": 250,
    "connectivity": 2,
    "show_zero_contour": True,
    "show_component_centres": True,
    "symmetric_color": True,
    "n_components": 4,
}

PRESETS = {
    "Three negative wells": [
        {"weight": 1.35, "mx": 0.0, "my": 0.0, "sx": 1.48, "sy": 1.34, "rho": 0.00},
        {"weight": -1.20, "mx": -1.8, "my": 0.6, "sx": 0.42, "sy": 0.47, "rho": 0.00},
        {"weight": -1.10, "mx": 1.6, "my": -0.4, "sx": 0.45, "sy": 0.42, "rho": 0.00},
        {"weight": -0.90, "mx": 0.2, "my": 1.5, "sx": 0.40, "sy": 0.40, "rho": 0.00},
    ],
    "Two symmetric lobes": [
        {"weight": 1.20, "mx": 0.0, "my": 0.0, "sx": 1.80, "sy": 1.50, "rho": 0.00},
        {"weight": -1.10, "mx": -1.6, "my": 0.0, "sx": 0.35, "sy": 0.35, "rho": 0.00},
        {"weight": -1.10, "mx": 1.6, "my": 0.0, "sx": 0.35, "sy": 0.35, "rho": 0.00},
    ],
    "Single persistent negative region": [
        {"weight": 0.95, "mx": 0.0, "my": 0.0, "sx": 2.20, "sy": 2.20, "rho": 0.00},
        {"weight": -1.60, "mx": 0.0, "my": 0.0, "sx": 0.55, "sy": 0.55, "rho": 0.00},
    ],
    "Near-merging domains": [
        {"weight": 1.15, "mx": 0.0, "my": 0.0, "sx": 1.90, "sy": 1.60, "rho": 0.00},
        {"weight": -1.10, "mx": -0.9, "my": 0.0, "sx": 0.42, "sy": 0.42, "rho": 0.00},
        {"weight": -1.10, "mx": 0.9, "my": 0.0, "sx": 0.42, "sy": 0.42, "rho": 0.00},
    ],
    "Mixed anisotropic case": [
        {"weight": 1.25, "mx": 0.0, "my": 0.0, "sx": 1.80, "sy": 1.10, "rho": 0.20},
        {"weight": -1.05, "mx": -1.8, "my": -0.6, "sx": 0.55, "sy": 0.28, "rho": 0.35},
        {"weight": -0.95, "mx": 1.2, "my": 1.1, "sx": 0.30, "sy": 0.55, "rho": -0.40},
        {"weight": 0.35, "mx": 2.4, "my": -1.6, "sx": 0.75, "sy": 0.50, "rho": 0.00},
    ],
}

DEFAULT_COMPONENTS = [dict(c) for c in PRESETS["Three negative wells"]]


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    if "components" not in st.session_state:
        st.session_state.components = [dict(c) for c in DEFAULT_COMPONENTS]

    for key, value in DEFAULT_GLOBALS.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# CORE MATH
# ============================================================

def covariance_from_params(sx: float, sy: float, rho: float) -> np.ndarray:
    return np.array([
        [sx**2, rho * sx * sy],
        [rho * sx * sy, sy**2]
    ], dtype=float)


def gaussian_2d(X: np.ndarray, Y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    pos = np.stack([X, Y], axis=-1)
    diff = pos - mean

    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    quad = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    return norm_const * np.exp(-0.5 * quad)


def signed_gaussian_mixture(X: np.ndarray, Y: np.ndarray, components: list[dict], t: float = 0.0) -> np.ndarray:
    Z = np.zeros_like(X, dtype=float)
    I = np.eye(2)

    for comp in components:
        w = float(comp["weight"])
        mu = np.array([comp["mx"], comp["my"]], dtype=float)
        Sigma = covariance_from_params(comp["sx"], comp["sy"], comp["rho"])
        Sigma_t = Sigma + t * I
        Z += w * gaussian_2d(X, Y, mu, Sigma_t)

    return Z


def negative_domains(Z: np.ndarray, threshold: float = 0.0, connectivity: int = 2):
    neg_mask = Z < threshold
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 1 else 2)
    labels, num = ndimage.label(neg_mask, structure=structure)
    return neg_mask, labels, num


def component_areas(labels: np.ndarray, num: int) -> list[int]:
    return [int(np.sum(labels == k)) for k in range(1, num + 1)]


# ============================================================
# CONFIG SERIALIZATION
# ============================================================

def build_config_dict() -> dict:
    return {
        "config_version": CONFIG_VERSION,
        "globals": {
            "mode": st.session_state.mode,
            "preset_name": st.session_state.preset_name,
            "t_value": float(st.session_state.t_value),
            "threshold": float(st.session_state.threshold),
            "xlim": float(st.session_state.xlim),
            "ylim": float(st.session_state.ylim),
            "resolution": int(st.session_state.resolution),
            "connectivity": int(st.session_state.connectivity),
            "show_zero_contour": bool(st.session_state.show_zero_contour),
            "show_component_centres": bool(st.session_state.show_component_centres),
            "symmetric_color": bool(st.session_state.symmetric_color),
            "n_components": len(st.session_state.components),
        },
        "components": st.session_state.components,
    }


def config_to_json_bytes() -> bytes:
    return json.dumps(build_config_dict(), indent=2).encode("utf-8")


def validate_component(comp: dict) -> dict:
    required = ["weight", "mx", "my", "sx", "sy", "rho"]
    out = {}
    for key in required:
        if key not in comp:
            raise ValueError(f"Missing component field: {key}")
        out[key] = float(comp[key])

    if out["sx"] <= 0 or out["sy"] <= 0:
        raise ValueError("sigma_x and sigma_y must be strictly positive.")
    if not (-0.999 < out["rho"] < 0.999):
        raise ValueError("rho must lie strictly between -1 and 1.")
    return out


def load_config_from_json_bytes(data: bytes):
    obj = json.loads(data.decode("utf-8"))
    if "components" not in obj:
        raise ValueError("Invalid config: missing 'components'.")

    globals_dict = obj.get("globals", {})
    comps = [validate_component(c) for c in obj["components"]]

    st.session_state.components = comps
    for key, default_value in DEFAULT_GLOBALS.items():
        st.session_state[key] = globals_dict.get(key, default_value)

    st.session_state.n_components = len(comps)


# ============================================================
# HELPERS
# ============================================================

def ensure_component_count(target_count: int):
    current = st.session_state.components
    current_count = len(current)

    if target_count > current_count:
        for k in range(current_count, target_count):
            current.append({
                "weight": 0.5,
                "mx": 0.0,
                "my": 0.0,
                "sx": 0.7,
                "sy": 0.7,
                "rho": 0.0
            })
    elif target_count < current_count:
        st.session_state.components = current[:target_count]


def apply_preset(name: str):
    preset_components = [dict(c) for c in PRESETS[name]]
    st.session_state.components = preset_components
    st.session_state.n_components = len(preset_components)
    st.session_state.preset_name = name


def plot_current_state(
    X: np.ndarray,
    Y: np.ndarray,
    extent: list[float],
    components: list[dict],
    t_value: float,
    threshold: float,
    connectivity: int,
    show_zero_contour: bool,
    show_component_centres: bool,
    symmetric_color: bool
):
    Z = signed_gaussian_mixture(X, Y, components, t=t_value)
    neg_mask, labels, num_domains = negative_domains(Z, threshold=threshold, connectivity=connectivity)
    areas = component_areas(labels, num_domains)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    ax1, ax2, ax3 = axes

    if symmetric_color:
        absmax = max(abs(Z.min()), abs(Z.max()), 1e-12)
        vmin, vmax = -absmax, absmax
    else:
        vmin, vmax = Z.min(), Z.max()

    im1 = ax1.imshow(
        Z, origin="lower", extent=extent, aspect="equal",
        cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    if show_zero_contour:
        ax1.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=2)
    if show_component_centres:
        for comp in components:
            ax1.plot(comp["mx"], comp["my"], "ko", markersize=4)
    ax1.set_title(r"Scalar field $u(x,y,t)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2.imshow(
        neg_mask.astype(int), origin="lower", extent=extent,
        aspect="equal", cmap="Blues"
    )
    if show_zero_contour:
        ax2.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=2)
    if show_component_centres:
        for comp in components:
            ax2.plot(comp["mx"], comp["my"], "ko", markersize=4)
    ax2.set_title(f"Negative mask | count = {num_domains}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3.imshow(
        labels, origin="lower", extent=extent,
        aspect="equal", cmap="tab20"
    )
    if show_zero_contour:
        ax3.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=1.5)
    if show_component_centres:
        for comp in components:
            ax3.plot(comp["mx"], comp["my"], "ko", markersize=4)
    ax3.set_title("Connected negative domains")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    return fig, Z, num_domains, areas


def build_animation_gif(
    X: np.ndarray,
    Y: np.ndarray,
    extent: list[float],
    components: list[dict],
    threshold: float,
    connectivity: int,
    show_zero_contour: bool,
    show_component_centres: bool,
    symmetric_color: bool,
    gif_tmax: float,
    gif_frames: int,
    fps: int
) -> bytes:
    fig_anim, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    ax1, ax2, ax3 = axes
    t_values = np.linspace(0.0, gif_tmax, gif_frames)

    def draw_frame(t: float):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        Zt = signed_gaussian_mixture(X, Y, components, t=t)
        neg_mask_t, labels_t, num_t = negative_domains(Zt, threshold=threshold, connectivity=connectivity)

        if symmetric_color:
            absmax_t = max(abs(Zt.min()), abs(Zt.max()), 1e-12)
            vmin_t, vmax_t = -absmax_t, absmax_t
        else:
            vmin_t, vmax_t = Zt.min(), Zt.max()

        ax1.imshow(
            Zt, origin="lower", extent=extent, aspect="equal",
            cmap="coolwarm", vmin=vmin_t, vmax=vmax_t
        )
        if show_zero_contour:
            ax1.contour(X, Y, Zt, levels=[0.0], colors="black", linewidths=2)
        if show_component_centres:
            for comp in components:
                ax1.plot(comp["mx"], comp["my"], "ko", markersize=4)
        ax1.set_title(rf"Scalar field $u(x,y,t)$, $t={t:.2f}$")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2.imshow(
            neg_mask_t.astype(int), origin="lower", extent=extent,
            aspect="equal", cmap="Blues"
        )
        if show_zero_contour:
            ax2.contour(X, Y, Zt, levels=[0.0], colors="black", linewidths=2)
        if show_component_centres:
            for comp in components:
                ax2.plot(comp["mx"], comp["my"], "ko", markersize=4)
        ax2.set_title(f"Negative mask | count = {num_t}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        ax3.imshow(
            labels_t, origin="lower", extent=extent,
            aspect="equal", cmap="tab20"
        )
        if show_zero_contour:
            ax3.contour(X, Y, Zt, levels=[0.0], colors="black", linewidths=1.5)
        if show_component_centres:
            for comp in components:
                ax3.plot(comp["mx"], comp["my"], "ko", markersize=4)
        ax3.set_title("Connected negative domains")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")

    draw_frame(0.0)

    anim = FuncAnimation(
        fig_anim,
        lambda i: draw_frame(t_values[i]),
        frames=len(t_values),
        interval=1000 / fps,
        blit=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = Path(tmpdir) / "negative_domains_evolution.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=fps))
        with open(gif_path, "rb") as f:
            gif_bytes = f.read()

    plt.close(fig_anim)
    return gif_bytes


# ============================================================
# HEADER
# ============================================================

st.title("Negative Domains Explorer")

st.write("This interactive page shows how the negative regions of a signed Gaussian mixture evolve under Gaussian smoothing.")
st.latex(r"f(x,y)=\sum_{k=1}^{K} w_k\,\mathcal{N}\!\left((x,y)\mid \mu_k,\Sigma_k\right)")
st.write("At scale " + r"$t$" + ", each covariance becomes " + r"$\Sigma_k+tI$" + ".")
st.latex(r"u(x,y,t)=\sum_{k=1}^{K} w_k\,\mathcal{N}\!\left((x,y)\mid \mu_k,\Sigma_k+tI\right)")
st.write("The negative domains are the connected components of")
st.latex(r"\{(x,y):u(x,y,t)<\tau\}")

with st.expander("How to use this page", expanded=True):
    st.markdown(
        """
        1. Choose a preset scenario.  
        2. Move the smoothing slider `t`.  
        3. Observe how the negative regions shrink, merge, or disappear.  
        4. Switch to **Advanced mode** only if you want to edit the Gaussian components directly.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Mode")
st.session_state.mode = st.sidebar.radio("Interface mode", ["Basic", "Advanced"], index=0 if st.session_state.mode == "Basic" else 1)

st.sidebar.header("Preset scenarios")
selected_preset = st.sidebar.selectbox(
    "Choose a preset",
    options=list(PRESETS.keys()),
    index=list(PRESETS.keys()).index(st.session_state.preset_name) if st.session_state.preset_name in PRESETS else 0
)

if st.sidebar.button("Load preset"):
    apply_preset(selected_preset)
    st.rerun()

st.sidebar.header("Main controls")

st.session_state.t_value = st.sidebar.slider(
    "Smoothing parameter t",
    min_value=0.0,
    max_value=3.0,
    value=float(st.session_state.t_value),
    step=0.01
)

st.session_state.threshold = st.sidebar.slider(
    "Threshold τ",
    min_value=-0.5,
    max_value=0.5,
    value=float(st.session_state.threshold),
    step=0.001
)

st.session_state.xlim = st.sidebar.slider(
    "Horizontal range",
    min_value=2.0,
    max_value=100.0,
    value=float(min(st.session_state.xlim, 100.0)),
    step=1.0
)

st.session_state.ylim = st.sidebar.slider(
    "Vertical range",
    min_value=2.0,
    max_value=100.0,
    value=float(min(st.session_state.ylim, 100.0)),
    step=1.0
)

st.session_state.resolution = st.sidebar.slider(
    "Grid resolution",
    min_value=100,
    max_value=800,
    value=int(min(st.session_state.resolution, 800)),
    step=50
)

st.session_state.show_zero_contour = st.sidebar.checkbox(
    "Show zero contour",
    value=bool(st.session_state.show_zero_contour)
)

st.session_state.show_component_centres = st.sidebar.checkbox(
    "Show component centres",
    value=bool(st.session_state.show_component_centres)
)

st.session_state.symmetric_color = st.sidebar.checkbox(
    "Use symmetric color scale",
    value=bool(st.session_state.symmetric_color)
)

st.session_state.connectivity = st.sidebar.selectbox(
    "Connectivity for components",
    options=[1, 2],
    index=0 if int(st.session_state.connectivity) == 1 else 1
)

st.sidebar.header("Configuration")

st.sidebar.download_button(
    label="Download current config (JSON)",
    data=config_to_json_bytes(),
    file_name="negative_domains_config.json",
    mime="application/json"
)

uploaded_config = st.sidebar.file_uploader("Load config from JSON", type=["json"])

if uploaded_config is not None:
    try:
        load_config_from_json_bytes(uploaded_config.read())
        st.sidebar.success("Configuration loaded successfully.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Could not load config: {e}")

if st.sidebar.button("Reset everything"):
    st.session_state.components = [dict(c) for c in DEFAULT_COMPONENTS]
    for key, value in DEFAULT_GLOBALS.items():
        st.session_state[key] = value
    st.rerun()


# ============================================================
# ADVANCED COMPONENT EDITOR
# ============================================================

if st.session_state.mode == "Advanced":
    st.sidebar.header("Advanced component editor")

    st.session_state.n_components = st.sidebar.slider(
        "Number of components",
        min_value=1,
        max_value=10,
        value=int(len(st.session_state.components)),
        step=1
    )
    ensure_component_count(st.session_state.n_components)

    for k in range(len(st.session_state.components)):
        comp = st.session_state.components[k]
        with st.sidebar.expander(f"Component {k+1}", expanded=(k < 4)):
            comp["weight"] = st.slider(
                f"w_{k+1}", -3.0, 3.0, float(comp["weight"]), 0.01, key=f"weight_{k}"
            )
            comp["mx"] = st.slider(
                f"mu_x_{k+1}", -float(st.session_state.xlim), float(st.session_state.xlim),
                float(comp["mx"]), 0.01, key=f"mx_{k}"
            )
            comp["my"] = st.slider(
                f"mu_y_{k+1}", -float(st.session_state.ylim), float(st.session_state.ylim),
                float(comp["my"]), 0.01, key=f"my_{k}"
            )
            comp["sx"] = st.slider(
                f"sigma_x_{k+1}", 0.10, 5.00, float(comp["sx"]), 0.01, key=f"sx_{k}"
            )
            comp["sy"] = st.slider(
                f"sigma_y_{k+1}", 0.10, 5.00, float(comp["sy"]), 0.01, key=f"sy_{k}"
            )
            comp["rho"] = st.slider(
                f"rho_{k+1}", -0.95, 0.95, float(comp["rho"]), 0.01, key=f"rho_{k}"
            )


# ============================================================
# GRID + PLOTS
# ============================================================

x = np.linspace(-st.session_state.xlim, st.session_state.xlim, st.session_state.resolution)
y = np.linspace(-st.session_state.ylim, st.session_state.ylim, st.session_state.resolution)
X, Y = np.meshgrid(x, y)
extent = [x.min(), x.max(), y.min(), y.max()]

fig, Z, num_domains, areas = plot_current_state(
    X=X,
    Y=Y,
    extent=extent,
    components=st.session_state.components,
    t_value=st.session_state.t_value,
    threshold=st.session_state.threshold,
    connectivity=st.session_state.connectivity,
    show_zero_contour=st.session_state.show_zero_contour,
    show_component_centres=st.session_state.show_component_centres,
    symmetric_color=st.session_state.symmetric_color
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Negative domains", num_domains)
col2.metric("Minimum value", f"{Z.min():.4f}")
col3.metric("Maximum value", f"{Z.max():.4f}")
col4.metric("Current t", f"{st.session_state.t_value:.2f}")

if areas:
    st.write("Areas of the detected negative domains (in grid pixels):", areas)
else:
    st.write("No negative domains were detected at the current scale.")

st.pyplot(fig, clear_figure=True)


# ============================================================
# INTERPRETATION PANEL
# ============================================================

with st.expander("Interpretation", expanded=True):
    st.markdown(
        """
        - The **left panel** shows the scalar field.  
        - The **middle panel** shows the set where the field is below the chosen threshold.  
        - The **right panel** splits that set into connected negative domains.  

        When `t` increases, the field is smoothed more strongly.  
        In many cases, negative regions tend to become simpler: they may shrink, merge, or disappear.
        """
    )


# ============================================================
# PARAMETER TABLE
# ============================================================

if st.session_state.mode == "Advanced":
    st.subheader("Current mixture parameters")

    table_rows = []
    for idx, comp in enumerate(st.session_state.components, start=1):
        Sigma = covariance_from_params(comp["sx"], comp["sy"], comp["rho"])
        table_rows.append({
            "component": idx,
            "weight": comp["weight"],
            "mu_x": comp["mx"],
            "mu_y": comp["my"],
            "sigma_x": comp["sx"],
            "sigma_y": comp["sy"],
            "rho": comp["rho"],
            "Sigma_11": Sigma[0, 0],
            "Sigma_12": Sigma[0, 1],
            "Sigma_22": Sigma[1, 1],
        })

    st.dataframe(table_rows, use_container_width=True)


# ============================================================
# GIF EXPORT
# ============================================================

st.subheader("Export animation")

g1, g2, g3 = st.columns(3)
with g1:
    gif_frames = st.slider("Number of frames", 20, 120, 40, 5)
with g2:
    gif_tmax = st.slider(
        "Maximum t for animation",
        min_value=0.1,
        max_value=3.0,
        value=max(1.5, float(st.session_state.t_value)),
        step=0.05
    )
with g3:
    fps = st.slider("GIF fps", 5, 30, 12, 1)

if st.button("Generate GIF"):
    with st.spinner("Building animation..."):
        gif_bytes = build_animation_gif(
            X=X,
            Y=Y,
            extent=extent,
            components=st.session_state.components,
            threshold=st.session_state.threshold,
            connectivity=st.session_state.connectivity,
            show_zero_contour=st.session_state.show_zero_contour,
            show_component_centres=st.session_state.show_component_centres,
            symmetric_color=st.session_state.symmetric_color,
            gif_tmax=gif_tmax,
            gif_frames=gif_frames,
            fps=fps
        )
    st.success("GIF ready.")
    st.download_button(
        label="Download GIF",
        data=gif_bytes,
        file_name="negative_domains_evolution.gif",
        mime="image/gif"
    )