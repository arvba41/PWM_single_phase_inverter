"""
PWM Inverter Simulator — Streamlit version of main.py
Covers: Half-bridge, Full-bridge (bipolar/unipolar), Harmonic analysis
Dependencies: streamlit, numpy, scipy, matplotlib
Run with: streamlit run main_streamlit.py
"""

import numpy as np
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────
#  Core simulation functions (translated from MATLAB)
# ─────────────────────────────────────────────


def fft_function_1d(x, t):
    """Single-sided amplitude spectrum."""
    L = len(x)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    Y = np.fft.fft(x)
    P2 = Y / L
    X = P2[: L // 2 + 1]
    X[1:-1] = 2 * X[1:-1]
    f = fs * np.arange(L // 2 + 1) / L
    return X, f


def euler_forward(fy, y0, t, h):
    """Forward Euler ODE solver."""
    y = np.zeros(len(t))
    y[0] = y0
    for ii in range(1, len(t)):
        y[ii] = y[ii - 1] + h * fy(t[ii - 1], y[ii - 1])
    return y


def carrier_wave(fsw, tsim, carrier_type):
    if carrier_type == "Triangular":
        return sawtooth(2 * np.pi * fsw * tsim, width=0.5)
    else:  # Sawtooth
        return sawtooth(2 * np.pi * fsw * tsim, width=0)


def hb_inverter(M, f1, fsw, h, carrier_type):
    tsim = np.arange(0, 1 / f1, h)
    ref = M * np.sin(2 * np.pi * f1 * tsim)
    car = carrier_wave(fsw, tsim, carrier_type)
    g1_p = (ref > car).astype(float)
    vout = (g1_p - 0.5) * 2
    return vout, tsim, ref, car


def fb_inverter(M, f1, fsw, h, carrier_type, mod_type):
    tsim = np.arange(0, 1 / f1, h)
    ref = M * np.sin(2 * np.pi * f1 * tsim)
    car = carrier_wave(fsw, tsim, carrier_type)
    g1_p = (ref > car).astype(float)
    g1_n = (ref < car).astype(float)
    if mod_type == "bipolar":
        g3_p = g1_n
    else:  # unipolar
        g3_p = (-ref > car).astype(float)
    vout = (g1_p - 0.5) * 2 - (g3_p - 0.5) * 2
    return vout, tsim, ref, car


# ─────────────────────────────────────────────
#  Theme colours
# ─────────────────────────────────────────────

BG = "#0f1117"
PANEL = "#1a1d27"
ACCENT = "#4f8ef7"
ACCENT2 = "#e05c5c"
ACCENT3 = "#4fcf8e"
TEXT = "#e8eaf0"
SUBTEXT = "#8b93a8"
BORDER = "#2a2e3e"

PLOT_COLORS = ["#4f8ef7", "#e05c5c", "#4fcf8e", "#f0c040", "#c97bff", "#ff8c42"]


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#13161f")
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    ax.title.set_color(TEXT)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=BORDER, linewidth=0.6, linestyle="--")


def make_fig():
    fig = plt.figure(clear=True, layout="constrained")
    fig.patch.set_facecolor(BG)
    return fig


# ─────────────────────────────────────────────
#  Simulation runner
# ─────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def run_simulation(f1, fsw, R_load, L_load, M, carrier, mod_type):
    h = 1e-5  # coarser step for speed

    # ── HB ──
    vout_hb, tsim, ref, car = hb_inverter(M, f1, fsw, h, carrier)
    Vout_hb, Fsim = fft_function_1d(vout_hb, tsim)
    i_eqn_hb = lambda t, y: (np.interp(t, tsim, vout_hb) - y * R_load) / L_load
    iout_hb = euler_forward(i_eqn_hb, 0, tsim, h)
    Iout_hb, _ = fft_function_1d(iout_hb, tsim)

    # ── FB ──
    vout_fb, _, ref_fb, car_fb = fb_inverter(M, f1, fsw, h, carrier, mod_type)
    Vout_fb, Fsim_fb = fft_function_1d(vout_fb, tsim)
    i_eqn_fb = lambda t, y: (np.interp(t, tsim, vout_fb) - y * R_load) / L_load
    iout_fb = euler_forward(i_eqn_fb, 0, tsim, h)
    Iout_fb, _ = fft_function_1d(iout_fb, tsim)

    # ── Harmonic sweep (lighter step) ──
    M_vec = np.arange(0, 1.41, 0.02)
    Vout1 = np.zeros((len(M_vec), 3))
    Vout3 = np.zeros((len(M_vec), 3))
    Voutfsw = np.zeros((len(M_vec), 3))
    Vout2fsw = np.zeros((len(M_vec), 3))
    Vout4fsw = np.zeros((len(M_vec), 3))

    for ii, Mi in enumerate(M_vec):
        for col, (inv_type, mtype) in enumerate(
            [("hb", None), ("fb", "bipolar"), ("fb", "unipolar")]
        ):
            if inv_type == "hb":
                v, t, *_ = hb_inverter(Mi, f1, fsw, h, "Triangular")
            else:
                v, t, *_ = fb_inverter(Mi, f1, fsw, h, "Triangular", mtype)
            V, F = fft_function_1d(v, t)
            Fr = np.round(F).astype(int)

            def amp(target):
                idx = np.where(Fr == target)[0]
                return np.abs(V[idx[0]]) if len(idx) else 0.0

            Vout1[ii, col] = amp(int(round(f1)))
            Vout3[ii, col] = amp(int(round(3 * f1)))
            Voutfsw[ii, col] = amp(int(round(fsw)))
            Vout2fsw[ii, col] = amp(int(round(2 * fsw - f1)))
            Vout4fsw[ii, col] = amp(int(round(4 * fsw - f1)))

    return {
        "tsim": tsim,
        "ref": ref,
        "car": car,
        "vout_hb": vout_hb,
        "Vout_hb": Vout_hb,
        "Fsim": Fsim,
        "iout_hb": iout_hb,
        "Iout_hb": Iout_hb,
        "ref_fb": ref_fb,
        "car_fb": car_fb,
        "vout_fb": vout_fb,
        "Vout_fb": Vout_fb,
        "Fsim_fb": Fsim_fb,
        "iout_fb": iout_fb,
        "Iout_fb": Iout_fb,
        "M_vec": M_vec,
        "Vout1": Vout1,
        "Vout3": Vout3,
        "Voutfsw": Voutfsw,
        "Vout2fsw": Vout2fsw,
        "Vout4fsw": Vout4fsw,
    }


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────


def draw_hb_fig(res, f1, fsw):
    xlim_fft = (0, fsw * 4 + 3 * f1)
    lw = 1.2
    fig = make_fig()
    axes = fig.subplots(3, 1)
    fig.subplots_adjust(hspace=0.42, top=0.94, bottom=0.08, left=0.09, right=0.97)

    tsim, ref, car = res["tsim"], res["ref"], res["car"]
    vout_hb, iout_hb = res["vout_hb"], res["iout_hb"]
    Vout_hb, Fsim = res["Vout_hb"], res["Fsim"]

    ax = axes[0]
    ax.plot(tsim * 1000, ref, color=PLOT_COLORS[0], lw=lw, label="ref")
    ax.plot(tsim * 1000, car, color=PLOT_COLORS[1], lw=lw, label="carrier", alpha=0.8)
    style_ax(ax, "Modulation Signal", xlabel="time [ms]")
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)

    ax = axes[1]
    ax.plot(tsim * 1000, vout_hb, color=PLOT_COLORS[0], lw=lw)
    ax.plot(tsim * 1000, iout_hb, color=PLOT_COLORS[2], lw=lw, linestyle="--")
    style_ax(ax, "Output Voltage & Current", xlabel="time [ms]", ylabel="[p.u]")
    ax.legend(
        ["v_out", "i_out"],
        fontsize=8,
        facecolor=PANEL,
        labelcolor=TEXT,
        edgecolor=BORDER,
    )

    ax = axes[2]
    mask = Fsim <= xlim_fft[1]
    ax.bar(Fsim[mask], np.abs(Vout_hb[mask]), width=fsw / 200, color=ACCENT, alpha=0.85)
    style_ax(ax, "Voltage Spectrum", xlabel="Frequency [Hz]", ylabel="|V_out| [p.u]")
    ax.set_xlim(xlim_fft)

    return fig


def draw_fb_fig(res, f1, fsw, mod_type):
    xlim_fft = (0, fsw * 4 + 3 * f1)
    lw = 1.2
    fig = make_fig()
    axes = fig.subplots(3, 1)
    fig.subplots_adjust(hspace=0.42, top=0.94, bottom=0.08, left=0.09, right=0.97)

    ref_fb, car_fb = res["ref_fb"], res["car_fb"]
    vout_fb, iout_fb = res["vout_fb"], res["iout_fb"]
    Vout_fb, Fsim_fb = res["Vout_fb"], res["Fsim_fb"]
    tsim = res["tsim"]

    ax = axes[0]
    ax.plot(tsim * 1000, ref_fb, color=PLOT_COLORS[0], lw=lw, label="ref")
    if mod_type == "unipolar":
        ax.plot(
            tsim * 1000, -ref_fb, color=PLOT_COLORS[3], lw=lw, label="-ref", alpha=0.8
        )
    ax.plot(
        tsim * 1000, car_fb, color=PLOT_COLORS[1], lw=lw, label="carrier", alpha=0.75
    )
    style_ax(ax, f"Modulation ({mod_type.capitalize()})", xlabel="time [ms]")
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)

    ax = axes[1]
    ax.plot(tsim * 1000, vout_fb, color=PLOT_COLORS[0], lw=lw)
    ax.plot(tsim * 1000, iout_fb, color=PLOT_COLORS[2], lw=lw, linestyle="--")
    style_ax(ax, "Output Voltage & Current", xlabel="time [ms]", ylabel="[p.u]")
    ax.legend(
        ["v_out", "i_out"],
        fontsize=8,
        facecolor=PANEL,
        labelcolor=TEXT,
        edgecolor=BORDER,
    )

    ax = axes[2]
    mask = Fsim_fb <= xlim_fft[1]
    ax.bar(
        Fsim_fb[mask], np.abs(Vout_fb[mask]), width=fsw / 200, color=ACCENT2, alpha=0.85
    )
    style_ax(ax, "Voltage Spectrum", xlabel="Frequency [Hz]", ylabel="|V_out| [p.u]")
    ax.set_xlim(xlim_fft)

    return fig


def draw_harmonic_fig(res, f1, fsw):
    lw = 1.2
    fig = make_fig()
    axes = fig.subplots(3, 1)
    fig.subplots_adjust(hspace=0.52, top=0.94, bottom=0.07, left=0.09, right=0.78)

    M_vec = res["M_vec"]
    Vout1, Vout3 = res["Vout1"], res["Vout3"]
    Voutfsw, Vout2fsw, Vout4fsw = res["Voutfsw"], res["Vout2fsw"], res["Vout4fsw"]

    legends = [
        f"f₁={f1:.0f}Hz",
        f"f₃={3*f1:.0f}Hz",
        f"f_sw={fsw:.0f}Hz",
        f"2f_sw-f₁",
        f"4f_sw-f₁",
    ]
    colors_h = PLOT_COLORS[:5]

    for col, (ax, title, norm) in enumerate(
        zip(
            axes,
            ["Half-Bridge Inverter", "Full-Bridge (Bipolar)", "Full-Bridge (Unipolar)"],
            [1, 2, 2],
        )
    ):
        for harm, leg, c in zip(
            [
                Vout1[:, col],
                Vout3[:, col],
                Voutfsw[:, col],
                Vout2fsw[:, col],
                Vout4fsw[:, col],
            ],
            legends,
            colors_h,
        ):
            ax.plot(M_vec, harm / norm, color=c, lw=lw, label=leg)
        style_ax(
            ax, title, ylabel="|V_out(h)| [p.u]", xlabel="M [-]" if col == 2 else ""
        )
        ax.legend(
            fontsize=7.5,
            facecolor=PANEL,
            labelcolor=TEXT,
            edgecolor=BORDER,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )

    return fig


# ─────────────────────────────────────────────
#  Streamlit app
# ─────────────────────────────────────────────


def main():
    st.set_page_config(page_title="PWM Inverter Simulator", layout="wide")

    st.title("⚡ PWM Inverter Simulator")
    st.caption("lecture 5 · power electronics")

    with st.sidebar:
        st.header("Parameters")
        f1 = st.number_input("Fundamental freq [Hz]", value=50.0, min_value=1.0)
        fsw = st.number_input("Switching freq [Hz]", value=1000.0, min_value=1.0)
        R_load = st.number_input("Load resistance [Ω]", value=1.0, min_value=0.0)
        L_load_mh = st.number_input("Load inductance [mH]", value=1.0, min_value=0.0)
        M = st.slider(
            "Modulation index M", min_value=0.0, max_value=1.2, value=0.6, step=0.01
        )
        carrier = st.selectbox("Carrier type", ["Triangular", "Sawtooth"])
        mod_type = st.selectbox("FB modulation type", ["bipolar", "unipolar"])
        run_clicked = st.button("▶ Run Simulation", use_container_width=True)
        if st.button("⟳ Reset Defaults", use_container_width=True):
            st.rerun()

    if run_clicked or "results" in st.session_state:
        if run_clicked:
            with st.spinner("Simulating…"):
                st.session_state["results"] = run_simulation(
                    f1, fsw, R_load, L_load_mh * 1e-3, M, carrier, mod_type
                )
                st.session_state["params"] = (f1, fsw, mod_type)

        res = st.session_state["results"]
        f1, fsw, mod_type = st.session_state["params"]

        tab_hb, tab_fb, tab_harm = st.tabs(
            [" Half-Bridge Inverter ", " Full-Bridge Inverter ", " Harmonic Analysis "]
        )

        with tab_hb:
            st.pyplot(draw_hb_fig(res, f1, fsw))

        with tab_fb:
            st.pyplot(draw_fb_fig(res, f1, fsw, mod_type))

        with tab_harm:
            st.pyplot(draw_harmonic_fig(res, f1, fsw))

        st.success(f"Done  |  M={M:.2f}  f₁={f1:.0f}Hz  fsw={fsw:.0f}Hz")
    else:
        st.info("Set parameters in the sidebar and click **Run Simulation**.")


if __name__ == "__main__":
    main()
