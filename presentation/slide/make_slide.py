"""
make_slide.py — one-page EN 649 "Project Interact" slide on the course template: motivation, then one result that
was reproduced and two that were not, each with a small figure.

    python make_slide.py          # -> EN649_project_slide.pptx (+ fig_*.png)

template_bg.png is the course template (Google Slides PDF) rendered without its placeholder text. The paper image
for the third figure is read from the poster's extracted paper pages (../../../poster/panels/paper_p4_0.jpeg).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "poster"))
import make_poster as mp                                  # data helpers: hold(), _ramp_files(), momentum_residual()

PAPER_P4 = os.path.join(HERE, "..", "..", "..", "poster", "panels", "paper_p4_0.jpeg")
INK, BROWN, MUTED = "233a44", "af7b51", "5b6b72"
GOOD, GOODBG, BAD, BADBG = "1b7a4b", "eaf4ee", "b3261e", "fbeceb"
plt.rcParams.update({"font.family": "Calibri", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": "#" + MUTED,
                     "axes.labelcolor": "#" + INK, "text.color": "#" + INK, "xtick.color": "#" + MUTED,
                     "ytick.color": "#" + MUTED, "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.3, "axes.linewidth": 0.6})

# card geometry (pt, slide is 720 x 405)
CARD_X, CARD_W, CARD_Y, CARD_H = (22, 251, 480), 218, 178, 177
FIG_W, FIG_H = 206, 106                                   # figure slot inside a card


def figsave(fig, name, bg):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=400, facecolor="#" + bg)
    plt.close(fig)
    return p


def fig_success():
    edges = np.arange(0, 2.25, 0.05)
    c = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(FIG_W / 72, FIG_H / 72))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.25, top=0.95)
    for br, ls, lab in ((1, "-", "increasing U$_g$"), (-1, "--", "decreasing U$_g$")):
        rows = []
        for f in mp._ramp_files("twoway"):
            r = np.load(f)
            m = r["branch"] == br
            idx = np.digitize(r["Ug"][m], edges) - 1
            rows.append([r["dP"][m][idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(c))])
        ax.plot(c, np.nanmean(np.array(rows), 0), color="#" + INK, ls=ls, label=lab)
    ax.axhline(905, color="#" + MUTED, lw=0.7, ls=":")
    ax.text(0.03, 925, "bed weight", fontsize=6.5, color="#" + MUTED)
    ax.text(1.08, 560, "u$_{mf}$ (m/s)", fontsize=6.5, color="#" + INK)
    for u, col, lab, y in ((0.99, "#" + GOOD, "ours 0.99", 420), (1.00, "#888888", "paper 1.00", 290),
                           (1.05, "#" + BROWN, "expt 1.05", 160)):
        ax.axvline(u, color=col, lw=0.8)
        ax.text(1.08, y, lab, fontsize=6.5, color=col)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1050)
    ax.set_xlabel("U$_g$ (m/s)", labelpad=1)
    ax.set_ylabel("ΔP$_{bed}$ (Pa)", labelpad=1)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.5, handlelength=1.8)
    return figsave(fig, "fig_success_umf.png", GOODBG)


def fig_plug():
    fig, ax = plt.subplots(figsize=(FIG_W / 72, FIG_H / 72))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.25, top=0.95)
    ax.axhspan(0.85, 0.94, color="#" + GOOD, alpha=0.35, lw=0)
    ax.text(14.8, 1.4, "paper: 0.85–0.94", ha="right", fontsize=6.5, color="#" + GOOD)
    for u, lab, col in (("2.19", "2 u$_{mf}$", "#9aa5aa"), ("3.28", "3 u$_{mf}$", "#" + BROWN),
                        ("4.38", "4 u$_{mf}$", "#" + BAD)):
        r = mp.hold(u, "2way")
        rat = r["dP"] / r["dP_theory"]
        ax.plot(r["t"], rat, color=col)
        ax.text(5, rat[r["t"] > 5].mean() + 0.6, lab, fontsize=6.5, color=col, fontweight="bold")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 13.5)
    ax.set_xlabel("time (s)", labelpad=1)
    ax.set_ylabel("ΔP / bed weight", labelpad=1)
    return figsave(fig, "fig_fail_plug.png", BADBG)


def fig_momentum():
    fig = plt.figure(figsize=(FIG_W / 72, FIG_H / 72))
    w = 0.21
    axs = [fig.add_axes([0.02 + 0.25 * i, 0.03, w, 0.80]) for i in range(3)]
    img = plt.imread(PAPER_P4)
    h, wd = img.shape[:2]
    axs[0].imshow(img[int(0.035 * h):int(0.225 * h), int(0.655 * wd):int(0.955 * wd)], aspect="auto")
    axs[0].set_title("paper ε$_g$:\nbubbles", pad=2)
    r = mp.hold("2.19", "2way")
    k = int(round(14.5 * float(r["sample_hz"]))) - 1
    W, H = float(r["W"]), float(r["H"])
    axs[1].imshow(r["eps"][k].T, origin="lower", extent=(0, W, 0, H), cmap="RdBu", vmin=0.4, vmax=1.0,
                  aspect="auto")
    axs[1].set_title("ours ε$_g$:\none void", pad=2)
    R, _, _, _ = mp.momentum_residual("2.19", "2way")
    im = axs[2].imshow(R.T, origin="lower", extent=(0, W, 0, H), cmap="Reds", vmin=0, vmax=1.5, aspect="auto")
    axs[2].set_title("ours: unbalanced\ndrag |R|", pad=2)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
    cb = fig.colorbar(im, cax=fig.add_axes([0.78, 0.08, 0.03, 0.70]), ticks=[0, 0.5, 1.0, 1.5])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    fig.text(0.96, 0.43, "× bed weight per volume", fontsize=6, va="center", ha="center", rotation=90,
             color="#" + MUTED)
    return figsave(fig, "fig_fail_momentum.png", BADBG)


# ------------------------------------------------------------------ pptx
def text(slide, x, y, w, h, runs, size, color=INK, bold=False, font="Calibri", align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """runs: str, or list of paragraphs, each a str or a list of (text, {bold, color}) runs."""
    tb = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        if spacing:
            para.space_after = Pt(spacing)
        for t, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run()
            r.text = t
            f = r.font
            f.name, f.size = font, Pt(size)
            f.bold = o.get("bold", bold)
            f.color.rgb = RGBColor.from_string(o.get("color", color))
    return tb


def card(slide, x, ok, tag, head, fig, caption):
    bg, col = (GOODBG, GOOD) if ok else (BADBG, BAD)
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Pt(x), Pt(CARD_Y), Pt(CARD_W), Pt(CARD_H))
    sh.adjustments[0] = 0.05
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor.from_string(bg)
    sh.line.fill.background()
    sh.shadow.inherit = False
    text(slide, x + 7, CARD_Y + 5, CARD_W - 14, 14,
         [[(("✔ " if ok else "✘ ") + tag + "  ", {"bold": True, "color": col}), (head, {"bold": True})]], 10.5)
    slide.shapes.add_picture(fig, Pt(x + 6), Pt(CARD_Y + 21), Pt(FIG_W), Pt(FIG_H))
    text(slide, x + 7, CARD_Y + 21 + FIG_H + 3, CARD_W - 14, CARD_H - FIG_H - 26, caption, 9.5)


def main():
    figs = fig_success(), fig_plug(), fig_momentum()
    R, E, _, _ = mp.momentum_residual("2.19", "2way")
    pct = 100 * R[E < 0.8].mean()                                       # mean unbalanced drag in bed cells
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(9144000), Emu(5143500)       # 10 x 5.625 in, as the template
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(os.path.join(HERE, "template_bg.png"), 0, 0, prs.slide_width, prs.slide_height)

    text(s, 25, 20, 670, 34, "Reproducing a CFD-DEM Fluidized-Bed Study in Python", 24, BROWN, font="Nunito")
    text(s, 34, 64, 500, 18, [[("Kaustubh Ashok", {}), ("  ·  24B1522", {"color": MUTED})]], 13)

    text(s, 25, 90, 400, 20, "Motivation and Objective", 16, BROWN)
    text(s, 25, 111, 670, 44,
         "Gas-fluidized beds run reactors, dryers and combustors, and CFD-DEM is the standard way to simulate "
         "them. Objective: reproduce Liu & van Wachem (2019) for the NETL Geldart-D bed with a soft-sphere DEM "
         "code written from scratch in Python (Numba, 3,380 particles), coupled to the gas one-way and "
         "quasi-two-way (1-D), and find where such a simple gas model stops working.", 11.5)

    text(s, 25, 159, 670, 17, [[("●   ", {"color": INK}), ("Key findings", {"bold": True}),
                                ("  —  one result reproduced, two not", {"color": MUTED})]], 13)

    card(s, CARD_X[0], True, "Reproduced", "Minimum fluidization", figs[0],
         "ΔP rises with gas velocity and levels off at the bed weight. u_mf = 0.99 ± 0.01 m/s over 14 beds, vs "
         "1.00 m/s (paper) and 1.05 m/s (experiment).")
    card(s, CARD_X[1], False, "Failed", "Plug lift at 3–4 u_mf", figs[1],
         "ΔP climbs to 6× and 12× the bed weight (paper: 0.85–0.94×). Gas solved only per row cannot bypass "
         "the bed, so it lifts the whole bed against the lid.")
    card(s, CARD_X[2], False, "Failed", "Bubbles at 2 u_mf", figs[2],
         f"One central void instead of discrete bubbles. No gas momentum equation: in bed cells "
         f"≈ {pct:.0f} % of the local weight is drag that no pressure gradient balances.")

    text(s, 25, 366, 200, 18, "EN 649", 13)
    text(s, 410, 360, 288, 26,
         ["Main reference: D. Liu, B. van Wachem, Powder Technol. 343 (2019) 145–158",
          "Code: github.com/kaustubh2005ashok-tech/en649-fluidized-bed"], 9, MUTED, align=PP_ALIGN.RIGHT)

    s.notes_slide.notes_text_frame.text = (
        "Success: u_mf from 14 beds per coupling matches the paper within 1 %. Failure 1: at 3 and 4 u_mf the 1-D "
        "gas model lifts the bed as a plug. Failure 2: at 2 u_mf the bed fluidizes but without discrete bubbles, "
        "because gas momentum is not conserved cell by cell: the drag in a cell is balanced only by one pressure "
        f"gradient per row, which leaves on average {pct:.0f} % of the local bed weight unbalanced in bed cells.")
    out = os.path.join(HERE, "EN649_project_slide.pptx")
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
