# Final submission — EN 649 course project

Kaustubh Ashok (24B1522) · Instructor: Prof. Manaswita Bose · IIT Bombay

Reproduction of Liu & van Wachem (2019), Powder Technol. 343, 145–158, with a 2-D DEM fluidized-bed code in Python.

- `poster_A1.pdf`: A1 portrait poster (print version); `poster_A1_300dpi.png`: the same poster as a 300 dpi image.
- `EN649_project_slide.pptx` / `.pdf`: the one-page slide on the course template, shown with the poster
  (one result reproduced: u_mf; two not: plug lift at 3–4 u_mf, no discrete bubbles at 2 u_mf).
- `results_video_1to4umf.mp4`: our one-way and quasi-two-way beds at 1, 2, 3 and 4 u_mf, played in real time next to
  the paper's Fig. 5 snapshot (static, t = 14.5 s) and the running ΔP / bed weight. The paper has no 1 u_mf snapshot.

Sources: the poster is built by `presentation/poster/make_poster.py`; the video by `make_video.py` from the hold runs
(`run_hold.py`). See `presentation/README.md` for the results table and limitations.
