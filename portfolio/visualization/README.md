# Neural Observatory

Open [index.html](index.html) in a browser. It works directly from a local file, with no server, network connection, package installation or model training. Keep `index.html`, `style.css`, `app.js` and `data.js` together.

![Animated neuron cloud and running mice with transformer, MLP and ridge predictions](neural-observatory.gif)

[Static preview at the fixed midpoint of TX103, seed 401](preview.png).

The shared 3D view shows the exact 512 cells supplied to the independently fitted transformer, population MLP and ridge regression. An observed-speed mouse provides the reference; three model animations sit side by side. All seven separate-cohort recordings, all three fixed seeds and every scored test frame are available.

- Drag the neuron cloud to orbit, scroll to zoom, or focus it and use arrow keys. Hover a cell to inspect its original row index, position and approximate activity above baseline.
- Play, pause, step a frame, or scrub/click the full-interval timeline. Playback stops at the final frame; pressing Play then restarts.
- Switch recording or seed. All panels stay synchronized. Ridge is deterministic and stays unchanged when the neural training seed changes.
- The full-test R² and error relative to zero speed remain visible. They describe the selected individual seed, not the seed-averaged primary comparison in the research report.

## What is recorded and what is illustrated?

**Recorded:** published neuron x/y coordinates and imaging-plane index, the selected neurons' prepared activity, and running speed at the aligned native frame. **Predicted:** existing held-out model outputs. **Illustrated:** mouse gait and the visual spacing between imaging planes. This is not video, a recovered body pose, an atlas reconstruction, a connectivity map or a neural generator.

The input-channel mapping is the essential implementation detail: `panel[i]` identifies the original neuron row, selects its published position, and corresponds to channel `i` of the prepared activity. The first scored test target is 63 frames into the prepared test segment; the exporter applies that same offset to the activity. Each animation and the timeline cursor use that one frame index.

Colors show positive activity relative to each cell's training mean/SD. Display intensities are clipped at +5 SD and quantized to eight bits; values at or below the mean remain dim. No decoder or metric is computed from those quantized colors. x/y are jointly scaled to preserve their relative aspect ratio; depth is an ordinal plane index with schematic spacing, not verified micrometers. These model recipes do not use coordinates as inputs.

Speed is nonnegative published running divided by the recording's training-target SD. Physical speed units and a native-frame-to-seconds conversion were not verified, so the interface uses SD and native frames. Playback “frames/s” is a display rate. One shared gait mapping converts the displayed speed to animation; cumulative distance gives deterministic poses when seeking.

## Rebuild and checks

The README GIF is a 12-second looping capture of 144 consecutive test frames, 2495–2638, from TX103 and seed 401. The first recording, first seed and fixed midpoint determine the excerpt; model error did not determine it. The cloud rotates while recorded activity, observed running and all three saved predictions advance together. All 512 cells remain in view. Compact capture styling removes playback controls; the interactive viewer itself is unchanged. Full-test metrics remain visible and are not calculated from the excerpt.

GIF playback averages 12 frames per display second, not verified acquisition seconds. This is an illustrative replay, not evidence that one model won. A shared 256-color palette and changed-region encoding keep the animated asset compact. [GIF checks and provenance](gif_review.json) record its dimensions, duration, source hashes and captured frame alignment. No training or inference was run to make it.

The maintained viewer has no third-party JavaScript dependencies. The optional data exporter requires NumPy and the existing local study archive:

```bash
python3 portfolio/visualization/build_data.py
```

It reads only the small position arrays from the large source NPZs, prepared test activity, and saved predictions. It copies no raw full-population recording and never imports training code. The generated `data.js` is approximately 25 MB because it includes all 34,411 test frames and 512 activity colors per frame across the seven mice. The file is local; there is no API request or CDN.

[Data checks](data_review.json) record exact target/channel alignment, coordinate provenance, display quantization error and agreement of compact display speeds with archived errors. [Browser checks](browser_review.json) cover 63 first/middle/last-frame cases across all recording/seed pairs, 504 speed/display checks, 567 archived metric checks, playback/seek/boundaries, orbit/zoom/keyboard/reset, reduced-motion preferences, desktop/mobile layouts and absence of runtime errors or external page requests. The desktop preview uses the fixed midpoint of the first recording; it was not chosen for favorable model performance.

Original experimental results are unchanged. No fits or inference were repeated. The browser testing dependency and temporary diagnostic files live outside the repository and are not needed to open the viewer. [Research context](../RESEARCH_REPORT.md) and [all aggregate results](../results/RESULTS.md) remain separate from this illustrative replay.
