## Visualization

`psm-viz` renders side-by-side video playback and a spatial memory heatmap with GPS trace overlay, synchronized by timestamp.

```bash
# Config file (defaults < config < CLI)
targets/psm-viz -c psm-viz.toml.example
targets/psm-viz -c /path/to/psm-viz.toml -g jepa
targets/psm-viz -c configs/psm-viz-balanced.toml -d /path/to/session
targets/psm-viz -c configs/psm-viz-low-hitch.toml -d /path/to/session

# Directory mode — finds *.mp4 and features.h5 automatically
targets/psm-viz -d /path/to/session/
targets/psm-viz -d /path/to/session/ -g jepa

# Explicit flags
targets/psm-viz -v video.mp4 -f features.h5 -g dino -m total

# Legacy positional args
targets/psm-viz video.mp4 features.h5 dino 5.0 10
```

| Flag | Arg | Default | Description |
|------|-----|---------|-------------|
| `-c` | `<path>` | — | TOML-style config file |
| `-d` | `<dir>` | — | Directory containing `*.mp4` and `features.h5` |
| `-v` | `<path>` | — | Video file path |
| `-f` | `<path>` | — | HDF5 features file path |
| `-g` | `<name>` | `dino` | HDF5 group name (`dino` or `jepa`) |
| `-t` | `<sec>` | `5.0` | Time window (seconds) |
| `-r` | `<res>` | `10` | H3 resolution (0-15) |
| `-m` | `<mode>` | `total` | Heatmap mode (`total`, `current`, `recency`) |
| `-h` | — | — | Print help |

`psm-viz.toml` supports:

```toml
session_dir = "./session"
# video_path = "./session/video.mp4"
# features_path = "./session/features.h5"

group = "dino"
time_window_sec = 5.0
h3_resolution = 10
start_paused = true
debug_hud_enabled = true
scrub_sensitivity_sec = 2.0
map_follow_smoothing = 8.0
video_decode_budget = 6
ingest_record_budget = 128
imu_sample_budget = 512
gps_point_budget = 64
tile_uploads_per_frame = 1
tile_disk_cache_enabled = true
tile_disk_cache_max_mb = 512
heatmap_mode = "total"
hex_extrude_scale = 0.0
tile_style = "CartoDB.Positron"

# Required for Stadia.* presets and any custom template using {api_key}
# tile_api_key = "..."

# Optional override; supports {z}, {x}, {y}, {s}, and {api_key}
# tile_url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
```

Relative paths in the config resolve relative to the config file itself. CLI flags override config values.

Ready-made presets:
- `configs/psm-viz-balanced.toml`: a good default balance between responsiveness and tile fill speed.
- `configs/psm-viz-low-hitch.toml`: prioritizes smoother interaction with fewer tile uploads per frame and smaller per-frame catch-up budgets.

Tuning keys:
- `start_paused`: when `true`, `psm-viz` opens on the first decoded frame and waits for `Space` before playback starts.
- `debug_hud_enabled`: enables the live window-title HUD by default. You can still toggle it at runtime with `H`.
- `scrub_sensitivity_sec`: seconds moved per horizontal scroll step on the video pane.
- `map_follow_smoothing`: exponential follow rate for GPS/IMU-driven recentering. Higher values snap faster.
- `video_decode_budget`: baseline video decode steps per frame at 1x playback. Faster playback scales up from this value, and sustained decode backlog can temporarily raise it further before it decays back down.
- `ingest_record_budget`: baseline max feature/embedding records applied per frame. Under sustained ingest backlog the runtime can temporarily raise this catch-up budget, then ease it back to the configured base.
- `imu_sample_budget`: baseline max IMU samples drained per frame, with the same temporary backlog-driven catch-up behavior.
- `gps_point_budget`: baseline max standalone GPS points drained per frame, with the same temporary backlog-driven catch-up behavior.
- `tile_uploads_per_frame`: baseline max ready tile textures uploaded per frame. Lower values reduce GL-side hitches; higher values fill tiles faster, and the runtime can temporarily raise this when decoded tiles are piling up.
- `tile_disk_cache_enabled`: enables or disables the on-disk raster tile cache.
- `tile_disk_cache_max_mb`: maximum on-disk tile cache size per configured tile source before older cached tiles are pruned.
- `heatmap_mode`: selects how H3 cells are scored before coloring. `total` shows the rolling merged count across the active ring buffer, `current` shows current-bucket activity only, and `recency` shows `current / total` to highlight cells that are active now relative to their longer-term history.
- `hex_extrude_scale`: cabinet-projection 3D extrusion. `0.0` (default) renders flat hexes; values in `0.10`–`0.50` raise the tallest cell by that fraction of the visible viewport so dominant memory cells read as terrain. Toggle at runtime with `E`.

Available `tile_style` presets:
- `CartoDB.Positron`
- `CartoDB.PositronNoLabels`
- `CartoDB.Voyager`
- `CartoDB.DarkMatter`
- `Stadia.AlidadeSmooth` (`tile_api_key` required)
- `Stadia.AlidadeSmoothDark` (`tile_api_key` required)
- `Stamen.Watercolor` (`tile_api_key` required) — painterly hand-drawn aesthetic, hosted by Stadia
- `Stamen.Toner` (`tile_api_key` required) — high-contrast black & white
- `Stamen.TonerLite` (`tile_api_key` required) — softer Toner variant
- `Stamen.Terrain` (`tile_api_key` required) — illustrated terrain shading

`Stadia.*` and `Stamen.*` styles all use the same Stadia Maps API key (free tier covers 200k tiles/month). Sign up at <https://stadiamaps.com/>, then drop the key into your config:

```toml
tile_style = "Stamen.Watercolor"
tile_api_key = "your-stadia-api-key"
```

Preview the providers here: <https://leaflet-extras.github.io/leaflet-providers/preview/>

Downloaded raster tiles are cached on disk and replay through the same threaded decode path as network tiles. The cache location is:
- macOS: `~/Library/Caches/psm-viz/tiles/...`
- other Unix-like systems: `$XDG_CACHE_HOME/psm-viz/tiles/...` or `~/.cache/psm-viz/tiles/...`

**Controls:**

| Key / Gesture | Action |
|---------------|--------|
| Space | Start / pause / resume playback (shows pause icon) |
| +/- | Zoom in / out around the map center |
| Left / Right | Slow down / speed up playback |
| Scroll H (video) | Scrub video timeline |
| Scroll V (map) | Zoom map toward the cursor |
| Drag (map) | Pan map manually |
| C | Re-center map and resume smooth follow |
| M | Cycle heatmap mode (`total` → `current` → `recency`) |
| E | Toggle 3D hex extrusion (cabinet projection) |
| L | Toggle the heatmap legend overlay |
| H | Toggle live debug title HUD |
| P | Save a screenshot of the current composed frame to `captures/` as `.png` |
| ? / F1 | Toggle the help overlay |
| Q / Esc | Quit |

The visualizer opens paused by default on the first decoded frame, shows the help/startup overlay immediately, and does not begin playback until you press `Space`. The first `Space` also dismisses that initial help panel.

Screenshots save into `<session_dir>/captures/` when a session directory is configured, otherwise `./captures/`, using names like `psm-viz-000042.png`.

The debug HUD lives in the window title and shows playback/decode budgets, ingest drain activity, tile pipeline queue counts, and tile disk-cache health in real time. For the `v`, `in`, `imu`, `gps`, and `up` fields, the HUD shows `work/current_budget`; when adaptive backpressure boosts a budget above its configured base, the base appears in parentheses, for example `256/384(128)`. A trailing `*` means that lane still had backlog after spending its frame budget. Tile fields are: `act` active network downloads, `rdy` compressed tiles ready for decode, `dec` tiles currently being decoded, `pix` decoded tiles waiting for GL upload, and `c` resident cached tile textures. Disk-cache fields are: `h` disk cache hits, `w` cache writes, `p` pruned files, and `m` cached MiB used versus cap.

**Layout:** Left half shows video with optional attention/prediction heatmap overlay. Right half shows configurable raster tiles (default: `CartoDB.Positron`) with H3 hex heatmap (RGB-cube tour: black → red → yellow → green → cyan → blue → magenta → white), GPS trace ribbon, and camera frustum. The map view follows the latest GPS/IMU-driven position smoothly by default; manual drag temporarily overrides that view until you re-center with `C`.

**Hex heatmap semantics:** The color ramp is always relative to the hottest tile currently rendered. The colormap is an RGB-cube tour — low-intensity tiles appear near-black, climbing through red, yellow, green, cyan, blue, magenta, and ending at white for the hottest tiles. Alpha also rises with intensity, so stronger cells look more solid. The ramp is striking but not perceptually uniform; cube-corner transitions (e.g. yellow→green) look like bigger jumps than mid-edge transitions of equal numeric distance.

Mode-specific behavior:
- `total`: colors by merged distinct-count across the full active ring-buffer horizon. This is the historical memory view.
- `current`: colors by the current time bucket only. This is the most immediate activity view.
- `recency`: colors by `current / total`, highlighting cells that are active now relative to their own accumulated history.

Time decay is only partly encoded in hue. In `total` mode, when a tile has historical count but little or no activity in the current bucket, the renderer reduces alpha so the cell lingers as a dim memory instead of vanishing immediately. The underlying forgetting is stepwise: as the ring buffer advances, the oldest bucket is overwritten, so hex intensity can drop in discrete steps at window boundaries rather than as a perfectly smooth fade.
