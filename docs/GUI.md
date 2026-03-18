# GUI — Strategic Map Viewer

Real-time interactive visualization for the GRAVITAS Engine Air Strip One 1984 scenario.

## Overview

The GUI provides a live strategic map view of the 35-sector war simulation, showing fleet movements, land combat, BLF resistance activity, and real-time faction scores. Built with Pygame for smooth real-time rendering.

## Features

### Map Display
- **35 sectors** across British Isles, France, Benelux, and Netherlands
- **Real geographic positions** from Natural Earth data
- **6 sea zones** with fleet indicators
- **Sector connections** showing adjacency
- **Contested sectors** highlighted in yellow

### Real-time Information
- **Faction scores** and military forces
- **Fleet positions** in sea zones (Oceania/Eurasia counts)
- **Land garrisons** with unit count badges
- **BLF resistance** escalation level and activities
- **Active invasions** with phase tracking
- **Research projects** in progress

### Interactive Controls
- **SPACE** — Pause/Resume auto-play
- **N** — Next turn (when paused)
- **+/-** — Speed up/slow down (0.1× to 10×)
- **S** — Toggle sector names
- **F** — Toggle fleet display
- **Click sectors** — View detailed information
- **ESC/Q** — Quit

### Visual Elements
- **Color-coded factions**: Oceania (blue), Eurasia (red), BLF (green)
- **Sector size**: Capitals (London, Paris) shown larger
- **Unit badges**: Number of land units in each sector
- **Sea zone fleets**: Ship counts by faction
- **Top bar**: Turn counter, date, scores, game status
- **Bottom bar**: War correspondent dispatches

## Installation

```bash
# Install Pygame if not already present
pip install pygame

# Generate map assets (one-time)
.venv/bin/python gui/generate_map.py

# Run the GUI
.venv/bin/python gui/main.py --seed 42 --turns 100 --speed 1.0
```

## Command Line Options

```bash
.venv/bin/python gui/main.py [OPTIONS]

Options:
  --seed SEED       Random seed for reproducible games (default: 42)
  --turns TURNS     Maximum number of turns to simulate (default: 100)
  --speed SPEED     Game speed multiplier (default: 1.0)
  --help, -h        Show this help message
```

## Map Generation

The `generate_map.py` script creates geographic assets from Natural Earth data:

```bash
.venv/bin/python gui/generate_map.py
```

This generates:
- `gui/assets/map_western_europe.png` — Base map image
- `gui/assets/sector_positions.json` — Sector pixel coordinates

The map uses real geographic coordinates projected to screen positions for accurate relative placement.

## GUI Architecture

### Main Components

```python
class AirStripOneGUI:
    def __init__(self, seed=42, max_turns=100, speed=1.0)
    def run()                     # Main game loop
    def _handle_events()         # Input processing
    def _update(dt)              # Game logic updates
    def _draw()                  # Rendering
```

### Rendering Pipeline

1. **Background**: Real map image or sea color
2. **Connections**: Sector adjacency lines
3. **Sea Zones**: Labels and fleet indicators
4. **Sectors**: Faction-colored circles with names
5. **Overlays**: Selected sector highlights, unit badges
6. **UI Panels**: Top bar, right panel, bottom dispatch bar

### Data Integration

The GUI connects directly to the GRAVITAS Engine:
- **Game state**: `gravitas.llm_game.create_game()`
- **Turn stepping**: `gravitas.llm_game.step_game()`
- **Event generation**: `gravitas.llm_game.generate_visible_events()`
- **Summaries**: `gravitas.llm_game.summarize_turn()`

## Performance

- **Target FPS**: 30 FPS for smooth animation
- **Turn speed**: 1 second per turn at 1× speed
- **Memory usage**: ~50MB for map assets + game state
- **CPU usage**: Light during pause, moderate during auto-play

## Troubleshooting

### Map Not Loading
```bash
# Regenerate map assets
.venv/bin/python gui/generate_map.py
```

### Performance Issues
- Reduce game speed with `-` key
- Toggle fleet display with `F` key
- Close other applications to free memory

### Display Issues
- Ensure Pygame is installed: `pip install pygame`
- Check that map assets exist in `gui/assets/`
- Try windowed mode if fullscreen fails

## File Structure

```
gui/
├── main.py              # Main GUI application
├── generate_map.py      # Asset generation script
├── __init__.py          # GUI package init
└── assets/              # Generated map assets
    ├── map_western_europe.png
    └── sector_positions.json
```

## Dependencies

- **Python 3.9+**
- **Pygame 2.5+**
- **NumPy** (from main project)

The GUI imports from the main GRAVITAS Engine, so ensure the project is properly installed with all dependencies.
