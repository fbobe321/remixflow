"""Catalog of steering controls and identity-preservation elements.

This module is the single source of truth for every slider, selector, and
lock described in the PRD (section "Evolution Controls" and "Identity
Preservation"). The FastAPI layer exposes it via ``GET /api/controls`` so the
React UI can render the panel dynamically instead of hard-coding it in two
places. The generation engine reads the same catalog when mapping UI values
onto audio transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ControlKind = Literal["bipolar", "unipolar", "enum", "multi"]


@dataclass(frozen=True)
class Control:
    """One steering control.

    ``kind`` semantics:
      * ``bipolar``   -> float in [-1, 1], 0 == "leave as-is" (e.g. Tempo).
      * ``unipolar``  -> float in [0, 1] (e.g. Variation Amount, genre influence).
      * ``enum``      -> one string from ``options``.
      * ``multi``     -> subset of ``options``.
    """

    key: str
    label: str
    kind: ControlKind
    group: str
    left: str = ""
    right: str = ""
    default: float = 0.0
    options: tuple[str, ...] = field(default_factory=tuple)
    help: str = ""


# --- Continuous / discrete steering controls (PRD §3) -----------------------

CONTROLS: tuple[Control, ...] = (
    # The critical one first — governs the magnitude of every other change.
    Control("variation_amount", "Variation Amount", "unipolar", "core",
            "Nearly identical", "Completely reimagined", default=0.2,
            help="Global magnitude. 0% = nearly identical, 100% = reimagined."),
    Control("energy", "Energy", "bipolar", "dynamics", "Low", "High"),
    Control("tempo", "Tempo", "bipolar", "dynamics", "Slower", "Faster"),
    Control("emotional_tone", "Emotional Tone", "bipolar", "dynamics", "Sad", "Happy"),
    Control("complexity", "Complexity", "bipolar", "dynamics", "Simple", "Complex"),

    Control("blues", "Blues Influence", "unipolar", "genre", "Less", "More"),
    Control("rock", "Rock Influence", "unipolar", "genre", "Less", "More"),
    Control("jazz", "Jazz", "unipolar", "genre", "Less", "More"),
    Control("electronic", "Electronic", "unipolar", "genre", "Less", "More"),
    Control("acoustic", "Acoustic", "unipolar", "genre", "Less", "More"),

    Control("instrument_density", "Instrument Density", "bipolar", "arrangement", "Sparse", "Dense"),
    Control("groove", "Groove", "bipolar", "arrangement", "Relaxed", "Tight"),
    Control("swing", "Swing", "unipolar", "arrangement", "None", "Heavy"),
    Control("chorus_strength", "Chorus Strength", "bipolar", "arrangement", "Subtle", "Anthemic"),

    Control("bass", "Bass", "bipolar", "tone", "Light", "Heavy"),
    Control("brightness", "Brightness", "bipolar", "tone", "Dark", "Bright"),
    Control("warmth", "Warmth", "bipolar", "tone", "Cold", "Warm"),

    Control("vocal_style", "Vocal Style", "enum", "vocals",
            options=("original", "softer", "aggressive", "whisper", "powerful"),
            help="Reshape the lead vocal character."),
    Control("instrument_focus", "Instrument Focus", "multi", "arrangement",
            options=("guitar", "piano", "strings", "synth", "bass", "drums"),
            help="Push selected instruments forward in the mix."),
)

CONTROLS_BY_KEY: dict[str, Control] = {c.key: c for c in CONTROLS}

# --- Identity preservation locks (PRD §4) -----------------------------------
# Elements the user can choose to hold fixed while everything else evolves.

IDENTITY_ELEMENTS: tuple[str, ...] = (
    "melody",
    "chord_progression",
    "vocal_phrasing",
    "rhythm",
    "hook",
    "chorus",
    "lyrics",
    "instrumentation",
)

GROUPS: tuple[tuple[str, str], ...] = (
    ("core", "Core"),
    ("dynamics", "Dynamics"),
    ("genre", "Genre Influence"),
    ("arrangement", "Arrangement"),
    ("tone", "Tone Color"),
    ("vocals", "Vocals"),
)


def default_steering() -> dict[str, object]:
    """A neutral steering payload: every control at its rest position."""
    out: dict[str, object] = {}
    for c in CONTROLS:
        if c.kind == "enum":
            out[c.key] = c.options[0] if c.options else ""
        elif c.kind == "multi":
            out[c.key] = []
        else:
            out[c.key] = c.default
    return out


def controls_manifest() -> dict[str, object]:
    """JSON-serializable description of the whole control surface for the UI."""
    return {
        "groups": [{"key": k, "label": v} for k, v in GROUPS],
        "controls": [
            {
                "key": c.key,
                "label": c.label,
                "kind": c.kind,
                "group": c.group,
                "left": c.left,
                "right": c.right,
                "default": c.default,
                "options": list(c.options),
                "help": c.help,
            }
            for c in CONTROLS
        ],
        "identityElements": list(IDENTITY_ELEMENTS),
    }
