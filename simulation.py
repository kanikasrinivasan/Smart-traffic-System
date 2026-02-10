#!/usr/bin/env python3
"""Traffic signal simulation and controller using live counts from main.py."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


COUNT_RE = re.compile(r"COUNTS\s+NORTH=(\d+)\s+SOUTH=(\d+)\s+EAST=(\d+)\s+WEST=(\d+)")
DIRECTIONS = ("NORTH", "SOUTH", "EAST", "WEST")


@dataclass
class SignalState:
    green_direction: str
    green_remaining: float


@dataclass
class SimConfig:
    detection_cmd: List[str]
    min_green: float = 4.0
    max_green: float = 20.0
    per_vehicle: float = 1.2
    fps: int = 30


def parse_counts(line: str) -> Optional[Dict[str, int]]:
    m = COUNT_RE.search(line.strip())
    if not m:
        return None
    north, south, east, west = map(int, m.groups())
    return {"NORTH": north, "SOUTH": south, "EAST": east, "WEST": west}


def choose_green_direction(counts: Dict[str, int], current: str) -> str:
    max_count = max(counts.values())
    busiest = [d for d, v in counts.items() if v == max_count]
    if current in busiest:
        return current
    return busiest[0]


def compute_green_time(counts: Dict[str, int], direction: str, cfg: SimConfig) -> float:
    duration = cfg.min_green + counts.get(direction, 0) * cfg.per_vehicle
    return max(cfg.min_green, min(cfg.max_green, duration))


def launch_detector(cmd: List[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def iter_detector_counts(proc: subprocess.Popen[str]) -> Iterable[Dict[str, int]]:
    assert proc.stdout is not None
    for line in proc.stdout:
        parsed = parse_counts(line)
        if parsed is not None:
            yield parsed


def run_simulation(cfg: SimConfig) -> int:
    try:
        import pygame  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing pygame. Install with: pip install pygame") from exc

    pygame.init()
    screen = pygame.display.set_mode((980, 620))
    pygame.display.set_caption("AI Smart Traffic Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)
    small = pygame.font.SysFont("arial", 18)

    counts = {d: 0 for d in DIRECTIONS}
    state = SignalState(green_direction="NORTH", green_remaining=cfg.min_green)

    detector = launch_detector(cfg.detection_cmd)
    count_stream = iter_detector_counts(detector)
    next_update = time.time()

    colors = {
        "bg": (25, 28, 36),
        "road": (50, 54, 64),
        "text": (240, 240, 240),
        "green": (40, 220, 90),
        "red": (220, 60, 60),
        "amber": (230, 180, 60),
        "card": (37, 41, 52),
    }

    def draw() -> None:
        screen.fill(colors["bg"])

        pygame.draw.rect(screen, colors["road"], (390, 0, 200, 620))
        pygame.draw.rect(screen, colors["road"], (0, 210, 980, 200))

        title = font.render("AI Smart Traffic Management - Live Simulation", True, colors["text"])
        screen.blit(title, (220, 20))

        positions = {
            "NORTH": (450, 100),
            "SOUTH": (450, 500),
            "EAST": (760, 270),
            "WEST": (140, 270),
        }

        for direction, (x, y) in positions.items():
            is_green = direction == state.green_direction
            signal_color = colors["green"] if is_green else colors["red"]

            pygame.draw.circle(screen, signal_color, (x, y), 18)
            label = small.render(f"{direction}: {counts[direction]}", True, colors["text"])
            screen.blit(label, (x - 60, y + 28))

        panel = pygame.Rect(40, 500, 280, 100)
        pygame.draw.rect(screen, colors["card"], panel, border_radius=10)

        active = small.render(f"Active Green: {state.green_direction}", True, colors["amber"])
        remain = small.render(f"Time Left: {state.green_remaining:0.1f}s", True, colors["text"])
        screen.blit(active, (56, 525))
        screen.blit(remain, (56, 555))

        pygame.display.flip()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0

            now = time.time()
            if now >= next_update:
                try:
                    counts = next(count_stream)
                    state.green_direction = choose_green_direction(counts, state.green_direction)
                    state.green_remaining = compute_green_time(counts, state.green_direction, cfg)
                except StopIteration:
                    # Detector stopped; keep displaying last known state.
                    pass
                next_update = now + 0.5

            state.green_remaining = max(0.0, state.green_remaining - (1.0 / cfg.fps))
            if state.green_remaining <= 0.0:
                state.green_direction = choose_green_direction(counts, state.green_direction)
                state.green_remaining = compute_green_time(counts, state.green_direction, cfg)

            draw()
            clock.tick(cfg.fps)
    finally:
        if detector.poll() is None:
            detector.terminate()
            try:
                detector.wait(timeout=2)
            except subprocess.TimeoutExpired:
                detector.kill()
        pygame.quit()


def parse_args(argv: List[str]) -> SimConfig:
    p = argparse.ArgumentParser(description="Run smart traffic signal simulation with AI counts.")
    p.add_argument(
        "--detector-cmd",
        default=f"{sys.executable} main.py --mock",
        help="Command to launch detector subprocess",
    )
    p.add_argument("--min-green", type=float, default=4.0)
    p.add_argument("--max-green", type=float, default=20.0)
    p.add_argument("--per-vehicle", type=float, default=1.2)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)

    return SimConfig(
        detection_cmd=args.detector_cmd.split(),
        min_green=args.min_green,
        max_green=args.max_green,
        per_vehicle=args.per_vehicle,
        fps=args.fps,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv or sys.argv[1:])
    return run_simulation(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
