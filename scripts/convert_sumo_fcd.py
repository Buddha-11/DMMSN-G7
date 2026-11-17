"""Convert a SUMO floating-car-data XML trace to fixed-population VEC CSV.

The converter keeps only vehicles present in every selected timestep. This is
intentional: the current environment has a fixed role population, and silently
imputing vehicles before entry or after exit would invalidate radio geometry.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vehicle-count", type=int, default=3)
    parser.add_argument("--vehicle-ids", default="")
    parser.add_argument("--begin", type=float)
    parser.add_argument("--end", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames: list[tuple[float, dict[str, tuple[float, float, float]]]] = []
    for _, element in ET.iterparse(args.input, events=("end",)):
        if element.tag != "timestep":
            continue
        time = float(element.attrib["time"])
        if args.begin is not None and time < args.begin:
            element.clear()
            continue
        if args.end is not None and time > args.end:
            element.clear()
            continue
        positions = {
            vehicle.attrib["id"]: (
                float(vehicle.attrib["x"]),
                float(vehicle.attrib["y"]),
                float(vehicle.attrib.get("z", 0.0)),
            )
            for vehicle in element.findall("vehicle")
        }
        frames.append((time, positions))
        element.clear()
    if not frames:
        raise SystemExit("no SUMO timesteps matched the requested interval")
    requested = tuple(value for value in args.vehicle_ids.split(",") if value)
    common = set(frames[0][1])
    for _, positions in frames[1:]:
        common.intersection_update(positions)
    selected = requested or tuple(sorted(common)[: args.vehicle_count])
    missing = set(selected) - common
    if missing:
        raise SystemExit(f"vehicles missing from at least one timestep: {sorted(missing)}")
    if len(selected) != args.vehicle_count:
        raise SystemExit(
            f"needed {args.vehicle_count} complete vehicles, found {len(selected)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("slot", "vehicle_id", "x", "y", "z", "sumo_time_s"))
        for slot, (time, positions) in enumerate(frames):
            for vehicle_id in selected:
                writer.writerow((slot, vehicle_id, *positions[vehicle_id], time))
    print(
        f"wrote {len(frames)} frames for {len(selected)} vehicles to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
