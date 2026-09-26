"""Command line entry: search usable combinations and emit a full plan.

Default mission: LEO store-and-forward / messaging service that must serve the
whole globe including the polar caps, with a worst-case revisit cadence.

    python3 -m orbitforge.constellation.cli
    python3 -m orbitforge.constellation.cli --json-out plan.json --text-out plan.txt

Every evaluated combination is either accepted with metrics or rejected with
explicit per-criterion reasons.
"""
from __future__ import annotations

import argparse
import math

from orbitforge.constellation.report import (
    plan_to_dict,
    render_plan,
    render_search_report,
    report_to_dict,
    write_json,
)
from orbitforge.constellation.requirements import ServiceRequirements
from orbitforge.constellation.search import (
    CandidateResult,
    DesignSpace,
    SearchConfig,
    refine_minimum_satellite_shortlist,
    search_design_space,
)


def default_requirements() -> ServiceRequirements:
    # Whole globe, both polar caps above 75 deg, minimum 10 deg elevation,
    # no point anywhere (or at the poles) waiting more than 2 hours between
    # access windows.
    return ServiceRequirements.degrees(
        min_elevation_deg=10.0,
        polar_latitude_deg=75.0,
        min_global_coverage=1.0,
        max_global_gap_s=2.0 * 3600.0,
        min_polar_coverage=1.0,
        max_polar_gap_s=2.0 * 3600.0,
    )


def default_space() -> DesignSpace:
    # Sparse enough that some families miss coverage/cadence, dense enough to
    # include the continuous-coverage designs. All F values are enumerated.
    return DesignSpace(
        altitudes_km=[1000.0, 1200.0],
        inclinations_deg=[53.0, 83.0],
        planes=[6, 7, 9],
        per_plane=[8, 10],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Walker constellation service designer")
    ap.add_argument("--elevation-deg", type=float, default=10.0)
    ap.add_argument("--polar-lat-deg", type=float, default=75.0)
    ap.add_argument("--global-gap-hours", type=float, default=2.0)
    ap.add_argument("--polar-gap-hours", type=float, default=2.0)
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--step-s", type=float, default=240.0)
    ap.add_argument("--grid-deg", type=float, default=10.0)
    ap.add_argument("--fine-grid-deg", type=float, default=4.0)
    ap.add_argument("--fine-window-hours", type=float, default=48.0)
    ap.add_argument("--fine-step-s", type=float, default=120.0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--text-out", default=None)
    ap.add_argument("--max-accepted-show", type=int, default=12)
    args = ap.parse_args(argv)

    req = ServiceRequirements.degrees(
        min_elevation_deg=args.elevation_deg,
        polar_latitude_deg=args.polar_lat_deg,
        min_global_coverage=1.0,
        max_global_gap_s=args.global_gap_hours * 3600.0,
        min_polar_coverage=1.0,
        max_polar_gap_s=args.polar_gap_hours * 3600.0,
    )
    space = default_space()
    config = SearchConfig(
        duration_s=args.window_hours * 3600.0,
        step_s=args.step_s,
        grid_lat_step_deg=args.grid_deg,
        grid_lon_step_deg=args.grid_deg,
    )

    done = [0]

    def progress(i, n, cand: CandidateResult):
        done[0] = i
        tag = "ACCEPT" if cand.accepted else f"reject:{cand.rejection_stage or 'sim'}"
        print(f"[{i:>3}/{n}] {cand.pattern.pattern_id:<46} {tag}", flush=True)

    print(
        f"Searching {space.altitudes_km} km x inc {space.inclinations_deg} deg, "
        f"P in {space.planes}, sats/plane in {space.per_plane}"
    )
    report = search_design_space(space, req, config, progress=progress)
    print()
    print(render_search_report(report, show_accepted=args.max_accepted_show))

    text_blocks = [render_search_report(report, show_accepted=args.max_accepted_show)]
    plan_payload = None
    selected = None
    if report.accepted:
        print("\nFine-grid re-evaluation of the minimum-satellite shortlist ...")

        def fine_progress(cand):
            g = cand.metrics.globe
            tag = "SURVIVES" if cand.accepted else "fails fine grid"
            print(f"   {cand.pattern.pattern_id:<46} {tag} "
                  f"(cont {g.continuous_coverage * 100:.2f}%, "
                  f"wait {(g.worst_interior_gap_s or 0) / 3600:.2f} h)", flush=True)

        survivors, tested = refine_minimum_satellite_shortlist(
            report, req,
            duration_s=args.fine_window_hours * 3600.0,
            step_s=args.fine_step_s,
            grid_lat_step_deg=args.fine_grid_deg,
            grid_lon_step_deg=args.fine_grid_deg,
        )
        for cand in tested:
            fine_progress(cand)
        if survivors:
            selected = survivors[0]
            print(f"\nSelected after fine verification: {selected.pattern.pattern_id}")
            print(render_plan(selected.pattern, selected.metrics, selected.verdict))
            text_blocks.append(render_plan(selected.pattern, selected.metrics, selected.verdict))
            plan_payload = plan_to_dict(selected.pattern, selected.metrics, selected.verdict)
            plan_payload["fine_shortlist"] = [
                {
                    "id": c.pattern.pattern_id,
                    "accepted": c.accepted,
                    "reasons": c.rejection_reasons,
                    "continuous_coverage": c.metrics.globe.continuous_coverage,
                    "worst_gap_s": c.metrics.globe.worst_interior_gap_s,
                }
                for c in tested
            ]
        else:
            print("\nNOTE: no coarse pass survived fine-grid verification.")
            text_blocks.append("No combination survived fine-grid verification.")
    else:
        print("\nNo combination met the requirements; no plan emitted.")
        text_blocks.append("No accepted combination.")

    if args.json_out:
        payload = {
            "search": report_to_dict(report),
            "selected_plan": plan_payload,
        }
        write_json(payload, args.json_out)
        print(f"\nWrote {args.json_out}")
    if args.text_out:
        with open(args.text_out, "w") as fh:
            fh.write("\n\n".join(text_blocks) + "\n")
        print(f"Wrote {args.text_out}")
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
