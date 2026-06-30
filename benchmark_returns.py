"""Validate or retune the ten pilot serve-return presets."""

from __future__ import annotations

import argparse
import json

from return_parameter_search import (
    PILOT_SERVICE_PARAMS,
    RETURN_PRESET_VECTORS,
    ContactSelection,
    RubberProperties,
    SearchConfig,
    ServiceTargets,
    StrokeTargets,
    build_return_preset,
    search_return,
    validate_return,
    validate_service,
)
from table_tennis_simulation import simulate_racket_impact


PROFILE_TARGETS = {
    "cut_short": ("short", (-15.0, 35.0, 10.0)),
    "cut_two_bounce": ("two_bounce", (-15.0, 35.0, 10.0)),
    "cut_long": ("long", (-15.0, 35.0, 10.0)),
    "top_two_bounce": ("two_bounce", (0.0, -45.0, 0.0)),
    "top_long": ("long", (0.0, -45.0, 0.0)),
}


def build_cases():
    service_result = simulate_racket_impact(PILOT_SERVICE_PARAMS, t_max=3.0)
    for profile, (depth, spin) in PROFILE_TARGETS.items():
        for stroke_side in ("forehand", "backhand"):
            targets = StrokeTargets(
                depth=depth,
                direction="elbow",
                spin_rps=spin,
                stroke_side=stroke_side,
            )
            contact, index, params = build_return_preset(profile, stroke_side, service_result)
            yield profile, stroke_side, targets, contact, index, params, service_result


def validate_bank() -> list[dict[str, object]]:
    rows = []
    service_result = simulate_racket_impact(PILOT_SERVICE_PARAMS, t_max=3.0)
    service_validation = validate_service(PILOT_SERVICE_PARAMS, service_result, ServiceTargets())
    if not service_validation.passed:
        raise RuntimeError(f"Pilot service is invalid: {service_validation.violations}")
    for profile, stroke_side, targets, contact, index, params, _ in build_cases():
        result = simulate_racket_impact(params, t_max=3.0)
        validation = validate_return(params, result, targets)
        rows.append(
            {
                "profile": profile,
                "stroke": stroke_side,
                "moment": contact.moment,
                "contact_index": index,
                "passed": validation.passed,
                "bounce_error_mm": validation.bounce_error_mm,
                "spin_error_rps": validation.spin_error_rps,
                "net_clearance_mm": validation.net_clearance_mm,
                "target_bounces": validation.bounces_on_target_side,
                "violations": validation.violations,
            }
        )
    return rows


def retune(workers: int) -> dict[str, object]:
    service_result = simulate_racket_impact(PILOT_SERVICE_PARAMS, t_max=3.0)
    output = {}
    for profile, (depth, spin) in PROFILE_TARGETS.items():
        targets = StrokeTargets(
            depth=depth,
            direction="elbow",
            spin_rps=spin,
            stroke_side="forehand",
        )
        result = search_return(
            service_result,
            targets,
            contact=ContactSelection(moment=4),
            rubber=RubberProperties(),
            config=SearchConfig(workers=workers),
            use_validated_preset=False,
        )
        output[profile] = {
            "success": result.success,
            "cost": result.cost,
            "vector": result.vector,
            "validation": result.validation.to_dict(),
            "optimizer": result.optimizer,
            "message": result.message,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune", action="store_true", help="Run exhaustive searches instead of validating presets.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.retune:
        print(json.dumps(retune(args.workers), indent=2))
        return

    rows = validate_bank()
    print("profile,stroke,moment,contact_index,passed,bounce_error_mm,spin_error_rps,net_clearance_mm,target_bounces")
    for row in rows:
        print(
            f"{row['profile']},{row['stroke']},{row['moment']},{row['contact_index']},"
            f"{int(row['passed'])},{row['bounce_error_mm']:.3f},{row['spin_error_rps']:.3f},"
            f"{row['net_clearance_mm']:.3f},{row['target_bounces']}"
        )
    failures = [row for row in rows if not row["passed"]]
    print(f"summary cases={len(rows)} passed={len(rows) - len(failures)} failed={len(failures)}")
    print(f"worst_bounce_error_mm={max(row['bounce_error_mm'] for row in rows):.3f}")
    print(f"worst_spin_error_rps={max(row['spin_error_rps'] for row in rows):.3f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
