"""Auto-generated standalone Python implementation for drone_altitude_pid_decompiled."""

from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass(frozen=True)
class GlassBoxOutput:
    thrust_cmd: float
    regime_id: int
    class_id: int | None = None

    def to_list(self) -> list[float]:
        return [self.thrust_cmd]


def infer_with_regime(x: list[float] | tuple[float, ...]) -> tuple[list[float], int, int | None]:
    """Evaluate drone_altitude_pid_decompiled and return (outputs, regime_id, class_id)."""
    alt_error = float(x[0])
    vert_vel = float(x[1])
    battery_v = float(x[2])
    if -0.74962351092099433 * alt_error + 0.64170055605210197 * vert_vel + 0.16212645754751145 * battery_v - 0.78491885482520085 <= 0.0:
        if 0.69655904566902527 * alt_error - 0.66064211815797913 * vert_vel - 0.27992407472812253 * battery_v + 1.690803498089037 <= 0.0:
            # Regime 0: regime_0_alt_error_lo (samples=166, local_err=0.00551433)
            values = [((((0.17200909175059864 * alt_error) - (0.13345172708318423 * vert_vel)) - (0.043794336395068877 * battery_v)) + 1.0627682490804946)]
            return values, 0, None
        else:
            if -0.35020144065054087 * alt_error - 0.51634167923251251 * vert_vel - 0.78150509995369521 * battery_v + 9.1532386499014375 <= 0.0:
                # Regime 1: regime_1_battery_v_hi (samples=116, local_err=0.000125982)
                values = [((((0.079567937678728579 * alt_error) - (0.043340847288694605 * vert_vel)) - (0.0058271432231129603 * battery_v)) + 0.81820993896822758)]
                return values, 1, None
            else:
                # Regime 2: regime_2_battery_v_lo (samples=272, local_err=0.00105086)
                values = [((((0.073557437375393969 * alt_error) - (0.052759296395063077 * vert_vel)) - (0.01979306104424567 * battery_v)) + 0.98122944197572248)]
                return values, 2, None
    else:
        if 0.75645688983004422 * alt_error - 0.61724984079968992 * vert_vel - 0.21627669282984199 * battery_v + 4.7382050720550879 <= 0.0:
            if 0.81640971945657925 * alt_error - 0.53395878687221876 * vert_vel - 0.21991631112488683 * battery_v + 4.8132150673354923 <= 0.0:
                # Regime 3: regime_3_alt_error_lo (samples=205, local_err=0.00298111)
                values = [((((0.0019157625035161527 * alt_error) - (0.0010577777091629447 * vert_vel)) + (0.00084224720242533438 * battery_v)) + 0.0071267175366798562)]
                return values, 3, None
            else:
                # Regime 4: regime_4_alt_error_hi (samples=38, local_err=0.00367024)
                values = [((((0.112831486947786 * alt_error) - (0.079971306471277967 * vert_vel)) - (0.012408911167648145 * battery_v)) + 0.49304250789764192)]
                return values, 4, None
        else:
            if -0.35020144065054087 * alt_error - 0.51634167923251251 * vert_vel - 0.78150509995369521 * battery_v + 9.1532386499014375 <= 0.0:
                if 0.4568155631117814 * alt_error - 0.33020270349233222 * vert_vel + 0.82600588127762198 * battery_v - 8.4848541592326168 <= 0.0:
                    # Regime 5: regime_5_battery_v_lo (samples=53, local_err=0.00123741)
                    values = [((((0.16991087825718212 * alt_error) - (0.12524755321297928 * vert_vel)) - (0.063000360211212122 * battery_v)) + 1.2213970920131891)]
                    return values, 5, None
                else:
                    # Regime 6: regime_6_battery_v_hi (samples=565, local_err=0.00723927)
                    values = [((((0.18304643001845006 * alt_error) - (0.13137747182019102 * vert_vel)) - (0.03056412780793934 * battery_v)) + 0.87713540153944192)]
                    return values, 6, None
            else:
                if 0.4568155631117814 * alt_error - 0.33020270349233222 * vert_vel + 0.82600588127762198 * battery_v - 8.4848541592326168 <= 0.0:
                    # Regime 7: regime_7_battery_v_lo (samples=143, local_err=6.29143e-05)
                    values = [((((0.16350309796675103 * alt_error) - (0.13495890188403151 * vert_vel)) - (0.08016935429785399 * battery_v)) + 1.4206134451813797)]
                    return values, 7, None
                else:
                    # Regime 8: regime_8_battery_v_hi (samples=503, local_err=0.00129268)
                    values = [((((0.17919503764774455 * alt_error) - (0.14619990945821282 * vert_vel)) - (0.051098190143886439 * battery_v)) + 1.121805866212239)]
                    return values, 8, None


def infer(x: list[float] | tuple[float, ...]) -> list[float]:
    """Evaluate drone_altitude_pid_decompiled and return output values."""
    values, _, _ = infer_with_regime(x)
    return values


def infer_struct(x: list[float] | tuple[float, ...]) -> GlassBoxOutput:
    """Evaluate drone_altitude_pid_decompiled and return a typed GlassBoxOutput."""
    values, regime_id, class_id = infer_with_regime(x)
    return GlassBoxOutput(
        thrust_cmd=values[0],
        regime_id=regime_id,
        class_id=class_id,
    )


if __name__ == "__main__":
    _golden = [
        ([0.0, 0.0, 11.550000000000001], [0.53162177005035061], 8),
        ([-2.8034883240321538, -0.85518058056367741, 11.932594507016679], [0.13472801086168329], 8),
        ([0.90471560846721566, -2.7869766480643077, 11.256465883391458], [0.9720170143369995], 2),
        ([-1.0270804590334159, 0.92122728443506174, 10.580337259766237], [0.28013597920635402], 7),
        ([2.6811234734659539, -1.0105687830655685, 11.878208636141018], [1.0061242887586987], 1),
    ]
    for idx, (gx, gy, grid) in enumerate(_golden):
        got_vals, got_reg, _ = infer_with_regime(gx)
        assert got_reg == grid, f'vector {idx}: regime {got_reg} != {grid}'
        for j, (got_v, exp_v) in enumerate(zip(got_vals, gy)):
            assert math.fabs(got_v - exp_v) <= 1e-9, (
                f'vector {idx} out[{j}]: {got_v} != {exp_v}'
            )
    print(f"PASS: {len(_golden)}/{len(_golden)} golden vectors verified.")
