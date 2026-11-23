// Auto-generated Rust implementation by GlassBox for drone_altitude_pid_decompiled
#![cfg_attr(not(test), no_std)]

#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub alt_error: f64,
    pub vert_vel: f64,
    pub battery_v: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Output {
    pub thrust_cmd: f64,
    pub regime_id: usize,
}

/// Evaluate decompiled `drone_altitude_pid_decompiled` across 9 regimes.
/// - Regime 0 (`regime_0_alt_error_lo`): samples=166, local_err=0.00551433
/// - Regime 1 (`regime_1_battery_v_hi`): samples=116, local_err=0.000125982
/// - Regime 2 (`regime_2_battery_v_lo`): samples=272, local_err=0.00105086
/// - Regime 3 (`regime_3_alt_error_lo`): samples=205, local_err=0.00298111
/// - Regime 4 (`regime_4_alt_error_hi`): samples=38, local_err=0.00367024
/// - Regime 5 (`regime_5_battery_v_lo`): samples=53, local_err=0.00123741
/// - Regime 6 (`regime_6_battery_v_hi`): samples=565, local_err=0.00723927
/// - Regime 7 (`regime_7_battery_v_lo`): samples=143, local_err=6.29143e-05
/// - Regime 8 (`regime_8_battery_v_hi`): samples=503, local_err=0.00129268
pub fn infer(input: &Input) -> Output {
    let alt_error: f64 = input.alt_error;
    let vert_vel: f64 = input.vert_vel;
    let battery_v: f64 = input.battery_v;
    if -0.74962351092099433 * alt_error + 0.64170055605210197 * vert_vel + 0.16212645754751145 * battery_v - 0.78491885482520085 <= 0.0 {
        if 0.69655904566902527 * alt_error - 0.66064211815797913 * vert_vel - 0.27992407472812253 * battery_v + 1.690803498089037 <= 0.0 {
            // Regime 0: regime_0_alt_error_lo (samples=166, local_err=0.00551433)
            let out_0: f64 = (((0.17200909175059864 * alt_error) - (0.13345172708318423 * vert_vel)) - (0.043794336395068877 * battery_v)) + 1.0627682490804946;
            Output {
                thrust_cmd: out_0,
                regime_id: 0,
            }
        } else {
            if -0.35020144065054087 * alt_error - 0.51634167923251251 * vert_vel - 0.78150509995369521 * battery_v + 9.1532386499014375 <= 0.0 {
                // Regime 1: regime_1_battery_v_hi (samples=116, local_err=0.000125982)
                let out_0: f64 = (((0.079567937678728579 * alt_error) - (0.043340847288694605 * vert_vel)) - (0.0058271432231129603 * battery_v)) + 0.81820993896822758;
                Output {
                    thrust_cmd: out_0,
                    regime_id: 1,
                }
            } else {
                // Regime 2: regime_2_battery_v_lo (samples=272, local_err=0.00105086)
                let out_0: f64 = (((0.073557437375393969 * alt_error) - (0.052759296395063077 * vert_vel)) - (0.01979306104424567 * battery_v)) + 0.98122944197572248;
                Output {
                    thrust_cmd: out_0,
                    regime_id: 2,
                }
            }
        }
    } else {
        if 0.75645688983004422 * alt_error - 0.61724984079968992 * vert_vel - 0.21627669282984199 * battery_v + 4.7382050720550879 <= 0.0 {
            if 0.81640971945657925 * alt_error - 0.53395878687221876 * vert_vel - 0.21991631112488683 * battery_v + 4.8132150673354923 <= 0.0 {
                // Regime 3: regime_3_alt_error_lo (samples=205, local_err=0.00298111)
                let out_0: f64 = (((0.0019157625035161527 * alt_error) - (0.0010577777091629447 * vert_vel)) + (0.00084224720242533438 * battery_v)) + 0.0071267175366798562;
                Output {
                    thrust_cmd: out_0,
                    regime_id: 3,
                }
            } else {
                // Regime 4: regime_4_alt_error_hi (samples=38, local_err=0.00367024)
                let out_0: f64 = (((0.112831486947786 * alt_error) - (0.079971306471277967 * vert_vel)) - (0.012408911167648145 * battery_v)) + 0.49304250789764192;
                Output {
                    thrust_cmd: out_0,
                    regime_id: 4,
                }
            }
        } else {
            if -0.35020144065054087 * alt_error - 0.51634167923251251 * vert_vel - 0.78150509995369521 * battery_v + 9.1532386499014375 <= 0.0 {
                if 0.4568155631117814 * alt_error - 0.33020270349233222 * vert_vel + 0.82600588127762198 * battery_v - 8.4848541592326168 <= 0.0 {
                    // Regime 5: regime_5_battery_v_lo (samples=53, local_err=0.00123741)
                    let out_0: f64 = (((0.16991087825718212 * alt_error) - (0.12524755321297928 * vert_vel)) - (0.063000360211212122 * battery_v)) + 1.2213970920131891;
                    Output {
                        thrust_cmd: out_0,
                        regime_id: 5,
                    }
                } else {
                    // Regime 6: regime_6_battery_v_hi (samples=565, local_err=0.00723927)
                    let out_0: f64 = (((0.18304643001845006 * alt_error) - (0.13137747182019102 * vert_vel)) - (0.03056412780793934 * battery_v)) + 0.87713540153944192;
                    Output {
                        thrust_cmd: out_0,
                        regime_id: 6,
                    }
                }
            } else {
                if 0.4568155631117814 * alt_error - 0.33020270349233222 * vert_vel + 0.82600588127762198 * battery_v - 8.4848541592326168 <= 0.0 {
                    // Regime 7: regime_7_battery_v_lo (samples=143, local_err=6.29143e-05)
                    let out_0: f64 = (((0.16350309796675103 * alt_error) - (0.13495890188403151 * vert_vel)) - (0.08016935429785399 * battery_v)) + 1.4206134451813797;
                    Output {
                        thrust_cmd: out_0,
                        regime_id: 7,
                    }
                } else {
                    // Regime 8: regime_8_battery_v_hi (samples=503, local_err=0.00129268)
                    let out_0: f64 = (((0.17919503764774455 * alt_error) - (0.14619990945821282 * vert_vel)) - (0.051098190143886439 * battery_v)) + 1.121805866212239;
                    Output {
                        thrust_cmd: out_0,
                        regime_id: 8,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_vectors() {
        let inp_0 = Input {
            alt_error: 0.0,
            vert_vel: 0.0,
            battery_v: 11.550000000000001,
        };
        let out_0 = infer(&inp_0);
        assert_eq!(out_0.regime_id, 8);
        assert!((out_0.thrust_cmd - 0.53162177005035061).abs() < 1e-9);
        let inp_1 = Input {
            alt_error: -2.8034883240321538,
            vert_vel: -0.85518058056367741,
            battery_v: 11.932594507016679,
        };
        let out_1 = infer(&inp_1);
        assert_eq!(out_1.regime_id, 8);
        assert!((out_1.thrust_cmd - 0.13472801086168329).abs() < 1e-9);
        let inp_2 = Input {
            alt_error: 0.90471560846721566,
            vert_vel: -2.7869766480643077,
            battery_v: 11.256465883391458,
        };
        let out_2 = infer(&inp_2);
        assert_eq!(out_2.regime_id, 2);
        assert!((out_2.thrust_cmd - 0.9720170143369995).abs() < 1e-9);
        let inp_3 = Input {
            alt_error: -1.0270804590334159,
            vert_vel: 0.92122728443506174,
            battery_v: 10.580337259766237,
        };
        let out_3 = infer(&inp_3);
        assert_eq!(out_3.regime_id, 7);
        assert!((out_3.thrust_cmd - 0.28013597920635402).abs() < 1e-9);
        let inp_4 = Input {
            alt_error: 2.6811234734659539,
            vert_vel: -1.0105687830655685,
            battery_v: 11.878208636141018,
        };
        let out_4 = infer(&inp_4);
        assert_eq!(out_4.regime_id, 1);
        assert!((out_4.thrust_cmd - 1.0061242887586987).abs() < 1e-9);
    }
}
