/* Auto-generated C99 implementation by GlassBox for drone_altitude_pid_decompiled */
#include <math.h>
#include <stddef.h>
#ifdef GLASSBOX_MAIN
#include <stdio.h>
#endif

#ifndef GLASSBOX_DRONE_ALTITUDE_PID_DECOMPILED_H
#define GLASSBOX_DRONE_ALTITUDE_PID_DECOMPILED_H
#ifdef _WIN32
#define GLASSBOX_API __declspec(dllexport)
#else
#define GLASSBOX_API
#endif

#define GLASSBOX_INPUT_DIM 3
#define GLASSBOX_OUTPUT_DIM 1

typedef struct {
    double alt_error;
    double vert_vel;
    double battery_v;
} GlassBoxInput;

typedef struct {
    double thrust_cmd;
    int regime_id;
    int class_id;
} GlassBoxOutput;

GLASSBOX_API int glassbox_infer(const double* x, double* out_values, int* out_regime);
GLASSBOX_API GlassBoxOutput glassbox_infer_struct(const GlassBoxInput* in);
GLASSBOX_API double glassbox_bench_loop(const double* xs, int n_samples, int n_repeats, double* out_last);
#endif

static void glassbox_eval_core(
    const double* x,
    double* out_values,
    int* out_regime,
    int* out_class
) {
    const double alt_error = x[0];
    const double vert_vel = x[1];
    const double battery_v = x[2];
    int regime_id = 0;
    int class_id = -1;

    if (-0.74962351092099433 * alt_error + 0.64170055605210197 * vert_vel + 0.16212645754751145 * battery_v - 0.78491885482520085 <= 0.0) {
        if (0.69655904566902527 * alt_error - 0.66064211815797913 * vert_vel - 0.27992407472812253 * battery_v + 1.690803498089037 <= 0.0) {
            /* Regime 0: regime_0_alt_error_lo (samples=166, local_err=0.00551433) */
            out_values[0] = ((((0.17200909175059864 * alt_error) - (0.13345172708318423 * vert_vel)) - (0.043794336395068877 * battery_v)) + 1.0627682490804946);
            regime_id = 0;
            class_id = -1;
        } else {
            if (-0.35020144065054087 * alt_error - 0.51634167923251251 * vert_vel - 0.78150509995369521 * battery_v + 9.1532386499014375 <= 0.0) {
                /* Regime 1: regime_1_battery_v_hi (samples=116, local_err=0.000125982) */
                out_values[0] = ((((0.079567937678728579 * alt_error) - (0.043340847288694605 * vert_vel)) - (0.0058271432231129603 * battery_v)) + 0.81820993896822758);
                regime_id = 1;
                class_id = -1;
            } else {
                /* Regime 2: regime_2_battery_v_lo (samples=272, local_err=0.00105086) */
                out_values[0] = ((((0.073557437375393969 * alt_error) - (0.052759296395063077 * vert_vel)) - (0.01979306104424567 * battery_v)) + 0.98122944197572248);
                regime_id = 2;
                class_id = -1;
            }
        }
    } else {
        if (0.75645688983004422 * alt_error - 0.61724984079968992 * vert_vel - 0.21627669282984199 * battery_v + 4.7382050720550879 <= 0.0) {
            if (0.81640971945657925 * alt_error - 0.53395878687221876 * vert_vel - 0.21991631112488683 * battery_v + 4.8132150673354923 <= 0.0) {
                /* Regime 3: regime_3_alt_error_lo (samples=205, local_err=0.00298111) */
                out_values[0] = ((((0.0019157625035161527 * alt_error) - (0.0010577777091629447 * vert_vel)) + (0.00084224720242533438 * battery_v)) + 0.0071267175366798562);
                regime_id = 3;
                class_id = -1;
            } else {
                /* Regime 4: regime_4_alt_error_hi (samples=38, local_err=0.00367024) */
                out_values[0] = ((((0.112831486947786 * alt_error) - (0.079971306471277967 * vert_vel)) - (0.012408911167648145 * battery_v)) + 0.49304250789764192);
                regime_id = 4;
                class_id = -1;
            }
        } else {
            if (-0.35020144065054087 * alt_error - 0.51634167923251251 * vert_vel - 0.78150509995369521 * battery_v + 9.1532386499014375 <= 0.0) {
                if (0.4568155631117814 * alt_error - 0.33020270349233222 * vert_vel + 0.82600588127762198 * battery_v - 8.4848541592326168 <= 0.0) {
                    /* Regime 5: regime_5_battery_v_lo (samples=53, local_err=0.00123741) */
                    out_values[0] = ((((0.16991087825718212 * alt_error) - (0.12524755321297928 * vert_vel)) - (0.063000360211212122 * battery_v)) + 1.2213970920131891);
                    regime_id = 5;
                    class_id = -1;
                } else {
                    /* Regime 6: regime_6_battery_v_hi (samples=565, local_err=0.00723927) */
                    out_values[0] = ((((0.18304643001845006 * alt_error) - (0.13137747182019102 * vert_vel)) - (0.03056412780793934 * battery_v)) + 0.87713540153944192);
                    regime_id = 6;
                    class_id = -1;
                }
            } else {
                if (0.4568155631117814 * alt_error - 0.33020270349233222 * vert_vel + 0.82600588127762198 * battery_v - 8.4848541592326168 <= 0.0) {
                    /* Regime 7: regime_7_battery_v_lo (samples=143, local_err=6.29143e-05) */
                    out_values[0] = ((((0.16350309796675103 * alt_error) - (0.13495890188403151 * vert_vel)) - (0.08016935429785399 * battery_v)) + 1.4206134451813797);
                    regime_id = 7;
                    class_id = -1;
                } else {
                    /* Regime 8: regime_8_battery_v_hi (samples=503, local_err=0.00129268) */
                    out_values[0] = ((((0.17919503764774455 * alt_error) - (0.14619990945821282 * vert_vel)) - (0.051098190143886439 * battery_v)) + 1.121805866212239);
                    regime_id = 8;
                    class_id = -1;
                }
            }
        }
    }

    if (out_regime != NULL) {
        *out_regime = regime_id;
    }
    if (out_class != NULL) {
        *out_class = class_id;
    }
}

GLASSBOX_API int glassbox_infer(const double* x, double* out_values, int* out_regime) {
    if (x == NULL || out_values == NULL) {
        return -1;
    }
    glassbox_eval_core(x, out_values, out_regime, NULL);
    return 0;
}

GLASSBOX_API GlassBoxOutput glassbox_infer_struct(const GlassBoxInput* in) {
    GlassBoxOutput result = {0};
    if (in == NULL) {
        result.regime_id = -1;
        result.class_id = -1;
        return result;
    }
    const double x[3] = {
        in->alt_error,
        in->vert_vel,
        in->battery_v
    };
    double out_vals[1];
    glassbox_eval_core(x, out_vals, &result.regime_id, &result.class_id);
    result.thrust_cmd = out_vals[0];
    return result;
}

GLASSBOX_API double glassbox_bench_loop(
    const double* xs,
    int n_samples,
    int n_repeats,
    double* out_last
) {
    if (xs == NULL || n_samples <= 0 || n_repeats <= 0) {
        return 0.0;
    }
    double local_out[GLASSBOX_OUTPUT_DIM] = {0};
    int local_regime = 0;
    volatile double acc = 0.0;
    for (int r = 0; r < n_repeats; ++r) {
        for (int i = 0; i < n_samples; ++i) {
            const double* row = xs + (size_t)i * GLASSBOX_INPUT_DIM;
            glassbox_eval_core(row, local_out, &local_regime, NULL);
            acc = acc + local_out[0] + (double)local_regime * 1e-12;
        }
    }
    if (out_last != NULL) {
        for (int j = 0; j < GLASSBOX_OUTPUT_DIM; ++j) {
            out_last[j] = local_out[j];
        }
    }
    return acc;
}

#ifdef GLASSBOX_MAIN
int main(void) {
    const double gv_x[5][3] = {
        {0.0, 0.0, 11.550000000000001},
        {-2.8034883240321538, -0.85518058056367741, 11.932594507016679},
        {0.90471560846721566, -2.7869766480643077, 11.256465883391458},
        {-1.0270804590334159, 0.92122728443506174, 10.580337259766237},
        {2.6811234734659539, -1.0105687830655685, 11.878208636141018}
    };
    const double gv_y[5][1] = {
        {0.53162177005035061},
        {0.13472801086168329},
        {0.9720170143369995},
        {0.28013597920635402},
        {1.0061242887586987}
    };
    const int gv_regime[5] = {8, 8, 2, 7, 1};
    for (int i = 0; i < 5; ++i) {
        double out[1];
        int regime = -1;
        if (glassbox_infer(gv_x[i], out, &regime) != 0) {
            fprintf(stderr, "FAIL vector %d: glassbox_infer returned non-zero\n", i);
            return 1;
        }
        if (regime != gv_regime[i]) {
            fprintf(stderr, "FAIL vector %d: regime %d != %d\n", i, regime, gv_regime[i]);
            return 1;
        }
        for (int j = 0; j < 1; ++j) {
            if (fabs(out[j] - gv_y[i][j]) > 1e-9) {
                fprintf(stderr, "FAIL vector %d out[%d]: %.17g != %.17g\n", i, j, out[j], gv_y[i][j]);
                return 1;
            }
        }
        GlassBoxInput s_in;
        s_in.alt_error = gv_x[i][0];
        s_in.vert_vel = gv_x[i][1];
        s_in.battery_v = gv_x[i][2];
        GlassBoxOutput s_out = glassbox_infer_struct(&s_in);
        if (s_out.regime_id != gv_regime[i]) {
            fprintf(stderr, "FAIL struct vector %d: regime %d != %d\n", i, s_out.regime_id, gv_regime[i]);
            return 1;
        }
        if (fabs(s_out.thrust_cmd - gv_y[i][0]) > 1e-9) {
            fprintf(stderr, "FAIL struct vector %d thrust_cmd: %.17g != %.17g\n", i, s_out.thrust_cmd, gv_y[i][0]);
            return 1;
        }
    }
    double bench_last[1] = {0};
    (void)glassbox_bench_loop(&gv_x[0][0], 5, 2, bench_last);
    for (int j = 0; j < 1; ++j) {
        if (fabs(bench_last[j] - gv_y[4][j]) > 1e-9) {
            fprintf(stderr, "FAIL bench_loop out[%d]\n", j);
            return 1;
        }
    }
    printf("PASS: %d/%d golden vectors verified.\n", 5, 5);
    return 0;
}
#endif /* GLASSBOX_MAIN */
