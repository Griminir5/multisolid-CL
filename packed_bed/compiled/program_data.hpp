// Runtime boundary-program ABI. Storage and cache validity belong to one run.
#include <cmath>
#include <limits>
struct ProgramData {
    int channels, ramps, outputs, valid;
    double width, time;
    const double* initial;
    const int* offsets;
    const int* ramp_index;
    const double* delta;
    const double* start;
    const double* end;
    const int* numerator;
    const int* denominator;
    double* fractions;
    double* raw;
    double* values;
};
static const double* program_values(double t, ProgramData* p) {
    if (p->valid && p->time == t) return p->values;
    const double width2 = p->width * p->width;
    for (int i=0; i<p->ramps; ++i) {
        const double a = t-p->start[i], b = t-p->end[i];
        p->fractions[i] = (0.5*(a+sqrt(a*a+width2))-0.5*(b+sqrt(b*b+width2)))
                       / (p->end[i]-p->start[i]);
    }
    for (int i=0; i<p->channels; ++i) {
        double value = p->initial[i];
        for (int j=p->offsets[i]; j<p->offsets[i+1]; ++j)
            value = value + p->delta[j]*p->fractions[p->ramp_index[j]];
        p->raw[i] = value;
    }
    for (int i=0; i<p->outputs; ++i) {
        double value = p->raw[p->numerator[i]];
        const int denominator = p->denominator[i];
        if (denominator >= 0) {
            const double flow = p->raw[denominator];
            value = (flow > 0 && std::isfinite(flow)) ? value/flow : std::numeric_limits<double>::quiet_NaN();
        }
        p->values[i] = value;
    }
    p->time = t;
    p->valid = 1;
    return p->values;
}
