// Runtime boundary-program ABI. Storage and cache validity belong to one run.
#include <algorithm>
#include <cmath>
#include <limits>
struct ProgramData {
    int channels, outputs, valid;
    double width, time;
    const double* initial;
    const int* offsets;
    const int* repeat_offsets;
    const double* period;
    const double* start;
    const double* end;
    const double* base;
    const double* delta;
    const int* numerator;
    const int* denominator;
    double* raw;
    double* values;
};
static double positive_time(double x, double width) {
    const double clipped = std::min(std::max(x + width, 0.), 2*width);
    return std::max(x - width, 0.) + clipped*clipped/(4*width);
}
static const double* program_values(double t, ProgramData* p) {
    if (p->valid && p->time == t) return p->values;
    for (int i=0; i<p->channels; ++i) {
        double value = p->initial[i];
        const double period = p->period[i];
        const double first = period > 0 ? std::max(0., std::floor((t-p->width)/period)) : 0.;
        const double last = period > 0 ? std::max(0., std::floor((t+p->width)/period)) : 0.;
        for (double cycle=first; cycle<=last; cycle+=1.) {
            const int begin = cycle > 0 ? p->repeat_offsets[i] : p->offsets[i];
            const int end = cycle > 0 ? p->offsets[i+1] : p->repeat_offsets[i];
            if (begin == end) continue;
            const double local = t - cycle*period;
            const int start = std::upper_bound(p->end+begin, p->end+end, local-p->width) - p->end;
            if (cycle == first)
                value = start < end ? p->base[start] : p->base[end-1]+p->delta[end-1];
            for (int j=start; j<end && p->start[j]<local+p->width; ++j) {
                const double fraction = (positive_time(local-p->start[j],p->width)
                                        -positive_time(local-p->end[j],p->width))/(p->end[j]-p->start[j]);
                value += p->delta[j]*fraction;
            }
        }
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
