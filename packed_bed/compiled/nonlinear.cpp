// Public SUNDIALS 7.5 nonlinear operation interface, with early stale-Jacobian
// refresh. Newton control flow is
// adapted from SUNDIALS (BSD-3-Clause).
// Copyright (c) 2025, Lawrence Livermore National Security, University of Maryland
// Baltimore County, and the SUNDIALS contributors.
// Copyright (c) 2013-2025, Lawrence Livermore National Security and Southern Methodist University.
// Copyright (c) 2002-2013, Lawrence Livermore National Security. All rights reserved.
// See licenses/SUNDIALS-LICENSE.txt and licenses/SUNDIALS-NOTICE.txt.
#include <algorithm>
#include <new>

using Ptr = void *;
struct Nonlinear {
    Ptr content;
    Ptr *ops;
    Ptr context;
};
using NewEmpty = Nonlinear *(*)(Ptr);
using FreeEmpty = void (*)(Nonlinear *);
using Clone = Ptr (*)(Ptr);
using Destroy = void (*)(Ptr);
using Scale = void (*)(double, Ptr, Ptr);
using Sum = void (*)(double, Ptr, double, Ptr, Ptr);
using Constant = void (*)(double, Ptr);
using System = int (*)(Ptr, Ptr, Ptr);
using Setup = int (*)(int, int *, Ptr);
using Solve = int (*)(Ptr, Ptr);
using Test = int (*)(Nonlinear *, Ptr, Ptr, double, Ptr, Ptr);
constexpr int Continue = 901, Recover = 902;

struct Content {
    System sys = nullptr;
    Setup setup = nullptr;
    Solve solve = nullptr;
    Test test = nullptr;
    Ptr test_data = nullptr;
    FreeEmpty release;
    Destroy destroy;
    Scale scale;
    Sum sum;
    Constant constant;
    Ptr delta = nullptr, previous = nullptr;
    int refresh, maxiters = 4, current = 0, jcur = 0;
    long iterations = 0, failures = 0;
    long long totals[3] = {}; // nonlinear solves, all restarts, proactive refreshes
    Content(int refresh, FreeEmpty release, Destroy destroy, Scale scale, Sum sum,
            Constant constant)
        : refresh(refresh), release(release), destroy(destroy), scale(scale), sum(sum),
          constant(constant) {}
};
static int type(Nonlinear *) { return 0; }
static int initialize(Nonlinear *s) {
    auto &c = *static_cast<Content *>(s->content);
    if (!c.sys || !c.solve || !c.test)
        return -1;
    c.iterations = c.failures = 0;
    c.jcur = 0;
    std::fill(c.totals, c.totals + 3, 0);
    return 0;
}
static int set_system(Nonlinear *s, System f) {
    static_cast<Content *>(s->content)->sys = f;
    return f ? 0 : -1;
}
static int set_setup(Nonlinear *s, Setup f) {
    static_cast<Content *>(s->content)->setup = f;
    return 0;
}
static int set_solve(Nonlinear *s, Solve f) {
    static_cast<Content *>(s->content)->solve = f;
    return f ? 0 : -1;
}
static int set_test(Nonlinear *s, Test f, Ptr p) {
    auto &c = *static_cast<Content *>(s->content);
    c.test = f;
    c.test_data = p;
    return f ? 0 : -1;
}
static int set_maximum(Nonlinear *s, int n) {
    if (n < 1)
        return -1;
    static_cast<Content *>(s->content)->maxiters = n;
    return 0;
}
static int get_iterations(Nonlinear *s, long *n) {
    *n = static_cast<Content *>(s->content)->iterations;
    return 0;
}
static int get_current(Nonlinear *s, int *n) {
    *n = static_cast<Content *>(s->content)->current;
    return 0;
}
static int get_failures(Nonlinear *s, long *n) {
    *n = static_cast<Content *>(s->content)->failures;
    return 0;
}

static int solve(Nonlinear *solver, Ptr, Ptr correction, Ptr weights, double tolerance,
                 int call_setup, Ptr memory) {
    auto &c = *static_cast<Content *>(solver->content);
    if (!c.sys || !c.solve || !c.test || (call_setup && !c.setup))
        return -1;
    c.iterations = c.failures = 0;
    c.totals[0]++;
    int flag = 0, bad = 0;
    for (;;) {
        c.current = 0;
        flag = c.sys(correction, c.delta, memory);
        if (flag != 0)
            break;
        if (call_setup) {
            flag = c.setup(bad, &c.jcur, memory);
            if (flag != 0)
                break;
        }
        bool proactive = false;
        for (;;) {
            c.iterations++;
            // This point has a successfully evaluated residual. Keep it even
            // if the following linear solve returns a recoverable failure.
            if (c.refresh)
                c.scale(1., correction, c.previous);
            c.scale(-1., c.delta, c.delta);
            flag = c.solve(c.delta, memory);
            if (flag != 0)
                break;
            c.sum(1., correction, 1., c.delta, correction);
            flag = c.test(solver, correction, c.delta, tolerance, weights, c.test_data);
            c.current++;
            if (flag == 0) {
                c.jcur = 0;
                return 0;
            }
            if (flag != Continue)
                break;
            if (c.refresh && !c.jcur && c.setup && c.current >= c.refresh) {
                proactive = true;
                flag = Recover;
                break;
            }
            if (c.current >= c.maxiters) {
                flag = Recover;
                break;
            }
            flag = c.sys(correction, c.delta, memory);
            if (flag != 0)
                break;
        }
        if (flag > 0 && !c.jcur && c.setup) {
            // A proactive refresh is work, but not a convergence failure.
            if (!proactive)
                c.failures++;
            else
                c.totals[2]++;
            c.totals[1]++;
            call_setup = 1;
            bad = 1;
            // Retry with a fresh Jacobian at the last evaluated iterate.
            if (c.refresh)
                c.scale(1., c.previous, correction);
            else
                c.constant(0., correction);
            continue;
        }
        break;
    }
    c.failures++;
    return flag;
}
static int release(Nonlinear *s) {
    if (!s)
        return 0;
    auto *c = static_cast<Content *>(s->content);
    auto empty = c->release;
    if (c->delta)
        c->destroy(c->delta);
    if (c->previous)
        c->destroy(c->previous);
    delete c;
    s->content = nullptr;
    empty(s);
    return 0;
}
extern "C" __declspec(dllexport) Nonlinear *make_solver(Ptr context, Ptr example, int refresh,
                                                        NewEmpty empty, FreeEmpty free_empty,
                                                        Clone clone, Destroy destroy, Scale scale,
                                                        Sum sum, Constant constant) {
    if (refresh < 0)
        return nullptr;
    auto *s = empty(context);
    if (!s)
        return nullptr;
    auto *c = new (std::nothrow) Content(refresh, free_empty, destroy, scale, sum, constant);
    if (!c) {
        free_empty(s);
        return nullptr;
    }
    s->content = c;
    c->delta = clone(example);
    c->previous = clone(example);
    if (!c->delta || !c->previous) {
        release(s);
        return nullptr;
    }
    Ptr ops[] = {reinterpret_cast<Ptr>(type),
                 reinterpret_cast<Ptr>(initialize),
                 nullptr,
                 reinterpret_cast<Ptr>(solve),
                 reinterpret_cast<Ptr>(release),
                 reinterpret_cast<Ptr>(set_system),
                 reinterpret_cast<Ptr>(set_setup),
                 reinterpret_cast<Ptr>(set_solve),
                 reinterpret_cast<Ptr>(set_test),
                 nullptr,
                 reinterpret_cast<Ptr>(set_maximum),
                 reinterpret_cast<Ptr>(get_iterations),
                 reinterpret_cast<Ptr>(get_current),
                 reinterpret_cast<Ptr>(get_failures)};
    std::copy(ops, ops + 14, s->ops);
    return s;
}
extern "C" __declspec(dllexport) void get_statistics(Nonlinear *s, long long *out) {
    auto &c = *static_cast<Content *>(s->content);
    std::copy(c.totals, c.totals + 3, out);
}
extern "C" __declspec(dllexport) int free_solver(Nonlinear *s) { return release(s); }
