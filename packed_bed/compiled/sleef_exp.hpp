// Copyright Naoki Shibata and contributors 2010 - 2025.
// Extracted from the generated SLEEF 3.9.0 AVX2 header: exp_u10 and its helpers.
// Distributed under the Boost Software License, Version 1.0.
// See licenses/SLEEF-LICENSE.txt. Requires AVX2 and FMA3.
#include <immintrin.h>
namespace packed_bed_sleef {
using vdouble_avx2_sleef=__m256d;
using vint_avx2_sleef=__m128i;
using vmask_avx2_sleef=__m256i;
using vopmask_avx2_sleef=__m256i;
#define SLEEF_ALWAYS_INLINE PB_INLINE
#define SLEEF_INLINE static inline
#define SLEEF_CONST
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vcast_vd_d_avx2_sleef(double d) { return _mm256_set1_pd(d); }
static SLEEF_ALWAYS_INLINE vmask_avx2_sleef vreinterpret_vm_vd_avx2_sleef(vdouble_avx2_sleef vd_avx2_sleef) { return _mm256_castpd_si256(vd_avx2_sleef); }
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vreinterpret_vd_vm_avx2_sleef(vmask_avx2_sleef vm) { return _mm256_castsi256_pd(vm);  }
static SLEEF_ALWAYS_INLINE vmask_avx2_sleef vandnot_vm_vo64_vm_avx2_sleef(vopmask_avx2_sleef x, vmask_avx2_sleef y) { return vreinterpret_vm_vd_avx2_sleef(_mm256_andnot_pd(vreinterpret_vd_vm_avx2_sleef(x), vreinterpret_vd_vm_avx2_sleef(y))); }
static SLEEF_ALWAYS_INLINE vint_avx2_sleef vrint_vi_vd_avx2_sleef(vdouble_avx2_sleef vd_avx2_sleef) { return _mm256_cvtpd_epi32(vd_avx2_sleef); }
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vrint_vd_vd_avx2_sleef(vdouble_avx2_sleef vd_avx2_sleef) { return _mm256_round_pd(vd_avx2_sleef, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC); }
static SLEEF_ALWAYS_INLINE vint_avx2_sleef vcast_vi_i_avx2_sleef(int i) { return _mm_set1_epi32(i); }
static SLEEF_ALWAYS_INLINE vmask_avx2_sleef vcastu_vm_vi_avx2_sleef(vint_avx2_sleef vi) {
  return _mm256_slli_epi64(_mm256_cvtepi32_epi64(vi), 32);
}
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vmul_vd_vd_vd_avx2_sleef(vdouble_avx2_sleef x, vdouble_avx2_sleef y) { return _mm256_mul_pd(x, y); }
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vmla_vd_vd_vd_vd_avx2_sleef(vdouble_avx2_sleef x, vdouble_avx2_sleef y, vdouble_avx2_sleef z) { return _mm256_fmadd_pd(x, y, z); }
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vfma_vd_vd_vd_vd_avx2_sleef(vdouble_avx2_sleef x, vdouble_avx2_sleef y, vdouble_avx2_sleef z) { return _mm256_fmadd_pd(x, y, z); }
static SLEEF_ALWAYS_INLINE vopmask_avx2_sleef vlt_vo_vd_vd_avx2_sleef(vdouble_avx2_sleef x, vdouble_avx2_sleef y) { return vreinterpret_vm_vd_avx2_sleef(_mm256_cmp_pd(x, y, _CMP_LT_OQ)); }
static SLEEF_ALWAYS_INLINE vopmask_avx2_sleef vgt_vo_vd_vd_avx2_sleef(vdouble_avx2_sleef x, vdouble_avx2_sleef y) { return vreinterpret_vm_vd_avx2_sleef(_mm256_cmp_pd(x, y, _CMP_GT_OQ)); }
static SLEEF_ALWAYS_INLINE vint_avx2_sleef vadd_vi_vi_vi_avx2_sleef(vint_avx2_sleef x, vint_avx2_sleef y) { return _mm_add_epi32(x, y); }
static SLEEF_ALWAYS_INLINE vint_avx2_sleef vsub_vi_vi_vi_avx2_sleef(vint_avx2_sleef x, vint_avx2_sleef y) { return _mm_sub_epi32(x, y); }
static SLEEF_ALWAYS_INLINE vint_avx2_sleef vsll_vi_vi_i_avx2_sleef(vint_avx2_sleef x, int c) { return _mm_slli_epi32(x, c); }
static SLEEF_ALWAYS_INLINE vint_avx2_sleef vsra_vi_vi_i_avx2_sleef(vint_avx2_sleef x, int c) { return _mm_srai_epi32(x, c); }
static SLEEF_ALWAYS_INLINE vdouble_avx2_sleef vsel_vd_vo_vd_vd_avx2_sleef(vopmask_avx2_sleef o, vdouble_avx2_sleef x, vdouble_avx2_sleef y) { return _mm256_blendv_pd(y, x, _mm256_castsi256_pd(o)); }
static SLEEF_ALWAYS_INLINE SLEEF_CONST  vdouble_avx2_sleef vpow2i_vd_vi_avx2_sleef(vint_avx2_sleef q) {
  q = vadd_vi_vi_vi_avx2_sleef(vcast_vi_i_avx2_sleef(0x3ff), q);
  vmask_avx2_sleef r = vcastu_vm_vi_avx2_sleef(vsll_vi_vi_i_avx2_sleef(q, 20));
  return vreinterpret_vd_vm_avx2_sleef(r);
}
static SLEEF_ALWAYS_INLINE SLEEF_CONST  vdouble_avx2_sleef vldexp2_vd_vd_vi_avx2_sleef(vdouble_avx2_sleef d, vint_avx2_sleef e) {
  return vmul_vd_vd_vd_avx2_sleef(vmul_vd_vd_vd_avx2_sleef(d, vpow2i_vd_vi_avx2_sleef(vsra_vi_vi_i_avx2_sleef(e, 1))), vpow2i_vd_vi_avx2_sleef(vsub_vi_vi_vi_avx2_sleef(e, vsra_vi_vi_i_avx2_sleef(e, 1))));
}
SLEEF_INLINE SLEEF_CONST  vdouble_avx2_sleef Sleef_expd4_u10avx2(vdouble_avx2_sleef d) {
  vdouble_avx2_sleef u = vrint_vd_vd_avx2_sleef(vmul_vd_vd_vd_avx2_sleef(d, vcast_vd_d_avx2_sleef(1.442695040888963407359924681001892137426645954152985934135449406931))), s;
  vint_avx2_sleef q = vrint_vi_vd_avx2_sleef(u);

  s = vmla_vd_vd_vd_vd_avx2_sleef(u, vcast_vd_d_avx2_sleef(-.69314718055966295651160180568695068359375), d);
  s = vmla_vd_vd_vd_vd_avx2_sleef(u, vcast_vd_d_avx2_sleef(-.28235290563031577122588448175013436025525412068e-12), s);

  vdouble_avx2_sleef s2 = vmul_vd_vd_vd_avx2_sleef(s, s), s4 = vmul_vd_vd_vd_avx2_sleef(s2, s2), s8 = vmul_vd_vd_vd_avx2_sleef(s4, s4);
  u = vmla_vd_vd_vd_vd_avx2_sleef((s8), (vmla_vd_vd_vd_vd_avx2_sleef((s), (vcast_vd_d_avx2_sleef(+0.2081276378237164457e-8)), (vcast_vd_d_avx2_sleef(+0.2511210703042288022e-7)))), (vmla_vd_vd_vd_vd_avx2_sleef((s4), (vmla_vd_vd_vd_vd_avx2_sleef((s2), (vmla_vd_vd_vd_vd_avx2_sleef((s), (vcast_vd_d_avx2_sleef(+0.2755762628169491192e-6)), (vcast_vd_d_avx2_sleef(+0.2755723402025388239e-5)))), (vmla_vd_vd_vd_vd_avx2_sleef((s), (vcast_vd_d_avx2_sleef(+0.2480158687479686264e-4)), (vcast_vd_d_avx2_sleef(+0.1984126989855865850e-3)))))), (vmla_vd_vd_vd_vd_avx2_sleef((s2), (vmla_vd_vd_vd_vd_avx2_sleef((s), (vcast_vd_d_avx2_sleef(+0.1388888888914497797e-2)), (vcast_vd_d_avx2_sleef(+0.8333333333314938210e-2)))), (vmla_vd_vd_vd_vd_avx2_sleef((s), (vcast_vd_d_avx2_sleef(+0.4166666666666602598e-1)), (vcast_vd_d_avx2_sleef(+0.1666666666666669072e+0)))))))));

  u = vfma_vd_vd_vd_vd_avx2_sleef(u, s, vcast_vd_d_avx2_sleef(+0.5000000000000000000e+0));
  u = vfma_vd_vd_vd_vd_avx2_sleef(u, s, vcast_vd_d_avx2_sleef(+0.1000000000000000000e+1));
  u = vfma_vd_vd_vd_vd_avx2_sleef(u, s, vcast_vd_d_avx2_sleef(+0.1000000000000000000e+1));

  u = vldexp2_vd_vd_vi_avx2_sleef(u, q);

  vopmask_avx2_sleef o = vgt_vo_vd_vd_avx2_sleef(d, vcast_vd_d_avx2_sleef(0x1.62e42fefa39efp+9));
  u = vsel_vd_vo_vd_vd_avx2_sleef(o, vcast_vd_d_avx2_sleef((1e+300 * 1e+300)), u);
  u = vreinterpret_vd_vm_avx2_sleef(vandnot_vm_vo64_vm_avx2_sleef(vlt_vo_vd_vd_avx2_sleef(d, vcast_vd_d_avx2_sleef(-1000)), vreinterpret_vm_vd_avx2_sleef(u)));

  return u;
}
#undef SLEEF_ALWAYS_INLINE
#undef SLEEF_INLINE
#undef SLEEF_CONST
}
