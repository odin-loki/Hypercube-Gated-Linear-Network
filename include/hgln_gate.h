// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_gate.h - the HGLN gate nonlinearity, shared by the C++ library and the CUDA kernel so both compute it the
// same way. Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#pragma once
#ifndef HGLN_NO_STD_HEADERS     // the header-free PTX build (cuda/hgln_ptx.cu) cannot include host headers
#include <cfloat>
#include <cmath>
#endif
#ifndef HGLN_HD
#if defined(__CUDACC__)
#define HGLN_HD __host__ __device__ __forceinline__
#else
#define HGLN_HD inline
#endif
#endif

// a = clip(1 / (1 + exp(-12 z)), 1e-4, 1 - 1e-4).
// Outside |12 z| <= 9.3 the clip always wins, so exp is only ever needed on [-9.3, 9.3]. There it is computed as
// 2^k * e^r with k = round(x / ln 2), |r| <= ln(2)/2 (Cody-Waite reduction, split ln 2) and a degree-13 Taylor
// polynomial (truncation error < 2e-16 relative), and 2^k (|k| <= 14) is assembled exactly from powers of two.
// The result agrees with 1/(1+std::exp(x)) to about 1 ulp, has no branches or table lookups, and so vectorises.
template <class T> HGLN_HD T hgln_gate(T z) {
  const T x = -(T)12 * z;
  const T xc = x > (T)9.3 ? (T)9.3 : (x < (T)-9.3 ? (T)-9.3 : x);
  // k = round-to-nearest(x / ln 2) with the 1.5 * 2^(mantissa bits) trick (exact for |x / ln 2| < 2^22; no libm call)
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
  const T kf = std::nearbyint(xc * (T)1.4426950408889634);         // excess-precision targets (e.g. x87): no trick
#else
  const T rnd = sizeof(T) == 4 ? (T)12582912.0 : (T)6755399441055744.0;
  const T kf = (xc * (T)1.4426950408889634 + rnd) - rnd;
#endif
  const T r = (xc - kf * (T)6.93147180369123816490e-01) - kf * (T)1.90821492927058770002e-10;
  T e = (T)(1.0 / 6227020800.0);                                   // 1/13!
  e = e * r + (T)(1.0 / 479001600.0); e = e * r + (T)(1.0 / 39916800.0); e = e * r + (T)(1.0 / 3628800.0);
  e = e * r + (T)(1.0 / 362880.0);    e = e * r + (T)(1.0 / 40320.0);    e = e * r + (T)(1.0 / 5040.0);
  e = e * r + (T)(1.0 / 720.0);       e = e * r + (T)(1.0 / 120.0);      e = e * r + (T)(1.0 / 24.0);
  e = e * r + (T)(1.0 / 6.0);         e = e * r + (T)0.5;                e = e * r + (T)1;
  e = e * r + (T)1;
  // 2^k for integer-valued kf in [-14, 14], from its binary digits; only float compares and selects (vectorises)
  T t = kf < 0 ? -kf : kf, s = 1;
  s = t >= 8 ? s * 256 : s; t = t >= 8 ? t - 8 : t;
  s = t >= 4 ? s * 16 : s;  t = t >= 4 ? t - 4 : t;
  s = t >= 2 ? s * 4 : s;   t = t >= 2 ? t - 2 : t;
  s = t >= 1 ? s * 2 : s;
  s = kf < 0 ? (T)1 / s : s;                                        // exact: s is a power of two
  T a = (T)1 / (1 + e * s);
  a = a < (T)1e-4 ? (T)1e-4 : (a > (T)(1 - 1e-4) ? (T)(1 - 1e-4) : a);
  return x > (T)9.3 ? (T)1e-4 : (x < (T)-9.3 ? (T)(1 - 1e-4) : a);
}
