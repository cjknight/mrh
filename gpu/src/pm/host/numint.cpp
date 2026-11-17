/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device_numint.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

// c[k] is column-major (ngrid x nocc): c[k][o*ngrid + g], component stride ngrid*nocc

void DeviceNumint::numint_contract_rho(double * rho, const double * c, int ngrid, int nocc, int nrow)
{
  const size_t stride = (size_t) ngrid * nocc;

#pragma omp parallel for schedule(static)
  for(int g=0; g<ngrid; ++g) {
    const double * c0 = c + g;

    double r0 = 0.0;
    for(int o=0; o<nocc; ++o) r0 += c0[(size_t) o * ngrid] * c0[(size_t) o * ngrid];
    rho[g] = r0;

    if(nrow < 4) continue;

    double tau = 0.0;
    for(int d=1; d<4; ++d) {
      const double * cd = c0 + d * stride;
      double rd = 0.0, td = 0.0;
      for(int o=0; o<nocc; ++o) {
	const double a = c0[(size_t) o * ngrid];
	const double e = cd[(size_t) o * ngrid];
	rd += a * e;
	td += e * e;
      }
      rho[(size_t) d * ngrid + g] = 2.0 * rd;
      tau += td;
    }

    if(nrow == 5) rho[(size_t) 4 * ngrid + g] = 0.5 * tau;
  }
}

/* ---------------------------------------------------------------------- */

// ao[k] is column-major (ngrid x nao): ao[k][i*ngrid + g]; out has the same layout

void DeviceNumint::numint_scale_ao(double * out, const double * ao, const double * wv, int ngrid, int nao, int ncomp)
{
  const size_t stride = (size_t) ngrid * nao; // between AO components

#pragma omp parallel for schedule(static)
  for(int i=0; i<nao; ++i) {
    double * o = out + (size_t) i * ngrid;
    const double * a = ao + (size_t) i * ngrid;

    for(int g=0; g<ngrid; ++g) o[g] = a[g] * wv[g];

    for(int k=1; k<ncomp; ++k) {
      const double * ak = a + k * stride;
      const double * wk = wv + (size_t) k * ngrid;
      for(int g=0; g<ngrid; ++g) o[g] += ak[g] * wk[g];
    }
  }
}

#endif
