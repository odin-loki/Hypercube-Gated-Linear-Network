/* SPDX-License-Identifier: AGPL-3.0-or-later
   mock_hgln.c - a CPU stand-in for libcuda.so.1 (and, built with -DMOCK_HIP, libamdhip64.so.6), used only by HGLN's
   tests. It exports the driver entry points FORGE's gpu::cuda_hip runtime dlsym's, keeps "device" memory in host
   memory, and runs a launch of hgln_run by executing the real kernel source on the CPU (cuda/hgln_emu_launch.cpp:
   blockDim threads per block with a barrier, dynamic shared memory pre-filled with NaN bytes), so the whole path
   Net -> hgln::gpu::Bank -> FORGE runtime -> driver -> kernel -> results is exercised without a GPU. Modelled on
   FORGE's tests/mock_cuda.c. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).

   What it checks like a driver would: the image's target (PTX ".target sm_NN" / the AMDGPU ELF machine) against the
   mock device, that function names exist in the module, the block size, and dynamic shared memory against the
   48 KB default / the cuFuncSetAttribute opt-in (CUDA) or 64 KB (HIP). FORGE's own images load (FORGE's runtime
   loads them at open); launching one of FORGE's kernels fails, since HGLN never does.
   It does NOT execute the PTX / code object: the kernel source is compiled for the CPU.

   Environment: MOCK_DEVICES (device count, default 1); MOCK_ARCH (CUDA: sm_70 or sm_80, default sm_80 - PTX for a
   newer sm is rejected; HIP: gfx90a, gfx942 or gfx1100, default gfx90a - other code objects are rejected);
   MOCK_SMEM_OPTIN (CUDA opt-in shared memory per block in bytes, default 101376 as on sm_86); MOCK_TRACE (log calls). */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MOCK_HIP
#define N_(cu, hip) hip
#define EXPORT_HIP 1
#else
#define N_(cu, hip) cu
#define EXPORT_HIP 0
#endif
#if defined(_WIN32)
#define MOCK_API __declspec(dllexport)
#else
#define MOCK_API __attribute__((visibility("default")))
#endif
typedef uint64_t dptr;
static void* P(dptr p) { return (void*)(uintptr_t)p; }
static int trace(void) { return getenv("MOCK_TRACE") != NULL; }

int hgln_emu_launch_f64(unsigned grid, unsigned block, unsigned smem, void** args);
int hgln_emu_launch_f32(unsigned grid, unsigned block, unsigned smem, void** args);

/* ---- module images */
typedef struct { int hgln, real; char arch[16]; } module;           /* real: 8 float64, 4 float32, 0 not HGLN */
typedef struct { char name[64]; module* mod; int max_dyn; } function;
static const char* mock_arch(void) { const char* e = getenv("MOCK_ARCH"); return e && *e ? e : N_("sm_80", "gfx90a"); }
static const void* find(const void* hay, size_t n, const char* needle) { return memmem(hay, n, needle, strlen(needle)); }
#ifndef MOCK_HIP
/* PTX: NUL-terminated text with ".target sm_NN"; a driver JIT-compiles it for any device of that sm or newer */
static int image_size_arch(const void* img, size_t* n, char* arch) {
    const char* t = (const char*)img; *n = strlen(t);
    const char* tg = strstr(t, ".target sm_"); if (!tg) return 218; /* CUDA_ERROR_INVALID_PTX */
    snprintf(arch, 16, "sm_%d", atoi(tg + 11));
    if (atoi(tg + 11) > atoi(mock_arch() + 3)) return 218;          /* PTX for a newer GPU */
    return 0;
}
#else
/* AMDGPU code object: an ELF whose e_flags hold the machine (EF_AMDGPU_MACH); it only loads on that gfx */
static int image_size_arch(const void* img, size_t* n, char* arch) {
    const unsigned char* e = (const unsigned char*)img; uint64_t shoff; uint32_t fl; uint16_t shentsize, shnum;
    if (memcmp(e, "\177ELF", 4) != 0) return 200;                   /* hipErrorInvalidImage */
    memcpy(&shoff, e + 40, 8); memcpy(&fl, e + 48, 4); memcpy(&shentsize, e + 58, 2); memcpy(&shnum, e + 60, 2);
    *n = (size_t)(shoff + (uint64_t)shentsize * shnum);
    switch (fl & 0xffu) { case 0x3f: strcpy(arch, "gfx90a"); break; case 0x4c: strcpy(arch, "gfx942"); break;
                          case 0x41: strcpy(arch, "gfx1100"); break; default: snprintf(arch, 16, "mach%#x", fl & 0xffu); }
    return strcmp(arch, mock_arch()) ? 209 /* hipErrorNoBinaryForGpu */ : 0;
}
#endif
MOCK_API int N_(cuModuleLoadData, hipModuleLoadData)(void** m, const void* img) {
    size_t n = 0; module mm; memset(&mm, 0, sizeof mm);
    int r = image_size_arch(img, &n, mm.arch);
    if (trace()) fprintf(stderr, "mock: module load %s (%zu bytes) -> %d\n", mm.arch, n, r);
    if (r) return r;
    mm.hgln = find(img, n, "hgln_run") != NULL;
    mm.real = find(img, n, "hgln_real_f64") ? 8 : find(img, n, "hgln_real_f32") ? 4 : 0;
    if (mm.hgln && !mm.real) return 218;
    module* p = malloc(sizeof mm); *p = mm; *m = p; return 0;
}
MOCK_API int N_(cuModuleUnload, hipModuleUnload)(void* m) { free(m); return 0; }
MOCK_API int N_(cuModuleGetFunction, hipModuleGetFunction)(void** f, void* m, const char* name) {
    module* mod = (module*)m;
    if (mod->hgln && strcmp(name, "hgln_run") && strcmp(name, mod->real == 8 ? "hgln_real_f64" : "hgln_real_f32")) return 500;   /* NOT_FOUND */
    function* fn = calloc(1, sizeof(function)); snprintf(fn->name, sizeof fn->name, "%s", name); fn->mod = mod; fn->max_dyn = 48 << 10;
    *f = fn; return 0;   /* (leaked, like the handles of a real driver until the module is unloaded) */
}

/* ---- devices, contexts */
static int g_cur = 0;
MOCK_API int N_(cuInit, hipInit)(unsigned f) { (void)f; return 0; }
MOCK_API int N_(cuDeviceGetCount, hipGetDeviceCount)(int* n) { const char* e = getenv("MOCK_DEVICES"); *n = e ? atoi(e) : 1; return 0; }
MOCK_API int N_(cuDeviceGetName, hipDeviceGetName)(char* b, int n, int d) { snprintf(b, (size_t)n, "hgln mock %s %d (%s)", N_("cuda", "hip"), d, mock_arch()); return 0; }
static int smem_optin(void) { const char* e = getenv("MOCK_SMEM_OPTIN"); return e ? atoi(e) : 101376; }
#ifndef MOCK_HIP
MOCK_API int cuDeviceGet(int* d, int o) { *d = o; return 0; }
MOCK_API int cuDeviceGetAttribute(int* v, int attr, int d) {
    (void)d;
    switch (attr) { case 8: *v = 48 << 10; return 0;            /* MAX_SHARED_MEMORY_PER_BLOCK */
                    case 16: *v = 4; return 0;                  /* MULTIPROCESSOR_COUNT */
                    case 97: *v = smem_optin(); return 0;       /* MAX_SHARED_MEMORY_PER_BLOCK_OPTIN */
                    default: return 1; }
}
MOCK_API int cuCtxCreate_v2(void** c, unsigned f, int d) { (void)f; *c = (void*)(uintptr_t)(d + 1); return 0; }
MOCK_API int cuCtxDestroy_v2(void* c) { (void)c; return 0; }
MOCK_API int cuCtxSetCurrent(void* c) { g_cur = (int)(uintptr_t)c - 1; return 0; }
MOCK_API int cuFuncSetAttribute(void* f, int attr, int v) {
    function* fn = (function*)f;
    if (attr != 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */ || v < 0 || v > smem_optin()) return 1;   /* INVALID_VALUE */
    fn->max_dyn = v; if (trace()) fprintf(stderr, "mock: %s max dynamic shared memory %d\n", fn->name, v);
    return 0;
}
#else
MOCK_API int hipSetDevice(int d) { g_cur = d; return 0; }
#endif

/* ---- memory, streams (everything runs synchronously on the host) */
MOCK_API int N_(cuMemAlloc_v2, hipMalloc)(dptr* p, size_t n) { void* b = malloc(n ? n : 1); if (!b) return 2; *p = (dptr)(uintptr_t)b; return 0; }
MOCK_API int N_(cuMemFree_v2, hipFree)(dptr p) { free(P(p)); return 0; }
MOCK_API int N_(cuMemcpyHtoD_v2, hipMemcpyHtoD)(dptr d, const void* s, size_t n) { memcpy(P(d), s, n); return 0; }
MOCK_API int N_(cuMemcpyDtoH_v2, hipMemcpyDtoH)(void* d, dptr s, size_t n) { memcpy(d, P(s), n); return 0; }
MOCK_API int N_(cuMemcpyHtoDAsync_v2, hipMemcpyHtoDAsync)(dptr d, const void* s, size_t n, void* st) { (void)st; memcpy(P(d), s, n); return 0; }
MOCK_API int N_(cuMemcpyDtoHAsync_v2, hipMemcpyDtoHAsync)(void* d, dptr s, size_t n, void* st) { (void)st; memcpy(d, P(s), n); return 0; }
MOCK_API int N_(cuMemcpyDtoDAsync_v2, hipMemcpyDtoDAsync)(dptr d, dptr s, size_t n, void* st) { (void)st; memmove(P(d), P(s), n); return 0; }
MOCK_API int N_(cuMemsetD32Async, hipMemsetD32Async)(dptr d, unsigned v, size_t n, void* st) {
    (void)st; uint32_t* p = (uint32_t*)P(d); for (size_t i = 0; i < n; ++i) p[i] = v; return 0; }
MOCK_API int N_(cuMemGetInfo_v2, hipMemGetInfo)(size_t* fr, size_t* tot) { *tot = (size_t)1 << 32; *fr = *tot / 2; return 0; }
MOCK_API int N_(cuCtxSynchronize, hipDeviceSynchronize)(void) { return 0; }
MOCK_API int N_(cuStreamCreate, hipStreamCreateWithFlags)(void** s, unsigned f) { (void)f; *s = malloc(1); return 0; }
MOCK_API int N_(cuStreamDestroy_v2, hipStreamDestroy)(void* s) { free(s); return 0; }
MOCK_API int N_(cuStreamSynchronize, hipStreamSynchronize)(void* s) { (void)s; return 0; }

/* ---- launches */
static struct { unsigned launches, grid, block, smem, real; char arch[16]; } g_last;
/* For tests (dlopen the same library and dlsym this): the last hgln_run launch as the driver saw it. */
MOCK_API void mock_hgln_last_launch(unsigned* launches, unsigned* grid, unsigned* block, unsigned* smem, unsigned* real, char arch[16]) {
    *launches = g_last.launches; *grid = g_last.grid; *block = g_last.block; *smem = g_last.smem; *real = g_last.real; memcpy(arch, g_last.arch, 16); }
MOCK_API int N_(cuLaunchKernel, hipModuleLaunchKernel)(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                                                      unsigned sh, void* stream, void** args, void** extra) {
    function* fn = (function*)f; (void)stream;
    const unsigned limit = EXPORT_HIP ? 64u << 10 : (unsigned)fn->max_dyn;
    if (trace()) fprintf(stderr, "mock: launch %s grid %u block %u smem %u (limit %u)\n", fn->name, gx, bx, sh, limit);
    if (!fn->mod->hgln) return 1;                                   /* FORGE's own kernels are not emulated here */
    if (extra || gy != 1 || gz != 1 || by != 1 || bz != 1 || bx * by * bz > 1024 || sh > limit) return 1;
    if (strcmp(fn->name, "hgln_run")) return 0;                     /* the empty marker kernel */
    g_last.launches++; g_last.grid = gx; g_last.block = bx; g_last.smem = sh; g_last.real = (unsigned)fn->mod->real;
    memcpy(g_last.arch, fn->mod->arch, 16);
    return fn->mod->real == 8 ? hgln_emu_launch_f64(gx, bx, sh, args) : hgln_emu_launch_f32(gx, bx, sh, args);
}
