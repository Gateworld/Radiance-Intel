/* SPDX-License-Identifier: Apache-2.0
 * Part of the Intel port of radiance (milestone 3, run 279, owner note 219 B.0: the all-reduce's in-kernel phase
 * timestamps). The desk question first: can a SYCL kernel read the device clock on the B70 through SPIR-V's
 * OpReadClockKHR (SPV_KHR_shader_clock), compiled ahead of time by this image's IGC / ocloc, and what does a tick
 * measure? One work-item reads the clock, spins a fixed integer loop, reads it again; the host prints the delta and,
 * timed against the host clock over a longer loop, the tick's length. The all-reduce's phase stamps (B.0) are this
 * read at the phase boundaries, thread 0 of a block, into a side buffer.
 *   xe_clock_probe [LOOP=1000000]
 */
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* the SYCL headers declare it (sycl/__spirv/spirv_ops_runtime.hpp: `extern uint64_t __spirv_ReadClockKHR(int)`) */
#ifdef __SYCL_DEVICE_ONLY__
#define RI_CLOCK() __spirv_ReadClockKHR(1)   /* scope Device */
#else
#define RI_CLOCK() 0ull
#endif

int main(int argc, char** argv) {
    const uint32_t loop = argc > 1 ? (uint32_t)std::atol(argv[1]) : 1000000u;
    sycl::queue q;
    std::printf("clock_probe: device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    uint64_t* out = sycl::malloc_shared<uint64_t>(4, q);
    for (int pass = 0; pass < 3; ++pass) {
        const uint32_t n = pass == 0 ? 1000u : loop * (pass == 2 ? 4u : 1u);
        const auto h0 = std::chrono::steady_clock::now();
        q.single_task([=]() {
            const uint64_t t0 = RI_CLOCK();
            uint32_t acc = n;
            for (uint32_t i = 0; i < n; ++i) acc = acc * 1664525u + 1013904223u;
            const uint64_t t1 = RI_CLOCK();
            out[0] = t0; out[1] = t1; out[2] = acc;
        }).wait();
        const double host_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count();
        const uint64_t d = out[1] - out[0];
        std::printf("RESULT loop %u: clock %llu -> %llu, delta %llu ticks; host %.1f us for the launch (acc %llu)\n", n,
                    (unsigned long long)out[0], (unsigned long long)out[1], (unsigned long long)d, host_us,
                    (unsigned long long)out[2]);
    }
    sycl::free(out, q);
    return 0;
}
