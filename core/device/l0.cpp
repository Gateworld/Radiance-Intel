/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 1).
 *
 * l0.cpp -- the Level Zero device backend.
 *
 * Compiled only in the Level Zero build (RAD_HAVE_L0, cmake/RadL0.cmake); elsewhere this file is
 * empty. Skeleton: discovery only, no device is offered yet, so the host backend runs.
 */
#if defined(RAD_HAVE_L0) && RAD_HAVE_L0

#include "device.h"
#include "../rad_internal.h"

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <cstring>

namespace rad {

const DeviceBackend* l0_backend_if_present(void) {
    const char* env = std::getenv("RADIANCE_L0");
    if (env && std::strcmp(env, "0") == 0) return nullptr;
    int n = 0;
    try {
        for (const auto& p : sycl::platform::get_platforms())
            if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
                n += (int)p.get_devices(sycl::info::device_type::gpu).size();
    } catch (const sycl::exception& e) {
        RAD_INFO("device: Level Zero discovery failed: %s", e.what());
        return nullptr;
    }
    RAD_INFO("device: Level Zero sees %d GPU(s); backend not implemented yet", n);
    return nullptr;
}

}  /* namespace rad */

#endif /* RAD_HAVE_L0 */
