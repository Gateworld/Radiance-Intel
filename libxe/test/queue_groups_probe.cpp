/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (Level Zero backend + SYCL kernel library, milestone 3).
 *
 * queue_groups_probe.cpp -- owner note 137 step 1 (run 111): what each card's Level Zero driver offers for a second
 * copy path. Prints every device's command-queue groups (ordinal, flags COMPUTE / COPY / COOPERATIVE / METRICS,
 * numQueues, the fill-pattern size) and, per group, whether an immediate list can be made on each queue index at
 * low / normal priority. Nothing is copied; no model, no compute. A card probe.
 *
 *   xe_queue_groups_probe
 */
#include <level_zero/ze_api.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::string flags_of(ze_command_queue_group_property_flags_t f) {
    std::string s;
    if (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) s += "COMPUTE ";
    if (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) s += "COPY ";
    if (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS) s += "COOPERATIVE ";
    if (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS) s += "METRICS ";
    return s.empty() ? "(none)" : s;
}

}  // namespace

int main() {
    if (zeInit(0) != ZE_RESULT_SUCCESS) { std::printf("zeInit failed\n"); return 2; }
    uint32_t nd = 0;
    zeDriverGet(&nd, nullptr);
    std::vector<ze_driver_handle_t> drivers(nd);
    zeDriverGet(&nd, drivers.data());
    for (uint32_t di = 0; di < nd; ++di) {
        ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        ze_context_handle_t ctx = nullptr;
        if (zeContextCreate(drivers[di], &cd, &ctx) != ZE_RESULT_SUCCESS) { std::printf("driver %u: no context\n", di); continue; }
        uint32_t ndev = 0;
        zeDeviceGet(drivers[di], &ndev, nullptr);
        std::vector<ze_device_handle_t> devs(ndev);
        zeDeviceGet(drivers[di], &ndev, devs.data());
        for (uint32_t d = 0; d < ndev; ++d) {
            ze_device_properties_t dp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
            zeDeviceGetProperties(devs[d], &dp);
            ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES, nullptr};
            const bool havepci = zeDevicePciGetPropertiesExt(devs[d], &pci) == ZE_RESULT_SUCCESS;
            std::printf("driver %u device %u: %s%s", di, d, dp.name, havepci ? "" : "\n");
            if (havepci)
                std::printf(" (pci %04x:%02x:%02x.%x)\n", pci.address.domain, pci.address.bus, pci.address.device,
                            pci.address.function);
            uint32_t ng = 0;
            zeDeviceGetCommandQueueGroupProperties(devs[d], &ng, nullptr);
            std::vector<ze_command_queue_group_properties_t> gp(ng);
            for (auto& g : gp) g = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr};
            zeDeviceGetCommandQueueGroupProperties(devs[d], &ng, gp.data());
            for (uint32_t o = 0; o < ng; ++o) {
                std::printf("  group ordinal %u: flags %s numQueues %u maxMemoryFillPatternSize %zu\n", o,
                            flags_of(gp[o].flags).c_str(), gp[o].numQueues, (size_t)gp[o].maxMemoryFillPatternSize);
                for (uint32_t qi = 0; qi < gp[o].numQueues && qi < 8; ++qi) {
                    std::string res;
                    const ze_command_queue_priority_t pr[2] = { ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW,
                                                                ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
                    for (int pi = 0; pi < 2; ++pi) {
                        ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, o, qi, 0,
                                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, pr[pi]};
                        ze_command_list_handle_t l = nullptr;
                        const ze_result_t r = zeCommandListCreateImmediate(ctx, devs[d], &qd, &l);
                        res += std::string(pi ? " normal " : " low ") + (r == ZE_RESULT_SUCCESS ? "ok" : "refused");
                        if (l) zeCommandListDestroy(l);
                    }
                    std::printf("    queue index %u: immediate list at priority%s\n", qi, res.c_str());
                }
            }
        }
        zeContextDestroy(ctx);
    }
    return 0;
}
