/* main.cpp -- the entry point: turns a command line into engine.h's startup sequence and a status
 * into an exit code.
 */
#include "engine.h"
#include "hf_model.h"
#include "rad_internal.h"
#include "runtime/step_poison.h"
#include "util/stallwatch.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }

    /* Intel port, run 294 (owner note 236): the recorded passes' Level Zero command lists take 1.13 MiB each with
     * NEO's default command buffer (l0.cpp measures it at start-up), ~2.5K of them live at 2048 passes a kept tape
     * -- ~2.8 GiB outside the headroom's accounting, which pushed a new batch shape at headroom 2048 out of device
     * memory (GEN0 / STALL1-5). NEO's debug key OverrideCmdListCmdBufferSizeInKb=64 makes a list 0.25 MiB (run 80's
     * X9; read by the driver only with NEOReadDebugKeys=1), so both are set here, before the first driver call, unless
     * the environment already has either -- then nothing is touched and the user's keys stand (a stray key would go
     * live with the read key; the image and the service set none). RADIANCE_L0_CMDBUF_KB=0 opts out, =N another size.
     * l0.cpp's start-up line says whether the driver took it ("in effect"). */
    {
        const char* kb = std::getenv("RADIANCE_L0_CMDBUF_KB");
        const long n = kb ? std::strtol(kb, nullptr, 10) : 64;
        const char* ovr = std::getenv("OverrideCmdListCmdBufferSizeInKb");
        const char* rdk = std::getenv("NEOReadDebugKeys");
        if (n > 0 && !ovr && !rdk) {
            char v[32];
            std::snprintf(v, sizeof v, "%ld", n);
            setenv("OverrideCmdListCmdBufferSizeInKb", v, 1);
            setenv("NEOReadDebugKeys", "1", 1);
            RAD_INFO("l0: NEO's command buffer a command list set to %ld KiB for this process "
                     "(OverrideCmdListCmdBufferSizeInKb=%ld NEOReadDebugKeys=1; owner note 236; RADIANCE_L0_CMDBUF_KB=0 "
                     "leaves the driver's default)", n, n);
        } else if (ovr || rdk) {
            RAD_INFO("l0: NEO debug keys from the environment left as they are (NEOReadDebugKeys=%s "
                     "OverrideCmdListCmdBufferSizeInKb=%s)", rdk ? rdk : "unset", ovr ? ovr : "unset");
        }
    }

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and --model points at it */

    rad::Engine engine(std::move(cfg));

    /* A startup that stops logging reports where it waits (util/stallwatch.h). Two minutes
     * comfortably covers the slowest normal step, such as a container read from a slow disk. The
     * watch goes on into serve() and ends when the server listens. */
    auto watch = std::make_unique<rad::StallWatch>("startup", 120000);
    /* RADIANCE_STALL_EXIT_S=N (owner note 112, opt-in): a startup quiet for N s inside a driver call
     * exits (code 75) instead of waiting -- a stuck bind killed later, with the loader's copies queued
     * behind it, has reset a card. RADIANCE_STALL_EXIT_ANY=1 counts any wait (the gate's injected
     * one, RADIANCE_DEBUG_STARTUP_WAIT_S: a sleep before start()). */
    if (const char* e = std::getenv("RADIANCE_STALL_EXIT_S"); e && std::atoi(e) > 0) {
        const char* any = std::getenv("RADIANCE_STALL_EXIT_ANY");
        watch->exit_after((uint64_t)std::atoi(e) * 1000, any && any[0] == '1');
    }
    if (const char* e = std::getenv("RADIANCE_DEBUG_STARTUP_WAIT_S"); e && std::atoi(e) > 0) {
        RAD_WARN("startup: debug wait of %d s (RADIANCE_DEBUG_STARTUP_WAIT_S)", std::atoi(e));
        std::this_thread::sleep_for(std::chrono::seconds(std::atoi(e)));
    }
    s = engine.start();
    if (s < 0 || engine.config().debug_graph) watch.reset();
    if (s < 0) {
        /* start() logged the error; the report adds the machine and the decisions made so far. */
        engine.print_startup_report(s);
        return 1;
    }

    if (engine.config().debug_graph) return engine.dump();
    s = engine.serve(std::move(watch));
    if (s < 0) {
        /* serve() can also refuse before listening (e.g. a template default the template
         * rejects, an unholdable thinking budget). */
        if (!engine.startup_report_printed()) engine.print_startup_report(s);
        if (rad::step_poisoned()) {
            /* Owner note 236: a rank's step failed and a rank thread may never leave its device wait; no
             * teardown (the engine's destructor would wait on the streams), the process ends and the driver
             * tears the context down. */
            RAD_ERR("exit 1 without the engine's teardown: a rank's step failed and the device may never go idle "
                    "(owner note 236)");
            std::fflush(nullptr);
            std::_Exit(1);
        }
        return 1;
    }
    return 0;
}
