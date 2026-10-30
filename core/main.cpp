/* main.cpp -- the entry point: turns a command line into engine.h's startup sequence and a status
 * into an exit code.
 */
#include "engine.h"
#include "hf_model.h"
#include "rad_internal.h"
#include "util/stallwatch.h"

#include <chrono>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }

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
        return 1;
    }
    return 0;
}
