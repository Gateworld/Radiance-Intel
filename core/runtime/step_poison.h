/* step_poison.h -- one rank's step has failed (Intel port, run 294-295, owner note 236).
 *
 * Set by the thread that sees a rank's step fail: the rank thread itself, or the Level Zero layer when a
 * held tape play fails inside a library launch's hook (which cannot return an error to the step). The
 * first call runs the hook the engine registered (it marks the step failed and aborts the step barrier,
 * so every rank thread leaves and the scheduler's step loop stops the engine with an error for the
 * requests), and starts a watchdog: if the process is still alive RADIANCE_STEP_POISON_EXIT_S seconds
 * later (default 300; 0 = no watchdog) -- a thread stuck in a device wait that only the failed rank's
 * work would end -- it says so and ends the process (exit code 1). Read by the append retry, whose
 * drain then ends at once, and by the rank loops. Process-wide and never cleared: a failed step stops
 * the engine (spec §17); the flag makes every thread agree on it within a bound. Defined in
 * rank_barrier.cpp. */
#pragma once

namespace rad {
void step_poison();
bool step_poisoned();
/* The engine's reaction to the first poison (called once, from the poisoning thread). */
void step_poison_set_hook(void (*fn)(void*), void* arg);
}  /* namespace rad */
