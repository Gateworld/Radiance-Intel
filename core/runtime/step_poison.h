/* step_poison.h -- one rank's step has failed (Intel port, run 294, owner note 236).
 *
 * Set by the rank thread that fails its step, before it aborts the step barrier; read by the Level
 * Zero layer's append retry, whose drain of the device then ends at once (the peer's half of the
 * step's collectives will never come, so a wait for the device to go idle could be forever), and by
 * the other rank's loop, which leaves instead of running another step. Process-wide and never
 * cleared: a failed step stops the engine (spec §17), the flag only makes every thread agree on it
 * within a bound. Defined in rank_barrier.cpp. */
#pragma once

namespace rad {
void step_poison();
bool step_poisoned();
}  /* namespace rad */
