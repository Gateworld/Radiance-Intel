/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"
#include "../rad_internal.h"

#include <chrono>

namespace rad {

void RankBarrier::arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (aborted_) return;

    const uint64_t my_gen = gen_;
    if (++waiting_ == n_) {
        /* Last one in resets the counter and flips the generation, so a rank that re-enters at
         * once waits on the new generation instead of racing through. */
        waiting_ = 0;
        ++gen_;
        lk.unlock();
        cv_.notify_all();
        return;
    }
    /* a 10 s wait says so once (milestone 3, run 18: which thread holds a stalled step) */
    bool told = false;
    while (!cv_.wait_for(lk, std::chrono::seconds(10), [&] { return aborted_ || gen_ != my_gen; })) {
        if (told) continue;
        told = true;
        RAD_WARN("rank barrier: a thread has waited 10 s at the step barrier (generation %llu, %d of %d arrived)",
                 (unsigned long long)my_gen, waiting_, n_);
    }
}

void RankBarrier::abort() {
    {
        std::lock_guard<std::mutex> lk(m_);
        aborted_ = true;
        waiting_ = 0;
        ++gen_;
    }
    cv_.notify_all();
}

}  /* namespace rad */
