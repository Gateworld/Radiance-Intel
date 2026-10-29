/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"
#include "../rad_internal.h"
#include "../util/stallwatch.h"

#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace rad {

namespace {
/* Run 87 (owner note 82: the first-step hang with boost off): with the barrier's 10 s warning, every thread of
 * the process that is not asleep in a futex -- so not one of the waiters -- with its state, the call it is in
 * and its stack (stallwatch's describer), up to 16: the rank thread that did not arrive, and where it is. */
void describe_busy_threads() {
    const pid_t me = (pid_t)syscall(SYS_gettid);
    DIR* d = opendir("/proc/self/task");
    if (!d) return;
    int n = 0;
    while (dirent* e = readdir(d)) {
        const pid_t t = (pid_t)std::atoi(e->d_name);
        if (t <= 0 || t == me) continue;
        std::ifstream f("/proc/self/task/" + std::to_string(t) + "/syscall");
        std::string sc;
        std::getline(f, sc);
        long nr = -2;
        std::sscanf(sc.c_str(), "%ld", &nr);
        if (nr == SYS_futex) continue;   /* the barrier's waiters and idle pool threads */
        if (n++ >= 16) break;
        RAD_WARN("rank barrier: not at a futex: %s", stall_describe_thread(t, true).c_str());
    }
    closedir(d);
}
}  /* namespace */

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
        /* once a process: the other waiter would print the same threads */
        static std::atomic<bool> described{false};
        if (!described.exchange(true)) {
            lk.unlock();
            describe_busy_threads();
            lk.lock();
        }
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
