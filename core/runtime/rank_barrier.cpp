/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"
#include "step_poison.h"
#include "../rad_internal.h"
#include "../util/stallwatch.h"

#include <atomic>
#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

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
    /* a 10 s wait says so once (milestone 3, run 18: which thread holds a stalled step) -- counted from when the
     * engine last left idle: a rank waiting while the engine has no step to issue is not stalled (owner note 267) */
    bool told = false;
    const int64_t t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()).count();
    while (!cv_.wait_for(lk, std::chrono::seconds(10), [&] { return aborted_ || gen_ != my_gen; })) {
        if (told) continue;
        const int64_t busy = busy_since_ns_.load(std::memory_order_relaxed);
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (busy < 0 || now - (busy > t0 ? busy : t0) < 10'000'000'000LL) continue;   /* idle, or not 10 s busy yet */
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

/* step_poison.h (owner note 236) */
namespace {
std::atomic<bool> g_step_poisoned{false};
std::atomic<void (*)(void*)> g_poison_fn{nullptr};
std::atomic<void*> g_poison_arg{nullptr};
}
void step_poison_set_hook(void (*fn)(void*), void* arg) {
    g_poison_arg.store(arg, std::memory_order_release);
    g_poison_fn.store(fn, std::memory_order_release);
}
void step_poison() {
    if (g_step_poisoned.exchange(true, std::memory_order_acq_rel)) return;   /* once */
    RAD_ERR("a rank's step failed: every rank leaves the step and the engine stops (owner note 236)");
    if (void (*fn)(void*) = g_poison_fn.load(std::memory_order_acquire)) fn(g_poison_arg.load(std::memory_order_acquire));
    const char* e = std::getenv("RADIANCE_STEP_POISON_EXIT_S");
    /* 300 s: past the all-reduce's first give-up (~2 min) and the queued launches' short ones, so the normal stop
     * (the device idle, then the teardown) finishes first; the watchdog is for a wait nothing else will end */
    const long s = e ? std::strtol(e, nullptr, 10) : 300;
    if (s <= 0) return;
    std::thread([s] {
        std::this_thread::sleep_for(std::chrono::seconds(s));
        RAD_ERR("the engine is still alive %ld s after a rank's step failed (a thread in a device wait the failed "
                "rank's work would have ended): exit 1 now (RADIANCE_STEP_POISON_EXIT_S; owner note 236)", s);
        std::fflush(nullptr);
        std::_Exit(1);
    }).detach();
}
bool step_poisoned() { return g_step_poisoned.load(std::memory_order_acquire); }

}  /* namespace rad */
