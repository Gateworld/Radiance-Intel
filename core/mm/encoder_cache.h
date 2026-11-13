/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (owner note 266 item 2): an encoder-output cache.
 *
 * mm/encoder_cache.h -- the encoder's rows, kept by the item's content key.
 *
 * An item's key covers its patches and its grid, and the encoder is a function of them alone
 * (mm::Item::key_hi / key_lo), so the rows an earlier pass produced for the same key are the rows a
 * new pass would produce. A prefix-cache hit already skips the encoder, but only up to the last
 * linear-state checkpoint on a hybrid model: a picture that ends between two checkpoints -- the
 * evaluation's six questions about one image -- is encoded again on every request. This cache answers
 * those: run_encodes() consults it before the pass and stores what a pass encoded.
 *
 * Least recently used, bounded in bytes (RADIANCE_MM_ENCODER_CACHE_MIB, default 0 = off). A hit must
 * match the key AND the item's geometry (kind, grids, segments, rows, row width), so an item can never
 * take the rows of a different shape under a colliding key. Host memory only; thread-safe.
 */
#pragma once

#include "mm/processor.h"

#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rad::mm {

class EncoderCache {
public:
    explicit EncoderCache(int64_t cap_bytes) : cap_(cap_bytes > 0 ? cap_bytes : 0) {}

    /* RADIANCE_MM_ENCODER_CACHE_MIB, in bytes; 0 (off) when unset or not a positive number. */
    static int64_t cap_from_env() {
        const char* v = std::getenv("RADIANCE_MM_ENCODER_CACHE_MIB");
        const long long mib = v ? std::atoll(v) : 0;
        return mib > 0 ? (int64_t)mib << 20 : 0;
    }

    bool enabled() const { return cap_ > 0; }

    /* On a hit, `it.embd` holds the cached rows and true is returned. */
    bool lookup(Item& it, int64_t n_embd) {
        if (!enabled() || unkeyed(it)) return false;
        const Shape s = shape_of(it, n_embd);
        std::lock_guard<std::mutex> lk(m_);
        const auto f = map_.find(Key{ it.key_hi, it.key_lo });
        if (f == map_.end() || !(f->second->shape == s)) {
            ++misses_;
            return false;
        }
        lru_.splice(lru_.begin(), lru_, f->second);   /* most recently used first */
        it.embd = f->second->rows;
        ++hits_;
        return true;
    }

    /* Keeps `it`'s rows (encoded by a pass), evicting the least recently used to fit. */
    void insert(const Item& it, int64_t n_embd) {
        if (!enabled() || unkeyed(it) || it.embd.empty()) return;
        const int64_t bytes = (int64_t)(it.embd.size() * sizeof(uint16_t));
        if (bytes > cap_) return;
        const Key k{ it.key_hi, it.key_lo };
        std::lock_guard<std::mutex> lk(m_);
        if (const auto f = map_.find(k); f != map_.end()) {   /* the same key again: keep the newest */
            bytes_ -= f->second->bytes;
            lru_.erase(f->second);
            map_.erase(f);
        }
        while (bytes_ + bytes > cap_ && !lru_.empty()) {
            bytes_ -= lru_.back().bytes;
            map_.erase(lru_.back().key);
            lru_.pop_back();
            ++evictions_;
        }
        lru_.push_front(Entry{ k, shape_of(it, n_embd), it.embd, bytes });
        map_[k] = lru_.begin();
        bytes_ += bytes;
    }

    int64_t bytes() const { std::lock_guard<std::mutex> lk(m_); return bytes_; }
    int64_t items() const { std::lock_guard<std::mutex> lk(m_); return (int64_t)lru_.size(); }
    int64_t hits() const { std::lock_guard<std::mutex> lk(m_); return hits_; }
    int64_t misses() const { std::lock_guard<std::mutex> lk(m_); return misses_; }
    int64_t evictions() const { std::lock_guard<std::mutex> lk(m_); return evictions_; }
    int64_t cap() const { return cap_; }

private:
    /* An item nobody keyed (both halves 0) is never cached: every such item would share one key. */
    static bool unkeyed(const Item& it) { return it.key_hi == 0 && it.key_lo == 0; }
    struct Key {
        uint64_t hi, lo;
        bool operator==(const Key& o) const { return hi == o.hi && lo == o.lo; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const { return (size_t)(k.hi ^ (k.lo * 0x9e3779b97f4a7c15ull)); }
    };
    /* Everything the rows' layout depends on besides the patches themselves. */
    struct Shape {
        int kind;
        int32_t grid_t, grid_h, grid_w, n_seg, seg_rows, rows_w, rows_h;
        int64_t n_patch, n_rows, n_embd, n_segments;
        bool operator==(const Shape& o) const {
            return kind == o.kind && grid_t == o.grid_t && grid_h == o.grid_h && grid_w == o.grid_w &&
                   n_seg == o.n_seg && seg_rows == o.seg_rows && rows_w == o.rows_w && rows_h == o.rows_h &&
                   n_patch == o.n_patch && n_rows == o.n_rows && n_embd == o.n_embd && n_segments == o.n_segments;
        }
    };
    struct Entry {
        Key key;
        Shape shape;
        std::vector<uint16_t> rows;
        int64_t bytes;
    };
    static Shape shape_of(const Item& it, int64_t n_embd) {
        return Shape{ (int)it.kind, it.grid_t, it.grid_h, it.grid_w, it.n_seg, it.seg_rows, it.rows_w,
                      it.rows_h, it.n_patch, it.n_rows(), n_embd, (int64_t)it.segments.size() };
    }

    const int64_t cap_;
    mutable std::mutex m_;
    std::list<Entry> lru_;
    std::unordered_map<Key, std::list<Entry>::iterator, KeyHash> map_;
    int64_t bytes_ = 0, hits_ = 0, misses_ = 0, evictions_ = 0;
};

}  /* namespace rad::mm */
