/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Deadcode and the radiance contributors
 * Part of the Intel port of radiance (owner note 266 item 2): mm::EncoderCache, the encoder-output cache.
 *
 * A hit returns the stored rows byte for byte; two different keys never answer for each other; the
 * same key with another geometry misses; eviction is least recently used and bounded in bytes; an
 * unkeyed item and a disabled cache keep nothing. */
#include "rad_test.h"

#include "mm/encoder_cache.h"

#include <cstdint>
#include <vector>

using rad::mm::EncoderCache;
using rad::mm::Item;

namespace {

/* A picture of grid_h x grid_w patches (merge 2: one row per 4 patches), keyed (hi, lo), rows
 * filled from `seed` so every item's rows differ. */
Item picture(uint64_t hi, uint64_t lo, int32_t gh, int32_t gw, int64_t n_embd, uint16_t seed) {
    Item it;
    it.kind = rad::mm::Kind::Image;
    it.grid_t = 1;
    it.grid_h = gh;
    it.grid_w = gw;
    it.n_patch = (int64_t)gh * gw;
    it.n_seg = 1;
    it.rows_h = gh / 2;
    it.rows_w = gw / 2;
    it.seg_rows = it.rows_h * it.rows_w;
    it.key_hi = hi;
    it.key_lo = lo;
    it.embd.resize((size_t)(it.n_rows() * n_embd));
    for (size_t i = 0; i < it.embd.size(); ++i) it.embd[i] = (uint16_t)(seed + i * 7919u);
    return it;
}

/* The same item as a fresh request would carry it: geometry and key, no rows yet. */
Item asked(const Item& like) {
    Item it = like;
    it.embd.clear();
    return it;
}

}  /* namespace */

TEST(encoder_cache_hit_is_byte_identical) {
    EncoderCache c(64 << 20);
    const Item a = picture(1, 2, 8, 8, 16, 11);
    c.insert(a, 16);
    Item q = asked(a);
    REQUIRE(c.lookup(q, 16));
    CHECK(q.embd == a.embd);
    CHECK_EQ(c.hits(), 1);
    CHECK_EQ(c.items(), 1);
}

TEST(encoder_cache_different_keys_never_collide) {
    EncoderCache c(64 << 20);
    const Item a = picture(1, 2, 8, 8, 16, 11);
    const Item b = picture(1, 3, 8, 8, 16, 99);   /* same geometry, other key, other rows */
    const Item d = picture(4, 2, 8, 8, 16, 55);
    c.insert(a, 16);
    c.insert(b, 16);
    Item qa = asked(a), qb = asked(b), qd = asked(d);
    REQUIRE(c.lookup(qa, 16));
    REQUIRE(c.lookup(qb, 16));
    CHECK(qa.embd == a.embd);
    CHECK(qb.embd == b.embd);
    CHECK(qa.embd != qb.embd);
    CHECK(!c.lookup(qd, 16));   /* a key never stored misses */
    CHECK(qd.embd.empty());
}

TEST(encoder_cache_same_key_other_geometry_misses) {
    EncoderCache c(64 << 20);
    const Item a = picture(7, 7, 8, 8, 16, 1);
    c.insert(a, 16);
    Item wide = asked(picture(7, 7, 8, 12, 16, 1));   /* another grid under the same key */
    CHECK(!c.lookup(wide, 16));
    Item other_embd = asked(a);
    CHECK(!c.lookup(other_embd, 32));                 /* another row width */
    Item video = asked(a);
    video.kind = rad::mm::Kind::Video;
    CHECK(!c.lookup(video, 16));
    Item same = asked(a);
    CHECK(c.lookup(same, 16));
}

TEST(encoder_cache_evicts_least_recently_used_within_its_bytes) {
    /* each item: 16 rows x 16 x 2 bytes = 512 B; room for two */
    EncoderCache c(1100);
    const Item a = picture(1, 1, 8, 8, 16, 1);
    const Item b = picture(2, 2, 8, 8, 16, 2);
    const Item d = picture(3, 3, 8, 8, 16, 3);
    c.insert(a, 16);
    c.insert(b, 16);
    Item qa = asked(a);
    REQUIRE(c.lookup(qa, 16));   /* a is now the most recently used: b goes first */
    c.insert(d, 16);
    CHECK_EQ(c.evictions(), 1);
    CHECK_EQ(c.items(), 2);
    CHECK(c.bytes() <= c.cap());
    Item qb = asked(b), qa2 = asked(a), qd = asked(d);
    CHECK(!c.lookup(qb, 16));
    CHECK(c.lookup(qa2, 16));
    CHECK(c.lookup(qd, 16));
    CHECK(qd.embd == d.embd);
}

TEST(encoder_cache_reinsert_keeps_one_copy) {
    EncoderCache c(64 << 20);
    const Item a = picture(5, 6, 8, 8, 16, 1);
    c.insert(a, 16);
    c.insert(a, 16);
    CHECK_EQ(c.items(), 1);
    CHECK_EQ(c.bytes(), (int64_t)(a.embd.size() * sizeof(uint16_t)));
}

TEST(encoder_cache_too_large_unkeyed_or_off_keeps_nothing) {
    EncoderCache small(100);   /* smaller than one item */
    const Item a = picture(1, 2, 8, 8, 16, 1);
    small.insert(a, 16);
    CHECK_EQ(small.items(), 0);

    EncoderCache c(64 << 20);
    const Item z = picture(0, 0, 8, 8, 16, 1);   /* nobody keyed it */
    c.insert(z, 16);
    Item qz = asked(z);
    CHECK(!c.lookup(qz, 16));
    CHECK_EQ(c.items(), 0);

    EncoderCache off(0);
    CHECK(!off.enabled());
    off.insert(a, 16);
    Item qa = asked(a);
    CHECK(!off.lookup(qa, 16));
}

RAD_TEST_MAIN()
