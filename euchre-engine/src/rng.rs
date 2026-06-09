//! A tiny, self-contained pseudo-random number generator.
//!
//! The engine needs to shuffle the deck, but pulling in an external RNG crate
//! would add a dependency for a job a few lines of arithmetic can do. This is a
//! [SplitMix64] generator: fast, statistically fine for dealing cards, and —
//! crucially — *seedable*, so a match can be replayed bit-for-bit by reusing a
//! seed. It is **not** cryptographically secure and must not be used where that
//! matters.
//!
//! [SplitMix64]: https://prng.di.unimi.it/splitmix64.c

/// A seedable SplitMix64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from an explicit 64-bit seed.
    ///
    /// Two generators created with the same seed produce the same sequence,
    /// which makes a whole match reproducible.
    pub const fn seed_from_u64(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// Creates a generator seeded from the system clock.
    ///
    /// Use this when you want a different game every time and do not care about
    /// reproducibility. Falls back to a fixed seed if the clock is unavailable.
    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        // Mix in the address of a stack local for a little extra variation
        // between near-simultaneous calls.
        let local = 0u8;
        let addr = (&local as *const u8) as u64;
        Rng::seed_from_u64(nanos ^ addr.rotate_left(17))
    }

    /// Returns the next 64-bit value and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniformly distributed value in `0..bound`.
    ///
    /// Uses Lemire's multiply-and-shift method, which avoids modulo bias.
    /// Panics if `bound` is zero.
    pub fn below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "bound must be positive");
        // Multiply a 32-bit random by the bound and take the high 32 bits.
        ((self.next_u64() >> 32).wrapping_mul(bound as u64) >> 32) as u32
    }

    /// Shuffles a slice in place using the Fisher–Yates algorithm.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.below(i as u32 + 1) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::seed_from_u64(42);
        let mut b = Rng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::seed_from_u64(1);
        let mut b = Rng::seed_from_u64(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Rng::seed_from_u64(7);
        for _ in 0..1000 {
            assert!(rng.below(5) < 5);
        }
    }

    #[test]
    fn shuffle_preserves_multiset() {
        let mut rng = Rng::seed_from_u64(99);
        let mut v: Vec<u32> = (0..24).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..24).collect::<Vec<_>>());
        // With overwhelming probability a 24-element shuffle reorders things.
        assert_ne!(v, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn shuffle_is_reproducible() {
        let mut a = Rng::seed_from_u64(2024);
        let mut b = Rng::seed_from_u64(2024);
        let mut va: Vec<u32> = (0..24).collect();
        let mut vb = va.clone();
        a.shuffle(&mut va);
        b.shuffle(&mut vb);
        assert_eq!(va, vb);
    }
}
