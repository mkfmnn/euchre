//! A small, dependency-free PRNG used by the agents that need to make random
//! choices.
//!
//! This is the same SplitMix64 generator the engine uses to shuffle; it is for
//! variety, not cryptographic security. It is reproduced here rather than shared
//! so the agents crate stays decoupled from the engine (the engine keeps its
//! shuffler private, and agents only depend on `euchre-interface`).

use std::sync::atomic::{AtomicU64, Ordering};

/// A SplitMix64 pseudo-random number generator.
///
/// Seed it explicitly with [`Rng::new`] for reproducible behavior, or with
/// [`Rng::from_entropy`] to vary from run to run.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Creates a generator from a fixed `seed`, for reproducible sequences.
    pub fn new(seed: u64) -> Self {
        // Avoid a zero state degenerating the first few outputs.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// Creates a generator seeded from the system clock and a process-wide
    /// counter, so independently constructed agents diverge even when built in
    /// the same instant.
    pub fn from_entropy() -> Self {
        Rng::new(entropy_seed())
    }

    /// Returns the next 64-bit output and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index in `0..n`. Panics if `n` is zero.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "cannot pick an index below zero");
        (self.next_u64() % n as u64) as usize
    }

    /// A uniformly chosen reference into a non-empty slice. Panics if `items` is
    /// empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// A seed derived from the wall clock mixed with a monotonically increasing
/// counter, so two generators created back-to-back still differ.
fn entropy_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9ABC_DEF0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    nanos ^ count.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_seed_is_reproducible() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            assert!(rng.below(5) < 5);
        }
    }

    #[test]
    fn entropy_seeds_differ() {
        let a = Rng::from_entropy();
        let b = Rng::from_entropy();
        // The counter guarantees distinct seeds even within one clock tick.
        assert_ne!(a.0, b.0);
    }
}
