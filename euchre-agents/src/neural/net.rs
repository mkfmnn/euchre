//! A small, self-contained multilayer perceptron with hand-written
//! backpropagation, Adam, and a compact serialization format.
//!
//! The whole point of keeping this in-tree rather than reaching for a deep
//! learning framework is the same one the [`stats`](../../euchre_eval/stats)
//! module makes: the numerics stay verifiable and the inference path stays
//! dependency-free and deterministic. The networks here are tiny (a couple of
//! hundred inputs, one or two hidden layers, at most 24 outputs), so an explicit
//! forward/backward pass is both fast and easy to audit — and a finite-difference
//! gradient check (see the tests) pins the backward pass to the forward one.
//!
//! A [`Net`] is a stack of fully-connected [`Linear`] layers with ReLU between
//! them and a linear output that produces *logits*. Training treats those logits
//! as a masked softmax over a set of legal classes (see
//! [`masked_softmax_cross_entropy`]); inference just takes the arg-max over the
//! legal classes. Optimisation state (the Adam moments) lives in a separate
//! [`NetTrainer`] so a saved [`Net`] carries only its weights.

use rand::{Rng, RngExt};

/// A fully-connected layer: `out = W·x + b`, with `W` stored row-major
/// (`w[o * in_dim + i]` is the weight from input `i` to output `o`).
#[derive(Debug, Clone)]
struct Linear {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Linear {
    /// A layer initialised with Kaiming-uniform weights (suited to the ReLU
    /// activations) and zero biases.
    fn new<R: Rng + ?Sized>(in_dim: usize, out_dim: usize, rng: &mut R) -> Self {
        // Kaiming/He uniform bound: U(-limit, limit) with limit = sqrt(6 / fan_in)
        // keeps the pre-activation variance roughly stable through ReLU layers.
        let limit = (6.0 / in_dim as f32).sqrt();
        let w = (0..in_dim * out_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * limit)
            .collect();
        Linear {
            in_dim,
            out_dim,
            w,
            b: vec![0.0; out_dim],
        }
    }

    /// Computes `W·x + b` into a fresh vector.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut out = self.b.clone();
        for (out_o, row) in out.iter_mut().zip(self.w.chunks_exact(self.in_dim)) {
            let mut acc = *out_o;
            for (wi, xi) in row.iter().zip(x) {
                acc += wi * xi;
            }
            *out_o = acc;
        }
        out
    }
}

/// A multilayer perceptron: ReLU between hidden layers, a linear output of
/// logits. Construct one with [`Net::new`], evaluate it with [`Net::forward`].
#[derive(Debug, Clone)]
pub struct Net {
    layers: Vec<Linear>,
}

impl Net {
    /// Builds a net whose layer widths are given by `dims` (`dims[0]` is the
    /// input width, `dims[last]` the number of output logits). Requires at least
    /// one layer, i.e. `dims.len() >= 2`.
    pub fn new<R: Rng + ?Sized>(dims: &[usize], rng: &mut R) -> Self {
        assert!(dims.len() >= 2, "a net needs an input and an output width");
        let layers = dims
            .windows(2)
            .map(|w| Linear::new(w[0], w[1], rng))
            .collect();
        Net { layers }
    }

    /// The number of inputs the net expects.
    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }

    /// The number of output logits the net produces.
    pub fn output_dim(&self) -> usize {
        self.layers.last().expect("a net has layers").out_dim
    }

    /// Evaluates the net, returning the output logits. This is the inference
    /// path: no intermediate state is kept.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let mut y = layer.apply(&x);
            if i != last {
                relu(&mut y);
            }
            x = y;
        }
        x
    }

    /// Forward pass that retains each layer's output activations, needed by
    /// [`Net::backward`]. `acts[0]` is the input; `acts[i + 1]` is the output of
    /// layer `i` (post-ReLU for hidden layers, raw logits for the last).
    fn forward_train(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut acts = Vec::with_capacity(self.layers.len() + 1);
        acts.push(input.to_vec());
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let mut y = layer.apply(acts.last().expect("seeded with input"));
            if i != last {
                relu(&mut y);
            }
            acts.push(y);
        }
        acts
    }

    /// Accumulates the gradient of the loss w.r.t. every weight and bias into
    /// `grads`, given the activations from [`Net::forward_train`] and the gradient
    /// of the loss w.r.t. the output logits (`d_logits`).
    ///
    /// Standard reverse-mode: the output layer takes `d_logits` directly (the
    /// logits are linear), and each hidden layer multiplies the incoming gradient
    /// by the ReLU derivative — which is simply whether that unit's stored output
    /// is positive.
    fn backward(&self, acts: &[Vec<f32>], d_logits: &[f32], grads: &mut Grads) {
        let mut delta = d_logits.to_vec();
        for li in (0..self.layers.len()).rev() {
            let layer = &self.layers[li];
            let input = &acts[li];
            let (dw, db) = &mut grads.0[li];
            // Parameter gradients: dW = delta ⊗ input, db = delta.
            for ((&d, db_o), dw_row) in delta
                .iter()
                .zip(db.iter_mut())
                .zip(dw.chunks_exact_mut(layer.in_dim))
            {
                *db_o += d;
                for (g, xi) in dw_row.iter_mut().zip(input) {
                    *g += d * xi;
                }
            }
            if li == 0 {
                break;
            }
            // Propagate to the previous layer's outputs, then through its ReLU.
            let mut prev = vec![0.0f32; layer.in_dim];
            for (&d, row) in delta.iter().zip(layer.w.chunks_exact(layer.in_dim)) {
                for (p, wi) in prev.iter_mut().zip(row) {
                    *p += d * wi;
                }
            }
            let prev_act = &acts[li];
            for (p, a) in prev.iter_mut().zip(prev_act) {
                if *a <= 0.0 {
                    *p = 0.0;
                }
            }
            delta = prev;
        }
    }

    /// A zeroed gradient accumulator shaped like this net's parameters.
    fn zero_grads(&self) -> Grads {
        Grads(
            self.layers
                .iter()
                .map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()]))
                .collect(),
        )
    }

    /// Serialises the net into `out`: per layer, `in_dim` and `out_dim` as
    /// little-endian `u32`s, then the weights and biases as little-endian `f32`s.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            out.extend_from_slice(&(layer.in_dim as u32).to_le_bytes());
            out.extend_from_slice(&(layer.out_dim as u32).to_le_bytes());
            for &x in &layer.w {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &layer.b {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }

    /// Reads a net previously written by [`Net::write`], advancing `cursor` past
    /// the bytes consumed. Returns `None` on any truncation or shape mismatch.
    pub fn read(cursor: &mut &[u8]) -> Option<Net> {
        let n_layers = read_u32(cursor)? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let in_dim = read_u32(cursor)? as usize;
            let out_dim = read_u32(cursor)? as usize;
            let w = read_f32s(cursor, in_dim * out_dim)?;
            let b = read_f32s(cursor, out_dim)?;
            layers.push(Linear {
                in_dim,
                out_dim,
                w,
                b,
            });
        }
        if layers.is_empty() {
            return None;
        }
        Some(Net { layers })
    }
}

/// Per-layer parameter gradients, shaped to match a [`Net`]: one `(dW, db)` pair
/// per layer.
struct Grads(Vec<(Vec<f32>, Vec<f32>)>);

impl Grads {
    /// Scales every accumulated gradient by `scale` (used to average a batch).
    fn scale(&mut self, scale: f32) {
        for (dw, db) in &mut self.0 {
            for g in dw.iter_mut().chain(db.iter_mut()) {
                *g *= scale;
            }
        }
    }
}

/// A [`Net`] plus the Adam optimiser state needed to train it. Kept separate
/// from [`Net`] so a saved model carries only weights, never optimiser moments.
pub struct NetTrainer {
    net: Net,
    /// First and second moment estimates, mirroring each layer's `(w, b)`.
    m: Vec<(Vec<f32>, Vec<f32>)>,
    v: Vec<(Vec<f32>, Vec<f32>)>,
    /// Adam time step.
    t: u64,
}

/// One training example for a single head: the input features, the index of the
/// teacher's chosen class, and the bitmask of which classes were legal.
pub struct Example<'a> {
    pub features: &'a [f32],
    pub target: usize,
    pub legal: u32,
}

/// One reinforcement-learning example for a single head: the input features, the
/// class that was actually *sampled* during self-play, the legal-class mask, and
/// the `advantage` — the (baseline-subtracted) return that scales the
/// policy-gradient update. A positive advantage reinforces the sampled action; a
/// negative one discourages it.
pub struct PolicyExample<'a> {
    pub features: &'a [f32],
    pub action: usize,
    pub legal: u32,
    pub advantage: f32,
}

impl NetTrainer {
    /// Wraps a freshly initialised net with zeroed Adam state.
    pub fn new(net: Net) -> Self {
        let m: Vec<_> = net
            .layers
            .iter()
            .map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()]))
            .collect();
        let v = m.clone();
        NetTrainer { net, m, v, t: 0 }
    }

    /// The net being trained.
    pub fn net(&self) -> &Net {
        &self.net
    }

    /// Consumes the trainer, yielding the trained net.
    pub fn into_net(self) -> Net {
        self.net
    }

    /// Runs one Adam step over a mini-batch, returning the mean batch loss.
    ///
    /// Gradients are summed over the batch then averaged, so the learning rate is
    /// batch-size independent. Examples whose target is illegal are skipped (they
    /// would carry no usable signal); in practice a teacher never produces one.
    pub fn train_batch(&mut self, batch: &[Example<'_>], lr: f32) -> f32 {
        let mut grads = self.net.zero_grads();
        let mut total_loss = 0.0;
        let mut counted = 0usize;
        for ex in batch {
            if ex.legal & (1 << ex.target) == 0 {
                continue;
            }
            let acts = self.net.forward_train(ex.features);
            let logits = acts.last().expect("forward produced logits");
            let (loss, d_logits) = masked_softmax_cross_entropy(logits, ex.legal, ex.target);
            total_loss += loss;
            counted += 1;
            self.net.backward(&acts, &d_logits, &mut grads);
        }
        if counted == 0 {
            return 0.0;
        }
        grads.scale(1.0 / counted as f32);
        self.adam_step(&grads, lr);
        total_loss / counted as f32
    }

    /// Runs one Adam step of the REINFORCE policy gradient over a mini-batch,
    /// returning the mean policy entropy across the batch (a handy convergence
    /// monitor).
    ///
    /// `temperature` is the temperature of the behaviour policy `π_T =
    /// softmax(logits / T)` that the actions were *sampled* from (see
    /// [`sample_masked`]); the score function is taken w.r.t. that same policy, so
    /// training stays on-policy when sampling explores with `T > 1`. For each
    /// example the gradient w.r.t. the logits is
    ///
    /// ```text
    ///   (1/T) · [ advantage · (π_T − onehot(action))  −  entropy_coef · dH/du ]
    /// ```
    ///
    /// i.e. gradient *ascent* on `advantage · ln π_T(action) + entropy_coef ·
    /// H(π_T)`. The first term reuses exactly the `softmax − onehot` gradient that
    /// backs [`masked_softmax_cross_entropy`] (and is gradient-checked there): with
    /// `advantage = 1`, `entropy_coef = 0`, and `temperature = 1` a step is
    /// identical to a supervised [`Self::train_batch`] step toward `action`.
    /// Gradients are summed over the batch and averaged, so the learning rate is
    /// batch-size independent.
    pub fn policy_gradient_step(
        &mut self,
        batch: &[PolicyExample<'_>],
        lr: f32,
        entropy_coef: f32,
        temperature: f32,
    ) -> f32 {
        let t = temperature.max(1e-6);
        let mut grads = self.net.zero_grads();
        let mut total_entropy = 0.0;
        let mut counted = 0usize;
        for ex in batch {
            if ex.legal & (1 << ex.action) == 0 {
                continue;
            }
            let acts = self.net.forward_train(ex.features);
            let logits = acts.last().expect("forward produced logits");
            // The behaviour policy: softmax over temperature-scaled logits.
            let scaled: Vec<f32> = logits.iter().map(|z| z / t).collect();
            let (h, dh) = masked_entropy(&scaled, ex.legal);
            total_entropy += h;
            let probs = masked_softmax(&scaled, ex.legal);
            // (1/T) · [ advantage · (π_T − onehot(action)) − entropy_coef · dH/du ].
            let mut d_logits = vec![0.0f32; logits.len()];
            for ((d, &p), &g) in d_logits.iter_mut().zip(&probs).zip(&dh) {
                *d = (ex.advantage * p - entropy_coef * g) / t;
            }
            d_logits[ex.action] -= ex.advantage / t;
            self.net.backward(&acts, &d_logits, &mut grads);
            counted += 1;
        }
        if counted == 0 {
            return 0.0;
        }
        grads.scale(1.0 / counted as f32);
        self.adam_step(&grads, lr);
        total_entropy / counted as f32
    }

    /// Applies one Adam update using the accumulated `grads`.
    fn adam_step(&mut self, grads: &Grads, lr: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for (li, layer) in self.net.layers.iter_mut().enumerate() {
            let (dw, db) = &grads.0[li];
            adam_update(
                &mut layer.w,
                dw,
                &mut self.m[li].0,
                &mut self.v[li].0,
                lr,
                bc1,
                bc2,
                B1,
                B2,
                EPS,
            );
            adam_update(
                &mut layer.b,
                db,
                &mut self.m[li].1,
                &mut self.v[li].1,
                lr,
                bc1,
                bc2,
                B1,
                B2,
                EPS,
            );
        }
    }
}

/// Applies the per-parameter Adam update to one weight (or bias) vector.
#[allow(clippy::too_many_arguments)]
fn adam_update(
    params: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    bc1: f32,
    bc2: f32,
    b1: f32,
    b2: f32,
    eps: f32,
) {
    for i in 0..params.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
        v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// In-place ReLU.
fn relu(v: &mut [f32]) {
    for x in v {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// The softmax probabilities taken over only the *legal* classes.
///
/// `legal` is a bitmask whose bit `k` marks class `k` as available; the softmax
/// is normalised over those classes alone, so illegal classes always receive
/// probability zero regardless of their logit. The returned vector has the same
/// length as `logits`, with zeros in the illegal positions.
pub fn masked_softmax(logits: &[f32], legal: u32) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for (k, &z) in logits.iter().enumerate() {
        if legal & (1 << k) != 0 && z > max {
            max = z;
        }
    }
    let mut sum = 0.0;
    let mut probs = vec![0.0f32; logits.len()];
    for (k, &z) in logits.iter().enumerate() {
        if legal & (1 << k) != 0 {
            let e = (z - max).exp();
            probs[k] = e;
            sum += e;
        }
    }
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

/// The cross-entropy loss of a [`masked_softmax`] over the legal classes, plus
/// its gradient w.r.t. the logits.
///
/// The returned gradient is the textbook `softmax - onehot(target)` restricted to
/// the legal set (illegal logits receive zero gradient). `target` must be legal.
pub fn masked_softmax_cross_entropy(logits: &[f32], legal: u32, target: usize) -> (f32, Vec<f32>) {
    debug_assert!(legal & (1 << target) != 0, "target class must be legal");
    let probs = masked_softmax(logits, legal);
    let loss = -(probs[target].max(1e-30)).ln();
    let mut grad = probs;
    grad[target] -= 1.0;
    (loss, grad)
}

/// The Shannon entropy `H = -Σ p·ln p` of the [`masked_softmax`] over `logits`,
/// together with its gradient w.r.t. the logits.
///
/// Used as an entropy *bonus* in policy-gradient training: adding `coef·H` to the
/// objective discourages the policy from collapsing prematurely onto a single
/// action while it is still learning. The gradient is `dH/dz_k = -p_k·(ln p_k +
/// H)` on the legal classes (and zero elsewhere); it is pinned to the entropy
/// definition by a finite-difference test.
pub fn masked_entropy(logits: &[f32], legal: u32) -> (f32, Vec<f32>) {
    let probs = masked_softmax(logits, legal);
    let mut h = 0.0f32;
    for (k, &p) in probs.iter().enumerate() {
        if legal & (1 << k) != 0 && p > 0.0 {
            h -= p * p.ln();
        }
    }
    let mut grad = vec![0.0f32; logits.len()];
    for (k, &p) in probs.iter().enumerate() {
        if legal & (1 << k) != 0 && p > 0.0 {
            grad[k] = -p * (p.ln() + h);
        }
    }
    (h, grad)
}

/// Samples a legal class from the [`masked_softmax`] of `logits`, sharpened or
/// softened by `temperature` (1.0 leaves the distribution unchanged; smaller is
/// greedier, larger is flatter). Sampling drives exploration during self-play; the
/// deployed agent instead takes the arg-max.
pub fn sample_masked<R: Rng + ?Sized>(
    logits: &[f32],
    legal: u32,
    temperature: f32,
    rng: &mut R,
) -> usize {
    let t = temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|z| z / t).collect();
    let probs = masked_softmax(&scaled, legal);
    let r = rng.random::<f32>();
    let mut acc = 0.0;
    let mut last = 0;
    for (k, &p) in probs.iter().enumerate() {
        if legal & (1 << k) != 0 {
            last = k;
            acc += p;
            if r <= acc {
                return k;
            }
        }
    }
    // Floating-point rounding can leave `r` just past the accumulated mass; fall
    // back to the last legal class.
    last
}

fn read_u32(cursor: &mut &[u8]) -> Option<u32> {
    let bytes = cursor.get(..4)?;
    let value = u32::from_le_bytes(bytes.try_into().ok()?);
    *cursor = &cursor[4..];
    Some(value)
}

fn read_f32s(cursor: &mut &[u8], n: usize) -> Option<Vec<f32>> {
    let bytes = cursor.get(..n * 4)?;
    let values = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
        .collect();
    *cursor = &cursor[n * 4..];
    Some(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn forward_shapes_match_dims() {
        let mut rng = SmallRng::seed_from_u64(1);
        let net = Net::new(&[5, 8, 3], &mut rng);
        assert_eq!(net.input_dim(), 5);
        assert_eq!(net.output_dim(), 3);
        let out = net.forward(&[0.1, -0.2, 0.3, 0.0, 0.5]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn masked_softmax_ignores_illegal_classes() {
        // Class 1 is illegal; its probability mass must be zero regardless of how
        // large its logit is.
        let logits = [0.0, 100.0, 0.0];
        let legal = 0b101; // classes 0 and 2
        let (_loss, grad) = masked_softmax_cross_entropy(&logits, legal, 0);
        assert_eq!(grad[1], 0.0, "illegal class gets no gradient");
        // The two legal classes are symmetric, so each carries probability 0.5;
        // the gradient is softmax - onehot.
        assert!((grad[0] - (0.5 - 1.0)).abs() < 1e-6);
        assert!((grad[2] - 0.5).abs() < 1e-6);
    }

    /// The backward pass must equal a finite-difference estimate of the forward
    /// pass's gradient. This is the load-bearing correctness test for the whole
    /// module: if backprop is wrong, training silently does nothing useful.
    #[test]
    fn backward_matches_finite_differences() {
        let mut rng = SmallRng::seed_from_u64(7);
        let net = Net::new(&[4, 6, 5], &mut rng);
        let input = [0.5, -0.3, 0.8, -0.1];
        let legal = 0b11011; // classes 0,1,3,4 legal; 2 illegal
        let target = 3;

        // Analytic gradient.
        let acts = net.forward_train(&input);
        let logits = acts.last().unwrap();
        let (_loss, d_logits) = masked_softmax_cross_entropy(logits, legal, target);
        let mut grads = net.zero_grads();
        net.backward(&acts, &d_logits, &mut grads);

        // Numerical gradient of the loss w.r.t. every parameter.
        let eps = 1e-3f32;
        let loss_of = |net: &Net| {
            let logits = net.forward(&input);
            masked_softmax_cross_entropy(&logits, legal, target).0
        };
        for li in 0..net.layers.len() {
            for wi in 0..net.layers[li].w.len() {
                let mut up = net.clone();
                up.layers[li].w[wi] += eps;
                let mut dn = net.clone();
                dn.layers[li].w[wi] -= eps;
                let numeric = (loss_of(&up) - loss_of(&dn)) / (2.0 * eps);
                let analytic = grads.0[li].0[wi];
                assert!(
                    (numeric - analytic).abs() < 1e-2,
                    "weight grad mismatch at layer {li} idx {wi}: numeric {numeric}, analytic {analytic}"
                );
            }
            for bi in 0..net.layers[li].b.len() {
                let mut up = net.clone();
                up.layers[li].b[bi] += eps;
                let mut dn = net.clone();
                dn.layers[li].b[bi] -= eps;
                let numeric = (loss_of(&up) - loss_of(&dn)) / (2.0 * eps);
                let analytic = grads.0[li].1[bi];
                assert!(
                    (numeric - analytic).abs() < 1e-2,
                    "bias grad mismatch at layer {li} idx {bi}: numeric {numeric}, analytic {analytic}"
                );
            }
        }
    }

    /// Adam should drive the loss on a single fixed example toward zero: the net
    /// can trivially memorise one (features, target) pair.
    #[test]
    fn trains_down_on_one_example() {
        let mut rng = SmallRng::seed_from_u64(11);
        let net = Net::new(&[6, 16, 4], &mut rng);
        let mut trainer = NetTrainer::new(net);
        let features = [0.2, 0.4, -0.1, 0.7, -0.5, 0.3];
        let batch = [Example {
            features: &features,
            target: 2,
            legal: 0b1111,
        }];
        let first = trainer.train_batch(&batch, 0.05);
        for _ in 0..200 {
            trainer.train_batch(&batch, 0.05);
        }
        let last = trainer.train_batch(&batch, 0.05);
        assert!(last < first, "loss did not decrease ({first} -> {last})");
        assert!(last < 0.05, "loss did not approach zero: {last}");
        // And the argmax should now be the target class.
        let logits = trainer.net().forward(&features);
        let best = (0..4)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap();
        assert_eq!(best, 2);
    }

    /// The entropy gradient must match a finite-difference estimate of the
    /// entropy itself — the same correctness bar the cross-entropy gradient meets,
    /// applied to the entropy bonus used in policy-gradient training.
    #[test]
    fn entropy_gradient_matches_finite_differences() {
        let logits = [0.4f32, -0.7, 1.1, 0.2, -0.3];
        let legal = 0b11101; // class 1 illegal
        let (_h, grad) = masked_entropy(&logits, legal);
        let eps = 1e-3f32;
        for k in 0..logits.len() {
            let mut up = logits;
            up[k] += eps;
            let mut dn = logits;
            dn[k] -= eps;
            let numeric =
                (masked_entropy(&up, legal).0 - masked_entropy(&dn, legal).0) / (2.0 * eps);
            assert!(
                (numeric - grad[k]).abs() < 1e-2,
                "entropy grad mismatch at {k}: numeric {numeric}, analytic {}",
                grad[k]
            );
            if legal & (1 << k) == 0 {
                assert_eq!(grad[k], 0.0, "illegal class must get zero entropy gradient");
            }
        }
    }

    /// A REINFORCE step with `advantage = 1` and no entropy bonus must reproduce a
    /// supervised step toward the same class exactly — anchoring the new
    /// policy-gradient path to the already gradient-checked cross-entropy path.
    #[test]
    fn policy_gradient_reduces_to_cross_entropy_when_advantage_is_one() {
        let mut rng = SmallRng::seed_from_u64(42);
        let net = Net::new(&[6, 12, 4], &mut rng);
        let features = [0.3, -0.2, 0.5, 0.1, -0.4, 0.6];
        let legal = 0b1111;
        let action = 2;

        let mut supervised = NetTrainer::new(net.clone());
        supervised.train_batch(
            &[Example {
                features: &features,
                target: action,
                legal,
            }],
            0.05,
        );

        let mut policy = NetTrainer::new(net);
        policy.policy_gradient_step(
            &[PolicyExample {
                features: &features,
                action,
                legal,
                advantage: 1.0,
            }],
            0.05,
            0.0,
            1.0,
        );

        let a = supervised.net().forward(&features);
        let b = policy.net().forward(&features);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "outputs diverged: {x} vs {y}");
        }
    }

    /// A positive advantage must raise the sampled action's probability; a
    /// negative one must lower it. This is the load-bearing behaviour of the
    /// policy gradient.
    #[test]
    fn advantage_sign_moves_probability_the_right_way() {
        let mut rng = SmallRng::seed_from_u64(9);
        let net = Net::new(&[6, 12, 4], &mut rng);
        let features = [0.1, 0.2, -0.3, 0.4, -0.5, 0.2];
        let legal = 0b1111;
        let action = 1;
        let prob = |t: &NetTrainer| masked_softmax(&t.net().forward(&features), legal)[action];

        let mut up = NetTrainer::new(net.clone());
        let before = prob(&up);
        for _ in 0..30 {
            up.policy_gradient_step(
                &[PolicyExample {
                    features: &features,
                    action,
                    legal,
                    advantage: 1.0,
                }],
                0.05,
                0.0,
                1.0,
            );
        }
        assert!(
            prob(&up) > before,
            "positive advantage did not raise p(action)"
        );

        let mut down = NetTrainer::new(net);
        let before = prob(&down);
        for _ in 0..30 {
            down.policy_gradient_step(
                &[PolicyExample {
                    features: &features,
                    action,
                    legal,
                    advantage: -1.0,
                }],
                0.05,
                0.0,
                1.0,
            );
        }
        assert!(
            prob(&down) < before,
            "negative advantage did not lower p(action)"
        );
    }

    /// Sampling only ever returns a legal class, and concentrates on the class
    /// whose logit dominates.
    #[test]
    fn sampling_respects_legality_and_logits() {
        let mut rng = SmallRng::seed_from_u64(123);
        let legal = 0b1011; // classes 0, 1, 3
        let logits = [0.0f32, 8.0, 0.0, 0.0]; // class 1 dominates
        let mut hits = 0;
        for _ in 0..2000 {
            let k = sample_masked(&logits, legal, 1.0, &mut rng);
            assert!(legal & (1 << k) != 0, "sampled an illegal class {k}");
            if k == 1 {
                hits += 1;
            }
        }
        assert!(hits > 1900, "dominant class sampled only {hits}/2000 times");
    }

    #[test]
    fn serialization_round_trips() {
        let mut rng = SmallRng::seed_from_u64(3);
        let net = Net::new(&[7, 10, 4], &mut rng);
        let mut bytes = Vec::new();
        net.write(&mut bytes);
        let mut cursor = &bytes[..];
        let restored = Net::read(&mut cursor).expect("valid bytes");
        assert!(cursor.is_empty(), "all bytes consumed");
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        assert_eq!(net.forward(&input), restored.forward(&input));
    }

    #[test]
    fn read_rejects_truncated_input() {
        let mut rng = SmallRng::seed_from_u64(5);
        let net = Net::new(&[3, 3], &mut rng);
        let mut bytes = Vec::new();
        net.write(&mut bytes);
        bytes.truncate(bytes.len() - 1);
        let mut cursor = &bytes[..];
        assert!(Net::read(&mut cursor).is_none());
    }
}
