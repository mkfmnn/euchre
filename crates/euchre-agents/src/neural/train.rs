//! The trainable model and the supervised training loop.
//!
//! A [`NeuralModel`] is the four policy networks bundled together, with a compact
//! on-disk format ([`NeuralModel::save`] / [`NeuralModel::load`]). [`train`]
//! fits one from a pile of [`Sample`]s — `(features, teacher's choice)` pairs
//! labelled by a strong agent. This is **behavioural cloning**: the net learns to
//! reproduce the teacher's decisions, so at play time it needs only a forward
//! pass and no search.
//!
//! The module is deliberately free of any dependency on the engine: it trains on
//! already-collected samples. Collecting those samples (driving real games with a
//! teacher) lives in the `train_neural` example, which is free to use the engine.

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use super::features::Head;
use super::net::{Example, Net, NetTrainer, PolicyExample};

/// One labelled decision: the head it belongs to, the input features, the index
/// of the class the teacher chose, and the mask of classes that were legal.
#[derive(Debug, Clone)]
pub struct Sample {
    pub head: Head,
    pub features: Vec<f32>,
    pub target: usize,
    pub legal: u32,
}

/// The four trained policy networks, indexed by [`Head::index`].
#[derive(Debug, Clone)]
pub struct NeuralModel {
    nets: [Net; 4],
}

/// Magic bytes prefixing a serialized model.
const MAGIC: &[u8; 4] = b"EUNN";
/// Serialization format version.
const VERSION: u32 = 1;

impl NeuralModel {
    /// The network for `head`.
    pub fn net(&self, head: Head) -> &Net {
        &self.nets[head.index()]
    }

    /// Serializes the model: magic, version, then the four nets in [`Head::ALL`]
    /// order.
    pub fn save(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        for head in Head::ALL {
            self.nets[head.index()].write(&mut out);
        }
        out
    }

    /// Reads a model written by [`NeuralModel::save`], returning `None` on a bad
    /// header, version, or any truncation, or if a net's shape does not match the
    /// head it is loaded for.
    pub fn load(bytes: &[u8]) -> Option<NeuralModel> {
        let mut cursor = bytes;
        if cursor.get(..4)? != MAGIC {
            return None;
        }
        cursor = &cursor[4..];
        let version = u32::from_le_bytes(cursor.get(..4)?.try_into().ok()?);
        cursor = &cursor[4..];
        if version != VERSION {
            return None;
        }
        let mut nets: Vec<Net> = Vec::with_capacity(4);
        for head in Head::ALL {
            let net = Net::read(&mut cursor)?;
            if net.input_dim() != head.input_dim() || net.output_dim() != head.output_dim() {
                return None;
            }
            nets.push(net);
        }
        let nets: [Net; 4] = nets.try_into().ok()?;
        Some(NeuralModel { nets })
    }
}

/// The four policy networks plus the Adam optimiser state needed to fine-tune
/// them by reinforcement learning, built from an already-trained [`NeuralModel`].
///
/// Where [`train`] fits the heads from a fixed pile of teacher-labelled samples,
/// `PolicyTrainer` improves them *online* from self-play: the caller samples
/// actions from the current policy ([`PolicyTrainer::net`]), plays them out to
/// collect rewards, and feeds the resulting [`PolicyExample`]s back through
/// [`PolicyTrainer::step`] (REINFORCE). The engine-dependent half — playing the
/// games and turning hand outcomes into advantages — lives in the `train_rl`
/// example, keeping this module free of any dependency on the engine.
pub struct PolicyTrainer {
    trainers: [NetTrainer; 4],
}

impl PolicyTrainer {
    /// Wraps a model's four nets with fresh Adam state for fine-tuning. The
    /// model's weights are the warm start (typically the behaviourally cloned
    /// model); the optimiser moments begin at zero.
    pub fn from_model(model: &NeuralModel) -> Self {
        let trainers = std::array::from_fn(|i| NetTrainer::new(model.nets[i].clone()));
        PolicyTrainer { trainers }
    }

    /// A snapshot of the current four nets as a [`NeuralModel`] (for evaluation,
    /// checkpointing, or saving). Cheap relative to a training step but it does
    /// clone the weights, so snapshot at iteration boundaries rather than per step.
    pub fn model(&self) -> NeuralModel {
        let nets = std::array::from_fn(|i| self.trainers[i].net().clone());
        NeuralModel { nets }
    }

    /// The current network for `head`, used to score logits while sampling
    /// self-play actions.
    pub fn net(&self, head: Head) -> &Net {
        self.trainers[head.index()].net()
    }

    /// Runs one REINFORCE Adam step for `head` over `batch`, returning the mean
    /// policy entropy (a convergence monitor). See
    /// [`NetTrainer::policy_gradient_step`].
    pub fn step(
        &mut self,
        head: Head,
        batch: &[PolicyExample<'_>],
        lr: f32,
        entropy_coef: f32,
        temperature: f32,
    ) -> f32 {
        self.trainers[head.index()].policy_gradient_step(batch, lr, entropy_coef, temperature)
    }
}

/// Hyperparameters for [`train`].
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Width of the single hidden layer in each head's network.
    pub hidden: usize,
    /// Passes over the data per head.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Initial Adam learning rate (decayed on a fixed step schedule).
    pub lr: f32,
    /// Fraction of samples held out to report validation accuracy.
    pub val_fraction: f32,
    /// Seed for weight initialisation and shuffling.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            hidden: 128,
            epochs: 14,
            batch_size: 256,
            lr: 1e-3,
            val_fraction: 0.05,
            seed: 0,
        }
    }
}

/// What [`train`] learned about one head.
#[derive(Debug, Clone, Copy)]
pub struct HeadReport {
    pub head: Head,
    /// Training samples for this head.
    pub train_samples: usize,
    /// Final mean training loss over the last epoch.
    pub train_loss: f32,
    /// Fraction of held-out samples whose arg-max matched the teacher.
    pub val_accuracy: f32,
}

/// Trains a [`NeuralModel`] from `samples`, returning the model and a per-head
/// report. Each head is trained independently on the samples that belong to it.
pub fn train(samples: &[Sample], config: TrainConfig) -> (NeuralModel, Vec<HeadReport>) {
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut nets: Vec<Net> = Vec::with_capacity(4);
    let mut reports = Vec::with_capacity(4);

    for head in Head::ALL {
        let mut idxs: Vec<usize> = (0..samples.len())
            .filter(|&i| samples[i].head == head)
            .collect();
        idxs.shuffle(&mut rng);

        let n_val = ((idxs.len() as f32 * config.val_fraction) as usize).min(idxs.len());
        let (val, tr) = idxs.split_at(n_val);

        let dims = [head.input_dim(), config.hidden, head.output_dim()];
        let mut trainer = NetTrainer::new(Net::new(&dims, &mut rng));

        let mut train_loss = 0.0;
        let mut order: Vec<usize> = tr.to_vec();
        for epoch in 0..config.epochs {
            order.shuffle(&mut rng);
            let lr = config.lr * lr_scale(epoch, config.epochs);
            let mut epoch_loss = 0.0;
            let mut batches = 0;
            for chunk in order.chunks(config.batch_size.max(1)) {
                let batch: Vec<Example<'_>> = chunk
                    .iter()
                    .map(|&i| Example {
                        features: &samples[i].features,
                        target: samples[i].target,
                        legal: samples[i].legal,
                    })
                    .collect();
                epoch_loss += trainer.train_batch(&batch, lr);
                batches += 1;
            }
            train_loss = if batches > 0 {
                epoch_loss / batches as f32
            } else {
                0.0
            };
        }

        let val_accuracy = accuracy(trainer.net(), samples, val);
        reports.push(HeadReport {
            head,
            train_samples: tr.len(),
            train_loss,
            val_accuracy,
        });
        nets.push(trainer.into_net());
    }

    let nets: [Net; 4] = nets.try_into().expect("exactly four heads");
    (NeuralModel { nets }, reports)
}

/// A step learning-rate schedule: full rate, then halved at 60% and again at 85%
/// of training. Gentle annealing tightens the fit late without much tuning.
fn lr_scale(epoch: usize, epochs: usize) -> f32 {
    let frac = epoch as f32 / epochs.max(1) as f32;
    if frac < 0.6 {
        1.0
    } else if frac < 0.85 {
        0.5
    } else {
        0.25
    }
}

/// The fraction of `idxs` whose masked arg-max under `net` equals the teacher's
/// label.
fn accuracy(net: &Net, samples: &[Sample], idxs: &[usize]) -> f32 {
    if idxs.is_empty() {
        return f32::NAN;
    }
    let mut correct = 0;
    for &i in idxs {
        let s = &samples[i];
        let logits = net.forward(&s.features);
        let pred = (0..logits.len())
            .filter(|&k| s.legal & (1 << k) != 0)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .expect("at least one legal class");
        if pred == s.target {
            correct += 1;
        }
    }
    correct as f32 / idxs.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model trained on a tiny, perfectly separable synthetic dataset should
    /// reach high validation accuracy — a smoke test that the whole pipeline
    /// (init, batching, Adam, accuracy) hangs together.
    #[test]
    fn learns_a_separable_synthetic_task() {
        // Two play-head classes keyed by a single feature; everything else zero.
        let mut samples = Vec::new();
        for i in 0..400 {
            let mut features = vec![0.0f32; Head::Play.input_dim()];
            let target = if i % 2 == 0 { 0 } else { 5 };
            features[target] = 1.0;
            samples.push(Sample {
                head: Head::Play,
                features,
                target,
                legal: (1 << 0) | (1 << 5),
            });
        }
        let config = TrainConfig {
            hidden: 16,
            epochs: 10,
            batch_size: 32,
            val_fraction: 0.2,
            seed: 1,
            ..TrainConfig::default()
        };
        let (model, reports) = train(&samples, config);
        let play = reports.iter().find(|r| r.head == Head::Play).unwrap();
        assert!(
            play.val_accuracy > 0.9,
            "synthetic task accuracy too low: {}",
            play.val_accuracy
        );
        // The other heads have no samples, but must still produce valid nets.
        assert_eq!(
            model.net(Head::Upcard).input_dim(),
            Head::Upcard.input_dim()
        );
    }

    #[test]
    fn model_save_load_round_trips() {
        let samples = vec![Sample {
            head: Head::Upcard,
            features: vec![0.0; Head::Upcard.input_dim()],
            target: 0,
            legal: 0b111,
        }];
        let (model, _) = train(&samples, TrainConfig::default());
        let bytes = model.save();
        let restored = NeuralModel::load(&bytes).expect("valid model bytes");
        let input = vec![0.2; Head::Play.input_dim()];
        assert_eq!(
            model.net(Head::Play).forward(&input),
            restored.net(Head::Play).forward(&input)
        );
    }

    #[test]
    fn load_rejects_bad_magic() {
        assert!(NeuralModel::load(b"nope and some trailing bytes").is_none());
    }

    /// A `PolicyTrainer` snapshots back to a model identical to its start, and a
    /// step with a reinforcing advantage raises the chosen action's probability —
    /// the contract the `train_rl` example relies on.
    #[test]
    fn policy_trainer_snapshots_and_improves() {
        let (model, _) = train(
            &[Sample {
                head: Head::Play,
                features: vec![0.0; Head::Play.input_dim()],
                target: 0,
                legal: 0b11,
            }],
            TrainConfig::default(),
        );
        let mut pt = PolicyTrainer::from_model(&model);

        // A fresh snapshot reproduces the warm-start weights exactly.
        let snap = pt.model();
        let input = vec![0.1; Head::Play.input_dim()];
        assert_eq!(
            snap.net(Head::Play).forward(&input),
            model.net(Head::Play).forward(&input)
        );

        let features = vec![0.2; Head::Play.input_dim()];
        let action = 1;
        let legal = 0b11;
        let p0 = super::super::net::masked_softmax(&pt.net(Head::Play).forward(&features), legal)
            [action];
        for _ in 0..20 {
            pt.step(
                Head::Play,
                &[PolicyExample {
                    features: &features,
                    action,
                    legal,
                    advantage: 1.0,
                }],
                0.02,
                0.0,
                1.0,
            );
        }
        let p1 = super::super::net::masked_softmax(&pt.net(Head::Play).forward(&features), legal)
            [action];
        assert!(
            p1 > p0,
            "policy gradient did not reinforce the action: {p0} -> {p1}"
        );
    }
}
