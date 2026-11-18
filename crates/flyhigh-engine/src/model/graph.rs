//! Helpers over the operation graph (`graph.json`): operand names, read counts, the pair / mid boundary.

use serde_json::{Map, Value};
use std::collections::HashMap;

/// The operand names under `key` of an op. A graph that passed [`validate`] has only strings there.
pub fn names(op: &Value, key: &str) -> Vec<String> {
    op[key].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// The weights the FiLM MLP reads by fixed name.
const FILM_WEIGHTS: [&str; 4] = ["film.film.0.weight", "film.film.0.bias", "film.film.2.weight", "film.film.2.bias"];

/// Checks what the builder takes for granted: every op names itself, its operands are strings, a repeat body is a
/// list of ops, and every weight an op reads exists in the manifest; so a model that loads can be built without a
/// panic, whatever directory it came from.
pub(crate) fn validate(ops: &[Value], tensors: &Map<String, Value>) -> anyhow::Result<()> {
    for (i, op) in ops.iter().enumerate() {
        let name = op["op"].as_str().ok_or_else(|| anyhow::anyhow!("graph.json: op {i} has no name"))?;
        for key in ["in", "out", "w"] {
            if let Some(list) = op.get(key) {
                let list = list.as_array().ok_or_else(|| anyhow::anyhow!("graph.json: {name}.{key} is not a list"))?;
                anyhow::ensure!(list.iter().all(Value::is_string), "graph.json: {name}.{key} has a non-string entry");
            }
        }
        for w in names(op, "w") {
            anyhow::ensure!(tensors.contains_key(&w), "graph.json: {name} reads tensor {w}, which the manifest lacks");
        }
        match name {
            "repeat" => {
                let body = op["body"].as_array().ok_or_else(|| anyhow::anyhow!("graph.json: repeat without a body"))?;
                validate(body, tensors)?;
            }
            "film_mlp" => {
                for w in FILM_WEIGHTS {
                    anyhow::ensure!(
                        tensors.contains_key(w),
                        "graph.json: film_mlp needs tensor {w}, which the manifest lacks"
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// count how many ops read each tensor name (recursing into repeat bodies once: peepholes and dead-output detection)
pub(crate) fn count_uses(ops: &[Value], uses: &mut HashMap<String, usize>) {
    for op in ops {
        for n in names(op, "in") {
            *uses.entry(n).or_insert(0) += 1;
        }
        if op["op"].as_str() == Some("repeat") {
            if let Some(b) = op["body"].as_array() {
                count_uses(b, uses);
            }
        }
    }
}
// the same over the UNROLLED plan (repeat bodies K times): how many reads each name still has (buffer liveness)
pub(crate) fn count_uses_unrolled(ops: &[Value], k: u32, uses: &mut HashMap<String, usize>) {
    for op in ops {
        for n in names(op, "in") {
            *uses.entry(n).or_insert(0) += 1;
        }
        if op["op"].as_str() == Some("repeat") {
            if let Some(b) = op["body"].as_array() {
                for _ in 0..repeat_times(op, k) {
                    count_uses_unrolled(b, k, uses);
                }
            }
        }
    }
}
/// How often a repeat op runs its body: its own number, or the session's K where the graph says "K".
pub fn repeat_times(op: &Value, k: u32) -> u32 {
    op["times"].as_u64().map_or(k, |n| n as u32)
}

fn renamed(op: &Value, suffix: &str) -> Value {
    let mut op = op.clone();
    if suffix.is_empty() {
        return op;
    }
    for key in ["in", "out", "carry"] {
        if let Some(list) = op.get_mut(key).and_then(Value::as_array_mut) {
            for v in list {
                if let Some(n) = v.as_str() {
                    *v = Value::String(format!("{n}{suffix}"));
                }
            }
        }
    }
    if let Some(body) = op.get_mut("body").and_then(Value::as_array_mut) {
        for b in body {
            *b = renamed(b, suffix);
        }
    }
    op
}

/// The cascade form of a graph: the pair stage runs whole at the coarsest of `levels` pyramid levels (each half the
/// size of the one before), then at every finer level the encoders run again and the fine block (the graph's repeat
/// body) continues from the flow, visibility and hidden state of the level below, upsampled: `fine_iters[i]` times
/// at the i-th level above the coarsest (the last entry repeats).
/// The coarse level keeps the motion range of a low internal size; the fine levels give the flow the precision of the
/// frame's own resolution, which a low internal size alone lacks (0.4 px of error at 1080p from a 480-line grid).
/// The mid stage is unchanged and reads the finest level under the graph's own names. `None` when the graph has no
/// repeat with carried tensors before its mid stage.
pub fn cascade(ops: &[Value], levels: u32, fine_iters: &[u32]) -> Option<Vec<Value>> {
    use std::collections::HashSet;
    let ms = mid_start(ops);
    let rep = ops[..ms].iter().position(|o| o["op"] == "repeat")?;
    let carry = names(&ops[rep], "carry");
    if levels < 2 || carry.is_empty() || fine_iters.is_empty() {
        return None;
    }
    let produced_before: HashSet<String> = ops[..rep].iter().flat_map(|o| names(o, "out")).collect();
    // what a finer level needs from the ops before the repeat: the closure of the given reads, stopping at the
    // carried tensors (they come from the level below) and at the graph inputs
    let closure = |seed: &[String]| -> Vec<bool> {
        let mut need: HashSet<String> =
            seed.iter().filter(|n| produced_before.contains(*n) && !carry.contains(*n)).cloned().collect();
        let mut keep = vec![false; rep];
        for i in (0..rep).rev() {
            if names(&ops[i], "out").iter().any(|o| need.contains(o)) {
                keep[i] = true;
                need.extend(names(&ops[i], "in").into_iter().filter(|n| !carry.contains(n)));
            }
        }
        keep
    };
    let mut reads = HashMap::new();
    count_uses(ops[rep]["body"].as_array()?, &mut reads);
    let body_reads: Vec<String> = reads.into_keys().collect();
    // the finest level also feeds every op after the repeat whose result is read at all
    let mut used = HashMap::new();
    count_uses(ops, &mut used);
    let mut tail_reads = body_reads.clone();
    for o in &ops[rep + 1..] {
        if o["op"] == "render_pix" || names(o, "out").iter().any(|n| used.contains_key(n)) {
            tail_reads.extend(names(o, "in"));
        }
    }
    let sfx = |l: u32| if l == 0 { String::new() } else { format!("#{l}") };
    let up = |src: String, dst: String| serde_json::json!({"op": "interpolate", "in": [src], "out": [dst], "scale": 2, "mode": "bilinear", "align_corners": false});
    let mut out = Vec::new();
    for l in 1..levels {
        for img in ["A", "B"] {
            out.push(serde_json::json!({"op": "interpolate", "in": [format!("{img}{}", sfx(l - 1))], "out": [format!("{img}{}", sfx(l))],
                "scale": 0.5, "mode": "bilinear", "align_corners": false, "antialias": true}));
        }
    }
    let top = levels - 1;
    out.extend(ops[..=rep].iter().map(|o| renamed(o, &sfx(top))));
    for l in (0..top).rev() {
        let keep = closure(if l == 0 { &tail_reads } else { &body_reads });
        out.extend(ops[..rep].iter().zip(&keep).filter(|(_, k)| **k).map(|(o, _)| renamed(o, &sfx(l))));
        for n in &carry {
            let (below, here) = (format!("{n}{}", sfx(l + 1)), format!("{n}{}", sfx(l)));
            if n == "flow" {
                // flow is in pixels of its own level: twice the grid, twice the value
                out.push(up(below, format!("{here}:up")));
                out.push(
                    serde_json::json!({"op": "scale", "in": [format!("{here}:up")], "out": [here], "factor": 2.0}),
                );
            } else {
                out.push(up(below, here));
            }
        }
        let mut r = renamed(&ops[rep], &sfx(l));
        let step = (top - 1 - l) as usize;
        r["times"] = Value::from(fine_iters[step.min(fine_iters.len() - 1)].max(1));
        out.push(r);
    }
    out.extend(ops[rep + 1..].iter().cloned());
    Some(out)
}

// the first op of the t-dependent (per generated frame) stage: everything from here on is rebuilt per mid in a split plan
pub fn mid_start(ops: &[Value]) -> usize {
    ops.iter()
        .position(|o| o["op"] == "mul_t" || o["op"] == "film_mlp" || names(o, "in").iter().any(|n| n == "t"))
        .unwrap_or(ops.len())
}
