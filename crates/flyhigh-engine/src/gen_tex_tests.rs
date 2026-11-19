use super::*;
use crate::api::Engine;
use crate::gpu::readback::Readback;
use std::path::Path;

const W: u32 = 256;
const H: u32 = 192;

fn engine() -> Engine {
    Engine::load_with(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"), &crate::EngineOptions::from_env())
        .expect("FlyHigh model and native GPU")
}

fn anime(e: &Engine, w: u32, h: u32) -> GenTex {
    let mut opts = TexOptions::new(w, h);
    opts.mode = TexMode::Video;
    opts.cadence = true;
    opts.protection_mask = true;
    GenTex::new(&e.gpu, &e.blob, &e.ops, opts, None).unwrap()
}

fn cel(draw: u32, pan: u32, cut: bool) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let bx = x + pan;
            let wall = (bx / 17 + y / 23) % 3;
            let ink = if (bx % 17 == 0) || (y % 23 == 0) { 12 } else { 52 + wall as u8 * 23 };
            let cx = 75 + ((draw % 4) * 31) as i32;
            let dx = (x as i32 - cx).abs();
            let px = if cut {
                [242, 229, 215, 255]
            } else if dx < 27 && (58..136).contains(&y) {
                if dx > 23 || (58..62).contains(&y) || (132..136).contains(&y) {
                    [17, 12, 23, 255]
                } else if (80..99).contains(&y) && (10..17).contains(&dx) {
                    [250, 220, 160, 255]
                } else {
                    [196, 57, 78, 255]
                }
            } else {
                [ink, ink + 23, ink + 44, 255]
            };
            rgba.extend_from_slice(&px);
        }
    }
    rgba
}

fn write(gpu: &Gpu, gen: &GenTex, slot: usize, rgba: &[u8]) {
    gpu.queue.write_texture(
        gen.ins[slot].0.as_image_copy(),
        rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn fingerprint(gpu: &Gpu, t: &wgpu::Texture) -> u64 {
    let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(W * H * 4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &stage,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    gpu.queue.submit(Some(enc.finish()));
    Readback::new(
        stage,
        Box::new(|d: &[u8]| {
            d.iter().fold(0xcbf29ce484222325_u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x100000001b3))
        }),
        None,
    )
    .block(gpu)
    .unwrap()
}

#[test]
#[ignore = "requires a GPU"]
fn cadence_pixels_and_decisions() {
    let e = engine();
    let gpu = &e.gpu;
    let mut gen = anime(&e, W, H);
    gen.write_mask(gpu, &(0..W * H).map(|i| u8::from(i % W < 21)).collect::<Vec<_>>()).unwrap();
    for _ in 0..4 {
        gen.add_input(gpu);
    }
    for _ in 0..2 {
        gen.add_output(gpu);
    }
    let mut report = String::new();
    for (hold, pan) in [(2, false), (3, false), (2, true), (3, true)] {
        gen.reset_history();
        let mut retimed = 0;
        for i in 0..22 {
            let slot = i % 4;
            let cut = (13..15).contains(&i);
            if i == 15 {
                gen.reset_history();
            }
            let image = cel((i / hold) as u32, if pan { 3 * i as u32 } else { 0 }, cut);
            write(gpu, &gen, slot, &image);
            let pushed = gen.push_slot(gpu, slot).unwrap();
            gen.resolve_stats(gpu).unwrap();
            assert_eq!(pushed.ready, i >= 2 && !(15..17).contains(&i));
            if i == 13 {
                assert!(gen.stats()[2].cut);
            }
            if !pushed.ready {
                continue;
            }
            for t in [0.0, 0.25, 0.5, 0.75] {
                let decision = gen.retime(t);
                if cut {
                    assert!(decision.is_none(), "never retime through a detected cut");
                }
                retimed += usize::from(decision.is_some());
                let written = gen.generate(gpu, t, 0).unwrap();
                assert_eq!(written, t > 0.0 || decision.is_some());
                let pixels = written.then(|| fingerprint(gpu, &gen.outs[0].0));
                report.push_str(&format!("{hold} {pan} {i} {t} {decision:?} {pixels:016x?}\n"));
            }
            if i == 7 {
                let sc = gen.scratch.as_ref().unwrap();
                let bg = &gen.sess[gen.order[1]];
                for (full, which, out) in [(true, 1, 0), (false, 0, 1), (false, 2, 0)] {
                    let elem = &gen.sess[gen.order[which]];
                    gmhs::composite_frame(gpu, sc, (bg, 0.25), (elem, 0.75), (elem, full), out, &gen.outs[out].1)
                        .unwrap();
                    report.push_str(&format!(
                        "composite {hold} {pan} {full} {which} {:016x}\n",
                        fingerprint(gpu, &gen.outs[out].0)
                    ));
                }
            }
        }
        if !pan {
            assert!(retimed > 0, "held drawings must exercise retiming");
        }
    }
    if let Some(path) = std::env::var_os("FH_CADENCE_CAPTURE") {
        std::fs::write(path, &report).unwrap();
    }
}

#[test]
#[ignore = "requires a GPU"]
fn cadence_1080p_memory() {
    let e = engine();
    let core = crate::Engine::from_shared(e.gpu.clone());
    let before = core.report().memory;
    let mut gen = anime(&e, 1920, 1080);
    for _ in 0..4 {
        gen.add_input(&e.gpu);
    }
    for _ in 0..2 {
        gen.add_output(&e.gpu);
    }
    let active = core.report().memory;
    eprintln!("1080p cadence memory: {}", serde_json::to_string(&active).unwrap());
    drop(gen);
    assert_eq!(core.report().memory.live_bytes, before.live_bytes);
}

#[test]
#[ignore = "requires a GPU"]
fn video_plans_share_immutable_convolution_weights() {
    let e = engine();
    let generator = anime(&e, W, H);
    let first_conv_weights: Vec<_> = generator
        .sess
        .iter()
        .map(|s| {
            let conv = s
                .plan_pair
                .iter()
                .find(|s| {
                    matches!(s, Step::Compute { pipe, .. }
            if ["conv", "implicit", "fused", "tconv", "fdirect"].iter().any(|prefix| pipe.starts_with(prefix)))
                })
                .unwrap();
            conv.allocations().nth(1).unwrap().clone()
        })
        .collect();
    assert!(first_conv_weights.iter().all(|w| std::rc::Rc::ptr_eq(w, &first_conv_weights[0])));
}
