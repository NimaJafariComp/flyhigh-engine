//! GPU time of a plan through timestamp queries: blocking, asynchronous and per-step profiles.

use crate::{map_read, Gpu, MapWait, Step};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

const TIMING_SLOTS: usize = 4;
const TIMING_BYTES: u64 = 32;

pub(crate) struct TimingPool {
    queries: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    slots: [Arc<TimingSlot>; TIMING_SLOTS],
}

struct TimingSlot {
    stage: wgpu::Buffer,
    device: wgpu::Device,
    queries: Option<wgpu::QuerySet>,
    busy: AtomicBool,
}

struct TimingLease {
    slot: Arc<TimingSlot>,
    mapping: bool,
}

impl Drop for TimingLease {
    fn drop(&mut self) {
        if self.mapping {
            // Cancellation may race a failed map, which has already unmapped the staging buffer.
            let _cancel = self.slot.device.push_error_scope(wgpu::ErrorFilter::Validation);
            self.slot.stage.unmap();
        }
        self.slot.busy.store(false, Ordering::Release);
    }
}

#[derive(Default)]
pub(crate) enum TimingState {
    #[default]
    Cold,
    Checking(Pin<Box<dyn Future<Output = Option<TimingPool>>>>),
    Ready(TimingPool),
    Unavailable,
}

impl TimingState {
    fn pool(&mut self, device: &wgpu::Device, deferred: bool) -> Option<&TimingPool> {
        if matches!(self, Self::Cold) {
            let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let buffer = |label, usage| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: TIMING_BYTES,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let queries = || {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("fh_live_timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 4,
                })
            };
            let pool = TimingPool {
                queries: (!deferred).then(queries),
                resolve: buffer("fh_timing_resolve", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                slots: std::array::from_fn(|_| {
                    Arc::new(TimingSlot {
                        stage: buffer(
                            "fh_timing_readback",
                            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        ),
                        device: device.clone(),
                        queries: deferred.then(queries),
                        busy: AtomicBool::new(false),
                    })
                }),
            };
            let (validation, internal, oom) = (validation.pop(), internal.pop(), oom.pop());
            *self = Self::Checking(Box::pin(async move {
                let (validation, internal, oom) = (validation.await, internal.await, oom.await);
                (validation.is_none() && internal.is_none() && oom.is_none()).then_some(pool)
            }));
        }
        if let Self::Checking(check) = self {
            match check.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
                Poll::Pending => return None,
                Poll::Ready(Some(pool)) => *self = Self::Ready(pool),
                Poll::Ready(None) => *self = Self::Unavailable,
            }
        }
        match self {
            Self::Ready(pool) => Some(pool),
            _ => None,
        }
    }
}

impl Gpu {
    fn defer_timing_resolve(&self) -> bool {
        cfg!(target_os = "macos") && self.adapter.get_info().backend == wgpu::Backend::Metal
    }

    // pure GPU execution time of the whole command buffer, via timestamp queries (best of N).
    // wall_clock(replay+wait) - this = CPU-side overhead (encode + submit + driver launch + poll).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_time_ms(&self, plan: &[Step]) -> f64 {
        self.gpu_time_ms_n(plan, 60)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_time_ms_n(&self, plan: &[Step], reps: usize) -> f64 {
        pollster::block_on(self.gpu_time_ms_async(plan, reps))
    }
    // min-of-reps GPU time of `plan` from timestamp queries (encoder-level where available, else pass-level — the
    // WebGPU-standard kind, which Chrome exposes); NaN when the device has no timestamps at all
    pub async fn gpu_time_ms_async(&self, plan: &[Step], reps: usize) -> f64 {
        if !self.timestamps {
            return f64::NAN;
        }
        #[cfg(target_os = "macos")]
        if self.defer_timing_resolve() {
            return (0..reps)
                .filter_map(|_| self.begin_timed(plan).and_then(|r| r.block_parts(self)).map(|p| p.total_ms))
                .reduce(f64::min)
                .unwrap_or(f64::NAN);
        }
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period = self.queue.get_timestamp_period() as f64; // ns per tick
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            if self.ts_inside {
                enc.write_timestamp(&qs, 0);
                self.encode_plan(&mut enc, plan);
                enc.write_timestamp(&qs, 1);
            } else {
                self.encode_plan_ts(&mut enc, plan, Some(&qs));
            }
            enc.resolve_query_set(&qs, 0..2, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &stage, 0, 16);
            self.queue.submit(Some(enc.finish()));
            let fut = map_read(&stage);
            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            if fut.await.is_err() {
                return f64::NAN;
            }
            let ms = {
                let data = stage.slice(..).get_mapped_range().unwrap();
                let ts: &[u64] = bytemuck::cast_slice(&data);
                ts[1].wrapping_sub(ts[0]) as f64 * period / 1e6
            };
            stage.unmap();
            if ms > 0.0 && ms < best {
                best = ms;
            }
        }
        best
    }
    // Timed run in two halves for hosts that must keep GPU submission order: `begin_timed` encodes `plan` with
    // timestamps, SUBMITS IT NOW and requests the readback; `TimedRun::finish` awaits the result later. (An async
    // export that submits inside its body runs that body in a microtask — after whatever the host submitted next.)
    pub fn begin_timed(&self, plan: &[Step]) -> Option<TimedRun> {
        self.begin_timed_parts(&[], plan)
    }

    pub(crate) fn begin_timed_parts(&self, setup: &[Step], plan: &[Step]) -> Option<TimedRun> {
        let has_compute = |steps: &[Step]| steps.iter().any(|s| matches!(s, Step::Compute { .. }));
        let can_time = |steps: &[Step]| self.ts_inside || has_compute(steps);
        if !self.timestamps || plan.is_empty() || !can_time(plan) || (!setup.is_empty() && !can_time(setup)) {
            return None;
        }
        let deferred = self.defer_timing_resolve();
        let mut timing = self.timing.borrow_mut();
        let pool = timing.pool(&self.device, deferred)?;
        let mut lease = pool.slots.iter().find_map(|slot| {
            slot.busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| TimingLease { slot: slot.clone(), mapping: false })
        })?;
        let queries = lease.slot.queries.as_ref().or(pool.queries.as_ref())?;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut section = |steps, index| {
            if has_compute(steps) {
                self.encode_plan_query_range(&mut enc, steps, Some((queries, index)));
            } else {
                enc.write_timestamp(queries, index);
                self.encode_plan(&mut enc, steps);
                enc.write_timestamp(queries, index + 1);
            }
        };
        let has_setup = !setup.is_empty();
        let index = if has_setup {
            section(setup, 0);
            2
        } else {
            0
        };
        section(plan, index);
        let count = index + 2;
        let read = if deferred {
            // Metal can resolve a counter before its samples become visible. Keep the query set leased
            // until a separate resolve is submitted after the sampling command has completed.
            self.queue.submit(Some(enc.finish()));
            TimingRead::Deferred(DeferredResolve {
                ready: super::readback::queue_done(&self.queue),
                device: self.device.clone(),
                queue: self.queue.clone(),
                queries: queries.clone(),
                resolve: pool.resolve.clone(),
                count,
            })
        } else {
            record_readback(&mut enc, queries, &pool.resolve, &lease.slot.stage, count);
            self.queue.submit(Some(enc.finish()));
            lease.mapping = true;
            TimingRead::Mapping(map_read(&lease.slot.stage))
        };
        Some(TimedRun { lease, read, period: self.queue.get_timestamp_period() as f64, has_setup })
    }

    pub(crate) fn submit_optional_timing(&self, setup: &[Step], plan: &[Step], timed: bool) -> Option<TimedRun> {
        if timed {
            if let Some(run) = self.begin_timed_parts(setup, plan) {
                return Some(run);
            }
        }
        self.replay(setup);
        self.replay(plan);
        None
    }
    // per-step GPU time: one timestamp per step, each compute in its own pass so encoder-level
    // write_timestamp can bracket it; aggregated by pipeline name, averaged over REPS. Profiling only —
    // pass splitting adds inter-pass gaps, so the summed total exceeds replay time; the SHARES are the signal.
    pub fn profile_plan(&self, plan: &[Step]) -> Vec<(String, f64)> {
        if !self.ts_inside {
            println!("  (no encoder-level timestamp queries on this device: per-op profile unavailable)");
            return Vec::new();
        }
        let n = plan.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: n + 1,
        });
        let bytes = ((n + 1) * 8) as u64;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period = self.queue.get_timestamp_period() as f64;
        const REPS: usize = 20;
        let mut acc: Vec<f64> = vec![0.0; plan.len()];
        for _ in 0..REPS {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for (i, st) in plan.iter().enumerate() {
                enc.write_timestamp(&qs, i as u32);
                match st {
                    Step::Copy { src, so, dst, dof, size } => enc.copy_buffer_to_buffer(src, *so, dst, *dof, *size),
                    Step::Compute { pipe, bg, groups, .. } => {
                        let mut cp = enc
                            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cp.set_pipeline(&self.pipeline(pipe));
                        cp.set_bind_group(0, bg, &[]);
                        cp.dispatch_workgroups(groups.0, groups.1, groups.2);
                    }
                }
            }
            enc.write_timestamp(&qs, n);
            enc.resolve_query_set(&qs, 0..n + 1, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &stage, 0, bytes);
            self.queue.submit(Some(enc.finish()));
            let slice = stage.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().unwrap().unwrap();
            {
                let data = slice.get_mapped_range().unwrap();
                let ts: &[u64] = bytemuck::cast_slice(&data);
                for i in 0..plan.len() {
                    acc[i] += ts[i + 1].wrapping_sub(ts[i]) as f64 * period / 1e6;
                }
            }
            stage.unmap();
        }
        let mut top: Vec<(usize, f64)> = acc.iter().enumerate().map(|(i, a)| (i, a / REPS as f64)).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  top steps (idx pipe groups ms): naive conv groups = (ceil(w/8), ceil(h/8), cout); fused = (np/nblk, pixels/16)");
        for (i, ms) in top.iter().take(14) {
            match &plan[*i] {
                Step::Compute { pipe, groups, .. } => println!(
                    "    #{:<3} {:<8} ({:>4},{:>4},{:>4})  {:>6.3} ms",
                    i, pipe, groups.0, groups.1, groups.2, ms
                ),
                Step::Copy { size, .. } => println!("    #{:<3} copy     {:>10} B           {:>6.3} ms", i, size, ms),
            }
        }
        let mut by: HashMap<String, f64> = HashMap::new();
        for (i, st) in plan.iter().enumerate() {
            let key = match st {
                Step::Copy { .. } => "copy".to_string(),
                Step::Compute { pipe, .. } => pipe.to_string(),
            };
            *by.entry(key).or_insert(0.0) += acc[i] / REPS as f64;
        }
        let mut v: Vec<(String, f64)> = by.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    }
}

// a submitted timed run whose timestamp readback is still in flight (see `Gpu::begin_timed`)
pub struct TimedRun {
    lease: TimingLease,
    read: TimingRead,
    period: f64,
    has_setup: bool,
}

enum TimingRead {
    Mapping(MapWait),
    Deferred(DeferredResolve),
}

struct DeferredResolve {
    ready: super::readback::QueueWait,
    device: wgpu::Device,
    queue: wgpu::Queue,
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    count: u32,
}

fn record_readback(
    enc: &mut wgpu::CommandEncoder,
    queries: &wgpu::QuerySet,
    resolve: &wgpu::Buffer,
    stage: &wgpu::Buffer,
    count: u32,
) {
    enc.resolve_query_set(queries, 0..count, resolve, 0);
    enc.copy_buffer_to_buffer(resolve, 0, stage, 0, u64::from(count) * 8);
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimingParts {
    pub setup_ms: Option<f64>,
    pub frame_ms: f64,
    pub total_ms: f64,
}
impl TimedRun {
    pub async fn finish(self) -> f64 {
        self.finish_parts().await.map_or(f64::NAN, |p| p.total_ms)
    }

    pub async fn finish_parts(self) -> Option<TimingParts> {
        self.finish_parts_impl(false).await
    }

    async fn finish_parts_impl(mut self, blocking: bool) -> Option<TimingParts> {
        let stage = &self.lease.slot.stage;
        let mapping = match self.read {
            TimingRead::Mapping(fut) => fut,
            TimingRead::Deferred(res) => {
                res.ready.await;
                let mut enc = res.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                record_readback(&mut enc, &res.queries, &res.resolve, stage, res.count);
                let submitted = res.queue.submit(Some(enc.finish()));
                self.lease.mapping = true;
                let mapping = map_read(stage);
                if blocking
                    && res
                        .device
                        .poll(wgpu::PollType::Wait { submission_index: Some(submitted), timeout: None })
                        .is_err()
                {
                    self.lease.mapping = false;
                    return None;
                }
                mapping
            }
        };
        if mapping.await.is_err() {
            self.lease.mapping = false;
            return None;
        }
        let parts = {
            let data = stage.slice(..).get_mapped_range().ok()?;
            let ts: &[u64] = bytemuck::cast_slice(&data);
            let ms = |i: usize| {
                let value = ts[i + 1].checked_sub(ts[i])? as f64 * self.period / 1e6;
                (value.is_finite() && value > 0.0).then_some(value)
            };
            if self.has_setup {
                ms(0).zip(ms(2)).map(|(setup, frame)| (Some(setup), frame))
            } else {
                ms(0).map(|frame| (None, frame))
            }
        };
        stage.unmap();
        self.lease.mapping = false;
        let (setup_ms, frame_ms) = parts?;
        Some(TimingParts { setup_ms, frame_ms, total_ms: setup_ms.unwrap_or(0.0) + frame_ms })
    }

    /// Native: poll the device and wait for the result.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn block(self, gpu: &Gpu) -> f64 {
        self.block_parts(gpu).map_or(f64::NAN, |p| p.total_ms)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn block_parts(self, gpu: &Gpu) -> Option<TimingParts> {
        gpu.wait();
        pollster::block_on(self.finish_parts_impl(true))
    }
}
