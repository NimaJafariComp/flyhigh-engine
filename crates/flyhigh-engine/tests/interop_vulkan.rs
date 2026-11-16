// Frames from another Vulkan device: a host device of its own writes two frames into exported buffers, signals a
// timeline semaphore, the engine copies them in after the wait, generates the in-between, copies it out and signals;
// the host waits and reads it back. It must match the engine's own upload path byte for byte. Ignored by default; needs
// a Vulkan GPU (`cargo test --release --test interop_vulkan -- --ignored`).
#![cfg(any(windows, target_os = "linux"))]
use ash::vk;
use flyhigh_engine::vulkan_interop::{self, buffer_size, row_pitch};
use flyhigh_engine::{Engine, EngineOptions, Model, Session, SessionOptions};
use std::path::PathBuf;

const W: u32 = 200;
const H: u32 = 120;

fn frame(dx: f32) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let u = (x as f32 - dx) * 0.07;
            let w = y as f32 * 0.05;
            let o = ((y * W + x) * 4) as usize;
            v[o] = (128.0 + 100.0 * u.sin()) as u8;
            v[o + 1] = (128.0 + 100.0 * (u * 0.7 + w).cos()) as u8;
            v[o + 2] = (128.0 + 90.0 * (u + w).sin()) as u8;
            v[o + 3] = 255;
        }
    }
    v
}

fn pitched(rgba: &[u8]) -> Vec<u8> {
    let pitch = row_pitch(W) as usize;
    let mut v = vec![0u8; pitch * H as usize];
    for y in 0..H as usize {
        let row = W as usize * 4;
        v[y * pitch..y * pitch + row].copy_from_slice(&rgba[y * row..(y + 1) * row]);
    }
    v
}

fn unpitched(bytes: &[u8]) -> Vec<u8> {
    let pitch = row_pitch(W) as usize;
    let row = W as usize * 4;
    (0..H as usize).flat_map(|y| bytes[y * pitch..y * pitch + row].iter().copied()).collect()
}

#[cfg(windows)]
const MEMORY_HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
#[cfg(not(windows))]
const MEMORY_HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
#[cfg(windows)]
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32;
#[cfg(not(windows))]
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

struct Host {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
}

struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocation_size: u64,
    memory_type_index: u32,
}

impl Host {
    fn new(uuid: [u8; 16]) -> Option<Host> {
        let entry = unsafe { ash::Entry::load() }.ok()?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance =
            unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }.ok()?;
        let physical = unsafe { instance.enumerate_physical_devices() }.ok()?.into_iter().find(|&p| {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe { instance.get_physical_device_properties2(p, &mut props) };
            id.device_uuid == uuid
        })?;
        let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }.iter().position(|f| {
            f.queue_flags.contains(vk::QueueFlags::TRANSFER) || f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })? as u32;
        #[cfg(windows)]
        let extensions =
            [ash::khr::external_memory_win32::NAME.as_ptr(), ash::khr::external_semaphore_win32::NAME.as_ptr()];
        #[cfg(not(windows))]
        let extensions = [ash::khr::external_memory_fd::NAME.as_ptr(), ash::khr::external_semaphore_fd::NAME.as_ptr()];
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&[1.0])];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extensions)
            .push_next(&mut timeline);
        let device = unsafe { instance.create_device(physical, &info, None) }.ok()?;
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool = unsafe {
            device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None)
        }
        .ok()?;
        Some(Host { _entry: entry, instance, physical, device, queue, pool })
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        let types = unsafe { self.instance.get_physical_device_memory_properties(self.physical) };
        (0..types.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && types.memory_types[i as usize].property_flags.contains(flags))
            .expect("memory type")
    }

    fn buffer(&self, size: u64, exported: bool) -> HostBuffer {
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(MEMORY_HANDLE);
        let mut info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        if exported {
            info = info.push_next(&mut external);
        }
        let buffer = unsafe { self.device.create_buffer(&info, None) }.unwrap();
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let flags = if exported {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        } else {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        };
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(MEMORY_HANDLE);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let memory_type_index = self.memory_type(req.memory_type_bits, flags);
        let mut alloc =
            vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(memory_type_index);
        if exported {
            alloc = alloc.push_next(&mut export).push_next(&mut dedicated);
        }
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }.unwrap();
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        HostBuffer { buffer, memory, allocation_size: req.size, memory_type_index }
    }

    fn export_memory(&self, b: &HostBuffer) -> vulkan_interop::ExternalMemory {
        vulkan_interop::ExternalMemory {
            handle: self.export_handle(b),
            allocation_size: b.allocation_size,
            memory_type_index: b.memory_type_index,
        }
    }

    fn export_handle(&self, b: &HostBuffer) -> isize {
        #[cfg(windows)]
        {
            let ext = ash::khr::external_memory_win32::Device::new(&self.instance, &self.device);
            let info = vk::MemoryGetWin32HandleInfoKHR::default().memory(b.memory).handle_type(MEMORY_HANDLE);
            unsafe { ext.get_memory_win32_handle(&info) }.unwrap()
        }
        #[cfg(not(windows))]
        {
            let ext = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);
            let info = vk::MemoryGetFdInfoKHR::default().memory(b.memory).handle_type(MEMORY_HANDLE);
            unsafe { ext.get_memory_fd(&info) }.unwrap() as isize
        }
    }

    fn timeline(&self) -> (vk::Semaphore, isize) {
        let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(SEMAPHORE_HANDLE);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind).push_next(&mut export);
        let semaphore = unsafe { self.device.create_semaphore(&info, None) }.unwrap();
        #[cfg(windows)]
        let handle = {
            let ext = ash::khr::external_semaphore_win32::Device::new(&self.instance, &self.device);
            let info = vk::SemaphoreGetWin32HandleInfoKHR::default().semaphore(semaphore).handle_type(SEMAPHORE_HANDLE);
            unsafe { ext.get_semaphore_win32_handle(&info) }.unwrap()
        };
        #[cfg(not(windows))]
        let handle = {
            let ext = ash::khr::external_semaphore_fd::Device::new(&self.instance, &self.device);
            let info = vk::SemaphoreGetFdInfoKHR::default().semaphore(semaphore).handle_type(SEMAPHORE_HANDLE);
            unsafe { ext.get_semaphore_fd(&info) }.unwrap() as isize
        };
        (semaphore, handle)
    }

    fn write(&self, b: &HostBuffer, bytes: &[u8]) {
        unsafe {
            let p = self.device.map_memory(b.memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p as *mut u8, bytes.len());
            self.device.unmap_memory(b.memory);
        }
    }

    fn read(&self, b: &HostBuffer, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        unsafe {
            let p = self.device.map_memory(b.memory, 0, len as u64, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(p as *const u8, v.as_mut_ptr(), len);
            self.device.unmap_memory(b.memory);
        }
        v
    }

    /// Copies `copies` (from, to) on the GPU after `wait` and signals `signal` on `semaphore`, then returns.
    fn submit(
        &self,
        copies: &[(&HostBuffer, &HostBuffer)],
        size: u64,
        semaphore: vk::Semaphore,
        wait: u64,
        signal: u64,
    ) {
        unsafe {
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1),
                )
                .unwrap()[0];
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            for (from, to) in copies {
                self.device.cmd_copy_buffer(cmd, from.buffer, to.buffer, &[vk::BufferCopy::default().size(size)]);
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.device.end_command_buffer(cmd).unwrap();
            let waits = if wait > 0 { vec![semaphore] } else { vec![] };
            let wait_values = [wait];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let signals = [semaphore];
            let signal_values = [signal];
            let mut values = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values[..waits.len()])
                .signal_semaphore_values(&signal_values);
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages[..waits.len()])
                .command_buffers(&cmds)
                .signal_semaphores(&signals)
                .push_next(&mut values);
            self.device.queue_submit(self.queue, &[submit], vk::Fence::null()).unwrap();
        }
    }

    fn wait(&self, semaphore: vk::Semaphore, value: u64) {
        let semaphores = [semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        unsafe { self.device.wait_semaphores(&info, 5_000_000_000) }.expect("the engine signals within 5 s");
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

#[test]
#[ignore]
fn frames_from_another_vulkan_device_come_back_generated() {
    let engine = match Engine::new(&EngineOptions { backend: Some(wgpu::Backends::VULKAN), ..EngineOptions::default() })
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipped: no Vulkan engine ({e:#})");
            return;
        }
    };
    if !vulkan_interop::supported(&engine) {
        eprintln!("skipped: {} has no external memory and semaphore extensions", engine.info().name);
        return;
    }
    let uuid = vulkan_interop::device_uuid(&engine).expect("a Vulkan engine has a device UUID");
    let Some(host) = Host::new(uuid) else {
        eprintln!("skipped: no host Vulkan device on the engine's GPU");
        return;
    };
    let size = buffer_size(W, H);
    let (f0, f1) = (frame(0.0), frame(8.0));
    let staging = [host.buffer(size, false), host.buffer(size, false)];
    host.write(&staging[0], &pitched(&f0));
    host.write(&staging[1], &pitched(&f1));
    let shared = [host.buffer(size, true), host.buffer(size, true), host.buffer(size, true)];
    let (semaphore, semaphore_handle) = host.timeline();

    let imported: Vec<wgpu::Buffer> = shared
        .iter()
        .map(|b| vulkan_interop::import_buffer(&engine, &host.export_memory(b), size).expect("import buffer"))
        .collect();
    let sem = vulkan_interop::import_semaphore(&engine, semaphore_handle, true).expect("import semaphore");
    assert!(sem.is_timeline());

    let model =
        Model::from_dir(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models")).unwrap();
    let mut s = Session::new(&engine, &model, &SessionOptions::new(W, H)).unwrap();
    let (a, b, out) = (s.add_input(), s.add_input(), s.add_output());

    // the engine's waits go in before the host's signal: a timeline semaphore allows it
    let pitch = row_pitch(W);
    vulkan_interop::copy_in(&engine, &imported[0], pitch, s.input_texture(a), Some((&sem, 1))).unwrap();
    vulkan_interop::copy_in(&engine, &imported[1], pitch, s.input_texture(b), Some((&sem, 1))).unwrap();
    host.submit(&[(&staging[0], &shared[0]), (&staging[1], &shared[1])], size, semaphore, 0, 1);
    s.pair(a, b).unwrap();
    s.mid(0.5, out).unwrap();
    vulkan_interop::copy_out(&engine, s.output_texture(out), &imported[2], pitch, Some((&sem, 2))).unwrap();
    let readback = host.buffer(size, false);
    host.submit(&[(&shared[2], &readback)], size, semaphore, 2, 3);
    host.wait(semaphore, 3);
    let got = unpitched(&host.read(&readback, size as usize));

    let mut reference = Session::new(&engine, &model, &SessionOptions::new(W, H)).unwrap();
    let (ra, rb, rout) = (reference.add_input(), reference.add_input(), reference.add_output());
    reference.write_rgba(ra, &f0).unwrap();
    reference.write_rgba(rb, &f1).unwrap();
    reference.generate(ra, rb, 0.5, rout).unwrap();
    let want = reference.read_rgba(rout).unwrap().block(engine.gpu()).unwrap();
    assert_eq!(got.len(), want.len());
    let differing = got.iter().zip(&want).filter(|(x, y)| x != y).count();
    eprintln!("{}: in-between through another Vulkan device, {differing} bytes differ", engine.info().name);
    assert_eq!(differing, 0, "the interop path gives the same frame as the upload path");

    assert!(vulkan_interop::copy_in(&engine, &imported[0], 100, s.input_texture(a), None).is_err());
    assert!(vulkan_interop::copy_in(&engine, &imported[0], pitch, s.input_texture(a), Some((&sem, 0))).is_err());

    drop(sem);
    drop(imported);
    unsafe {
        let _ = host.device.device_wait_idle();
        host.device.destroy_semaphore(semaphore, None);
        for b in shared.iter().chain(&staging).chain([&readback]) {
            host.device.destroy_buffer(b.buffer, None);
            host.device.free_memory(b.memory, None);
        }
    }
}
