//! Frames from a host's own Vulkan device (an emulator, a game, a capture) on the engine's Vulkan device, with no copy
//! through the CPU. The host allocates exportable memory and semaphores and hands their handles over: an NT handle on
//! Windows (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`, the same for semaphores), a file descriptor elsewhere
//! (`OPAQUE_FD`). The host keeps a Windows handle and may close it after the import; a file descriptor belongs to the
//! engine once the import succeeds.
//!
//! Frames cross as buffers, not images: an image's layout and compression are private to the device that made it, so
//! two devices sharing one need matching create parameters and layout handoffs on every frame, while a buffer is the
//! same bytes on both sides. Rows are RGBA8, `bytes_per_row` a multiple of 256 (`row_pitch`). The copies in and out
//! cost about 0.05 ms per 1080p frame each.
//!
//! Ordering is the semaphores' job. The host signals after writing a frame and `copy_in` makes the engine's next
//! submit wait for it; `copy_out` signals after the engine has written its frame, and the host waits before reading.
//! A binary semaphore needs its signal submitted before the wait (value 0); a timeline semaphore waits for a value.
use crate::Engine;
use anyhow::{bail, ensure, Context, Result};
use ash::vk;
use wgpu::hal::api::Vulkan;

#[cfg(windows)]
const MEMORY_HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
#[cfg(not(windows))]
const MEMORY_HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
#[cfg(windows)]
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32;
#[cfg(not(windows))]
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

#[cfg(windows)]
const EXTENSIONS: [&std::ffi::CStr; 2] =
    [ash::khr::external_memory_win32::NAME, ash::khr::external_semaphore_win32::NAME];
#[cfg(not(windows))]
const EXTENSIONS: [&std::ffi::CStr; 2] = [ash::khr::external_memory_fd::NAME, ash::khr::external_semaphore_fd::NAME];

/// The bytes per row of an RGBA8 frame `width` wide in an interop buffer.
pub fn row_pitch(width: u32) -> u32 {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// The size of an interop buffer for an RGBA8 frame.
pub fn buffer_size(width: u32, height: u32) -> u64 {
    row_pitch(width) as u64 * height as u64
}

/// Opens the device with the external memory and semaphore extensions when the adapter has them, else `None` and the
/// caller requests the device as usual.
pub(crate) fn open_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Option<Result<(wgpu::Device, wgpu::Queue)>> {
    if adapter.get_info().backend != wgpu::Backend::Vulkan {
        return None;
    }
    let open = unsafe {
        let hal = adapter.as_hal::<Vulkan>()?;
        if !EXTENSIONS.iter().all(|e| hal.physical_device_capabilities().supports_extension(e)) {
            return None;
        }
        hal.open_with_callback(
            desc.required_features,
            &desc.required_limits,
            &desc.memory_hints,
            Some(Box::new(|args: wgpu::hal::vulkan::CreateDeviceCallbackArgs<'_, '_, '_>| {
                for e in EXTENSIONS {
                    if !args.extensions.contains(&e) {
                        args.extensions.push(e);
                    }
                }
            })),
        )
    };
    let open = match open {
        Ok(open) => open,
        Err(e) => return Some(Err(anyhow::anyhow!("opening the Vulkan device: {e:?}"))),
    };
    Some(unsafe { adapter.create_device_from_hal(open, desc) }.map_err(|e| anyhow::anyhow!("no wgpu device: {e:?}")))
}

fn has_extensions(engine: &Engine) -> bool {
    let device = &engine.gpu().device;
    match unsafe { device.as_hal::<Vulkan>() } {
        Some(hal) => EXTENSIONS.iter().all(|e| hal.enabled_device_extensions().contains(e)),
        None => false,
    }
}

/// Whether this engine's device can take buffers and semaphores from another Vulkan device.
pub fn supported(engine: &Engine) -> bool {
    has_extensions(engine)
}

/// The engine's `VkPhysicalDeviceIDProperties::deviceUUID`: memory and semaphores are shared only between devices on
/// the same GPU and driver, so a host compares it with its own before exporting anything.
pub fn device_uuid(engine: &Engine) -> Option<[u8; 16]> {
    let device = &engine.gpu().device;
    let hal = unsafe { device.as_hal::<Vulkan>() }?;
    let instance = hal.shared_instance().raw_instance();
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    unsafe { instance.get_physical_device_properties2(hal.raw_physical_device(), &mut props) };
    Some(id.device_uuid)
}

/// Exported memory as the host allocated it. Vulkan requires the import to repeat the exporter's allocation size and
/// memory type index (the devices share one physical device, so the indices agree); the allocation is a dedicated one
/// for a buffer with `TRANSFER_SRC | TRANSFER_DST` usage.
#[derive(Clone, Copy, Debug)]
pub struct ExternalMemory {
    pub handle: isize,
    pub allocation_size: u64,
    pub memory_type_index: u32,
}

/// Imports `size` bytes of another device's memory as a buffer the engine copies frames to and from.
pub fn import_buffer(engine: &Engine, memory: &ExternalMemory, size: u64) -> Result<wgpu::Buffer> {
    ensure!(memory.handle != 0 && memory.handle != -1, "memory handle is not valid");
    ensure!(size > 0, "buffer size is 0");
    ensure!(supported(engine), "this device cannot import Vulkan memory");
    let device = &engine.gpu().device;
    let hal_buffer = unsafe {
        let hal = device.as_hal::<Vulkan>().context("the engine device is not Vulkan")?;
        let raw = hal.raw_device();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(MEMORY_HANDLE);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer = raw.create_buffer(&info, None).context("vkCreateBuffer")?;
        let device_memory = match import_memory(raw, buffer, memory) {
            Ok(device_memory) => device_memory,
            Err(e) => {
                raw.destroy_buffer(buffer, None);
                return Err(e);
            }
        };
        if let Err(e) = raw.bind_buffer_memory(buffer, device_memory, 0) {
            raw.destroy_buffer(buffer, None);
            raw.free_memory(device_memory, None);
            bail!("vkBindBufferMemory: {e}");
        }
        wgpu::hal::vulkan::Buffer::from_raw_managed(buffer, device_memory, 0, size)
    };
    Ok(unsafe {
        device.create_buffer_from_hal::<Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("fh_vk_imported"),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
    })
}

unsafe fn import_memory(raw: &ash::Device, buffer: vk::Buffer, memory: &ExternalMemory) -> Result<vk::DeviceMemory> {
    let req = unsafe { raw.get_buffer_memory_requirements(buffer) };
    ensure!(
        req.memory_type_bits & (1 << memory.memory_type_index) != 0,
        "memory type {} cannot back a transfer buffer here",
        memory.memory_type_index
    );
    ensure!(
        memory.allocation_size >= req.size,
        "allocation of {} bytes is smaller than the {} the buffer needs",
        memory.allocation_size,
        req.size
    );
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
    #[cfg(windows)]
    let mut import =
        vk::ImportMemoryWin32HandleInfoKHR::default().handle_type(MEMORY_HANDLE).handle(memory.handle as vk::HANDLE);
    #[cfg(not(windows))]
    let mut import = vk::ImportMemoryFdInfoKHR::default().handle_type(MEMORY_HANDLE).fd(memory.handle as i32);
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(memory.allocation_size)
        .memory_type_index(memory.memory_type_index)
        .push_next(&mut import)
        .push_next(&mut dedicated);
    unsafe { raw.allocate_memory(&alloc, None) }.context("importing the memory (vkAllocateMemory)")
}

/// A semaphore of another device, imported into the engine's device.
pub struct Semaphore {
    raw: vk::Semaphore,
    timeline: bool,
    device: wgpu::Device,
}

impl Semaphore {
    pub fn is_timeline(&self) -> bool {
        self.timeline
    }

    fn value(&self, value: u64) -> Result<Option<u64>> {
        match (self.timeline, value) {
            (true, 0) => bail!("a timeline semaphore needs a value above 0"),
            (true, v) => Ok(Some(v)),
            (false, 0) => Ok(None),
            (false, _) => bail!("a binary semaphore takes value 0"),
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // the queue may still hold a wait or a signal on it
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(hal) = unsafe { self.device.as_hal::<Vulkan>() } {
            unsafe { hal.raw_device().destroy_semaphore(self.raw, None) };
        }
    }
}

/// Imports the semaphore behind `handle`; `timeline` says which kind the host created.
pub fn import_semaphore(engine: &Engine, handle: isize, timeline: bool) -> Result<Semaphore> {
    ensure!(handle != 0 && handle != -1, "semaphore handle is not valid");
    ensure!(supported(engine), "this device cannot import Vulkan semaphores");
    let device = &engine.gpu().device;
    let raw = unsafe {
        let hal = device.as_hal::<Vulkan>().context("the engine device is not Vulkan")?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(if timeline {
            vk::SemaphoreType::TIMELINE
        } else {
            vk::SemaphoreType::BINARY
        });
        let semaphore = raw.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None)?;
        #[cfg(windows)]
        let imported = ash::khr::external_semaphore_win32::Device::new(instance, raw).import_semaphore_win32_handle(
            &vk::ImportSemaphoreWin32HandleInfoKHR::default()
                .semaphore(semaphore)
                .handle_type(SEMAPHORE_HANDLE)
                .handle(handle as vk::HANDLE),
        );
        #[cfg(not(windows))]
        let imported = ash::khr::external_semaphore_fd::Device::new(instance, raw).import_semaphore_fd(
            &vk::ImportSemaphoreFdInfoKHR::default()
                .semaphore(semaphore)
                .handle_type(SEMAPHORE_HANDLE)
                .fd(handle as i32),
        );
        if let Err(e) = imported {
            raw.destroy_semaphore(semaphore, None);
            bail!("importing the semaphore: {e}");
        }
        semaphore
    };
    Ok(Semaphore { raw, timeline, device: device.clone() })
}

fn check_rows(texture: &wgpu::Texture, buffer: &wgpu::Buffer, bytes_per_row: u32) -> Result<()> {
    ensure!(texture.format() == wgpu::TextureFormat::Rgba8Unorm, "interop frames are rgba8unorm");
    ensure!(
        bytes_per_row.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) && bytes_per_row >= texture.width() * 4,
        "bytes_per_row {bytes_per_row}: at least {} and a multiple of 256",
        texture.width() * 4
    );
    ensure!(
        buffer.size() >= bytes_per_row as u64 * texture.height() as u64,
        "buffer of {} bytes is smaller than {} rows of {bytes_per_row}",
        buffer.size(),
        texture.height()
    );
    Ok(())
}

fn layout(bytes_per_row: u32, height: u32) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(height) }
}

/// Copies a frame from `buffer` into `texture` (a stream or session input), after `wait` when given. The wait gates
/// this submit; everything later that reads the texture is ordered after the copy on the queue.
pub fn copy_in(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    bytes_per_row: u32,
    texture: &wgpu::Texture,
    wait: Option<(&Semaphore, u64)>,
) -> Result<()> {
    check_rows(texture, buffer, bytes_per_row)?;
    let gpu = engine.gpu();
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fh_vk_in") });
    enc.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo { buffer, layout: layout(bytes_per_row, texture.height()) },
        texture.as_image_copy(),
        texture.size(),
    );
    if let Some((semaphore, value)) = wait {
        let value = semaphore.value(value)?;
        let queue = unsafe { gpu.queue.as_hal::<Vulkan>() }.context("the engine queue is not Vulkan")?;
        queue.add_wait_semaphore(semaphore.raw, value, vk::PipelineStageFlags::TRANSFER);
    }
    gpu.queue.submit([enc.finish()]);
    Ok(())
}

/// Copies `texture` (a generated output, or an input to pass a real frame through) into `buffer`, then signals
/// `signal` when given, once this and every earlier submit of the engine is done.
pub fn copy_out(
    engine: &Engine,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    bytes_per_row: u32,
    signal: Option<(&Semaphore, u64)>,
) -> Result<()> {
    check_rows(texture, buffer, bytes_per_row)?;
    let gpu = engine.gpu();
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fh_vk_out") });
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer, layout: layout(bytes_per_row, texture.height()) },
        texture.size(),
    );
    if let Some((semaphore, value)) = signal {
        let value = semaphore.value(value)?;
        let queue = unsafe { gpu.queue.as_hal::<Vulkan>() }.context("the engine queue is not Vulkan")?;
        queue.add_signal_semaphore(semaphore.raw, value);
    }
    gpu.queue.submit([enc.finish()]);
    Ok(())
}

/// Signals `semaphore` once every submit so far is done, with nothing copied: for a frame the host shows without
/// anything from the engine, so its wait still pairs with a signal.
pub fn signal(engine: &Engine, semaphore: &Semaphore, value: u64) -> Result<()> {
    let value = semaphore.value(value)?;
    let gpu = engine.gpu();
    let queue = unsafe { gpu.queue.as_hal::<Vulkan>() }.context("the engine queue is not Vulkan")?;
    queue.add_signal_semaphore(semaphore.raw, value);
    let enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fh_vk_signal") });
    gpu.queue.submit([enc.finish()]);
    Ok(())
}
