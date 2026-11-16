//! `fh_vk_*`, `fh_stream_copy_*`, `fh_session_copy_*`: frames from the host's own Vulkan device through exported
//! memory and semaphores (`vulkan_interop`). Windows and Linux; elsewhere every call fails with a message.
use crate::{guard, set_err};
use flyhigh_engine::stream::Stream;
use flyhigh_engine::{Engine, Session};

#[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
use flyhigh_engine::vulkan_interop::{self as vi, ExternalMemory, Semaphore};

#[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
const UNSUPPORTED: &str = "Vulkan interop is built for Windows and Linux only";

/// Exported memory: the handle (an NT handle on Windows, a file descriptor on Linux) and the exporter's allocation.
#[repr(C)]
pub struct FhVkMemory {
    pub handle: isize,
    pub allocation_size: u64,
    pub memory_type_index: u32,
}

pub struct FhVkBuffer(wgpu::Buffer);

pub struct FhVkSemaphore {
    #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
    inner: Semaphore,
}

macro_rules! live {
    ($p:expr) => {
        if $p.is_null() {
            set_err("null argument");
            return -1;
        }
    };
}

/// 1 when the engine's device takes memory and semaphores from another Vulkan device (then `uuid`, if not null,
/// receives its 16-byte deviceUUID to match the host's GPU), 0 when it cannot, -1 on error.
/// # Safety
/// `e` is a live handle; `uuid` is null or points to 16 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_engine_vulkan_interop(e: *const Engine, uuid: *mut u8) -> i32 {
    guard(-1, || {
        live!(e);
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            let e = unsafe { &*e };
            if !vi::supported(e) {
                return 0;
            }
            if !uuid.is_null() {
                let Some(id) = vi::device_uuid(e) else { return 0 };
                unsafe { std::ptr::copy_nonoverlapping(id.as_ptr(), uuid, 16) };
            }
            1
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = uuid;
            0
        }
    })
}

/// Bytes per row of an RGBA8 frame `width` wide in an interop buffer (a multiple of 256).
#[no_mangle]
pub extern "C" fn fh_vk_row_pitch(width: u32) -> u32 {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// Imports `size` bytes of exported memory as a buffer (a dedicated allocation for a buffer with TRANSFER_SRC |
/// TRANSFER_DST usage on the host). NULL on error. A file descriptor belongs to the engine once this succeeds.
/// # Safety
/// `e` is a live handle; `m` points to a valid `FhVkMemory`.
#[no_mangle]
pub unsafe extern "C" fn fh_vk_buffer_import(e: *const Engine, m: *const FhVkMemory, size: u64) -> *mut FhVkBuffer {
    guard(std::ptr::null_mut(), || {
        if e.is_null() || m.is_null() {
            set_err("null argument");
            return std::ptr::null_mut();
        }
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            let m = unsafe { &*m };
            let memory = ExternalMemory {
                handle: m.handle,
                allocation_size: m.allocation_size,
                memory_type_index: m.memory_type_index,
            };
            match vi::import_buffer(unsafe { &*e }, &memory, size) {
                Ok(b) => Box::into_raw(Box::new(FhVkBuffer(b))),
                Err(err) => {
                    set_err(format!("{err:#}"));
                    std::ptr::null_mut()
                }
            }
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = size;
            set_err(UNSUPPORTED);
            std::ptr::null_mut()
        }
    })
}

/// # Safety
/// `b` is null or a handle from `fh_vk_buffer_import` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_vk_buffer_destroy(b: *mut FhVkBuffer) {
    guard((), || {
        if !b.is_null() {
            drop(unsafe { Box::from_raw(b) });
        }
    })
}

/// Imports an exported semaphore; `timeline` 1 for a timeline semaphore, 0 for a binary one. NULL on error.
/// # Safety
/// `e` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_vk_semaphore_import(e: *const Engine, handle: isize, timeline: u32) -> *mut FhVkSemaphore {
    guard(std::ptr::null_mut(), || {
        if e.is_null() {
            set_err("null argument");
            return std::ptr::null_mut();
        }
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            match vi::import_semaphore(unsafe { &*e }, handle, timeline != 0) {
                Ok(inner) => Box::into_raw(Box::new(FhVkSemaphore { inner })),
                Err(err) => {
                    set_err(format!("{err:#}"));
                    std::ptr::null_mut()
                }
            }
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = (handle, timeline);
            set_err(UNSUPPORTED);
            std::ptr::null_mut()
        }
    })
}

/// Waits for the engine's queue to finish with the semaphore, then destroys it.
/// # Safety
/// `s` is null or a handle from `fh_vk_semaphore_import` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_vk_semaphore_destroy(s: *mut FhVkSemaphore) {
    guard((), || {
        if !s.is_null() {
            drop(unsafe { Box::from_raw(s) });
        }
    })
}

/// Signals `s` (at `value` for a timeline semaphore, 0 for a binary one) once every earlier submit of the engine is
/// done, with nothing copied. 0 / -1.
/// # Safety
/// `e` and `s` are live handles.
#[no_mangle]
pub unsafe extern "C" fn fh_vk_signal(e: *const Engine, s: *const FhVkSemaphore, value: u64) -> i32 {
    guard(-1, || {
        live!(e);
        live!(s);
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            match vi::signal(unsafe { &*e }, unsafe { &(*s).inner }, value) {
                Ok(()) => 0,
                Err(err) => {
                    set_err(format!("{err:#}"));
                    -1
                }
            }
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = value;
            set_err(UNSUPPORTED);
            -1
        }
    })
}

#[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
unsafe fn semaphore<'a>(s: *const FhVkSemaphore, value: u64) -> Option<(&'a Semaphore, u64)> {
    (!s.is_null()).then(|| (unsafe { &(*s).inner }, value))
}

#[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
fn status(r: anyhow::Result<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(err) => {
            set_err(format!("{err:#}"));
            -1
        }
    }
}

fn check_slot(slot: u32, count: usize, what: &str) -> bool {
    if slot as usize >= count {
        set_err(format!("bad {what} slot {slot} (there are {count})"));
        return false;
    }
    true
}

/// Copies a frame from `b` (rows of `bytes_per_row`) into input slot `slot`, after `wait` (null: no wait) reaches
/// `value` (0 for a binary semaphore). 0 / -1.
/// # Safety
/// `s`, `e` and `b` are live handles; `wait` is null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_copy_in(
    s: *mut Stream,
    e: *const Engine,
    slot: u32,
    b: *const FhVkBuffer,
    bytes_per_row: u32,
    wait: *const FhVkSemaphore,
    value: u64,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        live!(b);
        let s = unsafe { &*s };
        if !check_slot(slot, s.inputs(), "input") {
            return -1;
        }
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            let tex = s.input_texture(slot as usize);
            status(vi::copy_in(unsafe { &*e }, unsafe { &(*b).0 }, bytes_per_row, tex, unsafe {
                semaphore(wait, value)
            }))
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = (bytes_per_row, wait, value);
            set_err(UNSUPPORTED);
            -1
        }
    })
}

/// Copies output slot `out` into `b`, then signals `signal` (null: none) at `value` (0 for a binary semaphore) once
/// this and every earlier submit of the engine is done. 0 / -1.
/// # Safety
/// `s`, `e` and `b` are live handles; `signal` is null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_copy_out(
    s: *const Stream,
    e: *const Engine,
    out: u32,
    b: *const FhVkBuffer,
    bytes_per_row: u32,
    signal: *const FhVkSemaphore,
    value: u64,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        live!(b);
        let s = unsafe { &*s };
        if !check_slot(out, s.outputs(), "output") {
            return -1;
        }
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            let tex = s.output_texture(out as usize);
            status(vi::copy_out(unsafe { &*e }, tex, unsafe { &(*b).0 }, bytes_per_row, unsafe {
                semaphore(signal, value)
            }))
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = (bytes_per_row, signal, value);
            set_err(UNSUPPORTED);
            -1
        }
    })
}

/// `fh_stream_copy_in` for a session's input slot.
/// # Safety
/// As `fh_stream_copy_in`.
#[no_mangle]
pub unsafe extern "C" fn fh_session_copy_in(
    s: *mut Session,
    e: *const Engine,
    slot: u32,
    b: *const FhVkBuffer,
    bytes_per_row: u32,
    wait: *const FhVkSemaphore,
    value: u64,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        live!(b);
        let s = unsafe { &*s };
        if !check_slot(slot, s.inputs(), "input") {
            return -1;
        }
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            let tex = s.input_texture(slot as usize);
            status(vi::copy_in(unsafe { &*e }, unsafe { &(*b).0 }, bytes_per_row, tex, unsafe {
                semaphore(wait, value)
            }))
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = (bytes_per_row, wait, value);
            set_err(UNSUPPORTED);
            -1
        }
    })
}

/// `fh_stream_copy_out` for a session's output slot.
/// # Safety
/// As `fh_stream_copy_out`.
#[no_mangle]
pub unsafe extern "C" fn fh_session_copy_out(
    s: *const Session,
    e: *const Engine,
    out: u32,
    b: *const FhVkBuffer,
    bytes_per_row: u32,
    signal: *const FhVkSemaphore,
    value: u64,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        live!(b);
        let s = unsafe { &*s };
        if !check_slot(out, s.outputs(), "output") {
            return -1;
        }
        #[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
        {
            let tex = s.output_texture(out as usize);
            status(vi::copy_out(unsafe { &*e }, tex, unsafe { &(*b).0 }, bytes_per_row, unsafe {
                semaphore(signal, value)
            }))
        }
        #[cfg(not(all(any(windows, target_os = "linux"), not(target_arch = "wasm32"))))]
        {
            let _ = (bytes_per_row, signal, value);
            set_err(UNSUPPORTED);
            -1
        }
    })
}
