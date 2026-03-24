// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor};
use cudarc::driver::CudaContext;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// Level Zero support
use level_zero;

/// Ze context wrapper
pub struct ZeContext {
    pub device_id: u32,
    pub driver: level_zero::Driver,
    pub device: level_zero::Device,
    pub context: Arc<level_zero::Context>,
}

unsafe impl Send for ZeContext {}
unsafe impl Sync for ZeContext {}

#[derive(Clone)]
pub enum DeviceContext {
    Cuda(Arc<CudaContext>),
    Ze(Arc<ZeContext>),
}

impl DeviceContext {
    /// Get the backend type of this context
    pub fn backend(&self) -> DeviceBackend {
        match self {
            Self::Cuda(_) => DeviceBackend::Cuda,
            Self::Ze(_) => DeviceBackend::Ze,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceBackend {
    Cuda,
    Ze,
}

/// Trait for types that can provide a device context.
pub trait DeviceContextProvider {
    /// Get a reference-counted device context (CUDA or ZE).
    fn device_context(&self) -> Arc<DeviceContext>;
}

/// Allocator for DeviceStorage
pub struct DeviceAllocator {
    ctx: Arc<DeviceContext>,
}

impl Default for DeviceAllocator {
    fn default() -> Self {
        Self::new(0, DeviceBackend::Cuda).expect("Failed to create CUDA context")
    }
}


impl DeviceAllocator {
    /// Create a new device allocator
    pub fn new(device_id: usize, backend: DeviceBackend) -> Result<Self, StorageError> {
        let ctx = Arc::new(match backend {
            DeviceBackend::Cuda => DeviceContext::Cuda(cuda_context(device_id as u32)?),
            DeviceBackend::Ze => DeviceContext::Ze(ze_context(device_id as u32)?),
        });
        Ok(Self { ctx })
    }

    pub fn ctx(&self) -> Arc<DeviceContext> {
        self.ctx.clone()
    }
}

impl StorageAllocator<DeviceStorage> for DeviceAllocator {
    fn allocate(&self, size: usize) -> Result<DeviceStorage, StorageError> {
        DeviceStorage::new(&self.ctx, size)
    }
}

/// An enum indicating the type of device storage.
/// This is needed to ensure ownership of memory is correctly handled.
/// When building a [`DeviceStorage`] from a torch tensor, we need to ensure that
/// the torch tensor is not GCed until the [`DeviceStorage`] is dropped.
/// Because of this, we need to store a reference to the torch tensor in the [`DeviceStorage`]
#[derive(Debug)]
enum DeviceStorageType {
    Owned,                                   // Memory that we allocated ourselves.
    Torch { _tensor: Arc<dyn TorchTensor> }, // Memory that came from a torch tensor.
}

/// Get or create a CUDA context for the given device.
pub(crate) fn cuda_context(device_id: u32) -> Result<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<u32, Arc<CudaContext>>>> = OnceLock::new();
    let mut map = CONTEXTS.get_or_init(Default::default).lock().unwrap();

    if let Some(existing) = map.get(&device_id) {
        return Ok(existing.clone());
    }

    let ctx = CudaContext::new(device_id as usize)?;
    map.insert(device_id, ctx.clone());
    Ok(ctx)
}

/// Get or create a Ze context for the given device.
pub(crate) fn ze_context(device_id: u32) -> Result<Arc<ZeContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<u32, Arc<ZeContext>>>> = OnceLock::new();
    let mut map = CONTEXTS.get_or_init(Default::default).lock().unwrap();

    if let Some(existing) = map.get(&device_id) {
        return Ok(existing.clone());
    }

    // Initialize Level Zero
    level_zero::init()
        .map_err(|e| StorageError::OperationFailed(format!("Ze init failed: {:?}", e)))?;

    let drivers = level_zero::drivers()
        .map_err(|e| StorageError::OperationFailed(format!("Ze drivers failed: {:?}", e)))?;

    let driver = drivers.first().copied()
        .ok_or_else(|| StorageError::OperationFailed("No Ze driver found".into()))?;

    let devices = driver.devices()
        .map_err(|e| StorageError::OperationFailed(format!("Ze devices failed: {:?}", e)))?;

    let device = devices.get(device_id as usize).copied()
        .ok_or_else(|| StorageError::OperationFailed(format!("Ze device {} not found", device_id)))?;

    let context = Arc::new(level_zero::Context::create(&driver)
        .map_err(|e| StorageError::OperationFailed(format!("Ze context creation failed: {:?}", e)))?);

    let ctx = Arc::new(ZeContext {
        device_id,
        driver,
        device,
        context,
    });

    map.insert(device_id, ctx.clone());
    Ok(ctx)
}

/// CUDA device memory allocated via cudaMalloc.
#[derive(Debug)]
pub struct DeviceStorage {
    /// CUDA context used for allocation and deallocation.
    ctx: Arc<CudaContext>,
    /// Device pointer to the allocated memory.
    ptr: u64,
    /// CUDA device ID where memory is allocated.
    device_id: u32,
    /// Size of the allocation in bytes.
    len: usize,
}

unsafe impl Send for DeviceStorage {}
unsafe impl Sync for DeviceStorage {}

impl DeviceStorage {
    /// Allocate new device memory of the given size.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `device_id` - CUDA device on which to allocate
    pub fn new(len: usize, device_id: u32) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let ctx = cuda_context(device_id)?;
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let ptr = unsafe { cudarc::driver::result::malloc_sync(len).map_err(StorageError::Cuda)? };

        Ok(Self {
            ctx,
            ptr,
            device_id,
            len,
        })
    }

    /// Get the device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the CUDA device ID this memory is allocated on.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::debug!("failed to bind CUDA context for free: {e}");
        }
        unsafe {
            if let Err(e) = cudarc::driver::result::free_sync(self.ptr) {
                tracing::debug!("failed to free device memory: {e}");
            }
        };
    }
}

impl MemoryDescriptor for DeviceStorage {
    fn addr(&self) -> usize {
        self.device_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Device(self.device_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for DeviceStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Vram,
            self.device_id as u64,
        )
    }
}

impl StorageBackendOps for DeviceContext {
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8, StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::alloc_pinned(ctx, size),
            Self::Ze(ctx) => StorageBackendOps::alloc_pinned(ctx, size),
        }
    }

    unsafe fn free_pinned(&self, ptr: u64, size: usize) -> Result<(), StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::free_pinned(ctx, ptr, size),
            Self::Ze(ctx) => StorageBackendOps::free_pinned(ctx, ptr, size),
        }
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, DeviceStorageType), StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::alloc_device(ctx, size),
            Self::Ze(ctx) => StorageBackendOps::alloc_device(ctx, size),
        }
    }

    unsafe fn free_device(&self, ptr: u64) -> Result<(), StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::free_device(ctx, ptr),
            Self::Ze(ctx) => StorageBackendOps::free_device(ctx, ptr),
        }
    }

    fn device_id(&self) -> u32 {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::device_id(ctx),
            Self::Ze(ctx) => StorageBackendOps::device_id(ctx),
        }
    }
}



impl super::StorageBackendOps for std::sync::Arc<CudaContext> {
    unsafe fn alloc_pinned(&self, len: usize, device_id: Option<u32>) -> Result<super::PinnedStorage, super::StorageError> {
        if len == 0 {
            return Err(super::StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let gpu_id = device_id.unwrap_or(0);
        let ctx = crate::device::cuda_context(gpu_id)?;

        // Try NUMA-aware allocation unless explicitly disabled
        #[cfg(target_os = "linux")]
        let numa_ptr = if let Some(gpu_id) = device_id {
            if !super::numa::is_numa_disabled() {
                match super::numa::worker_pool::NumaWorkerPool::global()
                    .allocate_pinned_for_gpu(len, gpu_id)
                {
                    Ok(Some(ptr)) => {
                        tracing::debug!(
                            "Using NUMA-aware allocation for {} bytes on GPU {}",
                            len,
                            gpu_id
                        );
                        Some(ptr as usize)
                    }
                    Ok(None) => None, // NUMA node unknown, fall through
                    Err(e) => return Err(super::StorageError::AllocationFailed(e)),
                }
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(not(target_os = "linux"))]
        let numa_ptr: Option<usize> = None;

        let ptr = if let Some(ptr) = numa_ptr {
            ptr
        } else {
            ctx.bind_to_thread().map_err(super::StorageError::Cuda)?;

            let ptr = unsafe { super::pinned::malloc_host_prefer_writecombined(len)? };

            assert!(!ptr.is_null(), "Failed to allocate pinned memory");
            assert!(len < isize::MAX as usize);

            ptr as usize
        };

        Ok(super::PinnedStorage::from_raw_parts(ptr, len, Arc::new(DeviceContext::Cuda(ctx))))
    }

    unsafe fn free_pinned(&self, ptr: u64, _size: usize) -> Result<(), super::StorageError> {
        unsafe { cudarc::driver::result::free_host(ptr as _) }.map_err(super::StorageError::Cuda)
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, super::DeviceStorageType), super::StorageError> {
        self.bind_to_thread()
            .map_err(super::StorageError::Cuda)?;

        let ptr = unsafe { cudarc::driver::result::malloc_sync(size) }.map_err(super::StorageError::Cuda)?;

        Ok((
            ptr,
            self.cu_device() as u32,
            super::DeviceStorageType::Owned {
                _ze_device_buffer: None,
            },
        ))
    }

    unsafe fn free_device(&self, ptr: u64) -> Result<(), super::StorageError> {
        unsafe { cudarc::driver::result::free_sync(ptr as _) }.map_err(super::StorageError::Cuda)
    }

    fn device_id(&self) -> u32 {
        self.cu_device() as u32
    }
}