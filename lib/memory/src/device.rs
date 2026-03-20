// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor};
use cudarc::driver::CudaContext;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};


#[derive(Clone)]
pub enum DeviceContext {
    Cuda(Arc<CudaContext>),
    Ze(Arc<ZeContext>),
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

impl DeviceContextProivder for DeviceStorage {
    fn device_context(&self) -> &Arc<DeviceContext> {
        // todo
    }
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
        let ctx = match backend {
            DeviceBackend::Cuda => DeviceContext::Cuda(cuda::Cuda::device_or_create(device_id)?),
            DeviceBackend::Ze => DeviceContext::Ze(Ze::device_or_create(device_id)?),
        };
        Ok(Self { ctx })
    }

    pub fn ctx(&self) -> Arc<DeviceContext> {
        match &self.ctx {
            DeviceContext::Cuda(ctx) => Arc::new(DeviceContext::Cuda(ctx.clone())),
            DeviceContext::Ze(ctx) => Arc::new(DeviceContext::Ze(ctx.clone())),
        }
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
