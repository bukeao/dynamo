// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # CUDA Storage Support
//!
//! This module provides CUDA-specific storage implementations for the block manager.
//! It is conditionally compiled based on the `cuda` feature flag.
//!
//! ## Features
//!
//! The following types are available when the `cuda` feature is enabled:
//! - [`PinnedStorage`] - Page-locked host memory for efficient GPU transfers
//! - [`DeviceStorage`] - Direct GPU memory allocation
//!
//! ## Storage Allocators
//!
//! The module provides allocators for each storage type:
//! - [`PinnedAllocator`] - Creates pinned host memory allocations
//! - [`DeviceAllocator`] - Creates device memory allocations
//!
//! ## CUDA Context Management
//!
//! The module provides a singleton [`Cuda`] type for managing CUDA contexts:
//! - Thread-safe context management
//! - Lazy initialization of device contexts
//! - Automatic cleanup of resources
//!
//! ## Usage
//!
//! ### Using Allocators
//! ```rust,ignore
//! use dynamo_llm::block_manager::storage::{DeviceAllocator, PinnedAllocator, StorageAllocator};
//!
//! // Create a pinned memory allocator
//! let pinned_allocator = PinnedAllocator::default();
//! let pinned_storage = pinned_allocator.allocate(1024).unwrap();
//!
//! // Create a device memory allocator for a specific device
//! let device_allocator = DeviceAllocator::new(1).unwrap();  // Use device 1
//! let device_storage = device_allocator.allocate(1024).unwrap();
//! ```
//!
//! ### Memory Operations
//! ```rust,ignore
//! use dynamo_llm::block_manager::storage::{
//!     PinnedAllocator, StorageAllocator, Storage, StorageMemset
//! };
//!
//! // Initialize memory
//! let mut storage = PinnedAllocator::default().allocate(1024).unwrap();
//!
//! // Initialize memory
//! storage.memset(0, 0, 1024).unwrap();
//!
//! // Access memory through raw pointers (requires unsafe)
//! unsafe {
//!     let ptr = storage.as_mut_ptr();
//!     // Use the pointer...
//! }
//! ```
//!
//! ## Safety
//!
//! All CUDA operations are wrapped in safe Rust interfaces that ensure:
//! - Proper resource cleanup
//! - Thread safety
//! - Memory alignment requirements
//! - Error handling for CUDA operations

use super::*;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::CudaContext;
use dynamo_memory::MemoryDescriptor as _;

/// Singleton for managing CUDA contexts.
pub struct Cuda {
    contexts: HashMap<usize, Arc<CudaContext>>,
}

impl Cuda {
    // Private constructor
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Get a CUDA context for a specific device_id.
    /// If the context does not exist, it will return None.
    ///
    /// This will not lazily instantiate a context for a device. Use
    /// [Cuda::device_or_create]
    pub fn device(device_id: usize) -> Option<Arc<CudaContext>> {
        Cuda::instance()
            .lock()
            .unwrap()
            .get_existing_context(device_id)
    }

    /// Get or initialize a CUDA context for a specific device_id.
    /// If the context does not exist, it will be created or fail.
    ///
    /// This will lazily instantiate a context for a device. Use
    /// [CudaContextManager::device] to get an existing context.
    pub fn device_or_create(device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
        Cuda::instance().lock().unwrap().get_context(device_id)
    }

    /// Check if a CUDA context exists for a specific device_id.
    pub fn is_initialized(device_id: usize) -> bool {
        Cuda::instance().lock().unwrap().has_context(device_id)
    }

    // Get the singleton instance
    fn instance() -> &'static Mutex<Cuda> {
        static INSTANCE: OnceLock<Mutex<Cuda>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(Cuda::new()))
    }

    // Get or create a CUDA context for a specific device
    fn get_context(&mut self, device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
        // Check if we already have a context for this device
        if let Some(ctx) = self.contexts.get(&device_id) {
            return Ok(ctx.clone());
        }

        // Create a new context for this device
        let ctx = CudaContext::new(device_id)?;

        // Store the context
        self.contexts.insert(device_id, ctx.clone());

        Ok(ctx)
    }

    // Get a context if it exists, but don't create one
    pub fn get_existing_context(&self, device_id: usize) -> Option<Arc<CudaContext>> {
        self.contexts.get(&device_id).cloned()
    }

    // Check if a context exists for a device
    pub fn has_context(&self, device_id: usize) -> bool {
        self.contexts.contains_key(&device_id)
    }
}

/// Pinned host memory storage using CUDA page-locked memory.
/// Wraps [`dynamo_memory::PinnedStorage`] and adds registration handle support.
#[derive(Debug)]
pub struct PinnedStorage {
    inner: dynamo_memory::PinnedStorage,
    handles: RegistrationHandles,
}

impl Local for PinnedStorage {}
impl SystemAccessible for PinnedStorage {}
impl CudaAccessible for PinnedStorage {}

impl PinnedStorage {
    /// Create a new pinned storage with the given size.
    ///
    /// Uses write-combined allocation with NUMA-awareness when enabled.
    /// Prefer [`new_for_device`](Self::new_for_device) for new code.
    ///
    /// TODO(KVBM-336): remove PinnedStorage::new in the future
    #[deprecated(since = "1.0.0", note = "Use PinnedStorage::new_for_device instead")]
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, StorageError> {
        let inner =
            dynamo_memory::PinnedStorage::new_for_device(size, Some(ctx.cu_device() as u32))?;
        Ok(Self {
            inner,
            handles: RegistrationHandles::new(),
        })
    }

    /// Create a new pinned storage, optionally NUMA-aware for a specific GPU.
    ///
    /// Delegates NUMA-aware allocation and write-combined selection to
    /// [`dynamo_memory::PinnedStorage::new_for_device`].
    ///
    /// When `device_id` is `None`, allocates on device 0 without NUMA awareness.
    pub fn new_for_device(size: usize, device_id: Option<u32>) -> Result<Self, StorageError> {
        // Warn once if the legacy opt-in env var is still set.
        static DEPRECATION_WARN: std::sync::Once = std::sync::Once::new();
        if std::env::var("DYN_KVBM_ENABLE_NUMA").is_ok() {
            DEPRECATION_WARN.call_once(|| {
                tracing::warn!(
                    "DYN_KVBM_ENABLE_NUMA is deprecated for PinnedStorage::new_for_device; \
                     NUMA is now enabled by default. Use DYN_MEMORY_DISABLE_NUMA=1 to disable."
                );
            });
        }
        let inner = dynamo_memory::PinnedStorage::new_for_device(size, device_id)?;
        Ok(Self {
            inner,
            handles: RegistrationHandles::new(),
        })
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        self.handles.release();
        // inner Drop handles free_host
    }
}
impl super::StorageBackendOps for std::sync::Arc<CudaContext> {
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8, super::StorageError> {
        self.bind_to_thread()
            .map_err(super::StorageError::Cuda)?;

        // Try NUMA-aware allocation if enabled, otherwise use direct allocation.
        if crate::block_manager::numa_allocator::is_numa_enabled() {
            let device_id = self.cu_device() as u32;
            match crate::block_manager::numa_allocator::worker_pool::NumaWorkerPool::global()
                .allocate_pinned_for_gpu(size, device_id)
            {
                Ok(ptr) => return Ok(ptr),
                Err(e) => {
                    tracing::warn!("NUMA allocation failed: {}, using direct allocation", e);
                }
            }
        }

        malloc_host_prefer_writecombined(size)
    }

    unsafe fn free_pinned(&self, ptr: u64, _size: usize) -> Result<(), super::StorageError> {
        cudarc::driver::result::free_host(ptr as _).map_err(super::StorageError::Cuda)
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, super::DeviceStorageType), super::StorageError> {
        self.bind_to_thread()
            .map_err(super::StorageError::Cuda)?;

        let ptr = cudarc::driver::result::malloc_sync(size).map_err(super::StorageError::Cuda)?;

        Ok((
            ptr,
            self.cu_device() as u32,
            super::DeviceStorageType::Owned {
                _ze_device_buffer: None,
            },
        ))
    }

    unsafe fn free_device(&self, ptr: u64) -> Result<(), super::StorageError> {
        cudarc::driver::result::free_sync(ptr as _).map_err(super::StorageError::Cuda)
    }

    fn device_id(&self) -> u32 {
        self.cu_device() as u32
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct MockTensor {
        device: TorchDevice,
        data_ptr: u64,
        size_bytes: usize,
    }

    impl MockTensor {
        pub fn new(device: TorchDevice, data_ptr: u64, size_bytes: usize) -> Self {
            Self {
                device,
                data_ptr,
                size_bytes,
            }
        }
    }

    impl TorchTensor for MockTensor {
        fn device(&self) -> TorchDevice {
            self.device.clone()
        }

        fn data_ptr(&self) -> u64 {
            self.data_ptr
        }

        fn size_bytes(&self) -> usize {
            self.size_bytes
        }

        fn shape(&self) -> Vec<usize> {
            vec![self.size_bytes]
        }

        fn stride(&self) -> Vec<usize> {
            vec![1]
        }
    }

    #[test]
    fn test_device_storage_from_torch_valid_tensor() {
        let ctx = Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let size_bytes = 1024;

        let actual_storage =
            std::mem::ManuallyDrop::new(DeviceStorage::new(&ctx, size_bytes).unwrap());

        let tensor = MockTensor::new(TorchDevice::Cuda(0), actual_storage.addr(), size_bytes);

        let storage = DeviceStorage::new_from_torch(&ctx, Arc::new(tensor)).unwrap();

        assert_eq!(storage.size(), size_bytes);
        assert_eq!(storage.storage_type(), StorageType::Device(0));
        assert_eq!(storage.addr(), actual_storage.addr());
    }

    #[test]
    fn test_device_storage_from_torch_cpu_tensor_fails() {
        let ctx = Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let size_bytes = 1024;

        let actual_storage = DeviceStorage::new(&ctx, size_bytes).unwrap();

        let tensor = MockTensor::new(
            TorchDevice::Other("cpu".to_string()),
            actual_storage.addr(),
            size_bytes,
        );

        let result = DeviceStorage::new_from_torch(&ctx, Arc::new(tensor));
        assert!(result.is_err());

        if let Err(StorageError::InvalidConfig(msg)) = result {
            assert!(msg.contains("Tensor is not CUDA"));
        } else {
            panic!("Expected InvalidConfig error for CPU tensor");
        }
    }

    #[test]
    fn test_device_storage_wrong_device() {
        let ctx = Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let size_bytes = 1024;

        let actual_storage = DeviceStorage::new(&ctx, size_bytes).unwrap();

        let tensor = MockTensor::new(TorchDevice::Cuda(1), actual_storage.addr(), size_bytes);

        let result = DeviceStorage::new_from_torch(&ctx, Arc::new(tensor));
        assert!(result.is_err());
    }

    /// Test PinnedStorage::new (deprecated) allocates usable pinned memory.
    #[allow(deprecated)]
    #[test]
    fn test_pinned_storage_new_without_numa() {
        let ctx = Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let size = 8192;

        let mut storage =
            PinnedStorage::new(&ctx, size).expect("PinnedStorage::new should succeed");

        // Verify storage properties
        assert_eq!(storage.size(), size);
        assert_eq!(storage.storage_type(), StorageType::Pinned);
        assert_ne!(storage.addr(), 0, "Address should be non-zero");

        // Verify memory is accessible
        unsafe {
            let ptr = storage.as_mut_ptr();
            assert!(!ptr.is_null(), "Pointer should not be null");

            // Write a pattern to verify memory is usable
            for i in 0..size {
                std::ptr::write_volatile(ptr.add(i), (i & 0xFF) as u8);
            }

            // Read back and verify
            for i in 0..size {
                let val = std::ptr::read_volatile(ptr.add(i));
                assert_eq!(
                    val,
                    (i & 0xFF) as u8,
                    "Memory content mismatch at offset {}",
                    i
                );
            }
        }
    }
}
