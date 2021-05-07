/*!
 *  Copyright (c) 2016 by Contributors
 * \file cpu_device_api.cc
 */
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/device_api.h>
#include <cstdlib>
#include <cstring>
#include "workspace_pool.h"

namespace dgl {
namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DGLContext ctx) final {}
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(DGLContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DGLType type_hint) final {
    void* ptr;
#if _MSC_VER || defined(__MINGW32__)
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG)
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
#ifndef _WIN32
    constexpr size_t _HugePage2MB_ = 1<<21;
    if (nbytes >= _HugePage2MB_) {
       int ret = posix_memalign(&ptr, _HugePage2MB_, nbytes);
       if (ret != 0) throw std::bad_alloc();
       if ((ret= madvise(ptr, nbytes, MADV_HUGEPAGE )) != 0 )
       throw std::bad_alloc();
    } else {
#endif
       int ret = posix_memalign(&ptr, alignment, nbytes);
       if (ret != 0) throw std::bad_alloc();
#ifndef _WIN32
    }
#endif
#endif
    return ptr;
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
#if _MSC_VER || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      DGLContext ctx_from,
                      DGLContext ctx_to,
                      DGLType type_hint,
                      DGLStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<const char*>(from) + from_offset,
           size);
  }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {
  }

  void* AllocWorkspace(DGLContext ctx, size_t size, DGLType type_hint) final;
  void FreeWorkspace(DGLContext ctx, void* data) final;

  static const std::shared_ptr<CPUDeviceAPI>& Global() {
    static std::shared_ptr<CPUDeviceAPI> inst =
        std::make_shared<CPUDeviceAPI>();
    return inst;
  }
};

struct CPUWorkspacePool : public WorkspacePool {
  CPUWorkspacePool() :
      WorkspacePool(kDLCPU, CPUDeviceAPI::Global()) {}
};

void* CPUDeviceAPI::AllocWorkspace(DGLContext ctx,
                                   size_t size,
                                   DGLType type_hint) {
  return dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()
      ->AllocWorkspace(ctx, size);
}

void CPUDeviceAPI::FreeWorkspace(DGLContext ctx, void* data) {
  dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

DGL_REGISTER_GLOBAL("device_api.cpu")
.set_body([](DGLArgs args, DGLRetValue* rv) {
    DeviceAPI* ptr = CPUDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace dgl
