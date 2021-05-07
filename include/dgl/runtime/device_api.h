/*!
 *  Copyright (c) 2016 by Contributors
 * \file dgl/runtime/device_api.h
 * \brief Abstract device memory management API
 */
#ifndef DGL_RUNTIME_DEVICE_API_H_
#define DGL_RUNTIME_DEVICE_API_H_

#include <string>
#include "packed_func.h"
#include "c_runtime_api.h"

namespace dgl {
namespace runtime {
/*!
 * \brief the query type into GetAttr
 */
enum DeviceAttrKind : int {
  kExist = 0,
  kMaxThreadsPerBlock = 1,
  kWarpSize = 2,
  kMaxSharedMemoryPerBlock = 3,
  kComputeVersion = 4,
  kDeviceName = 5,
  kMaxClockRate = 6,
  kMultiProcessorCount = 7,
  kMaxThreadDimensions = 8
};

/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 64;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 64;

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

/*!
 * \brief DGL Runtime Device API, abstracts the device
 *  specific interface for memory management.
 */
class DeviceAPI {
 public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {}
  /*!
   * \brief Set the environment device id to ctx
   * \param ctx The context to be set.
   */
  virtual void SetDevice(DGLContext ctx) = 0;
  /*!
   * \brief Get attribute of specified device.
   * \param ctx The device context
   * \param kind The result kind
   * \param rv The return value.
   * \sa DeviceAttrKind
   */
  virtual void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) = 0;
  /*!
   * \brief Allocate a data space on device.
   * \param ctx The device context to perform operation.
   * \param nbytes The number of bytes in memory.
   * \param alignment The alignment of the memory.
   * \param type_hint The type of elements. Only needed by certain backends such
   * as OpenGL, as nbytes & alignment are sufficient for most backends.
   * \return The allocated device pointer.
   */
  virtual void* AllocDataSpace(DGLContext ctx,
                               size_t nbytes,
                               size_t alignment,
                               DGLType type_hint) = 0;
  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   */
  virtual void FreeDataSpace(DGLContext ctx, void* ptr) = 0;
  /*!
   * \brief copy data from one place to another
   * \param from The source array.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param num_bytes The size of the memory in bytes
   * \param ctx_from The source context
   * \param ctx_to The target context
   * \param type_hint The type of elements, only neded by certain backends.
   *                  can be useful for cross device endian converison.
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(const void* from,
                              size_t from_offset,
                              void* to,
                              size_t to_offset,
                              size_t num_bytes,
                              DGLContext ctx_from,
                              DGLContext ctx_to,
                              DGLType type_hint,
                              DGLStreamHandle stream) = 0;
    /*!
   * \brief Create a new stream of execution.
   *
   * \param ctx The context of allocation.
   */
  DGL_DLL virtual DGLStreamHandle CreateStream(DGLContext ctx);

  /*!
   * \brief Free a stream of execution
   *
   * \param ctx The context of the stream
   * \param stream The pointer to be freed.
   */
  DGL_DLL virtual void FreeStream(DGLContext ctx, DGLStreamHandle stream);

  /*!
   * \brief Synchronize the stream
   * \param ctx The context to perform operation.
   * \param stream The stream to be sync.
   */
  virtual void StreamSync(DGLContext ctx, DGLStreamHandle stream) = 0;
  /*!
   * \brief Set the stream
   * \param ctx The context to set stream.
   * \param stream The stream to be set.
   */
  virtual void SetStream(DGLContext ctx, DGLStreamHandle stream) {}
  /**
   * \brief Get the underlying stream.
   *
   * \param ctx The context to get the stream of.
   *
   * \return The stream.
   */
  virtual DGLStreamHandle GetStream(DGLContext ctx) = 0;
  /*!
   * \brief Synchronize 2 streams of execution.
   *
   * An event is created in event_src stream that the second then
   * stream waits on.  Neither event_src or event_dst need to be of
   * the same device ID as the context, but they must be of the same
   * device type.
   *
   * \param ctx The context of the streams.
   * \param event_src The source stream to synchronize.
   * \param event_dst The destination stream to synchronize.
   */
  DGL_DLL virtual void SyncStreamFromTo(DGLContext ctx,
                                        DGLStreamHandle event_src,
                                        DGLStreamHandle event_dst);
  /*!
   * \brief Allocate temporal workspace for backend execution.
   *
   *  \note We have the following assumption about backend temporal
   *   workspace allocation, and backend will optimize for such assumption:
   *
   *  - Only a few allocation will happen, and space will be released after use.
   *  - The release order is usually in reverse order of allocate (stack style).
   *  - Repeative pattern of same allocations over different runs.
   *  - Workspace should not overlap between different threads(i.e. be threadlocal)
   *
   * \param ctx The context of allocation.
   * \param nbytes The size to be allocated.
   * \param type_hint The type of elements. Only needed by certain backends such
   * as OpenGL, as nbytes is sufficient for most backends.
   */
  DGL_DLL virtual void* AllocWorkspace(DGLContext ctx,
                                       size_t nbytes,
                                       DGLType type_hint = {});
  /*!
   * \brief Free temporal workspace in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  DGL_DLL virtual void FreeWorkspace(DGLContext ctx, void* ptr);

  /*!
   * \brief Get device API base don context.
   * \param ctx The context
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
  DGL_DLL static DeviceAPI* Get(DGLContext ctx, bool allow_missing = false);
};

/*! \brief The device type bigger than this is RPC device */
constexpr int kRPCSessMask = 128;
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_DEVICE_API_H_
