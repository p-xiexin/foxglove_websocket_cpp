// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: foxglove/CubePrimitive.proto

#ifndef PROTOBUF_INCLUDED_foxglove_2fCubePrimitive_2eproto
#define PROTOBUF_INCLUDED_foxglove_2fCubePrimitive_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "foxglove/Color.pb.h"
#include "foxglove/Pose.pb.h"
#include "foxglove/Vector3.pb.h"
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fCubePrimitive_2eproto 

namespace protobuf_foxglove_2fCubePrimitive_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_foxglove_2fCubePrimitive_2eproto
namespace foxglove {
class CubePrimitive;
class CubePrimitiveDefaultTypeInternal;
extern CubePrimitiveDefaultTypeInternal _CubePrimitive_default_instance_;
}  // namespace foxglove
namespace google {
namespace protobuf {
template<> ::foxglove::CubePrimitive* Arena::CreateMaybeMessage<::foxglove::CubePrimitive>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace foxglove {

// ===================================================================

class CubePrimitive : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:foxglove.CubePrimitive) */ {
 public:
  CubePrimitive();
  virtual ~CubePrimitive();

  CubePrimitive(const CubePrimitive& from);

  inline CubePrimitive& operator=(const CubePrimitive& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CubePrimitive(CubePrimitive&& from) noexcept
    : CubePrimitive() {
    *this = ::std::move(from);
  }

  inline CubePrimitive& operator=(CubePrimitive&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const CubePrimitive& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CubePrimitive* internal_default_instance() {
    return reinterpret_cast<const CubePrimitive*>(
               &_CubePrimitive_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(CubePrimitive* other);
  friend void swap(CubePrimitive& a, CubePrimitive& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CubePrimitive* New() const final {
    return CreateMaybeMessage<CubePrimitive>(NULL);
  }

  CubePrimitive* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<CubePrimitive>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const CubePrimitive& from);
  void MergeFrom(const CubePrimitive& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(CubePrimitive* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .foxglove.Pose pose = 1;
  bool has_pose() const;
  void clear_pose();
  static const int kPoseFieldNumber = 1;
  private:
  const ::foxglove::Pose& _internal_pose() const;
  public:
  const ::foxglove::Pose& pose() const;
  ::foxglove::Pose* release_pose();
  ::foxglove::Pose* mutable_pose();
  void set_allocated_pose(::foxglove::Pose* pose);

  // .foxglove.Vector3 size = 2;
  bool has_size() const;
  void clear_size();
  static const int kSizeFieldNumber = 2;
  private:
  const ::foxglove::Vector3& _internal_size() const;
  public:
  const ::foxglove::Vector3& size() const;
  ::foxglove::Vector3* release_size();
  ::foxglove::Vector3* mutable_size();
  void set_allocated_size(::foxglove::Vector3* size);

  // .foxglove.Color color = 3;
  bool has_color() const;
  void clear_color();
  static const int kColorFieldNumber = 3;
  private:
  const ::foxglove::Color& _internal_color() const;
  public:
  const ::foxglove::Color& color() const;
  ::foxglove::Color* release_color();
  ::foxglove::Color* mutable_color();
  void set_allocated_color(::foxglove::Color* color);

  // @@protoc_insertion_point(class_scope:foxglove.CubePrimitive)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::foxglove::Pose* pose_;
  ::foxglove::Vector3* size_;
  ::foxglove::Color* color_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_foxglove_2fCubePrimitive_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CubePrimitive

// .foxglove.Pose pose = 1;
inline bool CubePrimitive::has_pose() const {
  return this != internal_default_instance() && pose_ != NULL;
}
inline const ::foxglove::Pose& CubePrimitive::_internal_pose() const {
  return *pose_;
}
inline const ::foxglove::Pose& CubePrimitive::pose() const {
  const ::foxglove::Pose* p = pose_;
  // @@protoc_insertion_point(field_get:foxglove.CubePrimitive.pose)
  return p != NULL ? *p : *reinterpret_cast<const ::foxglove::Pose*>(
      &::foxglove::_Pose_default_instance_);
}
inline ::foxglove::Pose* CubePrimitive::release_pose() {
  // @@protoc_insertion_point(field_release:foxglove.CubePrimitive.pose)
  
  ::foxglove::Pose* temp = pose_;
  pose_ = NULL;
  return temp;
}
inline ::foxglove::Pose* CubePrimitive::mutable_pose() {
  
  if (pose_ == NULL) {
    auto* p = CreateMaybeMessage<::foxglove::Pose>(GetArenaNoVirtual());
    pose_ = p;
  }
  // @@protoc_insertion_point(field_mutable:foxglove.CubePrimitive.pose)
  return pose_;
}
inline void CubePrimitive::set_allocated_pose(::foxglove::Pose* pose) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(pose_);
  }
  if (pose) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      pose = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, pose, submessage_arena);
    }
    
  } else {
    
  }
  pose_ = pose;
  // @@protoc_insertion_point(field_set_allocated:foxglove.CubePrimitive.pose)
}

// .foxglove.Vector3 size = 2;
inline bool CubePrimitive::has_size() const {
  return this != internal_default_instance() && size_ != NULL;
}
inline const ::foxglove::Vector3& CubePrimitive::_internal_size() const {
  return *size_;
}
inline const ::foxglove::Vector3& CubePrimitive::size() const {
  const ::foxglove::Vector3* p = size_;
  // @@protoc_insertion_point(field_get:foxglove.CubePrimitive.size)
  return p != NULL ? *p : *reinterpret_cast<const ::foxglove::Vector3*>(
      &::foxglove::_Vector3_default_instance_);
}
inline ::foxglove::Vector3* CubePrimitive::release_size() {
  // @@protoc_insertion_point(field_release:foxglove.CubePrimitive.size)
  
  ::foxglove::Vector3* temp = size_;
  size_ = NULL;
  return temp;
}
inline ::foxglove::Vector3* CubePrimitive::mutable_size() {
  
  if (size_ == NULL) {
    auto* p = CreateMaybeMessage<::foxglove::Vector3>(GetArenaNoVirtual());
    size_ = p;
  }
  // @@protoc_insertion_point(field_mutable:foxglove.CubePrimitive.size)
  return size_;
}
inline void CubePrimitive::set_allocated_size(::foxglove::Vector3* size) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(size_);
  }
  if (size) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      size = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, size, submessage_arena);
    }
    
  } else {
    
  }
  size_ = size;
  // @@protoc_insertion_point(field_set_allocated:foxglove.CubePrimitive.size)
}

// .foxglove.Color color = 3;
inline bool CubePrimitive::has_color() const {
  return this != internal_default_instance() && color_ != NULL;
}
inline const ::foxglove::Color& CubePrimitive::_internal_color() const {
  return *color_;
}
inline const ::foxglove::Color& CubePrimitive::color() const {
  const ::foxglove::Color* p = color_;
  // @@protoc_insertion_point(field_get:foxglove.CubePrimitive.color)
  return p != NULL ? *p : *reinterpret_cast<const ::foxglove::Color*>(
      &::foxglove::_Color_default_instance_);
}
inline ::foxglove::Color* CubePrimitive::release_color() {
  // @@protoc_insertion_point(field_release:foxglove.CubePrimitive.color)
  
  ::foxglove::Color* temp = color_;
  color_ = NULL;
  return temp;
}
inline ::foxglove::Color* CubePrimitive::mutable_color() {
  
  if (color_ == NULL) {
    auto* p = CreateMaybeMessage<::foxglove::Color>(GetArenaNoVirtual());
    color_ = p;
  }
  // @@protoc_insertion_point(field_mutable:foxglove.CubePrimitive.color)
  return color_;
}
inline void CubePrimitive::set_allocated_color(::foxglove::Color* color) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(color_);
  }
  if (color) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      color = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, color, submessage_arena);
    }
    
  } else {
    
  }
  color_ = color;
  // @@protoc_insertion_point(field_set_allocated:foxglove.CubePrimitive.color)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace foxglove

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_foxglove_2fCubePrimitive_2eproto
