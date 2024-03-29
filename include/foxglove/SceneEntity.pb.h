// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: foxglove/SceneEntity.proto

#ifndef PROTOBUF_INCLUDED_foxglove_2fSceneEntity_2eproto
#define PROTOBUF_INCLUDED_foxglove_2fSceneEntity_2eproto

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
#include "foxglove/ArrowPrimitive.pb.h"
#include "foxglove/CubePrimitive.pb.h"
#include "foxglove/CylinderPrimitive.pb.h"
#include "foxglove/KeyValuePair.pb.h"
#include "foxglove/LinePrimitive.pb.h"
#include "foxglove/ModelPrimitive.pb.h"
#include "foxglove/SpherePrimitive.pb.h"
#include "foxglove/TextPrimitive.pb.h"
#include "foxglove/TriangleListPrimitive.pb.h"
#include <google/protobuf/duration.pb.h>
#include <google/protobuf/timestamp.pb.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fSceneEntity_2eproto 

namespace protobuf_foxglove_2fSceneEntity_2eproto {
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
}  // namespace protobuf_foxglove_2fSceneEntity_2eproto
namespace foxglove {
class SceneEntity;
class SceneEntityDefaultTypeInternal;
extern SceneEntityDefaultTypeInternal _SceneEntity_default_instance_;
}  // namespace foxglove
namespace google {
namespace protobuf {
template<> ::foxglove::SceneEntity* Arena::CreateMaybeMessage<::foxglove::SceneEntity>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace foxglove {

// ===================================================================

class SceneEntity : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:foxglove.SceneEntity) */ {
 public:
  SceneEntity();
  virtual ~SceneEntity();

  SceneEntity(const SceneEntity& from);

  inline SceneEntity& operator=(const SceneEntity& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  SceneEntity(SceneEntity&& from) noexcept
    : SceneEntity() {
    *this = ::std::move(from);
  }

  inline SceneEntity& operator=(SceneEntity&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const SceneEntity& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const SceneEntity* internal_default_instance() {
    return reinterpret_cast<const SceneEntity*>(
               &_SceneEntity_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(SceneEntity* other);
  friend void swap(SceneEntity& a, SceneEntity& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline SceneEntity* New() const final {
    return CreateMaybeMessage<SceneEntity>(NULL);
  }

  SceneEntity* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<SceneEntity>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const SceneEntity& from);
  void MergeFrom(const SceneEntity& from);
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
  void InternalSwap(SceneEntity* other);
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

  // repeated .foxglove.KeyValuePair metadata = 6;
  int metadata_size() const;
  void clear_metadata();
  static const int kMetadataFieldNumber = 6;
  ::foxglove::KeyValuePair* mutable_metadata(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::KeyValuePair >*
      mutable_metadata();
  const ::foxglove::KeyValuePair& metadata(int index) const;
  ::foxglove::KeyValuePair* add_metadata();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::KeyValuePair >&
      metadata() const;

  // repeated .foxglove.ArrowPrimitive arrows = 7;
  int arrows_size() const;
  void clear_arrows();
  static const int kArrowsFieldNumber = 7;
  ::foxglove::ArrowPrimitive* mutable_arrows(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::ArrowPrimitive >*
      mutable_arrows();
  const ::foxglove::ArrowPrimitive& arrows(int index) const;
  ::foxglove::ArrowPrimitive* add_arrows();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::ArrowPrimitive >&
      arrows() const;

  // repeated .foxglove.CubePrimitive cubes = 8;
  int cubes_size() const;
  void clear_cubes();
  static const int kCubesFieldNumber = 8;
  ::foxglove::CubePrimitive* mutable_cubes(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::CubePrimitive >*
      mutable_cubes();
  const ::foxglove::CubePrimitive& cubes(int index) const;
  ::foxglove::CubePrimitive* add_cubes();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::CubePrimitive >&
      cubes() const;

  // repeated .foxglove.SpherePrimitive spheres = 9;
  int spheres_size() const;
  void clear_spheres();
  static const int kSpheresFieldNumber = 9;
  ::foxglove::SpherePrimitive* mutable_spheres(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::SpherePrimitive >*
      mutable_spheres();
  const ::foxglove::SpherePrimitive& spheres(int index) const;
  ::foxglove::SpherePrimitive* add_spheres();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::SpherePrimitive >&
      spheres() const;

  // repeated .foxglove.CylinderPrimitive cylinders = 10;
  int cylinders_size() const;
  void clear_cylinders();
  static const int kCylindersFieldNumber = 10;
  ::foxglove::CylinderPrimitive* mutable_cylinders(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::CylinderPrimitive >*
      mutable_cylinders();
  const ::foxglove::CylinderPrimitive& cylinders(int index) const;
  ::foxglove::CylinderPrimitive* add_cylinders();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::CylinderPrimitive >&
      cylinders() const;

  // repeated .foxglove.LinePrimitive lines = 11;
  int lines_size() const;
  void clear_lines();
  static const int kLinesFieldNumber = 11;
  ::foxglove::LinePrimitive* mutable_lines(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::LinePrimitive >*
      mutable_lines();
  const ::foxglove::LinePrimitive& lines(int index) const;
  ::foxglove::LinePrimitive* add_lines();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::LinePrimitive >&
      lines() const;

  // repeated .foxglove.TriangleListPrimitive triangles = 12;
  int triangles_size() const;
  void clear_triangles();
  static const int kTrianglesFieldNumber = 12;
  ::foxglove::TriangleListPrimitive* mutable_triangles(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::TriangleListPrimitive >*
      mutable_triangles();
  const ::foxglove::TriangleListPrimitive& triangles(int index) const;
  ::foxglove::TriangleListPrimitive* add_triangles();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::TriangleListPrimitive >&
      triangles() const;

  // repeated .foxglove.TextPrimitive texts = 13;
  int texts_size() const;
  void clear_texts();
  static const int kTextsFieldNumber = 13;
  ::foxglove::TextPrimitive* mutable_texts(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::TextPrimitive >*
      mutable_texts();
  const ::foxglove::TextPrimitive& texts(int index) const;
  ::foxglove::TextPrimitive* add_texts();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::TextPrimitive >&
      texts() const;

  // repeated .foxglove.ModelPrimitive models = 14;
  int models_size() const;
  void clear_models();
  static const int kModelsFieldNumber = 14;
  ::foxglove::ModelPrimitive* mutable_models(int index);
  ::google::protobuf::RepeatedPtrField< ::foxglove::ModelPrimitive >*
      mutable_models();
  const ::foxglove::ModelPrimitive& models(int index) const;
  ::foxglove::ModelPrimitive* add_models();
  const ::google::protobuf::RepeatedPtrField< ::foxglove::ModelPrimitive >&
      models() const;

  // string frame_id = 2;
  void clear_frame_id();
  static const int kFrameIdFieldNumber = 2;
  const ::std::string& frame_id() const;
  void set_frame_id(const ::std::string& value);
  #if LANG_CXX11
  void set_frame_id(::std::string&& value);
  #endif
  void set_frame_id(const char* value);
  void set_frame_id(const char* value, size_t size);
  ::std::string* mutable_frame_id();
  ::std::string* release_frame_id();
  void set_allocated_frame_id(::std::string* frame_id);

  // string id = 3;
  void clear_id();
  static const int kIdFieldNumber = 3;
  const ::std::string& id() const;
  void set_id(const ::std::string& value);
  #if LANG_CXX11
  void set_id(::std::string&& value);
  #endif
  void set_id(const char* value);
  void set_id(const char* value, size_t size);
  ::std::string* mutable_id();
  ::std::string* release_id();
  void set_allocated_id(::std::string* id);

  // .google.protobuf.Timestamp timestamp = 1;
  bool has_timestamp() const;
  void clear_timestamp();
  static const int kTimestampFieldNumber = 1;
  private:
  const ::google::protobuf::Timestamp& _internal_timestamp() const;
  public:
  const ::google::protobuf::Timestamp& timestamp() const;
  ::google::protobuf::Timestamp* release_timestamp();
  ::google::protobuf::Timestamp* mutable_timestamp();
  void set_allocated_timestamp(::google::protobuf::Timestamp* timestamp);

  // .google.protobuf.Duration lifetime = 4;
  bool has_lifetime() const;
  void clear_lifetime();
  static const int kLifetimeFieldNumber = 4;
  private:
  const ::google::protobuf::Duration& _internal_lifetime() const;
  public:
  const ::google::protobuf::Duration& lifetime() const;
  ::google::protobuf::Duration* release_lifetime();
  ::google::protobuf::Duration* mutable_lifetime();
  void set_allocated_lifetime(::google::protobuf::Duration* lifetime);

  // bool frame_locked = 5;
  void clear_frame_locked();
  static const int kFrameLockedFieldNumber = 5;
  bool frame_locked() const;
  void set_frame_locked(bool value);

  // @@protoc_insertion_point(class_scope:foxglove.SceneEntity)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::KeyValuePair > metadata_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::ArrowPrimitive > arrows_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::CubePrimitive > cubes_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::SpherePrimitive > spheres_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::CylinderPrimitive > cylinders_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::LinePrimitive > lines_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::TriangleListPrimitive > triangles_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::TextPrimitive > texts_;
  ::google::protobuf::RepeatedPtrField< ::foxglove::ModelPrimitive > models_;
  ::google::protobuf::internal::ArenaStringPtr frame_id_;
  ::google::protobuf::internal::ArenaStringPtr id_;
  ::google::protobuf::Timestamp* timestamp_;
  ::google::protobuf::Duration* lifetime_;
  bool frame_locked_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_foxglove_2fSceneEntity_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// SceneEntity

// .google.protobuf.Timestamp timestamp = 1;
inline bool SceneEntity::has_timestamp() const {
  return this != internal_default_instance() && timestamp_ != NULL;
}
inline const ::google::protobuf::Timestamp& SceneEntity::_internal_timestamp() const {
  return *timestamp_;
}
inline const ::google::protobuf::Timestamp& SceneEntity::timestamp() const {
  const ::google::protobuf::Timestamp* p = timestamp_;
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.timestamp)
  return p != NULL ? *p : *reinterpret_cast<const ::google::protobuf::Timestamp*>(
      &::google::protobuf::_Timestamp_default_instance_);
}
inline ::google::protobuf::Timestamp* SceneEntity::release_timestamp() {
  // @@protoc_insertion_point(field_release:foxglove.SceneEntity.timestamp)
  
  ::google::protobuf::Timestamp* temp = timestamp_;
  timestamp_ = NULL;
  return temp;
}
inline ::google::protobuf::Timestamp* SceneEntity::mutable_timestamp() {
  
  if (timestamp_ == NULL) {
    auto* p = CreateMaybeMessage<::google::protobuf::Timestamp>(GetArenaNoVirtual());
    timestamp_ = p;
  }
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.timestamp)
  return timestamp_;
}
inline void SceneEntity::set_allocated_timestamp(::google::protobuf::Timestamp* timestamp) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(timestamp_);
  }
  if (timestamp) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(timestamp)->GetArena();
    if (message_arena != submessage_arena) {
      timestamp = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, timestamp, submessage_arena);
    }
    
  } else {
    
  }
  timestamp_ = timestamp;
  // @@protoc_insertion_point(field_set_allocated:foxglove.SceneEntity.timestamp)
}

// string frame_id = 2;
inline void SceneEntity::clear_frame_id() {
  frame_id_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& SceneEntity::frame_id() const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.frame_id)
  return frame_id_.GetNoArena();
}
inline void SceneEntity::set_frame_id(const ::std::string& value) {
  
  frame_id_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:foxglove.SceneEntity.frame_id)
}
#if LANG_CXX11
inline void SceneEntity::set_frame_id(::std::string&& value) {
  
  frame_id_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:foxglove.SceneEntity.frame_id)
}
#endif
inline void SceneEntity::set_frame_id(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  frame_id_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:foxglove.SceneEntity.frame_id)
}
inline void SceneEntity::set_frame_id(const char* value, size_t size) {
  
  frame_id_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:foxglove.SceneEntity.frame_id)
}
inline ::std::string* SceneEntity::mutable_frame_id() {
  
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.frame_id)
  return frame_id_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* SceneEntity::release_frame_id() {
  // @@protoc_insertion_point(field_release:foxglove.SceneEntity.frame_id)
  
  return frame_id_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void SceneEntity::set_allocated_frame_id(::std::string* frame_id) {
  if (frame_id != NULL) {
    
  } else {
    
  }
  frame_id_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), frame_id);
  // @@protoc_insertion_point(field_set_allocated:foxglove.SceneEntity.frame_id)
}

// string id = 3;
inline void SceneEntity::clear_id() {
  id_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& SceneEntity::id() const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.id)
  return id_.GetNoArena();
}
inline void SceneEntity::set_id(const ::std::string& value) {
  
  id_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:foxglove.SceneEntity.id)
}
#if LANG_CXX11
inline void SceneEntity::set_id(::std::string&& value) {
  
  id_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:foxglove.SceneEntity.id)
}
#endif
inline void SceneEntity::set_id(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  id_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:foxglove.SceneEntity.id)
}
inline void SceneEntity::set_id(const char* value, size_t size) {
  
  id_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:foxglove.SceneEntity.id)
}
inline ::std::string* SceneEntity::mutable_id() {
  
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.id)
  return id_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* SceneEntity::release_id() {
  // @@protoc_insertion_point(field_release:foxglove.SceneEntity.id)
  
  return id_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void SceneEntity::set_allocated_id(::std::string* id) {
  if (id != NULL) {
    
  } else {
    
  }
  id_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), id);
  // @@protoc_insertion_point(field_set_allocated:foxglove.SceneEntity.id)
}

// .google.protobuf.Duration lifetime = 4;
inline bool SceneEntity::has_lifetime() const {
  return this != internal_default_instance() && lifetime_ != NULL;
}
inline const ::google::protobuf::Duration& SceneEntity::_internal_lifetime() const {
  return *lifetime_;
}
inline const ::google::protobuf::Duration& SceneEntity::lifetime() const {
  const ::google::protobuf::Duration* p = lifetime_;
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.lifetime)
  return p != NULL ? *p : *reinterpret_cast<const ::google::protobuf::Duration*>(
      &::google::protobuf::_Duration_default_instance_);
}
inline ::google::protobuf::Duration* SceneEntity::release_lifetime() {
  // @@protoc_insertion_point(field_release:foxglove.SceneEntity.lifetime)
  
  ::google::protobuf::Duration* temp = lifetime_;
  lifetime_ = NULL;
  return temp;
}
inline ::google::protobuf::Duration* SceneEntity::mutable_lifetime() {
  
  if (lifetime_ == NULL) {
    auto* p = CreateMaybeMessage<::google::protobuf::Duration>(GetArenaNoVirtual());
    lifetime_ = p;
  }
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.lifetime)
  return lifetime_;
}
inline void SceneEntity::set_allocated_lifetime(::google::protobuf::Duration* lifetime) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(lifetime_);
  }
  if (lifetime) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(lifetime)->GetArena();
    if (message_arena != submessage_arena) {
      lifetime = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, lifetime, submessage_arena);
    }
    
  } else {
    
  }
  lifetime_ = lifetime;
  // @@protoc_insertion_point(field_set_allocated:foxglove.SceneEntity.lifetime)
}

// bool frame_locked = 5;
inline void SceneEntity::clear_frame_locked() {
  frame_locked_ = false;
}
inline bool SceneEntity::frame_locked() const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.frame_locked)
  return frame_locked_;
}
inline void SceneEntity::set_frame_locked(bool value) {
  
  frame_locked_ = value;
  // @@protoc_insertion_point(field_set:foxglove.SceneEntity.frame_locked)
}

// repeated .foxglove.KeyValuePair metadata = 6;
inline int SceneEntity::metadata_size() const {
  return metadata_.size();
}
inline ::foxglove::KeyValuePair* SceneEntity::mutable_metadata(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.metadata)
  return metadata_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::KeyValuePair >*
SceneEntity::mutable_metadata() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.metadata)
  return &metadata_;
}
inline const ::foxglove::KeyValuePair& SceneEntity::metadata(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.metadata)
  return metadata_.Get(index);
}
inline ::foxglove::KeyValuePair* SceneEntity::add_metadata() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.metadata)
  return metadata_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::KeyValuePair >&
SceneEntity::metadata() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.metadata)
  return metadata_;
}

// repeated .foxglove.ArrowPrimitive arrows = 7;
inline int SceneEntity::arrows_size() const {
  return arrows_.size();
}
inline ::foxglove::ArrowPrimitive* SceneEntity::mutable_arrows(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.arrows)
  return arrows_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::ArrowPrimitive >*
SceneEntity::mutable_arrows() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.arrows)
  return &arrows_;
}
inline const ::foxglove::ArrowPrimitive& SceneEntity::arrows(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.arrows)
  return arrows_.Get(index);
}
inline ::foxglove::ArrowPrimitive* SceneEntity::add_arrows() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.arrows)
  return arrows_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::ArrowPrimitive >&
SceneEntity::arrows() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.arrows)
  return arrows_;
}

// repeated .foxglove.CubePrimitive cubes = 8;
inline int SceneEntity::cubes_size() const {
  return cubes_.size();
}
inline ::foxglove::CubePrimitive* SceneEntity::mutable_cubes(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.cubes)
  return cubes_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::CubePrimitive >*
SceneEntity::mutable_cubes() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.cubes)
  return &cubes_;
}
inline const ::foxglove::CubePrimitive& SceneEntity::cubes(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.cubes)
  return cubes_.Get(index);
}
inline ::foxglove::CubePrimitive* SceneEntity::add_cubes() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.cubes)
  return cubes_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::CubePrimitive >&
SceneEntity::cubes() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.cubes)
  return cubes_;
}

// repeated .foxglove.SpherePrimitive spheres = 9;
inline int SceneEntity::spheres_size() const {
  return spheres_.size();
}
inline ::foxglove::SpherePrimitive* SceneEntity::mutable_spheres(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.spheres)
  return spheres_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::SpherePrimitive >*
SceneEntity::mutable_spheres() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.spheres)
  return &spheres_;
}
inline const ::foxglove::SpherePrimitive& SceneEntity::spheres(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.spheres)
  return spheres_.Get(index);
}
inline ::foxglove::SpherePrimitive* SceneEntity::add_spheres() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.spheres)
  return spheres_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::SpherePrimitive >&
SceneEntity::spheres() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.spheres)
  return spheres_;
}

// repeated .foxglove.CylinderPrimitive cylinders = 10;
inline int SceneEntity::cylinders_size() const {
  return cylinders_.size();
}
inline ::foxglove::CylinderPrimitive* SceneEntity::mutable_cylinders(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.cylinders)
  return cylinders_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::CylinderPrimitive >*
SceneEntity::mutable_cylinders() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.cylinders)
  return &cylinders_;
}
inline const ::foxglove::CylinderPrimitive& SceneEntity::cylinders(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.cylinders)
  return cylinders_.Get(index);
}
inline ::foxglove::CylinderPrimitive* SceneEntity::add_cylinders() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.cylinders)
  return cylinders_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::CylinderPrimitive >&
SceneEntity::cylinders() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.cylinders)
  return cylinders_;
}

// repeated .foxglove.LinePrimitive lines = 11;
inline int SceneEntity::lines_size() const {
  return lines_.size();
}
inline ::foxglove::LinePrimitive* SceneEntity::mutable_lines(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.lines)
  return lines_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::LinePrimitive >*
SceneEntity::mutable_lines() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.lines)
  return &lines_;
}
inline const ::foxglove::LinePrimitive& SceneEntity::lines(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.lines)
  return lines_.Get(index);
}
inline ::foxglove::LinePrimitive* SceneEntity::add_lines() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.lines)
  return lines_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::LinePrimitive >&
SceneEntity::lines() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.lines)
  return lines_;
}

// repeated .foxglove.TriangleListPrimitive triangles = 12;
inline int SceneEntity::triangles_size() const {
  return triangles_.size();
}
inline ::foxglove::TriangleListPrimitive* SceneEntity::mutable_triangles(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.triangles)
  return triangles_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::TriangleListPrimitive >*
SceneEntity::mutable_triangles() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.triangles)
  return &triangles_;
}
inline const ::foxglove::TriangleListPrimitive& SceneEntity::triangles(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.triangles)
  return triangles_.Get(index);
}
inline ::foxglove::TriangleListPrimitive* SceneEntity::add_triangles() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.triangles)
  return triangles_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::TriangleListPrimitive >&
SceneEntity::triangles() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.triangles)
  return triangles_;
}

// repeated .foxglove.TextPrimitive texts = 13;
inline int SceneEntity::texts_size() const {
  return texts_.size();
}
inline ::foxglove::TextPrimitive* SceneEntity::mutable_texts(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.texts)
  return texts_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::TextPrimitive >*
SceneEntity::mutable_texts() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.texts)
  return &texts_;
}
inline const ::foxglove::TextPrimitive& SceneEntity::texts(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.texts)
  return texts_.Get(index);
}
inline ::foxglove::TextPrimitive* SceneEntity::add_texts() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.texts)
  return texts_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::TextPrimitive >&
SceneEntity::texts() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.texts)
  return texts_;
}

// repeated .foxglove.ModelPrimitive models = 14;
inline int SceneEntity::models_size() const {
  return models_.size();
}
inline ::foxglove::ModelPrimitive* SceneEntity::mutable_models(int index) {
  // @@protoc_insertion_point(field_mutable:foxglove.SceneEntity.models)
  return models_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::foxglove::ModelPrimitive >*
SceneEntity::mutable_models() {
  // @@protoc_insertion_point(field_mutable_list:foxglove.SceneEntity.models)
  return &models_;
}
inline const ::foxglove::ModelPrimitive& SceneEntity::models(int index) const {
  // @@protoc_insertion_point(field_get:foxglove.SceneEntity.models)
  return models_.Get(index);
}
inline ::foxglove::ModelPrimitive* SceneEntity::add_models() {
  // @@protoc_insertion_point(field_add:foxglove.SceneEntity.models)
  return models_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::foxglove::ModelPrimitive >&
SceneEntity::models() const {
  // @@protoc_insertion_point(field_list:foxglove.SceneEntity.models)
  return models_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace foxglove

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_foxglove_2fSceneEntity_2eproto
