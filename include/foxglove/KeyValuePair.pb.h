// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: foxglove/KeyValuePair.proto

#ifndef PROTOBUF_INCLUDED_foxglove_2fKeyValuePair_2eproto
#define PROTOBUF_INCLUDED_foxglove_2fKeyValuePair_2eproto

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
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fKeyValuePair_2eproto 

namespace protobuf_foxglove_2fKeyValuePair_2eproto {
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
}  // namespace protobuf_foxglove_2fKeyValuePair_2eproto
namespace foxglove {
class KeyValuePair;
class KeyValuePairDefaultTypeInternal;
extern KeyValuePairDefaultTypeInternal _KeyValuePair_default_instance_;
}  // namespace foxglove
namespace google {
namespace protobuf {
template<> ::foxglove::KeyValuePair* Arena::CreateMaybeMessage<::foxglove::KeyValuePair>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace foxglove {

// ===================================================================

class KeyValuePair : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:foxglove.KeyValuePair) */ {
 public:
  KeyValuePair();
  virtual ~KeyValuePair();

  KeyValuePair(const KeyValuePair& from);

  inline KeyValuePair& operator=(const KeyValuePair& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  KeyValuePair(KeyValuePair&& from) noexcept
    : KeyValuePair() {
    *this = ::std::move(from);
  }

  inline KeyValuePair& operator=(KeyValuePair&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const KeyValuePair& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const KeyValuePair* internal_default_instance() {
    return reinterpret_cast<const KeyValuePair*>(
               &_KeyValuePair_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(KeyValuePair* other);
  friend void swap(KeyValuePair& a, KeyValuePair& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline KeyValuePair* New() const final {
    return CreateMaybeMessage<KeyValuePair>(NULL);
  }

  KeyValuePair* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<KeyValuePair>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const KeyValuePair& from);
  void MergeFrom(const KeyValuePair& from);
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
  void InternalSwap(KeyValuePair* other);
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

  // string key = 1;
  void clear_key();
  static const int kKeyFieldNumber = 1;
  const ::std::string& key() const;
  void set_key(const ::std::string& value);
  #if LANG_CXX11
  void set_key(::std::string&& value);
  #endif
  void set_key(const char* value);
  void set_key(const char* value, size_t size);
  ::std::string* mutable_key();
  ::std::string* release_key();
  void set_allocated_key(::std::string* key);

  // string value = 2;
  void clear_value();
  static const int kValueFieldNumber = 2;
  const ::std::string& value() const;
  void set_value(const ::std::string& value);
  #if LANG_CXX11
  void set_value(::std::string&& value);
  #endif
  void set_value(const char* value);
  void set_value(const char* value, size_t size);
  ::std::string* mutable_value();
  ::std::string* release_value();
  void set_allocated_value(::std::string* value);

  // @@protoc_insertion_point(class_scope:foxglove.KeyValuePair)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr key_;
  ::google::protobuf::internal::ArenaStringPtr value_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_foxglove_2fKeyValuePair_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// KeyValuePair

// string key = 1;
inline void KeyValuePair::clear_key() {
  key_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& KeyValuePair::key() const {
  // @@protoc_insertion_point(field_get:foxglove.KeyValuePair.key)
  return key_.GetNoArena();
}
inline void KeyValuePair::set_key(const ::std::string& value) {
  
  key_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:foxglove.KeyValuePair.key)
}
#if LANG_CXX11
inline void KeyValuePair::set_key(::std::string&& value) {
  
  key_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:foxglove.KeyValuePair.key)
}
#endif
inline void KeyValuePair::set_key(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  key_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:foxglove.KeyValuePair.key)
}
inline void KeyValuePair::set_key(const char* value, size_t size) {
  
  key_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:foxglove.KeyValuePair.key)
}
inline ::std::string* KeyValuePair::mutable_key() {
  
  // @@protoc_insertion_point(field_mutable:foxglove.KeyValuePair.key)
  return key_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* KeyValuePair::release_key() {
  // @@protoc_insertion_point(field_release:foxglove.KeyValuePair.key)
  
  return key_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void KeyValuePair::set_allocated_key(::std::string* key) {
  if (key != NULL) {
    
  } else {
    
  }
  key_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), key);
  // @@protoc_insertion_point(field_set_allocated:foxglove.KeyValuePair.key)
}

// string value = 2;
inline void KeyValuePair::clear_value() {
  value_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& KeyValuePair::value() const {
  // @@protoc_insertion_point(field_get:foxglove.KeyValuePair.value)
  return value_.GetNoArena();
}
inline void KeyValuePair::set_value(const ::std::string& value) {
  
  value_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:foxglove.KeyValuePair.value)
}
#if LANG_CXX11
inline void KeyValuePair::set_value(::std::string&& value) {
  
  value_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:foxglove.KeyValuePair.value)
}
#endif
inline void KeyValuePair::set_value(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  value_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:foxglove.KeyValuePair.value)
}
inline void KeyValuePair::set_value(const char* value, size_t size) {
  
  value_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:foxglove.KeyValuePair.value)
}
inline ::std::string* KeyValuePair::mutable_value() {
  
  // @@protoc_insertion_point(field_mutable:foxglove.KeyValuePair.value)
  return value_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* KeyValuePair::release_value() {
  // @@protoc_insertion_point(field_release:foxglove.KeyValuePair.value)
  
  return value_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void KeyValuePair::set_allocated_value(::std::string* value) {
  if (value != NULL) {
    
  } else {
    
  }
  value_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set_allocated:foxglove.KeyValuePair.value)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace foxglove

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_foxglove_2fKeyValuePair_2eproto