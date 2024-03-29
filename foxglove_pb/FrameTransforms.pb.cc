// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: foxglove/FrameTransforms.proto

#include "foxglove/FrameTransforms.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace protobuf_foxglove_2fFrameTransform_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fFrameTransform_2eproto ::google::protobuf::internal::SCCInfo<3> scc_info_FrameTransform;
}  // namespace protobuf_foxglove_2fFrameTransform_2eproto
namespace foxglove {
class FrameTransformsDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<FrameTransforms>
      _instance;
} _FrameTransforms_default_instance_;
}  // namespace foxglove
namespace protobuf_foxglove_2fFrameTransforms_2eproto {
static void InitDefaultsFrameTransforms() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::foxglove::_FrameTransforms_default_instance_;
    new (ptr) ::foxglove::FrameTransforms();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::foxglove::FrameTransforms::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_FrameTransforms =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsFrameTransforms}, {
      &protobuf_foxglove_2fFrameTransform_2eproto::scc_info_FrameTransform.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_FrameTransforms.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::FrameTransforms, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::FrameTransforms, transforms_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::foxglove::FrameTransforms)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::foxglove::_FrameTransforms_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "foxglove/FrameTransforms.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\036foxglove/FrameTransforms.proto\022\010foxglo"
      "ve\032\035foxglove/FrameTransform.proto\"\?\n\017Fra"
      "meTransforms\022,\n\ntransforms\030\001 \003(\0132\030.foxgl"
      "ove.FrameTransformb\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 146);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "foxglove/FrameTransforms.proto", &protobuf_RegisterTypes);
  ::protobuf_foxglove_2fFrameTransform_2eproto::AddDescriptors();
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_foxglove_2fFrameTransforms_2eproto
namespace foxglove {

// ===================================================================

void FrameTransforms::InitAsDefaultInstance() {
}
void FrameTransforms::clear_transforms() {
  transforms_.Clear();
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int FrameTransforms::kTransformsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

FrameTransforms::FrameTransforms()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_foxglove_2fFrameTransforms_2eproto::scc_info_FrameTransforms.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:foxglove.FrameTransforms)
}
FrameTransforms::FrameTransforms(const FrameTransforms& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      transforms_(from.transforms_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:foxglove.FrameTransforms)
}

void FrameTransforms::SharedCtor() {
}

FrameTransforms::~FrameTransforms() {
  // @@protoc_insertion_point(destructor:foxglove.FrameTransforms)
  SharedDtor();
}

void FrameTransforms::SharedDtor() {
}

void FrameTransforms::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* FrameTransforms::descriptor() {
  ::protobuf_foxglove_2fFrameTransforms_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_foxglove_2fFrameTransforms_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const FrameTransforms& FrameTransforms::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_foxglove_2fFrameTransforms_2eproto::scc_info_FrameTransforms.base);
  return *internal_default_instance();
}


void FrameTransforms::Clear() {
// @@protoc_insertion_point(message_clear_start:foxglove.FrameTransforms)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  transforms_.Clear();
  _internal_metadata_.Clear();
}

bool FrameTransforms::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:foxglove.FrameTransforms)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .foxglove.FrameTransform transforms = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_transforms()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:foxglove.FrameTransforms)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:foxglove.FrameTransforms)
  return false;
#undef DO_
}

void FrameTransforms::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:foxglove.FrameTransforms)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .foxglove.FrameTransform transforms = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->transforms_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->transforms(static_cast<int>(i)),
      output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:foxglove.FrameTransforms)
}

::google::protobuf::uint8* FrameTransforms::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:foxglove.FrameTransforms)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .foxglove.FrameTransform transforms = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->transforms_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->transforms(static_cast<int>(i)), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:foxglove.FrameTransforms)
  return target;
}

size_t FrameTransforms::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:foxglove.FrameTransforms)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .foxglove.FrameTransform transforms = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->transforms_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->transforms(static_cast<int>(i)));
    }
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void FrameTransforms::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:foxglove.FrameTransforms)
  GOOGLE_DCHECK_NE(&from, this);
  const FrameTransforms* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const FrameTransforms>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:foxglove.FrameTransforms)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:foxglove.FrameTransforms)
    MergeFrom(*source);
  }
}

void FrameTransforms::MergeFrom(const FrameTransforms& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:foxglove.FrameTransforms)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  transforms_.MergeFrom(from.transforms_);
}

void FrameTransforms::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:foxglove.FrameTransforms)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void FrameTransforms::CopyFrom(const FrameTransforms& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:foxglove.FrameTransforms)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool FrameTransforms::IsInitialized() const {
  return true;
}

void FrameTransforms::Swap(FrameTransforms* other) {
  if (other == this) return;
  InternalSwap(other);
}
void FrameTransforms::InternalSwap(FrameTransforms* other) {
  using std::swap;
  CastToBase(&transforms_)->InternalSwap(CastToBase(&other->transforms_));
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata FrameTransforms::GetMetadata() const {
  protobuf_foxglove_2fFrameTransforms_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_foxglove_2fFrameTransforms_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace foxglove
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::foxglove::FrameTransforms* Arena::CreateMaybeMessage< ::foxglove::FrameTransforms >(Arena* arena) {
  return Arena::CreateInternal< ::foxglove::FrameTransforms >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
