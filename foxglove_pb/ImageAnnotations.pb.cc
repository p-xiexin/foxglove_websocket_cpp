// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: foxglove/ImageAnnotations.proto

#include "foxglove/ImageAnnotations.pb.h"

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

namespace protobuf_foxglove_2fCircleAnnotation_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fCircleAnnotation_2eproto ::google::protobuf::internal::SCCInfo<3> scc_info_CircleAnnotation;
}  // namespace protobuf_foxglove_2fCircleAnnotation_2eproto
namespace protobuf_foxglove_2fPointsAnnotation_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fPointsAnnotation_2eproto ::google::protobuf::internal::SCCInfo<3> scc_info_PointsAnnotation;
}  // namespace protobuf_foxglove_2fPointsAnnotation_2eproto
namespace protobuf_foxglove_2fTextAnnotation_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_foxglove_2fTextAnnotation_2eproto ::google::protobuf::internal::SCCInfo<3> scc_info_TextAnnotation;
}  // namespace protobuf_foxglove_2fTextAnnotation_2eproto
namespace foxglove {
class ImageAnnotationsDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<ImageAnnotations>
      _instance;
} _ImageAnnotations_default_instance_;
}  // namespace foxglove
namespace protobuf_foxglove_2fImageAnnotations_2eproto {
static void InitDefaultsImageAnnotations() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::foxglove::_ImageAnnotations_default_instance_;
    new (ptr) ::foxglove::ImageAnnotations();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::foxglove::ImageAnnotations::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<3> scc_info_ImageAnnotations =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 3, InitDefaultsImageAnnotations}, {
      &protobuf_foxglove_2fCircleAnnotation_2eproto::scc_info_CircleAnnotation.base,
      &protobuf_foxglove_2fPointsAnnotation_2eproto::scc_info_PointsAnnotation.base,
      &protobuf_foxglove_2fTextAnnotation_2eproto::scc_info_TextAnnotation.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_ImageAnnotations.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::ImageAnnotations, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::ImageAnnotations, circles_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::ImageAnnotations, points_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::ImageAnnotations, texts_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::foxglove::ImageAnnotations)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::foxglove::_ImageAnnotations_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "foxglove/ImageAnnotations.proto", schemas, file_default_instances, TableStruct::offsets,
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
      "\n\037foxglove/ImageAnnotations.proto\022\010foxgl"
      "ove\032\037foxglove/CircleAnnotation.proto\032\037fo"
      "xglove/PointsAnnotation.proto\032\035foxglove/"
      "TextAnnotation.proto\"\224\001\n\020ImageAnnotation"
      "s\022+\n\007circles\030\001 \003(\0132\032.foxglove.CircleAnno"
      "tation\022*\n\006points\030\002 \003(\0132\032.foxglove.Points"
      "Annotation\022\'\n\005texts\030\003 \003(\0132\030.foxglove.Tex"
      "tAnnotationb\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 299);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "foxglove/ImageAnnotations.proto", &protobuf_RegisterTypes);
  ::protobuf_foxglove_2fCircleAnnotation_2eproto::AddDescriptors();
  ::protobuf_foxglove_2fPointsAnnotation_2eproto::AddDescriptors();
  ::protobuf_foxglove_2fTextAnnotation_2eproto::AddDescriptors();
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
}  // namespace protobuf_foxglove_2fImageAnnotations_2eproto
namespace foxglove {

// ===================================================================

void ImageAnnotations::InitAsDefaultInstance() {
}
void ImageAnnotations::clear_circles() {
  circles_.Clear();
}
void ImageAnnotations::clear_points() {
  points_.Clear();
}
void ImageAnnotations::clear_texts() {
  texts_.Clear();
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int ImageAnnotations::kCirclesFieldNumber;
const int ImageAnnotations::kPointsFieldNumber;
const int ImageAnnotations::kTextsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

ImageAnnotations::ImageAnnotations()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_foxglove_2fImageAnnotations_2eproto::scc_info_ImageAnnotations.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:foxglove.ImageAnnotations)
}
ImageAnnotations::ImageAnnotations(const ImageAnnotations& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      circles_(from.circles_),
      points_(from.points_),
      texts_(from.texts_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:foxglove.ImageAnnotations)
}

void ImageAnnotations::SharedCtor() {
}

ImageAnnotations::~ImageAnnotations() {
  // @@protoc_insertion_point(destructor:foxglove.ImageAnnotations)
  SharedDtor();
}

void ImageAnnotations::SharedDtor() {
}

void ImageAnnotations::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* ImageAnnotations::descriptor() {
  ::protobuf_foxglove_2fImageAnnotations_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_foxglove_2fImageAnnotations_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const ImageAnnotations& ImageAnnotations::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_foxglove_2fImageAnnotations_2eproto::scc_info_ImageAnnotations.base);
  return *internal_default_instance();
}


void ImageAnnotations::Clear() {
// @@protoc_insertion_point(message_clear_start:foxglove.ImageAnnotations)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  circles_.Clear();
  points_.Clear();
  texts_.Clear();
  _internal_metadata_.Clear();
}

bool ImageAnnotations::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:foxglove.ImageAnnotations)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .foxglove.CircleAnnotation circles = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_circles()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated .foxglove.PointsAnnotation points = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_points()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated .foxglove.TextAnnotation texts = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(26u /* 26 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_texts()));
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
  // @@protoc_insertion_point(parse_success:foxglove.ImageAnnotations)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:foxglove.ImageAnnotations)
  return false;
#undef DO_
}

void ImageAnnotations::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:foxglove.ImageAnnotations)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .foxglove.CircleAnnotation circles = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->circles_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->circles(static_cast<int>(i)),
      output);
  }

  // repeated .foxglove.PointsAnnotation points = 2;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->points_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2,
      this->points(static_cast<int>(i)),
      output);
  }

  // repeated .foxglove.TextAnnotation texts = 3;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->texts_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      3,
      this->texts(static_cast<int>(i)),
      output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:foxglove.ImageAnnotations)
}

::google::protobuf::uint8* ImageAnnotations::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:foxglove.ImageAnnotations)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .foxglove.CircleAnnotation circles = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->circles_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->circles(static_cast<int>(i)), deterministic, target);
  }

  // repeated .foxglove.PointsAnnotation points = 2;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->points_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, this->points(static_cast<int>(i)), deterministic, target);
  }

  // repeated .foxglove.TextAnnotation texts = 3;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->texts_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        3, this->texts(static_cast<int>(i)), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:foxglove.ImageAnnotations)
  return target;
}

size_t ImageAnnotations::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:foxglove.ImageAnnotations)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .foxglove.CircleAnnotation circles = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->circles_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->circles(static_cast<int>(i)));
    }
  }

  // repeated .foxglove.PointsAnnotation points = 2;
  {
    unsigned int count = static_cast<unsigned int>(this->points_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->points(static_cast<int>(i)));
    }
  }

  // repeated .foxglove.TextAnnotation texts = 3;
  {
    unsigned int count = static_cast<unsigned int>(this->texts_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->texts(static_cast<int>(i)));
    }
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ImageAnnotations::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:foxglove.ImageAnnotations)
  GOOGLE_DCHECK_NE(&from, this);
  const ImageAnnotations* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const ImageAnnotations>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:foxglove.ImageAnnotations)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:foxglove.ImageAnnotations)
    MergeFrom(*source);
  }
}

void ImageAnnotations::MergeFrom(const ImageAnnotations& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:foxglove.ImageAnnotations)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  circles_.MergeFrom(from.circles_);
  points_.MergeFrom(from.points_);
  texts_.MergeFrom(from.texts_);
}

void ImageAnnotations::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:foxglove.ImageAnnotations)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ImageAnnotations::CopyFrom(const ImageAnnotations& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:foxglove.ImageAnnotations)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ImageAnnotations::IsInitialized() const {
  return true;
}

void ImageAnnotations::Swap(ImageAnnotations* other) {
  if (other == this) return;
  InternalSwap(other);
}
void ImageAnnotations::InternalSwap(ImageAnnotations* other) {
  using std::swap;
  CastToBase(&circles_)->InternalSwap(CastToBase(&other->circles_));
  CastToBase(&points_)->InternalSwap(CastToBase(&other->points_));
  CastToBase(&texts_)->InternalSwap(CastToBase(&other->texts_));
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata ImageAnnotations::GetMetadata() const {
  protobuf_foxglove_2fImageAnnotations_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_foxglove_2fImageAnnotations_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace foxglove
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::foxglove::ImageAnnotations* Arena::CreateMaybeMessage< ::foxglove::ImageAnnotations >(Arena* arena) {
  return Arena::CreateInternal< ::foxglove::ImageAnnotations >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
