// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: foxglove/CameraCalibration.proto

#include "foxglove/CameraCalibration.pb.h"

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

namespace protobuf_google_2fprotobuf_2ftimestamp_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_google_2fprotobuf_2ftimestamp_2eproto ::google::protobuf::internal::SCCInfo<0> scc_info_Timestamp;
}  // namespace protobuf_google_2fprotobuf_2ftimestamp_2eproto
namespace foxglove {
class CameraCalibrationDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<CameraCalibration>
      _instance;
} _CameraCalibration_default_instance_;
}  // namespace foxglove
namespace protobuf_foxglove_2fCameraCalibration_2eproto {
static void InitDefaultsCameraCalibration() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::foxglove::_CameraCalibration_default_instance_;
    new (ptr) ::foxglove::CameraCalibration();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::foxglove::CameraCalibration::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_CameraCalibration =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsCameraCalibration}, {
      &protobuf_google_2fprotobuf_2ftimestamp_2eproto::scc_info_Timestamp.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_CameraCalibration.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, timestamp_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, frame_id_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, width_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, height_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, distortion_model_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, d_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, k_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, r_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::foxglove::CameraCalibration, p_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::foxglove::CameraCalibration)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::foxglove::_CameraCalibration_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "foxglove/CameraCalibration.proto", schemas, file_default_instances, TableStruct::offsets,
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
      "\n foxglove/CameraCalibration.proto\022\010foxg"
      "love\032\037google/protobuf/timestamp.proto\"\271\001"
      "\n\021CameraCalibration\022-\n\ttimestamp\030\001 \001(\0132\032"
      ".google.protobuf.Timestamp\022\020\n\010frame_id\030\t"
      " \001(\t\022\r\n\005width\030\002 \001(\007\022\016\n\006height\030\003 \001(\007\022\030\n\020d"
      "istortion_model\030\004 \001(\t\022\t\n\001D\030\005 \003(\001\022\t\n\001K\030\006 "
      "\003(\001\022\t\n\001R\030\007 \003(\001\022\t\n\001P\030\010 \003(\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 273);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "foxglove/CameraCalibration.proto", &protobuf_RegisterTypes);
  ::protobuf_google_2fprotobuf_2ftimestamp_2eproto::AddDescriptors();
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
}  // namespace protobuf_foxglove_2fCameraCalibration_2eproto
namespace foxglove {

// ===================================================================

void CameraCalibration::InitAsDefaultInstance() {
  ::foxglove::_CameraCalibration_default_instance_._instance.get_mutable()->timestamp_ = const_cast< ::google::protobuf::Timestamp*>(
      ::google::protobuf::Timestamp::internal_default_instance());
}
void CameraCalibration::clear_timestamp() {
  if (GetArenaNoVirtual() == NULL && timestamp_ != NULL) {
    delete timestamp_;
  }
  timestamp_ = NULL;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int CameraCalibration::kTimestampFieldNumber;
const int CameraCalibration::kFrameIdFieldNumber;
const int CameraCalibration::kWidthFieldNumber;
const int CameraCalibration::kHeightFieldNumber;
const int CameraCalibration::kDistortionModelFieldNumber;
const int CameraCalibration::kDFieldNumber;
const int CameraCalibration::kKFieldNumber;
const int CameraCalibration::kRFieldNumber;
const int CameraCalibration::kPFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

CameraCalibration::CameraCalibration()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_foxglove_2fCameraCalibration_2eproto::scc_info_CameraCalibration.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:foxglove.CameraCalibration)
}
CameraCalibration::CameraCalibration(const CameraCalibration& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      d_(from.d_),
      k_(from.k_),
      r_(from.r_),
      p_(from.p_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  distortion_model_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.distortion_model().size() > 0) {
    distortion_model_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.distortion_model_);
  }
  frame_id_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.frame_id().size() > 0) {
    frame_id_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.frame_id_);
  }
  if (from.has_timestamp()) {
    timestamp_ = new ::google::protobuf::Timestamp(*from.timestamp_);
  } else {
    timestamp_ = NULL;
  }
  ::memcpy(&width_, &from.width_,
    static_cast<size_t>(reinterpret_cast<char*>(&height_) -
    reinterpret_cast<char*>(&width_)) + sizeof(height_));
  // @@protoc_insertion_point(copy_constructor:foxglove.CameraCalibration)
}

void CameraCalibration::SharedCtor() {
  distortion_model_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  frame_id_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&timestamp_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&height_) -
      reinterpret_cast<char*>(&timestamp_)) + sizeof(height_));
}

CameraCalibration::~CameraCalibration() {
  // @@protoc_insertion_point(destructor:foxglove.CameraCalibration)
  SharedDtor();
}

void CameraCalibration::SharedDtor() {
  distortion_model_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  frame_id_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (this != internal_default_instance()) delete timestamp_;
}

void CameraCalibration::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* CameraCalibration::descriptor() {
  ::protobuf_foxglove_2fCameraCalibration_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_foxglove_2fCameraCalibration_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const CameraCalibration& CameraCalibration::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_foxglove_2fCameraCalibration_2eproto::scc_info_CameraCalibration.base);
  return *internal_default_instance();
}


void CameraCalibration::Clear() {
// @@protoc_insertion_point(message_clear_start:foxglove.CameraCalibration)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  d_.Clear();
  k_.Clear();
  r_.Clear();
  p_.Clear();
  distortion_model_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  frame_id_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (GetArenaNoVirtual() == NULL && timestamp_ != NULL) {
    delete timestamp_;
  }
  timestamp_ = NULL;
  ::memset(&width_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&height_) -
      reinterpret_cast<char*>(&width_)) + sizeof(height_));
  _internal_metadata_.Clear();
}

bool CameraCalibration::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:foxglove.CameraCalibration)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // .google.protobuf.Timestamp timestamp = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
               input, mutable_timestamp()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // fixed32 width = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(21u /* 21 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_FIXED32>(
                 input, &width_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // fixed32 height = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(29u /* 29 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_FIXED32>(
                 input, &height_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string distortion_model = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(34u /* 34 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_distortion_model()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->distortion_model().data(), static_cast<int>(this->distortion_model().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "foxglove.CameraCalibration.distortion_model"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated double D = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(42u /* 42 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, this->mutable_d())));
        } else if (
            static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(41u /* 41 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 1, 42u, input, this->mutable_d())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated double K = 6;
      case 6: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(50u /* 50 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, this->mutable_k())));
        } else if (
            static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(49u /* 49 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 1, 50u, input, this->mutable_k())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated double R = 7;
      case 7: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(58u /* 58 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, this->mutable_r())));
        } else if (
            static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(57u /* 57 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 1, 58u, input, this->mutable_r())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated double P = 8;
      case 8: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(66u /* 66 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, this->mutable_p())));
        } else if (
            static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(65u /* 65 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 1, 66u, input, this->mutable_p())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string frame_id = 9;
      case 9: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(74u /* 74 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_frame_id()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->frame_id().data(), static_cast<int>(this->frame_id().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "foxglove.CameraCalibration.frame_id"));
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
  // @@protoc_insertion_point(parse_success:foxglove.CameraCalibration)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:foxglove.CameraCalibration)
  return false;
#undef DO_
}

void CameraCalibration::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:foxglove.CameraCalibration)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .google.protobuf.Timestamp timestamp = 1;
  if (this->has_timestamp()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->_internal_timestamp(), output);
  }

  // fixed32 width = 2;
  if (this->width() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteFixed32(2, this->width(), output);
  }

  // fixed32 height = 3;
  if (this->height() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteFixed32(3, this->height(), output);
  }

  // string distortion_model = 4;
  if (this->distortion_model().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->distortion_model().data(), static_cast<int>(this->distortion_model().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "foxglove.CameraCalibration.distortion_model");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      4, this->distortion_model(), output);
  }

  // repeated double D = 5;
  if (this->d_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(5, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(static_cast< ::google::protobuf::uint32>(
        _d_cached_byte_size_));
    ::google::protobuf::internal::WireFormatLite::WriteDoubleArray(
      this->d().data(), this->d_size(), output);
  }

  // repeated double K = 6;
  if (this->k_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(6, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(static_cast< ::google::protobuf::uint32>(
        _k_cached_byte_size_));
    ::google::protobuf::internal::WireFormatLite::WriteDoubleArray(
      this->k().data(), this->k_size(), output);
  }

  // repeated double R = 7;
  if (this->r_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(7, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(static_cast< ::google::protobuf::uint32>(
        _r_cached_byte_size_));
    ::google::protobuf::internal::WireFormatLite::WriteDoubleArray(
      this->r().data(), this->r_size(), output);
  }

  // repeated double P = 8;
  if (this->p_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(8, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(static_cast< ::google::protobuf::uint32>(
        _p_cached_byte_size_));
    ::google::protobuf::internal::WireFormatLite::WriteDoubleArray(
      this->p().data(), this->p_size(), output);
  }

  // string frame_id = 9;
  if (this->frame_id().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->frame_id().data(), static_cast<int>(this->frame_id().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "foxglove.CameraCalibration.frame_id");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      9, this->frame_id(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:foxglove.CameraCalibration)
}

::google::protobuf::uint8* CameraCalibration::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:foxglove.CameraCalibration)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .google.protobuf.Timestamp timestamp = 1;
  if (this->has_timestamp()) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->_internal_timestamp(), deterministic, target);
  }

  // fixed32 width = 2;
  if (this->width() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFixed32ToArray(2, this->width(), target);
  }

  // fixed32 height = 3;
  if (this->height() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFixed32ToArray(3, this->height(), target);
  }

  // string distortion_model = 4;
  if (this->distortion_model().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->distortion_model().data(), static_cast<int>(this->distortion_model().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "foxglove.CameraCalibration.distortion_model");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        4, this->distortion_model(), target);
  }

  // repeated double D = 5;
  if (this->d_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      5,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        static_cast< ::google::protobuf::int32>(
            _d_cached_byte_size_), target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteDoubleNoTagToArray(this->d_, target);
  }

  // repeated double K = 6;
  if (this->k_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      6,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        static_cast< ::google::protobuf::int32>(
            _k_cached_byte_size_), target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteDoubleNoTagToArray(this->k_, target);
  }

  // repeated double R = 7;
  if (this->r_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      7,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        static_cast< ::google::protobuf::int32>(
            _r_cached_byte_size_), target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteDoubleNoTagToArray(this->r_, target);
  }

  // repeated double P = 8;
  if (this->p_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      8,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        static_cast< ::google::protobuf::int32>(
            _p_cached_byte_size_), target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteDoubleNoTagToArray(this->p_, target);
  }

  // string frame_id = 9;
  if (this->frame_id().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->frame_id().data(), static_cast<int>(this->frame_id().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "foxglove.CameraCalibration.frame_id");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        9, this->frame_id(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:foxglove.CameraCalibration)
  return target;
}

size_t CameraCalibration::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:foxglove.CameraCalibration)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated double D = 5;
  {
    unsigned int count = static_cast<unsigned int>(this->d_size());
    size_t data_size = 8UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast< ::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _d_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated double K = 6;
  {
    unsigned int count = static_cast<unsigned int>(this->k_size());
    size_t data_size = 8UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast< ::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _k_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated double R = 7;
  {
    unsigned int count = static_cast<unsigned int>(this->r_size());
    size_t data_size = 8UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast< ::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _r_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated double P = 8;
  {
    unsigned int count = static_cast<unsigned int>(this->p_size());
    size_t data_size = 8UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast< ::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _p_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // string distortion_model = 4;
  if (this->distortion_model().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->distortion_model());
  }

  // string frame_id = 9;
  if (this->frame_id().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->frame_id());
  }

  // .google.protobuf.Timestamp timestamp = 1;
  if (this->has_timestamp()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSize(
        *timestamp_);
  }

  // fixed32 width = 2;
  if (this->width() != 0) {
    total_size += 1 + 4;
  }

  // fixed32 height = 3;
  if (this->height() != 0) {
    total_size += 1 + 4;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void CameraCalibration::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:foxglove.CameraCalibration)
  GOOGLE_DCHECK_NE(&from, this);
  const CameraCalibration* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const CameraCalibration>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:foxglove.CameraCalibration)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:foxglove.CameraCalibration)
    MergeFrom(*source);
  }
}

void CameraCalibration::MergeFrom(const CameraCalibration& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:foxglove.CameraCalibration)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  d_.MergeFrom(from.d_);
  k_.MergeFrom(from.k_);
  r_.MergeFrom(from.r_);
  p_.MergeFrom(from.p_);
  if (from.distortion_model().size() > 0) {

    distortion_model_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.distortion_model_);
  }
  if (from.frame_id().size() > 0) {

    frame_id_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.frame_id_);
  }
  if (from.has_timestamp()) {
    mutable_timestamp()->::google::protobuf::Timestamp::MergeFrom(from.timestamp());
  }
  if (from.width() != 0) {
    set_width(from.width());
  }
  if (from.height() != 0) {
    set_height(from.height());
  }
}

void CameraCalibration::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:foxglove.CameraCalibration)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void CameraCalibration::CopyFrom(const CameraCalibration& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:foxglove.CameraCalibration)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool CameraCalibration::IsInitialized() const {
  return true;
}

void CameraCalibration::Swap(CameraCalibration* other) {
  if (other == this) return;
  InternalSwap(other);
}
void CameraCalibration::InternalSwap(CameraCalibration* other) {
  using std::swap;
  d_.InternalSwap(&other->d_);
  k_.InternalSwap(&other->k_);
  r_.InternalSwap(&other->r_);
  p_.InternalSwap(&other->p_);
  distortion_model_.Swap(&other->distortion_model_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  frame_id_.Swap(&other->frame_id_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(timestamp_, other->timestamp_);
  swap(width_, other->width_);
  swap(height_, other->height_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata CameraCalibration::GetMetadata() const {
  protobuf_foxglove_2fCameraCalibration_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_foxglove_2fCameraCalibration_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace foxglove
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::foxglove::CameraCalibration* Arena::CreateMaybeMessage< ::foxglove::CameraCalibration >(Arena* arena) {
  return Arena::CreateInternal< ::foxglove::CameraCalibration >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
