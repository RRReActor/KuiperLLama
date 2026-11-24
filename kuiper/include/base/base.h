#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <glog/logging.h>

namespace model {
    enum class ModelBufferType {
        kInputTokens = 0,
        kInputEmbeddings = 1,
        kAttnRMSNorm = 2,
        kKeyCache = 3,
        kValueCache = 4,
        kQuery = 5,
        kInputPos = 6,
        kScoreStorage = 7,
        kMHAOutput = 8,
        kAttnOutput = 9,
        kWGateOutput = 10,
        kWDownOutput = 11,
        kWUpOutput = 12,
        kFFNRMSNorm = 13,
        kLMOutput = 15,
        kLMOutputCPU = 16,

        kSinCache = 17,
        kCosCache = 18,
    };
}

namespace base {
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3
};

enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
    kModelTypeQwen3 = 2,        
};

inline std::size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;
        NoCopyable(const NoCopyable&) = delete;
        NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplemented = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7
};

enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
    kENcodeBpe = kEncodeBpe
};

class Status {
    public:
        Status(int code = StatusCode::kSuccess, std::string message = "");
        Status(const Status&) = default;
        Status& operator=(const Status&) = default;
        Status& operator=(int code);
        bool operator==(int code) const;
        bool operator!=(int code) const;
        operator int() const;
        operator bool() const;
        int32_t get_err_code() const;
        const std::string& get_err_msg() const;
        void set_err_msg(const std::string& message);
    private:
        int code_ = StatusCode::kSuccess;
        std::string message_;
};
    namespace error {
        Status Success(const std::string& err_msg = "");
        
        Status FunctionNotImplement(const std::string& err_msg = "");

        Status PathNotValid(const std::string& err_msg = "");

        Status ModelParseError(const std::string& err_msg = "");

        Status InternalError(const std::string& err_msg = "");

        Status KeyHasExits(const std::string& err_msg = "");

        Status InvalidArgument(const std::string& err_msg = "");
    } // namespace error

    std::ostream& operator<<(std::ostream& os, const Status& x);

} // namespace base