#include "base/base.h"
#include <string>

namespace base {
Status::Status(int code, std::string message) : code_(code), message_(message) {}

Status& Status::operator=(int code) {
    code_ = code;
    return *this;
}

bool Status::operator==(int code) const {
    return code_ == code;
}

bool Status::operator!=(int code) const {
    return code_ != code;
}

Status::operator int() const {
    return code_;
}

Status::operator bool() const {
    return code_ == StatusCode::kSuccess;
}

int32_t Status::get_err_code() const {
    return code_;
}

const std::string& Status::get_err_msg() const {
    return message_;
}

void Status::set_err_msg(const std::string& message) {
    message_ = message;
}

namespace error {
Status Success(const std::string& message) { return Status(kSuccess, message); }
Status FunctionNotImplement(const std::string& message) { return Status(kFunctionUnImplemented, message); }
Status PathNotValid(const std::string& message) { return Status(kPathNotValid, message); }
Status ModelParseError(const std::string& message) { return Status(kModelParseError, message); }
Status InternalError(const std::string& message) { return Status(kInternalError, message); }
Status KeyHasExits(const std::string& message) { return Status(kKeyValueHasExist, message); }
Status InvalidArgument(const std::string& message) { return Status(kInvalidArgument, message); }
} // namespace error

std::ostream& operator<<(std::ostream& os, const Status& status) {
    return os << status.get_err_code() << ": " << status.get_err_msg();
}

} // namespace base