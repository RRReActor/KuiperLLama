#include "op/layer.h"
#include "base/cuda_config.h"
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <limits>
#include <utility>

namespace op {
    BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, 
        base::DataType data_type, std::string layer_name):
            device_type_(device_type),
            data_type_(data_type),
            layer_type_(layer_type),
            layer_name_(std::move(layer_name)) {}
    
    base::DataType BaseLayer::data_type() const { return data_type_;}
    
    LayerType BaseLayer::layer_type() const { return layer_type_;}

    base::Status BaseLayer::set_weight(int32_t, const tensor::Tensor&) {
        return base::error::FunctionNotImplement();
    }

    base::Status BaseLayer::set_weight(int32_t, const std::vector<int32_t>&, 
                const void*, base::DeviceType) {
                    return base::error::FunctionNotImplement();
                }

    const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

    void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }
    base::DeviceType BaseLayer::device_type() const { return device_type_; }

    void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

    Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
        : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

    base::Status Layer::init() { return base::error::Success(); }

    base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

    base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                    base::DataType data_type) const {
    if (tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }
    return base::error::Success();
    }
    
    base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                            base::DeviceType device_type, base::DataType data_type,
                                            ...) const {
        std::va_list args;
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor has a wrong device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor has a wrong data type.");
        }

        va_start(args, data_type);
        int32_t dims = tensor.dims_size();
        for (int32_t i = 0; i < dims; ++i) {
            int32_t dim = va_arg(args, int32_t);
            if (dim != tensor.get_dim(i)) {
            va_end(args);
            return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
            }
        }
        va_end(args);
        return base::error::Success();
    }

    void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    this->inputs_.at(idx) = input;
    }

    void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    this->outputs_.at(idx) = output;
    }

    const tensor::Tensor& Layer::get_input(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_input(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_output(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
    }

    base::Status Layer::check() const {
    return base::error::FunctionNotImplement("The check function is not implement yet");
    }

    const tensor::Tensor& Layer::get_output(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
    }

    void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

    void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

    void Layer::to_cuda() {
    for (auto& input : inputs_) {
        if (!input.is_empty()) {
        input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
    for (auto& output : outputs_) {
        if (!output.is_empty()) {
        output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
    }

    void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
        if (!config) {
            return;
        }
        this->cuda_config_ = config;
    }
    
    std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

    
    size_t Layer::input_size() const { return inputs_.size(); }

    size_t Layer::output_size() const { return outputs_.size(); }

    LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                        std::string layer_name)
        : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

    base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
        if (!weight.is_empty()) {
            CHECK(weight.device_type() == device_type_);
        }
        weights_.at(idx) = weight;
        return base::error::Success();
    }
    
    base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight,
                                       const tensor::Tensor& scales) {
        CHECK(is_quant_layer_);
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK(!weight.is_empty());
        CHECK(weight.data_type() == base::DataType::kDataTypeInt8);
        CHECK(weight.device_type() == device_type_);
        CHECK(!scales.is_empty());
        CHECK(scales.data_type() == base::DataType::kDataTypeFp32);
        CHECK(scales.device_type() == device_type_);
        CHECK_GT(group_size_, 0);
        CHECK_EQ(weight.size() % static_cast<size_t>(group_size_), 0);
        CHECK_EQ(scales.size(), weight.size() / static_cast<size_t>(group_size_));

        weights_.at(idx) = weight;
        scales_ = scales;
        return base::error::Success();
    }

    const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

    void LayerParam::to_cuda() {
        Layer::to_cuda();
        for (auto& weight : weights_) {
            if (!weight.is_empty()) {
                weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
            }
        }
        if (!scales_.is_empty()) {
            scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }

    
    base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                        const void* weight_ptr, base::DeviceType device_type) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK_NE(weight_ptr, nullptr);

        CHECK(!dims.empty());
        size_t size = is_quant_layer_ ? sizeof(int8_t) : sizeof(float);
        for (int32_t dim : dims) {
            CHECK_GT(dim, 0);
            CHECK_LE(static_cast<size_t>(dim), std::numeric_limits<size_t>::max() / size);
            size *= static_cast<size_t>(dim);
        }
        auto view = [&](base::DataType type, const std::vector<int32_t>& shape, const void* ptr) {
            tensor::Tensor tensor(type, shape, false, nullptr, const_cast<void*>(ptr));
            tensor.set_device_type(device_type);
            // The checkpoint has no padding after an odd number of INT8 codes.
            if (type == base::DataType::kDataTypeFp32 && device_type == base::DeviceType::kDeviceCPU &&
                reinterpret_cast<std::uintptr_t>(ptr) % alignof(float) != 0) {
                return tensor.clone();
            }
            return tensor;
        };
        if (is_quant_layer_) {
            if (group_size_ <= 0 || size % group_size_ != 0 ||
                size / group_size_ > std::numeric_limits<int32_t>::max()) {
                return base::error::InvalidArgument("Invalid packed INT8 weight or group size.");
            }
            const auto* scales = static_cast<const int8_t*>(weight_ptr) + size;
            scales_ = view(base::DataType::kDataTypeFp32, {static_cast<int32_t>(size / group_size_)}, scales);
            weights_.at(idx) = view(base::DataType::kDataTypeInt8, dims, weight_ptr);
        } else {
            weights_.at(idx) = view(base::DataType::kDataTypeFp32, dims, weight_ptr);
        }
        return base::error::Success();
    }

    void LayerParam::set_scales(const tensor::Tensor& scales) {
        CHECK(!scales.is_empty());
        CHECK(scales.data_type() == base::DataType::kDataTypeFp32);
        CHECK(scales.device_type() == device_type_);
        this->scales_ = scales;
    }

    void LayerParam::set_group_size(int32_t group_size) {
        CHECK_GT(group_size, 0);
        this->group_size_ = group_size;
    }

    size_t LayerParam::weight_size() const { return weights_.size(); }

    void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

    int32_t LayerParam::get_scale_num() const { return static_cast<int32_t>(scales_.size()); }

    base::Status Layer::forward(const tensor::Tensor& input, tensor::Tensor output) {
        this->set_input(0, input);
        this->set_output(0, output);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2, tensor::Tensor output) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_output(0, output);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                const tensor::Tensor& input3, tensor::Tensor output) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_output(0, output);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                const tensor::Tensor& input3, const tensor::Tensor& input4, tensor::Tensor output) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_output(0, output);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                const tensor::Tensor& input3, const tensor::Tensor& input4, const tensor::Tensor& input5, tensor::Tensor output) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);
        this->set_output(0, output);
        return this->forward();
    }

    tensor::Tensor& LayerParam::get_weight(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

}