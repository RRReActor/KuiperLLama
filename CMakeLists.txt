cmake_minimum_required(VERSION 3.18)
project(KuiperLLama LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(KUIPER_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
if(KUIPER_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Sanitizers require GCC or Clang")
    endif()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    add_link_options(-fsanitize=address,undefined)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(sentencepiece CONFIG REQUIRED)
option(LLAMA3_SUPPORT "Enable Llama 3 BPE tokenizer" OFF)
option(QWEN2_SUPPORT "Enable Qwen 2 BPE tokenizer" OFF)
option(QWEN3_SUPPORT "Enable Qwen3 model and BPE tokenizer" OFF)

# Compile SentencePiece against its own Abseil headers independently of RE2.
add_library(kuiper_sentencepiece OBJECT kuiper/source/op/encode.cpp)
target_include_directories(kuiper_sentencepiece PRIVATE kuiper/include)
set(KUIPER_SENTENCEPIECE_ABSL_INCLUDE_DIR "" CACHE PATH
    "Include root for the Abseil version used to build SentencePiece")
if(KUIPER_SENTENCEPIECE_ABSL_INCLUDE_DIR)
    get_target_property(sentencepiece_includes sentencepiece::sentencepiece INTERFACE_INCLUDE_DIRECTORIES)
    find_file(sentencepiece_header sentencepiece_processor.h HINTS ${sentencepiece_includes} REQUIRED)
    # The installed header uses quoted Abseil includes, which would otherwise
    # prefer an incompatible Abseil installation alongside it.
    configure_file("${sentencepiece_header}"
        "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece-headers/sentencepiece_processor.h" COPYONLY)
    target_include_directories(kuiper_sentencepiece SYSTEM PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/sentencepiece-headers"
        "${KUIPER_SENTENCEPIECE_ABSL_INCLUDE_DIR}")
endif()
target_link_libraries(kuiper_sentencepiece PRIVATE sentencepiece::sentencepiece glog::glog CUDA::cudart)

add_library(kuiper STATIC
    kuiper/source/base/base.cpp
    kuiper/source/base/alloc.cpp
    kuiper/source/base/alloc_cpu.cpp
    kuiper/source/base/alloc_cuda.cpp
    kuiper/source/base/buffer.cpp
    kuiper/source/base/unicode.cpp
    kuiper/source/base/unicode-data.cpp
    kuiper/source/tensor/tensor.cpp
    kuiper/source/sampler/argmax_sampler.cpp
    kuiper/source/op/layer.cpp
    kuiper/source/op/linear.cpp
    kuiper/source/op/rmsnorm.cpp
    kuiper/source/op/rope.cpp
    kuiper/source/op/activation.cpp
    kuiper/source/op/embedding.cpp
    kuiper/source/op/kernels/kernels_interface.cpp
    kuiper/source/op/kernels/cpu/emb_kernel.cpp
    kuiper/source/op/kernels/cpu/rmsnorm_kernel.cpp
    kuiper/source/op/kernels/cuda/argmax_kernel.cu
    kuiper/source/op/kernels/cuda/emb_kernel.cu
    kuiper/source/op/kernels/cuda/rmsnorm_kernel.cu
    kuiper/source/op/kernels/cuda/rope_kernel.cu
    kuiper/source/op/kernels/cuda/gemv_kernel.cu
    kuiper/source/op/kernels/cuda/add_kernel.cu
    kuiper/source/op/kernels/cuda/activation_kernel.cu)
target_compile_features(kuiper PUBLIC cxx_std_17)
target_include_directories(kuiper PUBLIC kuiper/include kuiper/source)
target_link_libraries(kuiper PUBLIC glog::glog CUDA::cudart re2::re2 PRIVATE kuiper_sentencepiece)
if(QWEN3_SUPPORT)
    target_sources(kuiper PRIVATE kuiper/source/model/model.cpp kuiper/source/model/qwen3.cpp
        kuiper/source/model/raw_model_data.cpp kuiper/source/op/mha.cpp
        kuiper/source/op/kernels/cuda/mha_kernel.cu)
    add_executable(qwen3_infer demo/main_qwen3.cpp demo/qwen3_demo.cpp)
    target_link_libraries(qwen3_infer PRIVATE kuiper)
    add_executable(qwen3_infer_int8 demo/main_qwen3_int8.cpp demo/qwen3_demo.cpp)
    target_link_libraries(qwen3_infer_int8 PRIVATE kuiper)
endif()
if(LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    target_sources(kuiper PRIVATE kuiper/source/op/bpe_encode.cpp)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    target_link_libraries(kuiper PRIVATE nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
    foreach(feature LLAMA3_SUPPORT QWEN2_SUPPORT QWEN3_SUPPORT)
        if(${feature})
            target_compile_definitions(kuiper PUBLIC ${feature})
        endif()
    endforeach()
endif()
target_compile_options(kuiper PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

include(CTest)
if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    add_executable(kuiper_tests tests/argmax_sampler_test.cpp tests/base_test.cpp tests/buffer_test.cpp tests/tensor_test.cpp tests/cuda_test.cpp tests/layer_test.cpp tests/linear_test.cpp tests/gemv_kernel_test.cpp tests/activation_test.cpp tests/embedding_test.cpp tests/rope_test.cpp)
    target_link_libraries(kuiper_tests PRIVATE kuiper GTest::gtest_main)
    add_executable(kuiper_rmsnorm_tests tests/rmsnorm_test.cpp)
    target_link_libraries(kuiper_rmsnorm_tests PRIVATE kuiper GTest::gtest_main)
    add_executable(kuiper_rmsnorm_legacy_tests test/test_op/test_cu_rmsnorm.cpp)
    target_link_libraries(kuiper_rmsnorm_legacy_tests PRIVATE kuiper GTest::gtest_main)
    include(GoogleTest)
    if(QWEN3_SUPPORT)
        add_executable(kuiper_qwen3_encode_test tests/qwen3_encode_test.cpp tests/qwen3_model_test.cpp)
        target_compile_definitions(kuiper_qwen3_encode_test PRIVATE
            QWEN3_TEST_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/model/qwen0.6.bin2"
            QWEN3_TEST_INT8_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/model/qwen0.6.int8.bin"
            QWEN3_TEST_VOCAB="${CMAKE_CURRENT_SOURCE_DIR}/model/tokenizer.json")
        target_link_libraries(kuiper_qwen3_encode_test PRIVATE kuiper GTest::gtest_main)
        gtest_discover_tests(kuiper_qwen3_encode_test DISCOVERY_MODE PRE_TEST)
    endif()
    gtest_discover_tests(kuiper_tests DISCOVERY_MODE PRE_TEST)
    gtest_discover_tests(kuiper_rmsnorm_tests DISCOVERY_MODE PRE_TEST)
    gtest_discover_tests(kuiper_rmsnorm_legacy_tests DISCOVERY_MODE PRE_TEST)
endif()
