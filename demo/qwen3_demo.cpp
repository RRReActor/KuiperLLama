#include "qwen3_demo.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string_view>
#include <tuple>
#include <glog/logging.h>
#include "model/qwen3.h"

namespace {
auto first_token = std::chrono::steady_clock::now();
int32_t output_count = 0;

int32_t generate(const model::Qwen3Model& model, const std::string& prompt, int total_steps, bool need_output = false) {
    auto tokens = model.encode(prompt);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    std::vector<int32_t> words;

    const int context_size = model.get_buffer(model::ModelBufferType::kKeyCache).get_dim(1);
    while (pos < std::min(total_steps, context_size)) {
        pos_tensor.index<int32_t>(0) = pos;
        if (pos < prompt_len - 1) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            CHECK(model.predict(input, pos_tensor, is_prompt, next));
        } else {
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& prompt_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            CHECK(model.predict(input, pos_tensor, is_prompt, next));
            if (output_count++ == 0) first_token = std::chrono::steady_clock::now();
            if (model.is_sentence_ending(next)) {
                ++pos;
                break;
            }
            words.push_back(next);
        }
        pos += 1;
        if (is_prompt) next = tokens.at(pos);
    }

    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    return pos;
}

std::string fill_template(const std::string& content) {
    const std::string format =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    std::string result = format;
    size_t pos = result.find("%s");
    if (pos != std::string::npos) {
        result.replace(pos, 2, content);
    }
    return result;
}

int benchmark(const model::Qwen3Model& model, const std::string& prompt, int new_tokens) {
    const auto prompt_ids = model.encode(prompt);
    const int context = model.get_buffer(model::ModelBufferType::kKeyCache).get_dim(1);
    if (prompt_ids.empty() || prompt_ids.size() > static_cast<size_t>(context) ||
        new_tokens - 1 > context - static_cast<int>(prompt_ids.size())) {
        std::cerr << "Prompt and generation exceed the model context\n";
        return 1;
    }
    auto position = model.get_buffer(model::ModelBufferType::kInputPos);
    auto run = [&](int count) {
        using Clock = std::chrono::steady_clock;
        std::vector<int> generated;
        generated.reserve(count);
        int next = -1;
        const auto start = Clock::now();
        const auto prompt_embedding = model.embedding(prompt_ids);
        for (size_t index = 0; index < prompt_ids.size(); ++index) {
            position.index<int32_t>(0) = static_cast<int32_t>(index);
            auto input = model.fill_input(position, prompt_embedding, true);
            CHECK(model.predict(input, position, index + 1 < prompt_ids.size(), next));
        }
        const auto first = Clock::now();
        generated.push_back(next);
        for (int index = 1; index < count; ++index) {
            position.index<int32_t>(0) = static_cast<int32_t>(prompt_ids.size()) + index - 1;
            auto input = model.fill_input(position, model.embedding({next}), false);
            CHECK(model.predict(input, position, false, next));
            generated.push_back(next);
        }
        const auto end = Clock::now();
        using Milliseconds = std::chrono::duration<double, std::milli>;
        return std::make_tuple(Milliseconds(first - start).count(),
                               Milliseconds(end - first).count() / (count - 1), generated);
    };
    run(2);
    std::cout << "prompt_ids";
    for (int id : prompt_ids) std::cout << ' ' << id;
    std::cout << '\n';
    for (int index = 0; index < 3; ++index) {
        const auto [ttft, tpot, generated] = run(new_tokens);
        std::cout << "sample " << ttft << ' ' << tpot;
        for (int id : generated) std::cout << ' ' << id;
        std::cout << '\n';
    }
    return 0;
}

}  // namespace

int run_qwen3_demo(int argc, char* argv[], bool is_quant_model) {
    const bool benchmark_mode = argc == 6 && std::string_view(argv[5]) == "--benchmark";
    if (argc < 3 || (argc > 5 && !benchmark_mode)) {
        std::cerr << "Usage: " << argv[0] << " CHECKPOINT TOKENIZER [cuda|cpu] [steps] [--benchmark]\n";
        return 1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];
    const std::string_view device = argc >= 4 ? argv[3] : "cuda";
    if (device != "cpu" && device != "cuda") {
        std::cerr << "Expected cpu or cuda\n";
        return 1;
    }
    int total_steps = 1024;
    if (argc >= 5) {
        const std::string_view steps = argv[4];
        const auto end = steps.data() + steps.size();
        const auto parsed = std::from_chars(steps.data(), end, total_steps);
        if (parsed.ec != std::errc{} || parsed.ptr != end || total_steps < (benchmark_mode ? 2 : 1)) {
            std::cerr << (benchmark_mode ? "Expected at least two new tokens\n" : "Expected positive steps\n");
            return 1;
        }
    }

    model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, is_quant_model);
    auto init_status = model.init(device == "cpu" ? base::DeviceType::kDeviceCPU
                                                  : base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        std::cerr << "Qwen3 initialization failed (" << init_status.get_err_code()
                  << ") " << init_status.get_err_msg() << '\n';
        return 1;
    }

    const std::string hi = "什么是AI?";
    if (benchmark_mode) return benchmark(model, hi, total_steps);
    std::cout << hi << "\n";
    const std::string& sentence = fill_template(hi);
    auto start = std::chrono::steady_clock::now();
    fflush(stdout);
    int steps = generate(model, sentence, total_steps, true);
    auto end = std::chrono::steady_clock::now();
    auto duration_total = std::chrono::duration<double>(end - start).count();
    printf("\nsteps:%d\n", steps);
    printf("\nduration:%lf\n", duration_total);
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration_total);
    if (output_count > 0) {
        const double ttft = std::chrono::duration<double>(first_token - start).count();
        const double tpot = output_count > 1 ? (duration_total - ttft) / (output_count - 1) : 0.0;
        printf("\nttft: %lfms tpot: %lfms\n", ttft * 1000, tpot * 1000);
    }
    fflush(stdout);
    return 0;
}
