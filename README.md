# KuiperLLama

KuiperLLama 是一个使用 C++ 和 CUDA 的 Qwen3-0.6B 推理项目，支持在 CPU 和 NVIDIA GPU 上运行 FP32 与 INT8 权重量化模型。仓库包含内存管理、张量、分词器、推理算子、命令行 demo 和回归测试。

## 依赖

| 用途 | 依赖 |
| --- | --- |
| 基础构建 | CMake 3.18+、支持 C++17 且与所用 CUDA Toolkit 兼容的宿主编译器、CUDA Toolkit（`nvcc`、头文件、`cudart`）、glog、RE2、SentencePiece |
| Qwen3 模型和 BPE 分词 | nlohmann_json、ICU（`uc` 和 `i18n`） |
| C++ 测试 | GoogleTest（提供 `GTest::gtest_main`） |
| CUDA 推理和测试 | 可访问的 NVIDIA GPU 和支持所用 CUDA 运行时的驱动 |
| INT8 权重导出 | Python 3.8+、NumPy |
| Python 性能基准 | PyTorch 和支持 Qwen3 的 Transformers。Kuiper 对比脚本还直接使用 NumPy 和 tokenizers。 |
| 显存采样 | Bash、`nvidia-smi`、`stdbuf`、`awk` 和 `mktemp` |

CMake 从本机查找依赖。当前构建同时启用 C++ 和 CUDA，因此即使只运行 CPU 推理，编译也需要 CUDA Toolkit。

自定义安装路径可用 `-DCMAKE_PREFIX_PATH=/path/to/prefix` 指定。若依赖来自已激活的 Conda 环境，可传入 `-DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DICU_ROOT="$CONDA_PREFIX"`。有多套 CUDA 工具包时，可用 `CUDACXX` 选择 `nvcc`，并用 `-DCUDAToolkit_ROOT=/path/to/cuda` 指向同一套工具包。需要指定目标 GPU 时，传入与目标硬件匹配的 `-DCMAKE_CUDA_ARCHITECTURES`。

若 SentencePiece 和 RE2 使用不同版本的 Abseil，导致 SentencePiece 头文件编译冲突，可用 `-DKUIPER_SENTENCEPIECE_ABSL_INCLUDE_DIR=/path/to/abseil-cpp` 指向构建 SentencePiece 时所用的 Abseil 头文件根目录。

## 构建与运行

以下命令在仓库根目录执行。打开 `QWEN3_SUPPORT` 后，CMake 会生成 FP32 和 INT8 两个推理目标。只构建基础静态库时，可设置 `-DQWEN3_SUPPORT=OFF -DBUILD_TESTING=OFF`。

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
    -DQWEN3_SUPPORT=ON -DBUILD_TESTING=OFF
cmake --build build-bench --target qwen3_infer qwen3_infer_int8 --parallel 3
```

### 准备模型

| 文件 | 用途 |
| --- | --- |
| `model/qwen0.6.bin2` | Qwen3-0.6B 的 Kuiper FP32 权重 |
| `model/qwen0.6.int8.bin` | 从上述 FP32 文件导出的 Kuiper INT8 权重 |
| `model/tokenizer.json` | 与权重对应的 Qwen3 分词器 |

请在运行前确认本地有 FP32 权重和分词器。若未生成 INT8 权重，运行以下命令。导出在 CPU 上分块执行，三个参数均可省略。

```sh
python3 tools/export_qwen3_int8.py --input model/qwen0.6.bin2 \
    --output model/qwen0.6.int8.bin --group-size 64
```

### 生成文本

```sh
./build-bench/qwen3_infer model/qwen0.6.bin2 model/tokenizer.json cpu 20
./build-bench/qwen3_infer_int8 model/qwen0.6.int8.bin model/tokenizer.json cuda 64
```

两个程序都接受 `CHECKPOINT TOKENIZER [cuda|cpu] [steps] [--benchmark]`。默认设备为 CUDA，普通生成默认最多执行 1024 步，包含提示词处理，遇到结束 token 可以提前退出。demo 使用内置提示词“什么是AI?”，普通生成会套用对话模板。两种权重均可选择 `cpu` 或 `cuda`。

## 推理性能和显存

### TTFT 与 TPOT

```sh
./build-bench/qwen3_infer model/qwen0.6.bin2 model/tokenizer.json cuda 8 --benchmark
./build-bench/qwen3_infer_int8 model/qwen0.6.int8.bin model/tokenizer.json cuda 8 --benchmark
```

`--benchmark` 直接使用原始提示词，步数参数表示固定生成的新 token 数，最少为 2。程序预热一次，随后输出三行 `sample`，每行依次包含 TTFT（ms）、TPOT（ms/token）和生成的 token ID。计时从已分词的提示词开始，包含 embedding、前向和贪心采样，模型加载和分词在计时区间外。TTFT 衡量取得首个 token 的时间，TPOT 衡量首个 token 之后每个 token 的平均耗时。

```sh
# 使用同一份 Kuiper FP32 权重和分词器比较 Kuiper 与 Transformers
python3 benchmarks/qwen3_fp32.py --device cpu --new-tokens 8

# 独立测量 Transformers 的 Qwen3 FP32 推理
python3 hf_infer/benchmark_qwen3.py --device cpu --new-tokens 8
```

对比脚本报告三轮中位数，并检查两端提示词和生成 token ID 是否一致。独立脚本的 `--model` 接受 Hugging Face 模型 ID 或本地模型目录，默认为 `Qwen/Qwen3-0.6B`。Python 基准也可指定 `--device cuda`，此时需使用支持 CUDA 的 PyTorch。

### 峰值显存

脚本的 FP32 路径使用 `build/qwen3_infer`，INT8 路径使用上面编译的 `build-bench/qwen3_infer_int8`。使用 FP32 采样前，先构建对应目标。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN3_SUPPORT=ON -DBUILD_TESTING=OFF
cmake --build build --target qwen3_infer --parallel 3

./demo/measure_peak_vram.sh 64
./demo/measure_peak_vram.sh --int8 64
```

脚本每 50 毫秒通过 `nvidia-smi` 采样 GPU 0 的整卡显存，报告推理前基线、采样峰值和峰值增量，单位为 MiB。省略步数时上限为 1024。测量期间保持其他 GPU 负载稳定，峰值增量可用于估算本次运行的显存占用。

如果 CUDA 初始化失败，先运行 `nvidia-smi -L` 检查当前进程是否能访问 GPU，再检查 NVIDIA 驱动与程序所用 CUDA 运行时的兼容性。

## 测试

```sh
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Debug \
    -DQWEN3_SUPPORT=ON -DBUILD_TESTING=ON
cmake --build build-tests --parallel 3
ctest --test-dir build-tests --output-on-failure

ctest --test-dir build-tests -R 'Buffer|RMSNorm' --output-on-failure
ctest --test-dir build-tests -R 'Qwen3|GemvCUDA|LinearLayer' --output-on-failure
```

测试覆盖内存与 Buffer 所有权、Tensor、分词、CPU/CUDA 算子以及 Qwen3 预填充和逐 token 解码。GEMV 测试使用 CPU 双精度参考值对照 FP32 和分组 INT8 CUDA 内核。Qwen3 测试从 `model/` 读取分词器和权重，缺少对应权重时会跳过该模型用例。没有可用 GPU 时，设备用例会报告 `Skipped`。

### 消毒器检查

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
    -DKUIPER_ENABLE_SANITIZERS=ON
cmake --build build-sanitize --parallel 3
ctest --test-dir build-sanitize --output-on-failure
```

`KUIPER_ENABLE_SANITIZERS` 在 GCC 或 Clang 的宿主 C++ 代码上启用 ASan 和 UBSan。设备侧可用 NVIDIA Compute Sanitizer 检查。以下命令使用普通 Debug 测试构建。

```sh
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    ./build-tests/kuiper_tests --gtest_filter='CUDA.*:GemvCUDA.*'
compute-sanitizer --tool initcheck --error-exitcode 99 \
    ./build-tests/kuiper_tests --gtest_filter='CUDA.*'
```

如果在 WSL 中复现了 ASan shadow gap 导致的 CUDA 初始化失败，以及 WSL 驱动的泄漏报告，可用以下设置放宽 shadow gap 保护并抑制对应的驱动泄漏报告。记录结果时应注明启用的设置。

```sh
ASAN_OPTIONS=protect_shadow_gap=0 \
LSAN_OPTIONS="suppressions=$PWD/tests/lsan-wsl.supp:print_suppressions=1" \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build-sanitize --output-on-failure
```

## INT8 权重布局

Q、K、V、O、MLP 的 gate、down、up 投影和 `lm_head` 使用 INT8，词嵌入、RMSNorm 权重（含 Q/K norm）、激活和 KV cache 使用 FP32。CPU 与 CUDA 的量化 GEMV 使用 FP32 累加。CUDA 内核在满足条件时对 64 元素分组采用向量化读取。

- 文件头包含 9 个小端 `int32`，依次为 `dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len, intermediate_dim, group_size`。前八项与 FP32 格式相同。
- 张量按 input norm、post-attention norm、final norm、embedding、Q、Q norm、K、K norm、V、O、gate、down、up、`lm_head` 排列。同类张量按层号升序存放。
- 量化矩阵按行优先展平后连续分组。组大小须整除每个量化矩阵的元素数。每组使用 FP32 scale `max(abs(w)) / 127`，量化编码取最近整数（中点取偶数），限幅至 `[-127, 127]`。全零组的编码和 scale 均为零。
- 每个含 N 个元素的量化矩阵写入 N 个 INT8 编码，随后紧跟 `N / group_size` 个小端 FP32 scale。第 i 个元素的还原值为 `q[i] * scale[i // group_size]`。其余张量沿用 FP32 文件的字节布局。

## 代码位置

- [kuiper/include/](kuiper/include/) 和 [kuiper/source/](kuiper/source/) 包含库的接口和实现，按 `base`、`tensor`、`op`、`model` 和 `sampler` 划分。
- [model.cpp](kuiper/source/model/model.cpp) 将 checkpoint 映射到内存并解析文件头。[qwen3.cpp](kuiper/source/model/qwen3.cpp) 按偏移为各层绑定权重。[layer.cpp](kuiper/source/op/layer.cpp) 构造 INT8 权重和 FP32 scale 张量并负责上传到 CUDA。
- [demo/](demo/) 包含两个推理入口、共享的生成逻辑和显存采样脚本。
- [tests/](tests/) 存放当前测试，[test/](test/) 中的旧版 CUDA RMSNorm 测试也由 CTest 注册。
