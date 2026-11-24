# KuiperLLama

C++ CPU/CUDA 内存管理及张量基础模块。

## 依赖

- CMake 3.18 或更新版本、支持 C++17 的编译器。
- CUDA Toolkit（包含 nvcc、头文件与 cudart，构建启用 CMake CUDA 语言）。
- Google glog。
- GoogleTest（启用测试时需要，必须提供 `GTest::gtest_main`）。

依赖通过 CMake 查找，不会在配置时自动下载。非标准安装路径可通过
`-DCMAKE_PREFIX_PATH=/path/to/prefix` 和 `-DCUDAToolkit_ROOT=/path/to/cuda` 指定。

## 编译与测试

在仓库根目录执行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

构建包含基础模块、张量、算子及 CUDA 内核，生成静态库 `build/libkuiper.a` 和测试程序
`build/kuiper_tests`。仅构建库时设置 `-DBUILD_TESTING=OFF`。

```sh
# 只运行 Buffer 测试（包括非法输入检查）。
./build/kuiper_tests --gtest_filter='Buffer*'

# 1024 行 × 1024 列 RMSNorm 正确性测试（CPU/CUDA）。
./build/kuiper_tests --gtest_filter='RMSNorm*.Matrix1024By1024'

# 地址和未定义行为消毒器检查。
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTING=ON -DKUIPER_ENABLE_SANITIZERS=ON
cmake --build build-sanitize -j2
ctest --test-dir build-sanitize --output-on-failure
```

测试覆盖状态类型、CPU 分配器、Buffer 生命周期及复制、Tensor 元数据及独立克隆。
Buffer 的设备方向分派测试使用记录型分配器；真实 CUDA 测试执行设备迁移、复制、
清零和流操作。没有可访问的 GPU 时，CUDA 测试会显示为 `Skipped`，其余测试仍可运行；
完整编译始终需要 CUDA Toolkit。

受限或调试环境中，LeakSanitizer 可能无法工作。仅在确认是该环境限制时使用
`ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-sanitize --output-on-failure`，
此时不验证内存泄漏，地址及未定义行为检查仍启用。

### CUDA 消毒器检查

使用普通 Debug 构建执行设备内存检查（需要 NVIDIA Compute Sanitizer）：

```sh
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    ./build/kuiper_tests --gtest_filter='CUDA.*'
compute-sanitizer --tool initcheck --error-exitcode 99 \
    ./build/kuiper_tests --gtest_filter='CUDA.*'
```

`CUDA.AllocatorPoolCleanup` 覆盖大小两类缓存池的复用和析构清理；
是否存在显存泄漏由 memcheck 判定。

在本机 WSL 环境中，默认 ASan 设置使 CUDA 初始化报 `out of memory`；
`protect_shadow_gap=0` 可使初始化正常，但会放宽 ASan shadow gap 保护。
仅调用 `cudaGetDeviceCount` 和 `cudaDeviceReset` 的独立程序也会报告
WSL 驱动内部 300 字节主机内存泄漏。遇到相同问题时可使用以下命令，
仅抑制来自该驱动的泄漏报告，保留项目代码的泄漏检测：

```sh
ASAN_OPTIONS=protect_shadow_gap=0 \
LSAN_OPTIONS="suppressions=$PWD/tests/lsan-wsl.supp:print_suppressions=1" \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build-sanitize --output-on-failure
```

此规则仅用于已复现该问题的 WSL 驱动环境；报告通过结果时应注明使用了抑制规则。

### GEMV 内核测试

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build -j2
./build/kuiper_tests --gtest_filter='GemvCUDA.*'
ctest --test-dir build --output-on-failure

# 有可访问 GPU 时检查越界与未对齐访问。
compute-sanitizer --tool memcheck --error-exitcode 99 \
    ./build/kuiper_tests --gtest_filter='GemvCUDA.*'
```

`tests/gemv_kernel_test.cpp` 使用 CPU 双精度参考值验证 FP32 和 INT8 分组量化
GEMV，覆盖多种行列数（含非 4 对齐列数、1024×1024）、跨行量化分组、
默认流和非阻塞自定义流，以及零输入覆盖输出。量化 scale 按展平后的权重索引分组。
没有可访问 GPU 时测试明确跳过；跳过不代表已验证内核数值正确性。

### Qwen3 FP32 调试

`QWEN3_SUPPORT` 会将 Qwen3 模型、BPE tokenizer 和注意力算子编入库。构建依赖
SentencePiece、RE2，启用 Qwen3 时还需要 nlohmann_json 和 ICU（uc、i18n）。
本机的 RE2、JSON 和 ICU 位于 Conda 环境，可在仓库根目录执行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
    -DQWEN3_SUPPORT=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DICU_ROOT="$CONDA_PREFIX"
cmake --build build -j3
ctest --test-dir build -R Qwen3 --output-on-failure
```

模型回归测试使用 `model/tokenizer.json` 和 `model/qwen0.6.bin2`，逐位置核对
预填充与单 token 解码的 logit 和采样 ID。CPU 可运行整个 FP32 流程。
有可用 GPU 时，同一测试也会验证 CUDA 路径。

### 编译并运行 Qwen3 demo

启用 `QWEN3_SUPPORT` 后，可单独编译 `demo/main_qwen3.cpp` 对应的可执行目标。

```sh
cmake -S . -B build -DQWEN3_SUPPORT=ON
cmake --build build --target qwen3_infer -j3
./build/qwen3_infer model/qwen0.6.bin2 model/tokenizer.json
```

默认使用 CUDA。可在第三个参数指定 `cpu`，在第四个参数指定步数上限（包含提示词）。

```sh
./build/qwen3_infer model/qwen0.6.bin2 model/tokenizer.json cpu 20
```

### Qwen3-0.6B FP32 性能对比

在仓库根目录编译 Release 版本，然后使用装有 PyTorch、Transformers、NumPy 和
tokenizers 的 Python 环境运行基准。本次验证使用 Transformers 4.57.6。

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DQWEN3_SUPPORT=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DICU_ROOT="$CONDA_PREFIX"
cmake --build build-bench --target qwen3_infer -j3
python benchmarks/qwen3_fp32.py --device cpu --new-tokens 8
```

如果本机的 SentencePiece 使用了独立版本的 Abseil，可在配置时用
`-DKUIPER_SENTENCEPIECE_ABSL_INCLUDE_DIR=/path/to/abseil-cpp` 指定对应头文件根目录。
具备 CUDA 运行环境时可指定 `--device cuda`。

两端使用同一份 `model/qwen0.6.bin2` FP32 权重和 `model/tokenizer.json`，直接输入
`什么是AI?`，每轮执行固定长度的贪心解码。CPU 测试使用单计算线程。每端预热一次，
报告三轮的中位数，并检查提示词和生成 token ID 是否一致。
TTFT 从输入已分词的提示词计时到取得第一个 token，单位为 ms。TPOT 用第一个 token
到最后一个 token 的时间除以其余 token 数，单位为 ms/token。模型加载和输出文本解码在计时区间外。

### Qwen3-0.6B INT8 weight-only 权重导出

在仓库根目录使用 Python 3.8+ 和 NumPy 运行以下命令。输入采用参考工具
`tools/export_qwen3/write_bin.py` 的 FP32 格式，仓库自带的权重可直接使用。

```sh
python3 tools/export_qwen3_int8.py --input model/qwen0.6.bin2 \
    --output model/qwen0.6.int8.bin --group-size 64
```

这三个参数也都是默认值。导出在 CPU 上分块执行。注意力的 Q、K、V、O 投影、
MLP 的 gate、down、up 投影和 `lm_head` 均使用 INT8。词嵌入和全部 RMSNorm
权重（含 Q/K norm）保留 FP32。

**二进制布局**

- 文件头为 9 个小端 int32，依次为 `dim, hidden_dim, layer_num, head_num,
  kv_head_num, vocab_size, seq_len, intermediate_dim, group_size`。前八项直接复制自输入。
- 张量沿用 FP32 文件的顺序，同类张量按层号升序排列。依次为 input norm、post-attention
  norm、final norm、embedding、Q、Q norm、K、K norm、V、O、gate、down、up、lm_head。
- 每个量化矩阵按行优先展平，按 `group_size` 连续分组。组大小须整除每个量化矩阵的元素数。
  每组使用 FP32 scale `max(abs(w)) / 127`，取最近整数（中点取偶数），限幅至
  `[-127, 127]`。全零组的编码与 scale 均为零。
- 含 N 个元素的量化矩阵先写 N 个 int8，再写 N / group_size 个小端 float32 scale，
  无填充。展平索引 i 处的还原值为 `q[i] * scale[i // group_size]`。其余张量按原字节写入。

### Qwen3-0.6B INT8 推理与性能测试

`qwen3_infer_int8` 通过 `Qwen3Model(..., true)` 加载上述导出文件。CPU 和 CUDA
均按组读取 INT8 权重与 scale，在 GEMV 中以 FP32 累加。激活与 KV cache 使用 FP32。
默认的 64 元素分组在 Qwen3 的 CUDA 投影中使用向量化读取，其他分组使用通用内核。

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DQWEN3_SUPPORT=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DICU_ROOT="$CONDA_PREFIX"
cmake --build build-bench --target qwen3_infer_int8 qwen3_infer -j3

# 使用对话模板生成文本，64 是包含提示词的最大步数。
./build-bench/qwen3_infer_int8 model/qwen0.6.int8.bin model/tokenizer.json cuda 64

# 使用相同提示词分别测量 INT8 和 FP32，固定生成 8 个 token。
./build-bench/qwen3_infer_int8 model/qwen0.6.int8.bin model/tokenizer.json cuda 8 --benchmark
./build-bench/qwen3_infer model/qwen0.6.bin2 model/tokenizer.json cuda 8 --benchmark
```

将 `cuda` 改为 `cpu` 可在主机上运行。两个 demo 共用同一套分词、生成和计时代码。
`--benchmark` 使用原始提示词“什么是AI?”，预热一次后输出三行 `sample`。
每行依次给出 TTFT（ms）、TPOT（ms/token）和生成的 token ID。计时从已分词的
提示词开始，包含 embedding、模型前向和贪心采样。模型加载耗时单独排除。

启用测试并生成默认分组的模型文件后，可运行
`ctest --test-dir build -R 'Qwen3Model|GemvCUDA|LinearLayer' --output-on-failure`。
INT8 模型测试核对预填充和解码位置的 logits 与贪心结果。
