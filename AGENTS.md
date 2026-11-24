# Repository Guidelines

## 项目结构与模块划分

本仓库是 KuiperLLama 的早期 C++ 实现，目前主要包含 CPU/CUDA 内存管理基础设施。

- `kuiper/include/base/`：状态类型、设备分配器和缓冲区的公共头文件。
- `kuiper/source/base/`：基础模块实现。其中 `alloc_cpu.cpp` 负责主机内存分配，`alloc_cu.cpp` 管理 CUDA 内存池，`alloc.cpp` 实现内存复制和清零。
- `kuiper/source/tensor/tensor.cpp`：目前为空，预留给张量功能。
- `demo/`：目前为空，用于存放可运行示例。

仓库目前没有测试目录、资源文件或构建配置。

## 构建、测试与开发命令

在仓库根目录执行命令。需要安装 C++ 编译器和 Google glog 头文件；CUDA 源码还依赖 CUDA 工具包及运行时库。

```sh
# 检查缓冲区代码的语法和编译警告，不执行链接。
g++ -std=c++11 -Wall -Wextra -Werror -Ikuiper/include \
    -fsyntax-only kuiper/source/base/buffer.cpp

# 检查 CPU 分配器代码的语法。
g++ -std=c++11 -Wall -Wextra -Ikuiper/include \
    -fsyntax-only kuiper/source/base/alloc_cpu.cpp
```

上述命令要求依赖已安装。目前没有完整构建流程、测试运行器或应用入口，请勿假设已有可用的 `cmake`、`make` 或示例运行命令。

## 编码风格与命名约定

遵循相邻 C++ 代码的风格：使用四空格缩进，左花括号放在声明所在行，基础设施代码放入 `base` 命名空间。类名使用 `PascalCase`，方法和局部变量使用 `snake_case`，成员变量以下划线结尾，枚举值以 `k` 开头。

头文件使用 `#pragma once`，项目头文件使用 `base/buffer.h` 这样的包含路径。除非明确修改接口，否则保留现有 API 拼写。仓库尚未配置格式化或静态检查工具。

## 测试要求

目前没有指定测试框架或覆盖率门槛。修改内存管理代码时，应验证所有权、重复分配、分配失败、外部缓冲区、空指针输入及复制方向。

CPU 测试可使用记录分配、释放及复制操作的测试分配器；CUDA 行为需在具备相应硬件的环境中单独验证。新增测试放入 `tests/`，文件名使用 `*_test.cpp`，并记录编译和运行命令。条件允许时使用地址消毒器和未定义行为消毒器，并明确说明环境限制。

## 提交与拉取请求要求

当前检出环境无法读取 Git 历史，因此不能确认既有提交规范。建议使用简短、明确的动作描述作为提交标题，例如 `Fix buffer ownership on repeated allocation`，每次提交聚焦一个问题。

拉取请求应说明缺陷、修改后的行为、验证命令及结果，以及依赖或硬件限制；存在关联问题时附上链接。
