#include "op/encode.h"
#include <gtest/gtest.h>

TEST(Qwen3Encode, MatchesTokenizerJson) {
    op::QwenEncodeLayer tokenizer(QWEN3_TEST_VOCAB, false, false);

    EXPECT_EQ(tokenizer.encode("a  b"), (std::vector<int32_t>{64, 220, 293}));
    EXPECT_EQ(tokenizer.encode("  hello"), (std::vector<int32_t>{220, 23811}));
    EXPECT_EQ(tokenizer.encode("def f(x):\n    return x+1"),
              (std::vector<int32_t>{750, 282, 2075, 982, 262, 470, 856, 10, 16}));

    const auto ids = tokenizer.encode("Cafe\u0301<|im_end|>\u1100\u1161");
    EXPECT_EQ(ids, (std::vector<int32_t>{34, 2577, 963, 151645, 19969}));
    EXPECT_EQ(tokenizer.decode(ids), "Café<|im_end|>가");
}
