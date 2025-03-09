library(readxl)
library(effsize)

# 读取Excel文件
file_path <- "D:/rpdc.xlsx"  # 请替换为你的文件路径
data <- read_excel(file_path)

# 提取基线条件和非基线条件的数据
baseline <- data$`基线条件`
non_baseline <- data$`非基线条件`

# 执行配对t检验
t_test_result <- t.test(baseline, non_baseline, paired = TRUE)

# 计算Cohen's d
cohen_d_value <- cohen.d(baseline, non_baseline, paired = TRUE)$estimate

# 生成报告
report <- paste("配对t检验结果:\n",
                "t-value: ", round(t_test_result$statistic, 3), "\n",
                "p-value: ", round(t_test_result$p.value, 5), "\n",
                "95%置信区间: [", round(t_test_result$conf.int[1], 3), ", ", round(t_test_result$conf.int[2], 3), "]\n",
                "Cohen's d 值: ", round(cohen_d_value, 3))

# 打印报告
cat(report)
