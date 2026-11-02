# 农业图像批次评价

批次：20261102_230817_98a40f97；开始：2026-11-02T15:08:17.388581+00:00
来源：public_dataset；设备证据：hardware；评价标签：source
Git：3ae13554d3716269eaf488f1cf31f751817060dc；dirty=True
固件 / 模型：0.2.0 / 0.1.0
模型 SHA：`bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa`
固件 SHA：`bab257fda4b57c0f922cc86a8ec76e9037135208c3ba7adda93957eb6711f622`

| 项目 | 数值 |
| --- | ---: |
| 总图片 / 成功 / 排除 | 60 / 60 / 0 |
| 独立参考标签 / 人工复核 | 60 / 0 |
| 概率可用二分类 Accuracy / n | 0.8833 / 60 |
| 接受覆盖率 | 0.9500 |
| 接受输入准确率 | 0.8947 |
| 接受输入错误率 | 0.1053 |
| UNKNOWN 拒绝率 | 0.0500 |
| SUSPECT→NORMAL / SUSPECT→UNKNOWN | 0.0208 / 0.0417 |
| 正常叶片误报率 | 0.4167 |
| 人工质量评价 | PENDING，n=0 |
| 设备推理 P50 / P95 ms | 523.1980 / 523.2453 |

## 指标口径

UNKNOWN 计为拒绝，不计为正确分类。二分类使用冻结 0.30；系统使用冻结 0.20/0.40。
概率可用二分类、质量通过、已接受与人工 USABLE 的分母分别保存，不能相互替代。
质量拒绝设备没有执行模型，其反事实分类未知。概率区间拒绝可对照同一输入强制二分类。

## UNKNOWN

```json
{
  "count": 3,
  "quality_rejected": 0,
  "probability_rejected": 3,
  "other_rejected": 0,
  "boundary_forced_binary_errors_intercepted": 1,
  "boundary_forced_binary_correct_withheld": 2,
  "forced_binary_error_rate_probability_cohort": 0.1166666666666667,
  "quality_rejection_counterfactual": "not observed: device model was not executed"
}
```

## 内存

```json
{
  "free_internal": {
    "minimum": 286447,
    "maximum": 286447
  },
  "free_psram": {
    "minimum": 5046852,
    "maximum": 5046852
  }
}
```

## 推理环境对照

```json
{
  "samples": 60,
  "scenarios": {
    "UNRECORDED": 60
  },
  "fp32_vs_host": {
    "state_pairs": 0,
    "probability_pairs": 0,
    "state_agreement": "PENDING",
    "suspect_probability_mae": "PENDING",
    "suspect_probability_max_error": "PENDING"
  },
  "fp32_vs_device": {
    "state_pairs": 60,
    "probability_pairs": 60,
    "state_agreement": 0.9333333333333333,
    "suspect_probability_mae": 0.04320632386225043,
    "suspect_probability_max_error": 0.5766705721616745
  },
  "host_vs_device": {
    "state_pairs": 0,
    "probability_pairs": 0,
    "state_agreement": "PENDING",
    "suspect_probability_mae": "PENDING",
    "suspect_probability_max_error": "PENDING"
  },
  "three_way_state_agreement": {
    "count": 0,
    "rate": "PENDING"
  },
  "saturation_fraction": {
    "p50": 0.02234375,
    "p95": 0.5753671874999999
  },
  "failures": []
}
```

## 失败与延后判断样本

| Sample | 参考 | 设备 |
| --- | --- | --- |
| 88a5742453ad4bc49a49e169048a332e | NORMAL | SUSPECT |
| ec26832f92fa4e3094be9db5e28ff828 | NORMAL | UNKNOWN |
| c37a3075613f46e6a7aa80d2838d03ca | NORMAL | SUSPECT |
| f6d371cb5cd346fc943a14672c61c31f | SUSPECT | UNKNOWN |
| 0c65dd29e0244b798d7230af8ea6fefa | NORMAL | SUSPECT |
| b831095ee1944e8eb25563a5530dbe40 | NORMAL | SUSPECT |
| 5bcbe43a3660422295e6c6d8f95266b1 | NORMAL | SUSPECT |
| f6668cdc792d4932b9c18294d1102a9f | SUSPECT | NORMAL |
| 03fc2fadd30749b4bb7c019720b6efc0 | SUSPECT | UNKNOWN |

## 实验范围

来源标签是独立数据集作者的类别或显式人工标注；预测不生成真实标签。
人工质量/光照/距离未记录时相应评价为 PENDING 或 UNRECORDED；小场景只报告计数与样本不足。
公开互联网农业图像与本地田间摄像头采集分别记录；诊断性子集指标不外推到完整田间总体。
植物存在性、完整 OOD 和具体病害诊断不由质量门控提供。
执行错误单列，不转成 UNKNOWN。分组划分结果单独保存，组不足时不制造独立验证/测试集合。
