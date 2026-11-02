# 首轮公开农业图像实机评价

本轮实现与操作说明见 [FIELD_VALIDATION.md](FIELD_VALIDATION.md)，架构见 [FIELD_VALIDATION_REVIEW.md](FIELD_VALIDATION_REVIEW.md)。

## 数据和证据

使用 PlantDoc 官方仓库 revision `5467f6012d78d1c446145d5f582da6096f852ae8`。在查看模型结果前固定 seed=41、官方 test、苹果/甜椒/马铃薯/番茄十五个来源类别、每类四张，共 60 张。参考 NORMAL 12 张、SUSPECT 48 张。
来源是作者人工整理的互联网农业图像；这是公开复杂背景外部评测，不是本地田间相机采集。标签采用作者类别，不从本项目模型推断。质量、植物存在性单独判断、光照、距离、角度、背景和植物/会话 ID 未提供，不填造人工值。
下载保留选择清单、每图 URL、Git blob SHA1、SHA256 和 CC-BY-4.0 许可/作者署名。与已有 2,792 张 PlantVillage 图像进行精确文件/像素及 pHash Hamming≤4 审查，本批次匹配数为 0。此审查排查已定义的重复，不证明所有潜在来源关联都不存在。
60/60 图像在 COM3 的真实 ESP32-S3 完成推理，并保存原始 RGB、实际发送 RGB 和结果。再由原图和记录方向/镜像/双线性缩放重建，60 对输入逐像素一致。实际模型和固件哈希、冻结状态规则与 v0.3 相同；没有重新训练、量化、刷写或调阈值。

## 二分类与三状态

| 指标 | 实测 |
| --- | ---: |
| 设备二分类 Accuracy，阈值 0.30 | 88.33% (60 张) |
| Precision / Recall / F1 | 0.9020 / 0.9583 / 0.9293 |
| Specificity / FPR / FNR | 0.5833 / 0.4167 / 0.0417 |
| 接受数 / 覆盖率 | 57/60 / 95.00% |
| 接受准确率 / 错误率 | 89.47% / 10.53% |
| UNKNOWN | 3/60 / 5.00% |
| SUSPECT→NORMAL 漏检 | 1/48 / 2.08% |
| SUSPECT→UNKNOWN 延后 | 2/48 / 4.17% |
| NORMAL 误报 | 5/12 / 41.67% |

| 参考 / 设备 | NORMAL | SUSPECT | UNKNOWN |
| --- | ---: | ---: | ---: |
| NORMAL | 6 | 5 | 1 |
| SUSPECT | 1 | 45 | 2 |

UNKNOWN 不记为正确分类。二分类矩阵为 [[7,5],[2,46]]，接受子集为 [[6,5],[1,45]]。50 张输出 SUSPECT、7 张 NORMAL、3 张 UNKNOWN。类别比例为 12:48，因此总体准确率不能替代正常叶片的误报分析。

## UNKNOWN 的实际作用

本批次三个 UNKNOWN 均来自概率区间，质量拒绝为 0。按同一图的 0.30 强制二分类对照，拦截错误 1 个，延后原本正确 2 个。接受错误从强制二分类 7/60 变为 6/57，覆盖率从 100% 变为 95%。这是当前固定子集上的覆盖—风险取舍，不是通过改阈值取得的优化。

## 作物与场景

| 作物 | NORMAL / SUSPECT 参考数 | 总数 | 接受覆盖率 | 接受准确率 |
| --- | ---: | ---: | ---: | ---: |
| Apple | 4 / 8 | 12 | 100.00% | 100.00% |
| Bell pepper | 4 / 4 | 8 | 87.50% | 71.43% |
| Potato | 0 / 8 | 8 | 87.50% | 100.00% |
| Tomato | 4 / 28 | 32 | 96.88% | 87.10% |

马铃薯子集中没有 NORMAL 参考图，不能由其接受准确率推断正常叶片误报。光照、距离、角度、背景和异常程度均为 UNRECORDED；对应分组 support=unrecorded，不能据此比较这些场景。样本量标签仅表示是否达到计数阈值，不表示独立性或统计显著性。

## 质量门控

本批次 60/60 通过设备门控。独立人工 image_quality 标签为 0，质量评价为 PENDING，模糊/曝光拦截率、正常误拒绝和不合格错放率均为 null。通过率 100% 不能解释为质量检测准确率 100%。标注页面已提供这些维度，补充后可独立生成质量混淆矩阵。

## FP32 / INT8 一致性

同一保存 RGB 上，FP32 与设备状态一致 56/60 (93.33%)，p(SUSPECT) MAE=0.043206，最大绝对差=0.576671。FP32 二分类准确率为 90.00%，设备为 88.33%。

| 样本 | 来源类别 | FP32 p / 状态 | 设备 p / 状态 |
| --- | --- | --- | --- |
| ec26832f92fa4e3094be9db5e28ff828 | Bell_pepper leaf | 0.068178 / NORMAL | 0.201813 / UNKNOWN |
| f6d371cb5cd346fc943a14672c61c31f | Potato leaf late blight | 0.788739 / SUSPECT | 0.212069 / UNKNOWN |
| 598262b4c2c944ce90bc8dbb30b74c99 | Tomato leaf | 0.271499 / UNKNOWN | 0.112795 / NORMAL |
| 03fc2fadd30749b4bb7c019720b6efc0 | Tomato leaf yellow virus | 0.791199 / SUSPECT | 0.307358 / UNKNOWN |

Host Simulation 和三方一致率为 PENDING：没有原正式模型的 frozen .native 图和绑定摘要。不用合成图、FP32 或重新校准图代替。全部逐图差异和来源引用见 [INFERENCE_PARITY.md](../reports/field_validation/INFERENCE_PARITY.md) 与 [SOURCE_MANIFEST.json](../reports/field_validation/SOURCE_MANIFEST.json)。

## 划分、软件与硬件验收

公开数据缺少植物/拍摄会话标识，采用一个保守来源组；新 Field Train/Validation/Test 全部 unassigned，独立分组不足为 PENDING。工具支持多批次按植物/会话/批次、精确和近重复连通分组，并固定 seed，避免相邻帧跨集。

软件、模拟传输/主机 C++ 核心、实际设备和农业数据结果分别列于本轮验收文件；没有远端推送，Actions 未触发。两条软件 warning 来自原 PyTorch ONNX legacy exporter。

| 证据类别 | 结果 | 记录 |
| --- | --- | --- |
| Ruff | All checks passed | [ruff_field.txt](../reports/ruff_field.txt) |
| 软件（含模拟串口、HTTP 和主机 C++ 核心） | 241 passed，0 skipped，5 deselected | [文本](../reports/tests_field_software.txt)、[JUnit](../reports/tests_field_software.xml) |
| 实机 / 模型文件 | 5 passed：3 项真实 ESP32、2 项本地模型文件 | [文本](../reports/tests_field_hardware.txt)、[JUnit](../reports/tests_field_hardware.xml) |
| 公开农业图像与真实 ESP32 | 60/60 返回、全部图像对与原图重建一致 | [逐图结果](../reports/field_validation/INFERENCE_RESULTS.csv)、[来源](../reports/field_validation/SOURCE_MANIFEST.json) |
| 人工质量与受控场景农业验证 | PENDING：公开源没有这些标签 | [质量报告](../reports/field_validation/QUALITY_GATE_REPORT.json)、[场景统计](../reports/field_validation/SCENARIO_STATISTICS.json) |
| 本地真实田间摄像头采集 | 工具与管线已验收，农业拍摄实验 PENDING | 原图采集管线使用生成探针做实机测试，未计入农业准确率 |
| 真实非目标场景误接受率 | PENDING，独立 presence=NO 标注数为 0 | 软件评价与采集/标注维度已具备，来源类别不能替代植物存在性真值 |

普通测试显式 deselect 三项设备和两项本地模型测试，这五项本轮另行执行通过。测试日志的
本机 hostname 和不必要的绝对路径已脱敏。相机采集、标注保存/导出、三状态/质量分母、
来源下载、字节一致性和连通分组均有软件测试；浏览器页面可查看 60 张原图/实际输入，
上一张/下一张验证通过，脚本错误为零。手工标签未因界面测试写入真实农业批次。

## 下一步

先独立复核这 60 张图的质量与目标叶片存在性，检查五张正常误报和一张接受漏检；不按预测改真实标签。随后进行多植物、多拍摄会话的真实摄像头采集，补齐低光/强光、距离、角度、遮挡和非目标场景。每个会话整体分组，数量足够后固定 Validation/Test。恢复原 frozen native 图后再定位量化图误差与设备执行差异。当前正式模型、质量阈值与 UNKNOWN 边界保持冻结。

所有规定的批次 CSV/JSON/Markdown/PNG 见 [汇总目录](../reports/field_validation/README.md)；照片和无损模型输入位于被 Git 忽略的本地批次。文件变化清单见 [FIELD_CHANGESET.md](FIELD_CHANGESET.md)。
