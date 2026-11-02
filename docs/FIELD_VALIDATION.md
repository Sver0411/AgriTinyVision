# 采集、独立标注与三状态评价

本阶段复用 v0.3 摄像头、单帧在途通信、设备身份和冻结参考后端。正式模型、质量门控、
0.30 阈值和 float32 0.20/0.40 边界保持一致。数据使用普通文件，不使用数据库。

## 摄像头采集

1. 启动 `python scripts/run_pc_camera.py --port COM3`，在设备工作台连接相机和 ESP32。
2. 打开 [采集与标注](http://127.0.0.1:8765/field)。填写作物、植物个体 ID、拍摄会话、
   场景、光照、距离、角度和背景，设置间隔后创建批次。未观察的条件留空。
3. “采集一张”保存一次；“按间隔采集”持续执行；“暂停”后可继续；“结束批次”结束本次。
   单帧在途，设备处理期间不堆积待处理图片。实际采集数和完成数分别显示。
4. 暂停并等待当前帧完成，下载“批次（含照片）”。普通工作台推理保持不存图。

每批次在 `runs/field_validation/<run_id>/` 保存：

| 路径 | 内容 |
| --- | --- |
| batch.json | 开始/结束、来源、项目/Git、设备身份、模型/固件摘要、配置 |
| images/original | 原始 BGR 转回 RGB 的无损 PNG，保持原始尺寸 |
| images/model_input | 同一次发送的 160×160 RGB 无损 PNG；不使用 JPEG 预览替代 |
| metadata | 每图时间、尺寸、配置、场景、原图文件/像素 SHA、输入文件/像素 SHA |
| predictions | 成功设备结果或明确的执行错误；CRC、序列、时间、内存、质量、概率与饱和 |
| annotations | manual.jsonl 为独立人工记录，source.jsonl 为公开来源标签 |
| reports | 统计、CSV、Markdown 和 PNG |

本地相机时间为 PC 读取开始时间，非传感器硬件曝光时间。公开数据的 captured_at 为入库/
回放时间，页面显示“入库”；原始照片拍摄时间、植物个体和场景条件不从文件名猜测。
程序出错属于 status=error/pending，不能视为设备 UNKNOWN；原图和输入在发送前保存，
已知通信错误另存 error，不生成预测概率。原始图片与运行目录默认被 Git 忽略。

## 独立人工标注

选择批次即可对照原图、实际模型输入、预测与质量指标。人工表单独立填写，不从预测或
来源标签自动填充。三类模型输出和人工 UNCERTAIN / NOT_APPLICABLE 标签属于不同维度。
填写目标叶片存在性、人工图像质量、视觉标签、判断是否确定；作物、环境、个体/会话、
异常程度与备注可补充。病害名称需填写可靠依据，不能以模型二分类结果推断病名。

保存后以追加 JSONL 保留修订；读取每张图最后一次记录。上一张/下一张切换前会保存已填写
且有效的修改；页面未保存时离开会提示。批次列表显示人工完成数量，可导出 CSV。

也可使用 CSV：

```powershell
python scripts/field_validation.py annotation-template --batch BATCH_DIRECTORY --file annotations.csv
# 用独立观察填写 CSV；judgment_certain / expert_confirmed 使用 true / false
python scripts/field_validation.py import-annotations --batch BATCH_DIRECTORY --file annotations.csv
```

## 公开农业数据

按用户要求使用 PlantDoc 官方仓库，revision `5467f6012d78d1c446145d5f582da6096f852ae8`，
作者为 Singh、Jain、Jain、Kayal、Kumawat、Batra（2020），仓库许可 CC-BY-4.0。
项目链接：[PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)。
它是作者整理标注的互联网农业图像；与本地田间摄像头采集分开记录。

```powershell
python scripts/download_field_data.py --output data/local/NEW_PLANTDOC --per-class 4 --seed 41
python scripts/run_field_dataset.py --manifest data/local/NEW_PLANTDOC/manifest.json --port COM3
```

选择协议先固定 test、四种作物、十五个来源类别、每类四张、seed=41，再运行任何预测。
每图记录 URL、Git blob SHA1、文件 SHA256、类别、来源划分和许可。回放前与已有全部
PlantVillage 图片检查原始文件/RGB 精确重复和 pHash Hamming≤4 近重复；匹配样本记录后
排除评价，不补选以美化结果。新增来源文件不能覆盖已固定的 selection.json。

来源类别映射 NORMAL/SUSPECT 作为作者参考标签；没有提供的人工质量、存在性单独判断、
判断确定性、光照、距离和拍摄个体均保持未填写。来源标签不冒充本地人工复核。

## 三状态和质量评价

```powershell
python scripts/field_validation.py parity --batch BATCH_DIRECTORY
# 有原模型对应的可信 frozen native 图时增加 --host-model ... --host-report ...
python scripts/field_validation.py report --batch BATCH_DIRECTORY --labels source
# 人工复核完成后另行评价
python scripts/field_validation.py report --batch BATCH_DIRECTORY --labels manual
```

两套标签依据显式选择，报告记录标签文件和批次/manifest 摘要。设备二分类以 p≥0.30 为
SUSPECT，分母仅包含有概率和独立参考标签的样本。系统以 2×3 表记录参考 NORMAL/SUSPECT
与设备 NORMAL/SUSPECT/UNKNOWN，不把 UNKNOWN 记为正确类别。

| 指标 | 分母 |
| --- | --- |
| 二分类 Accuracy / Precision / Recall / F1 / Specificity / FPR / FNR | 有概率且有参考二分类标签的样本；各比率的类别分母写入混淆矩阵 |
| 接受覆盖率 / UNKNOWN 拒绝率 | 有参考标签并有有效设备结果的样本 |
| 接受输入准确率、错误率 | 上述样本中输出 NORMAL/SUSPECT 的子集 |
| 有效输入覆盖率 | 人工标为 USABLE 且有二分类参考标签的子集 |
| SUSPECT→NORMAL 漏检 / SUSPECT→UNKNOWN 延后 | 所有参考 SUSPECT 的有效响应；二者分别报告 |
| SUSPECT 未告警率 | 上述漏检加延后，不能与二分类 FNR 混称 |
| NORMAL 误报 | 参考 NORMAL 中输出 SUSPECT 的比例 |
| 质量误拒/错放 | 人工质量 USABLE / 不合格；按 BLUR、TOO_DARK、OVEREXPOSED、LOW_CONTRAST、OCCLUDED 单列 |

没有分母时指标为 null，文档显示 PENDING，不用零替代。非目标场景另计误接受率，不作为
疾病分类真值。UNKNOWN 分为质量、概率区间和其他拒绝；只对有概率的区间拒绝进行 0.30
强制分类对照，记录拦截错误与延后正确的数量。质量拒绝设备没有执行模型，其反事实不推断。

FP32、host 与设备均复用保存的同一 RGB；核对输入文件和像素 SHA、设备结果 RGB SHA，
参考归一化与冻结边界沿用 v0.3。各后端独立记录概率、状态、耗时和误差。
host 缺原 frozen 图时 PENDING，不以重新校准图或 FP32 输出填充。

场景按实际采集/人工信息分组，缺失为 UNRECORDED，默认少于五个有参考标签样本为
insufficient；同时保留每组计数、混淆矩阵、覆盖率和错误率，不能只比较单个百分比。

## 分组划分和报告

```powershell
python scripts/field_validation.py split --batches BATCH_A BATCH_B BATCH_C --output NEW_FIELD_SPLIT.json --seed 41
python scripts/collect_field_report.py --batch BATCH_DIRECTORY --output reports/field_validation
```

分组连通合并同一批次、拍摄会话、植物个体、公开来源组、精确 RGB 重复及 pHash 近重复。
同视频相邻帧随整个批次保留一组。默认 Train/Validation/Test 的组配额为 .2/.4/.4；没有
足够独立组则全部 unassigned，不制造三套数据。公开 PlantDoc 子集缺个体/会话标识，
保守作为一个来源组，本轮新 Field 分组为 PENDING；原作者 test 来源仍被保留。

每批次生成 EXPERIMENT_SUMMARY.md、DATASET_STATISTICS.json、INFERENCE_RESULTS.csv、
QUALITY_GATE_REPORT.json、THREE_STATE_REPORT.json、INFERENCE_PARITY.json 和三张图：
CONFUSION_MATRIX.png、SCENARIO_COMPARISON.png、UNKNOWN_DISTRIBUTION.png。
另提供 INFERENCE_PARITY.md、场景 JSON 与划分/来源审查。collect_field_report 仅收集
汇总、图表和来源引用，不复制照片，也不上传文件。历史 v0.2 / v0.3 报告保持原样。

本轮结果和完整验收见 [FIELD_VALIDATION_RESULTS.md](FIELD_VALIDATION_RESULTS.md)，架构审查见
[FIELD_VALIDATION_REVIEW.md](FIELD_VALIDATION_REVIEW.md)。
