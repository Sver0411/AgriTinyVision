# 农业场景验证：架构和实施顺序

基线为本地 v0.3 工作区，Git HEAD `3ae13554d3716269eaf488f1cf31f751817060dc`。
保留开始时的全部未提交改动，清单哈希存于本地 `.cache/field_start_inventory.json`。
正式模型、0.30 阈值、0.20/0.40 边界与质量配置不变；不训练、不刷写、不推送。

## A：复用架构

现有 CameraService 是相机/串口唯一所有者，InferenceController 单帧在途，RunRecorder
记录测量。新增轻量 field_validation 包，保存批次和离线评价；继续使用 JSON/JSONL/CSV。
在 InferenceController 的同一帧快照、同一 RGB 字节处接入可选采集批次。普通推理默认不保存图片。
Web 增加独立的采集/标注页面，沿用本地会话、CSRF 与文件路径约束。

| 阶段 | 文件 | 验收 |
| --- | --- | --- |
| B 采集 | field_validation/capture.py、controller、service、web | 原始 RGB PNG、实际 160 RGB PNG、像素/文件 SHA、元数据和设备结果对应；手动/定时/暂停/继续；导出受控批次 |
| C 标注 | field_validation/annotations.py、Web 页面 | 标签与预测独立；无预测填充标签；保存、翻页、CSV；病害标签需依据；源数据作者标签单独标明 |
| D 评价 | field_validation/evaluation.py | 2×2 二分类与 2×3 系统矩阵分开；UNKNOWN 不计正确；分母、覆盖、错误、质量与概率拒绝分开；人工质量缺失则 PENDING |
| E 对照 | field_validation/parity.py | 复用 OnnxReference / FrozenHostReference、设备身份和实际 RGB SHA；缺原 frozen native 时保留空缺 |
| F 报告/划分 | reporting.py、split.py、CLI | 九类批次产物与独立 INFERENCE_PARITY.md；按植物/会话/批次和精确/近重复连通组划分；组不足时保留未分配 |
| G 回归 | tests/test_field_*.py | 软件、模拟串口、实际 ESP32 与外部农业数据证据分别记录；原 v0.3 测试通过 |

## 公开数据补充

用户要求从网上寻找真实农业样本。选用作者维护的 PlantDoc 官方公开仓库，revision
`5467f6012d78d1c446145d5f582da6096f852ae8`，许可 CC-BY-4.0。
只取官方 test 中苹果、甜椒、马铃薯、番茄类别，固定 seed 和每类数量，在查看预测前决定样本。
保留每图 URL、Git blob、原始类别、来源拆分和文件 SHA。与已有 PlantVillage 数据检查精确
和 pHash 近重复，发现重叠即在评价分母中排除并记录。

PlantDoc 为作者人工整理的互联网农业图像；这是公开复杂背景外部评测，不冒充本地摄像头
田间采集。作者类别只用于二分类来源标签；缺少的人工质量、光照、距离、植物个体等不推断。
本地标注页面可补充这些维度，来源标签、人工复核、模型预测分开存储。

## 指标约定

二分类以冻结 p(SUSPECT)≥0.30 为阳性，只计算有合法概率和独立参考标签的样本。
系统以 NORMAL/SUSPECT/UNKNOWN 三输出计算 2×3 表、接受覆盖率和选择性准确率。
UNKNOWN 是拒绝，不是正确分类。SUSPECT→NORMAL 漏检与 SUSPECT→UNKNOWN 延后判断单列；
另列未告警率，避免将拒绝隐藏在 recall 中。质量拒绝设备没有概率，不计算其强制分类反事实。
概率区间拒绝可与同一图的 0.30 强制二分类对照，分别记录拦截错误和拒绝正确的数量。

人工质量评价仅使用已填写的质量标签，不把设备门控结果反填成人工标签。
分场景图表以实际元数据分组，未记录维度明确显示 UNRECORDED，少于设定数量显示样本不足。
公开数据缺少植物/拍摄会话标识，采用保守来源批次组；不会把独立文件名假设为独立植物。
