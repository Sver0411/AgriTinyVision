# Field Edge Audit

研究以 AgriTinyVision 的已发布摄像头为起点。新一轮实验让摄像头在不知道评估标签的情况下，用同一字节账本选择购买当地标签或把带宽留给逐帧复核。真实图片与 JPEG 成本覆盖 Cornell、Kashmir 和 Latvia，涵盖原 V2、V2 的 ESP32-S3 主机量化代理和 V4。原有 60 条硬件记录给量化敏感性提供参考。

- [修订稿 PDF](paper/manuscript.pdf) · [自主购买协议](protocol/PURCHASE_STUDY.md) · [同预算配对对照](results/purchase/paired_summary.csv) · [主实验汇总](results/purchase/summary.csv)
- [Latvia 逐图误报与漏报](results/purchase/latvia_recall_primary.csv) · [来源分数与阈值诊断](results/purchase/backbone_diagnostic.csv) · [定量复现实验](protocol/QUANTIZATION_RANKS.md)
- [端云流程](results/purchase/purchase_decision_flow.pdf) · [跨骨干及额度方式图](results/purchase/paired_effects.pdf)

在主预算的 36 个三站点、三数值管线、两种标注上限和两种额度条件中，自主购买策略只在 8 个条件中胜过同标签、同安装费用的 EdgeBoost 排名对照。在从零攒流量且允许买 32 张的 180 次决策中，全部购买 16 张，随后有 149 次保留原头。Latvia 的 V2 后期照片上，保留原警报并复核部分阴性，无需购买标签即可将 49 张漏报降至 31 张，同时保持 67 张误报。

在公开目录用 Python 标准库运行 `python3 scripts/verify_purchase_results.py` 可复核主结论。恢复逐例参考标签或重跑模型时，使用下方数据说明和 [新实验的私有缓存生成协议](protocol/PURCHASE_COHORTS.md)；新逐运行记录分站分骨干保存在 `results/purchase/`，合并表可从分片生成。旧实验相关的索引和复核路径仍列于下方。

## 公开数据与复核

可直接阅读所有主表、论文和协议。公开目录保留研究生成的分数、模型头、合规的逐图记录与聚合统计。Cornell 原图、原始参考标签和受限的图像特征请依 [FGVC7 竞赛条款](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules) 自行取得。Latvia 原图与逐图视觉特征留在[官方数据提供方](https://www.kaggle.com/datasets/projectlzp201910094/applescablds)。第三方代码、数据及模型保留原作者的许可，COMMECT 提取指标采用 [CC BY-SA 4.0](protocol/DEVICE_AND_LINK.md)。

在此研究目录安装下方 Python 依赖后，可以用 `python scripts/verify_revision.py` 检查论文主表。若要逐图核对 Cornell 结果，先运行 `python scripts/public_release.py restore /path/to/train.csv` 在本机补回参考标签。公开索引 `PUBLIC_DATA_INDEX.json` 记录需要恢复的字段及重建条件。部分从头训练的实验还需要重建特征缓存。

## 阅读顺序

- 英文稿在 [paper/manuscript.pdf](paper/manuscript.pdf)，可编辑源文件在 [paper/manuscript.tex](paper/manuscript.tex)。
- [选题审查](protocol/PORTFOLIO.txt) 和[期刊定位](protocol/VENUES.txt) 说明对原仓库的取舍。[分析设计和追加对照顺序](protocol/analysis.txt) 记录了部分对照是在观察到哪些问题后加入的。本研究未作公开预注册。
- [60 张板端配对数据](data/board_pairs.csv)、[PlantDoc 数据沿革](data/PLANTDOC_PROVENANCE.md)、[后端权重和图像编码](data/BACKEND_PROVENANCE.md) 说明已发表于仓库的测量与本次重算的边界。
- [逐种子误差与真实通信字节](results/edge_adaptation_metrics.csv)、[固定后端配对结果](results/upload_outcomes.csv)、[逐次抽样最少上传字节](results/head_vs_oracle.csv)、[跨来源冻结测试](results/transfer_outcomes.csv) 支撑主要结论。
- [分歧抽样](protocol/FIXED_POOL_CERTIFICATE.md)、[适配协议](protocol/EDGE_ADAPTATION.md)、[高召回阈值实验](protocol/SENSITIVITY_CALIBRATION.md) 和 [C99 核查](c_reference/README.md) 提供实现细节和假设。

## 复核已保存的结果

需要 Python 3.12、NumPy 和 SciPy。绘图还需要 Matplotlib，C99 核查需要 C 编译器。在项目根目录执行以下命令。按上一节完成标签恢复后，以下复核可以使用保存的逐图决策和已拟合分类头。依赖特征的重训命令还需按下一节准备对应数据。文件中带有 train 的 FGVC7 编号是竞赛公开标注部分的命名，研究所用训练池与评估池由独立清单给出。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/joint_error_evidence.py
.venv/bin/python scripts/random_query_budget.py
.venv/bin/python scripts/audit_joint_budget.py
.venv/bin/python scripts/matched_collaboration.py
.venv/bin/python scripts/verify_bundle.py
.venv/bin/python scripts/oracle_cost.py
.venv/bin/python scripts/verify_certificate.py --compare-primary
.venv/bin/python scripts/evaluate_upload.py
.venv/bin/python scripts/transfer_eval.py
.venv/bin/python scripts/plot_results.py
.venv/bin/python c_reference/verify.py
```

## 从原始公开数据重新执行

工作站需要 Git、联网、数 GB 工作空间。公开照片与第三方模型权重留在原始提供方，压缩交付包不附这些原件或衍生 JPEG。本地项目的 data/raw 及缓存 JPEG 保留本次计算使用的副本。Kaggle 公共镜像能否匿名下载取决于提供方；脚本也接受已下载的版本化 ZIP 和独立标签文件。模型作者的训练集、DINOv2 预训练语料以及来源中的所有植株身份无法完整审核。

```sh
python3.12 -m venv .venv-jpeg
.venv-jpeg/bin/python -m pip install -r requirements-backend.txt
.venv/bin/python scripts/fetch_plantdoc.py
.venv/bin/python scripts/fetch_backend.py
.venv/bin/python scripts/prepare_orchard.py data/raw/fgvc7-mirror.zip
.venv/bin/python field_pairs.py --prepare --source data/raw/agritinyvision
.venv/bin/python scripts/run_plantdoc.py
.venv/bin/python scripts/orchard_edge.py
.venv/bin/python scripts/cross_source.py
.venv-jpeg/bin/python scripts/server_backend.py
.venv-jpeg/bin/python scripts/server_backend.py --fgvc --images data/raw/fgvc7 --csv results/fgvc_backend.csv --payloads data/fgvc_backend_jpegs
.venv-jpeg/bin/python scripts/audit_backend.py
.venv/bin/python scripts/field_specialist.py plantdoc
.venv/bin/python scripts/field_specialist.py fgvc
.venv/bin/python scripts/field_specialist.py head
.venv/bin/python scripts/edge_adaptation_features.py
.venv/bin/python scripts/edge_adaptation.py
.venv/bin/python scripts/specialist_ablation.py
.venv/bin/python scripts/sensitivity_calibration.py
.venv/bin/python scripts/transfer_features.py data/raw/kashmir.parquet
.venv/bin/python scripts/evaluate_upload.py
.venv/bin/python scripts/transfer_eval.py
.venv/bin/python scripts/plot_results.py
.venv/bin/python scripts/verify_bundle.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python c_reference/verify.py
```

[网络回放协议与来源](data/NETWORK_REPLAY.md) 给出三段轨迹的授权、起点、复现和恒定容量测试。便携包保留这六个公开轨迹及其许可，运行 `.venv/bin/python scripts/mahimahi_batch.py` 可重建 9,600 行回放。

LaTeX 稿件可在 paper 目录用 `tectonic manuscript.tex` 编译。实验中以真实产生的应用层报文长度计通信量。主实验的 INT8 分类头使用 FP32 骨干特征。按设备发布的张量量化尺度进行的主机重放另列于正文第 6 节。

## 同时降低误报和漏报

[三地逐图配对结果汇总](results/joint_error_evidence.csv)给出每个来源的阴性和阳性分母、两类错误的纠正数与新增数，以及采用同一规则需要传输的字节数。[外测规则锁定记录](protocol/LATVIA_ENDPOINT.txt)区分历史对照和查看拉脱维亚分数前已指定的策略。[日期分割](protocol/LATVIA_LOCAL_DATES.md)描述一个需要当地标签、按拍摄日先后评估的独立问题。正式发布的无黑星病标签只代表未观察到目标病害。

在已经保存的逐图决策上运行 `scripts/joint_error_evidence.py` 即可重算双降结论。`scripts/random_query_budget.py` 计算独立随机抽取一部分照片上送时，这批固定图像同时少报错、少漏诊的确切随机化概率。它以每张照片的发送抽签为唯一随机源。[概率和期望字节](results/random_query_budget.csv)不包含未来果园的预测区间。

从原始资料复现时，先按[拉脱维亚数据审计说明](data/latvia_source/README.md)取得和验证官方 ZIP。以下命令在已重建上述源模型、JPEG 缓存及分类头的项目目录执行。用本地实际授权取得的原图存放路径替换最后一组命令的 `data/raw/latvia.zip`。

```sh
.venv/bin/python scripts/directional_audit.py fit
.venv/bin/python scripts/directional_audit.py replay
.venv/bin/python scripts/dual_fusion.py fit
.venv/bin/python scripts/dual_fusion.py evaluate
.venv/bin/python scripts/dual_existing.py
.venv/bin/python scripts/dual_multiview.py data/raw/kashmir.parquet
.venv/bin/python scripts/latvia_blind_features.py extract data/raw/latvia.zip data/latvia_source/blind_index.csv
.venv/bin/python scripts/latvia_blind_features.py verify data/latvia_source/manifest.csv data/latvia_source/provenance.json
.venv/bin/python scripts/latvia_blind_features.py score
.venv/bin/python scripts/dual_external.py
.venv/bin/python scripts/latvia_local_dates.py prepare
.venv/bin/python scripts/latvia_local_dates.py fit
.venv/bin/python scripts/latvia_local_dates.py test
.venv/bin/python scripts/joint_error_evidence.py
.venv/bin/python scripts/random_query_budget.py
.venv/bin/python scripts/verify_bundle.py
```

## 数据和方法来源

[AgriTinyVision 发布页](https://github.com/Sver0411/AgriTinyVision/releases/tag/v0.1.0) · [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) · [Cornell 果园图像和专家标注](https://doi.org/10.1002/aps3.11390) · [Kashmir 图像数据](https://doi.org/10.17632/9m2dcb5mmr.3) · [Kashmir 版本化镜像](https://huggingface.co/datasets/Project-AgML/apple_leaf_disease_classification/tree/38e4c0600bbf962f2379bf1574b8049c801e089d) · [DINOv2](https://arxiv.org/abs/2304.07193)

## 相同总预算下的端云协同

[协同协议](protocol/JOINT_BUDGET.txt)在每批次的总上限内，比较只更新端侧、只按原端侧不确定性选图上传、以及先更新端侧再把剩余带宽用在低置信度图像的组合。服务器和 JPEG 编码固定，校准及部署开销均从预算内扣除。这组离线批处理实验以参考完整上传的 25% 为上限，另报告 10% 和 50%。20 次抽取共享同一批评估图。

[各站点和预算的均值](results/joint_budget_summary.csv) · [每次运行的两类错误和实际字节](results/joint_budget_runs.csv) · [随机上传对照与远端纠错记录](results/joint_budget_ablation.csv) · [逐病种的纠正和新增错误](results/joint_error_by_label.csv)

有本地拉脱维亚特征缓存时，运行 `python scripts/joint_budget.py` 可从分数和分类头重建三地协同决策。便携包包含每张图的路由选择、两端决策和真实字节数，运行 `python scripts/audit_joint_budget.py` 可独立核对每个配对比较并重算 20 种随机上传次序的对照，无需原图。运行 `python scripts/plot_collaboration.py` 和 `python scripts/plot_joint_errors.py` 可重绘协同与误报漏报两张图。

[同标注量的单端、单服务器与端云联合阈值](protocol/MATCHED_COLLABORATION.md)还比较了相同 450 个标签下穷举所有单分数阈值的强对照，区别于 25% 总流量下的选择性上传实验。
