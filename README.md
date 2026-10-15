# 基于YOLOv8的远景行人检测

本项目对应《智能系统工程训练任务书》课题三。实现YOLOv8n基线、ECA注意力、P2小目标分支以及ECA+P2四组模型，提供数据清洗、共同骨干初始化、训练、忽略区域感知评估、推理、计时与结果归档。

Git仓库保存源代码、配置、测试、报告草稿、实验汇总、少量展示图片和[四组正式最佳权重](weights)。VisDrone原始数据、预训练权重、完整结果ZIP、虚拟环境和录屏不随仓库上传；下面提到的`data/`、`downloads/`、`outputs/`路径须在运行环境中自行准备或由程序生成。四组最佳权重的SHA-256见[`weights/SHA256SUMS`](weights/SHA256SUMS)。已核实的汇总数字见[课程报告](materials/课程报告.md)与[模型结构统计](materials/模型结构统计.md)。项目内附带的Ultralytics源码遵循其[上游许可与修改说明](vendor/UPSTREAM.md)。

## 目前交付状态

- 四组模型已在Linux服务器各完成100轮，正式结果包为本地`outputs/results_bundle_e100_complete.zip`，未上传Git仓库。原包`results_bundle_e100.zip`也保留在本地；完整包另含640和1280输入速度补测。
- test-dev上，baseline、eca、p2、eca_p2的mAP50–95依次为14.78%、15.01%、17.35%、17.28%；AP_small依次为11.08%、11.04%、13.43%、13.45%。详细验证集、测试集和速度口径见[`materials/课程报告.md`](materials/课程报告.md)第8节。
- 课程报告保留Markdown版；15页答辩内容另有[`materials/基于YOLOv8的远景行人检测_PPT制作稿.docx`](materials/基于YOLOv8的远景行人检测_PPT制作稿.docx)，供豆包等工具生成PPT。真实演示录屏仍待制作；姓名、学号、班级和实际分工由学生填写。
- 本地`outputs/demo_e100/`保存正式权重对两张test-dev静态图片的实际推理输出，包含标注JPG、JSON和CSV；仓库中的[展示图片](materials/figures)可供答辩排版使用，但不是录屏视频。
- 密集图像的可视化默认只画细框，防止大量类别文字遮挡目标；类别、置信度和原图像素坐标仍保存在JSON/CSV。
- 本地RTX 4060流程检查与正式Linux RTX PRO 5000训练分开记录，见[`materials/verification.md`](materials/verification.md)。四组25轮脚本作为时间受限时的备用方案，**不属于本次正式结果**。

完成环境安装后，可以用仓库内正式权重对自己的图片做推理（`your_image.jpg`须换成实际图片路径）：

```bash
python run.py predict --weights weights/eca_p2_e100.pt --source your_image.jpg --output outputs/inference --imgsz 960 --conf 0.25
```

## Linux 服务器（100轮）

部署目录为 `~/private/code/zcp/yolo_pedestrian`。本服务器的 CUDA 数字编号与 `nvidia-smi` 物理索引不一致，因此 Linux 脚本用 GPU UUID 固定物理 GPU 1 和 3，并在启动前通过 NVML 检查真实映射；程序内统一为 `cuda:0`。四组各 100 轮，统一使用 `configs/train.yaml`，输出目录后缀为 `_e100`。GPU UUID 更换后可通过 `GPU1_UUID`、`GPU3_UUID` 环境变量更新映射。

```bash
cd ~/private/code/zcp/yolo_pedestrian
bash scripts/setup_linux.sh
bash scripts/run_experiments_linux.sh 2>&1 | tee outputs/linux_e100.log
```

当物理 GPU 1 和 3 都空闲时，可用 `bash scripts/run_experiments_linux.sh --parallel` 将四组分成两轮并行运行：第一轮 baseline/eca_p2，第二轮 eca/p2，每组仍独立占用一张卡，统一 batch=8、100 轮。各组日志写入 `outputs/logs/`。长时间 SSH 会话建议用 `nohup` 或 `tmux` 保持任务；启动前确认两张卡的占用。GPU 2 已被其他任务占用，不纳入本脚本。

服务器运行记录的查看方式：

```bash
cd ~/private/code/zcp/yolo_pedestrian
tail -f outputs/linux_e100_launcher.log
tail -f outputs/logs/baseline_e100.log
tail -f outputs/logs/eca_p2_e100.log
nvidia-smi
```

每组100轮`results.csv`位于结果包内的`outputs/runs/<组名>_e100/`。独立评估位于`outputs/evaluation/<组名>_e100_{val,test}_960/`，速度结果位于`outputs/benchmark_<组名>_e100.json`。补测的640、1280速度JSON仅在`results_bundle_e100_complete.zip`中。若需复核，不要直接重启整套脚本以免新目录混入正式结果。

服务器不能访问 GitHub 时，先把完整的 `VisDrone2019-DET-train.zip`、`VisDrone2019-DET-val.zip`、`VisDrone2019-DET-test-dev.zip` 放在 `data/raw/`。也可放对应的已解压目录。脚本以 `download --offline` 检查三个集合和标注，不会联网。训练按验证集选出最佳权重并完成验证分析后，脚本才自动评估 test-dev；`bash scripts/run_experiments_linux.sh --final-test` 可单独重跑最终测试。不要把测试集用于调参。

离线服务器已有的 `geneval_blackwell2` 是 Python 3.10、PyTorch 2.7.1+cu128。`setup_linux.sh` 只在 `zcp_train` 不存在时克隆它，随后核对实际版本并运行 GPU 前向/反向测试。原计划的 Python 3.11 在服务器无法出站下载安装时无法满足，报告和结果包须如实记录此偏差。项目内附带 THOP 2.0.10 的纯 Python 源码及许可证以完成计算量统计。环境和数据准备完成后，先依据服务器实测每轮耗时决定是否需要两卡并行；不能仅根据显存容量推算速度。

## Windows服务器最快运行方法

将完整项目目录复制到服务器，建议放在短路径，例如 `D:\pedestrian`。更新支持RTX 5080的NVIDIA驱动，在项目根目录打开PowerShell。安装脚本优先使用已有Python 3.11；若只有Conda，会在项目`.venv`内创建独立Python 3.11环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_experiments_25.ps1 -OfflineData
```

若不使用自动检测，可指定已有Python 3.11的完整路径：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Python 'C:\Python311\python.exe'
```

脚本在项目`.venv`中安装固定版本，不修改全局环境。Conda回退时解释器位于`.venv\python.exe`；以下手动命令通过`run_python.ps1`自动识别两种环境。首次下载约2GB原始压缩包；请预留至少15GB磁盘空间供原图副本、训练图像、检查点和输出使用。失败时脚本立即停止，不会静默跳过实验。

若Conda已创建`.venv`但安装脚本中途失败，直接重新运行`setup_windows.ps1`；脚本会复用已有环境并继续安装。只有依赖安装和严格GPU检查都通过后，训练脚本才允许开始。

### GitHub下载失败时离线导入数据

在能够访问GitHub下载地址的电脑上，下载缺失的VisDrone2019-DET压缩包。当前如果只在`test-dev`下载失败，只需补齐`VisDrone2019-DET-test-dev.zip`；若train或val也缺失，则一并下载对应ZIP。三个文件的地址是：

- `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip`
- `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip`
- `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip`

通过U盘、局域网共享等方式，把完整ZIP复制到服务器项目的`data/raw/`，保留上述原文件名。`.zip.part`是失败的临时文件，不能当作完整ZIP使用。也可以在另一台电脑先解压，再把完整的`VisDrone2019-DET-{train,val,test-dev}`目录复制到`data/raw/`。离线入口会核对ZIP完整性及官方图像、标注数量；已完整解压的目录可以直接复用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiments_25.ps1 -OfflineData
```

此参数只关闭数据下载，不跳过校验和解压。若已经手动解压，也可用同一命令。传输中断时重新复制缺失的ZIP即可，无须重建Python环境。

## 分步执行

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py check --strict
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py download --offline
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py prepare --strict-counts
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py preflight --config configs/train_25.yaml --strict
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py train-all --config configs/train_25.yaml --tag e25
```

`preflight`用最大模型、密集样本的标注及一次优化器更新检查显存，并将四组共同batch记录在 `outputs/experiment_settings.json`。探测图像为随机像素，仅用于内存检查，不参与训练或精度评估。正常训练图片来自VisDrone。

25轮短预算设置：960输入、batch=8起步、seed=42、AdamW、lr=0.001、预热1轮、最后3轮关闭Mosaic。原100轮方案仍在`configs/train.yaml`与`run_experiments.ps1`。若显存不足按8、4、2、1统一降低batch。Mosaic可能把多幅密集图像组合在一起，若正式训练仍溢出，请停止所有组、统一减小batch并重新开始，不能只修改一个组。

### 25轮备用短预算

短预算脚本为`scripts/run_experiments_25.ps1`，在`configs/train_25.yaml`上重新做显存检查，再顺序训练四组，每组25轮。结果保存在`outputs/runs/baseline_e25`、`eca_e25`、`p2_e25`、`eca_p2_e25`，结果包为`outputs/results_bundle_e25.zip`。本项目正式结果已采用100轮；25轮如另行运行，需从头开始并保持四组共同日程，不能与本次100轮指标混作一张消融表。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiments_25.ps1 -OfflineData
```

25轮全部验证完成后，独立运行最终test-dev测试：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiments_25.ps1 -FinalTest
```

这个脚本若发现已有`*_e25`训练目录会拒绝从头重跑，避免自动生成带编号目录却评估不完整权重；中断时需要先按检查点恢复对应组。

单组执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py train --variant eca_p2 --config configs/train_25.yaml --name eca_p2_e25
```

中途断电或关闭进程后，使用尚未结束训练且包含优化器状态的last.pt恢复：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py train --variant eca_p2 --config configs/train_25.yaml --resume outputs/runs/eca_p2_e25/weights/last.pt
```

已经完成25轮且被Ultralytics去除优化器的权重不适用于断点恢复。完成一组后不要重新启动 `train-all`，否则会新建带编号目录；正式四组应使用相同配置、保留一套明确命名的最终运行目录。

## 验证与最终测试

训练过程每轮使用项目自定义PersonValidator，按**忽略区感知mAP50–95**选择best.pt。该指标与上游默认加权fitness不同。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py evaluate --weights outputs/runs/eca_p2_e25/weights/best.pt --split val --output outputs/evaluation/eca_p2_e25_val_960
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py analyze --split val --tag e25
```

`analyze`对基线和完整模型在640、960、1280三种推理分辨率重新预测，并对置信度阈值0.05至0.95作分析。它不是重新训练，也不能称为“不同训练分辨率实验”。

确定方案后独立执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiments_25.ps1 -FinalTest
```

默认正式测试使用960、FP32、置信度工作点0.25；AP计算使用0.001低阈值采集预测。测试数据不得用于选择最佳轮次、调整模型或阈值。

## 评估口径

- 数据源类别1 pedestrian与类别2 people合并为person，因此包含坐、蹲等其他姿态人体。
- 原始train/val/test-dev划分不变；正式入口核对6471/548/1610张图像，数据转换保留所有无人体图像作为负样本。
- 同一划分或跨划分的字节重复图像记录在`duplicate_images.json`，数据摘要另列数量；保留官方图像与划分。若发现跨划分重复，报告中应说明潜在评估偏倚。上次中断留下的`data/prepared`部分文件可由同一`prepare`命令重新覆盖，清单仅在成功完成后生成。
- 训练图片遮罩原始忽略区域，人体框内像素保护；评估读取未遮罩原图。
- 对每个IoU阈值按置信度顺序先进行一对一有效GT匹配，再排除与忽略区域IoA≥0.5的未匹配预测。IoA分母是预测框面积。
- AP采用101点召回插值，IoU=0.50:0.05:0.95，max_det=1000；这是课程项目的单类扩展指标，不是VisDrone官方多类别排行榜成绩。
- AP_small按**原图面积<1024像素²**划分。与其他尺度GT匹配的预测不计小目标误检。
- 数据准备输出`data_summary.json`、`object_sizes.csv`和目标面积、遮挡程度、原始类别三张分布图；独立评估输出按场景标记的`case_selection.json`和对应检测图。
- Precision、Recall、F1、TP、FP、FN使用明确的0.25置信度工作点；GT为空时AP保存为JSON null，避免把未定义指标当成0。

## 图片与视频检测

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py predict --weights outputs/runs/eca_p2_e25/weights/best.pt --source your_image.jpg --output outputs/demo_image
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py predict --weights outputs/runs/eca_p2_e25/weights/best.pt --source your_video.mp4 --output outputs/demo_video
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py predict --weights outputs/runs/eca_p2_e25/weights/best.pt --source 0 --max-frames 1000 --output outputs/demo_camera
```

输入可以是图片文件、图片目录、视频或摄像头编号。输出JSON包含每帧记录，包括没有检测的空帧；CSV只保存实际检测框。坐标均为原图像素。目录图片按文件名排序；视频输出保留原视频帧率，摄像头或无法读取帧率时采用25fps。

## 速度测试与结果返回

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py benchmark --weights outputs/runs/eca_p2_e25/weights/best.pt --image data/prepared/originals/val/实际文件名.jpg --output outputs/benchmark_eca_p2_e25.json
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py summary
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 run.py pack --output outputs/results_bundle_e25.zip
```

速度测试每组采用同一图像、输入尺寸和精度，batch=1，预热50次、计时500次。模型延迟不含预处理/NMS；端到端延迟包含预处理及NMS，但不含磁盘读取、画框和视频编码。若使用 `--half`，四组都需要使用FP16并另列结果。

请返回 `outputs/results_bundle_e25.zip`。它包含配置、数据清单、环境、日志、曲线、权重、预测与指标，不包含完整原图和虚拟环境。正式报告与PPT据此更新。

## 测试与许可

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 -m pytest tests -q
```

项目内的Ultralytics源码为8.3.24，带AGPL-3.0完整许可。自定义模块、解析器改动和来源见 `vendor/UPSTREAM.md`。只加载官方权重或本项目自己生成的可信权重。公开数据仅用于课程研究，重新分发前核对数据集提供方的许可与条款。
