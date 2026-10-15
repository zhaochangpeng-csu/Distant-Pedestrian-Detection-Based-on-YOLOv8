# 上游源码说明

来源：本机已安装的 Ultralytics 8.3.24 Python发行包。
官方仓库：https://github.com/ultralytics/ultralytics/tree/v8.3.24
许可：AGPL-3.0，完整许可文本见 LICENSE.ultralytics。

本项目修改 nn/tasks.py：导入 C2fECA，在 parse_model 的普通模块与内部重复模块集合中注册它。
新增实现位于 pedestrian/modules.py。其他上游文件保留原实现。
项目在导入任何Ultralytics模块前设置 vendor 优先路径和项目内缓存目录。
无需安装全局ultralytics；使用项目自己的Python虚拟环境。
