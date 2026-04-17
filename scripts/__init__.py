"""Jointdiff 可执行脚本入口（训练 / 评估 / Gradio / 工具）。

从仓库根目录 ``Jointdiff/`` 运行，例如::

    python scripts/train/train_jointdiff_new_full_withcamparam_doubleloss.py --help

脚本会在文件开头将仓库根目录加入 ``sys.path``，以便 ``import utils``、``patch`` 等包级导入。
"""
