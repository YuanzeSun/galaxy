# Galaxy

把一次星空照片处理过程中能复用的方法整理成 Python 工具和说明。

处理顺序是：保护原片 → 暗场校正 → 恒星定位与对齐 → 相对照度和曝光归一化 → 叠加 → 流星合成 → 颜色与显示 → 标注和导出。

目前提供数值处理模块、文件校验命令和合成数据示例。RAW 读取、星表定位、流星筛选、调色和成片导出仍需接入自己的工具；这里还不是一个从任意 RAW 文件夹一键出图的软件。

实际使用看[按步骤连接工具和脚本](docs/manual-run.md)：说明外部RAW/定位工具输出什么、怎样接到测光和照度拟合、如何对齐及叠加。可以一步一步执行，不要求自动跑完整批。

换设备或天区时，按[参数复用与适用条件](docs/portability.md)重新提供元数据、测光和坐标；示例中的数字不作为新拍摄的默认校准值。

## 可以直接复用什么

| 环节 | 解决的问题 | 代码 |
| --- | --- | --- |
| 暗场与整数转换 | 扣除暗场；必须交给整数解码器时，避免直接截断造成单向偏差 | `calibration` |
| 对齐与像素面积 | 用镜头模型映射星点，计算每个像素对应的天空面积 | `geometry` |
| 照度自标定 | 利用同一颗星经过画面不同位置的光量变化，估计低频照度响应及曝光间变化 | `selfcal` |
| 重采样 | 明确哪些输入像素有效，保留负值，估计单像素方差 | `resampling` |
| 叠加 | 记录实际有效覆盖；没有数据的位置保留为空，不当成黑色平均 | `stacking` |
| 背景与测光 | 排除受保护天体后取背景样本；测量固定位置的恒星孔径光量 | `background`、`photometry` |
| 方法检查 | 用未参与拟合的区域检查预测，按空间块估计误差 | `validation` |
| 文件保护 | 记录并复核文件大小、修改时间与 SHA256 | `provenance` |

这里的照度自标定是从恒星估计的**相对低频响应**，不等于拍摄得到的平场，也不保证恢复灰尘斑、完整二维响应或天空绝对背景。

## 安装和运行

Python 3.10 或以上，依赖 NumPy 和 SciPy。无需下载照片或星表。

```bash
git clone git@github.com:YuanzeSun/galaxy.git
cd galaxy
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
galaxy-astro demo
```

`demo` 用已知答案的合成数据检查校准、照度拟合、背景保护和叠加，不读取真实照片，也不联网。

原片校验示例，报告放在原片目录之外：

```bash
galaxy-astro manifest /path/to/originals --output /path/to/session-inputs.json
galaxy-astro verify /path/to/originals /path/to/session-inputs.json
```

命令不会覆盖已有报告。校验包括新增、缺失或变化的文件。报告包含相对文件名，分享前仍需检查这些名字是否包含私人信息。

## 最小代码示例

```python
import numpy as np
from galaxy_astro.calibration import calibrate_linear
from galaxy_astro.stacking import weighted_mean

# 数字仅为示例。实际 dark、response、variance 应来自本次数据。
raw = np.array([[101., 99.], [102., 98.]])
signal, variance, valid = calibrate_linear(
    raw, dark=100., response=1., variance=4.
)
result = weighted_mean(signal, variance, valid)
print(result["mean"])  # [1.5, -1.5]；保留负信号
```

更多例子见 [examples/methods.py](examples/methods.py)。图像可使用任意尺寸；相机黑电平、白电平、几何、响应和掩膜都由调用者提供，仓库不附带某次拍摄的拟合参数。

## 工作流和方法记录

- [完整处理流程](docs/workflow.md)：每一步读什么、输出什么，以及哪些环节尚未封装。
- [按步骤操作](docs/manual-run.md)：可直接照着接的代码、文件格式和外部工具入口；[串联示例](examples/connect_steps.py)演示测光到叠加的完整调用。
- [换设备与天区](docs/portability.md)：哪些代码共用、哪些参数要重算，以及当前测试覆盖的范围。
- [照度自标定与平场](docs/flat-field.md)：模型、可辨识性和容易混入的误差。
- [验证与停止条件](docs/validation.md)：合成控制、空间留出、光量和弥散结构保护。
- [一次处理的取舍](docs/lessons.md)：保留了什么、否决了什么，以及下方亮带修正为何停止。
- [函数接口与单位](docs/api.md)：数组形状、坐标、方差和校验行为。
- [代码来源与参考](docs/references.md)。

原片、成片、缓存、星表和拍摄专属参数没有上传。亮带修正已停止，相关失败实验只保留文字记录，没有作为推荐算法发布。仓库中的软件测试也不能证明真实银河的绝对光量或颜色已经标定准确。

## 许可证

[Apache-2.0](LICENSE)。第三方依赖使用各自的许可证。

