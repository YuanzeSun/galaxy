# 按步骤连接工具和脚本

不需要一键运行。下面按每一步的输入、调用和输出说明连接方式；你可以做完一步、检查结果，再继续下一步。

所有相机与拍摄参数来自自己的输入。[换设备与天区](portability.md)列出了需要重算的部分和本例的适用条件；不使用原项目的固定机型参数。

微单＋广角镜头拍摄银河以外的天区也使用这些接口，测光样本和天体保护区按[不同星野](star-fields.md)选择。没有流星、没有标注需求时不添加这些步骤；背景无法可靠估计时保留母版，不要求一定扣除。

小图的完整调用示例在 [examples/connect_steps.py](../examples/connect_steps.py)。其中用到的读取、测光和WCS转换函数在 [examples/manual_steps.py](../examples/manual_steps.py)。示例只生成合成数据，不读取真实照片，也不写中间文件。

在仓库根目录安装并运行：

```bash
python -m pip install ".[manual]"
python examples/connect_steps.py
```

以下代码也从仓库根目录运行。`session/` 是新建的处理目录，和原片分开；文件名仅作示例。大图需要按行分块，后文给出写法。

```python
from pathlib import Path

for name in ["linear", "solved", "checks", "measurements", "final"]:
    Path("session", name).mkdir(parents=True, exist_ok=True)
```

## 先约定要传递什么

| 数据 | 约定 | 谁产生 |
| --- | --- | --- |
| 线性图 | `H×W×3` RGB或`H×W`单通道，浮点数；无显示拉伸、自动亮度、局部对比度 | RAW/天文处理工具 |
| 坐标 | `(x,y)`是从0开始的像素中心；数组索引用`[y,x]` | WCS或已有镜头模型 |
| 恒星表 | 相同`star_id`始终表示同一颗星；保存每帧位置和测光质量 | 星表＋WCS＋人工/独立质量检查 |
| 响应模型 | 中心、归一化尺度、特征定义、拟合系数必须一起保存 | `selfcal` |
| 曝光偏移 | 与`frame_id`对应；参考曝光的偏移为0 | `selfcal` |
| 对齐图 | 统一参考坐标下的表面亮度，另有`H×W`有效掩膜 | `resampling` |
| 组平均 | `H×W×C`均值和`H×W`实际曝光计数 | 自己累加；再交给`stacking` |

先按机身、镜头、焦距、光圈、对焦和读出设置分组。不同光学配置不要共用一套径向响应。相机矩阵、白平衡、黑/白电平与解码尺度要记录；一次拟合内保持通道处理方式一致。若曝光时间或增益不同，先换算到共同单位，或明确曝光偏移同时吸收了这些变化，不能把它全称为大气透明度。

## 1. RAW → 已校准的线性图

可以用 Siril 完成RAW转换、匹配的暗场校准和去马赛克，导出32位浮点FITS。暗场校准应在CFA阶段完成。保留原始构图和方向，不应用显示拉伸或自动亮度。

本说明采用的入口是：**已经减过暗场、尚未做照度/平场和天空背景校正的线性图**。如果外部工具已经做过平场，就跳过后面对应的照度除法；如果已经减过暗场，后面不能再减一次。

若自行接 RAW 解码器，`calibrated_cfa_bridge` 接在“读取CFA”和“去马赛克”之间：

```python
from galaxy_astro.calibration import calibrated_cfa_bridge

# 四个输入由RAW适配器和本次暗场提供；这里没有机型通用的黑电平常数。
decoder_cfa = calibrated_cfa_bridge(
    observed_cfa, dark_including_bias, black_level_map, white_level
)
# 把decoder_cfa交给解码器的内存CFA接口，再进行线性去马赛克。
```

这段只说明接口位置，不是一个完整RAW适配器。优先使用支持浮点校准的工具；只有解码器必须接收unsigned CFA时才用整数桥。照片里的ADC饱和无法通过后续放大恢复，旧机型的高光缩放经验也不能直接套在新解码器上。

Siril的操作依据见[校准说明](https://siril.readthedocs.io/en/stable/preprocessing/calibration.html)，自行适配时可查[rawpy参数](https://letmaik.github.io/rawpy/api/rawpy.Params.html)。

## 2. 线性图 → 带天区坐标的图

给每张图求解WCS，例如使用Siril的星空定位功能，或已安装的Astrometry.net：

```bash
solve-field --no-plots --dir session/solved/frame_001 session/linear/frame_001_green.fits
```

该命令及所需星表索引见[Astrometry.net说明](https://astrometry.net/doc/readme.html)。求解输出通常包含带WCS的`.new` FITS。索引文件需要另行准备；仓库不附星表，也不自动上传照片。

求解可以用绿色通道，但它必须和待处理RGB图有**相同尺寸、方向、裁切和像素位置**。不能直接把缩小预览的WCS套在完整图上。

若RGB图还没有WCS，可以把相同构图绿色通道的求解结果写入一个新的RGB FITS，原文件不改动：

```python
from astropy.io import fits
from astropy.wcs import WCS

with fits.open("session/linear/frame_001_rgb.fits", mode="readonly") as hdus:
    data = hdus[0].data.copy()  # FITS通道顺序为(3,H,W)
    unit = hdus[0].header.get("BUNIT")
wcs = WCS(fits.getheader("session/solved/frame_001/frame_001_green.new")).celestial
header = wcs.to_header(relax=True)
if unit is not None:
    header["BUNIT"] = unit
fits.PrimaryHDU(data, header=header).writeto("session/linear/frame_001_solved.fits", overwrite=False)
```

然后读取：

```python
from examples.manual_steps import read_linear_fits

counts, wcs = read_linear_fits("session/linear/frame_001_solved.fits")
```

`counts`是`H×W×3`，`wcs`描述这张图的像素→天空关系。先用独立星点检查全图及边缘位置误差，再用于对齐或强制测光。定位成功本身不证明边缘精度足够。

## 3. 同一恒星目录 → 各帧测光表

准备一份目录CSV，列为`star_id,ra_deg,dec_deg`，坐标在ICRS下、历元和自行处理与定位相符。ID当作字符串读，避免大整数经过浮点转换后失真。

```python
import csv
import json
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from examples.manual_steps import measure_rows

with open("session/catalog.csv", encoding="utf-8", newline="") as f:
    catalog = list(csv.DictReader(f))
ids = np.array([row["star_id"] for row in catalog])
sky = SkyCoord(
    [float(row["ra_deg"]) for row in catalog]*u.deg,
    [float(row["dec_deg"]) for row in catalog]*u.deg,
    frame="icrs",
)
xy = np.column_stack(wcs.world_to_pixel(sky))

# 每帧一份事先检查的质量表：{"star-id": true/false, ...}。
# true表示未饱和、没有明显重叠/缺陷等；缺失项默认不采用。
with open("session/checks/frame_001_quality.json", encoding="utf-8") as f:
    quality = json.load(f)
admitted = np.array([quality.get(str(star), False) for star in ids])
rows, selected = measure_rows(
    counts, xy, ids, "frame_001", admitted,
    radius=5, annulus=(8, 12), minimum_snr=20, fractional_error_floor=.01,
)
with open("session/measurements/frame_001.npz", "xb") as f:
    np.savez_compressed(f, **rows)
```

这里的孔径和误差底限是示范值，应在看拟合结果前按自己的星像、噪声和测光稳定性确定；坐标与半径都用完整图像像素。饱和检查尽量回到原始ADC，不能只看经过白平衡的16位图有没有到65535。保留未采用对象及原因，不按拟合残差事后挑星。

`measure_rows`默认测绿色通道，把`flux`变成`log_flux`，误差按`error/flux`转换并加指定底限。孔径误差只是背景对角噪声近似；真实数据有更好的光子/相机噪声模型时，先更新`sigma_log`再拟合。它不是完整的相关噪声误差条。

对明确选定的其他曝光重复这一步，保存相同列。WCS给出同一目录星在各帧的位置，所以星ID可以直接连接起来，不需要再凭最近坐标随意认作同一颗星。[Astropy坐标接口](https://docs.astropy.org/en/stable/wcs/wcsapi.html)说明了这些转换。

## 4. 测光表 → 照度系数和曝光偏移

```python
from examples.manual_steps import join_rows
from galaxy_astro.selfcal import radial_features, fit_stellar_response

files = ["session/measurements/frame_001.npz", "session/measurements/frame_002.npz"]
# files可以继续列出自己已检查的曝光。两张只是文件名示意，不保证信息足够。
tables = []
for name in files:
    with np.load(name, allow_pickle=False) as table:
        tables.append(dict(table))
table = join_rows(tables)
h, w = counts.shape[:2]
center, scale = ((w-1)/2, (h-1)/2), w/2
positions = np.column_stack([table["x"], table["y"]])
features = radial_features(positions, center, scale)

# 从目录中事先划出的留出星列表；不是看残差后再决定。
with open("session/checks/held_star_ids.json", encoding="utf-8") as f:
    held_ids = np.array(json.load(f), dtype=str)
train = ~np.isin(table["star_id"], held_ids)
fit = fit_stellar_response(
    table["log_flux"][train], features[train], table["star_id"][train],
    table["frame_id"][train], table["sigma_log"][train],
)
frame_offsets = {str(k): float(v) for k, v in zip(fit.frame_levels, fit.frame_log_offsets)}
```

输出的`fit.coefficients`描述`log R = theta[0]·r² + theta[1]·r⁴`；`frame_offsets`描述相对于参考曝光的乘性变化，应用时取`exp(offset)`。参考是**排序后的第一个曝光ID**，不一定是输入列表的第一张，给ID补零可以避免字符串排序误会。

求解器会拒绝信息不足或不连通的观测表。纯转动、恒星半径不变时，不能强求径向响应。应有足够位置变化和共有恒星。

把`center`、`scale`、特征定义、系数、参考曝光、测光孔径、输入单位和曝光偏移一起保存。不能只保存两个系数；新图像的坐标约定、裁切或光学设置一变，它们就不再等价。

留出星的检查方式见`connect_steps.py`：减掉响应与曝光偏移后，每颗留出星只拟合自己的常量光量，再比较时间散布及位置趋势。若新曝光没有参与拟合，不能把其偏移设为零；应先用已定响应和共有训练星的光量估计它的曝光偏移。

## 5. 原生线性图 → 相对表面亮度 → 对齐图

这里采用“恒星孔径总计数拟合响应”的约定：点源响应是`R`；均匀天空每像素还乘以其天空面积。因此在原生传感器坐标下除以`R × 曝光因子 × 相对像素面积`，再对齐。

```python
from examples.manual_steps import pixel_solid_angles, reference_to_source, register_surface_brightness
from galaxy_astro.calibration import calibrate_linear

reference_counts, reference_wcs = read_linear_fits("session/linear/frame_001_solved.fits")
reference_shape = reference_counts.shape[:2]
reference_area = pixel_solid_angles(reference_wcs, reference_shape)
rh, rw = reference_shape
omega0 = reference_area[rh//2, rw//2]  # 所有帧都使用这一个归一化常数

# counts、wcs为当前源帧；这里以同一机身/尺寸的图为例。
yy, xx = np.mgrid[:h, :w]
sensor_xy = np.stack([xx, yy], axis=-1)
response = np.exp(radial_features(sensor_xy, center, scale) @ fit.coefficients)
area = pixel_solid_angles(wcs, (h, w))/omega0
transmission = np.exp(frame_offsets["frame_001"])
denominator = response*area*transmission
sb, _, valid_rgb = calibrate_linear(counts, dark=0., response=denominator[..., None])

source_xy = reference_to_source(reference_wcs, wcs, reference_shape)
aligned, valid = register_surface_brightness(sb, source_xy, valid_rgb.all(axis=-1))
```

曝光ID必须换成当前帧的ID。`dark=0`是因为入口已减暗场；不要再次扣除暗场。上面把G通道测得的响应共同应用到RGB，属于无色低频响应假设，各颜色通道仍要验证。

`source_xy[y,x]`回答“参考图这个位置应去源图哪里取值”，不是反方向。无效边缘保持NaN及`valid=False`；不填黑、不镜像补边。已经转成表面亮度后，不再额外乘一次重采样Jacobian。

WCS面积函数使用半像素方向差分，是局部导数近似；不会把整幅广角图当作一个恒定像素面积。已有经过验证的相机多项式模型时，也可用`geometry.solid_angle`和`geometry.map_pixels`产生相同接口的数据，不必更换为WCS。

完整分辨率下不要一次生成四套天空方向向量。按条带建立面积图和对齐坐标：

```python
area = np.empty((h, w), dtype=float)
for y0 in range(0, h, 96):
    y1 = min(h, y0+96)
    area[y0:y1] = pixel_solid_angles(wcs, (h, w), y_start=y0, y_stop=y1)/omega0
# register_surface_brightness(sb, ...)也可按参考图的96行逐条调用；
# reference_to_source的y_start/y_stop会保留全图坐标，不把条带顶部误当y=0。
```

若用Siril等工具完成了完整校准及配准，也可从已经统一表面亮度单位的对齐帧直接进入下一步，不要再做一次上述校正。普通“对齐RGB文件”不能未经单位核对就当作表面亮度输入。

### 曝光间需要匹配背景时

这一步放在累加之前，和最后的显示背景扣除分开。先选一张已经转成相同单位的对齐图作共同参照；对每张曝光只拟合它与参照的差值，并使用相同的天体保护规则：

```python
from galaxy_astro.background import sample_background_cells, fit_background_plane, BackgroundPlane

# aligned_reference为共同参照，protected为参考坐标下的天体保护区。
bad = ~np.isfinite(aligned).all(axis=-1) | ~np.isfinite(aligned_reference).all(axis=-1)
xy, delta, cells = sample_background_cells(
    aligned-aligned_reference, protected | bad, cell_size=64
)
candidate = fit_background_plane(xy, delta, reference_shape)
# 先按第7步的空间留出方式检查差值平面，采用后保存其coefficients。
```

所有选定曝光都完成检查后，令各曝光的平面系数平均为零，再用于匹配：

```python
# models是所有已通过检查的差值平面；frame_index对应当前曝光。
coefficients = np.stack([model.coefficients for model in models])
relative = coefficients[frame_index]-coefficients.mean(axis=0)
plane = BackgroundPlane(relative, reference_shape)
yy, xx = np.mgrid[:rh, :rw]
correction = plane.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(rh, rw, -1)
matched = aligned-correction
# 用matched及原valid累加；大图按参考行分块生成correction。
```

这只是相对平面匹配的连接例子，不提供绝对天空零点。不同曝光覆盖和权重不同时，曝光平均系数为零也不保证画面每处的加权修正都为零，需要额外检查边缘。无可靠背景区域、留出结果不好或天体保护失败时，跳过它，不强行提高阶数。

## 6. 对齐帧 → 组平均 → 线性母版

在一个组内，把选定曝光逐张加进去：

```python
sum_rgb = np.zeros((*reference_shape, 3), float)
count = np.zeros(reference_shape, np.uint32)
# 对每张已检查的aligned、valid重复这两行；负值和零值都可参与。
sum_rgb[valid] += aligned[valid]
count[valid] += 1
group_mean = np.full(sum_rgb.shape, np.nan)
np.divide(sum_rgb, count[..., None], out=group_mean, where=count[..., None] > 0)
```

按自己的分组保存`group_mean`和`count`。再调用：

```python
from galaxy_astro.stacking import combine_group_means

# means/groups_counts是自己选定的各组数组列表。
result = combine_group_means(np.stack(means), np.stack(groups_counts))
linear_sb = result["mean"]
coverage = result["coverage"]
```

这会按组里的实际曝光数合并，不是把每组当成一张。默认没有剔除；若要用历史MAD规则，显式提供`clip_sigma`和自己数据单位下的`scale_floor`，并先做验证。九组是历史处理选择，不是任意素材的固定要求。

组数较多时，用`np.load(..., mmap_mode="r")`读取各组，按64行分别`np.stack`及合并，写入新的输出数组，避免把所有全图堆进内存。

这里得到的是相对表面亮度。要输出“参考像素光量”版本，再乘一次`reference_area/omega0`；该换算只在输出时做，不能又拿它作为照度响应除一次。

## 7. 可选背景估计与验证

另建一个参考坐标下的`H×W`保护掩膜，覆盖本次需要保留的真实天体及弥散结构，并排除空覆盖。普通星野、星团、星云和银河分别选择保护范围；不使用默认银河带。新天区要重新画/生成掩膜，不能照搬另一张图的保护区。

```python
from galaxy_astro.background import sample_background_cells, fit_background_plane
from galaxy_astro.validation import spatial_folds

protected = np.load("session/checks/protected.npy") | ~np.isfinite(linear_sb).all(axis=-1)
xy, sky, cells = sample_background_cells(linear_sb, protected, cell_size=64)
folds = spatial_folds(cells)
for fold in np.unique(folds):
    model = fit_background_plane(xy[folds != fold], sky[folds != fold], reference_shape)
    held_residual = model.predict(xy[folds == fold])-sky[folds == fold]
    # 保存这组留出残差，再按docs/validation.md比较空间结构和保护条件。
```

背景函数只给出候选估计，不是自动采用决定；它也不是原来未通过验收的亮带样条算法。先保存原始线性母版，把背景候选放在单独的显示分支。需要曝光间背景匹配时，只匹配相对变化并规定共同零点约束，不要把每帧独立估计的整幅天空都扣掉。

## 8. 母版交给后期工具

例如写一份新的线性FITS，再交给Siril或其他支持线性浮点图的软件：

```python
header = reference_wcs.to_header(relax=True)
header["BUNIT"] = "relative surface brightness"
fits.PrimaryHDU(
    np.moveaxis(linear_sb.astype(np.float32), -1, 0), header=header
).writeto("session/final/linear_sb.fits", overwrite=False)
fits.PrimaryHDU(coverage).writeto("session/final/coverage.fits", overwrite=False)
```

之后再做经过验证的颜色、显示拉伸、流星合成、标注和导出。FITS数组的通道顺序是`C×H×W`，读取后再转回`H×W×C`；不要把轴顺序错误当成画幅旋转。

每一步记录输入文件、单位、坐标、参数和输出SHA256。`provenance`命令用于核对原片，`write_new_json`可以独占创建参数记录，避免覆盖已有文件。所有中间文件都放在新目录，不写回原片。

## 换一批拍摄时

| 项目 | 处理方式 |
| --- | --- |
| 解码规则、整数转换方法 | 沿用方法，黑/白电平及缩放从新元数据取得 |
| 相机颜色配置 | 同机身和同解码约定可沿用基础配置；白平衡重新读取/明确选择 |
| 已有镜头模型 | 相同光学条件可作初值，用新星点核验；也可直接使用新求解的WCS |
| 照度响应 | 用新测光表重拟合并留出验证；不要直接拿旧系数当真实平场 |
| 曝光偏移、坐标变换、覆盖和背景 | 用新照片重新计算 |
| 孔径、分组、保护区、噪声尺度 | 看新数据事先设置，不照抄示例数值 |

重算时真正需要交接的是这些明确的输入和输出，不需要先写一个自动跑完整批的程序。

