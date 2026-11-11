# 接口与单位

数组计算不读写照片，也不联网。除文件校验命令外，所有函数由调用者传入数组和参数。无效输出一般为 NaN，并同时返回显式有效掩膜或计数；不要用数值大于零来判断覆盖。

| 模块/函数 | 输入与输出约定 |
| --- | --- |
| `calibration.calibrate_linear` | 可广播的 raw、含偏置 dark、正 response；返回信号、可选方差、有效掩膜。RGB 的二维响应需显式加 `[..., None]` |
| `calibration.calibrated_cfa_bridge` | 同形 CFA raw/dark、可广播黑电平、整数白电平；返回 uint16，只用于必须接收 unsigned CFA 的解码器 |
| `geometry.polynomial_map` | xy 为 `(..., 2)`，坐标是零起点像素中心；系数为总次数三角序 `1,x,y,x²,xy,y²,...`，形状 `(terms,2)`；可返回解析 Jacobian |
| `geometry.grid_map` | 一维 x、y 坐标；同样的三角系数、origin、scale；返回两个 `len(y)×len(x)` 网格 |
| `geometry.map_pixels` | 相机模型与3×3旋转；模型中 inverse 是像素→切平面，forward 是切平面→像素。旋转方向由调用者明确，非任意形变 |
| `geometry.solid_angle` | 同一相机模型，返回每像素立体角，约为球面度；不等于点源照度响应 |
| `selfcal.radial_features` | xy、中心和正尺度，默认列为归一化 r²、r⁴，不使用固定相机尺寸 |
| `selfcal.response_information` | `N×P` 特征、N个星ID、N个曝光ID、正 log-flux 标准差；报告去掉星/曝光偏移后的秩及条件数 |
| `selfcal.fit_stellar_response` | 同上再加 N个 log-flux；返回 `ResponseFit`，第一排序曝光偏移为零。输入已做适当饱和/SNR筛选；必要可辨识性不通过则抛错 |
| `resampling.resample_cubic` | 二维图、`(...,2)`源坐标；返回重采样值、可选对角方差、有效掩膜。非零权重贡献中任一输入无效则输出无效 |
| `stacking.weighted_mean` | 沿轴0按给定正方差的倒数加权；返回均值、条件方差、权重、有效样本数及权重集中度 `n_eff` |
| `stacking.combine_group_means` | 值为 `(groups,...,channels)`，计数为 `(groups,...)`；返回均值、可用覆盖、接受计数、拒绝组数和回退标志 |
| `background.sample_background_cells` | `H×W`或`H×W×C`图及`H×W`保护掩膜；返回 `N×2`中心、`N×C`中值和`N×2`整数格坐标 |
| `background.fit_background_plane` | 样本中心、通道值和 `(H,W)`；返回具有 `.predict(xy)` 的 `BackgroundPlane`，只估计加性平面 |
| `photometry.aperture_measure` | 二维图和固定星点，孔径/环带均为完整像素单位；返回孔径光量、对角噪声近似误差和局部噪声尺度。默认孔径较大，需按自己的星像配置 |
| `validation.spatial_folds` | 非负整数采样格坐标，返回整块留出的折号；不执行拟合 |
| `validation.block_mean_residuals` | 带符号通道残差及采样格坐标；返回块坐标和平均残差 |
| `validation.bootstrap_mean` | 标量评分与块ID；返回均值、块重采样标准误差和区间，需要至少两块 |
| `validation.one_standard_error_choice` | 各模型的评分、标准误差与复杂度；返回选中的输入索引，不包含其他验收门槛 |
| `provenance.snapshot / verify_snapshot` | 目录与JSON兼容清单；只读原字节，拒绝符号链接，检查完整文件集合 |

## 方差的范围

`calibrate_linear` 的输入方差必须已经包含原始测量与暗场的不确定性。响应的不确定性、共享暗场相关性及模型偏差没有自动计入。

`resample_cubic` 传播单个输出样本的对角方差；相邻输出共享输入，协方差并未输出。`weighted_mean` 的方差同样以独立输入和正确给定方差为条件。`photometry` 的误差不是重采样相关噪声下的完整误差条。

`area_scale` 是对 count-per-source-pixel 输入进行局部 Jacobian 面积换算时的近似。已经是表面亮度时通常应省略；这不是精确像素重叠积分或 drizzle。

