# 代码来源与参考

基础数组函数从本项目早期的几何、校准和重采样方法中提取；照度拟合、覆盖叠加、空间统计和文件校验接口在整理时去掉了固定尺寸、个人路径、缓存依赖及本次拟合常数，并添加了已知答案测试。

没有复制第三方源代码，也没有随仓库分发星表。下面链接说明所用工具和相关方法；本仓库不是这些项目的官方实现。

- [NumPy `rint`](https://numpy.org/doc/stable/reference/generated/numpy.rint.html)：最近整数与半整数取偶。
- [SciPy `lsqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)：稀疏最小二乘。
- [Siril 背景提取](https://siril.readthedocs.io/en/stable/processing/background.html)：背景建模与天体信号保护。
- [rawpy API](https://letmaik.github.io/rawpy/api/index.html)：需要接入 RAW 解码时参考；本包不分发 RAW 适配器。
- [Astrometry.net](https://astrometry.net/doc/readme.html)：星空定位工具；本包只计算已知相机模型，不自带定位服务。
- [Gaia DR3 文档](https://gea.esac.esa.int/archive/documentation/GDR3/)：恒星位置与光谱参考来源。
- [Astropy WCS接口](https://docs.astropy.org/en/stable/wcs/wcsapi.html)及[FITS图像](https://docs.astropy.org/en/stable/io/fits/usage/image.html)：手工连接示例的坐标和文件接口。

历史处理中查过 Gaia、SIMBAD 和 CIE；本包测试与示例不查询这些服务，不需要账号或网络。需要真实数据时，应单独记录目录版本、历元、引用及服务要求，不把外部目录数据自动纳入本仓库的 Apache-2.0 许可。

