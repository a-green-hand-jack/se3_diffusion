"""SO(3) 扩散方法.
这个模块实现了在 SO(3) 李群上的扩散概率模型。
SO(3) 是三维旋转群，表示所有可能的三维旋转。
"""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch

from data import utils as du


def igso3_expansion(
    omega: Union[np.ndarray, torch.Tensor],
    eps: Union[np.ndarray, torch.Tensor],
    L: int = 1000,
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """IGSO(3) 分布的截断幂级数展开计算.

    理论背景:
    - IGSO(3) 是 Isotropic Gaussian distribution on SO(3) 的缩写
    - SO(3) 是特殊正交群，描述三维空间中的旋转
    - 这个分布是 SO(3) 上的高斯分布的类比，具有旋转不变性

    数学原理:
    幂级数展开的形式为:
    p(R|σ) = Σ (2l + 1) * exp(-l(l+1)σ²/2) * χl(R)
    其中:
    - l 是级数项的索引
    - σ 是分布的扩散参数
    - χl 是 SO(3) 的特征函数
    - R 是旋转矩阵

    实现细节:
    1. 使用欧拉角度 omega 参数化旋转
    2. 将无限级数截断到 L 项
    3. 支持批量计算和 GPU 加速

    参数:
        omega: Union[np.ndarray, torch.Tensor]
            欧拉向量的旋转角度
            shape: [batch_size] 或 [batch_size, num_res]
        eps: Union[np.ndarray, torch.Tensor]
            IGSO(3) 的标准差参数
            shape: 与 omega 相匹配
        L: int, 默认值=1000
            级数截断长度，决定了近似的精度
        use_torch: bool, 默认值=False
            True: 使用 PyTorch 进行计算（支持 GPU 和自动微分）
            False: 使用 NumPy 进行计算

    返回:
        Union[np.ndarray, torch.Tensor]:
            计算得到的幂级数和
            shape: 与输入的 omega 相同

    异常:
        ValueError: 当 omega 的维度不是 1D 或 2D 时抛出

    注意:
        1. eps = sqrt(2) * eps_leach，其中 eps_leach 是原论文中的参数
        2. 当 use_torch=True 时，返回值会保持在与输入相同的设备上
        3. 函数支持批处理操作，可以同时处理多个旋转
    """

    # 选择计算库（PyTorch 或 NumPy）
    lib = torch if use_torch else np

    # 生成级数项的索引 [0, 1, ..., L-1]
    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)  # 确保在正确的设备上

    # 处理不同的输入维度情况
    if len(omega.shape) == 2:
        # 用于预测评分计算时的批处理
        # [1, 1, L] 便于广播
        ls = ls[None, None]
        # [num_batch, num_res, 1]
        omega = omega[..., None]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # 用于缓存计算时的批处理
        # [1, L] 便于广播
        ls = ls[None]
        # [num_batch, 1]
        omega = omega[..., None]
    else:
        raise ValueError("Omega 必须是 1D 或 2D 数组.")

    # 计算幂级数展开的每一项
    # (2l + 1): 维度因子
    # exp(-l(l+1)*eps^2/2): 高斯衰减项
    # sin(omega*(l+1/2))/sin(omega/2): 特征函数项
    # 幂级数展开的形式为:
    # P(R|σ) = Σp; 
    # p = (2l + 1) * exp(-l(l+1)σ²/2) * χl(R)
    p = (
        (2 * ls + 1)
        * lib.exp(-ls * (ls + 1) * eps**2 / 2)
        * (lib.sin(omega * (ls + 1 / 2)) / lib.sin(omega / 2))
    )

    # 沿最后一个维度求和得到最终结果
    if use_torch:
        return p.sum(dim=-1)
    else:
        return p.sum(axis=-1)


def density(
    expansion: Union[np.ndarray, torch.Tensor],
    omega: Union[np.ndarray, torch.Tensor],
    marginal: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """计算 IGSO(3) 分布的概率密度函数.

    理论背景:
    expansion 参数解释:
    - expansion 是 IGSO(3) 分布的幂级数展开结果
    - 它来自 igso3_expansion 函数的计算
    - 其数学形式为：Σ (2l + 1) * exp(-l(l+1)σ²/2) * χl(R)
    - 这个展开捕获了 SO(3) 上高斯分布的特征

    密度计算过程:
    1. 边缘密度（marginal=True）:
       - 将 expansion 与 Haar 测度项 (1-cos(ω))/π 相乘
       - Haar 测度考虑了 SO(3) 的几何结构
       - 最终得到在旋转角度空间上的概率密度

    2. 完整密度（marginal=False）:
       - 将 expansion 除以归一化常数 8π²
       - 得到在整个 SO(3) 空间上的概率密度

    举例说明:
    假设我们要计算旋转角度 θ=π/4 处的密度：
    1. 首先通过 igso3_expansion 计算幂级数展开
    2. 然后根据 marginal 参数选择计算方式：
       - marginal=True: density = expansion * (1-cos(π/4))/π
       - marginal=False: density = expansion/(8π²)

    参数:
        expansion: Union[np.ndarray, torch.Tensor]
            IGSO(3) 密度的幂级数展开值
            这是 igso3_expansion 函数的输出结果
            表示了 SO(3) 上高斯分布的基本形式
            shape: 与 omega 相匹配
        omega: Union[np.ndarray, torch.Tensor]
            欧拉向量的旋转角度（弧度）
            表示要计算密度的具体旋转角度
            shape: 任意形状，但必须与 expansion 兼容
        marginal: bool, 默认值=True
            True: 计算旋转角度的边缘密度
            False: 计算 SO(3) 上的完整密度

    返回:
        Union[np.ndarray, torch.Tensor]:
            计算得到的概率密度值
            shape: 与输入的 expansion 和 omega 相同

    注意:
        1. expansion 的值已经包含了 SO(3) 上高斯分布的主要特征
        2. 边缘密度和完整密度的选择取决于具体应用场景
        3. 边缘密度更适合分析旋转角度的分布
        4. 完整密度则考虑了完整的旋转信息
    """
    if marginal:
        # 计算边缘密度：在旋转角度 [0, π] 上的密度
        # (1-cos(omega))/pi 项来自 SO(3) 的 Haar 测度
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        # 计算完整密度：在整个 SO(3) 空间上的密度
        # 1/(8π²) 是归一化常数
        return expansion / 8 / np.pi**2


def score(
    exp: Union[np.ndarray, torch.Tensor],
    omega: Union[np.ndarray, torch.Tensor],
    eps: Union[np.ndarray, torch.Tensor],
    L: int = 1000,
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """计算 IGSO(3) 密度的评分函数.

    理论背景:
    - 评分函数是概率密度函数的对数梯度: ∇ log p(x)
    - 在扩散模型中，评分函数用于指导反向扩散过程
    - 这个函数计算的是 SO(3) 切空间中的评分

    数学原理:
    使用商规则计算对数导数:
    1. 对于一般函数 f(x) = hi(x)/lo(x):
       d/dx log(f(x)) = (lo(x) * d/dx hi(x) - hi(x) * d/dx lo(x)) / (lo(x)^2)

    2. 在 IGSO(3) 的情况下:
       - hi(x) = sin(ω*(l+1/2))
       - lo(x) = sin(ω/2)
       - 其中 ω 是旋转角度

    实现细节:
    1. 计算高阶项 hi 和其导数 dhi
    2. 计算低阶项 lo 和其导数 dlo
    3. 应用商规则计算最终的评分

    参数:
        exp: Union[np.ndarray, torch.Tensor]
            IGSO(3) 密度的幂级数展开值
            来自 igso3_expansion 函数
            shape: 与 omega 相匹配
        omega: Union[np.ndarray, torch.Tensor]
            欧拉向量的旋转角度（弧度）
            shape: [batch_size] 或 [batch_size, num_res]
        eps: Union[np.ndarray, torch.Tensor]
            IGSO(3) 的尺度参数
            注意：这里的 eps = sqrt(2) * eps_leach
            shape: 与 omega 相匹配
        L: int, 默认值=1000
            级数截断长度
        use_torch: bool, 默认值=False
            True: 使用 PyTorch 计算
            False: 使用 NumPy 计算

    返回:
        Union[np.ndarray, torch.Tensor]:
            SO(3) 切空间中的评分值
            shape: 与输入的 omega 相同

    注意:
        1. 评分函数在扩散模型的训练和采样中都很重要
        2. 返回值用于在 SO(3) 上进行梯度步进
        3. 添加了小常数 1e-4 以避免数值不稳定性
        4. 支持批量计算以提高效率
    """

    # 选择计算库
    lib = torch if use_torch else np

    # 生成级数项索引并处理设备位置（如果使用 PyTorch）
    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)
    ls = ls[None]  # [1, L] 用于广播

    # 处理输入维度
    if len(omega.shape) == 2:
        ls = ls[None]  # [1, 1, L] 用于二维输入
    elif len(omega.shape) > 2:
        raise ValueError("Omega 必须是 1D 或 2D 数组.")
    omega = omega[..., None]  # 添加最后一维用于广播
    eps = eps[..., None]

    # 计算评分函数的组成部分
    hi = lib.sin(omega * (ls + 1 / 2))  # 高阶项
    dhi = (ls + 1 / 2) * lib.cos(omega * (ls + 1 / 2))  # 高阶项的导数
    lo = lib.sin(omega / 2)  # 低阶项
    dlo = 1 / 2 * lib.cos(omega / 2)  # 低阶项的导数

    # 使用商规则计算评分
    # (lo * dhi - hi * dlo) / lo^2 是对数导数的形式
    dSigma = (
        (2 * ls + 1)
        * lib.exp(-ls * (ls + 1) * eps**2 / 2)
        * (lo * dhi - hi * dlo)
        / lo**2
    )

    # 沿最后一维求和并归一化
    if use_torch:
        dSigma = dSigma.sum(dim=-1)
    else:
        dSigma = dSigma.sum(axis=-1)

    # 返回归一化的评分，添加小常数避免除零
    return dSigma / (exp + 1e-4)


class SO3Diffuser:
    """SO(3) 扩散模型的实现类.

    理论背景:
    - SO(3) 是三维旋转群，描述了所有可能的三维旋转
    - 扩散模型在 SO(3) 上通过逐步添加和移除噪声来生成旋转
    - 使用 IGSO(3)（各向同性高斯分布）作为噪声分布

    主要功能:
    1. 前向扩散：将干净的旋转数据逐步加噪
    2. 反向扩散：从噪声数据中恢复原始旋转
    3. 评分计算：计算用于指导反向扩散的梯度信息
    4. 采样：从指定时间点的分布中采样

    实现细节:
    - 使用对数时间表进行噪声调度
    - 支持缓存计算结果以提高效率
    - 提供批量处理能力
    """

    # 1. 基础设置
    def __init__(self, so3_conf) -> None:
        """初始化 SO3Diffuser.

        参数:
            so3_conf: 配置对象，必须包含以下属性:
                schedule: str - 时间表类型 ('logarithmic')
                min_sigma: float - 最小噪声尺度
                max_sigma: float - 最大噪声尺度
                num_sigma: int - sigma 离散化数量
                use_cached_score: bool - 是否使用缓存的评分
                num_omega: int - omega 离散化数量
                cache_dir: str - 缓存目录路径
        """
        self.schedule = so3_conf.schedule
        self.min_sigma = so3_conf.min_sigma
        self.max_sigma = so3_conf.max_sigma
        self.num_sigma = so3_conf.num_sigma
        self.use_cached_score = so3_conf.use_cached_score
        self._log = logging.getLogger(__name__)

        # 离散化 omega 用于计算 CDF，跳过 omega=0
        self.discrete_omega = np.linspace(0, np.pi, so3_conf.num_omega + 1)[1:]

        # 预计算 IGSO3 值并处理缓存
        def replace_period(x: float) -> str:
            """将浮点数中的小数点替换为下划线"""
            return str(x).replace(".", "_")

        # 构建缓存目录路径
        cache_dir = os.path.join(
            so3_conf.cache_dir,
            f"eps_{so3_conf.num_sigma}_omega_{so3_conf.num_omega}_"
            f"min_sigma_{replace_period(so3_conf.min_sigma)}_"
            f"max_sigma_{replace_period(so3_conf.max_sigma)}_"
            f"schedule_{so3_conf.schedule}",
        )

        # 如果缓存目录不存在则创建
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        # 定义缓存文件路径
        pdf_cache = os.path.join(cache_dir, "pdf_vals.npy")
        cdf_cache = os.path.join(cache_dir, "cdf_vals.npy")
        score_norms_cache = os.path.join(cache_dir, "score_norms.npy")

        # 尝试加载缓存或重新计算
        if (
            os.path.exists(pdf_cache)
            and os.path.exists(cdf_cache)
            and os.path.exists(score_norms_cache)
        ):
            # 使用缓存的值
            self._log.info(f"使用缓存的 IGSO3 数据: {cache_dir}")
            self._pdf = np.load(pdf_cache)
            self._cdf = np.load(cdf_cache)
            self._score_norms = np.load(score_norms_cache)
        else:
            # 重新计算所有值
            self._log.info(f"计算 IGSO3 数据并保存到: {cache_dir}")

            # 计算幂级数展开
            exp_vals = np.asarray(
                [
                    igso3_expansion(self.discrete_omega, sigma)
                    for sigma in self.discrete_sigma
                ]
            )

            # 计算边缘分布的 PDF 和 CDF
            self._pdf = np.asarray(
                [density(x, self.discrete_omega, marginal=True) for x in exp_vals]
            )
            self._cdf = np.asarray(
                [pdf.cumsum() / so3_conf.num_omega * np.pi for pdf in self._pdf]
            )

            # 计算评分范数
            self._score_norms = np.asarray(
                [
                    score(exp_vals[i], self.discrete_omega, x)
                    for i, x in enumerate(self.discrete_sigma)
                ]
            )

            # 保存计算结果到缓存
            np.save(pdf_cache, self._pdf)
            np.save(cdf_cache, self._cdf)
            np.save(score_norms_cache, self._score_norms)

        # 计算评分缩放因子
        self._score_scaling = np.sqrt(
            np.abs(
                np.sum(self._score_norms**2 * self._pdf, axis=-1)
                / np.sum(self._pdf, axis=-1)
            )
        ) / np.sqrt(3)

    # 2. 时间和 sigma 相关的基础函数
    @property
    def discrete_sigma(self) -> np.ndarray:
        """获取离散化的 sigma 值数组.

        返回:
            np.ndarray: 在 [0, 1] 区间上均匀分布的 num_sigma 个点
        """
        return self.sigma(np.linspace(0.0, 1.0, self.num_sigma))

    def sigma(self, t: np.ndarray) -> np.ndarray:
        """根据选择的时间表计算 sigma(t) 值.

        理论背景:
        - sigma(t) 控制扩散过程中噪声的大小
        - t 是归一化的时间参数，从 0 到 1
        - 对数时间表可以更好地控制噪声的变化率

        数学原理:
        在对数时间表下:
        sigma(t) = log(t * exp(max_sigma) + (1-t) * exp(min_sigma))
        这提供了从 min_sigma 到 max_sigma 的平滑过渡

        参数:
            t: np.ndarray
                时间点，必须在范围 [0, 1] 内
                shape: 任意，但通常是标量或一维数组

        返回:
            np.ndarray:
                对应时间点的 sigma 值
                shape: 与输入 t 相同

        异常:
            ValueError: 当 t 不在 [0, 1] 范围内时抛出

        注意:
            1. t=0 对应最小噪声 (min_sigma)
            2. t=1 对应最大噪声 (max_sigma)
            3. 对数时间表确保了噪声的平滑变化
        """
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"无效的时间值 t={t}")

        if self.schedule == "logarithmic":
            return np.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))
        else:
            raise ValueError(f"未知的时间表类型 {self.schedule}")

    def sigma_idx(self, sigma: np.ndarray) -> np.ndarray:
        """计算给定 sigma 值对应的离散索引.

        参数:
            sigma: np.ndarray - 需要查找索引的 sigma 值

        返回:
            np.ndarray: sigma 值对应的离散索引
        """
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t: np.ndarray) -> np.ndarray:
        """将连续时间 t 转换为对应的 sigma 索引.

        参数:
            t: np.ndarray - 连续时间值

        返回:
            np.ndarray: 对应的 sigma 索引
        """
        return self.sigma_idx(self.sigma(t))

    def diffusion_coef(self, t: float) -> np.ndarray:
        """计算扩散系数 (g_t).

        理论背景:
        - 扩散系数控制了噪声添加的强度
        - 在对数时间表下，系数需要考虑时间的非线性变化

        数学原理:
        g_t = sqrt(2 * (exp(max_σ) - exp(min_σ)) * σ(t) / exp(σ(t)))

        参数:
            t: float - 时间点，范围 [0, 1]

        返回:
            np.ndarray: 对应时间点的扩散系数

        异常:
            ValueError: 当时间表类型未知时
        """
        if self.schedule == "logarithmic":
            g_t = np.sqrt(
                2
                * (np.exp(self.max_sigma) - np.exp(self.min_sigma))
                * self.sigma(t)
                / np.exp(self.sigma(t))
            )
        else:
            raise ValueError(f"未知的时间表类型 {self.schedule}")
        return g_t

    # 3. 采样相关方法
    def sample_igso3(self, t: float, n_samples: int = 1) -> np.ndarray:
        """使用逆 CDF 方法从 IGSO(3) 分布采样旋转角度.

        理论背景:
        - 使用逆变换采样方法生成符合特定分布的随机数
        - CDF（累积分布函数）的逆函数用于将均匀分布转换为目标分布

        实现细节:
        1. 生成均匀分布的随机数
        2. 使用预计算的 CDF 查找表进行插值
        3. 返回对应的旋转角度

        参数:
            t: float
                连续时间值，必须是标量
            n_samples: int, 默认值=1
                需要生成的样本数量

        返回:
            np.ndarray: shape [n_samples] 的旋转角度数组

        异常:
            ValueError: 当 t 不是标量时
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} 必须是标量.")
        x = np.random.rand(n_samples)
        return np.interp(x, self._cdf[self.t_to_idx(t)], self.discrete_omega)

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """从 IGSO(3) 分布生成旋转向量.

        理论背景:
        - 从 X(0) 到 X(T) 的过程中，遵循 dx = f(x,t)dt + g(t)dw
        - 这里的 f(x,t) 和 g(t) 都是事先约定好的关于 x 和 t 的函数
        - 但是因为我们这里是在 SO3 上，所以需要一些特殊的 f(x,t) 和 g(t)
        - 这里其实是定义了 f(x, t) = 0 和 g(t) = sigma(t)

        - 生成的旋转向量遵循 IGSO(3) 分布
        - 向量的方向均匀分布在单位球面上
        - 向量的长度由 sample_igso3 确定

        实现步骤:
        1. 生成均匀分布的单位向量
        2. 从 IGSO(3) 采样旋转角度
        3. 将单位向量缩放到对应角度

        参数:
            t: float - 连续时间值
            n_samples: int, 默认值=1 - 样本数量

        返回:
            np.ndarray: shape [n_samples, 3] 的轴角旋转向量
        """
        # 生成随机单位向量,， 其实就是 dw
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        # 缩放到采样的旋转角度
        return x * self.sample_igso3(t, n_samples=n_samples)[:, None]

    def sample_ref(self, n_samples: int = 1) -> np.ndarray:
        """生成参考分布的样本.

        这是一个便捷方法，等价于在 t=1 时采样
        这里得到的就是前向过程的最后的情况了，也即是完全的噪音

        参数:
            n_samples: int, 默认值=1 - 样本数量

        返回:
            np.ndarray: shape [n_samples, 3] 的参考分布样本
        """
        return self.sample(1, n_samples=n_samples)

    # 4. Score 相关方法
    def score(self, vec: np.ndarray, t: float, eps: float = 1e-6) -> np.ndarray:
        """计算 IGSO(3) 密度的评分函数（以旋转向量形式）.

        理论背景:
        - 评分函数是概率密度函数的对数梯度
        - 在扩散模型中用于指导反向扩散过程
        - 这个方法是 torch_score 的 NumPy 包装器

        参数:
            vec: np.ndarray
                shape [..., 3] 的轴角旋转向量数组
            t: float
                连续时间值，范围 [0, 1]
            eps: float, 默认值=1e-6
                数值稳定性的小量

        返回:
            np.ndarray:
                shape [..., 3] 的评分向量
                方向与输入向量相同，大小由 _score_norms 决定

        异常:
            ValueError: 当 t 不是标量时
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} 必须是标量.")
        torch_score = self.torch_score(torch.tensor(vec), torch.tensor(t)[None])
        return torch_score.numpy()

    def torch_score(
        self, vec: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """使用 PyTorch 计算 IGSO(3) 密度的评分函数.

        理论背景:
        - 这是评分计算的核心实现
        - 使用预计算的查找表提高效率
        - 支持 GPU 加速和自动微分

        实现细节:
        1. 计算旋转向量的范数（角度）
        2. 根据使用缓存与否选择不同的计算路径：
           - 使用缓存：直接查表获取评分范数
           - 不使用缓存：实时计算评分范数
        3. 将评分范数与归一化的方向向量相乘

        参数:
            vec: torch.Tensor
                shape [..., 3] 的轴角旋转向量
            t: torch.Tensor
                连续时间值，范围 [0, 1]
            eps: float, 默认值=1e-6
                数值稳定性的小量

        返回:
            torch.Tensor:
                shape [..., 3] 的评分向量
                保持在与输入相同的设备上
        """
        # 计算旋转角度（添加 eps 避免除零）
        omega = torch.linalg.norm(vec, dim=-1) + eps

        if self.use_cached_score:
            # 使用预计算的评分范数
            score_norms_t = self._score_norms[self.t_to_idx(du.move_to_np(t))]
            score_norms_t = torch.tensor(score_norms_t).to(vec.device)
            # 找到离散化 omega 的索引
            omega_idx = torch.bucketize(
                omega, torch.tensor(self.discrete_omega[:-1]).to(vec.device)
            )
            # 使用索引获取对应的评分范数
            omega_scores_t = torch.gather(score_norms_t, 1, omega_idx)
        else:
            # 实时计算评分范数
            sigma = self.discrete_sigma[self.t_to_idx(du.move_to_np(t))]
            sigma = torch.tensor(sigma).to(vec.device)
            # 计算 IGSO(3) 展开
            omega_vals = igso3_expansion(omega, sigma[:, None], use_torch=True)
            # 计算评分
            omega_scores_t = score(omega_vals, omega, sigma[:, None], use_torch=True)

        # 返回评分向量：评分范数 * 归一化方向
        return omega_scores_t[..., None] * vec / (omega[..., None] + eps)

    def score_scaling(self, t: np.ndarray) -> np.ndarray:
        """计算训练过程中使用的评分缩放因子.

        理论背景:
        - 评分缩放用于归一化评分的大小
        - 这有助于训练的稳定性
        - 缩放因子是根据预计算的评分范数和 PDF 计算的

        数学原理:
        缩放因子 = sqrt(E[score²]) / sqrt(3)
        其中 E[score²] 是评分的二阶矩

        参数:
            t: np.ndarray
                时间点数组

        返回:
            np.ndarray:
                对应时间点的评分缩放因子
        """
        return self._score_scaling[self.t_to_idx(t)]

    # 5. 扩散过程相关方法
    def forward_marginal(
        self, rot_0: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在时间 t 对初始旋转进行前向扩散采样.

        理论背景:
        - 前向过程通过添加噪声来模糊初始旋转
        - 使用 IGSO(3) 分布作为噪声模型
        - 在 SO(3) 上使用右乘法进行旋转复合

        实现细节:
        1. 从 IGSO(3) 采样噪声旋转
        2. 计算噪声旋转的评分
        3. 使用右乘法将噪声应用到初始旋转

        参数:
            rot_0: np.ndarray
                shape [..., 3] 的初始旋转向量
                也就是真实的蛋白质的旋转的情况
            t: float
                连续时间值，范围 [0, 1]

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - rot_t: shape [..., 3] 扩散后的旋转向量
                - rot_score: shape [..., 3] 对应的评分向量
        """
        # 计算样本总数
        n_samples = np.cumprod(rot_0.shape[:-1])[-1]
        # 采样噪声旋转
        sampled_rots = self.sample(t, n_samples=n_samples)
        # 计算评分
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)
        # 右乘复合旋转
        rot_t = du.compose_rotvec(rot_0, sampled_rots).reshape(rot_0.shape)
        return rot_t, rot_score

    def reverse(
        self,
        rot_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        mask: Optional[np.ndarray] = None,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """模拟一步反向 SDE，使用测地线随机游走.

        理论背景:
        - 反向扩散过程通过评分函数指导去噪
        - 使用随机微分方程（SDE）描述反向过程
        - 在 SO(3) 上使用测地线保持流形结构

        数学原理:
        更新公式：
        rot_t+1 = rot_t ⊕ (g_t²⋅score_t⋅dt + g_t⋅sqrt(dt)⋅z)
        其中：
        - ⊕ 表示 SO(3) 上的右乘
        - g_t 是扩散系数
        - z 是标准正态噪声
        - dt 是时间步长

        参数:
            rot_t: np.ndarray
                shape [..., 3] 当前旋转向量
            score_t: np.ndarray
                shape [..., 3] 当前评分向量
            t: float
                当前时间点
            dt: float
                时间步长
            mask: Optional[np.ndarray]
                可选的掩码，指示哪些部分需要更新
            noise_scale: float, 默认值=1.0
                噪声强度的缩放因子

        返回:
            np.ndarray:
                shape [..., 3] 更新后的旋转向量

        异常:
            ValueError: 当 t 不是标量时

        注意:
            1. 使用测地线确保结果仍在 SO(3) 上
            2. mask 可用于选择性更新
            3. noise_scale 控制随机性强度
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} 必须是标量.")

        # 计算扩散系数
        g_t = self.diffusion_coef(t)
        # 生成随机噪声
        z = noise_scale * np.random.normal(size=score_t.shape)
        # 计算更新量
        # math: g(t)^2 * score_d * dt + g(t) * d(\bar{w})
        #       = g(t)^2 * score_t * dt + g(t) * sqrt(dt) * z
        perturb = (g_t**2) * score_t * dt + g_t * np.sqrt(dt) * z

        # 应用掩码（如果提供）
        if mask is not None:
            perturb *= mask[..., None]

        # 计算样本总数
        n_samples = np.cumprod(rot_t.shape[:-1])[-1]

        # 使用测地线复合旋转
        rot_t_1 = du.compose_rotvec(
            rot_t.reshape(n_samples, 3), perturb.reshape(n_samples, 3)
        ).reshape(rot_t.shape)

        return rot_t_1
