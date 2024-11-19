"""R^3 扩散方法。
这是一个实现3D空间中扩散模型的模块，主要用于处理分子构象的转换。
R3表示三维欧几里得空间，使用VP-SDE(Variance Preserving Stochastic Differential Equation)
作为扩散过程的数学基础。
"""

from typing import Optional, Union, Tuple, Any
import numpy as np
import torch


class R3Diffuser:
    """VP-SDE扩散器类，用于处理三维空间中的分子构象扩散过程。

    该类实现了基于方差保持随机微分方程(VP-SDE)的扩散模型，专门用于处理分子在三维欧几里得空间(R³)中的构象变化。

    核心功能：
    1. 前向扩散过程：通过逐步添加高斯噪声，将初始分子构象转换为纯噪声
    2. 反向扩散过程：通过学习的score function，将噪声数据逐步恢复为有意义的分子构象
    3. 边际采样：实现了p(x(t)|x(0))和p(x(t)|x(t-1))的采样
    4. Score计算：计算用于指导反向扩散的score function

    主要特点：
    - 使用方差保持SDE作为扩散过程的数学基础
    - 支持连续时间扩散过程，时间t在[0,1]范围内
    - 包含坐标缩放机制，用于处理不同尺度的分子系统
    - 支持带掩码的扩散，可以选择性地对部分原子进行扩散

    数学原理：
    - 使用β(t)作为时变方差调度
    - 扩散系数和漂移系数分别由diffusion_coef和drift_coef定义
    - 通过边际概率分布marginal_b_t控制噪声的添加过程

    使用场景：
    - 分子构象生成
    - 构象转换模拟
    - 分子动力学研究
    """

    def __init__(self, r3_conf: Any) -> None:
        """初始化R3Diffuser对象。

        Args:
            r3_conf: 配置对象，包含以下关键参数：
                - min_b: 方差调度的起始值，控制初始扩散强度
                - max_b: 方差调度的终止值，控制最大扩散强度
                - coordinate_scaling: 坐标缩放因子，用于归一化分子坐标

        数学背景：
        min_b和max_b定义了β(t)函数的范围，该函数控制扩散过程中噪声的添加速率。
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """对输入坐标进行缩放，使其适合扩散过程。

        Args:
            x: 原始分子坐标，形状为[..., 3]或[..., N, 3]

        Returns:
            缩放后的坐标

        说明：
        缩放操作可以帮助模型更好地处理不同尺度的分子系统，
        通常使用埃米单位到模型单位的转换。
        """
        return x * self._r3_conf.coordinate_scaling

    def _unscale(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """将缩放后的坐标转换回原始尺度。

        Args:
            x: 缩放后的坐标，形状为[..., 3]或[..., N, 3]

        Returns:
            原始尺度的坐标

        说明：
        这是_scale的逆操作，用于将模型输出转换回实际物理单位。
        """
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """计算时间t处的方差调度值。

        Args:
            t: 时间点，范围必须在[0,1]内

        Returns:
            对应时间点的β(t)值

        数学原理：
        β(t) = min_b + t(max_b - min_b)
        这是一个线性方差调度，控制扩散过程中噪声的添加速率。
        """
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """计算扩散系数，即SDE中的g(t)。

        Args:
            t: 时间点，范围在[0,1]内

        Returns:
            扩散系数值 g(t) = √(β(t))

        数学背景：
        在VP-SDE中，扩散系数控制随机噪声的强度，
        它是方差调度的平方根。
        """
        return np.sqrt(self.b_t(t))

    def drift_coef(
        self, x: Union[np.ndarray, torch.Tensor], t: Union[float, np.ndarray]
    ) -> Union[np.ndarray, torch.Tensor]:
        """计算漂移系数，即SDE中的f(x,t)。

        Args:
            x: 当前状态坐标
            t: 时间点

        Returns:
            漂移系数值 f(x,t) = -1/2 * β(t) * x

        数学背景：
        漂移项决定了系统的确定性演化部分，
        在VP-SDE中采用线性形式。
        """
        return -1 / 2 * self.b_t(t) * x

    def sample_ref(self, n_samples: int = 1) -> np.ndarray:
        """采样参考噪声。

        Args:
            n_samples: 需要采样的数量

        Returns:
            标准正态分布采样，形状为(n_samples, 3)

        说明：
        生成用于扩散过程的基准噪声，
        采用标准正态分布N(0,1)。
        """
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """计算边际方差调度。

        Args:
            t: 时间点，范围在[0,1]内

        Returns:
            边际方差值

        数学原理：
        这个函数定义了从初始分布到目标噪声分布的插值路径，
        采用二次函数形式：t*min_b + (t²/2)*(max_b-min_b)
        """
        return t * self.min_b + (1 / 2) * (t**2) * (self.max_b - self.min_b)

    def calc_trans_0(
        self,
        score_t: Union[np.ndarray, torch.Tensor],
        x_t: Union[np.ndarray, torch.Tensor],
        t: Union[float, np.ndarray],
        use_torch: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """计算从时间t到时间0的转移。

        Args:
            score_t: t时刻的score值
            x_t: t时刻的状态
            t: 时间点
            use_torch: 是否使用PyTorch计算

        Returns:
            预测的时间0状态

        数学原理：
        基于score function的条件期望计算，
        使用指数缩放和条件方差进行状态转换。
        """
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1 / 2 * beta_t)

    def forward(self, x_t_1: torch.Tensor, t: float, num_t: int) -> torch.Tensor:
        """采样条件分布 p(x(t) | x(t-1))。

        该函数实现了扩散过程中的单步前向转换，将t-1时刻的状态转换为t时刻的状态。

        Args:
            x_t_1: 形状为[..., n, 3]的张量，表示t-1时刻的位置坐标（埃米单位）
            t: 连续时间值，必须在[0, 1]范围内的标量
            num_t: 总时间步数，用于归一化方差调度

        Returns:
            x_t: 形状为[..., n, 3]的张量，表示t时刻的位置坐标

        数学原理：
        使用重参数化技巧进行采样，结合方差调度实现平滑的状态转换：
        x_t = sqrt(1 - b_t) * x_t_1 + sqrt(b_t) * z_t_1
        其中z_t_1是标准正态分布的随机采样。
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(
        self,
        x_t: Union[np.ndarray, torch.Tensor],
        score_t: Union[np.ndarray, torch.Tensor],
        t: float,
        mask: Optional[np.ndarray],
        dt: float,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[float, np.ndarray]]:
        """计算扩散过程的分布参数。

        该函数计算扩散过程中下一时刻状态的分布参数（均值和标准差）。

        Args:
            x_t: 形状为[..., n, 3]的当前状态坐标
            score_t: 形状为[..., n, 3]的score函数值
            t: 当前时间点
            mask: 可选的掩码数组，指示哪些部分参与扩散
            dt: 时间步长

        Returns:
            Tuple[mu, std]:
                mu: 下一时刻状态的均值
                std: 下一时刻状态的标准差

        数学原理：
        基于Fokker-Planck方程，计算转移概率分布的参数：
        μ = x_t - (f_t - g_t²*score_t)*dt
        σ = g_t*sqrt(dt)
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(
        self, x_0: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """采样边际分布 p(x(t) | x(0))。

        该函数实现了从初始状态直接采样t时刻状态的操作，
        不需要逐步模拟中间过程。

        Args:
            x_0: 形状为[..., n, 3]的初始位置坐标（埃米单位）
            t: 目标时间点，必须在[0, 1]范围内

        Returns:
            Tuple[x_t, score_t]:
                x_t: 形状为[..., n, 3]的t时刻位置坐标
                score_t: 形状为[..., n, 3]的t时刻score值

        数学原理：
        利用扩散过程的高斯性质，可以直接计算任意时刻的边际分布：
        x_t ~ N(exp(-1/2*β(t))*x_0, (1-exp(-β(t)))*I)
        其中β(t)是累积方差函数。
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1 / 2 * self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t))),
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: float) -> float:
        """计算score函数的缩放因子。

        Args:
            t: 时间点，范围在[0,1]内

        Returns:
            score的缩放系数，用于归一化score值

        数学原理：
        score缩放系数是条件方差的倒数的平方根，
        用于保持score函数在不同时间尺度上的一致性。
        """
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
        self,
        *,
        x_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        mask: Optional[np.ndarray] = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """模拟一步反向SDE过程。

        该函数实现了扩散模型中的去噪步骤，通过score function指导
        将噪声数据逐步恢复为原始分子构象。

        Args:
            x_t: [..., 3] t时刻的位置坐标（埃米单位）
            score_t: [..., 3] t时刻的score值
            t: 连续时间值，范围[0, 1]
            dt: 时间步长，范围[0, 1]
            mask: 指示哪些残基需要扩散的掩码
            center: 是否对结果进行质心中心化
            noise_scale: 噪声缩放因子

        Returns:
            x_t_1: [..., 3] t-1时刻的位置坐标

        数学原理：
        反向扩散过程基于Langevin动力学，
        结合score function和随机扰动实现构象恢复：
        dx = (f_t - g_t²*score_t)*dt + g_t*sqrt(dt)*dW
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(
        self, t: Union[float, np.ndarray], use_torch: bool = False
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """计算条件方差 p(x_t|x_0)。

        Args:
            t: 时间点，范围[0,1]
            use_torch: 是否使用PyTorch计算

        Returns:
            条件方差值 Var[x_t|x_0] = conditional_var(t)*I

        数学原理：
        条件方差描述了给定初始状态时，t时刻状态的不确定性：
        Var[x_t|x_0] = 1 - exp(-β(t))
        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(
        self,
        x_t: Union[np.ndarray, torch.Tensor],
        x_0: Union[np.ndarray, torch.Tensor],
        t: Union[float, np.ndarray],
        use_torch: bool = False,
        scale: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """计算score function。

        Args:
            x_t: t时刻的状态
            x_0: 初始状态
            t: 时间点
            use_torch: 是否使用PyTorch计算
            scale: 是否对输入进行缩放

        Returns:
            score function值

        数学原理：
        score function是条件概率对状态的对数梯度：
        ∇_x log p(x_t|x_0) = -(x_t - exp(-β(t)/2)*x_0) / conditional_var(t)
        """
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(
            x_t - exp_fn(-1 / 2 * self.marginal_b_t(t)) * x_0
        ) / self.conditional_var(t, use_torch=use_torch)
