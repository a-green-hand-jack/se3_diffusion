"""SE(3) diffusion methods.
SE3扩散方法 - 用于处理3D刚体运动的扩散模型,包含旋转(SO3)和平移(R3)两部分。
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from data import so3_diffuser, r3_diffuser
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils as ru
from data import utils as du
import torch
import logging


def _extract_trans_rots(rigid: ru.Rigid) -> Tuple[np.ndarray, np.ndarray]:
    """从Rigid对象中提取平移和旋转信息

    Args:
        rigid: OpenFold的Rigid对象,包含旋转矩阵和平移向量

    Returns:
        tran: [..., 3] 平移向量
        rot: [..., 3] 旋转向量(以Rodriguez旋转向量形式表示)
    """
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] + (3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot


def _assemble_rigid(rotvec: np.ndarray, trans: np.ndarray) -> ru.Rigid:
    """将旋转向量和平移向量组装成Rigid对象

    Args:
        rotvec: [..., 3] Rodriguez旋转向量
        trans: [..., 3] 平移向量

    Returns:
        ru.Rigid对象
    """
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = (
        Rotation.from_rotvec(rotvec).as_matrix().reshape(rotvec_shape[:-1] + (3, 3))
    )
    return ru.Rigid(
        rots=ru.Rotation(rot_mats=torch.Tensor(rotmat)), trans=torch.tensor(trans)
    )


class SE3Diffuser:
    """SE(3)扩散模型,结合SO(3)旋转扩散和R3平移扩散

    用于对蛋白质骨架等3D结构进行扩散和生成。可以选择性地只对旋转或平移进行扩散。
    """

    def __init__(self, se3_conf):
        """初始化SE3扩散器

        Args:
            se3_conf: 配置对象,包含so3和r3的配置信息
        """
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        # 是否对旋转进行扩散
        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        # 是否对平移进行扩散
        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    def forward_marginal(
        self,
        rigids_0: ru.Rigid,
        t: float,
        diffuse_mask: Optional[np.ndarray] = None,
        as_tensor_7: bool = True,
    ) -> Dict[str, Union[ru.Rigid, np.ndarray, float]]:
        """前向扩散过程,将t=0时刻的构象扩散到t时刻

        Args:
            rigids_0: 初始构象,Rigid对象
            t: 扩散时间,范围[0,1]
            diffuse_mask: 指定哪些残基需要扩散的掩码
            as_tensor_7: 是否将结果转换为7维张量表示(四元数+平移)

        Returns:
            包含以下键的字典:
            - rigids_t: t时刻的构象
            - trans_score: 平移score
            - rot_score: 旋转score
            - trans_score_scaling: 平移score缩放因子
            - rot_score_scaling: 旋转score缩放因子
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t),
            )
        else:
            rot_t, rot_score = self._so3_diffuser.forward_marginal(rot_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t),
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(trans_0, t)
            trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            rot_t = self._apply_mask(rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score, np.zeros_like(trans_score), diffuse_mask[..., None]
            )
            rot_score = self._apply_mask(
                rot_score, np.zeros_like(rot_score), diffuse_mask[..., None]
            )
        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            "rigids_t": rigids_t,
            "trans_score": trans_score,
            "rot_score": rot_score,
            "trans_score_scaling": trans_score_scaling,
            "rot_score_scaling": rot_score_scaling,
        }

    def calc_trans_0(
        self, trans_score: np.ndarray, trans_t: np.ndarray, t: float
    ) -> np.ndarray:
        """计算t时刻的平移score对应的初始平移位置

        Args:
            trans_score: [..., 3] 平移score
            trans_t: [..., 3] t时刻的平移向量
            t: 时间点

        Returns:
            [..., 3] 估计的初始平移向量
        """
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(
        self,
        trans_t: np.ndarray,
        trans_0: np.ndarray,
        t: float,
        use_torch: bool = False,
        scale: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """计算平移部分的score function

        Args:
            trans_t: t时刻的平移向量
            trans_0: 初始平移向量
            t: 时间点
            use_torch: 是否使用PyTorch计算
            scale: 是否对结果进行缩放

        Returns:
            平移score向量
        """
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale
        )

    def calc_rot_score(
        self, rots_t: ru.Rotation, rots_0: ru.Rotation, t: float
    ) -> torch.Tensor:
        """计算旋转部分的score function

        Args:
            rots_t: t时刻的旋转
            rots_0: 初始旋转
            t: 时间点

        Returns:
            旋转score向量
        """
        # 计算rots_0的逆旋转
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        # 计算从0到t的旋转差异
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        # 转换为旋转向量表示
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._so3_diffuser.torch_score(rotvec_0t, t)

    def _apply_mask(
        self, x_diff: np.ndarray, x_fixed: np.ndarray, diff_mask: np.ndarray
    ) -> np.ndarray:
        """应用扩散掩码,选择性地更新部分位置

        Args:
            x_diff: 扩散后的值
            x_fixed: 固定的值
            diff_mask: 扩散掩码,1表示需要扩散,0表示保持固定

        Returns:
            混合后的结果
        """
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(
        self,
        trans_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算平移扩散的分布参数

        Args:
            trans_t: [..., 3] t时刻的平移向量
            score_t: [..., 3] t时刻的score
            t: 当前时间点
            dt: 时间步长
            mask: 可选的掩码,指定哪些位置需要计算

        Returns:
            mean: 条件分布的均值
            std: 条件分布的标准差
        """
        return self._r3_diffuser.distribution(trans_t, score_t, t, dt, mask)

    def score(
        self, rigid_0: ru.Rigid, rigid_t: ru.Rigid, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算给定构象对的score function

        Args:
            rigid_0: 初始构象
            rigid_t: t时刻的构象
            t: 时间点

        Returns:
            trans_score: 平移score
            rot_score: 旋转score
        """
        # 提取平移和旋转信息
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        # 如果不扩散旋转,score为0
        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(rot_t, t)

        # 如果不扩散平移,score为0
        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算score的缩放因子

        在训练和推理过程中,score需要根据时间t进行适当的缩放,
        以保证数值稳定性和训练效果。

        Args:
            t: 时间点

        Returns:
            rot_score_scaling: 旋转score的缩放因子
            trans_score_scaling: 平移score的缩放因子
        """
        # 分别获取旋转和平移的score缩放因子
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
        self,
        rigid_t: ru.Rigid,
        rot_score: np.ndarray,
        trans_score: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: Optional[np.ndarray] = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> ru.Rigid:
        """反向扩散采样,从t时刻采样到(t-dt)时刻

        Args:
            rigid_t: t时刻的构象
            rot_score: 旋转score
            trans_score: 平移score
            t: 当前时间点
            dt: 时间步长
            diffuse_mask: 扩散掩码
            center: 是否将质心置于原点
            noise_scale: 噪声强度缩放因子

        Returns:
            t-dt时刻的构象
        """
        # 提取平移和旋转信息
        trans_t, rot_t = _extract_trans_rots(rigid_t)

        # 处理旋转部分
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
            )

        # 处理平移部分
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale,
            )

        # 应用扩散掩码
        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref(
        self,
        n_samples: int,
        impute: Optional[ru.Rigid] = None,
        diffuse_mask: Optional[np.ndarray] = None,
        as_tensor_7: bool = False,
    ) -> Dict[str, Union[ru.Rigid, torch.Tensor]]:
        """从参考分布中采样构象

        Args:
            n_samples: 采样数量
            impute: 用于填充未扩散部分的构象
            diffuse_mask: 扩散掩码
            as_tensor_7: 是否转换为7维张量表示

        Returns:
            包含采样得到的构象的字典

        Raises:
            ValueError: 当需要填充值但未提供impute时抛出
        """
        # 验证输入参数
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError("使用diffuse_mask时必须提供impute值")

        if (not self._diffuse_rot) and impute is None:
            raise ValueError("不进行旋转扩散时必须提供impute值")

        if (not self._diffuse_trans) and impute is None:
            raise ValueError("不进行平移扩散时必须提供impute值")

        # 采样旋转部分
        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(n_samples=n_samples)
        else:
            rot_ref = rot_impute

        # 采样平移部分
        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(n_samples=n_samples)
        else:
            trans_ref = trans_impute

        # 应用扩散掩码
        if diffuse_mask is not None:
            rot_ref = self._apply_mask(rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None]
            )

        # 反归一化平移
        trans_ref = self._r3_diffuser._unscale(trans_ref)

        # 组装最终结果
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {"rigids_t": rigids_t}
