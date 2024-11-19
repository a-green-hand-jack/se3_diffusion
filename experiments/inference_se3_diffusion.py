"""Script for running inference and sampling.

这个脚本实现了一个蛋白质结构采样器,主要功能包括:
1. 使用SE(3)扩散模型生成蛋白质骨架结构
2. 使用ProteinMPNN为生成的骨架设计氨基酸序列
3. 使用ESMFold验证设计的序列能否折叠成目标结构
4. 计算各种结构相似性指标(TM-score, RMSD等)

主要组件:
- Sampler类: 封装了采样和评估的核心功能
- 支持批量生成不同长度的样本
- 自动选择可用GPU
- 结果保存在指定输出目录

Sample command:
> python scripts/run_inference.py

"""

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional, Any

import esm
import GPUtil
import hydra
import numpy as np
import pandas as pd
import torch
import tree
from biotite.sequence.io import fasta
from omegaconf import DictConfig, OmegaConf

from analysis import metrics
from analysis import utils as au
from data import residue_constants
from data import utils as du
from experiments import train_se3_diffusion
from openfold.data import data_transforms

CA_IDX = residue_constants.atom_order['CA']


from typing import TypedDict, Union



class PDBFeatures(TypedDict):
    aatype: np.ndarray  # [L] 氨基酸类型编号
    atom_positions: np.ndarray  # [L, 37, 3] 原子坐标
    atom_mask: np.ndarray  # [L, 37] 原子mask
    residue_index: np.ndarray  # [L] 残基编号
    bb_mask: np.ndarray  # [L] 骨架原子mask

class ChainFeatures(TypedDict):
    aatype: torch.Tensor  # [L] 氨基酸类型
    all_atom_positions: torch.Tensor  # [L, 37, 3] 原子坐标
    all_atom_mask: torch.Tensor  # [L, 37] 原子mask
    seq_idx: np.ndarray  # [L] 序列位置编号(从1开始)
    res_mask: np.ndarray  # [L] 残基mask
    residue_index: np.ndarray  # [L] 原始PDB中的残基编号
    # ... 其他由OpenFold transforms添加的特征

def process_chain(design_pdb_feats: PDBFeatures) -> ChainFeatures:
    """处理PDB特征,转换为模型所需的链级特征格式。
    
    该函数主要完成以下转换:
    1. 将numpy数组转为torch张量
    2. 使用OpenFold的transforms将原子坐标转换为内部表示:
       - atom37_to_frames: 将原子坐标转换为局部坐标系
       - make_atom14_*: 生成14原子表示相关特征
       - atom37_to_torsion_angles: 计算主链和侧链二面角
    3. 添加序列位置编号等额外特征
    
    Args:
        design_pdb_feats: 从PDB解析出的原始特征字典
            包含原子坐标、氨基酸类型等基本信息
            
    Returns:
        chain_feats: 处理后的特征字典
            包含模型训练和推理所需的所有特征
    """
    # 转换为torch张量
    chain_feats = {
        'aatype': torch.tensor(design_pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(design_pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(design_pdb_feats['atom_mask']).double()
    }
    
    # 应用OpenFold transforms进行坐标系转换
    chain_feats = data_transforms.atom37_to_frames(chain_feats)  # 生成局部坐标系
    chain_feats = data_transforms.make_atom14_masks(chain_feats)  # 生成14原子表示的mask
    chain_feats = data_transforms.make_atom14_positions(chain_feats)  # 转换为14原子表示
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)  # 计算二面角
    
    # 生成从1开始的序列位置编号
    seq_idx = design_pdb_feats['residue_index'] - np.min(design_pdb_feats['residue_index']) + 1
    
    # 添加额外特征
    chain_feats['seq_idx'] = seq_idx  # 序列位置编号
    chain_feats['res_mask'] = design_pdb_feats['bb_mask']  # 残基mask
    chain_feats['residue_index'] = design_pdb_feats['residue_index']  # 原始残基编号
    
    return chain_feats


class PaddingFeatures(TypedDict):
    res_mask: torch.Tensor  # [pad_amt] 残基mask,全1
    fixed_mask: torch.Tensor  # [pad_amt] 固定位置mask,全0
    rigids_impute: torch.Tensor  # [pad_amt, 4, 4] 刚体变换矩阵,全0
    torsion_impute: torch.Tensor  # [pad_amt, 7, 2] 7个二面角的sin/cos值,全0
                                 # 7个二面角包括:
                                 # - 主链二面角: φ(phi), ψ(psi), ω(omega)
                                 # - 侧链二面角: χ1, χ2, χ3, χ4
                                 # 每个角度用(sin,cos)值表示,所以最后一维是2

def create_pad_feats(pad_amt: int) -> PaddingFeatures:
    """创建用于序列填充的特征字典。
    
    在处理不等长序列时,需要将较短的序列填充到相同长度。
    该函数生成填充位置所需的特征值:
    - res_mask设为1表示这些位置在计算时需要考虑
    - fixed_mask设为0表示这些位置不是固定的模板残基
    - rigids_impute和torsion_impute设为0作为坐标和角度的初始值
    
    Args:
        pad_amt: 需要填充的残基数量
        
    Returns:
        pad_feats: 包含填充特征的字典
            res_mask: [pad_amt] 残基mask,全1
            fixed_mask: [pad_amt] 固定位置mask,全0  
            rigids_impute: [pad_amt, 4, 4] 刚体变换矩阵,全0
            torsion_impute: [pad_amt, 7, 2] 二面角sin/cos值,全0
    """
    return {        
        'res_mask': torch.ones(pad_amt),  # 标记填充位置为有效残基
        'fixed_mask': torch.zeros(pad_amt),  # 标记填充位置为非固定残基
        'rigids_impute': torch.zeros((pad_amt, 4, 4)),  # 初始化刚体变换矩阵
        'torsion_impute': torch.zeros((pad_amt, 7, 2)),  # 初始化7个二面角的sin/cos值
    }


class Sampler:
    """蛋白质结构采样器，用于生成和评估新的蛋白质结构。

    主要功能:
    1. 使用预训练的SE(3)扩散模型生成蛋白质骨架结构
    2. 调用ProteinMPNN为生成的骨架设计氨基酸序列
    3. 使用ESMFold验证设计序列的可折叠性
    4. 计算结构相似性指标(TM-score, RMSD等)

    关键组件:
    - SE(3)扩散模型: 生成蛋白质骨架的3D坐标
    - ProteinMPNN: 序列设计模型
    - ESMFold: 结构预测模型，用于验证
    
    工作流程:
    1. 初始化时加载预训练模型
    2. run_sampling(): 批量生成不同长度的样本
    3. 对每个样本:
       - sample(): 使用扩散模型生成骨架
       - run_self_consistency(): 进行序列设计和结构验证
    4. 结果保存在指定输出目录

    Args:
        conf (DictConfig): 配置参数
        conf_overrides (Dict, optional): 需要覆盖的配置项

    Attributes:
        device (str): 使用的计算设备(GPU/CPU)
        model: SE(3)扩散模型
        _folding_model: ESMFold模型
        _output_dir (str): 结果输出目录
    """

    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Optional[Dict] = None
        ) -> None:
        """初始化蛋白质结构采样器。
        
        主要完成以下初始化工作:
        1. 配置处理:
           - 加载基础配置
           - 应用配置覆盖
           - 设置随机种子
        2. 计算设备设置:
           - 自动选择可用GPU
           - 如无GPU则使用CPU
        3. 模型加载:
           - 加载预训练的SE(3)扩散模型
           - 加载ESMFold模型
        4. 输出目录设置:
           - 创建时间戳目录
           - 保存配置文件
        
        Args:
            conf: 基础配置,包含:
                inference: 推理相关配置
                    - seed: 随机种子
                    - pt_hub_dir: PyTorch模型目录
                    - gpu_id: 指定GPU ID
                    - weights_path: 模型权重路径
                    - output_dir: 输出目录
                    - pmpnn_dir: ProteinMPNN目录
                model: 模型相关配置
            conf_overrides: 需要覆盖的配置项
        
        Attributes:
            device (str): 使用的计算设备
            model: SE(3)扩散模型
            _folding_model: ESMFold模型
            _output_dir (str): 结果输出目录
            _log: 日志记录器
        """
        # 设置日志记录器
        self._log = logging.getLogger(__name__)

        # 配置处理
        OmegaConf.set_struct(conf, False)  # 允许动态修改配置
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples

        # 设置随机种子
        self._rng = np.random.default_rng(self._infer_conf.seed)

        # 设置PyTorch模型目录
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)

        # 设置计算设备
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                # 自动选择显存最大的GPU
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # 设置目录
        self._weights_path = self._infer_conf.weights_path
        output_dir = self._infer_conf.output_dir
        if self._infer_conf.name is None:
            # 使用时间戳作为目录名
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.join(output_dir, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        self._pmpnn_dir = self._infer_conf.pmpnn_dir

        # 保存配置
        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # 加载模型
        self._load_ckpt(conf_overrides)  # 加载SE(3)扩散模型
        self._folding_model = esm.pretrained.esmfold_v1().eval()  # 加载ESMFold
        self._folding_model = self._folding_model.to(self.device)

    def _load_ckpt(self, conf_overrides: Optional[Dict[str, Any]]) -> None:
        """加载预训练的SE(3)扩散模型检查点。

        主要步骤:
        1. 读取模型权重文件
        2. 合并基础配置和检查点配置
        3. 初始化模型并加载权重
        4. 将模型移至指定设备(GPU/CPU)

        Args:
            conf_overrides: 需要覆盖的配置项。
                可以覆盖模型的默认配置参数。

        注意:
            - 模型权重路径由self._weights_path指定
            - 会移除权重字典中的'module.'前缀(分布式训练产生)
            - 加载后的模型会被设置为评估模式(eval)
        """
        self._log.info(f'Loading weights from {self._weights_path}')

        # 读取检查点并创建实验
        weights_pkl = du.read_pkl(
            self._weights_path, 
            use_torch=True,
            map_location=self.device
        )

        # 合并基础实验配置和检查点配置
        self._conf.model = OmegaConf.merge(
            self._conf.model, 
            weights_pkl['conf'].model
        )
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # 准备模型
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_se3_diffusion.Experiment(conf=self._conf)
        self.model = self.exp.model

        # 移除module前缀并加载权重
        model_weights = weights_pkl['model']
        model_weights = {
            k.replace('module.', ''): v 
            for k, v in model_weights.items()
        }
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def init_data(
            self,
            *,  # 强制使用关键字参数
            rigids_impute: torch.Tensor,  # [N, 4, 4] 刚体变换矩阵
            torsion_impute: torch.Tensor,  # [N, 7, 2] 二面角sin/cos值
            fixed_mask: torch.Tensor,  # [N] 固定位置mask
            res_mask: torch.Tensor,  # [N] 残基mask
        ) -> Dict[str, torch.Tensor]:
        """初始化扩散模型的输入数据。

        主要步骤:
        1. 计算扩散mask(哪些位置需要进行扩散)
        2. 采样参考噪声
        3. 生成序列位置编号
        4. 组装所有特征并移至GPU

        Args:
            rigids_impute: 刚体变换矩阵初始值
            torsion_impute: 二面角sin/cos值初始值
            fixed_mask: 标记固定位置(如模板残基)的mask
            res_mask: 标记有效残基的mask

        Returns:
            初始化的特征字典,包含:
            - res_mask: 残基mask
            - seq_idx: 序列位置编号
            - fixed_mask: 固定位置mask
            - torsion_angles_sin_cos: 二面角值
            - sc_ca_t: 侧链Cα原子相对位置
            - 参考噪声采样等其他特征
            所有特征都添加了batch维度并移至指定设备
        """
        num_res = res_mask.shape[0]
        # 计算扩散mask: 非固定且有效的位置
        diffuse_mask = (1 - fixed_mask) * res_mask
        fixed_mask = fixed_mask * res_mask

        # 采样参考噪声
        ref_sample = self.diffuser.sample_ref(
            n_samples=num_res,
            rigids_impute=rigids_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True,
        )

        # 生成序列位置编号(从1开始)
        res_idx = torch.arange(1, num_res+1)

        # 组装特征字典
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx * res_mask,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': torsion_impute,
            'sc_ca_t': torch.zeros_like(rigids_impute.get_trans()),
            **ref_sample,
        }

        # 将所有特征转换为tensor并添加batch维度
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), 
            init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), 
            init_feats
        )

        return init_feats

    def run_sampling(self) -> None:
        """执行蛋白质结构采样的主流程。

        主要步骤:
        1. 遍历配置的序列长度范围
        2. 对每个长度生成多个样本
        3. 对每个样本:
           - 使用扩散模型生成骨架结构
           - 保存采样轨迹
           - 运行序列设计和结构验证
        
        配置参数(来自self._sample_conf):
        - min_length: 最小序列长度
        - max_length: 最大序列长度
        - length_step: 长度递增步长
        - samples_per_length: 每个长度生成的样本数

        输出目录结构:
        {output_dir}/
        ├── length_{length}/
        │   ├── sample_0/
        │   │   ├── sample.pdb  # 最终结构
        │   │   ├── bb_traj.pdb # 骨架轨迹
        │   │   ├── x0_traj.pdb # 预测轨迹
        │   │   └── self_consistency/  # 序列设计结果
        │   ├── sample_1/
        │   └── ...
        └── ...
        """
        # 生成需要采样的序列长度列表
        all_sample_lengths = range(
            self._sample_conf.min_length,
            self._sample_conf.max_length + 1,
            self._sample_conf.length_step
        )

        # 对每个长度进行采样
        for sample_length in all_sample_lengths:
            # 创建长度特定的输出目录
            length_dir = os.path.join(
                self._output_dir, f'length_{sample_length}')
            os.makedirs(length_dir, exist_ok=True)
            self._log.info(f'Sampling length {sample_length}: {length_dir}')

            # 生成指定数量的样本
            for sample_i in range(self._sample_conf.samples_per_length):
                sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
                if os.path.isdir(sample_dir):
                    continue  # 跳过已存在的样本
                os.makedirs(sample_dir, exist_ok=True)

                # 生成样本并保存轨迹
                sample_output = self.sample(sample_length)
                traj_paths = self.save_traj(
                    sample_output['prot_traj'],
                    sample_output['rigid_0_traj'],
                    np.ones(sample_length),
                    output_dir=sample_dir
                )

                # 运行序列设计和结构验证
                pdb_path = traj_paths['sample_path']
                sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                os.makedirs(sc_output_dir, exist_ok=True)
                shutil.copy(pdb_path, os.path.join(
                    sc_output_dir, os.path.basename(pdb_path)))
                _ = self.run_self_consistency(
                    sc_output_dir,
                    pdb_path,
                    motif_mask=None
                )
                self._log.info(f'Done sample {sample_i}: {pdb_path}')

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,  # [T, N, 37, 3] 
            x0_traj: np.ndarray,      # [T, N, 3]
            diffuse_mask: np.ndarray,  # [N]
            output_dir: str
        ) -> Dict[str, str]:
        """保存采样轨迹和最终结构。

        Args:
            bb_prot_traj: 骨架原子轨迹
                - T: 时间步数
                - N: 残基数
                - 37: 每个残基的原子数
                - 3: 3D坐标
                第一个时间步(t=eps)是最终样本
            x0_traj: Cα原子的x0预测轨迹
                每个时间步的预测值
            diffuse_mask: 标记哪些残基进行了扩散
            output_dir: 输出目录
            
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse 
                diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            包含保存文件路径的字典:
            - sample_path: 最终结构的PDB文件
            - traj_path: 完整轨迹的PDB文件
            - x0_traj_path: x0预测轨迹的PDB文件

        注意:
            - b-factors被设置为:
              - 100: 扩散的残基
              - 0: 模板残基(如果有)
            - 最终结构是轨迹的第一帧(t=eps)
        """
        # 准备文件路径
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, 'sample')
        prot_traj_path = os.path.join(output_dir, 'bb_traj')
        x0_traj_path = os.path.join(output_dir, 'x0_traj')

        # 设置b-factors来标记扩散残基
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        # 保存结构文件
        sample_path = au.write_prot_to_pdb(
            bb_prot_traj[0],  # 第一帧是最终结构
            sample_path,
            b_factors=b_factors
        )
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors
        )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,
            motif_mask: Optional[np.ndarray] = None
        ) -> pd.DataFrame:
        """运行序列设计和结构验证的自洽性评估。

        主要步骤:
        1. 使用ProteinMPNN对输入结构进行序列设计
        2. 对每个设计的序列:
           - 使用ESMFold预测其结构
           - 计算与参考结构的相似度指标
        3. 保存结果和评估指标

        Args:
            decoy_pdb_dir: 输出目录，用于存放设计序列和预测结构
            reference_pdb_path: 参考蛋白质结构的PDB文件路径
            motif_mask: 可选，标记motif残基的mask，用于计算局部RMSD

        Returns:
            包含评估结果的DataFrame:
            - tm_score: TM-score与参考结构的相似度
            - rmsd: 整体RMSD
            - motif_rmsd: (如果提供motif_mask)motif区域的RMSD
            - sample_path: 预测结构的文件路径
            - sequence: 设计的氨基酸序列
            
        目录结构:
        decoy_pdb_dir/
        ├── seqs/          # ProteinMPNN输出的序列
        ├── esmf/          # ESMFold预测的结构
        └── sc_results.csv # 评估结果
        """
        # 运行ProteinMPNN进行序列设计
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{self._pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            decoy_pdb_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._sample_conf.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        if self._infer_conf.gpu_id is not None:
            pmpnn_args.extend(['--device', str(self._infer_conf.gpu_id)])
            
        while ret < 0 and num_tries < 5:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e

        # 获取设计的序列文件路径
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )

        # 对每个设计序列进行结构预测和评估
        mpnn_results = {
            'tm_score': [], 'sample_path': [], 'header': [],
            'sequence': [], 'rmsd': []
        }
        if motif_mask is not None:
            mpnn_results['motif_rmsd'] = []
            
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        
        # 读取序列并进行评估
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        
        for i, (header, sequence) in enumerate(fasta_seqs.items()):
            # 运行ESMFold预测结构
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(sequence, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # 计算结构相似性指标
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], 
                esmf_feats['bb_positions'],
                sample_seq, sample_seq
            )
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], 
                esmf_feats['bb_positions']
            )
            
            # 如果提供了motif_mask，计算motif区域的RMSD
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
                
            # 收集结果
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(sequence)

        # 保存结果到CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
        return mpnn_results

    def run_folding(
            self, 
            sequence: str, 
            save_path: str
        ) -> str:
        """使用ESMFold预测蛋白质序列的结构。

        Args:
            sequence: 氨基酸序列
            save_path: 输出PDB文件路径

        Returns:
            生成的PDB文件内容字符串

        注意:
            - 使用预加载的ESMFold模型(_folding_model)
            - 预测在GPU上进行(如果可用)
            - 输出以PDB格式保存
        """
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def sample(
            self, 
            sample_length: int
        ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """生成指定长度的蛋白质结构样本。

        主要步骤:
        1. 创建残基mask和固定位置mask
        2. 初始化参考噪声和特征
        3. 运行扩散模型生成结构

        Args:
            sample_length: 要生成的蛋白质长度(残基数)

        Returns:
            包含采样结果的字典:
            - prot_traj: [T, L, 37, 3] 骨架原子轨迹
            - rigid_0_traj: [T, L, 4, 4] 刚体变换矩阵轨迹
            其中T是时间步数，L是序列长度

        注意:
            - 使用预训练的SE(3)扩散模型
            - 采样参数来自self._diff_conf配置
            - 返回完整的采样轨迹
        """
        # 初始化mask
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # 初始化参考噪声和特征
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, sample_length+1)
        
        # 组装初始特征
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        
        # 转换为tensor并添加batch维度
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), 
            init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), 
            init_feats
        )

        # 运行扩散模型采样
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t, 
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale,
        )
        
        # 移除batch维度
        return tree.map_structure(lambda x: x[:, 0], sample_out)



@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
