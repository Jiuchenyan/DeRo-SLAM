#!/usr/bin/env python3
# randlanet_precise_mapping.py
# 注意：RandLA-Net 工程目录与权重路径都由外部（ROS 参数）传入，见 setup()
import os
import torch
import numpy as np
import sys
import time
import faiss
from numba import jit, prange
import traceback

DEFAULT_RANDLANET_ROOT = "/home/heu/deep_learning/lib/RandLA-Net-pytorch-main"
RANDLANET_ROOT = DEFAULT_RANDLANET_ROOT
WEIGHT_PATH = None          # 由 setup() 填写
cfg = None                  # 由 setup() 填写
Network = None              # 由 setup() 填写
DP = None                   # 由 setup() 填写
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def setup(root=None, weight_path=None):
    """加载 RandLA-Net 工程（sys.path + 模块导入），并确定权重路径。
    root         : RandLA-Net-pytorch-main 目录
    weight_path  : checkpoint.tar 路径（空 = <root>/pretrain_model/checkpoint.tar）
    """
    global RANDLANET_ROOT, WEIGHT_PATH, cfg, Network, DP
    RANDLANET_ROOT = os.path.abspath(root or DEFAULT_RANDLANET_ROOT)
    if not os.path.isdir(RANDLANET_ROOT):
        raise RuntimeError(f"[randlanet] 工程目录不存在: {RANDLANET_ROOT}（用 ~randlanet_root 指定）")
    if RANDLANET_ROOT not in sys.path:
        sys.path.insert(0, RANDLANET_ROOT)

    from utils.config import ConfigSemanticKITTI as _cfg
    from network.RandLANet import Network as _Network
    from utils.data_process import DataProcessing as _DP
    cfg, Network, DP = _cfg, _Network, _DP

    WEIGHT_PATH = weight_path or os.path.join(RANDLANET_ROOT, "pretrain_model/checkpoint.tar")
    if not os.path.isabs(WEIGHT_PATH):
        WEIGHT_PATH = os.path.join(RANDLANET_ROOT, WEIGHT_PATH)
    if not os.path.exists(WEIGHT_PATH):
        raise RuntimeError(f"[randlanet] 权重不存在: {WEIGHT_PATH}（用 ~model_path 指定）")
    print(f"[INFO] RandLA-Net 工程: {RANDLANET_ROOT}")
    print(f"[INFO] RandLA-Net 权重: {WEIGHT_PATH}")
    return WEIGHT_PATH


class RandLANetPreciseInfer:
    def __init__(self, root=None, weight_path=None):
        if cfg is None or root or weight_path:
            setup(root, weight_path)
        # ========== GPU推理极致优化 ==========
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        
        self.model = Network(config=cfg).to(DEVICE)
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        if DEVICE.type == 'cuda':
            self.model = self.model.half()
        
        self.num_points = cfg.num_points          # 45056
        self.grid_size = cfg.sub_grid_size        # 0.06
        self.k_neighbors = 1                       # 语义传播K值
        self.knn_k = cfg.k_n                        # RandLA-Net每层KNN邻域数（默认16）

        # FAISS GPU资源（全局复用，避免重复初始化）
        self.faiss_res = None
        self.faiss_index = None  # 全局FAISS索引，复用减少创建开销
        if DEVICE.type == 'cuda':
            try:
                self.faiss_res = faiss.StandardGpuResources()
                self.faiss_res.setTempMemory(1024 * 1024 * 1024)  # 1GB临时内存
                # 预初始化FAISS索引（3维L2距离）
                self.faiss_index = faiss.IndexFlatL2(3)
                self.faiss_index = faiss.index_cpu_to_gpu(self.faiss_res, 0, self.faiss_index)
                print(f" FAISS GPU资源初始化完成 | 临时内存池: 1GB | 索引预初始化完成")
            except Exception as e:
                self.faiss_index = faiss.IndexFlatL2(3)  # 降级到CPU索引
                print(f" FAISS GPU初始化失败，降级到CPU: {e}")
        else:
            self.faiss_index = faiss.IndexFlatL2(3)  # CPU索引
        
        # 预分配内存缓冲区（减少动态内存申请）
        self.max_points = 130000
        self.point_buffer = np.zeros((self.max_points, 3), dtype=np.float32)
        self.label_buffer = np.zeros(self.max_points, dtype=np.int32)
        # 预分配GPU张量（避免推理时动态创建）
        self.sub_pts_t = torch.zeros((self.max_points, 3), dtype=torch.float32, device=DEVICE)
        self.pred_label_t = torch.zeros(self.num_points, dtype=torch.long, device=DEVICE)
        self.random_idx_t = torch.zeros(self.num_points, dtype=torch.long, device=DEVICE)
        self.all_labels = torch.zeros(self.max_points, dtype=torch.long, device=DEVICE)
        
        print(f"内存缓冲区预分配完成 | 最大支持点云数: {self.max_points}")
        print(f"RandLANet优化版加载完成 | 设备: {DEVICE}")
        print(f"KNN引擎: FAISS {'GPU' if self.faiss_res else 'CPU'} | 索引复用启用")

    def preallocate_buffers(self):
        """保持兼容，实际已在 __init__ 中预分配"""
        pass

    def precise_grid_subsampling(self, points, grid_size=0.06):
        """
        精确的网格下采样（优化哈希计算，减少内存占用）
        返回下采样点、下采样点索引、逆映射索引
        """
        inv_grid_size = 1.0 / grid_size
        grid_indices = (points * inv_grid_size).astype(np.int64)
        # 优化哈希计算：避免大数值相乘，改用更高效的哈希方式
        offset = np.min(grid_indices, axis=0)
        grid_indices_centered = grid_indices - offset
        max_coords = np.max(grid_indices_centered, axis=0) + 1
        hash_values = (
            grid_indices_centered[:, 0] * max_coords[1] * max_coords[2] +
            grid_indices_centered[:, 1] * max_coords[2] +
            grid_indices_centered[:, 2]
        )
        unique_hashes, unique_indices, inverse_indices = np.unique(
            hash_values, 
            return_index=True, 
            return_inverse=True
        )
        subsampled_points = points[unique_indices]
        subsampled_indices = unique_indices
        return subsampled_points, subsampled_indices, inverse_indices

    # 【FAISS优化】复用索引，减少创建开销（核心实时性优化）
    def faiss_knn_search(self, query_points, target_points, k):
        # 类型转换优化：避免重复astype
        if query_points.dtype != np.float32:
            query = query_points.astype(np.float32)
        else:
            query = query_points
        if target_points.dtype != np.float32:
            target = target_points.astype(np.float32)
        else:
            target = target_points
        
        # 复用全局索引，仅清空和重新添加数据（避免每次新建索引）
        self.faiss_index.reset()
        self.faiss_index.add(target)
        # 搜索时禁用临时内存（减少开销）
        distances, indices = self.faiss_index.search(query, k)
        return indices, distances

    def create_input_with_tf_map(self, points):
        try:
            is_cuda = DEVICE.type == 'cuda'
            num_layers = cfg.num_layers
            knn_k = self.knn_k
            sub_sampling_ratios = cfg.sub_sampling_ratio
            
            end_points = {
                'xyz': [None] * num_layers,
                'neigh_idx': [None] * num_layers,
                'sub_idx': [None] * num_layers,
                'interp_idx': [None] * num_layers
            }
            
            batch_pc = points[np.newaxis, ...]  # 已为float32，无需重复转换
            features = batch_pc.copy()
            
            dtype_xyz = torch.float16 if is_cuda else torch.float32
            # 优化数据拷贝：non_blocking=True + 减少中间变量
            features_gpu = torch.from_numpy(features).to(device=DEVICE, dtype=dtype_xyz, non_blocking=True)
            features_gpu = features_gpu.transpose(1, 2).contiguous()
            
            # t1 = time.time()
            for i in range(num_layers):
                pc_flat = batch_pc.squeeze(0)
                # KNN搜索（复用索引，减少耗时）
                neighbour_idx_flat, _ = self.faiss_knn_search(pc_flat, pc_flat, knn_k)
                neighbour_idx = neighbour_idx_flat[np.newaxis, ...]
                
                sub_points = batch_pc[:, :batch_pc.shape[1] // sub_sampling_ratios[i], :]
                pool_i = neighbour_idx[:, :batch_pc.shape[1] // sub_sampling_ratios[i], :]
                
                sub_points_flat = sub_points.squeeze(0)
                up_i_flat, _ = self.faiss_knn_search(pc_flat, sub_points_flat, 1)
                up_i = up_i_flat[np.newaxis, ...]
                
                dtype_idx = torch.long
                # 优化：直接转换，减少临时变量
                end_points['xyz'][i] = torch.from_numpy(batch_pc).to(device=DEVICE, dtype=dtype_xyz, non_blocking=True)
                end_points['neigh_idx'][i] = torch.from_numpy(neighbour_idx).to(device=DEVICE, dtype=dtype_idx, non_blocking=True)
                end_points['sub_idx'][i] = torch.from_numpy(pool_i).to(device=DEVICE, dtype=dtype_idx, non_blocking=True)
                end_points['interp_idx'][i] = torch.from_numpy(up_i).to(device=DEVICE, dtype=dtype_idx, non_blocking=True)
                
                batch_pc = sub_points
            
            # t2 = time.time()
            # print(f"  KNN_all: {(t2-t1)*1000:.1f}ms")
            # 仅在需要同步时执行（减少cuda同步开销）
            if is_cuda:
                torch.cuda.synchronize()
            
            end_points['features'] = features_gpu
            return end_points
        except Exception as e:
            print(f"创建输入失败: {e}")
            traceback.print_exc()
            return None

    def propagate_semantic_gpu(self, subsampled_points, random_sample_indices, pred_label):
        """
        修复NameError + 优化实时性：预分配张量、减少数据拷贝、简化逻辑
        """
        num_sub = len(subsampled_points)
        if num_sub > self.max_points:
            print(f"警告：点云数({num_sub})超过预分配缓冲区({self.max_points})，动态扩容")
            self.all_labels = torch.zeros(num_sub, dtype=torch.long, device=DEVICE)
        
        # 1. 修复核心错误：预分配张量复用，减少拷贝
        self.sub_pts_t[:num_sub] = torch.from_numpy(subsampled_points).float().to(DEVICE, non_blocking=True)
        self.pred_label_t[:len(pred_label)] = torch.from_numpy(pred_label).long().to(DEVICE, non_blocking=True)
        self.random_idx_t[:len(random_sample_indices)] = torch.from_numpy(random_sample_indices).long().to(DEVICE, non_blocking=True)
        
        # 2. 初始化标签（复用预分配张量）
        self.all_labels.zero_()  # 快速置零，比重新创建快
        self.all_labels[self.random_idx_t[:len(random_sample_indices)]] = self.pred_label_t[:len(pred_label)]
        
        # 3. 区分有标签/无标签点
        has_label_mask = self.all_labels[:num_sub] != 0
        has_label_idx = torch.nonzero(has_label_mask).squeeze(1)
        no_label_idx = torch.nonzero(~has_label_mask).squeeze(1)
        
        # 无需要传播的点，直接返回
        if no_label_idx.numel() == 0:
            return self.all_labels[:num_sub].cpu().numpy()
        
        # 4. K值限制（避免越界）
        k = min(self.k_neighbors, has_label_idx.size(0))
        
        # 5. KNN搜索（复用全局FAISS索引）
        query_pts = subsampled_points[no_label_idx.cpu().numpy()]
        target_pts = subsampled_points[has_label_idx.cpu().numpy()]
        indices_np, _ = self.faiss_knn_search(query_pts, target_pts, k)
        indices = torch.from_numpy(indices_np).long().to(DEVICE, non_blocking=True)
        
        # ========== 修复NameError：补充neighbor_labels定义 ==========
        # 获取邻居标签（核心漏写行）
        neighbor_labels = self.all_labels[has_label_idx][indices]
        
        # 6. 标签投票（优化计算逻辑，减少中间变量）
        neighbor_labels_0based = neighbor_labels - 1  # 转为0~19
        one_hot = torch.nn.functional.one_hot(neighbor_labels_0based, num_classes=20)
        hist = one_hot.sum(dim=1)
        pred_no_label = hist.argmax(dim=1) + 1  # 恢复为1~20
        
        # 7. 赋值传播后的标签
        self.all_labels[no_label_idx] = pred_no_label
        
        # 8. 仅必要时同步，减少开销
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        
        return self.all_labels[:num_sub].cpu().numpy()

    @torch.no_grad()
    def infer(self, ros_pc_np):
        try:
            mask = ros_pc_np[:, 2] != -100000.0
            valid_pc_xyz = ros_pc_np[mask][:, :3].astype(np.float32)  # 提前转换类型，避免重复转换
            if len(valid_pc_xyz) == 0:
                return None, None, None
            
            # t1 = time.time()
            # 1. 网格下采样（优化版）
            subsampled_points, subsampled_indices, inverse_indices = self.precise_grid_subsampling(valid_pc_xyz, self.grid_size)
            # t2 = time.time()

            # 2. 随机采样（简化逻辑）
            num_sub = len(subsampled_points)
            if num_sub > self.num_points:
                random_indices = np.random.choice(num_sub, self.num_points, replace=False)
            else:
                random_indices = np.random.choice(num_sub, self.num_points, replace=True)
            sampled_points = subsampled_points[random_indices]
            # t3 = time.time()

            # 3. 数据预处理（复用FAISS索引）
            end_points = self.create_input_with_tf_map(sampled_points)
            if end_points is None:
                return None, None, None
            # t4 = time.time()

            # 4. 模型推理（half精度加速）
            pred_end_points = self.model(end_points)
            # t5 = time.time()

            # 5. 解析推理结果（简化计算）
            probs = torch.nn.functional.softmax(pred_end_points['logits'], dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs, axis=0) + 1
            pred_confidence = probs[pred_label - 1, np.arange(len(pred_label))]

            # 6. 语义传播（修复后）
            sub_to_label = self.propagate_semantic_gpu(subsampled_points, random_indices, pred_label)
            # t6 = time.time()

            # 7. 映射回原始点（优化：仅必要时转换）
            S = len(subsampled_points)
            sub_to_confidence = np.zeros(S, dtype=np.float32)
            sub_to_confidence[random_indices] = pred_confidence  # 仅采样点有置信度
            
            orig_labels = sub_to_label[inverse_indices]
            orig_confidences = sub_to_confidence[inverse_indices]
            
            # 打印时间统计
            # print(f"  网格下采样: {(t2-t1)*1000:.1f}ms")
            # print(f"  随机采样: {(t3-t2)*1000:.1f}ms")
            # print(f"  数据预处理: {(t4-t3)*1000:.1f}ms")
            # print(f"  模型推理: {(t5-t4)*1000:.1f}ms")
            # print(f"  语义传播: {(t6-t5)*1000:.1f}ms")
            # print(f"  总耗时: {(t6-t1)*1000:.1f}ms")

            return valid_pc_xyz, orig_labels, orig_confidences

        except Exception as e:
            print(f"推理失败: {e}")
            traceback.print_exc()
            return None, None, None