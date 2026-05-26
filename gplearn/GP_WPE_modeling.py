# -*- coding: gbk -*-
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import pywt
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from typing import Tuple, List, Dict

# 中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def wucha_zhibiao(data_shiji: np.ndarray, data_yuce: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """计算误差指标：ERMS、EMA、EMS、R2、EMI、MAPE"""
    # 去除所有的NaN值
    valid_indices = ~np.isnan(data_shiji) & ~np.isnan(data_yuce)
    data_shiji = data_shiji[valid_indices]
    data_yuce = data_yuce[valid_indices]

    if len(data_shiji) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    n = len(data_shiji)

    # 均方根误差 (ERMS)
    erms = np.sqrt(np.mean((data_shiji - data_yuce) ** 2))

    # 平均绝对误差 (EMA)
    ema = np.mean(np.abs(data_shiji - data_yuce))

    # 均方误差 (EMS)
    ems = np.mean((data_shiji - data_yuce) ** 2)

    # 决定系数 (R2)
    ss_res = np.sum((data_shiji - data_yuce) ** 2)
    ss_tot = np.sum((data_shiji - np.mean(data_shiji)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # 最大绝对误差 (EMI)
    emi = np.max(np.abs(data_shiji - data_yuce))

    # 平均绝对百分比误差 (MAPE)，避免除以零
    mape = np.mean(np.abs((data_shiji - data_yuce) / np.clip(data_shiji, 1e-10, None))) * 100  # 转换为百分比

    return erms, ema, ems, r2, emi, mape


def wavelet_decomposition(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """进行小波分解，自动选择分解层数并处理边界效应"""
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([])

    # 确保是一维数组
    if len(data.shape) > 1:
        data = data.flatten()

    # 根据数据长度动态选择分解层数
    max_level = pywt.dwt_max_level(len(data), 'coif2')
    level = min(2, max_level)  # 最多使用2层

    try:
        coeffs = pywt.wavedec(data, 'coif2', level=level)
        if level >= 2:
            return coeffs[0], coeffs[1], coeffs[2]  # cA2, cD2, cD1
        elif level == 1:
            return coeffs[0], np.array([]), np.array([])  # cA1, empty, empty
        else:
            return data, np.array([]), np.array([])  # 数据太短，返回原始数据
    except Exception as e:
        print(f"小波分解失败: {e}，使用原始数据")
        return data, np.array([]), np.array([])


def extract_trend_period_random(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取时间序列中的趋势、周期和随机成分"""
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([])

    # 确保是一维数组
    if len(data.shape) > 1:
        data = data.flatten()

    # 移动平均法提取趋势成分
    window_size = min(3, len(data))
    if window_size == 1:
        trend = data.copy()
    else:
        trend = np.convolve(data, np.ones(window_size) / window_size, mode='same')

    # 傅里叶变换提取周期成分
    n = len(data)
    if n > 0:
        fft_data = np.fft.fft(data)
        fft_data[np.abs(fft_data) < 1e-10] = 0  # 去除小系数
        period = np.fft.ifft(fft_data).real
    else:
        period = np.array([])

    # 剩余部分为随机成分
    random = data - trend - period
    return trend, period, random


def extract_features(X: np.ndarray) -> np.ndarray:
    """提取样本的多个特征"""
    features_list = []

    for i in range(X.shape[0]):
        sample = X[i, :]

        # 小波分解
        cA2, cD2, cD1 = wavelet_decomposition(sample)

        # 趋势、周期和随机成分提取
        trend, period, random = extract_trend_period_random(cA2)

        # 筛选出非空特征
        valid_features = []
        for arr in [cD2, cD1, trend, period, random]:
            if len(arr) > 0:
                valid_features.append(arr)

        # 如果没有有效特征，使用原始数据的统计特征
        if not valid_features:
            mean_val = np.mean(sample)
            std_val = np.std(sample)
            valid_features = [np.array([mean_val, std_val, 0.5, 1.0])]  # 添加固定特征表示

        # 合并特征
        combined_features = np.concatenate(valid_features)
        features_list.append(combined_features)

    return np.array(features_list)


def ensure_dimension(features: np.ndarray, target_dim: int) -> np.ndarray:
    """确保特征维度一致"""
    if features.size == 0:
        return np.zeros((1, target_dim))

    current_dim = features.shape[1] if len(features.shape) > 1 else 1

    if current_dim < target_dim:
        # 填充维度
        padded = np.zeros((features.shape[0], target_dim))
        padded[:, :current_dim] = features
        return padded

    elif current_dim > target_dim:
        # 截断维度
        print(f"警告: 当前维度({current_dim})超过目标维度({target_dim})，进行截断")
        return features[:, :target_dim]

    return features


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """计算并返回误差指标字典（包含MAPE）"""
    ERMS, EMA, EMS, R2, EMI, MAPE = wucha_zhibiao(actual, predicted)

    return {
        'ERMS': ERMS,
        'EMA': EMA,
        'EMS': EMS,
        'R2': R2,
        'EMI': EMI,
        'MAPE': MAPE  # 新增MAPE指标
    }


def print_metrics(metrics: Dict[str, float], dataset_name: str) -> None:
    """格式化打印误差指标（包含MAPE）"""
    print(f"\n{dataset_name}误差指标:")
    print(f"ERMS(均方根误差):\t{metrics['ERMS']:.4f}")
    print(f"EMA(平均绝对误差):\t{metrics['EMA']:.4f}")
    print(f"EMS(均方误差):\t\t{metrics['EMS']:.4f}")
    print(f"R2(决定系数):\t\t{metrics['R2']:.4f}")
    print(f"EMI(最大绝对误差):\t{metrics['EMI']:.4f}")
    print(f"MAPE(平均绝对百分比误差):\t{metrics['MAPE']:.4f}%")  # 新增MAPE打印


def save_model_expression(column_name: str, expression: str) -> None:
    """将遗传规划生成的最佳表达式保存到单独文本文件，包含操作符说明"""
    save_dir = 'GP_data-yuan1+SZ2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f'GP_{column_name}_最佳表达式.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"列名: {column_name}\n\n")
        f.write("遗传规划生成的最佳表达式:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{expression}\n")
        f.write("=" * 50 + "\n\n")

        f.write("操作符说明:\n")
        f.write("- add: 加法运算，如 add(x, y) = x + y\n")
        f.write("- sub: 减法运算，如 sub(x, y) = x - y\n")
        f.write("- mul: 乘法运算，如 mul(x, y) = x * y\n")
        f.write("- div: 除法运算，如 div(x, y) = x / y\n")
        f.write("- sin: 正弦函数，如 sin(x) = 对x取正弦值\n")
        f.write("- cos: 余弦函数，如 cos(x) = 对x取余弦值\n")
        f.write("- log: 自然对数，如 log(x) = ln(x)（x需>0）\n")
        f.write("- sqrt: 平方根函数，如 sqrt(x) = √x（x需≥0）\n")
        f.write("\n表达式中的数字为模型自动生成的常数项，用于拟合时间序列特征")

    print(f"最佳表达式已保存至: {file_path}")


def save_results(column_name: str, optimal_lag: int, acf_values: np.ndarray,
                 y_train: np.ndarray, y_train_pred: np.ndarray,
                 y_validation: np.ndarray, y_validation_pred: np.ndarray,
                 y_predict: np.ndarray, y_predict_pred: np.ndarray,
                 metrics_train: Dict[str, float], metrics_validation: Dict[str, float],
                 metrics_predict: Dict[str, float], expression: str) -> None:
    """将结果保存到Excel文件（包含MAPE指标）"""
    valid_column_name = ''.join(c for c in column_name if c.isalnum() or c in ['_', '-'])
    save_dir = 'GP_data-yuan1+SZ2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存预测结果（包含误差百分比）
    if len(y_train) > 0:
        pd.DataFrame({
            '实际值': y_train,
            '预测值': y_train_pred,
            '误差': y_train - y_train_pred,
            '误差百分比(%)': ((y_train - y_train_pred) / np.clip(y_train, 1e-10, None)) * 100  # 与MAPE计算逻辑一致
        }).to_excel(os.path.join(save_dir, f'GP_{valid_column_name}_训练集.xlsx'), index=False)

    if len(y_validation) > 0:
        pd.DataFrame({
            '实际值': y_validation,
            '预测值': y_validation_pred,
            '误差': y_validation - y_validation_pred,
            '误差百分比(%)': ((y_validation - y_validation_pred) / np.clip(y_validation, 1e-10, None)) * 100
        }).to_excel(os.path.join(save_dir, f'GP_{valid_column_name}_验证集.xlsx'), index=False)

    if len(y_predict) > 0:
        pd.DataFrame({
            '实际值': y_predict,
            '预测值': y_predict_pred,
            '误差': y_predict - y_predict_pred,
            '误差百分比(%)': ((y_predict - y_predict_pred) / np.clip(y_predict, 1e-10, None)) * 100
        }).to_excel(os.path.join(save_dir, f'GP_{valid_column_name}_预测集.xlsx'), index=False)

    # 保存误差指标（包含MAPE）
    metrics_df = pd.DataFrame({
        '指标': ['ERMS', 'EMA', 'EMS', 'R2', 'EMI', 'MAPE'],  # 新增MAPE行
        '训练集': [metrics_train['ERMS'], metrics_train['EMA'], metrics_train['EMS'],
                   metrics_train['R2'], metrics_train['EMI'], metrics_train['MAPE']],
        '验证集': [metrics_validation['ERMS'], metrics_validation['EMA'], metrics_validation['EMS'],
                   metrics_validation['R2'], metrics_validation['EMI'], metrics_validation['MAPE']],
        '预测集': [metrics_predict['ERMS'], metrics_predict['EMA'], metrics_predict['EMS'],
                   metrics_predict['R2'], metrics_predict['EMI'], metrics_predict['MAPE']]
    })

    metrics_df.to_excel(os.path.join(save_dir, f'GP_{valid_column_name}_误差指标.xlsx'), index=False)

    # 保存最佳时滞信息和自相关系数
    with open(os.path.join(save_dir, f'GP_{valid_column_name}_时滞与表达式.txt'), 'w', encoding='gbk') as f:
        f.write(f"列名: {column_name}\n")
        f.write(f"最佳时滞: {optimal_lag}\n\n")

        # 记录自相关系数的关键值
        if len(acf_values) > 0:
            f.write("自相关系数关键值:\n")
            f.write(f"lag=0: {acf_values[0]:.6f}\n")
            for i in range(1, min(10, len(acf_values))):
                f.write(f"lag={i}: {acf_values[i]:.6f}\n")
            f.write(f"最大自相关系数: lag={np.argmax(acf_values)}, 值={np.max(acf_values):.6f}\n")
            f.write(
                f"显著时滞数量(>30%峰值): {np.sum(acf_values > 0.3 * np.max(acf_values[1:])) if len(acf_values) > 1 else 0}\n")

    # 保存自相关系数到Excel文件
    if len(acf_values) > 1:  # 确保有足够的数据计算峰值
        peak_lag = np.argmax(acf_values)
        peak_value = acf_values[peak_lag]

        acf_df = pd.DataFrame({
            '时滞(lag)': np.arange(len(acf_values)),
            '自相关系数(ACF)': acf_values,
            '是否显著(>30%峰值)': acf_values > 0.3 * peak_value
        })

        # 添加峰值标记行
        peak_row = pd.DataFrame({
            '时滞(lag)': [f'峰值位置'],
            '自相关系数(ACF)': [peak_value],
            '是否显著(>30%峰值)': ['-']
        })
        acf_df = pd.concat([acf_df.iloc[:peak_lag], peak_row, acf_df.iloc[peak_lag + 1:]], ignore_index=True)
    elif len(acf_values) == 1:  # 处理只有一个数据点的情况
        acf_df = pd.DataFrame({
            '时滞(lag)': [0],
            '自相关系数(ACF)': acf_values,
            '是否显著(>30%峰值)': [False]
        })
    else:  # 没有自相关系数数据
        acf_df = pd.DataFrame(columns=['时滞(lag)', '自相关系数(ACF)', '是否显著(>30%峰值)'])

    acf_df.to_excel(os.path.join(save_dir, f'GP_{valid_column_name}_自相关系数.xlsx'), index=False)
    print(f"自相关系数已保存: GP_{valid_column_name}_自相关系数.xlsx")

    # 保存自相关系数的统计摘要
    if len(acf_values) > 0:
        acf_stats = {
            '最大时滞': len(acf_values),
            '峰值时滞': np.argmax(acf_values) if len(acf_values) > 0 else 0,
            '峰值自相关系数': np.max(acf_values) if len(acf_values) > 0 else 0,
            '显著时滞数量(>30%峰值)': np.sum(acf_values > 0.3 * np.max(acf_values[1:])) if len(acf_values) > 1 else 0,
            '有效时滞范围': f"0-{np.argmax(acf_values > 0.3 * np.max(acf_values[1:])) if len(acf_values) > 1 else 0}"
        }

        acf_stats_df = pd.DataFrame(acf_stats, index=['值'])
        acf_stats_df.to_excel(os.path.join(save_dir, f'GP_{valid_column_name}_自相关系数统计.xlsx'), index=False)
        print(f"自相关系数统计摘要已保存: GP_{valid_column_name}_自相关系数统计.xlsx")

    # 保存最佳表达式到单独文档
    save_model_expression(valid_column_name, expression)


def process_column(data_column: np.ndarray, column_name: str, special_columns: List[str] = []) -> None:
    """处理数据集中的某一列数据并进行预测分析（改为单次时滞构建逻辑）"""
    print(f"\n{'=' * 40}")
    print(f"开始处理列: {column_name}")
    print(f"原始数据长度: {len(data_column)}")

    # 处理特殊列逻辑
    is_special = column_name in special_columns
    if is_special:
        print(f"注意: {column_name} 是特殊列，使用特殊处理逻辑")

    # 参数设置
    history = 4  # 历史数据窗口大小（时滞长度）
    predate = 1  # 预测步长
    max_lag = 20  # 最大时滞范围（未使用，保持原参数）
    optimal_lag = history  # 直接使用固定时滞
    acf_values = np.array([])  # 不再计算自相关系数

    # --------------------------
    # 单次时滞构建（核心修改：直接从原始数据构建特征-目标对）
    # --------------------------
    nums = len(data_column)
    # 总样本数 = 原始数据长度 - 时滞长度 - 预测步长 + 1（闭区间补偿）
    total_samples = nums - history - predate + 1
    if total_samples <= 0:
        print(f"添加时滞特征后数据为空，无法继续处理")
        return

    # 确保有足够的数据进行有效训练
    if total_samples < 3:
        print(f"数据量太少({total_samples}条)，无法进行有效训练")
        return

    # 直接构建特征(X)和目标(y)：一步到位，无二次时滞
    X = []
    y = []
    for i in range(total_samples):
        # 特征：前4个历史数据（时滞窗口）
        features = data_column[i:i + history].flatten()
        # 目标：第i+4+1-1 = i+4个数据（预测步长=1）
        target = data_column[i + history + predate - 1]
        X.append(features)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    print(f"构建时滞特征后：X形状 = {X.shape}, y形状 = {y.shape}")
    print(f"使用最佳时滞: {optimal_lag}")
    print(f"y的统计信息: 最小值={np.min(y):.4f}, 最大值={np.max(y):.4f}, 均值={np.mean(y):.4f}, 标准差={np.std(y):.4f}")

    # --------------------------
    # 数据集划分（保持原逻辑）
    # --------------------------
    train_ratio = 0.6
    validation_ratio = 0.2
    predict_ratio = 0.2

    train_end = int(total_samples * train_ratio)
    validation_end = train_end + int(total_samples * validation_ratio)

    x_train = X[:train_end]
    y_train = y[:train_end]
    x_validation = X[train_end:validation_end]
    y_validation = y[train_end:validation_end]
    x_predict = X[validation_end:]
    y_predict = y[validation_end:]

    print(f"数据集大小: 训练集={len(y_train)}, 验证集={len(y_validation)}, 预测集={len(y_predict)}")

    # --------------------------
    # 特征提取（保持原逻辑）
    # --------------------------
    print("开始特征提取...")
    x_train_wavelet = extract_features(x_train)
    x_validation_wavelet = extract_features(x_validation)
    x_predict_wavelet = extract_features(x_predict)

    # 确保特征维度一致
    feature_dim = x_train_wavelet.shape[1] if x_train_wavelet.size > 0 else 1
    x_validation_wavelet = ensure_dimension(x_validation_wavelet, feature_dim)
    x_predict_wavelet = ensure_dimension(x_predict_wavelet, feature_dim)

    print(
        f"特征维度: 训练集={x_train_wavelet.shape}, 验证集={x_validation_wavelet.shape}, 预测集={x_predict_wavelet.shape}")

    # --------------------------
    # 添加噪声（保持原逻辑）
    # --------------------------
    noise_level = 0.2 * np.std(y_train) if np.std(y_train) > 0 else 0.1
    print(f"添加噪声: 水平={noise_level:.4f}")

    x_train_noisy = x_train_wavelet + np.random.normal(0, noise_level, x_train_wavelet.shape)
    y_train_noisy = y_train + np.random.normal(0, noise_level, y_train.shape)

    # --------------------------
    # 缺失值处理（保持原逻辑）
    # --------------------------
    imputer = SimpleImputer(strategy='mean')
    x_train_noisy = imputer.fit_transform(x_train_noisy)
    x_validation_wavelet = imputer.transform(x_validation_wavelet)
    x_predict_wavelet = imputer.transform(x_predict_wavelet)

    # --------------------------
    # 遗传编程模型初始化（保持原逻辑）
    # --------------------------
    print("初始化遗传编程模型...")
    model = SymbolicRegressor(
        population_size=3000,  # 种群大小
        generations=50,  # 迭代次数
        tournament_size=50,  # 锦标赛选择规模
        stopping_criteria=0.001,
        const_range=(-10., 10.),  # 常数范围
        init_depth=(2, 10),  # 初始树深度
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'),  # 函数集
        metric='mean absolute error',
        parsimony_coefficient=0.002,  # 简约系数，控制模型复杂度
        p_crossover=0.8,
        p_subtree_mutation=0.05,
        p_hoist_mutation=0.05,
        p_point_mutation=0.05,
        p_point_replace=0.1,
        max_samples=0.9,
        verbose=1,
        random_state=42,
        init_method='half and half',  # 初始化方法
        warm_start=True,  # 热启动
        n_jobs=-1  # 使用所有CPU核心
    )

    # 特殊列使用不同的模型参数（保持原逻辑）
    if is_special:
        print(f"为特殊列 {column_name} 使用增强模型参数")
        model.population_size = 5000
        model.generations = 80
        model.function_set = ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt', 'exp', 'tan')
        model.const_range = (-20., 20.)  # 特殊列使用更宽的常数范围

    # --------------------------
    # 模型训练与预测（保持原逻辑）
    # --------------------------
    print("开始训练遗传编程模型...")
    model.fit(x_train_noisy, y_train_noisy)
    best_expression = str(model._program)
    print(f"最终表达式: {best_expression}")

    y_train_pred = model.predict(x_train_noisy)
    y_validation_pred = model.predict(x_validation_wavelet)
    y_predict_pred = model.predict(x_predict_wavelet)

    # --------------------------
    # 误差计算与结果保存（保持原逻辑）
    # --------------------------
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_validation = calculate_metrics(y_validation, y_validation_pred)
    metrics_predict = calculate_metrics(y_predict, y_predict_pred)

    print_metrics(metrics_train, "训练集")
    print_metrics(metrics_validation, "验证集")
    print_metrics(metrics_predict, "预测集")

    save_results(column_name, optimal_lag, acf_values, y_train, y_train_pred,
                 y_validation, y_validation_pred, y_predict, y_predict_pred,
                 metrics_train, metrics_validation, metrics_predict, best_expression)

    # 绘制并保存系数分布直方图（保持原逻辑）
    if hasattr(model._program, 'constants') and len(model._program.constants) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(model._program.constants, bins=20, alpha=0.7, color='skyblue')
        plt.title(f'{column_name} - 遗传规划生成表达式中的系数分布')
        plt.xlabel('系数值')
        plt.ylabel('频率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join('GP_data-yuan1+SZ2', f'GP_{column_name}_系数分布.png'))
        plt.close()
        print(f"系数分布直方图已保存: GP_{column_name}_系数分布.png")

    print(f"列 {column_name} 处理完成")
    print(f"{'=' * 40}")


def main():
    file = r"E:\读研_\论文与专利\遗传规划信号分解英文小论文\数据绘图计算等\使用数据.xlsx"
    special_columns = ["估计流域有效水头", "其他需要特殊处理的列"]  # 指定需要特殊处理的列

    # 读取数据（保持原逻辑）
    try:
        data = pd.read_excel(file, sheet_name="Sheet2")
        print(f"成功读取数据，共 {len(data)} 行，{len(data.columns)} 列")
        print(f"列名: {list(data.columns)}")
    except Exception as e:
        print(f"数据读取失败: {e}")
        return

    # 处理每一列数据（保持原逻辑）
    columns = data.columns[1:]  # 跳过第一列(可能是日期列)
    for column in columns:
        # 检查该列是否在特殊列列表中
        is_special = column in special_columns

        # 检查该列是否有足够的非NaN值
        column_data = data[column].values
        valid_data = column_data[~np.isnan(column_data)]

        if len(valid_data) < 10:
            print(f"列 {column} 有效数据太少，{len(valid_data)} 条，跳过")
            continue

        # 对该列进行预测分析
        process_column(column_data, column, special_columns)

    print("\n所有列处理完成")
    print(f"结果已保存到: {os.path.abspath('GP_data-yuan1+SZ2')}")


if __name__ == "__main__":
    main()