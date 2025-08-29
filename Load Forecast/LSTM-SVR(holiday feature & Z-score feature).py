# -- 必需的库 --
import os
import time
import json
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prettytable import PrettyTable
import warnings
import holidays  # <-- 库变更: 使用 holidays 库来支持更复杂的分类

# -- Scikit-learn 库 --
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV

# -- PyTorch 库 --
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -- 优化库 --
import optuna

warnings.filterwarnings("ignore")

# =============================================================================
# PyTorch 设备设置
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 设备准备就绪 ---")
print(f"当前使用的设备: {device.upper()}")
print("-" * 25)

start_time = time.time()

# =============================================================================
# === 1. 数据加载与特征工程 (*** 代码核心修改区域 ***) ===
# =============================================================================
dataset_raw = pd.read_csv("Tokyo Area Load Dataset.csv")
time_col = pd.to_datetime(dataset_raw.iloc[:, 0])
dataset_raw['datetime'] = time_col  # 将datetime列添加到原始数据中，方便后续处理

# 【核心修正 1】: 明确定义列名，并强制重排顺序
print("--> 正在重排数据列，确保'load'在第一列...")
original_columns = dataset_raw.columns.tolist()
load_col_name = original_columns[13]  # 'load'
weather_col_names = original_columns[1:13]  # 12个天气特征
# 将 'load' 列放在最前面
dataset = dataset_raw[[load_col_name] + weather_col_names].copy()
dataset['datetime'] = time_col  # 再次添加datetime，确保后续特征工程使用

# --- 【*** 新代码 ***】---
# --- 1.1 精致化节假日特征生成 (根据您的要求) ---
print("--> 正在生成精致化节假日特征...")
# 初始化日本的节假日对象
jp_holidays = holidays.JP()


def get_holiday_feature(timestamp):
    """根据日期返回节假日类别ID"""
    # 0: 平日, 1: 正月休み, 5: ゴールデンウィーク, 13: その他の祝日
    # 从 timestamp 中提取 date 对象进行比较
    date = timestamp.date()

    # 定义大型连休的范围
    if (date.month == 12 and date.day >= 29) or (date.month == 1 and date.day <= 5):  # 新年
        return 1
    if (date.month == 1 and date.day >= 11) or (date.month == 1 and date.day <= 13):  # 成人之日
        return 2
    if (date.month == 4 and date.day >= 29) or (date.month == 5 and date.day <= 5):  # 黄金周
        return 3
    if (date.month == 7 and date.day >= 19) or (date.month == 7 and date.day <= 21):  # 海之日
        return 4
    if (date.month == 8 and date.day >= 9) or (date.month == 8 and date.day <= 11):  # 山之日
        return 5
    if (date.month == 8 and date.day >= 13) or (date.month == 8 and date.day <= 16):  # 盂蘭盆節
        return 6
    if (date.month == 9 and date.day >= 13) or (date.month == 9 and date.day <= 15):  # 敬老日
        return 7
    if (date.month == 10 and date.day >= 11) or (date.month == 10 and date.day <= 13):  # 运动之日
        return 8
    if (date.month == 11 and date.day >= 1) or (date.month == 11 and date.day <= 3):  # 文化日
        return 9
    if date in jp_holidays:
        return 10  # 其他祝日
    return 0


dataset['holiday_feature'] = dataset['datetime'].apply(get_holiday_feature)
print("    精致化节假日特征生成完毕。")

# --- 新增特征2: 气象异常日特征 (Z-score > 2) ---
print("--> 正在创建气象异常日特征...")
# 1. 计算每日天气平均值
daily_weather = dataset_raw.set_index('datetime')[weather_col_names].resample('D').mean().dropna()
# 2. 计算每个天气特征的Z-score
z_scores = daily_weather.apply(lambda x: (x - x.mean()) / x.std())
# 3. 任何一个特征的Z-score绝对值大于2，则标记为异常日
anomaly_mask = (z_scores.abs() > 2.5).any(axis=1)
anomaly_dates = daily_weather[anomaly_mask].index.date
# 4. 将异常日标记映射回原始小时数据集
dataset['is_weather_anomaly'] = dataset['datetime'].dt.date.isin(anomaly_dates).astype(int)
print(f"    基于Z-score>2.5的阈值，检测到 {len(anomaly_dates)} 个气象异常日。")

# --- 周期性时间特征工程 ---
print("--> 正在创建周期性时间特征...")
dataset['hour'] = dataset['datetime'].dt.hour
dataset['day_of_week'] = dataset['datetime'].dt.dayofweek
dataset['month'] = dataset['datetime'].dt.month
dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24.0)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24.0)
dataset['day_of_week_sin'] = np.sin(2 * np.pi * dataset['day_of_week'] / 7.0)
dataset['day_of_week_cos'] = np.cos(2 * np.pi * dataset['day_of_week'] / 7.0)
dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12.0)
dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12.0)

# --- 清理并最终确定特征 ---
# 【*** 变更 ***】: 将 is_holiday 替换为 holiday_feature
cols_to_move = ['holiday_feature', 'is_weather_anomaly']
new_order = [col for col in dataset.columns if col not in cols_to_move]
insert_pos = new_order.index(weather_col_names[-1]) + 1
final_order = new_order[:insert_pos] + cols_to_move + new_order[insert_pos:]
dataset = dataset[final_order]

dataset.drop(['datetime', 'hour', 'day_of_week', 'month'], axis=1, inplace=True)
dataset.fillna(method='ffill', inplace=True)

# --- 最终确认特征数量 ---
values = dataset.values.astype('float32')
or_dim = values.shape[1]
print(f"--> 数据准备完成。最终用于模型的总特征数为: {or_dim}")
if or_dim == 21:
    # 【*** 变更 ***】: 更新了确认成功的提示信息
    print("    (确认成功: 1负荷 + 12天气 + 1精致化假日特征 + 1气象异常 + 6时间 = 21个)")
else:
    print(f"    (警告: 特征数量为 {or_dim}，不是预期的21个！请检查代码！)")
    exit()


# =============================================================================
# === 2. 窗口化处理函数 (Data Collation) ===
# =============================================================================

def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples):
    """一个清晰、明确的窗口化处理函数"""
    res = np.zeros((num_samples, n_in * or_dim + n_out))
    for i in range(num_samples):
        input_window = data[scroll_window * i: n_in + scroll_window * i, 0:or_dim]
        input_flat = input_window.reshape(1, n_in * or_dim)
        target_value = data[n_in + scroll_window * i: n_in + scroll_window * i + n_out, 0]
        res[i, :] = np.hstack((input_flat, target_value.reshape(1, n_out)))
    return res


# =============================================================================
# === 3. 数据准备与分割 ===
# =============================================================================
n_in = 9
n_out = 1
scroll_window = 1
num_samples = len(values) - n_in - n_out + 1

res = data_collation(values, n_in, n_out, or_dim, scroll_window, num_samples)
values_windowed = np.array(res)

n_train_number = int(num_samples * 0.7)
Xtrain = values_windowed[:n_train_number, :n_in * or_dim]
Ytrain = values_windowed[:n_train_number, n_in * or_dim:]
Xtest = values_windowed[n_train_number:, :n_in * or_dim]
Ytest = values_windowed[n_train_number:, n_in * or_dim:]

m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)

vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
vp_test = vp_test.reshape((vp_test.shape[0], n_in, or_dim))
print("数据预处理完成。")


# =============================================================================
# === 4. LSTM 模型定义与训练 ===
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dense_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim2, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x);
        out, _ = self.lstm2(out)
        out = out[:, -1, :];
        out = self.fc1(out)
        out = self.relu(out);
        out = self.fc2(out)
        out = self.relu(out);
        out = self.fc3(out)
        return out


# =============================================================================
# === Optuna 目标函数 (已移除早停) ===
# =============================================================================
def objective(trial, X_train_data, y_train_data):
    # 定义超参数搜索空间
    hidden_dim1 = trial.suggest_int('hidden_dim1', 100, 300)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 50, 200)
    dense_dim = trial.suggest_int('dense_dim', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # 数据准备
    vp_train_sub, vp_val, vt_train_sub, vt_val = train_test_split(X_train_data, y_train_data, test_size=0.3, random_state=42)
    X_train_sub_t = torch.from_numpy(vp_train_sub).float().to(device)
    y_train_sub_t = torch.from_numpy(vt_train_sub).float().to(device)
    X_val_t = torch.from_numpy(vp_val).float().to(device)
    y_val_t = torch.from_numpy(vt_val).float().to(device)

    # 模型、损失函数、优化器
    model = LSTMModel(or_dim, hidden_dim1, hidden_dim2, dense_dim, n_out).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=TensorDataset(X_train_sub_t, y_train_sub_t), batch_size=batch_size, shuffle=True)

    # --- 【修改】: 移除了早停机制的训练循环 ---
    for epoch in range(100):  # 循环将固定运行100次
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 在所有epoch结束后，计算最终的验证损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        final_val_loss = criterion(val_outputs, y_val_t).item()

    return final_val_loss


print("\n--- 开始阶段 1: 优化 PyTorch LSTM ---")
lstm_params_file = 'best_lstm_params_2.json'
if os.path.exists(lstm_params_file):
    print("检测到已存在的LSTM参数文件，直接加载...")
    with open(lstm_params_file, 'r') as f:
        best_lstm_params = json.load(f)
else:
    print("未找到LSTM参数文件，开始Optuna优化...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, vp_train, vt_train), n_trials=30)
    best_lstm_params = study.best_params
    with open(lstm_params_file, 'w') as f:
        json.dump(best_lstm_params, f, indent=4)
print(f"LSTM 最佳参数: {best_lstm_params}")

print("\n正在使用最佳参数训练最终的LSTM模型...")
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(vp_train, vt_train, test_size=0.3, random_state=42)
X_train_tensor = torch.from_numpy(X_train_final).float().to(device)
y_train_tensor = torch.from_numpy(y_train_final).float().to(device)
X_val_tensor = torch.from_numpy(X_val_final).float().to(device)
y_val_tensor = torch.from_numpy(y_val_final).float().to(device)

learning_rate = best_lstm_params.pop('learning_rate')
batch_size = best_lstm_params.pop('batch_size')

final_model = LSTMModel(input_dim=or_dim, **best_lstm_params, output_dim=n_out).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

best_model_path = "best_lstm_model_2.pth" # 模型路径可以保持不变或更改

# --- 【修改】: 移除了早停机制的最终训练循环 ---
for epoch in range(150): # 循环将固定运行150次
    epoch_start_time = time.time()
    final_model.train()
    total_train_loss = 0
    num_train_batches = 0

    for inputs, labels in train_loader:
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        num_train_batches += 1

    # 每个epoch结束后评估模型
    final_model.eval()
    with torch.no_grad():
        val_outputs = final_model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    avg_train_loss = total_train_loss / num_train_batches

    print(f"Epoch {epoch + 1:03d}/150 | "
          f"Train Loss: {avg_train_loss:.6e} | "
          f"Val Loss: {val_loss:.6e} | "
          f"Time: {epoch_duration:.2f}s")
    # --- 【修改】: 删除了所有与 val_loss 比较、保存最佳模型和提前中断的逻辑 ---

# --- 【修改】: 在所有epoch完成后，保存最终的模型 ---
torch.save(final_model.state_dict(), best_model_path)
print(f"--- 完成全部 150 个 epoch 的训练，最终模型已保存到 {best_model_path} ---")

# --- 【修改】: 由于我们已在循环后保存了最终模型，不再需要加载所谓的“最佳”模型 ---
# final_model.load_state_dict(torch.load(best_model_path)) # 此行不再需要

print("最终 LSTM 模型训练完成。")


print("\n正在使用最终模型为SVR生成特征...")
with torch.no_grad():
    lstm_train_features = final_model(torch.from_numpy(vp_train).float().to(device)).cpu().numpy()
    lstm_test_features = final_model(torch.from_numpy(vp_test).float().to(device)).cpu().numpy()

# =============================================================================
# === 5. SVR 模型优化与训练 ===
# =============================================================================
print("\n--- 开始阶段 2: 优化 SVR ---")
svr_params_file = 'best_svr_params_2.json'
if os.path.exists(svr_params_file):
    print("检测到已存在的SVR参数文件，直接加载...")
    with open(svr_params_file, 'r') as f:
        best_svr_params = json.load(f)
    best_svr_model = SVR(**best_svr_params)
    best_svr_model.fit(lstm_train_features, vt_train.ravel())
else:
    print("未找到SVR参数文件，开始GridSearchCV...")
    param_grid = {'C': [10, 20, 30, 40, 50, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'epsilon': [1e-5, 1e-4, 1e-3, 1e-2]}
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(lstm_train_features, vt_train.ravel())
    print("\n--- SVR GridSearchCV 详细结果 ---")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by='rank_test_score')
    print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
    print("-" * 40)
    best_svr_params = grid_search.best_params_
    best_svr_model = grid_search.best_estimator_
    with open(svr_params_file, 'w') as f:
        json.dump(best_svr_params, f, indent=4)

print(f"SVR 最佳参数: {best_svr_params}")

# =============================================================================
# === 6. 预测与模型评估 ===
# =============================================================================
print("\n--- 开始阶段 3: 评估最终模型 ---")

predicted_train_normalized = best_svr_model.predict(lstm_train_features)
predicted_train = m_out.inverse_transform(predicted_train_normalized.reshape(-1, 1))
predicted_test_normalized = best_svr_model.predict(lstm_test_features)
predicted_data = m_out.inverse_transform(predicted_test_normalized.reshape(-1, 1))


def evaluate_forecasts(Ytest, predicted_data, Ytrain, predicted_train, n_out):
    test_table = PrettyTable(
        ['テストセットの指標', 'RMSE', 'MAE', 'R2', 'Max Error', 'Mean Error', 'Error Var', '最大予測誤差率(%)',
         '最小予測誤差率(%)', '平均予測誤差率(%)', '標準偏差'])
    train_table = PrettyTable(
        ['トレーニングセットの指標', 'RMSE', 'MAE', 'R2', 'Max Error', 'Mean Error', 'Error Var', '最大予測誤差率(%)',
         '最小予測誤差率(%)', '平均予測誤差率(%)', '標準偏差'])

    for i in range(n_out):
        actual_test, predicted_test = Ytest[:, i], predicted_data[:, i]
        actual_train, predicted_train_col = Ytrain[:, i], predicted_train[:, i]
        for d_type in ['test', 'train']:
            if d_type == 'test':
                actual, predicted, table = actual_test, predicted_test, test_table
                label = '予測結果の指標(LSTM-SVR)' if n_out == 1 else f'第{i + 1}步预测结果'
            else:
                actual, predicted, table = actual_train, predicted_train_col, train_table
                label = '予測結果の指標(LSTM-SVR)' if n_out == 1 else f'第{i + 1}步预测结果'
            diff_errors = np.abs(actual - predicted)
            with np.errstate(divide='ignore', invalid='ignore'):
                error_rates = np.abs(diff_errors / actual) * 100.0
                error_rates = error_rates[np.isfinite(error_rates)]
            table.add_row([
                label,
                f"{sqrt(mean_squared_error(actual, predicted)):.4f}",
                f"{mean_absolute_error(actual, predicted):.4f}",
                f"{r2_score(actual, predicted) * 100:.2f}%",
                f"{np.max(diff_errors):.2f}",
                f"{np.mean(diff_errors):.2f}",
                f"{np.var(diff_errors):.2f}",
                f"{np.max(error_rates):.2f}%",
                f"{np.min(error_rates):.2f}%",
                f"{np.mean(error_rates):.2f}%",
                f"{np.std(error_rates):.4f}"
            ])
    return train_table, test_table


train_table_res, test_table_res = evaluate_forecasts(Ytest, predicted_data, Ytrain, predicted_train, n_out)
print("\nトレーニングセットの指標:")
print(train_table_res)
print("\nテストセットの指標:")
print(test_table_res)

# =============================================================================
# === 7. 保存结果与计时 ===
# =============================================================================
print("\n正在保存预测结果到文件...")
#pd.DataFrame(predicted_data, columns=['predicted_load']).to_csv('predicted_lstm_svr_2.csv', index=False)
#pd.DataFrame(Ytest, columns=['actual_load']).to_csv('y_test_actual.csv', index=False)
#test_sample_indices = [n_train_number + j * scroll_window + n_in for j in range(len(Ytest))]
#test_time = time_col.iloc[test_sample_indices]
#pd.DataFrame({'timestamp': test_time}).to_csv('test_timestamps.csv', index=False)
print("预测相关文件已保存完毕！")

end_time = time.time()
print(f"\n全部流程执行完毕，总耗时: {end_time - start_time:.2f} 秒")
