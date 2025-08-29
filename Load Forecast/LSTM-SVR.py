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
# === 1. 数据加载与特征工程 ===
# =============================================================================
dataset_raw = pd.read_csv("Tokyo Area Load Dataset.csv")
time_col = pd.to_datetime(dataset_raw.iloc[:, 0])

# 【核心修正 1】: 明确定义列名，并强制重排顺序
print("--> 正在重排数据列，确保'load'在第一列...")
original_columns = dataset_raw.columns.tolist()
load_col_name = original_columns[13]  # 'load'
weather_col_names = original_columns[1:13]  # 12个天气特征
# 将 'load' 列放在最前面
dataset = dataset_raw[[load_col_name] + weather_col_names].copy()

# --- 在重排后的数据集上进行特征工程 ---
dataset['datetime'] = time_col
dataset['hour'] = dataset['datetime'].dt.hour
dataset['day_of_week'] = dataset['datetime'].dt.dayofweek
dataset['month'] = dataset['datetime'].dt.month
dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24.0)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24.0)
dataset['day_of_week_sin'] = np.sin(2 * np.pi * dataset['day_of_week'] / 7.0)
dataset['day_of_week_cos'] = np.cos(2 * np.pi * dataset['day_of_week'] / 7.0)
dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12.0)
dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12.0)
dataset.drop(['datetime', 'hour', 'day_of_week', 'month'], axis=1, inplace=True)
dataset.fillna(method='ffill', inplace=True)

# --- 最终确认特征数量 ---
values = dataset.values.astype('float32')
or_dim = values.shape[1]
print(f"--> 数据准备完成。最终用于模型的总特征数为: {or_dim}")
if or_dim == 19:
    print("    (确认成功: 1个负荷特征 + 12个天气特征 + 6个时间特征 = 19个)")
else:
    print(f"    (警告: 特征数量为 {or_dim}，不是预期的19个！请检查代码！)")
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


def objective(trial, X_train_data, y_train_data):
    hidden_dim1 = trial.suggest_int('hidden_dim1', 100, 300)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 50, 200)
    dense_dim = trial.suggest_int('dense_dim', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    vp_train_sub, vp_val, vt_train_sub, vt_val = train_test_split(X_train_data, y_train_data, test_size=0.3,
                                                                  random_state=42)
    X_train_sub_t, y_train_sub_t = torch.from_numpy(vp_train_sub).float().to(device), torch.from_numpy(
        vt_train_sub).float().to(device)
    X_val_t, y_val_t = torch.from_numpy(vp_val).float().to(device), torch.from_numpy(vt_val).float().to(device)
    model = LSTMModel(or_dim, hidden_dim1, hidden_dim2, dense_dim, n_out).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=TensorDataset(X_train_sub_t, y_train_sub_t), batch_size=batch_size, shuffle=True)
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs);
            loss = criterion(outputs, labels)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t);
            val_loss = criterion(val_outputs, y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
        if patience_counter >= 10: break
    return best_val_loss


print("\n--- 开始阶段 1: 优化 PyTorch LSTM ---")
lstm_params_file = 'best_lstm_params.json'
if os.path.exists(lstm_params_file):
    print("检测到已存在的LSTM参数文件，直接加载...")
    with open(lstm_params_file, 'r') as f:
        best_lstm_params = json.load(f)
else:
    print("未找到LSTM参数文件，开始Optuna优化...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, vp_train, vt_train), n_trials=30)  # n_trials:贝叶斯优化训练轮数
    best_lstm_params = study.best_params
    with open(lstm_params_file, 'w') as f:
        json.dump(best_lstm_params, f, indent=4)
print(f"LSTM 最佳参数: {best_lstm_params}")

print("\n正在使用最佳参数训练最终的LSTM模型...")
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(vp_train, vt_train, test_size=0.3,
                                                                          random_state=42)
X_train_tensor, y_train_tensor = torch.from_numpy(X_train_final).float().to(device), torch.from_numpy(
    y_train_final).float().to(device)
X_val_tensor, y_val_tensor = torch.from_numpy(X_val_final).float().to(device), torch.from_numpy(y_val_final).float().to(
    device)
learning_rate = best_lstm_params.pop('learning_rate')
batch_size = best_lstm_params.pop('batch_size')
final_model = LSTMModel(input_dim=or_dim, **best_lstm_params, output_dim=n_out).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
patience_final, best_val_loss_final, patience_counter_final = 15, float('inf'), 0
best_model_path = "best_lstm_model.pth"

# 【修改代码】: 详细的最终训练循环
for epoch in range(150):
    epoch_start_time = time.time()  # 【新增代码】: 记录epoch开始时间
    final_model.train()

    # 【新增代码】: 初始化用于计算平均训练损失的变量
    total_train_loss = 0
    num_train_batches = 0

    for inputs, labels in train_loader:
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 【新增代码】: 累加每个batch的loss
        total_train_loss += loss.item()
        num_train_batches += 1

    final_model.eval()
    with torch.no_grad():
        val_outputs = final_model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    epoch_end_time = time.time()  # 【新增代码】: 记录epoch结束时间
    epoch_duration = epoch_end_time - epoch_start_time  # 【新增代码】: 计算epoch耗时
    avg_train_loss = total_train_loss / num_train_batches  # 【新增代码】: 计算平均训练损失

    # 【新增代码】: 打印每个epoch的详细结果
    print(f"Epoch {epoch + 1:03d}/150 | "
          f"Train Loss: {avg_train_loss:.6e} | "
          f"Val Loss: {val_loss:.6e} | "
          f"Time: {epoch_duration:.2f}s")

    if val_loss < best_val_loss_final:
        best_val_loss_final, patience_counter_final = val_loss, 0
        torch.save(final_model.state_dict(), best_model_path)
        # 【新增代码】: 增加提示信息
        print(f"    -> Val loss 从 {best_val_loss_final:.6f} 降低到 {val_loss:.6f}。保存模型。")
    else:
        patience_counter_final += 1
        # 【新增代码】: 增加提示信息
        print(f"    -> Val loss 没有改善。耐心计数: {patience_counter_final}/{patience_final}")

    if patience_counter_final >= patience_final:
        print(f"--- 在第 {epoch + 1} 个 epoch 触发早停 ---")
        break

final_model.load_state_dict(torch.load(best_model_path))
print("最终 LSTM 模型训练完成。")

print("\n正在使用最终模型为SVR生成特征...")
with torch.no_grad():
    lstm_train_features = final_model(torch.from_numpy(vp_train).float().to(device)).cpu().numpy()
    lstm_test_features = final_model(torch.from_numpy(vp_test).float().to(device)).cpu().numpy()

# =============================================================================
# === 5. SVR 模型优化与训练 ===
# =============================================================================
print("\n--- 开始阶段 2: 优化 SVR ---")
svr_params_file = 'best_svr_params.json'
if os.path.exists(svr_params_file):
    print("检测到已存在的SVR参数文件，直接加载...")
    with open(svr_params_file, 'r') as f:
        best_svr_params = json.load(f)
    best_svr_model = SVR(**best_svr_params)
    best_svr_model.fit(lstm_train_features, vt_train.ravel())
else:
    print("未找到SVR参数文件，开始GridSearchCV...")
    param_grid = {'C': [10, 20, 30, 40, 50, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'epsilon': [1e-5, 1e-4, 1e-3, 1e-2]}
    # 【修改代码】: 将 verbose 设置为 2 以获取更详细的输出
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(lstm_train_features, vt_train.ravel())

    # 【新增代码】: 打印GridSearchCV的详细结果
    print("\n--- SVR GridSearchCV 详细结果 ---")
    results_df = pd.DataFrame(grid_search.cv_results_)
    # 筛选出我们最关心的列并按排名排序
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

# --- 对训练集进行预测和反归一化 ---
predicted_train_normalized = best_svr_model.predict(lstm_train_features)
predicted_train = m_out.inverse_transform(predicted_train_normalized.reshape(-1, 1))

# --- 对测试集进行预测和反归一化 ---
predicted_test_normalized = best_svr_model.predict(lstm_test_features)
predicted_data = m_out.inverse_transform(predicted_test_normalized.reshape(-1, 1))


# --- 定义评估函数 ---
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


# --- 执行评估并打印结果 ---
train_table_res, test_table_res = evaluate_forecasts(Ytest, predicted_data, Ytrain, predicted_train, n_out)

print("\nトレーニングセットの指標:")
print(train_table_res)
print("\nテストセットの指標:")
print(test_table_res)

# =============================================================================
# === 7. 保存结果与计时 ===
# =============================================================================
print("\n正在保存预测结果到文件...")
pd.DataFrame(predicted_data, columns=['predicted_load']).to_csv('predictions_lstm_svr.csv', index=False)
pd.DataFrame(Ytest, columns=['actual_load']).to_csv('y_test_actual.csv', index=False)
test_sample_indices = [n_train_number + j * scroll_window + n_in for j in range(len(Ytest))]
test_time = time_col.iloc[test_sample_indices]
pd.DataFrame({'timestamp': test_time}).to_csv('test_timestamps.csv', index=False)
print("预测相关文件已保存完毕！")

end_time = time.time()
print(f"\n全部流程执行完毕，总耗时: {end_time - start_time:.2f} 秒")
