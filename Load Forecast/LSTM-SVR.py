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
# =============================================================================
# === 导入PSO库 ===
# =============================================================================
import pyswarms as ps
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# PyTorch 设备设置
# =============================================================================
# 自动检测并选择可用的设备 (GPU或CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 设备准备就绪 ---")
print(f"当前使用的设备: {device.upper()}")
print("-" * 25)

start_time = time.time()

dataset=pd.read_csv("Tokyo Area Load Dataset.csv")

time_col = pd.to_datetime(dataset.iloc[:, 0])

# ==============================================================================
# 提取周期性特征。
# 目的：为了让模型更容易学习到例如时间、星期、月份等周期性规律。
# 方法：使用正弦(sin)和余弦(cos)变换，将周期性数据映射到圆周上的(x, y)坐标。

# 将数据集的第一列（时间戳）转换为pandas的datetime格式，并存为一个新列。
# 这样就可以方便地使用.dt来提取小时、星期等信息。
dataset['datetime'] = pd.to_datetime(dataset.iloc[:, 0])

# --- 步骤1：分解时间戳 ---
# 从'datetime'列中提取具有周期性的“小时”、“星期”和“月份”信息。
dataset['hour'] = dataset['datetime'].dt.hour
dataset['day_of_week'] = dataset['datetime'].dt.dayofweek  # 在pandas中，星期一为0，星期日为6
dataset['month'] = dataset['datetime'].dt.month

# --- 步骤2：应用sin/cos变换 ---
# 将分解后的每个时间元素（如小时），用sin和cos转换成两个新特征。
# 这样模型就能理解23点和0点是相邻的连续关系。

# 转换“小时”信息 (周期最大值为24)
dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24.0)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24.0)

# 转换“星期”信息 (周期最大值为7)
dataset['day_of_week_sin'] = np.sin(2 * np.pi * dataset['day_of_week'] / 7.0)
dataset['day_of_week_cos'] = np.cos(2 * np.pi * dataset['day_of_week'] / 7.0)

# 转换“月份”信息 (周期最大值为12)
dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12.0)
dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12.0)

# --- 步骤3：删除不再需要的特征列 ---
# 创建了sin/cos特征之后，原来的中间列（'datetime', 'hour'等）就不再需要了。
dataset.drop(['datetime', 'hour', 'day_of_week', 'month'], axis=1, inplace=True)

# 同时，也删除原始CSV文件中的时间戳字符串列。
# 这样，数据框中就只剩下电力负荷值和我们新创建的sin/cos特征。
dataset.drop(dataset.columns[0], axis=1, inplace=True)

# ▲▲▲ 代码块结束 ▲▲▲
# ==============================================================================

dataset.fillna(method='ffill', inplace=True)

print(dataset)#显示dataset数据
print(dataset.isnull().sum())

values = dataset.values




# 确保所有数据是浮动的
values = values.astype('float32')
# 将values数组中的数据类型转换为float32。
# 这通常用于确保数据类型的一致性，特别是在准备输入到神经网络模型中时。



def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples):
    res = np.zeros((num_samples,n_in*or_dim+n_out))
    for i in range(0, num_samples):
        h1 = values[scroll_window*i: n_in+scroll_window*i,0:or_dim]
        h2 = h1.reshape( 1, n_in*or_dim)
        h3 = values[n_in+scroll_window*(i) : n_in+scroll_window*(i)+n_out,-7].T
        h4 = h3[np.newaxis, :]
        h5 = np.hstack((h2,h4))
        res[i,:] = h5
    return res




# In[7]:

n_in = 9    # 输入前5行的数据
n_out = 1  # 预测未来1步的数据 注意！SVR不可以多步预测！这是SVR定义决定的！！，这里只能为1
or_dim = values.shape[1]        # 记录特征数据维度
print(f"特征数据维度：{or_dim}")
scroll_window = 1  #如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取

# =====================================================================
# --- 【核心修正点】: 自动计算可生成的样本总数，避免硬编码和越界 ---
# =====================================================================
num_samples = len(values) - n_in - n_out + 1
print(f"数据集总行数: {len(values)}")
print(f"根据窗口大小，最多可生成 {num_samples} 个有效样本。")
# =====================================================================


res = data_collation(values, n_in, n_out, or_dim, scroll_window, num_samples)
# 把数据集分为训练集和测试集
values = np.array(res)
# 将前面处理好的DataFrame（data）转换成numpy数组，方便后续的数据操作。

n_train_number = int(num_samples * 0.7)
Xtrain = values[:n_train_number, :n_in*or_dim]
Ytrain = values[:n_train_number, n_in*or_dim:]
Xtest = values[n_train_number:, :n_in*or_dim]
Ytest = values[n_train_number:,  n_in*or_dim:]

# =====================================================================
# --- 【新增代码】: 在脚本开头打印训练集和测试集的时间范围 ---
# =====================================================================
# 计算与每个样本的目标值相对应的时间戳索引
# 训练集第一个样本的目标值对应的时间戳索引
train_start_index = n_in + scroll_window * 0
# 训练集最后一个样本的目标值对应的时间戳索引
train_end_index = n_in + scroll_window * (n_train_number - 1)
# 测试集第一个样本的目标值对应的时间戳索引
test_start_index = n_in + scroll_window * n_train_number
# 测试集最后一个样本的目标值对应的时间戳索引
test_end_index = n_in + scroll_window * (num_samples - 1)

# 从原始时间列中提取对应的时间戳
train_start_time = time_col.iloc[train_start_index]
train_end_time = time_col.iloc[train_end_index]
test_start_time = time_col.iloc[test_start_index]
test_end_time = time_col.iloc[test_end_index]

print("\n--- 数据集时间范围 ---")
print(f"【训练集】时间范围: 从 {train_start_time} 到 {train_end_time}")
print(f"【测试集】时间范围: 从 {test_start_time} 到 {test_end_time}")
print("-" * 25)
# =====================================================================


# 对训练集和测试集进行归一化
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)  # 注意fit_transform() 和 transform()的区别
vp_test = m_in.transform(Xtest)  # 注意fit_transform() 和 transform()的区别

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)  # 注意fit_transform() 和 transform()的区别
vt_test = m_out.transform(Ytest)  # 注意fit_transform() 和 transform()的区别

# -- 重塑数据以适应LSTM [样本数, 时间步, 特征数] --
vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
vp_test = vp_test.reshape((vp_test.shape[0], n_in, or_dim))
# 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。

print("数据预处理完成。")
# =============================================================================
# 阶段 1: 使用 Optuna 和 PyTorch 优化 LSTM 模型 (加入早停机制)
# =============================================================================
print("\n--- 开始阶段 1: 使用 Optuna 优化 PyTorch LSTM (加入早停机制) ---")


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


# 2. 为 Optuna 定义目标函数 (已加入早停机制)
def objective(trial, X_train, y_train):
    hidden_dim1 = trial.suggest_int('hidden_dim1', 100, 300)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 50, 200)
    dense_dim = trial.suggest_int('dense_dim', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    vp_train_sub, vp_val, vt_train_sub, vt_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    X_train_sub_t = torch.from_numpy(vp_train_sub).float().to(device)
    y_train_sub_t = torch.from_numpy(vt_train_sub).float().to(device)
    X_val_t = torch.from_numpy(vp_val).float().to(device)
    y_val_t = torch.from_numpy(vt_val).float().to(device)

    model = LSTMModel(or_dim, hidden_dim1, hidden_dim2, dense_dim, n_out).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train_sub_t, y_train_sub_t)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # --- 早停机制参数 ---
    patience = 10 # 如果验证损失连续10轮没有改善，就停止
    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = 100

    for epoch in range(max_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Trial #{trial.number}: Early stopping at epoch {epoch + 1}")
            break

    return best_val_loss


# 3. 检查、加载或运行优化
lstm_params_file = 'best_lstm_params_pytorch.json'
if os.path.exists(lstm_params_file):
    print(f"检测到已存在的参数文件'{lstm_params_file}'，直接加载参数。")
    with open(lstm_params_file, 'r') as f:
        best_lstm_params = json.load(f)
    print(f"加载的LSTM参数: {best_lstm_params}")
else:
    print(f"未找到参数文件'{lstm_params_file}'，开始执行贝叶斯优化...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, vp_train, vt_train), n_trials=100) # 共计运行100轮

    best_lstm_params = study.best_params
    print(f"LSTM 优化完成。找到的最佳参数: {best_lstm_params}")
    with open(lstm_params_file, 'w') as f:
        json.dump(best_lstm_params, f, indent=4)
    print(f"最佳参数已保存到 '{lstm_params_file}'。")

# 4. 使用最佳参数训练最终模型 (已加入早停机制)
print("\n正在使用最佳参数训练最终的LSTM模型...")

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    vp_train, vt_train, test_size=0.3, random_state=42
)

X_train_tensor = torch.from_numpy(X_train_final).float().to(device)
y_train_tensor = torch.from_numpy(y_train_final).float().to(device)
X_val_tensor = torch.from_numpy(X_val_final).float().to(device)
y_val_tensor = torch.from_numpy(y_val_final).float().to(device)

learning_rate = best_lstm_params.pop('learning_rate')
batch_size = best_lstm_params.pop('batch_size')

final_model = LSTMModel(input_dim=or_dim, **best_lstm_params, output_dim=n_out).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# --- 早停机制的参数 ---
patience_final = 15 # 如果验证损失连续15轮没有改善，就停止
best_val_loss_final = float('inf')
patience_counter_final = 0
max_epochs_final = 100
best_model_path = "final_best_lstm_model.pth"

for epoch in range(max_epochs_final):
    epoch_start_time = time.time()

    final_model.train()
    for inputs, labels in train_loader:
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_model.eval()
    with torch.no_grad():
        val_outputs = final_model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(
        f'Final Model Training - Epoch [{epoch + 1}/{max_epochs_final}], Loss: {loss.item():.2e}, Val Loss: {val_loss:.2e}, Time: {epoch_duration:.2f}s')

    if val_loss < best_val_loss_final:
        best_val_loss_final = val_loss
        patience_counter_final = 0
        torch.save(final_model.state_dict(), best_model_path)
        print(f"  -> 验证损失改善，保存最佳模型到 '{best_model_path}'")
    else:
        patience_counter_final += 1

    if patience_counter_final >= patience_final:
        print(f"最终训练: 在第 {epoch + 1} 轮早停")
        break

print(f"\n加载验证集上表现最佳的模型 (from '{best_model_path}')...")
final_model.load_state_dict(torch.load(best_model_path))
print("最终 LSTM 模型训练完成。")

# 5. 使用最终模型为SVR生成特征
final_model.eval()
# 使用完整的训练集和测试集来生成特征
full_X_train_tensor = torch.from_numpy(vp_train).float().to(device)
X_test_tensor = torch.from_numpy(vp_test).float().to(device)
with torch.no_grad():
    lstm_train_features = final_model(full_X_train_tensor).cpu().numpy()
    lstm_test_features = final_model(X_test_tensor).cpu().numpy()

# =============================================================================
# 阶段 2: 使用网格搜索 (GridSearchCV) 寻找最佳 SVR 参数
# =============================================================================
print("\n---开始阶段2:优化SVR---")

# 定义保存SVR最佳参数的文件名
svr_params_file = 'best_svr_params_pytorch.json'

# 检查SVR最佳参数文件是否已存在
if os.path.exists(svr_params_file):
    # 如果文件存在，直接加载参数，并用这些参数训练模型
    print(f"检测到已存在的参数文件'{svr_params_file}'，直接加载参数。")
    with open(svr_params_file, 'r') as f:
        best_svr_params = json.load(f)
    print(f"加载的SVR参数: {best_svr_params}")

    # **重要**: 使用加载的参数创建并训练最终的SVR模型
    best_svr_model = SVR(**best_svr_params)
    best_svr_model.fit(lstm_train_features, vt_train.ravel())

else:
    # --- 只有当文件不存在时，才执行以下所有网格搜索步骤 ---

    print(f"未找到参数文件'{svr_params_file}'，开始执行网格搜索...")

    # 1. 定义SVR的参数网格
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 20, 30, 40, 50, 70, 80, 90, 100],
        'gamma': [0.001, 0.01, 0.1,1],
        'epsilon': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    }

    # 2. 初始化SVR模型和网格搜索
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    # 3. 在LSTM生成的特征上执行网格搜索
    grid_search.fit(lstm_train_features, vt_train.ravel())

    # 4. 获取最佳SVR模型和参数
    best_svr_params = grid_search.best_params_
    best_svr_model = grid_search.best_estimator_
    print(f"SVR 优化完成。找到的最佳参数: {best_svr_params}")

    # 5. 将找到的最佳参数保存到文件中
    with open(svr_params_file, 'w') as f:
        json.dump(best_svr_params, f, indent=4)
    print(f"最佳参数已保存到 '{svr_params_file}'。")

# =============================================================================
# 阶段 3: 使用优化后的模型进行预测和评估 (基于您的原始代码)
# =============================================================================
print("\n--- 开始阶段 3: 评估最终模型 ---")

# 对训练集进行评估
predicted_train_normalized = best_svr_model.predict(lstm_train_features)
# ===================【修改点1：处理训练集预测结果】===================
# 准备一个“容器”来进行反归一化，其列数必须与归一化时输入的特征数相同
num_features_output = Ytrain.shape[1] # 获取输出特征的数量，这里应该是1
dummy_array_train = np.zeros((len(predicted_train_normalized), num_features_output))

# 将预测出的、被归一化的值，放入容器的第一列
dummy_array_train[:, 0] = predicted_train_normalized.flatten()

# 使用完整的容器进行反归一化
unscaled_train = m_out.inverse_transform(dummy_array_train)

# 从反归一化后的结果中，只取出第一列，这才是我们最终的预测值
predicted_train = unscaled_train[:, 0].reshape(-1, n_out)
# =================================================================

# 对测试集进行预测
yhat_normalized = best_svr_model.predict(lstm_test_features)
# 准备一个“容器”来进行反归一化
num_features_output = Ytrain.shape[1] # 获取输出特征的数量，这里应该是1
dummy_array_test = np.zeros((len(yhat_normalized), num_features_output))

# 将预测出的、被归一化的值，放入容器的第一列
dummy_array_test[:, 0] = yhat_normalized.flatten()

# 使用完整的容器进行反归一化
unscaled_test = m_out.inverse_transform(dummy_array_test)

# 从反归一化后的结果中，只取出第一列，这才是我们最终的预测值
predicted_data = unscaled_test[:, 0].reshape(-1, n_out)
# ================================================================


def mape(y_true, y_pred):
    # 定义一个计算平均绝对百分比误差（MAPE）的函数。
    record = []
    for index in range(len(y_true)):
        # 遍历实际值和预测值。
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        # 计算单个预测的MAPE。
        record.append(temp_mape)
        # 将MAPE添加到记录列表中。
    return np.mean(record) * 100
    # 返回所有记录的平均值，乘以100得到百分比。

def evaluate_forecasts(Ytest, predicted_data, Ytrain, predicted_train, n_out):
    # 定义一个函数来评估预测的性能。
    rmse_dic_test = []
    mae_dic_test = []
    r2_dic_test = []

    rmse_dic_train = []
    mae_dic_train = []
    r2_dic_train = []

    # 初始化新指标存储列表
    max_error_rate_test_list = []
    min_error_rate_test_list = []
    avg_error_rate_test_list = []
    std_error_rate_test_list = []

    max_error_rate_train_list = []
    min_error_rate_train_list = []
    avg_error_rate_train_list = []
    std_error_rate_train_list = []

    # 初始化存储各个评估指标的字典。
    test_table = PrettyTable(['テストセットの指標', 'RMSE', 'MAE', 'R2',
                             'Max Error', 'Mean Error', 'Error Var',
                             '最大予測誤差率(%)', '最小予測誤差率(%)', '平均予測誤差率(%)', '標準偏差'])
    train_table = PrettyTable(['トレーニングセットの指標', 'RMSE', 'MAE', 'R2',
                              'Max Error', 'Mean Error', 'Error Var',
                              '最大予測誤差率(%)', '最小予測誤差率(%)', '平均予測誤差率(%)', '標準偏差'])
    for i in range(n_out):
        # 遍历每一个预测步长。每一列代表一步预测，现在是在求每步预测的指标
        actual_test = [float(row[i]) for row in Ytest]
        predicted_test = [float(row[i]) for row in predicted_data]
        actual_train = [float(row[i]) for row in Ytrain]
        predicted_train = [float(row[i]) for row in predicted_train]

        # === 追加: 差分誤差・最大誤差・統計量の計算 ===
        diff_errors_test = np.abs(np.array(actual_test) - np.array(predicted_test))  # 差分誤差（絶対値）
        max_error_test = np.max(diff_errors_test)  # 最大誤差
        mean_error_test = np.mean(diff_errors_test)  # 誤差平均
        var_error_test = np.var(diff_errors_test)  # 誤差分散

        # トレーニングセットも同様に追加
        diff_errors_train = np.abs(np.array(actual_train) - np.array(predicted_train))
        max_error_train = np.max(diff_errors_train)
        mean_error_train = np.mean(diff_errors_train)
        var_error_train = np.var(diff_errors_train)



        rmse_test = sqrt(mean_squared_error(actual_test, predicted_test))
        mae_test = mean_absolute_error(actual_test, predicted_test)
        r2_test = r2_score(actual_test, predicted_test)

        rmse_train = sqrt(mean_squared_error(actual_train, predicted_train))
        mae_train = mean_absolute_error(actual_train, predicted_train)
        r2_train = r2_score(actual_train, predicted_train)

        rmse_dic_test.append(rmse_test)
        mae_dic_test.append(mae_test)
        r2_dic_test.append(r2_test)

        rmse_dic_train.append(rmse_train)
        mae_dic_train.append(mae_train)
        r2_dic_train.append(r2_train)

        # === 新增：计算预测误差率指标 ===
        # 计算每个样本的预测误差率（百分比形式）
        error_rates_test = np.abs((np.array(actual_test) - np.array(predicted_test)) / np.array(actual_test)) * 100.0

        # 计算新指标
        max_error_rate_test = np.max(error_rates_test)
        min_error_rate_test = np.min(error_rates_test)
        avg_error_rate_test = np.mean(error_rates_test)
        std_error_rate_test = np.std(error_rates_test)

        # 存储新指标
        max_error_rate_test_list.append(max_error_rate_test)
        min_error_rate_test_list.append(min_error_rate_test)
        std_error_rate_test_list.append(std_error_rate_test)

        # 训练集同样处理
        error_rates_train = np.abs((np.array(actual_train) - np.array(predicted_train)) / np.array(actual_train)) * 100.0
        max_error_rate_train = np.max(error_rates_train)
        min_error_rate_train = np.min(error_rates_train)
        avg_error_rate_train = np.mean(error_rates_train)
        std_error_rate_train = np.std(error_rates_train)

        max_error_rate_train_list.append(max_error_rate_train)
        min_error_rate_train_list.append(min_error_rate_train)
        std_error_rate_train_list.append(std_error_rate_train)

        if n_out == 1:
            strr = '予測結果の指標(LSTM-SVR)'
        else:
            strr = f'第{i + 1}步预测结果'

        test_table.add_row([strr, rmse_test, mae_test,
                            f"{r2_test * 100:.2f}%", f"{max_error_test:.2f}",
                            f"{mean_error_test:.2f}", f"{var_error_test:.2f}",
                            f"{max_error_rate_test:.2f}%", f"{min_error_rate_test:.2f}%",
                            f"{avg_error_rate_test:.2f}%", f"{std_error_rate_test:.4f}"])

        train_table.add_row([strr, rmse_train, mae_train,
                             f"{r2_train * 100:.2f}%", f"{max_error_train:.2f}",
                             f"{mean_error_train:.2f}", f"{var_error_train:.2f}",
                             f"{max_error_rate_train:.2f}%", f"{min_error_rate_train:.2f}%",
                             f"{avg_error_rate_train:.2f}%", f"{std_error_rate_train:.4f}"])

    return (rmse_dic_test, mae_dic_test,  r2_dic_test,  diff_errors_test, max_error_test, mean_error_test,
            var_error_test,max_error_rate_test_list, min_error_rate_test_list, avg_error_rate_test_list, std_error_rate_test_list,\
        rmse_dic_train, mae_dic_train,  r2_dic_train, diff_errors_train, max_error_train, mean_error_train, var_error_train,
            max_error_rate_train_list, min_error_rate_train_list, avg_error_rate_train_list, std_error_rate_train_list,
            test_table, train_table)
    # 返回包含所有评估指标的字典。



(rmse_dic_test, mae_dic_test, r2_dic_test, diff_errors_test, max_error_test, mean_error_test, var_error_test,
 max_error_rate_test_list, min_error_rate_test_list, avg_error_rate_test_list, std_error_rate_test_list,
rmse_dic_train, mae_dic_train, r2_dic_train,diff_errors_train, max_error_train, mean_error_train, var_error_train,
 max_error_rate_train_list, min_error_rate_train_list, avg_error_rate_train_list, std_error_rate_train_list,
test_table, train_table) = evaluate_forecasts(Ytest, predicted_data, Ytrain, predicted_train, n_out)

# In[16]:

print("\nトレーニングセットの指標:")
print(train_table)
print("テストセットの指標:")
print(test_table)

# =====================================================================
# --- 【最终修正版】结果可视化与文件保存 ---
# =====================================================================

# --- 提取与测试集样本完全对应的时间戳 ---
# 这里的逻辑是关键，它确保了时间戳与 Ytest 数组中的每一个点一一对应
test_sample_indices = [n_train_number + j * scroll_window + n_in for j in range(len(Ytest))]
# 从最开始加载的、完整的 time_col 中，根据索引提取出正确的时间戳
test_time = time_col.iloc[test_sample_indices]

# =====================================================================
# --- 保存预测结果、真实值和时间戳 ---
# =====================================================================
print("\n正在保存结果到文件...")

# 1. 保存LSTM-SVR的预测结果
#pd.DataFrame(predicted_data, columns=['predicted_load']).to_csv('predictions_lstm_svr.csv', index=False)
#print("  - LSTM-SVR 预测结果已保存到 'predictions_lstm_svr.csv'")

# 2. 保存测试集的真实值
# 注意：Ytest 和 test_time 已经在您的原始代码中被定义，我们直接使用即可
#pd.DataFrame(Ytest, columns=['actual_load']).to_csv('y_test_actual.csv', index=False)
#print("  - 测试集真实值已保存到 'y_test_actual.csv'")

# 3. 保存测试集的时间戳
#pd.DataFrame({'timestamp': test_time}).to_csv('test_timestamps.csv', index=False)
#print("  - 测试集时间戳已保存到 'test_timestamps.csv'")

#print("\n所有文件保存完毕！")

end_time = time.time()#记录结束时间
print(f"Code running time: {end_time-start_time}S")

