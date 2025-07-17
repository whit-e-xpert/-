import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import matplotlib.dates as mdates

# --- 基本设置 ---
warnings.filterwarnings('ignore')
# 正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据加载与预处理 ---
df_full = pd.read_csv("Tokyo Area Load Dataset2.csv",
                 usecols=['timestamp', 'load', 'temperature TK', 'rainfall TK', 'humidity TK', 'wind speed TK',
                          'temperature YH', 'rainfall YH', 'humidity YH', 'wind speed YH',
                          'temperature QB', 'rainfall QB', 'humidity QB', 'wind speed QB'],
                 parse_dates=['timestamp'],
                 index_col='timestamp')

df_full.fillna(method='ffill', inplace=True)

# =====================================================================
# --- 【修正点】: 截断数据集以确保与LSTM-SVR的数据量完全一致 ---
# =====================================================================
num_samples = 35055
# 使用 iloc[-num_samples:] 来选取最后 N 行数据
df = df_full.iloc[-num_samples:].copy()
print(f"为了与LSTM-SVR模型公平对比，数据集已被截断为后 {len(df)} 个样本。")
# =====================================================================

scaler = StandardScaler()
features = ['temperature TK', 'rainfall TK', 'humidity TK', 'wind speed TK',
            'temperature YH', 'rainfall YH', 'humidity YH', 'wind speed YH',
            'temperature QB', 'rainfall QB', 'humidity QB', 'wind speed QB']
df[features] = scaler.fit_transform(df[features])

# --- 2. 划分训练集与测试集 ---
split_point = df.index[int(len(df) * 0.7)]
train_data = df.loc[df.index < split_point]
test_data = df.loc[df.index >= split_point]

X_train = train_data[features].values
y_train = train_data['load'].values
X_test = test_data[features].values
y_test = test_data['load'].values

# =====================================================================
# --- 【新增代码】: 在脚本开头打印训练集和测试集的时间范围 ---
# =====================================================================
train_start_time = train_data.index[0]
train_end_time = train_data.index[-1]
test_start_time = test_data.index[0]
test_end_time = test_data.index[-1]

print("\n--- 数据集时间范围 ---")
print(f"【训练集】时间范围: 从 {train_start_time} 到 {train_end_time}")
print(f"【测试集】时间范围: 从 {test_start_time} 到 {test_end_time}")
print("-" * 25)
# =====================================================================

# --- 3. 定义并训练ARIMAX模型 ---
best_order = (29, 1, 24)
print(f"使用预先确定的最佳参数: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}")

print("正在使用最优参数训练ARIMAX模型...")
final_model = sm.tsa.ARIMA(endog=y_train, exog=X_train, order=best_order).fit()
print("模型训练完成。")

# --- 4. 获取预测结果 ---
# 4.1 获取对训练集的预测结果 (fittedvalues)
predicted_train = final_model.fittedvalues

# 4.2 对测试集进行分批滚动预测
print("正在进行滚动预测...")
predictions_test = []
batch_size = 1
num_batches = int(np.ceil(len(y_test) / batch_size))
updatable_model = final_model

for i in range(num_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(y_test))
    if start_index >= len(y_test):
        break

    current_exog = X_test[start_index:end_index]
    batch_predictions = updatable_model.forecast(steps=len(current_exog), exog=current_exog)
    predictions_test.extend(batch_predictions)

    if end_index < len(y_test):
        update_endog = y_test[start_index:end_index]
        update_exog = X_test[start_index:end_index]
        updatable_model = updatable_model.append(endog=update_endog, exog=update_exog, refit=False)

    print(f"  完成第 {i + 1}/{num_batches} 批次的预测与模型更新...")

predictions_test = np.array(predictions_test[:len(y_test)])
print("预测完成。")

# =====================================================================
# --- 5. 模型评估  ---
# =====================================================================
print("\n--- ARIMAX 模型性能评估 ---")

# --- 处理训练集 ---
# 【修正点】: 由于差分(d=1)，预测值和真实值都从第二个点开始对齐，以确保长度一致
y_train_aligned = y_train[1:]
predicted_train_aligned = predicted_train[1:]

rmse_train = np.sqrt(mean_squared_error(y_train_aligned, predicted_train_aligned))
mae_train = mean_absolute_error(y_train_aligned, predicted_train_aligned)
r2_train = r2_score(y_train_aligned, predicted_train_aligned)

epsilon = 1e-10
error_rates_train = np.abs((y_train_aligned - predicted_train_aligned) / (y_train_aligned + epsilon)) * 100.0
max_err_rate_train = np.max(error_rates_train)
min_err_rate_train = np.min(error_rates_train)
avg_err_rate_train = np.mean(error_rates_train)
std_err_rate_train = np.std(error_rates_train)

# 打印训练集结果
print("\n【训练集】性能指标:")
print(f"  均方根误差 (RMSE): {rmse_train:.2f}")
print(f"  平均绝对误差 (MAE): {mae_train:.2f}")
print(f"  决定系数 (R²): {r2_train:.4f}")
print(f"  最大预测误差率: {max_err_rate_train:.2f}%")
print(f"  最小预测误差率: {min_err_rate_train:.2f}%")
print(f"  平均预测误差率: {avg_err_rate_train:.2f}%")
print(f"  误差率标准差: {std_err_rate_train:.4f}")

# --- 处理测试集 ---
rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
mae_test = mean_absolute_error(y_test, predictions_test)
r2_test = r2_score(y_test, predictions_test)

error_rates_test = np.abs((y_test - predictions_test) / (y_test + epsilon)) * 100.0
max_err_rate_test = np.max(error_rates_test)
min_err_rate_test = np.min(error_rates_test)
avg_err_rate_test = np.mean(error_rates_test)
std_err_rate_test = np.std(error_rates_test)

# 打印测试集结果
print("\n【测试集】性能指标:")
print(f"  均方根误差 (RMSE): {rmse_test:.2f}")
print(f"  平均绝对误差 (MAE): {mae_test:.2f}")
print(f"  决定系数 (R²): {r2_test:.4f}")
print(f"  最大预测误差率: {max_err_rate_test:.2f}%")
print(f"  最小预测误差率: {min_err_rate_test:.2f}%")
print(f"  平均预测误差率: {avg_err_rate_test:.2f}%")
print(f"  误差率标准差: {std_err_rate_test:.4f}")

# --- 6. 结果可视化 ---
days_to_plot = 3
samples_to_plot = days_to_plot * 24
predictions_last3 = predictions_test[-samples_to_plot:]
y_test_last3 = y_test[-samples_to_plot:]
test_index_last3 = test_data.index[-samples_to_plot:]

plt.figure(figsize=(20, 8), dpi=300)
plt.plot(test_index_last3, predictions_last3, 'b--', linewidth=0.5, label='Predicted Load')
plt.plot(test_index_last3, y_test_last3, 'r-', linewidth=0.8, label='Actual Load')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.gcf().autofmt_xdate()

# 【核心修正】: 创建一个动态的、能反映真实日期的标题
start_date_str = test_index_last3[0].strftime('%Y/%m/%d')
end_date_str = test_index_last3[-1].strftime('%Y/%m/%d')
plt.title(f'ARIMAX Model Prediction ({start_date_str} - {end_date_str})', fontsize=15)

plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

# =====================================================================
# --- 7. 【新增代码】保存预测结果、真实值和时间戳 ---
# =====================================================================
print("\n正在保存结果到文件...")

# 7.1 保存ARIMAX的预测结果
#pd.DataFrame(predictions_test, columns=['predicted_load']).to_csv('predictions_arimax.csv', index=False)
#print("  - ARIMAX 预测结果已保存到 'predictions_arimax.csv'")

# 7.2 保存测试集的真实值
#pd.DataFrame(y_test, columns=['actual_load']).to_csv('y_test_actual.csv', index=False)
#print("  - 测试集真实值已保存到 'y_test_actual.csv'")

# 7.3 保存测试集的时间戳
pd.DataFrame(test_data.index, columns=['timestamp']).to_csv('test_timestamps.csv', index=False)
print("  - 测试集时间戳已保存到 'test_timestamps.csv'")

print("\n所有文件保存完毕！")
