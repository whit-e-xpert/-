import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

print("--- 开始加载绘图所需的数据文件 ---")

try:
    # 1. 加载测试集相关数据
    predicted_data_df = pd.read_csv('predictions_lstm_svr.csv')
    y_test_df = pd.read_csv('y_test_actual.csv')
    test_time_df = pd.read_csv('test_timestamps.csv')

    # 2. 将数据转换为Numpy数组和Pandas Series
    predicted_data = predicted_data_df.values
    y_test = y_test_df.values
    test_time = pd.to_datetime(test_time_df.iloc[:, 0])

    print("数据文件加载成功！\n")

except FileNotFoundError as e:
    print(f"错误: 缺少数据文件 -> {e}")
    print("请确保已运行LSTM-SVR主脚本，并生成了以下文件：")
    print("['predictions_lstm_svr.csv', 'y_test_actual.csv', 'test_timestamps.csv']")
    exit()


# =====================================================================
# --- 结果可视化绘图 ---
# =====================================================================

# --- 1. 截取最后三日的数据用于绘图 ---
days = 3
hours_per_day = 24
last_three_days_samples = days * hours_per_day

# 从测试集的末尾截取数据
predicted_data_last3 = predicted_data[-last_three_days_samples:]
y_test_last3 = y_test[-last_three_days_samples:]
test_time_last3 = test_time[-last_three_days_samples:]


# --- 2. 绘制最后三日预测结果对比图 ---
print("正在显示：1. 最后三日预测与实际对比图...")
plt.figure(figsize=(20, 8), dpi=150) # dpi可以适当调低，因为只在屏幕显示
plt.plot(test_time_last3, predicted_data_last3, 'b--', linewidth=0.8, label='Predicted Load')
plt.plot(test_time_last3, y_test_last3, 'r-', linewidth=1.0, label='Actual Load')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.gcf().autofmt_xdate()
if not test_time_last3.empty:
    start_date_str = test_time_last3.iloc[0].strftime('%Y/%m/%d')
    end_date_str = test_time_last3.iloc[-1].strftime('%Y/%m/%d')
    plt.title(f'LSTM-SVR Model Prediction ({start_date_str} - {end_date_str})', fontsize=15)
else:
    plt.title('LSTM-SVR Model Prediction (Last 3 Days)', fontsize=15)
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()


# ======================== 误差分析绘图 ========================
print("\n--- 开始进行误差分析与绘图 ---")

# 1. 创建一个用于分析的结果DataFrame
results_df = pd.DataFrame({
    'timestamp': test_time,
    'actual': y_test.flatten(),
    'predicted': predicted_data.flatten()
})
results_df['absolute_error'] = np.abs(results_df['actual'] - results_df['predicted'])


# 2. 绘制整个测试集期间的绝对误差时序图
print("正在显示：2. 整个测试集绝对误差时序图...")
plt.figure(figsize=(20, 8), dpi=150)
plt.plot(results_df['timestamp'], results_df['absolute_error'], label='Absolute Error', color='red', linewidth=0.8)
plt.title("Absolute Error Over Entire Test Period (LSTM-SVR)")
plt.xlabel('Time')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.show()


# 3. 绘制最后3天的误差图
print("正在显示：3. 最后三日绝对误差时序图...")
last_3days_errors = results_df['absolute_error'].iloc[-last_three_days_samples:]
plt.figure(figsize=(20, 8), dpi=150)
plt.plot(test_time_last3, last_3days_errors, color='red', linewidth=1, label='Test Error')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.gcf().autofmt_xdate()
if not test_time_last3.empty:
    plt.xlim(test_time_last3.min(), test_time_last3.max())
    start_date_str = test_time_last3.iloc[0].strftime('%Y/%m/%d')
    end_date_str = test_time_last3.iloc[-1].strftime('%Y/%m/%d')
    plt.title(f'LSTM-SVR Prediction Errors ({start_date_str} - {end_date_str})', fontsize=15)
else:
    plt.title('LSTM-SVR Prediction Errors (Last 3 Days)', fontsize=15)
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()


# 4. 绘制预测值 vs 实际值散点图
print("正在显示：4. 预测值 vs 实际值散点图...")
plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(results_df['actual'], results_df['predicted'], alpha=0.5, label='Predictions')
perfect_line = np.linspace(results_df['actual'].min(), results_df['actual'].max(), 100)
plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label='Perfect Prediction (y=x)')
plt.title('Predicted vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# --- 分组误差分析绘图 ---
results_df['hour'] = results_df['timestamp'].dt.hour
results_df['day_of_week'] = results_df['timestamp'].dt.day_name()
results_df['month'] = results_df['timestamp'].dt.month

# 5. 按小时分析平均绝对误差
print("正在显示：5. 按小时分析误差图...")
hourly_error = results_df.groupby('hour')['absolute_error'].mean().reset_index()
plt.figure(figsize=(12, 6), dpi=100)
plt.bar(hourly_error['hour'], hourly_error['absolute_error'], color='skyblue')
plt.title('Average Absolute Error by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Absolute Error')
plt.xticks(np.arange(0, 24, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 6. 按星期几分析平均绝对误差
print("正在显示：6. 按星期分析误差图...")
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_error = results_df.groupby('day_of_week')['absolute_error'].mean().reindex(weekday_order).reset_index()
plt.figure(figsize=(12, 6), dpi=100)
plt.bar(daily_error['day_of_week'], daily_error['absolute_error'], color='lightgreen')
plt.title('Average Absolute Error by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Absolute Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 7. 按月份分析平均绝对误差
print("正在显示：7. 按月份分析误差图...")
monthly_error = results_df.groupby('month')['absolute_error'].mean().reset_index()
plt.figure(figsize=(12, 6), dpi=100)
plt.bar(monthly_error['month'], monthly_error['absolute_error'], color='mediumpurple')
plt.title('Average Absolute Error by Month')
plt.xlabel('Month')
plt.ylabel('Average Absolute Error')
plt.xticks(np.arange(1, 13, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("\n所有分析图表已显示完毕。")
