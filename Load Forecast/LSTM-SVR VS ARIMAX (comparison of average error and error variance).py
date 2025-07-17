import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns # 导入seaborn库，用于绘制更美观的统计图表

def run_comprehensive_analysis():
    """
    一个完整的分析脚本，用于加载模型预测数据，进行量化误差分析，
    并生成高质量的对比图表。
    """
    try:
        # 优先使用'Yu Gothic'，如果找不到，则尝试'MS Gothic'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Yu Gothic', 'MS Gothic']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print("警告：未能成功设置日文字体。图表中的日文可能无法正确显示。")
        print("请确认您的系统中已安装 'Yu Gothic' 或 'MS Gothic' 字体。")
    # ==================================================================

    try:
        # --- 1. 加载所有需要的数据 ---
        preds_arimax = pd.read_csv('predictions_arimax.csv')
        preds_lstm_svr = pd.read_csv('predictions_lstm_svr.csv')
        actual_values = pd.read_csv('y_test_actual.csv')
        timestamps = pd.read_csv('test_timestamps.csv', parse_dates=['timestamp'])
        test_features_df = pd.read_csv('test_data.csv', parse_dates=['timestamp'])

    except FileNotFoundError as e:
        print(f"错误：文件未找到 {e.filename}。请确保所有CSV文件都在脚本所在的目录中。")
        return

    # --- 2. 合并成一个用于分析的DataFrame ---
    results_df = pd.DataFrame({
        'timestamp': timestamps['timestamp'],
        'actual': actual_values['actual_load'],
        'predicted_arimax': preds_arimax['predicted_load'],
        'predicted_lstm_svr': preds_lstm_svr['predicted_load']
    })

    results_df['abs_error_arimax'] = np.abs(results_df['actual'] - results_df['predicted_arimax'])
    results_df['abs_error_lstm_svr'] = np.abs(results_df['actual'] - results_df['predicted_lstm_svr'])

    print("--- 数据已成功加载并合并 ---")
    print(results_df.head())
    print("-" * 30)

    # --- 3. 计算并打印量化指标（平均值和方差）---
    arimax_error_mean = results_df['abs_error_arimax'].mean()
    lstm_svr_error_mean = results_df['abs_error_lstm_svr'].mean()
    arimax_error_var = results_df['abs_error_arimax'].var()
    lstm_svr_error_var = results_df['abs_error_lstm_svr'].var()

    print("--- 模型的量化误差指标分析 ---")
    print(f"\nARIMAX 模型:")
    print(f"  - 平均绝对误差 (Mean Absolute Error): {arimax_error_mean:.4f}")
    print(f"  - 绝对误差的方差 (Variance of Absolute Error): {arimax_error_var:.4f}")
    print(f"\nLSTM-SVR 模型:")
    print(f"  - 平均绝对误差 (Mean Absolute Error): {lstm_svr_error_mean:.4f}")
    print(f"  - 绝对误差的方差 (Variance of Absolute Error): {lstm_svr_error_var:.4f}")
    print("\n--- 分析图表正在生成... ---")

    # --- 4. 绘制【误差指标对比条形图】 ---
    fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
    labels = ['ARIMAX', 'LSTM-SVR']
    means = [arimax_error_mean, lstm_svr_error_mean]
    variances = [arimax_error_var, lstm_svr_error_var]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, means, width, label='平均絶対誤差', color='cornflowerblue')
    rects2 = ax1.bar(x + width / 2, variances, width, label='誤差の分散', color='sandybrown')

    ax1.set_ylabel('誤差値')
    ax1.set_title('ARIMAXとLSTM-SVRの誤差の平均値と分散の比較')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    ax1.bar_label(rects1, padding=3, fmt='%.2f')
    ax1.bar_label(rects2, padding=3, fmt='%.2f')
    fig1.tight_layout()

    # ===  5.计算优劣区间的比例，并用饼图可视化 ===

    # 情况1: LSTM-SVR的误差 > ARIMAX的误差 (表现不佳)
    underperform_condition = results_df['abs_error_lstm_svr'] > results_df['abs_error_arimax']
    underperform_count = underperform_condition.sum()

    # 为了计算情况2和3，首先计算出“改善率”
    # 仅在LSTM-SVR表现更好时，其改善率才有意义
    improvement_rate = (results_df['abs_error_arimax'] - results_df['abs_error_lstm_svr']) / results_df[
        'abs_error_arimax']

    # 情况2: LSTM-SVR的性能改善超过10%
    significant_improvement_condition = (~underperform_condition) & (improvement_rate > 0.10)
    significant_improvement_count = significant_improvement_condition.sum()

    # 情况3: 剩下的就是表现更好，但改善率在0%~10%之间
    minor_improvement_count = len(results_df) - underperform_count - significant_improvement_count

    # 计算各个情况占总体的百分比
    total_count = len(results_df)
    underperform_pct = (underperform_count / total_count) * 100
    significant_improvement_pct = (significant_improvement_count / total_count) * 100
    minor_improvement_pct = (minor_improvement_count / total_count) * 100

    print("\n--- 优劣区间比例分析 ---")
    print(f"LSTM-SVR 性能劣于 ARIMAX 的时段占比: {underperform_pct:.2f}%")
    print(f"LSTM-SVR 性能优于 ARIMAX 超过10%的时段 占比: {significant_improvement_pct:.2f}%")
    print(f"LSTM-SVR 性能优于 ARIMAX 0%～10%的时段: {minor_improvement_pct:.2f}%")

    # 绘制饼图
    labels = [
        f'LSTM-SVRの性能がARIMAXより劣る\n({underperform_pct:.1f}%)',
        f'LSTM-SVRがARIMAXより10%以上優れている\n({significant_improvement_pct:.1f}%)',
        f'LSTM-SVRがARIMAXより0%～10%優れている\n({minor_improvement_pct:.1f}%)'
    ]
    sizes = [underperform_pct, significant_improvement_pct, minor_improvement_pct]
    colors = ['lightcoral', 'lightgreen', 'skyblue']
    explode = (0.05, 0, 0)  # 少しだけ「性能が劣る」部分を強調

    fig3, ax3 = plt.subplots(figsize=(13, 9), dpi=150)
    ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors,
            textprops={'fontsize': 12})
    ax3.axis('equal')  # 円を真円にする
    ax3.set_title('テスト期間全体におけるARIMAXとLSTMーSVR性能比較の割合', fontsize=18, y=1.05)

    # --- 5. 关键点深度分析 ---
    print("\n--- 开始进行关键点深度分析 ---")

    # 5.1. 将所有需要的数据合并到一个总的分析表中
    # 将基础误差数据与原始特征数据通过时间戳合并
    analysis_df = pd.merge(test_features_df, results_df, left_on='timestamp', right_on='timestamp')

    # 计算误差的差值，用于排序
    analysis_df['Error_Diff'] = analysis_df['abs_error_arimax'] - analysis_df['abs_error_lstm_svr']

    # 5.2. 找出两种情况下的Top 10
    top10_arimax_fails = analysis_df.sort_values(by='Error_Diff', ascending=False).head(10)
    top10_lstm_fails = analysis_df.sort_values(by='Error_Diff', ascending=True).head(10)

    # 5.3. 定义一个函数，用于将表格绘制并保存为图片
    def plot_and_save_table(data_df, title, filename):
        cols_to_display = [
            'timestamp',
            'abs_error_arimax', 'abs_error_lstm_svr', 'Error_Diff',  # 誤差関連の列
            'TARGET', 'ARIMAX_Prediction', 'LSTM_SVR_Prediction',  # 実績値と予測値
            'temperature TK', 'rainfall TK', 'humidity TK', 'wind speed TK',  # 東京の気象データ
            'temperature YH', 'rainfall YH', 'humidity YH', 'wind speed YH',  # 横浜の気象データ
            'temperature QB', 'rainfall QB', 'humidity QB', 'wind speed QB'  # 千葉の気象データ
        ]
        cols_to_display = [col for col in cols_to_display if col in data_df.columns]
        plot_df = data_df[cols_to_display].copy()

        # 格式化数据以便显示
        plot_df['timestamp'] = plot_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        for col in plot_df.select_dtypes(include=np.number).columns:
            plot_df[col] = plot_df[col].round(2)

        fig, ax = plt.subplots(figsize=(22, 5))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=plot_df.values, colLabels=plot_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.title(title, fontsize=16, y=1.05)
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        print(f"表格图片已保存: {filename}")
        plt.close(fig)

    # 5.4. 执行函数，生成两张表格图片
    output_dir = "critical_points_analysis_final"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_and_save_table(top10_arimax_fails,
                        'ARIMAXの誤差が特に大きいトップ10時点のデータ特徴',
                        os.path.join(output_dir, 'top10_arimax_fails.png'))

    plot_and_save_table(top10_lstm_fails,
                        'LSTM-SVRの誤差が特に大きいトップ10時点のデータ特徴',
                        os.path.join(output_dir, 'top10_lstm_fails.png'))

    print("--- 关键点深度分析完成 ---")

    # --- 6. すべてのグラフをインタラクティブに表示 ---
    print("\nすべてのインタラクティブなグラフを生成しました。表示します...")
    plt.show()


if __name__ == '__main__':
     run_comprehensive_analysis()
