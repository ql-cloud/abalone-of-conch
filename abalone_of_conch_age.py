import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 将models定义在全局作用域
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Random Forest Regression': RandomForestRegressor(random_state=42)
}

# 数据加载函数
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight',
                                                    'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Age'])
    return df


# 数据预处理函数
def preprocess_data(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    scaler = StandardScaler()
    numerical_features = ['Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    X = df.drop('Age', axis=1)
    y = df['Age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# 模型训练及性能指标计算函数
def train_models(X_train, X_test, y_train, y_test):
    # 这里可以直接使用全局定义的models变量
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        print(f"{name} - MSE: {mse}, MAE: {mae}, R2: {r2}")
    return results


# 可视化结果函数
def visualize_results(X_test, y_test, results):
    # 对比不同回归模型的MSE、MAE和R2值
    print(pd.DataFrame(results).T)

    # 散点图展示模型预测值与真实值的对比
    plt.figure(figsize=(12, 6))
    for name in results.keys():
        plt.scatter(y_test, results[name]['model'].predict(X_test), label=name)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')  # 添加对角线
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.show()

    # 绘制残差分析图
    plt.figure(figsize=(12, 6))
    for name in results.keys():
        residuals = y_test - results[name]['model'].predict(X_test)
        plt.scatter(results[name]['model'].predict(X_test), residuals, label=name)
        # 可在此添加拟合曲线代码进一步分析残差
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.legend()
    plt.show()

    # 绘制Length与Age的关系图
    plt.figure(figsize=(6, 6))
    plt.scatter(load_data(file_path)['Length'], load_data(file_path)['Age'])
    plt.xlabel('Length')
    plt.ylabel('Age')
    plt.title('Length vs Age')
    plt.show()


def select_file_and_process():
    global file_path
    file_path = filedialog.askopenfilename(title="选择数据文件", filetypes=[("Data Files", "*.data;*.csv")])
    if file_path:
        df = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        results = train_models(X_train, X_test, y_train, y_test)
        results_with_models = {name: {'model': model,'metrics': metrics} for (name, model), (_, metrics) in
                               zip(models.items(), results.items())}
        visualize_results(X_test, y_test, results_with_models)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("数据处理与分析工具")
    # 设置窗口大小
    root.geometry("400x200")
    # 使用clam主题，让界面更美观（可选主题还有 'alt', 'default', 'classic'等）
    style = ttk.Style(root)
    style.theme_use('clam')

    # 创建一个框架用于布局
    main_frame = ttk.Frame(root)
    main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    # 添加提示信息标签，告知用户操作步骤
    instruction_label = ttk.Label(main_frame, text="请点击下方按钮选择要分析的数据文件。",
                                  font=("Arial", 12))
    instruction_label.pack(pady=(0, 15))

    # 创建选择文件并处理按钮，设置样式
    process_button = ttk.Button(main_frame, text="选择文件并处理", command=select_file_and_process)
    process_button.pack(pady=10)
    # 配置按钮样式，改变背景色、前景色和字体等（可根据喜好调整颜色值）
    style.configure('TButton',
                    background='#4CAF50',  # 按钮背景色
                    foreground='white',  # 按钮前景色（文字颜色）
                    font=('Arial', 12, 'bold'))  # 按钮字体

    root.mainloop()