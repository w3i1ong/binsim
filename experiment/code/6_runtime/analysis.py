from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


def plot_bar_chart(models, gpu_times, cpu_times, gpu_color, cpu_color):
    # 将GPU时间和CPU时间分别绘制在柱状图上
    bar_width = 0.35
    index = range(len(models))

    plt.figure(figsize=(12, 6))

    bar1 = plt.bar(index, gpu_times, bar_width, color=gpu_color, hatch='.', label='GPU Time')
    bar2 = plt.bar([i + bar_width for i in index], cpu_times, bar_width, color=cpu_color, hatch='o', label='CPU Time')

    # 添加标题和标签
    # plt.title('Comparison of Neural Network Models Run Time',fontsize=15)
    plt.xlabel('Models',fontsize=15)
    plt.ylabel('Run Time (seconds)',fontsize=15)
    plt.xticks([i + bar_width / 2 for i in index], models, fontsize=12)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(prop={'size': 15})
    # 显示数值
    # for i, gpu_time in enumerate(gpu_times):
    #     plt.text(i, gpu_time + 0.5, str(gpu_time), ha='center')
    #     plt.text(i + bar_width, cpu_times[i] + 0.5, str(cpu_times[i]), ha='center')
    # 显示图形
    plt.show()


# 模型名称
models = [r'$\alpha$-diff', 'Gemini', 'SAFE', 'GraphEmbed', 'Asteria',
          'jtrans']

# 模型的平均GPU运行时间和CPU运行时间（假设这里是随机生成的数据）
gpu_times = [10, 15, 20, 10, 10, 10]
cpu_times = [5, 10, 15, 100, 100, 100]

# GPU时间和CPU时间的颜色
gpu_color = "#8CD17D"
cpu_color = "#FF9D9A"

plot_bar_chart(models, gpu_times, cpu_times, gpu_color, cpu_color)


def main():
    pass


if __name__ == '__main__':
    main()
