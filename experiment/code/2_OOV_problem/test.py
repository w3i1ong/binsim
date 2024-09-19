import matplotlib.pyplot as plt

# 创建数据
x = range(10)
y1 = [i**2 for i in x]
y2 = [i*10 for i in x]



fig = draw_lines(x, y1, "a", y2, "b")
fig.show()
if __name__ == '__main__':
    pass
