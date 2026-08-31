'draw function curve by matplotlib'
import matplotlib.pyplot as plt
import numpy as np

import mpl_toolkits.axisartist as ax

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def exp_expand(x, beta=5):
    y = np.where(x > 1.2, 1.2 + 1/beta - np.exp(beta * (1.2-x))/beta, x)
    y = np.where(y < 0.8, 0.8 - 1/beta + np.exp(beta * (y-0.8))/beta, y)
    return y
def exp_expa_gra(x, beta=5):
    # y=x
    y = np.where(x < 0.8, np.exp(beta * (x-0.8)), x)
    y = np.where(y > 1.2, np.exp(beta * (1.2-y)), y)
    y[400:600] = 1
    return y

def draw_L():
    ax1 = ax.Subplot(fig, 111)
    fig.add_axes(ax1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.axis['top'].set_visible(False)
    ax1.axis['right'].set_visible(False)
    ax1.axis["bottom"] = ax1.new_floating_axis(0, 0,axis_direction="bottom")
    ax1.axis["left"]= ax1.new_floating_axis(1, 0, axis_direction="left")
    # ax1.axis["bottom"].major_ticklabels.set_visible(False)
    # ax1.axis["left"].major_ticklabels.set_visible(False)
    # ax1.axis["left"].set_axisline_style("->", size=1.5)
    # ax1.axis["left"].toggle(False)
    ax1.set_xticks([1,1.2])
    # ax1.set_xticklabels([0,0])
    ax1.set_yticks([1])
    x = np.linspace(-2, 2, 1000)
    sig = np.tanh(x) #sigmoid(2 * (x-1)) * 2
    # y_cli = np.clip(x,0,1.2)
    # y_exp = exp_expand(x)
    plt.plot(x, sig)
    # plt.plot(1, 1, 'ro')
    plt.plot(x, x)
    # plt.plot(x, y_exp)
    # plt.xlabel('r',loc='right')
    # plt.ylabel("L",loc="top")

def draw_grad_L():
    ax1 = ax.Subplot(fig, 111)
    fig.add_axes(ax1)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.axis['top'].set_visible(False)
    ax1.axis['right'].set_visible(False)
    ax1.axis["bottom"] = ax1.new_floating_axis(0, 0,axis_direction="bottom")
    ax1.axis["left"]= ax1.new_floating_axis(1, 0, axis_direction="left")
    # ax1.axis["bottom"].major_ticklabels.set_visible(False)
    # ax1.axis["left"].major_ticklabels.set_visible(False)
    # ax1.axis["left"].set_axisline_style("->", size=1.5)
    # ax1.axis["left"].toggle(False)
    ax1.set_xticks([1,1.2])
    # ax1.set_xticklabels([0,0])
    ax1.set_yticks([1])
    x = np.linspace(0, 2, 1000)
    sig = sigmoid(2 * (x-1))*(1 - sigmoid(2 * (x-1))) * 4
    # y_cli = np.clip(x,0,1.2)
    y_exp = exp_expa_gra(x)
    plt.plot(x, sig)
    plt.plot(1, 1, 'ro')
    # plt.plot(1.2*x/x,x)
    # plt.plot(x, y_cli)
    plt.plot(x, y_exp)
    # plt.xlabel('r',loc='right')
    # plt.ylabel("L",loc="top")

if __name__ == '__main__':
    fig = plt.figure()
    draw_L()

    plt.show()
