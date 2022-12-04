# Статистические методы анализа данных лабораторная работа 3
# Вариант №6 Бригада:Абраменко, Мак, Назаров

import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f, t
np.random.seed(1)

u, y, x1, x2, x3 = [], [], [], [], []
w = 0   # Мощность отклика


# Генерация равномерного распределения x1,x2,x3 в диапазоне [-1,1] с учётом уровней
def generate_x(x1, x2, x3, x_levels):
    for x_1 in np.linspace(1, -1, x_levels[0]):
        for x_2 in np.linspace(1, -1, x_levels[1]):
            for x_3 in np.linspace(1, -1, x_levels[2]):
                x1.append(x_1)
                x2.append(x_2)
                x3.append(x_3)


# Вычисление незашумленного и зашумленного отклика
def param_calc(x1, x2, x3, theta, n, data_print):
    global y, u, w
    e = []  # вектор шума

    for i in range(n):
        u = np.append(u, theta[0] + theta[1]*x1[i]+theta[2]*x2[i]+theta[3]*x3[i]+theta[4]*(x1[i])**2+\
               theta[5]*(x2[i])**2+theta[6]*(np.sin(x3[i]))+theta[7]*(x1[i]*x2[i])+theta[8]*(x1[i]*x3[i]) + theta[9]*(x2[i]*x3[i]))

    u_mean = np.sum(u)/n  # Среднее значение отклика

    w = np.dot((u - u_mean).transpose(), (u - u_mean)) / (n - 1)  # Мощность от-клика
    for i in range(n):
        e = np.append(e, np.random.normal(0, np.sqrt(0.7 * w)))  # Случайная ошибка
        y.append(u[i] + e[i])

    if data_print:
        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'u': u, 'e': e, 'y': y})
        print(data)


# Вычисление оценочных параметров
def gen_est(x1, x2, x3, y, n):
    z = np.ones(n)  # Вспомогательный вектор из единичек
    x1x2, x1x3, x2x3, x1_square, x2_square, x3_square = np.empty([n]), np.empty([n]), np.empty([n]), np.empty(
        [n]), np.empty([n]), np.empty([n])
    for i in range(n):
        x1x2[i], x1x3[i], x2x3[i] = x1[i] * x2[i], x1[i] * x3[i], x2[i] * x3[i]
        x1_square[i], x2_square[i], x3_square[i] = x1[i] ** 2, x2[i] ** 2, np.sin(x3[i])

    # Матрица X
    fx = np.vstack([z, x1, x2, x3, x1_square, x2_square, x3_square, x1x2, x1x3, x2x3]).transpose()

    # Оценочные значения тетта
    theta_est = np.dot(np.dot(inv(np.dot(fx.transpose(), fx)), fx.transpose()), y)
    e_est = y - np.dot(fx, theta_est)
    unb_est = np.dot(e_est.transpose(), e_est) / (n - 10)  # дисперсия несмещён-ной оценки

    return fx, theta_est, e_est, unb_est


# Лабораторная работа 3
def conf_interval(x1, x2, x3, y, n, theta, data_print):
    fx, theta_est, e_est, unb_est = gen_est(x1, x2, x3, y, n)
    fxt = fx.transpose()
    inv_fxt_fx = inv(np.dot(fxt, fx))

    l_interval = np.empty(len(theta_est))  # Минимальное значение
    r_interval = np.empty(len(theta_est))  # Максимальное значение
    theta_disp = np.empty(len(theta_est))  # Фактическое значение
    qst = t.ppf(0.975, 14)

    # Вычисляем доверительные интервалы для каждого параметра
    for j in range(len(theta_est)):
        theta_disp[j] = np.sqrt(unb_est*inv_fxt_fx[j][j])
        l_interval[j] = theta_est[j] - qst*theta_disp[j]
        r_interval[j] = theta_est[j] + qst*theta_disp[j]

    if data_print:
        data = pd.DataFrame({'left interval': l_interval, 'theta_est': theta_est, 'theta': theta, 'right interval': r_interval})
        print(data)


# Проверка гипотезы о незначимости каждого параметра
def param_hypothesis(x1, x2, x3, y, n, data_print):
    fx, theta_est, e_est, unb_est = gen_est(x1, x2, x3, y, n)
    f_t = f.ppf(0.95, 1, 14)
    fxt = fx.transpose()
    inv_fxt_fx = inv(np.dot(fxt, fx))
    f_est = np.empty(len(theta_est))

    for j in range(len(theta_est)):  # Вычисление статистики j-го параметра
        f_est[j] = theta_est[j]**2 / unb_est*inv_fxt_fx[j][j]

    if data_print:
        data = pd.DataFrame({'Статистика j-ого параметра': f_est, 'Незначимость параметра': f_est < f_t})
        print(data)


# Проверка гипотезы о незначимости регрессии
def regr_hypothesis(x1, x2, x3, y, n,data_print):
    fx, theta_est, e_est, unb_est = gen_est(x1, x2, x3, y, n)

    RSS = 14*unb_est
    RSSh = sum(np.square(y - (sum(y)/len(y))))
    f_est = ((RSSh-RSS)/9)/(RSS/14)
    f_t = f.ppf(0.95, 9, 14)

    if data_print:
        print("Гипотеза о незначимости регрессии: ", f_est > f_t, f_est, f_t)


# Прогнозные значения математического ожидания функции отклика для интервала действия x1
def mexp_est(x1, x2, x3, y, n, theta, data_print):
    fx, theta_est, e_est, unb_est = gen_est(x1, x2, x3, y, n)
    fxt = fx.transpose()
    inv_fxt_fx = inv(np.dot(fxt, fx))
    f_t = t.ppf(0.975, 14)

    # Зафиксируем значения фактора x2 и x3 в точках [1] и [1]
    # Фактор X1 изменяется на трёх уровнях {-1,0,1}
    # Посчитаем значения точной и оценочной модели в этих точках
    x_1 = [-1, 0, 1]
    x_2, x_3 = 1, 1
    fx_mexp, nu, nu_est, sigma_nu_est, left_interval, right_interval = np.empty([3, 10]), np.empty(3), np.empty(3), np.empty(3), np.empty(3), np.empty(3)
    for i in range(3):
        fx_mexp[i] = [1, x_1[i], x_2, x_3, x_1[i]**2, x_2**2, np.sin(x_3), x_1[i]*x_2, x_1[i]*x_3, x_2*x_3]
        nu[i] = np.dot(fx_mexp[i].transpose(), theta)
        nu_est[i] = np.dot(fx_mexp[i].transpose(), theta_est)
        sigma_nu_est[i] = np.sqrt(unb_est*(np.dot(np.dot(fx_mexp[i].transpose(), inv_fxt_fx), fx_mexp[i])))
        left_interval[i] = nu_est[i] - f_t*sigma_nu_est[i]
        right_interval[i] = nu_est[i] + f_t*sigma_nu_est[i]

    if data_print:
        data_nu = pd.DataFrame({'left interval': left_interval, 'nu': nu, 'right interval': right_interval})
        print(data_nu)
        return data_nu


# Прогнозные значения математического ожидания отклика для интервала действия фактора x1
def yexp_est(x1, x2, x3, y, n, theta, data_print):
    fx, theta_est, e_est, unb_est = gen_est(x1, x2, x3, y, n)
    fxt = fx.transpose()
    inv_fxt_fx = inv(np.dot(fxt, fx))
    f_t = t.ppf(0.975, 14)

    # Зафиксируем значения фактора x2 и x3 в точках [1] и [1]
    # Фактор X1 изменяется на трёх уровнях {-1,0,1}
    # Посчитаем значения точной и оценочной модели в этих точках
    x_1 = [-1, 0, 1]
    x_2, x_3 = 1, 1
    fx_mexp, nu, y_est, sigma_y_est, left_interval, right_interval = np.empty([3, 10]), np.empty(3), np.empty(3), np.empty(3), np.empty(3),np.empty(3)
    for i in range(3):
        fx_mexp[i] = [1, x_1[i], x_2, x_3, x_1[i]**2, x_2**2, np.sin(x_3), x_1[i]*x_2, x_1[i]*x_3, x_2*x_3]
        y_est[i] = np.dot(fx_mexp[i].transpose(), theta_est)
        sigma_y_est[i] = np.sqrt(unb_est*(1+(np.dot(np.dot(fx_mexp[i].transpose(), inv_fxt_fx), fx_mexp[i]))))
        left_interval[i] = y_est[i] - f_t*sigma_y_est[i]
        right_interval[i] = y_est[i] + f_t*sigma_y_est[i]

    if data_print:
        data_y = pd.DataFrame({'left interval': left_interval, 'y_est': y_est, 'right interval': right_interval})
        print(data_y)
        return data_y


def plots(data, data_print):
    # Для отрисовки графиков необходимо установить matplotlib версии не выше 3.5.3
    if data_print:
        x = [-1, 0, 1]
        y_1, y_2, y_3 = np.empty(3),np.empty(3),np.empty(3)
        for i in range(3):
            y_1[i] = data.to_numpy()[i][0]
            y_2[i] = data.to_numpy()[i][1]
            y_3[i] = data.to_numpy()[i][2]

        sns.lineplot(x=x, y=y_1)
        sns.lineplot(x=x, y=y_2)
        sns.lineplot(x=x, y=y_3)

        plt.show()


def mnk(x1, x2, x3, y, n):
    fx, theta_est, e_est, unb_est = gen_est(x1, x2, x3, y, n)
    base_est = 0.7 * w  # дисперсия из 1 лабораторной

    # Проверка адекватности модели
    print(base_est)
    print(unb_est)
    ft = unb_est / base_est
    ftt = f.ppf(0.95, 14, 1e+10)  # Табличное значение F-распределения
    print("ft", ft)
    print("f_tabl", ftt)
    print("Статус гипотезы об адекватности: ", ft < ftt)  # Если истина, то мо-дель признаётся адекватной


def main():
    n = 24                                              # кол-во генерируемых наблюдений
    x_levels = [3, 4, 2]                                # Уровни x в соответ-ствии с вариантом задания
    theta = [1, 1, 1, 1, 0.01, 0.01, 0.01, 1, 1, 0]
    data_print = True                                  # Вывод таблиц/графиков если True, иначе False

    generate_x(x1, x2, x3, x_levels)
    param_calc(x1, x2, x3, theta, n, data_print)
    mnk(x1, x2, x3, y, n)

    conf_interval(x1, x2, x3, y, n, theta, data_print)
    param_hypothesis(x1, x2, x3, y, n, data_print)
    regr_hypothesis(x1, x2, x3, y, n, data_print)
    me = mexp_est(x1, x2, x3, y, n, theta, data_print)
    ye = yexp_est(x1, x2, x3, y, n, theta, data_print)
    plots(ye, data_print)


main()
