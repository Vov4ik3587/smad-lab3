import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
import pandas as pd


class Model:  # Возьмем модель из лаб работы 1-2

    def __init__(self):
        self.amount_tests = 12
        self.x_max = 1
        self.x_min = -1
        self.x1 = []
        self.x2 = []
        self.signal = []  # сигнал
        self.response = []  # отклик
        self.power = 0  # мощность сигнала
        self.variance = 0  # дисперсия
        self.theta = np.array([1, 4, 0.001, 4])  # параметры модели
        self.func = lambda x1, x2: 1 + 4 * x1 + 0.001 * x1 ** 2 + 4 * x2 ** 2
        self.experiment_matrix = []  # Матрица наблюдений Х
        self.theta_mnk = []  # Оценка теты по МНК


class Calculator:

    @staticmethod
    def compute_signal(model: Model):  # Вычисление сигнала - незашумленного отклика
        signal = [model.func(model.x1[i], model.x2[i])
                  for i in range(model.amount_tests)]
        return np.array(signal)

    @staticmethod
    def compute_variance(model, percent) -> float:  # Вычисление дисперсии
        return model.power * (percent / 100)

    @staticmethod
    def compute_power(model):  # Вычисление мощности
        avg_signal = [
            np.sum(model.signal) / len(model.signal)
            for i in range(len(model.signal))
        ]
        vec_avg_signal = np.array(avg_signal)
        power = np.vdot(model.signal - vec_avg_signal,
                        model.signal - vec_avg_signal) / len(model.signal)
        return power

    @staticmethod
    def compute_response(model, error):  # вычисление зашумленного отклика
        return model.signal + error

    @staticmethod
    def mnk(model):  # Метод наименьших квадратов
        trans_experiment_matrix = model.experiment_matrix.T
        mnk_eval = np.matmul(np.linalg.inv(np.matmul(trans_experiment_matrix, model.experiment_matrix)),
                             trans_experiment_matrix)
        mnk_eval = np.matmul(mnk_eval, model.response)
        return mnk_eval

    @staticmethod
    def compute_experiment_matrix(model):  # Матрица наблюдений X
        if len(model.theta) == 5:  # То есть если есть доп регрессор
            experiment_matrix = np.array([
                np.array([1 for _ in range(model.amount_tests)]),
                model.x1,
                np.array([x1 ** 2 for x1 in model.x1]),
                np.array([x2 ** 2 for x2 in model.x2]),
                np.array([x2 ** 5 for x2 in model.x2])
            ], dtype=object)
            experiment_matrix = np.array([list(i) for i in zip(*experiment_matrix)])
            return experiment_matrix
        else:
            experiment_matrix = np.array([
                np.array([1 for _ in range(model.amount_tests)]),
                model.x1,
                np.array([x1 ** 2 for x1 in model.x1]),
                np.array([x2 ** 2 for x2 in model.x2])
            ], dtype=object)
            experiment_matrix = np.array([list(i) for i in zip(*experiment_matrix)])
            return experiment_matrix


class DataGenerator:

    @staticmethod
    def generate_couple(x_min, x_max, amount_tests):  # Генерация значений регрессоров
        x1 = np.random.uniform(x_min, x_max, amount_tests)
        x2 = np.random.uniform(x_min, x_max, amount_tests)
        return x1, x2

    @staticmethod
    def generate_error(standard_deviation, number_tests) -> float:  # генерация случайной ошибки
        error = np.random.normal(0, standard_deviation, number_tests)  # стандартное отклонение - sqrt(variance)
        return error


# %% Заполняем модель данными

model = Model()

model.x1, model.x2 = DataGenerator.generate_couple(
    model.x_min, model.x_max, model.amount_tests)

# Сохраним копию с исходными данными для последнего задания
last_model = copy.deepcopy(model)

model.signal = Calculator.compute_signal(model)
model.power = Calculator.compute_power(model)
model.variance = Calculator.compute_variance(model, 10)

error = DataGenerator.generate_error(
    np.sqrt(model.variance), model.amount_tests)

model.response = Calculator.compute_response(model, error)

# %% Добавим к исходной модели дополнительный регрессор
# Вычислим матрицу наблюдений и МНК-оценку параметров

model.func = lambda x1, x2: 1 + 4 * x1 + 0.001 * x1 ** 2 + 4 * x2 ** 2 + 0.001 * x2 ** 5
model.theta = np.array([1, 4, 0.001, 4, 0.001])
model.experiment_matrix = Calculator.compute_experiment_matrix(model)
model.theta_mnk = Calculator.mnk(model)

# %% Найдем вектор остатков и несмещенную оценку дисперсии

e_hat = model.response - np.matmul(model.experiment_matrix, model.theta_mnk)
eval_variance = np.vdot(e_hat.T, e_hat) / (model.amount_tests - len(model.theta_mnk))

# %% Проверим гипотезу об адекватности модели

f_quantile = scp.f.ppf(0.95, model.amount_tests - len(model.theta), 1e+10)
if eval_variance / model.variance <= f_quantile:
    print("Гипотеза не отвергается")
else:
    print("Гипотеза отвергается. Модель неадекватна.")

# %% Построим доверительные интервалы

# Найдем квантиль распределения Стьюдента.
# Уровень значимости - 0.05
st_quantile = scp.t.sf(0.05 / 2, model.amount_tests - len(model.theta))
m = len(model.theta)
matrix_d = np.linalg.inv(np.matmul(model.experiment_matrix.T, model.experiment_matrix))

sigma = np.empty(m)
left_inter = np.empty(m)
right_inter = np.empty(m)

for i in range(m):
    sigma[i] = np.sqrt(eval_variance * matrix_d[i][i])
    left_inter[i] = model.theta_mnk[i] - st_quantile * sigma[i]
    right_inter[i] = model.theta_mnk[i] + st_quantile * sigma[i]

confidence_interval = pd.DataFrame(
    {'left interval': left_inter, 'theta': model.theta, 'right interval': right_inter, 'eval theta': model.theta_mnk})
print(confidence_interval)

# %% Проверим гипотезу о незначимости каждого параметра модели

F_param = np.array([
    model.theta_mnk[i] ** 2 / (eval_variance * matrix_d[i][i])
    for i in range(m)
])
fisher_quantile = scp.f.ppf(0.95, 1, model.amount_tests - m)  # единица, так как это частный случай общей лин гипотезы
for i, value in enumerate(F_param):
    if value < fisher_quantile:
        print(f'Гипотеза о незначимости параметра theta[{i}] принимается')
    else:
        print(f'Гипотеза о незначимости параметра theta[{i}] НЕ принимается')

    # %% Проверим гипотезу о незначимости самой регрессии
f_quantile_regres = scp.f.ppf(0.95, m - 1, model.amount_tests - m)

tmp = (model.response - np.matmul(model.experiment_matrix, model.theta_mnk))
RSS = np.matmul(tmp.T, tmp)
RSS_H = np.sum(np.square(model.response - np.mean(model.response)))

F_regres = ((RSS_H - RSS) / (m - 1)) / (RSS / (model.amount_tests - m))

if F_regres < f_quantile_regres:
    print('Гипотеза о незначимости самой регрессии принимается')
else:
    print('Гипотеза о незначимости самой регрессии НЕ принимается')


# %% Рассчитаем прогнозные значения для мат ожидания функции отклика и для самого отклика


def create_vec_func(x1, x2):
    return np.array([1, x1, x1 ** 2, x2 ** 2, x2 ** 5])


left_inter_mat = np.empty(model.amount_tests)
right_inter_mat = np.empty(model.amount_tests)
nu_hat = np.empty(model.amount_tests)

vec_func_response = np.empty(model.amount_tests)
left_inter_response = np.empty(model.amount_tests)
right_inter_response = np.empty(model.amount_tests)

sort_x1 = np.sort(model.x1)

for i in range(model.amount_tests):
    # Интервал для мат ожидания
    vector_func = create_vec_func(sort_x1[i], 0)  # вектор значений функции факторов (х1, 0)
    nu_hat[i] = np.dot(vector_func.T, model.theta_mnk)
    sigma_mat = np.sqrt(np.dot(np.dot(vector_func.T, matrix_d), vector_func))
    left_inter_mat[i] = nu_hat[i] - st_quantile * sigma_mat
    right_inter_mat[i] = nu_hat[i] + st_quantile * sigma_mat
    # Интервал для отклика
    vec_func_response[i] = model.func(sort_x1[i], 0)
    sigma_response = np.sqrt(eval_variance * (1 + np.dot(np.dot(vector_func.T, matrix_d), vector_func)))
    left_inter_response[i] = vec_func_response[i] - st_quantile * sigma_response
    right_inter_response[i] = vec_func_response[i] + st_quantile * sigma_response

# %% Рисуем график доверительного интервала мат ожидания

plt.plot(sort_x1, left_inter_mat)
plt.plot(sort_x1, nu_hat)
plt.plot(sort_x1, right_inter_mat)
plt.show()

# %% Рисуем график доверительного интервала самого отклика

plt.plot(sort_x1, left_inter_response)
plt.plot(sort_x1, vec_func_response)
plt.plot(sort_x1, right_inter_response)
plt.show()

# %% Заново смоделировать исходные данные, увеличив мощность до 50..70%
# Провести оценку параметров, проверить гипотезы о незначимости параметров и самой регрессии

print('\nУвеличиваем мощность до 60%\n')

last_model.signal = Calculator.compute_signal(last_model)
last_model.power = Calculator.compute_power(last_model)
last_model.variance = Calculator.compute_variance(last_model, percent=60)

last_error = DataGenerator.generate_error(
    np.sqrt(last_model.variance), last_model.amount_tests)

last_model.response = Calculator.compute_response(last_model, last_error)

last_model.func = lambda x1, x2: 1 + 4 * x1 + 0.001 * x1 ** 2 + 4 * x2 ** 2 + 0.001 * x2 ** 5
last_model.theta = np.array([1, 4, 0.001, 4, 0.001])
last_model.experiment_matrix = Calculator.compute_experiment_matrix(last_model)
last_model.theta_mnk = Calculator.mnk(last_model)

last_e_hat = last_model.response - np.matmul(last_model.experiment_matrix, last_model.theta_mnk)
last_eval_variance = np.vdot(last_e_hat.T, last_e_hat) / (last_model.amount_tests - len(last_model.theta_mnk))

last_matrix_d = np.linalg.inv(np.matmul(last_model.experiment_matrix.T, last_model.experiment_matrix))

last_F_param = np.array([
    last_model.theta_mnk[i] ** 2 / (last_eval_variance * last_matrix_d[i][i])
    for i in range(m)
])
for i, value in enumerate(last_F_param):
    if value < fisher_quantile:
        print(f'Гипотеза о незначимости параметра theta[{i}] принимается')
    else:
        print(f'Гипотеза о незначимости параметра theta[{i}] НЕ принимается')

last_tmp = (last_model.response - np.matmul(last_model.experiment_matrix, last_model.theta_mnk))
last_RSS = np.matmul(last_tmp.T, last_tmp)
last_RSS_H = np.sum(np.square(last_model.response - np.mean(last_model.response)))

last_F_regres = ((last_RSS_H - last_RSS) / (m - 1)) / (last_RSS / (last_model.amount_tests - m))

if last_F_regres < f_quantile_regres:
    print('Гипотеза о незначимости самой регрессии принимается')
else:
    print('Гипотеза о незначимости самой регрессии НЕ принимается')
