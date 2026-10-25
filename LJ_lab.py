from pathlib import Path
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Параметры расчёта
out_dir = Path(__file__).resolve().parent / "results"
dt = 0.004                      # безразмерный шаг времени
t_end_sweep = 60.0              # длительность каждого расчёта для определения периода
amplitudes = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
                       0.10, 0.13, 0.16, 0.20, 0.25, 0.30,
                       0.35, 0.40, 0.45, 0.50])
a_example = 0.20                # пример для графиков r(t), E(t), GIF
max_time_plot = 12.0           # на графиках показываем первые 12 ед. времени
dt_steps = [0.016, 0.008, 0.004, 0.002, 0.001]
error_tol_pct = 1.0             # критерий применимости гармонического приближения
create_gif = True
gif_frames = 90
gif_fps = 18

r_eq = 2.0 ** (1.0 / 6.0)
k_harm = 72.0 / (r_eq * r_eq)
omega_harm = math.sqrt(k_harm)
t_harm = 2.0 * math.pi / omega_harm


def potential(r):
    """Потенциал Леннарда — Джонса в приведённых единицах."""
    r = np.asarray(r)
    return 4.0 * (r**-12 - r**-6)


def force(r):
    """Сила, действующая на относительную координату."""
    r = np.asarray(r)
    return 24.0 * (2.0 * r**-13 - r**-7)


def harmonic_potential(r):
    return -1.0 + 0.5 * k_harm * (np.asarray(r) - r_eq)**2


def simulate(a, dt=dt, t_end=t_end_sweep):
    """Расчёт относительного движения позиционным методом Верле."""
    n = int(math.ceil(t_end / dt))
    t = np.arange(n + 1, dtype=float) * dt
    r = np.empty(n + 1, dtype=float)
    r[0] = r_eq * (1.0 + a)
    r[1] = r[0] + 0.5 * dt * dt * float(force(r[0]))
    for i in range(1, n):
        r[i + 1] = 2.0*r[i] - r[i - 1] + dt*dt*float(force(r[i]))
        if not np.isfinite(r[i+1]) or r[i+1] <= 0:
            raise ArithmeticError(f"Потеря устойчивости: шаг {i}, r={r[i+1]}; уменьшите dt")
    v = np.empty_like(r)
    v[1:-1] = (r[2:] - r[:-2]) / (2.0*dt)
    v[0] = (-3*r[0]+4*r[1]-r[2])/(2*dt)
    v[-1] = (3*r[-1]-4*r[-2]+r[-3])/(2*dt)
    ep = potential(r)
    ek = 0.5 * v*v
    e = ek + ep
    return {"t": t, "r": r, "v": v, "ep": ep, "ek": ek, "e": e,
            "a": float(a), "dt": float(dt)}


def turning_max_times(t, r):
    """Времена максимумов с параболической интерполяцией."""
    idx = np.flatnonzero((r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:])) + 1
    times = []
    dt = float(t[1] - t[0])
    for i in idx:
        left, mid, right = r[i-1], r[i], r[i+1]
        denom = left - 2*mid + right
        shift = 0.5*(left-right)/denom if denom != 0 else 0
        times.append(float(t[i] + dt*np.clip(shift, -1, 1)))
    return np.array(times)


def measured_period(res):
    maxima = turning_max_times(res["t"], res["r"])
    if len(maxima) < 4:
        raise ValueError("Недостаточно максимумов для определения периода: увеличьте t_end_sweep")
    periods = np.diff(maxima)
    return float(np.mean(periods)), float(np.std(periods, ddof=1)), len(periods)


def energy_error_pct(res):
    # Нормируем на epsilon, а не на отрицательную E(0).
    return 100 * np.abs(res["e"] - res["e"][0])


def plot_save(fig, filename):
    fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_animation(res):
    """Анимация движения в системе центра масс."""
    t = res["t"]
    r = res["r"]
    period, _, _ = measured_period(res)
    last_t = min(float(t[-1]), 3*period)
    inds = np.linspace(0, int(last_t / res["dt"]), gif_frames).astype(int)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.axhline(0, color="0.8", lw=0.8)
    p1, = ax.plot([], [], "o", ms=17, color="tab:blue", label="Атом 1")
    p2, = ax.plot([], [], "o", ms=17, color="tab:orange", label="Атом 2")
    title = ax.set_title("")
    ax.set_xlabel("Координата x / σ")
    ax.legend(loc="upper right")
    def update(frame):
        i = inds[frame]
        p1.set_data([-r[i]/2], [0])
        p2.set_data([r[i]/2], [0])
        title.set_text(f"Колебания двух атомов, t* = {t[i]:.2f}, r* = {r[i]:.3f}")
        return p1, p2, title
    anim = FuncAnimation(fig, update, frames=gif_frames, blit=False)
    anim.save(out_dir / "animation.gif", writer=PillowWriter(fps=gif_fps), dpi=90)
    plt.close(fig)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    print("ЛР № 6: потенциал Леннарда–Джонса, позиционный метод Верле")
    print(f"Аналитическое равновесие: r_eq* = {r_eq:.10f}")
    print(f"Гармоническая частота: omega_0* = {omega_harm:.10f}")
    print(f"Гармонический период: T_0* = {t_harm:.10f}")

    # Потенциал и сила
    x = np.linspace(0.97, 2.5, 700)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, potential(x), label="Потенциал LJ")
    ax.plot(x, harmonic_potential(x), "--", label="Гармоническое приближение")
    ax.axvline(r_eq, color="black", ls=":", label=f"r_eq*={r_eq:.4f}")
    ax.set(xlabel="Межатомное расстояние r/σ", ylabel="U/ε", ylim=(-1.5, 4),
           title="Потенциальная энергия взаимодействия")
    ax.legend()
    plot_save(fig, "output1.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, force(x), label="F* = −dU*/dr*", color="tab:red")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(r_eq, color="black", ls=":", label="Равновесие")
    ax.set(xlabel="Межатомное расстояние r/σ", ylabel="Безразмерная сила F*",
           ylim=(-3, 25), title="Межатомная сила: отталкивание и притяжение")
    ax.legend()
    plot_save(fig, "output2.png")

    # Зависимость периода от амплитуды
    records = []
    for a in amplitudes:
        res = simulate(float(a))
        T, std_T, nperiod = measured_period(res)
        err = abs(T/t_harm-1)*100
        records.append([float(a), T, T/t_harm, err, std_T, nperiod,
                        float(np.max(energy_error_pct(res)))])
    with (out_dir / "period_sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["A_over_r_eq", "T_reduced", "T_over_T0", "harmonic_error_percent",
                     "period_std", "n_periods", "max_energy_error_percent"])
        wr.writerows(records)
    data = np.array(records, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a in (0.02, 0.10, a_example, 0.40):
        z = simulate(a, t_end=max_time_plot)
        ax.plot(z["t"], z["r"], label=f"A/r_eq = {a:.2f}")
    ax.set(xlabel="Время t*", ylabel="Расстояние r*", title="Колебания при разных начальных отклонениях")
    ax.legend(ncol=2)
    plot_save(fig, "output3.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data[:,0], data[:,2], "o-", label="Модель Леннарда–Джонса")
    ax.axhline(1, color="tab:red", ls="--", label="Гармоническое приближение")
    ax.set(xlabel="Начальное отклонение A/r_eq", ylabel="T / T₀",
           title="Влияние начального отклонения на период")
    ax.legend()
    plot_save(fig, "output4.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data[:,0], data[:,3], "o-", label="Относительное отклонение периода")
    ax.axhline(error_tol_pct, color="tab:red", ls="--", label=f"Критерий {error_tol_pct:g}%")
    ax.set(xlabel="Начальное отклонение A/r_eq", ylabel="|T/T₀ − 1| · 100%, %",
           title="Граница применимости гармонического приближения")
    ax.legend()
    plot_save(fig, "output5.png")

    # Энергия
    example = simulate(a_example, t_end=max_time_plot)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(example["t"], example["ek"], label="Кинетическая энергия Eₖ*")
    ax.plot(example["t"], example["ep"], label="Потенциальная энергия Eₚ*")
    ax.plot(example["t"], example["e"], label="Полная энергия E*", lw=2)
    ax.set(xlabel="Время t*", ylabel="Энергия / ε", title="Закон сохранения механической энергии")
    ax.legend()
    plot_save(fig, "output6.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(example["t"], energy_error_pct(example))
    ax.set(xlabel="Время t*", ylabel="|E*(t)−E*(0)| · 100%, %",
           title="Отклонение полной энергии от начального значения")
    plot_save(fig, "output7.png")

    # Сходимость по шагу
    conv = []
    for dt in dt_steps:
        z = simulate(a_example, dt=dt, t_end=40.0)
        Tp, _, _ = measured_period(z)
        conv.append([dt, float(np.max(energy_error_pct(z))), Tp, abs(Tp/t_harm-1)*100])
    with (out_dir / "time_step_convergence.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["dt", "max_energy_error_percent", "period_reduced", "deviation_from_harmonic_percent"])
        wr.writerows(conv)
    cv = np.array(conv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(cv[:,0], cv[:,1], "o-", label="Максимальная ошибка энергии")
    ax.set(xlabel="Шаг Δt*", ylabel="Макс. |E*−E*(0)| · 100%, %",
           title="Сходимость энергетической погрешности")
    ax.legend()
    plot_save(fig, "output8.png")

    # Проверка равновесия
    eq = simulate(a=0, dt=dt, t_end=10)
    eq_err = float(np.max(np.abs(eq["r"]-r_eq)))
    if eq_err > 1e-8:
        raise AssertionError(f"Не выполнен тест равновесия: ошибка {eq_err}")
    assert abs(float(force(r_eq))) < 1e-12
    assert abs(float(potential(r_eq)) + 1) < 1e-12

    # Пересчёт времени для аргона
    kb = 1.380649e-23            # Дж/К
    atomic_mass_unit = 1.66053906660e-27 # кг
    sigma_argon = 3.405e-10      # м, типичный параметр LJ для Ar
    eps_argon = 119.8 * kb       # Дж, пример параметра LJ для Ar
    m_argon = 39.948 * atomic_mass_unit
    mu_argon = m_argon/2
    tau_argon = sigma_argon*math.sqrt(mu_argon/eps_argon)

    valid = data[data[:,3] <= error_tol_pct, 0]
    print("\nЭксперимент: a, T*, T/T0, отклонение в %")
    for row in records:
        print(f"{row[0]:6.3f}  {row[1]:9.6f}  {row[2]:9.6f}  {row[3]:10.5f}")
    print("\nМаксимальное ИССЛЕДОВАННОЕ отклонение, отвечающее критерию: ",
          f"{max(valid):.4f}" if len(valid) else "нет среди исследованных точек")
    print(f"Ошибка стационарного положения: {eq_err:.3e}")
    print(f"Пример для Ar: sigma = {sigma_argon:.4e} м, epsilon/kB = 119.8 К, tau = {tau_argon:.4e} с")
    print(f"Пример для Ar: гармонический период = {t_harm*tau_argon:.4e} с")

    if create_gif:
        make_animation(simulate(a_example, t_end=20))
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("ЛР № 6: LJ; позиционный Верле\n")
        f.write(f"r_eq* = {r_eq:.12g}; T0* = {t_harm:.12g}\n")
        f.write(f"dt*={dt}, критерий {error_tol_pct}%\n")
        f.write(f"Максимальная проверенная амплитуда с ошибкой <= {error_tol_pct}%: {max(valid) if len(valid) else 'нет'}\n")
        f.write("Это дискретная проверка на сетке амплитуд, а не доказанная точная граница.\n")
        f.write("Результаты: period_sweep.csv, time_step_convergence.csv, output1-8.png, animation.gif\n")
    print(f"\nФайлы сохранены в: {out_dir}")


if __name__ == "__main__":
    main()
