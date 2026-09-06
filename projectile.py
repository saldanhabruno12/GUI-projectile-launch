import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def calcular_resultados(v0, theta_graus, y0, g):
    #conversao de graus para radianos
    theta = math.radians(theta_graus)  
    vy0 = v0 * math.sin(theta)
    vx0 = v0 * math.cos(theta)

    t_voo = (vy0 + math.sqrt(vy0**2 + 2 * g * y0)) / g
    y_max = y0 + (vy0**2) / (2 * g)
    R = vx0 * t_voo

    return t_voo, y_max, R

def calcular_trajetoria(v0, theta_graus, y0, g, n_pontos=300):
    theta = math.radians(theta_graus)
    vy0 = v0 * math.sin(theta)
    vx0 = v0 * math.cos(theta)

    t_voo, _, _ = calcular_resultados(v0, theta_graus, y0, g)
    t = np.linspace(0, t_voo, n_pontos)

    x = vx0 * t                        
    y = y0 + vy0 * t - 0.5 * g * t**2  

    return t, x, y

if __name__ == "__main__":
    # Valores iniciais
    v0_init, theta_init, y0_init, g_init = 50, 45, 0, 9.8

    t, x, y = calcular_trajetoria(v0_init, theta_init, y0_init, g_init)
    t_voo, y_max, R = calcular_resultados(v0_init, theta_init, y0_init, g_init)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)  # abre espaço embaixo para os sliders

    linha_trajetoria, = ax.plot(x, y, lw=2)

    ax.set_xlabel("Distância (m)")
    ax.set_ylabel("Altura (m)")
    ax.set_aspect('equal')
    ax.grid(True)

    ax.set_xlim(0, R * 1.1)
    ax.set_ylim(0, y_max * 1.2)
    ax.set_title(f"Alcance: {R:.1f} m | Altura máx: {y_max:.1f} m | Tempo de voo: {t_voo:.2f} s")

    # --- Sliders ---
    ax_v0 = plt.axes([0.25, 0.20, 0.5, 0.03])
    ax_theta = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_y0 = plt.axes([0.25, 0.10, 0.5, 0.03])
    ax_g = plt.axes([0.25, 0.05, 0.5, 0.03])

    slider_v0 = Slider(ax_v0, 'v0 (m/s)', 5, 150, valinit=v0_init)
    slider_theta = Slider(ax_theta, 'θ (°)', 1, 89, valinit=theta_init)
    slider_y0 = Slider(ax_y0, 'y0 (m)', 0, 50, valinit=y0_init)
    slider_g = Slider(ax_g, 'g (m/s²)', 1.6, 24.8, valinit=g_init)

    # --- Callback ---
    def atualizar_grafico(val):
        v0 = slider_v0.val
        theta = slider_theta.val
        y0 = slider_y0.val
        g = slider_g.val

        t, x, y = calcular_trajetoria(v0, theta, y0, g)
        t_voo, y_max, R = calcular_resultados(v0, theta, y0, g)

        linha_trajetoria.set_data(x, y)
        ax.set_xlim(0, max(R, 1) * 1.1)
        ax.set_ylim(0, max(y_max, 1) * 1.2)
        ax.set_title(f"Alcance: {R:.1f} m | Altura máx: {y_max:.1f} m | Tempo de voo: {t_voo:.2f} s")

        fig.canvas.draw_idle()

    slider_v0.on_changed(atualizar_grafico)
    slider_theta.on_changed(atualizar_grafico)
    slider_y0.on_changed(atualizar_grafico)
    slider_g.on_changed(atualizar_grafico)

    plt.show()