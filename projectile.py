"""
Simulador de Lançamento de Projéteis - Interface Gráfica Interativa
=====================================================================

Disciplina : Física - Práticas Extensivas
Atividade  : Lançamento de Projéteis (sem resistência do ar)

INSTRUÇÕES DE EXECUÇÃO
-----------------------
Bibliotecas necessárias:
    pip install numpy matplotlib

Para executar:
    python projectile.py

Biblioteca de interface utilizada: matplotlib.widgets
(Slider, Button, RadioButtons, CheckButtons)
Escolhida por não exigir dependências externas além do matplotlib,
que já é usado para plotar o gráfico da trajetória.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# 1. FÍSICA: funções que calculam a trajetória e os resultados do lançamento
#    (totalmente separadas dos widgets/interface, conforme pedido no enunciado)
# ---------------------------------------------------------------------------

def calcular_resultados(v0, theta_graus, y0, g):
    """Calcula tempo de voo, altura máxima e alcance (fórmulas analíticas,
    Seção 2 do enunciado). Assume-se lançamento até y = 0."""
    theta = np.radians(theta_graus)
    vy0 = v0 * np.sin(theta)

    # tempo de voo: raiz positiva de  y0 + vy0*t - 0.5*g*t^2 = 0
    discriminante = vy0**2 + 2 * g * y0
    t_voo = (vy0 + np.sqrt(discriminante)) / g

    y_max = y0 + (vy0**2) / (2 * g)
    alcance = v0 * np.cos(theta) * t_voo

    return t_voo, y_max, alcance


def calcular_trajetoria(v0, theta_graus, y0, g, n_pontos=300):
    """Retorna os vetores t, x(t) e y(t) da trajetória, do lançamento até o solo."""
    theta = np.radians(theta_graus)
    t_voo, _, _ = calcular_resultados(v0, theta_graus, y0, g)

    t = np.linspace(0, t_voo, n_pontos)
    x = v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t**2
    y = np.clip(y, 0, None)  # evita valores negativos por erro de arredondamento
    return t, x, y


def validar_entradas(v0, theta_graus, y0, g):
    """Verifica se os parâmetros atuais formam um lançamento fisicamente válido.
    Retorna (valido: bool, mensagem_de_erro: str)."""
    if v0 <= 0:
        return False, "Velocidade inicial deve ser maior que 0."
    if not (0 < theta_graus < 90):
        return False, "Ângulo deve estar entre 0° e 90° (exclusive)."
    if y0 < 0:
        return False, "Altura inicial não pode ser negativa."
    if g <= 0:
        return False, "Gravidade deve ser maior que 0."
    return True, ""


# ---------------------------------------------------------------------------
# 2. PLANETAS PRÉ-CONFIGURADOS (valores de g em m/s²)
# ---------------------------------------------------------------------------
PLANETAS = {
    "Lua": 1.62,
    "Marte": 3.71,
    "Terra": 9.81,
    "Júpiter": 24.79,
}
OPCOES_GRAVIDADE = list(PLANETAS.keys()) + ["Personalizado"]
INDICE_PERSONALIZADO = len(OPCOES_GRAVIDADE) - 1

# ---------------------------------------------------------------------------
# 3. INTERFACE GRÁFICA (widgets e eventos - separado da física acima)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(left=0.10, bottom=0.42, right=0.78)

linha_trajetoria, = ax.plot([], [], lw=2, color="tab:blue", label="Trajetória")
ponto_animado, = ax.plot([], [], 'o', color="tab:red", markersize=8)
ax.set_xlabel("Distância horizontal x (m)")
ax.set_ylabel("Altura y (m)")
ax.set_title("Trajetória do Projétil (sem resistência do ar)")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="box")

texto_resultados = ax.text(
    0.03, 0.97, "", transform=ax.transAxes, fontsize=10, va="top", ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
)
texto_erro = ax.text(
    1.05, 0.60, "", transform=ax.transAxes, fontsize=9, color="crimson",
    va="top", wrap=True, clip_on=False,
)

# --- Sliders (requisito: controles interativos para v0, theta, y0, g) ---
eixo_v0 = plt.axes([0.10, 0.30, 0.60, 0.03])
eixo_theta = plt.axes([0.10, 0.25, 0.60, 0.03])
eixo_y0 = plt.axes([0.10, 0.20, 0.60, 0.03])
eixo_g = plt.axes([0.10, 0.15, 0.60, 0.03])

slider_v0 = Slider(eixo_v0, "v0 (m/s)", 5, 150, valinit=40)
slider_theta = Slider(eixo_theta, "θ (graus)", 1, 89, valinit=45)
slider_y0 = Slider(eixo_y0, "y0 (m)", 0, 50, valinit=0)
slider_g = Slider(eixo_g, "g (m/s²)", 1.6, 24.8, valinit=9.81)

# --- Seletor de planeta pré-configurado ---
eixo_planetas = plt.axes([0.80, 0.12, 0.17, 0.24])
radio_planetas = RadioButtons(eixo_planetas, OPCOES_GRAVIDADE, active=2)

# --- Checkbox: sobrepor lançamentos anteriores (requisito opcional) ---
eixo_check = plt.axes([0.10, 0.03, 0.28, 0.05])
check_sobrepor = CheckButtons(eixo_check, ["Manter trajetórias anteriores"], [False])

# --- Botões ---
eixo_lancar = plt.axes([0.42, 0.03, 0.14, 0.05])
botao_lancar = Button(eixo_lancar, "Lançar")

eixo_limpar = plt.axes([0.60, 0.03, 0.14, 0.05])
botao_limpar = Button(eixo_limpar, "Limpar")

# Guarda a animação e as curvas sobrepostas para que não sejam
# destruídas pelo garbage collector / possam ser removidas depois.
_animacao_atual = [None]
_curvas_anteriores = []
_atualizando_planeta = [False]


def ler_parametros():
    """Lê os valores atuais dos sliders."""
    return slider_v0.val, slider_theta.val, slider_y0.val, slider_g.val


def ajustar_limites(x, y):
    x_max_dados = max(float(np.max(x)), 1.0)
    y_max_dados = max(float(np.max(y)), 1.0)

    for curva in _curvas_anteriores:
        x_anterior = np.asarray(curva.get_xdata())
        y_anterior = np.asarray(curva.get_ydata())
        if x_anterior.size:
            x_max_dados = max(x_max_dados, float(np.max(x_anterior)))
        if y_anterior.size:
            y_max_dados = max(y_max_dados, float(np.max(y_anterior)))

    margem = 0.1
    x_max = x_max_dados * (1 + margem)
    y_max = y_max_dados * (1 + margem)

    posicao = ax.get_position()
    largura = fig.get_figwidth() * posicao.width
    altura = fig.get_figheight() * posicao.height
    proporcao_area = largura / altura

    if x_max / y_max > proporcao_area:
        y_max = x_max / proporcao_area
    else:
        x_max = y_max * proporcao_area

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)


def atualizar_grafico(_=None):
    """Chamada automaticamente sempre que QUALQUER slider (v0, theta, y0, g)
    é movido (via on_changed). Recalcula a trajetória e redesenha o gráfico
    e os resultados numéricos em tempo real, sem reiniciar o programa."""
    v0, theta, y0, g = ler_parametros()
    valido, msg = validar_entradas(v0, theta, y0, g)

    if not valido:
        # entrada inválida: mostra mensagem amigável, não trava nem lança erro
        texto_erro.set_text("⚠ " + msg)
        texto_resultados.set_text("")
        linha_trajetoria.set_data([], [])
        ponto_animado.set_data([], [])
        fig.canvas.draw_idle()
        return

    texto_erro.set_text("")
    t_voo, y_max, alcance = calcular_resultados(v0, theta, y0, g)
    _, x, y = calcular_trajetoria(v0, theta, y0, g)

    linha_trajetoria.set_data(x, y)
    ponto_animado.set_data([], [])

    texto_resultados.set_text(
        f"Alcance R    = {alcance:8.2f} m\n"
        f"Altura máx.  = {y_max:8.2f} m\n"
        f"Tempo de voo = {t_voo:8.2f} s"
    )

    ajustar_limites(x, y)

    fig.canvas.draw_idle()


def ao_mudar_planeta(label):
    """Callback do RadioButtons: ao escolher um planeta, atualiza o slider
    de gravidade, o que por sua vez dispara atualizar_grafico() automaticamente."""
    if label == "Personalizado":
        return

    _atualizando_planeta[0] = True
    try:
        slider_g.set_val(PLANETAS[label])
    finally:
        _atualizando_planeta[0] = False


def ao_mudar_gravidade(_valor):
    if not _atualizando_planeta[0] and radio_planetas.value_selected != "Personalizado":
        radio_planetas.set_active(INDICE_PERSONALIZADO)
    atualizar_grafico()


def animar_lancamento(_evento):
    """Callback do botão 'Lançar': anima um ponto vermelho se deslocando
    sobre a curva y(x) já calculada, conforme o tempo avança."""
    v0, theta, y0, g = ler_parametros()
    valido, msg = validar_entradas(v0, theta, y0, g)
    if not valido:
        texto_erro.set_text("⚠ " + msg)
        return

    t, x, y = calcular_trajetoria(v0, theta, y0, g, n_pontos=150)
    t_voo_final, y_max_final, alcance_final = calcular_resultados(v0, theta, y0, g)

    # se "manter trajetórias anteriores" estiver marcado, fixa a curva atual
    # no gráfico antes de iniciar a nova animação (requisito opcional)
    if check_sobrepor.get_status()[0]:
        curva_fixa, = ax.plot(x, y, lw=1.5, alpha=0.5)
        _curvas_anteriores.append(curva_fixa)

    # altura máxima já atingida "até agora" em cada instante da animação
    altura_maxima_ate_agora = np.maximum.accumulate(y)

    def quadro_da_animacao(i):
        ponto_animado.set_data([x[i]], [y[i]])
        # atualiza os resultados em tempo real, acompanhando a bolinha:
        # distância e tempo decorridos até aqui, e a maior altura já alcançada
        texto_resultados.set_text(
            f"Alcance R    = {x[i]:8.2f} m\n"
            f"Altura máx.  = {altura_maxima_ate_agora[i]:8.2f} m\n"
            f"Tempo de voo = {t[i]:8.2f} s"
        )
        # no último quadro, garante que os valores batam exatamente com as
        # fórmulas analíticas (evita pequenas diferenças de arredondamento)
        if i == len(x) - 1:
            texto_resultados.set_text(
                f"Alcance R    = {alcance_final:8.2f} m\n"
                f"Altura máx.  = {y_max_final:8.2f} m\n"
                f"Tempo de voo = {t_voo_final:8.2f} s"
            )
        return ponto_animado, texto_resultados

    # interrompe uma animação anterior ainda em execução, se houver.
    # Após uma animação com repeat=False terminar, o matplotlib pode
    # liberar o event_source internamente (ele vira None) — por isso
    # verificamos os dois antes de chamar stop().
    animacao_anterior = _animacao_atual[0]
    if animacao_anterior is not None and animacao_anterior.event_source is not None:
        animacao_anterior.event_source.stop()

    _animacao_atual[0] = FuncAnimation(
        fig, quadro_da_animacao, frames=len(x), interval=15, blit=True, repeat=False
    )
    fig.canvas.draw_idle()


def limpar_sobreposicoes(_evento):
    """Callback do botão 'Limpar': remove todas as trajetórias sobrepostas."""
    for curva in _curvas_anteriores:
        curva.remove()
    _curvas_anteriores.clear()
    atualizar_grafico()


# --- Liga cada widget à função chamada quando ele é alterado ---
slider_v0.on_changed(atualizar_grafico)
slider_theta.on_changed(atualizar_grafico)
slider_y0.on_changed(atualizar_grafico)
slider_g.on_changed(ao_mudar_gravidade)
radio_planetas.on_clicked(ao_mudar_planeta)
botao_lancar.on_clicked(animar_lancamento)
botao_limpar.on_clicked(limpar_sobreposicoes)

atualizar_grafico()  # desenha o estado inicial da interface
plt.show()