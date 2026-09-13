"""
Simulador de Lançamento de Projéteis - Interface Gráfica Interativa
=====================================================================

Disciplina : Física - Práticas Extensivas
Atividade  : Lançamento de Projéteis (sem resistência do ar)

INSTRUÇÕES DE EXECUÇÃO
-----------------------
Bibliotecas necessárias:
    pip install numpy
    pip install matplotlib

Em caso de erro ao instalar, tente:
    py -m pip install numpy
    py -m pip install matplotlib

Para executar:
    python projectile.py

Biblioteca de interface utilizada: matplotlib.widgets
(Slider, Button, RadioButtons, CheckButtons, TextBox)
Escolhida por não exigir dependências externas além do matplotlib,
que já é usado para plotar o gráfico da trajetória.
"""
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# 1. FÍSICA: funções que calculam a trajetória e os resultados do lançamento
# ---------------------------------------------------------------------------

def calcular_resultados(v0, theta_graus, y0, g):
    """Calcula tempo de voo, altura máxima e alcance."""
    theta = np.radians(theta_graus)
    vy0 = v0 * np.sin(theta)

    discriminante = vy0**2 + 2 * g * y0
    t_voo = (vy0 + np.sqrt(discriminante)) / g

    y_max = y0 + (vy0**2) / (2 * g)
    alcance = v0 * np.cos(theta) * t_voo

    return t_voo, y_max, alcance


def calcular_trajetoria(v0, theta_graus, y0, g, n_pontos=300):
    """Retorna os vetores t, x(t) e y(t) da trajetória."""
    theta = np.radians(theta_graus)
    t_voo, _, _ = calcular_resultados(v0, theta_graus, y0, g)

    t = np.linspace(0, t_voo, n_pontos)
    x = v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t**2
    y = np.clip(y, 0, None)

    return t, x, y


def validar_entradas(v0, theta_graus, y0, g):
    """Valida os parâmetros físicos do lançamento."""
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
# 2. PLANETAS PRÉ-CONFIGURADOS
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
# 3. INTERFACE GRÁFICA
# ---------------------------------------------------------------------------

COR_FUNDO = "#F4F7FB"
COR_PAINEL = "#FFFFFF"
COR_GRAFICO = "#FFFFFF"
COR_PRIMARIA = "#2563EB"
COR_PRIMARIA_HOVER = "#1D4ED8"
COR_TEXTO = "#172033"
COR_TEXTO_SECUNDARIO = "#526078"
COR_BORDA = "#CBD5E1"
COR_GRID = "#DCE3EC"
COR_TRAJETORIA = "#2563EB"
COR_PROJETIL = "#EF4444"
COR_ERRO = "#B91C1C"
COR_BOTAO_SECUNDARIO = "#E8EDF4"
COR_BOTAO_SECUNDARIO_HOVER = "#D8E0EA"

CORES_TRAJETORIAS_ANTERIORES = [
    "#F59E0B",
    "#10B981",
    "#8B5CF6",
    "#EC4899",
    "#06B6D4",
    "#F97316",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
})

fig = plt.figure(figsize=(11.5, 7.6), facecolor=COR_FUNDO)

fig.text(
    0.075, 0.952,
    "Simulador de Lançamento de Projéteis",
    fontsize=18,
    fontweight="bold",
    color=COR_TEXTO,
    va="top"
)

fig.text(
    0.075, 0.915,
    "Explore como velocidade, ângulo, altura inicial e gravidade alteram a trajetória.",
    fontsize=10,
    color=COR_TEXTO_SECUNDARIO,
    va="top"
)

ax = fig.add_axes([0.075, 0.395, 0.675, 0.445], facecolor=COR_GRAFICO)

linha_trajetoria, = ax.plot(
    [], [],
    lw=2.8,
    color=COR_TRAJETORIA,
    solid_capstyle="round"
)

ponto_animado, = ax.plot(
    [], [],
    "o",
    color=COR_PROJETIL,
    markersize=9,
    markeredgecolor="white",
    markeredgewidth=1.3,
    zorder=5
)

ax.set_xlabel("Distância horizontal x (m)", color=COR_TEXTO_SECUNDARIO, labelpad=10)
ax.set_ylabel("Altura y (m)", color=COR_TEXTO_SECUNDARIO, labelpad=10)
ax.set_title(
    "Trajetória do projétil",
    loc="left",
    color=COR_TEXTO,
    fontweight="bold",
    pad=10
)

ax.grid(True, color=COR_GRID, linewidth=0.8, linestyle="--", alpha=0.95)
ax.set_axisbelow(True)
ax.set_aspect("equal", adjustable="box")

for spine in ax.spines.values():
    spine.set_color(COR_BORDA)
    spine.set_linewidth(1.0)

ax.tick_params(colors=COR_TEXTO_SECUNDARIO)

texto_resultados = ax.text(
    0.025,
    0.965,
    "",
    transform=ax.transAxes,
    fontsize=10,
    va="top",
    ha="left",
    color=COR_TEXTO,
    family="monospace",
    linespacing=1.45,
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        alpha=0.96,
        edgecolor=COR_BORDA,
        linewidth=1.0
    ),
)

texto_erro = fig.text(
    0.79,
    0.455,
    "",
    fontsize=9.5,
    color=COR_ERRO,
    va="top",
    ha="left",
    wrap=True
)

fig.text(
    0.79,
    0.842,
    "AMBIENTE",
    fontsize=9,
    fontweight="bold",
    color=COR_TEXTO_SECUNDARIO
)

eixo_planetas = fig.add_axes([0.79, 0.565, 0.17, 0.245], facecolor=COR_PAINEL)

for spine in eixo_planetas.spines.values():
    spine.set_color(COR_BORDA)

radio_planetas = RadioButtons(
    eixo_planetas,
    OPCOES_GRAVIDADE,
    active=2,
    activecolor=COR_PRIMARIA
)

for texto in radio_planetas.labels:
    texto.set_color(COR_TEXTO)
    texto.set_fontsize(10)

fig.text(
    0.075,
    0.345,
    "PARÂMETROS DO LANÇAMENTO",
    fontsize=9,
    fontweight="bold",
    color=COR_TEXTO_SECUNDARIO
)

# --- Sliders ---
eixo_v0 = fig.add_axes([0.14, 0.288, 0.43, 0.028], facecolor=COR_FUNDO)
eixo_theta = fig.add_axes([0.14, 0.238, 0.43, 0.028], facecolor=COR_FUNDO)
eixo_y0 = fig.add_axes([0.14, 0.188, 0.43, 0.028], facecolor=COR_FUNDO)
eixo_g = fig.add_axes([0.14, 0.138, 0.43, 0.028], facecolor=COR_FUNDO)

slider_v0 = Slider(
    eixo_v0, "v0 (m/s)", 5, 150,
    valinit=40, valfmt="%1.1f", color=COR_PRIMARIA
)

slider_theta = Slider(
    eixo_theta, "θ (graus)", 1, 89,
    valinit=45, valfmt="%1.1f°", color=COR_PRIMARIA
)

slider_y0 = Slider(
    eixo_y0, "y0 (m)", 0, 50,
    valinit=0, valfmt="%1.1f", color=COR_PRIMARIA
)

slider_g = Slider(
    eixo_g, "g (m/s²)", 1.6, 24.8,
    valinit=9.81, valfmt="%1.2f", color=COR_PRIMARIA
)

for slider in (slider_v0, slider_theta, slider_y0, slider_g):
    slider.label.set_color(COR_TEXTO)
    slider.label.set_fontweight("bold")
    slider.valtext.set_visible(False)

# --- Campos numéricos editáveis ---
eixo_input_v0 = fig.add_axes([0.595, 0.281, 0.105, 0.041])
eixo_input_theta = fig.add_axes([0.595, 0.231, 0.105, 0.041])
eixo_input_y0 = fig.add_axes([0.595, 0.181, 0.105, 0.041])
eixo_input_g = fig.add_axes([0.595, 0.131, 0.105, 0.041])

input_v0 = TextBox(
    eixo_input_v0, "",
    initial="40.0",
    color=COR_PAINEL,
    hovercolor="#F8FAFC",
    textalignment="center"
)

input_theta = TextBox(
    eixo_input_theta, "",
    initial="45.0",
    color=COR_PAINEL,
    hovercolor="#F8FAFC",
    textalignment="center"
)

input_y0 = TextBox(
    eixo_input_y0, "",
    initial="0.0",
    color=COR_PAINEL,
    hovercolor="#F8FAFC",
    textalignment="center"
)

input_g = TextBox(
    eixo_input_g, "",
    initial="9.81",
    color=COR_PAINEL,
    hovercolor="#F8FAFC",
    textalignment="center"
)

for eixo_input in (eixo_input_v0, eixo_input_theta, eixo_input_y0, eixo_input_g):
    for spine in eixo_input.spines.values():
        spine.set_color(COR_BORDA)
        spine.set_linewidth(1.0)

for campo in (input_v0, input_theta, input_y0, input_g):
    campo.text_disp.set_color(COR_TEXTO)
    campo.text_disp.set_fontweight("bold")

# --- Checkbox maior ---
eixo_check = fig.add_axes([0.075, 0.042, 0.32, 0.072], facecolor=COR_FUNDO)

for spine in eixo_check.spines.values():
    spine.set_visible(False)

check_sobrepor = CheckButtons(
    eixo_check,
    ["Manter trajetórias anteriores"],
    [False],
    label_props={
        "fontsize": [10.5],
        "color": [COR_TEXTO],
        "fontweight": ["medium"],
    },
    frame_props={
        "s": 115,
        "edgecolor": COR_TEXTO_SECUNDARIO,
        "linewidth": 1.4,
    },
    check_props={
        "s": 115,
        "color": COR_PRIMARIA,
        "linewidth": 2.0,
    }
)

eixo_lancar = fig.add_axes([0.405, 0.052, 0.14, 0.055])

botao_lancar = Button(
    eixo_lancar,
    "Lançar",
    color=COR_PRIMARIA,
    hovercolor=COR_PRIMARIA_HOVER
)

botao_lancar.label.set_color("white")
botao_lancar.label.set_fontweight("bold")

for spine in eixo_lancar.spines.values():
    spine.set_color(COR_PRIMARIA)

eixo_limpar = fig.add_axes([0.565, 0.052, 0.14, 0.055])

botao_limpar = Button(
    eixo_limpar,
    "Limpar",
    color=COR_BOTAO_SECUNDARIO,
    hovercolor=COR_BOTAO_SECUNDARIO_HOVER
)

botao_limpar.label.set_color(COR_TEXTO)
botao_limpar.label.set_fontweight("bold")

for spine in eixo_limpar.spines.values():
    spine.set_color(COR_BORDA)

_animacao_atual = [None]
_curvas_anteriores = []
_atualizando_planeta = [False]
_sincronizando_campos = [False]
_indice_cor_trajetoria = [0]


def ler_parametros():
    """Lê os valores atuais dos sliders."""
    return slider_v0.val, slider_theta.val, slider_y0.val, slider_g.val


def definir_texto_sem_evento(campo, texto):
    evento_anterior = campo.eventson
    campo.eventson = False
    campo.set_val(texto)
    campo.eventson = evento_anterior


def sincronizar_campos_com_sliders():
    if _sincronizando_campos[0]:
        return

    _sincronizando_campos[0] = True

    try:
        definir_texto_sem_evento(input_v0, f"{slider_v0.val:.1f}")
        definir_texto_sem_evento(input_theta, f"{slider_theta.val:.1f}")
        definir_texto_sem_evento(input_y0, f"{slider_y0.val:.1f}")
        definir_texto_sem_evento(input_g, f"{slider_g.val:.2f}")
    finally:
        _sincronizando_campos[0] = False


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


def atualizar_resultados(alcance, y_max, t_voo):
    texto_resultados.set_text(
        f"Alcance R    = {alcance:8.2f} m\n"
        f"Altura máx.  = {y_max:8.2f} m\n"
        f"Tempo de voo = {t_voo:8.2f} s"
    )


def atualizar_grafico(_=None):
    """Atualiza a trajetória e os resultados em tempo real."""

    parar_animacao_atual()

    sincronizar_campos_com_sliders()

    v0, theta, y0, g = ler_parametros()
    valido, msg = validar_entradas(v0, theta, y0, g)

    if not valido:
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

    atualizar_resultados(alcance, y_max, t_voo)
    ajustar_limites(x, y)

    fig.canvas.draw_idle()


def aplicar_valor_digitado(texto, slider, campo, nome, casas_decimais=1):
    if _sincronizando_campos[0]:
        return

    try:
        valor = float(texto.replace(",", "."))
    except ValueError:
        texto_erro.set_text(f"⚠ {nome}: informe um número válido.")
        formato = f"{{:.{casas_decimais}f}}"
        definir_texto_sem_evento(campo, formato.format(slider.val))
        fig.canvas.draw_idle()
        return

    if not slider.valmin <= valor <= slider.valmax:
        texto_erro.set_text(
            f"⚠ {nome}: informe um valor entre "
            f"{slider.valmin:g} e {slider.valmax:g}."
        )
        formato = f"{{:.{casas_decimais}f}}"
        definir_texto_sem_evento(campo, formato.format(slider.val))
        fig.canvas.draw_idle()
        return

    texto_erro.set_text("")
    slider.set_val(valor)


def ao_mudar_planeta(label):
    """Atualiza a gravidade ao selecionar um planeta."""
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

def parar_animacao_atual():
    animacao = _animacao_atual[0]
    if animacao is not None and animacao.event_source is not None:
        animacao.event_source.stop()
    _animacao_atual[0] = None

def animar_lancamento(_evento):
    """Anima o projétil ao longo da trajetória."""
    v0, theta, y0, g = ler_parametros()
    valido, msg = validar_entradas(v0, theta, y0, g)

    if not valido:
        texto_erro.set_text("⚠ " + msg)
        return

    n_pontos = 300
    t, x, y = calcular_trajetoria(v0, theta, y0, g, n_pontos=n_pontos)
    t_voo_final, y_max_final, alcance_final = calcular_resultados(v0, theta, y0, g)

    escala_tempo = 1.0
    
    if check_sobrepor.get_status()[0]:
        cor = CORES_TRAJETORIAS_ANTERIORES[
            _indice_cor_trajetoria[0] % len(CORES_TRAJETORIAS_ANTERIORES)
        ]
        _indice_cor_trajetoria[0] += 1

        curva_fixa, = ax.plot(
            x,
            y,
            lw=2.0,
            alpha=0.78,
            color=cor,
            linestyle="--",
            dashes=(5, 3)
        )

        _curvas_anteriores.append(curva_fixa)
        ajustar_limites(x, y)

    altura_maxima_ate_agora = np.maximum.accumulate(y)

    def quadro_da_animacao(i):
        tempo_decorrido_real = time.perf_counter() - tempo_inicio
        t_simulado = tempo_decorrido_real * escala_tempo

        chegou_ao_fim = t_simulado >= t_voo_final
        if chegou_ao_fim:
            t_simulado = t_voo_final

        if t_simulado >= t_voo_final:
            t_simulado = t_voo_final
            if _animacao_atual[0] is not None and _animacao_atual[0].event_source is not None:
                _animacao_atual[0].event_source.stop()

        # interpola x, y no instante t_simulado, em vez de indexar por frame
        x_atual = np.interp(t_simulado, t, x)
        y_atual = np.interp(t_simulado, t, y)
        altura_max_atual = np.interp(t_simulado, t, altura_maxima_ate_agora)

        ponto_animado.set_data([x_atual], [y_atual])

        if chegou_ao_fim:
            atualizar_resultados(
                alcance_final,
                y_max_final,
                t_voo_final
            )
            if _animacao_atual[0] is not None and _animacao_atual[0].event_source is not None:
                _animacao_atual[0].event_source.stop()
        else:
            atualizar_resultados(x_atual, altura_max_atual, t_simulado)

        if i == len(x) - 1:
            atualizar_resultados(
                alcance_final,
                y_max_final,
                t_voo_final
            )

        return ponto_animado, texto_resultados

    parar_animacao_atual()
    tempo_inicio = time.perf_counter()

    _animacao_atual[0] = FuncAnimation(
        fig,
        quadro_da_animacao,
        frames=itertools.count(),
        interval=15,
        blit=False,
        repeat=False,
        cache_frame_data=False
    )

    fig.canvas.draw_idle()


def limpar_sobreposicoes(_evento):
    """Remove as trajetórias sobrepostas."""

    parar_animacao_atual()

    for curva in _curvas_anteriores:
        curva.remove()

    _curvas_anteriores.clear()
    _indice_cor_trajetoria[0] = 0
    atualizar_grafico()


slider_v0.on_changed(atualizar_grafico)
slider_theta.on_changed(atualizar_grafico)
slider_y0.on_changed(atualizar_grafico)
slider_g.on_changed(ao_mudar_gravidade)

input_v0.on_submit(
    lambda texto: aplicar_valor_digitado(
        texto, slider_v0, input_v0, "Velocidade inicial", 1
    )
)

input_theta.on_submit(
    lambda texto: aplicar_valor_digitado(
        texto, slider_theta, input_theta, "Ângulo", 1
    )
)

input_y0.on_submit(
    lambda texto: aplicar_valor_digitado(
        texto, slider_y0, input_y0, "Altura inicial", 1
    )
)

input_g.on_submit(
    lambda texto: aplicar_valor_digitado(
        texto, slider_g, input_g, "Gravidade", 2
    )
)

radio_planetas.on_clicked(ao_mudar_planeta)
botao_lancar.on_clicked(animar_lancamento)
botao_limpar.on_clicked(limpar_sobreposicoes)

atualizar_grafico()
plt.show()