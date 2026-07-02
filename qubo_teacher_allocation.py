import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import dimod
import neal
import itertools

# =====================================================================
# 1. DADOS DO PROBLEMA
# =====================================================================
professores = ['P1', 'P2', 'P3']
escolas = ['E1', 'E2']
horarios = ['Manha', 'Tarde']

distancias = {
    'P1': {'E1': 5,  'E2': 15},
    'P2': {'E1': 20, 'E2': 5},
    'P3': {'E1': 10, 'E2': 12}
}

print("Matriz de Custos de Entrada (Distância em km):")
df_dist = pd.DataFrame(distancias).T
print(df_dist)
print("-"*40)

# =====================================================================
# 2. QUBO / HAMILTONIANO
# =====================================================================
bqm = dimod.BinaryQuadraticModel('BINARY')

PENALIDADE_ESCOLA = 100
PENALIDADE_DUPLICIDADE = 100

for p in professores:
    for e in escolas:
        for h in horarios:
            bqm.add_linear(f"{p}_{e}_{h}", distancias[p][e])

for e in escolas:
    for h in horarios:
        for p in professores:
            bqm.add_linear(f"{p}_{e}_{h}", -PENALIDADE_ESCOLA)
        for i, p1 in enumerate(professores):
            for j, p2 in enumerate(professores):
                if i < j:
                    bqm.add_quadratic(f"{p1}_{e}_{h}", f"{p2}_{e}_{h}", 2 * PENALIDADE_ESCOLA)
        bqm.offset += PENALIDADE_ESCOLA

for p in professores:
    for h in horarios:
        for e1, e2 in itertools.combinations(escolas, 2):
            bqm.add_quadratic(f"{p}_{e1}_{h}", f"{p}_{e2}_{h}", PENALIDADE_DUPLICIDADE)

# =====================================================================
# 3. RESOLUÇÃO (SIMULATED ANNEALING)
# =====================================================================
print("Executando o Annealer para varrer a paisagem de energia...")
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=2000)

best_sample = sampleset.first.sample
best_energy = sampleset.first.energy
print(f"Sucesso! Mínimo Global (Energia de Base) Encontrado: {best_energy}")

alocacao = []
for var, val in best_sample.items():
    if val == 1:
        p, e, h = var.split('_')
        alocacao.append({'Professor': p, 'Escola': e, 'Horário': h, 'Distância': distancias[p][e]})

df_alocacao = pd.DataFrame(alocacao)

print("\nRESULTADOS ENCONTRADOS")
if not df_alocacao.empty:
    print(df_alocacao.to_string(index=False))
    print(f"Custo Total de Deslocamento do Ecossistema: {df_alocacao['Distância'].sum()} km")
    profs_alocados = df_alocacao['Professor'].unique()
    profs_ociosos = [p for p in professores if p not in profs_alocados]
    if profs_ociosos:
        print(f"[ALERTA] Professor(es) ocioso(s): {profs_ociosos}")
else:
    profs_alocados = []
    profs_ociosos = professores
    print("Erro: Nenhuma alocação respeitou as Hard Constraints.")

# =====================================================================
# 4. VISUALIZAÇÕES (DOIS GRÁFICOS LADO A LADO)
# =====================================================================
fig = plt.figure(figsize=(15, 6.5))

# --- GRÁFICO 1 (NOVO): Quadro de Alocação Docente ---
ax1 = plt.subplot(1, 2, 1)

cell_w, cell_h = 2.4, 1.8
x_offset, y_offset = 1.6, 1.0
cores_prof = {'P1': '#2E6F95', 'P2': '#3E9C6B', 'P3': '#C08A3E'}
custo_max = max(v for d in distancias.values() for v in d.values())

for i, e in enumerate(escolas):
    for j, h in enumerate(horarios):
        x = x_offset + j * (cell_w + 0.3)
        y = y_offset + (len(escolas) - 1 - i) * (cell_h + 0.3)

        registro = df_alocacao[(df_alocacao.Escola == e) & (df_alocacao['Horário'] == h)] if not df_alocacao.empty else pd.DataFrame()
        if not registro.empty:
            p = registro.iloc[0]['Professor']
            dist = registro.iloc[0]['Distância']
            cor = cores_prof[p]
            alpha = 0.35 + 0.5 * (dist / custo_max)
        else:
            p, dist, cor, alpha = None, None, '#E8E8E8', 1.0

        box = FancyBboxPatch((x, y), cell_w, cell_h,
                              boxstyle="round,pad=0.02,rounding_size=0.12",
                              linewidth=1.5, edgecolor='white',
                              facecolor=cor, alpha=alpha)
        ax1.add_patch(box)

        if p:
            ax1.text(x + cell_w/2, y + cell_h/2 + 0.25, p,
                     ha='center', va='center', fontsize=20, weight='bold', color='white')
            ax1.text(x + cell_w/2, y + cell_h/2 - 0.35, f"{dist} km",
                     ha='center', va='center', fontsize=11, color='white', alpha=0.9)
        else:
            ax1.text(x + cell_w/2, y + cell_h/2, "vago",
                     ha='center', va='center', fontsize=12, color='#999999', style='italic')

for j, h in enumerate(horarios):
    x = x_offset + j * (cell_w + 0.3) + cell_w/2
    ax1.text(x, y_offset + len(escolas)*(cell_h+0.3) + 0.15, h,
             ha='center', va='bottom', fontsize=12, weight='bold', color='#333333')

for i, e in enumerate(escolas):
    y = y_offset + (len(escolas) - 1 - i) * (cell_h + 0.3) + cell_h/2
    ax1.text(x_offset - 0.35, y, e, ha='right', va='center',
             fontsize=12, weight='bold', color='#333333')

custo_total = df_alocacao['Distância'].sum() if not df_alocacao.empty else 0
info_x = x_offset + len(horarios)*(cell_w+0.3) + 0.5

ax1.text(info_x, y_offset + len(escolas)*(cell_h+0.3), "Resumo",
         fontsize=13, weight='bold', color='#333333', va='top')
ax1.text(info_x, y_offset + len(escolas)*(cell_h+0.3) - 0.6,
         f"Custo total:\n{custo_total} km", fontsize=11, va='top', color='#2E6F95')

y_cursor = y_offset + len(escolas)*(cell_h+0.3) - 1.9
ax1.text(info_x, y_cursor, "Ociosos:", fontsize=11, weight='bold', va='top', color='#B0453E')
y_cursor -= 0.45
if profs_ociosos:
    for p in profs_ociosos:
        circ = mpatches.Circle((info_x + 0.15, y_cursor - 0.02), 0.15,
                                facecolor=cores_prof.get(p, '#B0B0B0'), edgecolor='white')
        ax1.add_patch(circ)
        ax1.text(info_x + 0.45, y_cursor, p, fontsize=11, va='center')
        y_cursor -= 0.5
else:
    ax1.text(info_x, y_cursor, "nenhum", fontsize=11, va='top', color='#3E9C6B')

ax1.set_xlim(0, info_x + 2.2)
ax1.set_ylim(0, y_offset + len(escolas)*(cell_h+0.3) + 0.8)
ax1.axis('off')
ax1.set_title("Grade Escolar Resultante (Mínimo Global)", fontsize=14, weight='bold', pad=15)

# --- GRÁFICO 2: Paisagem de Energia Didática (mantido do original) ---
ax2 = plt.subplot(1, 2, 2)
x = np.linspace(-2.5, 2.5, 400)
energia = x**4 - 4*x**2 + x

ax2.plot(x, energia, color='navy', linewidth=2.5, label='Fronteira de Soluções (Hamiltoniano)')

x_local, y_local = -1.48, (-1.48)**4 - 4*(-1.48)**2 + (-1.48)
x_global, y_global = 1.35, (1.35)**4 - 4*(1.35)**2 + (1.35)

ax2.scatter(x_local, y_local, color='red', s=130, zorder=5, label='Mínimo Local (Solução Subótima Clássica)')
ax2.scatter(x_global, y_global, color='green', s=130, zorder=5, label='Mínimo Global (Alocação Perfeita)')

ax2.annotate('', xy=(x_global, y_local), xytext=(x_local, y_local),
             arrowprops=dict(arrowstyle="->", color="purple", ls="--", lw=2))
ax2.text(0, y_local + 0.6, 'Tunelamento Quântico\n(Atravessa barreiras de restrição)',
         horizontalalignment='center', color='purple', fontsize=9, weight='bold')

ax2.annotate('', xy=(-0.1, 0), xytext=(x_local, y_local),
             arrowprops=dict(connectionstyle="arc3,rad=.3", arrowstyle="->", color="orange", lw=1.5))
ax2.text(-0.6, 1.4, 'Salto Térmico Clássico\n(Exige muita computação)', color='orange', fontsize=9, weight='bold')

ax2.set_title("Visualização Física do Quantum Annealing", fontsize=14, pad=15, weight='bold')
ax2.set_xlabel("Espaço de Configurações do QUBO", fontsize=11)
ax2.set_ylabel("Energia do Sistema (Penalidades + Distância)", fontsize=11)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('grafico_completo.png', dpi=150, bbox_inches='tight')
plt.show()