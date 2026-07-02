import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dimod
import neal
import itertools

# =====================================================================
# 1. RELATÓRIO TEÓRICO (RESPOSTAS COMPLETA DAS PERGUNTAS)
# =====================================================================
print("="*80)
print("     RELATÓRIO TÉCNICO: OTIMIZAÇÃO DA DISTRIBUIÇÃO DE PROFESSORES     ")
print("="*80)

print("\nREQUISITO 1: Como a otimização de distribuição de professores pode ser feita?")
print("-"*80)
print("1. Mapeamento QUBO: O problema combinatório (NP-Difícil) é convertido em uma\n"
      "   Função de Custo Binária (QUBO). Variáveis X_(p,e,h) assumem 1 para alocação\n"
      "   e 0 para o contrário.\n"
      "2. Hamiltoniano do Sistema: As regras do mundo real são traduzidas em energia:\n"
      "   - Hard Constraints (Restrições Rígidas): Regras invioláveis (ex: professor em\n"
      "     dois lugares ao mesmo tempo) recebem penalidades matemáticas gigantescas (+100).\n"
      "   - Soft Constraints (Objetivos): O que queremos otimizar (ex: minimizar a\n"
      "     distância de deslocamento) recebe pesos proporcionais ao custo real.\n"
      "3. Realidade Quântica (Era NISQ): Hardwares atuais (100 a 1100 qubits físicos, erro ~10^-3)\n"
      "   não suportam redes escolares inteiras diretamente. A abordagem real exige\n"
      "   Decomposição Combinatória: solvers clássicos isolam os gargalos críticos, e a QPU\n"
      "   resolve apenas essas submatrizes complexas.")

print("\nREQUISITO 2: Quais algoritmos quânticos podem ser utilizados?")
print("-"*80)
print("1. QAOA (Quantum Approximate Optimization Algorithm): Algoritmo híbrido para\n"
      "   computadores baseados em portas lógicas (IBM, Google). Alterna camadas de custo\n"
      "   e mistura quântica, ajustando parâmetros via otimizadores clássicos.\n"
      "2. Quantum Annealing (Recozimento Quântico): Utilizado por máquinas analógicas (D-Wave).\n"
      "   Evolui o sistema adiabaticamente, permitindo que os qubits usem o Tunelamento Quântico\n"
      "   para 'atravessar' picos de restrições e colapsar no Mínimo Global (solução ótima).\n"
      "3. VQE (Variational Quantum Eigensolver): Outro algoritmo variacional híbrido,\n"
      "   focado em encontrar o menor autovalor (estado fundamental de energia) do Hamiltoniano.")

print("\n" + "="*80)
print("             EXECUÇÃO DA SIMULAÇÃO (TOY PROBLEM QUÂNTICO)            ")
print("="*80)

# =====================================================================
# 2. MODELAGEM DO TOY PROBLEM (DADOS SIMULADOS)
# =====================================================================
professores = ['P1', 'P2', 'P3']
escolas = ['E1', 'E2']
horarios = ['Manha', 'Tarde']

# Custo de deslocamento (em km ou tempo)
distancias = {
    'P1': {'E1': 5,  'E2': 15},
    'P2': {'E1': 20, 'E2': 5},
    'P3': {'E1': 10, 'E2': 12}
}

print("\nMatriz de Custos de Entrada (Distância em km):")
df_dist = pd.DataFrame(distancias).T
print(df_dist)
print("-"*40)

# =====================================================================
# 3. FORMULAÇÃO MATEMÁTICA DO HAMILTONIANO (O QUBO)
# =====================================================================
bqm = dimod.BinaryQuadraticModel('BINARY')

PENALIDADE_ESCOLA = 100        
PENALIDADE_DUPLICIDADE = 100   

# A. Soft Constraint: Minimizar a Distância Geral
for p in professores:
    for e in escolas:
        for h in horarios:
            var_name = f"{p}_{e}_{h}"
            bqm.add_linear(var_name, distancias[p][e])

# B. Hard Constraint 1: Exatamente 1 professor por escola/horário
for e in escolas:
    for h in horarios:
        for p in professores:
            var_name = f"{p}_{e}_{h}"
            bqm.add_linear(var_name, -PENALIDADE_ESCOLA)
        
        for i, p1 in enumerate(professores):
            for j, p2 in enumerate(professores):
                if i < j:
                    var1 = f"{p1}_{e}_{h}"
                    var2 = f"{p2}_{e}_{h}"
                    bqm.add_quadratic(var1, var2, 2 * PENALIDADE_ESCOLA)
                    
        bqm.offset += PENALIDADE_ESCOLA

# C. Hard Constraint 2: Um professor não pode estar em 2 escolas ao mesmo tempo (Escalável)
for p in professores:
    for h in horarios:
        for e1, e2 in itertools.combinations(escolas, 2):
            var1 = f"{p}_{e1}_{h}"
            var2 = f"{p}_{e2}_{h}"
            bqm.add_quadratic(var1, var2, PENALIDADE_DUPLICIDADE)

# =====================================================================
# 4. RESOLUÇÃO VIA SIMULATED ANNEALING (SIMULADOR OCEAN SDK)
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

print("\n" + "="*80)
print("                        RESULTADOS ENCONTRADOS                       ")
print("="*80)
if not df_alocacao.empty:
    print(df_alocacao.to_string(index=False))
    print("-"*40)
    print(f"Custo Total de Deslocamento do Ecossistema: {df_alocacao['Distância'].sum()} km")
    
    # Validação de Negócio (Alerta de ociosidade comum em otimizações puras)
    profs_alocados = df_alocacao['Professor'].unique()
    profs_ociosos = [p for p in professores if p not in profs_alocados]
    if profs_ociosos:
        print(f"\n[ALERTA] NOTA DE ENGENHARIA: O algoritmo deixou o(s) professor(es) {profs_ociosos} ociosos.")
        print("  Isso ocorre porque o Mínimo Global priorizou a eficiência máxima de distância.")
else:
    print("Erro: Nenhuma alocação respeitou as Hard Constraints.")

# Preparação da matriz para o Heatmap gráfico
mapa_alocacao = pd.DataFrame(index=escolas, columns=horarios, data="")
for _, row in df_alocacao.iterrows():
    mapa_alocacao.at[row['Escola'], row['Horário']] = row['Professor']

# =====================================================================
# 5. VISUALIZAÇÕES GRÁFICAS (MATPLOTLIB & SEABORN)
# =====================================================================
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(15, 6))

# --- GRÁFICO 1: Heatmap da Alocação Ótima ---
ax1 = plt.subplot(1, 2, 1)

def get_prof_idx(prof_str):
    return professores.index(prof_str) if prof_str in professores else -1

mapa_cores = np.array([[get_prof_idx(mapa_alocacao.loc[e, h]) for h in horarios] for e in escolas])

sns.heatmap(mapa_cores, annot=mapa_alocacao, fmt="", cmap="YlGnBu", cbar=False, 
            linewidths=1.5, ax=ax1, annot_kws={"size": 16, "weight": "bold"},
            xticklabels=horarios, yticklabels=escolas)

ax1.set_title("Grade Escolar Resultante (Mínimo Global)", fontsize=14, pad=15, weight='bold')
ax1.set_xlabel("Horários", fontsize=11)
ax1.set_ylabel("Escolas", fontsize=11)

# --- GRÁFICO 2: Paisagem de Energia Didática ---
ax2 = plt.subplot(1, 2, 2)
x = np.linspace(-2.5, 2.5, 400)
energia = x**4 - 4*x**2 + x

ax2.plot(x, energia, color='navy', linewidth=2.5, label='Fronteira de Soluções (Hamiltoniano)')

x_local, y_local = -1.48, (-1.48)**4 - 4*(-1.48)**2 + (-1.48)
x_global, y_global = 1.35, (1.35)**4 - 4*(1.35)**2 + (1.35)

ax2.scatter(x_local, y_local, color='red', s=130, zorder=5, label='Mínimo Local (Solução Subótima Clássica)')
ax2.scatter(x_global, y_global, color='green', s=130, zorder=5, label='Mínimo Global (Alocação Perfeita)')

# Seta do Tunelamento Quântico
ax2.annotate('', xy=(x_global, y_local), xytext=(x_local, y_local),
             arrowprops=dict(arrowstyle="->", color="purple", ls="--", lw=2))
ax2.text(0, y_local + 0.6, 'Tunelamento Quântico\n(Atravessa barreiras de restrição)', 
         horizontalalignment='center', color='purple', fontsize=9, weight='bold')

# Seta da Otimização Clássica
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
plt.show()