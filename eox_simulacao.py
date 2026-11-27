#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eox_simulacao.py
================
Simulação de Eventos Discretos (DES) da linha de montagem de elevadores EOX.

O script roda o modelo em simpy para 4 cenários, calcula os KPIs, executa as
verificações de V&V e grava um único arquivo HTML autocontido
(``eox_simulacao.html``) com:
  * layout fabril animado em Canvas 2D (replay do log de eventos da simulação);
  * gráficos temporais (Plotly via CDN);
  * tabela comparativa de KPIs;
  * seção de Verificação & Validação.

Uso
---
    pip install simpy numpy
    python eox_simulacao.py
    # abre eox_simulacao.html no navegador

O arquivo é gravado no diretório corrente. Não há caminhos absolutos nem dados
de ambiente no código.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import simpy

# =============================================================================
# 1. PARÂMETROS GLOBAIS
# =============================================================================
JORNADA = 480.0               # min de trabalho por dia (1 turno)
TAKT_BASE = 48.0              # min/elevador  -> 480/48 = 10 elevadores/dia
META_DIARIA = 10              # meta de produção (elevadores/dia)
DIAS_AQUECIMENTO = 3          # descartados das estatísticas
DIAS_MEDICAO = 18             # dias úteis medidos
T_AQUEC = DIAS_AQUECIMENTO * JORNADA
T_FIM = (DIAS_AQUECIMENTO + DIAS_MEDICAO) * JORNADA
JANELA = T_FIM - T_AQUEC      # 8.640 min de medição

SEED_PRINCIPAL = 42
SEEDS_REPLICAS = (42, 43, 44, 45, 46)   # 5 réplicas para o IC 95 %
T_STUDENT_975_GL4 = 2.776               # t(0,975; 4 g.l.)
# Estudo de convergência do AS-IS: 40 réplicas independentes (seeds 42…81)
SEEDS_CONVERGENCIA = tuple(range(42, 82))

# Quebras do BEOX-05 (AS-IS): processo de Poisson durante a operação
MTBF_05 = 387.0               # min (tempo médio entre falhas)
MTTR_05 = 44.35               # min (tempo médio de reparo, exponencial)

# Premissa: o prompt informa "36 % de chance de parada adicional" no BEOX-04,
# mas não a duração. Adotou-se Exponencial com média de 8 min (editável).
PARADA_EXTRA_BEOX04 = 8.0

# Modo "saturado": chegadas a 75 % do takt para medir a CAPACIDADE do sistema
# (sempre há trabalho na frente do gargalo, então a saída = capacidade).
FATOR_SATURACAO = 0.75

AMOSTRAGEM_SERIES = 10.0      # min entre pontos das séries temporais
N_AMOSTRAS_VV = 20_000        # amostras por distribuição na validação
ARQUIVO_SAIDA = "eox_simulacao.html"


def br(x: float, d: int = 2) -> str:
    """Formata número com vírgula decimal (padrão brasileiro)."""
    return f"{x:.{d}f}".replace(".", ",")


def g_br(x: float) -> str:
    return f"{x:g}".replace(".", ",")


# =============================================================================
# 2. DISTRIBUIÇÕES DE TEMPO DE PROCESSAMENTO
# -----------------------------------------------------------------------------
# Critério geral de escolha (CV = desvio-padrão / média):
#   * Tempos de tarefa são estritamente positivos e assimétricos à direita
#     (algo pode demorar muito mais que a média, nunca "muito menos que zero").
#     Por isso a LOGNORMAL é o padrão para tarefas manuais com CV entre 0,2 e
#     0,8: não gera valores negativos e reproduz a cauda longa.
#   * Com CV baixo (< 0,2) e comportamento simétrico, a NORMAL é aceitável:
#     P(X < 0) é desprezível (z < -5). Ela é truncada em zero por segurança.
#   * A GAMMA modela uma operação composta por várias subetapas em série
#     (soma de exponenciais -> Erlang/Gamma). Com forma k = (média/σ)^2 ≈ 20
#     ela equivale a ~20 subetapas: adequada ao posto gargalo BEOX-05, que
#     concentra muitas tarefas de montagem de painéis.
#   * OCORRÊNCIAS (falta de peça, retrabalho, ajuste) são eventos raros, de
#     duração curta e sem memória: modeladas como Bernoulli(p) x Exponencial.
#   * QUEBRAS do BEOX-05 são tratadas à parte, como processo de Poisson
#     (inter-falha exponencial, MTBF) com reparo exponencial (MTTR).
# =============================================================================
@dataclass(frozen=True)
class Distribuicao:
    codigo: str
    base: str            # 'lognormal' | 'normal' | 'gamma'
    media: float         # média da parcela base
    dp: float            # desvio-padrão da parcela base
    p_extra: float = 0.0     # probabilidade de ocorrência adicional
    media_extra: float = 0.0  # média (exponencial) da ocorrência
    const: float = 0.0       # parcela constante (ex.: setup)
    justificativa: str = ""

    # ---- parâmetros derivados -------------------------------------------
    @property
    def cv(self) -> float:
        return self.dp / self.media

    def _lognormal_mu_sigma(self) -> tuple[float, float]:
        s2 = math.log(1.0 + (self.dp / self.media) ** 2)
        return math.log(self.media) - s2 / 2.0, math.sqrt(s2)

    @property
    def media_teorica(self) -> float:
        return self.media + self.p_extra * self.media_extra + self.const

    @property
    def dp_teorico(self) -> float:
        # Var(B·E) com B~Bernoulli(p), E~Exp(m): p·2m² − (p·m)²
        var_extra = self.p_extra * 2 * self.media_extra ** 2 - (self.p_extra * self.media_extra) ** 2
        return math.sqrt(self.dp ** 2 + var_extra)

    @property
    def descricao(self) -> str:
        nomes = {"lognormal": "Lognormal", "normal": "Normal", "gamma": "Gamma"}
        txt = f"{nomes[self.base]}(μ={g_br(self.media)}; σ={g_br(self.dp)})"
        if self.p_extra >= 1.0:
            txt += f" + Exp({g_br(self.media_extra)})"
        elif self.p_extra > 0:
            txt += f" + {g_br(self.p_extra * 100)}%·Exp({g_br(self.media_extra)})"
        if self.const:
            txt += f" + {g_br(self.const)}"
        return txt

    # ---- amostragem -------------------------------------------------------
    def amostrar(self, rng: np.random.Generator) -> float:
        if self.base == "lognormal":
            mu, s = self._lognormal_mu_sigma()
            x = rng.lognormal(mu, s)
        elif self.base == "normal":
            x = rng.normal(self.media, self.dp)
            while x <= 0.0:                     # truncamento em zero
                x = rng.normal(self.media, self.dp)
        elif self.base == "gamma":
            k = (self.media / self.dp) ** 2
            theta = self.dp ** 2 / self.media
            x = rng.gamma(k, theta)
        else:
            raise ValueError(self.base)
        if self.p_extra > 0 and rng.random() < self.p_extra:
            x += rng.exponential(self.media_extra)
        return x + self.const

    # ---- densidade teórica (para a validação) ----------------------------
    def _pdf_base(self, x: np.ndarray) -> np.ndarray:
        if self.base == "lognormal":
            mu, s = self._lognormal_mu_sigma()
            return np.exp(-(np.log(x) - mu) ** 2 / (2 * s * s)) / (x * s * math.sqrt(2 * math.pi))
        if self.base == "normal":
            return np.exp(-(x - self.media) ** 2 / (2 * self.dp ** 2)) / (self.dp * math.sqrt(2 * math.pi))
        k = (self.media / self.dp) ** 2
        theta = self.dp ** 2 / self.media
        return np.exp((k - 1) * np.log(x) - x / theta - math.lgamma(k) - k * math.log(theta))

    def pdf_teorica(self, dx: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Densidade da soma base + Bernoulli·Exp + const, por convolução numérica."""
        xmax = self.media_teorica + 12 * self.dp_teorico
        x = np.arange(0.0, xmax, dx) + dx / 2
        f = self._pdf_base(x)
        if self.p_extra > 0:
            fe = np.exp(-x / self.media_extra) / self.media_extra
            fe /= fe.sum() * dx
            conv = np.convolve(f, fe)[: len(x)] * dx
            f = (1 - self.p_extra) * f + self.p_extra * conv
        k = int(round(self.const / dx))
        if k:
            f = np.concatenate([np.zeros(k), f[: len(f) - k]])
        f /= f.sum() * dx
        return x, f


DISTRIBUICOES: dict[str, Distribuicao] = {
    "BEOX-01": Distribuicao(
        "BEOX-01", "lognormal", 40.5, 12.0, 0.24, 5.0,
        justificativa="CV = 0,30. Montagem manual da estrutura do kit poço: tempo positivo e "
        "assimétrico (dificuldades pontuais alongam a cauda) → Lognormal. Ocorrências "
        "(falta de peça, retrabalho) em 24% das obras, duração Exp(5,0) sem memória."),
    "BEOX-02": Distribuicao(
        "BEOX-02", "lognormal", 19.9, 8.2, 0.11, 1.9,
        justificativa="CV = 0,41, a maior dispersão relativa da sequência 6N (tarefas de fixação "
        "variam com o modelo) → Lognormal evita tempos negativos que a Normal geraria "
        "(P≈0,8%). Ocorrências curtas: 11%·Exp(1,9)."),
    "BEOX-03": Distribuicao(
        "BEOX-03", "lognormal", 18.0, 4.5, 0.17, 5.2,
        justificativa="CV = 0,25. Processo mais padronizado, ainda assimétrico → Lognormal de "
        "baixa dispersão. Ocorrências 17%·Exp(5,2)."),
    "BEOX-04": Distribuicao(
        "BEOX-04", "lognormal", 33.7, 23.8, 0.36, PARADA_EXTRA_BEOX04,
        justificativa="CV = 0,71 (alto). Montagem por obra, fora do takt: o conteúdo de trabalho "
        "muda muito entre obras. A Lognormal reproduz a cauda longa das obras complexas; uma "
        "Normal teria 8% de valores negativos. Parada adicional em 36% das obras, "
        f"Exp({PARADA_EXTRA_BEOX04:g}) — duração adotada como premissa."),
    "MAQ-7C": Distribuicao(
        "MAQ-7C", "lognormal", 8.0, 2.0,
        justificativa="CV = 0,25. Célula automática SG1530 + EBe + LVD: a variação vem do programa "
        "e do tamanho da chapa, sempre positiva → Lognormal de baixa dispersão."),
    "BEOX-05": Distribuicao(
        "BEOX-05", "gamma", 40.7, 9.0, 1.0, 3.9, 0.2,
        justificativa="CV = 0,22. Posto gargalo com muitas subtarefas em série: Gamma com forma "
        "k = (40,7/9)² ≈ 20,4 equivale à soma de ~20 etapas exponenciais (Erlang). Paradas "
        "curtas Exp(3,9) em toda operação + setup constante de 0,2 min. Quebras modeladas à "
        "parte (Poisson, MTBF/MTTR)."),
    "MAQ-7D": Distribuicao(
        "MAQ-7D", "lognormal", 17.0, 3.5,
        justificativa="CV = 0,21. Automação programável Salvagnini + TOX: ciclo estável, "
        "levemente assimétrico → Lognormal."),
    "BEOX-07": Distribuicao(
        "BEOX-07", "normal", 40.2, 5.5, 0.24, 3.8,
        justificativa="CV = 0,14 (baixo) e comportamento simétrico → Normal é adequada "
        "(P(X<0) ≈ 10⁻¹³; truncada em zero por segurança). Ocorrências 24%·Exp(3,8)."),
    "BEOX-08": Distribuicao(
        "BEOX-08", "lognormal", 26.6, 5.0, 0.01, 0.5,
        justificativa="CV = 0,19, fronteira entre Normal e Lognormal; escolheu-se a Lognormal "
        "pela positividade. Ocorrência quase desprezível: 1%·Exp(0,5)."),
}

# Ordem fixa das estações (o índice é usado no log de animação)
ESTACOES = ["BEOX-01", "BEOX-02", "BEOX-03", "BEOX-04", "MAQ-7C",
            "BEOX-05", "MAQ-7D", "BEOX-07", "BEOX-08"]
IDX = {e: i for i, e in enumerate(ESTACOES)}
LINHA_DA_ESTACAO = {"BEOX-01": "6N", "BEOX-02": "6N", "BEOX-03": "6N", "BEOX-04": "6N",
                    "MAQ-7C": "7C", "BEOX-05": "7C", "MAQ-7D": "7D", "BEOX-07": "7D",
                    "BEOX-08": "7D"}
ESTACOES_6N = {"BEOX-01", "BEOX-02", "BEOX-03", "BEOX-04"}
MAQUINAS = {"MAQ-7C", "MAQ-7D"}

# "Tokens" = partes físicas de uma obra que andam separadas (para a animação)
#   0: kit 6N (cadeia principal)   1: ramo BEOX-04 (TO-BE)
#   2: kit 7C                      3: chapa 7D na máquina
#   4: ramo BEOX-07                5: ramo BEOX-08


# =============================================================================
# 3. CENÁRIOS
# =============================================================================
@dataclass(frozen=True)
class Cenario:
    id: str
    nome: str
    descricao: str
    takt: float
    demanda: int
    operadores_6n: int
    beox04_paralelo: bool
    quebras_05: bool


CENARIOS = [
    Cenario("C1", "AS-IS",
            "3 operadores na 6N; BEOX-04 no fim da sequência; BEOX-05 com quebras "
            "(MTBF 387 / MTTR 44,35 min).", TAKT_BASE, 10, 3, False, True),
    Cenario("C2", "TO-BE",
            "4 operadores na 6N; BEOX-04 em paralelo, fora do takt; manutenção preventiva "
            "elimina as quebras do BEOX-05.", TAKT_BASE, 10, 4, True, False),
    Cenario("C3", "Demanda +20%",
            "Layout TO-BE com demanda de 12 elevadores/dia (takt = 40 min).", 40.0, 12, 4, True, False),
    Cenario("C4", "TO-BE −1 operador",
            "TO-BE com 3 operadores na 6N (sensibilidade de recursos).", TAKT_BASE, 10, 3, True, False),
]


# =============================================================================
# 4. INSTRUMENTAÇÃO — variável de nível registrada a cada mudança
# =============================================================================
class Nivel:
    """Guarda a trajetória (t, valor) de uma variável em degraus: fila, WIP, ocupação."""

    def __init__(self, v0: float = 0.0):
        self.t = [0.0]
        self.v = [v0]

    def set(self, t: float, v: float) -> None:
        if t == self.t[-1]:
            self.v[-1] = v            # mudança instantânea: sobrescreve
        else:
            self.t.append(t)
            self.v.append(v)

    def add(self, t: float, d: float) -> None:
        self.set(t, self.v[-1] + d)

    def _duracoes(self, a: float, b: float):
        t = np.asarray(self.t)
        v = np.asarray(self.v, dtype=float)
        fim = np.append(t[1:], np.inf)
        dur = np.clip(np.minimum(fim, b) - np.maximum(t, a), 0.0, None)
        return v, dur

    def media(self, a: float, b: float) -> float:
        v, dur = self._duracoes(a, b)
        return float((v * dur).sum() / (b - a))

    def maximo(self, a: float, b: float) -> float:
        v, dur = self._duracoes(a, b)
        return float(v[dur > 0].max()) if (dur > 0).any() else 0.0

    def amostrar(self, grade: np.ndarray) -> np.ndarray:
        t = np.asarray(self.t)
        v = np.asarray(self.v, dtype=float)
        return v[np.searchsorted(t, grade, side="right") - 1]


# =============================================================================
# 5. MODELO DES (simpy)
# =============================================================================
class Modelo:
    def __init__(self, cen: Cenario, seed: int, saturado: bool = False):
        self.cen = cen
        self.saturado = saturado
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(seed)       # gerador único da réplica
        # Recursos: cada posto/máquina é um recurso de capacidade 1;
        # os operadores da 6N formam um pool compartilhado entre os 4 postos.
        self.postos = {e: simpy.Resource(self.env, capacity=1) for e in ESTACOES}
        self.op6n = simpy.Resource(self.env, capacity=cen.operadores_6n)
        # Instrumentação
        self.fila = {e: Nivel() for e in ESTACOES}
        self.ocupado = {e: Nivel() for e in ESTACOES}
        self.op6n_ocupados = Nivel()
        self.reparo_05 = Nivel()
        self.wip = Nivel()
        self.chegada: list[float] = []
        self.conclusao: list[float | None] = []
        self.kit_pronto: dict[str, list[tuple[int, float]]] = {"6N": [], "7C": [], "7D": []}
        self.ops: list[list] = []        # [obra, token, estação, t_fila, t_início, t_fim]
        self.reparos: list[tuple[float, float]] = []

    # ---- execução ---------------------------------------------------------
    def rodar(self) -> "Modelo":
        self.env.process(self._chegadas())
        self.env.run(until=T_FIM)
        return self

    def _chegadas(self):
        """Uma obra (elevador) chega a cada Takt × U(0,95; 1,05)."""
        j = 0
        while self.env.now < T_FIM:
            self.chegada.append(self.env.now)
            self.conclusao.append(None)
            self.env.process(self._elevador(j))
            j += 1
            if self.saturado:
                intervalo = self.cen.takt * FATOR_SATURACAO
            else:
                intervalo = self.cen.takt * self.rng.uniform(0.95, 1.05)
            yield self.env.timeout(intervalo)

    def _elevador(self, j: int):
        """O elevador só fica pronto quando os 3 kits estão concluídos."""
        env = self.env
        self.wip.add(env.now, +1)
        linhas = [env.process(self._linha_6n(j)),
                  env.process(self._linha_7c(j)),
                  env.process(self._linha_7d(j))]
        yield simpy.AllOf(env, linhas)
        self.conclusao[j] = env.now
        self.wip.add(env.now, -1)

    # ---- linhas -----------------------------------------------------------
    def _sequencia(self, j: int, token: int, estacoes: list[str]):
        for e in estacoes:
            yield self.env.process(self._operar(j, token, e))

    def _linha_6n(self, j: int):
        env = self.env
        if self.cen.beox04_paralelo:
            # TO-BE: BEOX-04 sai do takt e começa junto com o BEOX-01
            cadeia = env.process(self._sequencia(j, 0, ["BEOX-01", "BEOX-02", "BEOX-03"]))
            ramo04 = env.process(self._sequencia(j, 1, ["BEOX-04"]))
            yield cadeia & ramo04
        else:
            # AS-IS: BEOX-04 no fim da sequência
            yield env.process(self._sequencia(j, 0, ["BEOX-01", "BEOX-02", "BEOX-03", "BEOX-04"]))
        self.kit_pronto["6N"].append((j, env.now))

    def _linha_7c(self, j: int):
        yield self.env.process(self._sequencia(j, 2, ["MAQ-7C", "BEOX-05"]))
        self.kit_pronto["7C"].append((j, self.env.now))

    def _linha_7d(self, j: int):
        env = self.env
        yield env.process(self._operar(j, 3, "MAQ-7D"))
        p07 = env.process(self._operar(j, 4, "BEOX-07"))      # BEOX-07 ∥ BEOX-08
        p08 = env.process(self._operar(j, 5, "BEOX-08"))
        yield p07 & p08
        self.kit_pronto["7D"].append((j, env.now))

    # ---- operação em um posto ---------------------------------------------
    def _operar(self, j: int, token: int, est: str):
        env = self.env
        reg = [j, token, IDX[est], env.now, None, None]
        self.ops.append(reg)
        self.fila[est].add(env.now, +1)

        req_posto = self.postos[est].request()
        yield req_posto
        req_op = None
        if est in ESTACOES_6N:                       # precisa de um operador do pool 6N
            req_op = self.op6n.request()
            yield req_op
            self.op6n_ocupados.add(env.now, +1)

        reg[4] = env.now
        self.fila[est].add(env.now, -1)
        self.ocupado[est].set(env.now, 1)

        duracao = DISTRIBUICOES[est].amostrar(self.rng)
        if est == "BEOX-05" and self.cen.quebras_05:
            # Quebras (Poisson): enquanto houver tempo restante na operação,
            # sorteia o tempo até a próxima falha; se a falha ocorre antes do
            # fim, acrescenta um reparo exponencial e continua.
            restante = duracao
            while True:
                ttf = self.rng.exponential(MTBF_05)
                if ttf >= restante:
                    yield env.timeout(restante)
                    break
                yield env.timeout(ttf)
                restante -= ttf
                t0 = env.now
                self.reparo_05.set(t0, 1)
                yield env.timeout(self.rng.exponential(MTTR_05))
                self.reparo_05.set(env.now, 0)
                self.reparos.append((t0, env.now))
        else:
            yield env.timeout(duracao)

        reg[5] = env.now
        self.ocupado[est].set(env.now, 0)
        if req_op is not None:
            self.op6n.release(req_op)
            self.op6n_ocupados.add(env.now, -1)
        self.postos[est].release(req_posto)


# =============================================================================
# 6. KPIs
# =============================================================================
def calcular_kpis(m: Modelo) -> dict:
    a, b = T_AQUEC, T_FIM
    conc = np.array([c for c in m.conclusao if c is not None])
    diario = [int(((conc >= a + d * JORNADA) & (conc < a + (d + 1) * JORNADA)).sum())
              for d in range(DIAS_MEDICAO)]
    throughput = float(np.mean(diario))

    lead = {}
    for linha, lst in m.kit_pronto.items():
        lt = [t - m.chegada[j] for j, t in lst if a <= t < b]
        lead[linha] = float(np.mean(lt)) if lt else float("nan")
    lt_elev = [c - m.chegada[j] for j, c in enumerate(m.conclusao) if c is not None and a <= c < b]
    lead["elevador"] = float(np.mean(lt_elev)) if lt_elev else float("nan")

    util = {e: m.ocupado[e].media(a, b) for e in ESTACOES}
    fila_media = {e: m.fila[e].media(a, b) for e in ESTACOES}
    fila_max = {e: m.fila[e].maximo(a, b) for e in ESTACOES}
    quebras = sum(1 for t0, _ in m.reparos if a <= t0 < b)

    return {
        "throughput": throughput,
        "diario": diario,
        "lead": lead,
        "wip_medio": m.wip.media(a, b),
        "wip_max": m.wip.maximo(a, b),
        "util": util,
        "util_op6n": m.op6n_ocupados.media(a, b) / m.cen.operadores_6n,
        "fila_media": fila_media,
        "fila_max": fila_max,
        "ns_meta10": float(np.mean([d >= META_DIARIA for d in diario])),
        "ns_demanda": float(np.mean([d >= m.cen.demanda for d in diario])),
        "gargalo_util": max(util, key=util.get),
        # estações empatadas no topo (diferença ≤ 1 p.p.) — ex.: dois postos saturados
        "gargalos_util": [e for e in ESTACOES if util[e] >= max(util.values()) - 0.01],
        "gargalo_fila": max(fila_media, key=fila_media.get),
        "quebras": quebras,
        "frac_reparo_05": m.reparo_05.media(a, b),
        "chegadas_total": len(m.chegada),
        "concluidos_total": int(len(conc)),
        "wip_final": int(m.wip.v[-1]),
    }


def series_temporais(m: Modelo) -> dict:
    grade = np.arange(0.0, T_FIM + 1e-9, AMOSTRAGEM_SERIES)
    conc = np.sort(np.array([c for c in m.conclusao if c is not None]))
    grade_med = np.arange(T_AQUEC, T_FIM + 1e-9, AMOSTRAGEM_SERIES)
    acum = np.searchsorted(conc, grade_med, side="right") - np.searchsorted(conc, T_AQUEC, side="left")
    return {
        "x_dias": np.round(grade / JORNADA, 4).tolist(),
        "wip": m.wip.amostrar(grade).astype(int).tolist(),
        "fila05": m.fila["BEOX-05"].amostrar(grade).astype(int).tolist(),
        "acum_x": np.round((grade_med - T_AQUEC) / JORNADA, 4).tolist(),
        "acum": acum.astype(int).tolist(),
    }


def log_animacao(m: Modelo) -> dict:
    """Log compacto de eventos para o replay em Canvas (−1 = após o fim da simulação)."""
    plano = []
    for j, tok, e, tq, ts, te in m.ops:
        plano += [j, tok, e, round(tq, 2),
                  round(ts, 2) if ts is not None else -1,
                  round(te, 2) if te is not None else -1]
    return {
        "ops": plano,
        "cheg": [round(t, 2) for t in m.chegada],
        "conc": [round(c, 2) if c is not None else -1 for c in m.conclusao],
        "rep": [round(x, 2) for par in m.reparos for x in par],
    }


# =============================================================================
# 7. VERIFICAÇÃO & VALIDAÇÃO
# =============================================================================
def validar_distribuicoes() -> list[dict]:
    """Momentos amostrais vs. paramétricos, teste KS e histograma vs. densidade teórica."""
    rng = np.random.default_rng(SEED_PRINCIPAL)
    saida = []
    n = N_AMOSTRAS_VV
    d_crit = 1.36 / math.sqrt(n)                 # KS, α = 5 %
    for est in ESTACOES:
        dist = DISTRIBUICOES[est]
        amostra = np.array([dist.amostrar(rng) for _ in range(n)])
        x, f = dist.pdf_teorica()
        dx = x[1] - x[0]
        cdf = np.cumsum(f) * dx
        s = np.sort(amostra)
        F = np.interp(s, x + dx / 2, cdf)
        i = np.arange(1, n + 1)
        d_ks = float(max((i / n - F).max(), (F - (i - 1) / n).max()))

        lo, hi = np.percentile(amostra, [0.05, 99.8])
        cont, bordas = np.histogram(amostra, bins=45, range=(lo, hi))
        larg = bordas[1] - bordas[0]
        dens = cont / (n * larg)
        msk = (x >= lo) & (x <= hi)
        passo = max(1, int(msk.sum() / 240))
        media_a, dp_a = float(amostra.mean()), float(amostra.std(ddof=1))
        saida.append({
            "estacao": est,
            "dist": dist.descricao,
            "cv": dist.cv,
            "media_teo": dist.media_teorica, "media_amo": media_a,
            "dp_teo": dist.dp_teorico, "dp_amo": dp_a,
            "erro_media": (media_a - dist.media_teorica) / dist.media_teorica,
            "erro_dp": (dp_a - dist.dp_teorico) / dist.dp_teorico,
            "ks": d_ks, "ks_crit": d_crit, "ks_ok": d_ks < d_crit,
            "justificativa": dist.justificativa,
            "hist_x": np.round((bordas[:-1] + bordas[1:]) / 2, 3).tolist(),
            "hist_y": np.round(dens, 6).tolist(),
            "hist_w": float(larg),
            "pdf_x": np.round(x[msk][::passo], 3).tolist(),
            "pdf_y": np.round(f[msk][::passo], 6).tolist(),
        })
    return saida


def estudo_convergencia() -> dict:
    """Roda o AS-IS com várias seeds e acompanha a média acumulada do throughput."""
    valores = [calcular_kpis(Modelo(CENARIOS[0], s).rodar())["throughput"] for s in SEEDS_CONVERGENCIA]
    acum = np.cumsum(valores) / np.arange(1, len(valores) + 1)
    return {"seeds": list(SEEDS_CONVERGENCIA), "valores": [round(v, 3) for v in valores],
            "media_acum": np.round(acum, 4).tolist()}


def verificacao_estrutural(principais: dict[str, Modelo], kpis: dict[str, dict],
                           capacidade: dict[str, list[float]], convergencia: dict) -> list[dict]:
    checks = []

    def add(nome, resultado, criterio, ok):
        checks.append({"nome": nome, "resultado": resultado, "criterio": criterio, "ok": bool(ok)})

    conv = np.array(convergencia["valores"])
    media_c, dp_c = float(conv.mean()), float(conv.std(ddof=1))
    h_c = 1.96 * dp_c / math.sqrt(len(conv))
    add(f"Convergência do throughput AS-IS ({len(conv)} réplicas, seeds 42–{SEEDS_CONVERGENCIA[-1]})",
        f"{br(media_c)} ± {br(h_c)} elev./dia", "média ≈ 9,6 (|erro| ≤ 0,15)", abs(media_c - 9.6) <= 0.15)
    thr = kpis["C1"]["throughput"]
    z = (thr - media_c) / dp_c
    add("Throughput AS-IS da seed 42 (após aquecimento)", f"{br(thr)} elev./dia (z = {br(z)})",
        f"dentro da faixa de 95% das réplicas: {br(media_c - 1.96 * dp_c)} a {br(media_c + 1.96 * dp_c)}",
        abs(z) <= 1.96)
    cap_tobe = float(np.mean(capacidade["C2"]))
    add("Capacidade TO-BE (sistema saturado, 5 seeds)", f"{br(cap_tobe)} elev./dia",
        "≈ 10,6 ± 0,3", abs(cap_tobe - 10.6) <= 0.3)

    for cid, m in principais.items():
        k = kpis[cid]
        bal = k["chegadas_total"] == k["concluidos_total"] + k["wip_final"]
        add(f"Conservação de fluxo — {cid}",
            f"{k['chegadas_total']} = {k['concluidos_total']} + {k['wip_final']}",
            "chegadas = concluídos + WIP final", bal)

    erros_tempo = 0
    sobreposicoes = 0
    for m in principais.values():
        por_token: dict[tuple, list] = {}
        por_est: dict[int, list] = {}
        for j, tok, e, tq, ts, te in m.ops:
            if ts is not None and ts < tq - 1e-9:
                erros_tempo += 1
            if ts is not None and te is not None and te < ts - 1e-9:
                erros_tempo += 1
            por_token.setdefault((j, tok), []).append((tq, te))
            if ts is not None:
                por_est.setdefault(e, []).append((ts, te if te is not None else math.inf))
        for seq in por_token.values():
            for (tq1, te1), (tq2, _) in zip(seq, seq[1:]):
                if te1 is None or tq2 < te1 - 1e-9:
                    erros_tempo += 1
        for lst in por_est.values():
            lst.sort()
            for (s1, e1), (s2, _) in zip(lst, lst[1:]):
                if s2 < e1 - 1e-9:
                    sobreposicoes += 1
    add("Consistência temporal (todos os cenários)", f"{erros_tempo} violações",
        "t_fila ≤ t_início ≤ t_fim e precedência respeitada", erros_tempo == 0)
    add("Capacidade dos postos (capacidade = 1)", f"{sobreposicoes} sobreposições",
        "nenhuma operação simultânea no mesmo posto", sobreposicoes == 0)
    for cid, m in principais.items():
        pico = max(m.op6n_ocupados.v)
        add(f"Pool de operadores 6N — {cid}", f"pico = {pico:.0f} de {m.cen.operadores_6n}",
            "ocupados ≤ operadores disponíveis", pico <= m.cen.operadores_6n)
    for cid in ("C2", "C3", "C4"):
        add(f"Manutenção preventiva — {cid}", f"{kpis[cid]['quebras']} quebras no BEOX-05",
            "0 quebras", kpis[cid]["quebras"] == 0)
    rep = Modelo(CENARIOS[0], SEED_PRINCIPAL).rodar()
    igual = calcular_kpis(rep)["diario"] == kpis["C1"]["diario"]
    add("Reprodutibilidade (AS-IS, seed 42 rodado 2×)", "idêntico" if igual else "diferente",
        "mesma sequência diária de produção", igual)
    return checks


def verificacao_analitica(kpis: dict[str, dict], capacidade: dict[str, list[float]]) -> list[dict]:
    """U_teórica = tempo_ciclo / (takt × disponibilidade) para o BEOX-05."""
    tc = DISTRIBUICOES["BEOX-05"].media_teorica
    linhas = []
    for cen in CENARIOS:
        disp = MTBF_05 / (MTBF_05 + MTTR_05) if cen.quebras_05 else 1.0
        u_teo = tc / (cen.takt * disp)
        u_esp = min(u_teo, 1.0)
        u_sim = kpis[cen.id]["util"]["BEOX-05"]
        cap_teo = JORNADA * disp / tc
        cap_sim = float(np.mean(capacidade[cen.id]))
        linhas.append({
            "cenario": cen.id, "nome": cen.nome, "takt": cen.takt, "disp": disp, "tc": tc,
            "u_teo": u_teo, "u_esp": u_esp, "u_sim": u_sim,
            "ok_u": abs(u_sim - u_esp) <= 0.03,
            "cap_teo": cap_teo, "cap_sim": cap_sim, "ok_cap": abs(cap_sim - cap_teo) / cap_teo <= 0.03,
        })
    return linhas


def lei_de_little(kpis: dict[str, dict]) -> list[dict]:
    """L = λ·W: WIP médio vs. taxa de saída × lead time médio do elevador."""
    out = []
    for cen in CENARIOS:
        k = kpis[cen.id]
        lam = k["throughput"] / JORNADA
        lw = lam * k["lead"]["elevador"]
        estacionario = k["throughput"] >= cen.demanda * 0.97
        out.append({"cenario": cen.id, "nome": cen.nome, "L": k["wip_medio"], "lw": lw,
                    "erro": (lw - k["wip_medio"]) / k["wip_medio"], "estacionario": estacionario})
    return out


def intervalo_confianca(valores: list[float]) -> dict:
    v = np.array(valores, dtype=float)
    media, s = float(v.mean()), float(v.std(ddof=1))
    h = T_STUDENT_975_GL4 * s / math.sqrt(len(v))
    return {"reps": [round(x, 3) for x in v], "media": media, "s": s, "lo": media - h, "hi": media + h, "h": h}


# =============================================================================
# 8. EXECUÇÃO DOS EXPERIMENTOS
# =============================================================================
def executar() -> dict:
    principais: dict[str, Modelo] = {}
    kpis: dict[str, dict] = {}
    series: dict[str, dict] = {}
    anim: dict[str, dict] = {}
    thr_reps: dict[str, list[float]] = {}
    cap_reps: dict[str, list[float]] = {}

    for cen in CENARIOS:
        thr_reps[cen.id], cap_reps[cen.id] = [], []
        for seed in SEEDS_REPLICAS:
            m = Modelo(cen, seed).rodar()
            k = calcular_kpis(m)
            thr_reps[cen.id].append(k["throughput"])
            if seed == SEED_PRINCIPAL:
                principais[cen.id], kpis[cen.id] = m, k
                series[cen.id] = series_temporais(m)
                anim[cen.id] = log_animacao(m)
            ms = Modelo(cen, seed, saturado=True).rodar()
            cap_reps[cen.id].append(calcular_kpis(ms)["throughput"])
        kpis[cen.id]["capacidade"] = float(np.mean(cap_reps[cen.id]))
        print(f"  {cen.id} {cen.nome:<18} throughput={kpis[cen.id]['throughput']:.2f}  "
              f"capacidade={kpis[cen.id]['capacidade']:.2f}  "
              f"gargalo={kpis[cen.id]['gargalo_util']}  "
              f"U05={kpis[cen.id]['util']['BEOX-05']:.1%}")

    ic = {cid: intervalo_confianca(v) for cid, v in thr_reps.items()}
    ic_cap = {cid: intervalo_confianca(v) for cid, v in cap_reps.items()}

    print("  Estudo de convergência do AS-IS…")
    convergencia = estudo_convergencia()
    print(f"    média de {len(SEEDS_CONVERGENCIA)} réplicas = {np.mean(convergencia['valores']):.3f}")
    print("  Validando distribuições…")
    return {
        "param": {
            "jornada": JORNADA, "takt": TAKT_BASE, "meta": META_DIARIA,
            "dias_aquec": DIAS_AQUECIMENTO, "dias_med": DIAS_MEDICAO,
            "t_aquec": T_AQUEC, "t_fim": T_FIM, "mtbf": MTBF_05, "mttr": MTTR_05,
            "seed": SEED_PRINCIPAL, "seeds": list(SEEDS_REPLICAS),
            "fator_sat": FATOR_SATURACAO, "parada04": PARADA_EXTRA_BEOX04,
        },
        "cenarios": [cen.__dict__ for cen in CENARIOS],
        "estacoes": [{"codigo": e, "linha": LINHA_DA_ESTACAO[e], "maquina": e in MAQUINAS,
                      "media": DISTRIBUICOES[e].media_teorica, "dist": DISTRIBUICOES[e].descricao}
                     for e in ESTACOES],
        "kpis": kpis,
        "ic": ic,
        "ic_cap": ic_cap,
        "series": series,
        "anim": anim,
        "vv": {
            "estrutural": verificacao_estrutural(principais, kpis, cap_reps, convergencia),
            "convergencia": convergencia,
            "analitica": verificacao_analitica(kpis, cap_reps),
            "little": lei_de_little(kpis),
            "dist": validar_distribuicoes(),
        },
    }


# =============================================================================
# 9. GERAÇÃO DO HTML
# =============================================================================
def gerar_html(dados: dict) -> str:
    js = json.dumps(dados, ensure_ascii=False, separators=(",", ":"), allow_nan=True,
                    default=float).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DADOS__", js)


def main() -> None:
    print("Simulação DES — linha de elevadores EOX")
    dados = executar()
    html = gerar_html(dados)
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as fh:
        fh.write(html)
    ok = sum(c["ok"] for c in dados["vv"]["estrutural"])
    print(f"  V&V estrutural: {ok}/{len(dados['vv']['estrutural'])} verificações OK")
    print(f"Arquivo gerado: {ARQUIVO_SAIDA} ({len(html) / 1024:.0f} KB)")


# =============================================================================
# 10. TEMPLATE HTML (CSS + JavaScript embutidos; Plotly via CDN)
# =============================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulação DES — Linha de elevadores EOX</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{
    --surface-0:#f4f3ef; --surface-1:#fcfcfb; --border:#e2e0da; --grid:#ecebe7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#8a8984;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
    --l6n:#2a6fd6; --l7c:#d64545; --l7d:#1f9a55;
    --good-bg:#e4f3e8; --good:#1d7a3c; --bad:#b42318; --bad-bg:#fdecea; --warn:#a15c00;
    --accent:#1c4f91;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface-0);color:var(--text-primary);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{background:#14213d;color:#fff;padding:28px 32px 22px}
  header h1{margin:0 0 4px;font-size:24px;font-weight:650;letter-spacing:-.01em}
  header p{margin:0;color:#c9d3e6;max-width:980px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
  .chip{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:999px;
        padding:3px 11px;font-size:12.5px;color:#e8edf7}
  nav{position:sticky;top:0;z-index:5;background:var(--surface-1);border-bottom:1px solid var(--border);
      display:flex;gap:4px;padding:6px 24px;overflow-x:auto}
  nav a{color:var(--text-secondary);text-decoration:none;padding:6px 10px;border-radius:6px;white-space:nowrap;font-weight:500}
  nav a:hover{background:var(--surface-0);color:var(--text-primary)}
  main{max-width:1360px;margin:0 auto;padding:20px 24px 60px}
  section{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin:18px 0}
  h2{margin:0 0 4px;font-size:19px;font-weight:650}
  h3{margin:18px 0 8px;font-size:15px;font-weight:650}
  .sub{color:var(--text-secondary);margin:0 0 14px}
  .muted{color:var(--text-muted)}
  /* ---- animação ---- */
  .controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;margin-bottom:10px}
  button{font:inherit;font-weight:550;border:1px solid var(--border);background:#fff;border-radius:8px;
         padding:6px 14px;cursor:pointer;color:var(--text-primary)}
  button:hover{border-color:#b9b7b0}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff;min-width:96px}
  select{font:inherit;padding:6px 8px;border-radius:8px;border:1px solid var(--border);background:#fff}
  label.ctl{display:flex;align-items:center;gap:8px;color:var(--text-secondary)}
  input[type=range]{accent-color:var(--accent)}
  #vel{width:170px} #tempo{flex:1;min-width:220px}
  .hud{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:6px 0 10px}
  .hud div{background:var(--surface-0);border-radius:8px;padding:7px 10px}
  .hud b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
  .hud span{font-size:12px;color:var(--text-secondary)}
  #cv{width:100%;aspect-ratio:1280/640;display:block;border-radius:10px;border:1px solid var(--border);background:#fff}
  .legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12.5px;color:var(--text-secondary);margin-top:8px}
  .legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
  .desc-cen{margin:4px 0 0;color:var(--text-secondary);font-size:13px}
  /* ---- tabelas ---- */
  .tbl-wrap{overflow-x:auto}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
  th,td{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left;white-space:normal}
  thead th{font-weight:600;color:var(--text-secondary);font-size:12.5px;border-bottom:1px solid var(--border);vertical-align:bottom}
  tr.grp td{background:var(--surface-0);font-weight:650;color:var(--text-secondary);font-size:12px;
            text-transform:uppercase;letter-spacing:.04em;text-align:left}
  td.best{background:var(--good-bg);font-weight:650;color:var(--good)}
  .sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
  .ok{color:var(--good);font-weight:600} .fail{color:var(--bad);font-weight:600} .warn{color:var(--warn);font-weight:600}
  .ubar{display:inline-block;height:6px;border-radius:3px;background:#c7d9f3;vertical-align:middle;margin-left:6px}
  .note{font-size:12.5px;color:var(--text-secondary);margin-top:8px}
  /* ---- gráficos ---- */
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media (max-width:980px){.grid2{grid-template-columns:1fr}}
  .card{border:1px solid var(--grid);border-radius:10px;padding:12px 12px 4px}
  .card h3{margin:0 0 2px;font-size:14px} .card p{margin:0 0 4px;font-size:12.5px;color:var(--text-secondary)}
  .chart{height:330px}
  .hist-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  @media (max-width:980px){.hist-grid{grid-template-columns:1fr}}
  .hist{height:220px}
  .aviso{color:var(--warn);padding:20px}
  .callout{background:#eef4fc;border-left:3px solid var(--s1);padding:10px 14px;border-radius:6px;margin:10px 0;color:#1d2c44}
  details summary{cursor:pointer;color:var(--accent);font-weight:550}
  ul.prem li{margin:4px 0}
  footer{color:var(--text-muted);font-size:12px;text-align:center;padding:10px}
</style>
</head>
<body>
<header>
  <h1>Simulação de eventos discretos — linha de montagem de elevadores EOX</h1>
  <p>Modelo DES em Python/simpy das linhas 6N (kit poço), 7C (painéis/SFM) e 7D (teto/plataforma). Um elevador só fica pronto quando os três kits estão concluídos. Todos os números desta página vêm da execução do script <code>eox_simulacao.py</code>.</p>
  <div class="chips" id="chips"></div>
</header>
<nav>
  <a href="#sec-anim">Layout animado</a><a href="#sec-kpi">KPIs</a><a href="#sec-graf">Gráficos</a>
  <a href="#sec-vv">V&amp;V</a><a href="#sec-prem">Premissas</a>
</nav>
<main>

<section id="sec-anim">
  <h2>Layout fabril animado</h2>
  <p class="sub">Replay do log de eventos da simulação (seed 42). Cada caixinha é um kit de uma obra; os bonecos ficam nítidos quando o posto está trabalhando e esmaecidos quando ocioso.</p>
  <div class="controls">
    <button class="primary" id="btnPlay">▶ Play</button>
    <button id="btnReset">↺ Reset</button>
    <button id="btnSkip" title="Pular para o fim do aquecimento">⏭ Fim do aquecimento</button>
    <label class="ctl">Cenário <select id="selCen"></select></label>
    <label class="ctl">Velocidade <input type="range" id="vel" min="0" max="1000" value="544"><b id="velTxt" style="min-width:92px"></b></label>
  </div>
  <div class="controls"><label class="ctl" style="flex:1">Tempo <input type="range" id="tempo" min="0" max="1" step="1" value="0"></label></div>
  <p class="desc-cen" id="descCen"></p>
  <div class="hud">
    <div><span>Dia · hora do turno</span><b id="hDia">—</b></div>
    <div><span>Fase</span><b id="hFase">—</b></div>
    <div><span>Elevadores concluídos</span><b id="hConc">—</b></div>
    <div><span>WIP (obras no sistema)</span><b id="hWip">—</b></div>
    <div><span>Fila BEOX-05</span><b id="hFila">—</b></div>
    <div><span>Operadores 6N ocupados</span><b id="hOps">—</b></div>
  </div>
  <canvas id="cv"></canvas>
  <div class="legend">
    <span><i style="background:var(--l6n)"></i>Kit 6N</span>
    <span><i style="background:#86aee9"></i>Ramo BEOX-04 (TO-BE)</span>
    <span><i style="background:var(--l7c)"></i>Kit 7C</span>
    <span><i style="background:var(--l7d)"></i>Kit 7D</span>
    <span><i style="background:#7fcb9c"></i>Ramo BEOX-08</span>
    <span>Barra no posto = progresso da operação · ⚠ = BEOX-05 em reparo</span>
  </div>
</section>

<section id="sec-kpi">
  <h2>KPIs comparativos</h2>
  <p class="sub">Janela de medição: 18 dias úteis após 3 dias de aquecimento. Seed 42; o intervalo de confiança usa as seeds 42–46. Em verde, o melhor valor de cada linha (quando "melhor" faz sentido).</p>
  <div class="tbl-wrap"><table id="tblKpi"></table></div>
  <div class="callout" id="leitura"></div>
</section>

<section id="sec-graf">
  <h2>Gráficos temporais</h2>
  <p class="sub">Passe o mouse para ler os valores. A faixa cinza marca os 3 dias de aquecimento.</p>
  <div class="grid2">
    <div class="card"><h3>Throughput acumulado</h3><p>Elevadores completos desde o fim do aquecimento; tracejado = meta de 10/dia.</p><div class="chart" id="gAcum"></div></div>
    <div class="card"><h3>Produção diária</h3><p>Elevadores completos por dia de medição; tracejado = meta.</p><div class="chart" id="gDia"></div></div>
    <div class="card"><h3>WIP ao longo do tempo</h3><p>Obras em processo (chegaram e ainda não viraram elevador completo).</p><div class="chart" id="gWip"></div></div>
    <div class="card"><h3>Fila em BEOX-05</h3><p>Kits 7C aguardando o posto gargalo.</p><div class="chart" id="gFila"></div></div>
    <div class="card"><h3>Utilização por estação</h3><p>% do tempo de medição com o recurso ocupado (inclui reparo no BEOX-05).</p><div class="chart" id="gUtil"></div></div>
    <div class="card"><h3>Fila média por estação</h3><p>Número médio de kits aguardando cada posto.</p><div class="chart" id="gFilaEst"></div></div>
  </div>
</section>

<section id="sec-vv">
  <h2>Verificação &amp; validação</h2>
  <p class="sub">Verificação: o código faz o que o modelo conceitual diz? Validação: o modelo representa o sistema real e os dados de entrada?</p>

  <h3>1. Verificação estrutural</h3>
  <div class="tbl-wrap"><table id="tblEstr"></table></div>
  <div class="card" style="margin-top:12px"><h3>Convergência do throughput AS-IS</h3><p>Throughput de cada réplica (pontos) e média acumulada (linha). A seed 42 isolada fica abaixo de 9,6 por acaso amostral; a média converge para o valor esperado.</p><div class="chart" id="gConv" style="height:280px"></div></div>

  <h3>2. Verificação analítica — utilização do BEOX-05</h3>
  <p class="note" style="margin-top:0">U<sub>teórica</sub> = tempo_ciclo / (takt × disponibilidade), com tempo_ciclo = E[Gamma] + E[paradas] + setup = 40,7 + 3,9 + 0,2 = 44,8 min e disponibilidade A = MTBF/(MTBF+MTTR). Se U<sub>teórica</sub> &gt; 100%, o posto satura e a utilização simulada esperada é 100%. A capacidade teórica é 480·A/tempo_ciclo.</p>
  <div class="tbl-wrap"><table id="tblAnal"></table></div>
  <h3>Lei de Little (L = λ·W)</h3>
  <div class="tbl-wrap"><table id="tblLittle"></table></div>
  <p class="note">λ = throughput/480 (elev./min); W = lead time médio do elevador. Em cenários não estacionários (fila crescendo, C1 e C3) a igualdade vale só aproximadamente, porque o WIP não tem média estável na janela.</p>

  <h3>3. Validação das distribuições de entrada</h3>
  <p class="note" style="margin-top:0">20.000 amostras de cada gerador (seed 42) comparadas com a distribuição paramétrica: média e desvio-padrão, teste de Kolmogorov–Smirnov (α = 5%) e histograma vs. densidade teórica (obtida por convolução numérica da parcela base com a ocorrência exponencial).</p>
  <div class="tbl-wrap"><table id="tblDist"></table></div>
  <div class="hist-grid" id="hists" style="margin-top:14px"></div>
  <details style="margin-top:12px"><summary>Justificativa da escolha de cada distribuição</summary><div class="tbl-wrap"><table id="tblJust"></table></div></details>

  <h3>4. Intervalo de confiança do throughput (5 réplicas: seeds 42–46)</h3>
  <p class="note" style="margin-top:0">IC 95% = média ± t<sub>0,975;4</sub>·s/√5, com t = 2,776.</p>
  <div class="tbl-wrap"><table id="tblIC"></table></div>
</section>

<section id="sec-prem">
  <h2>Premissas e notas do modelo</h2>
  <ul class="prem" id="premissas"></ul>
</section>
<footer>Gerado por eox_simulacao.py · simpy + numpy · gráficos Plotly (CDN)</footer>
</main>

<script>
"use strict";
const D = __DADOS__;

/* ===================== utilidades ===================== */
const CORES_CEN = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];
const COR_LINHA = {'6N': '#2a6fd6', '7C': '#d64545', '7D': '#1f9a55'};
const P = D.param, J = P.jornada, TA = P.t_aquec, TF = P.t_fim;
const CEN = D.cenarios;
const EST = D.estacoes;
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : Number(v).toLocaleString('pt-BR', {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = (v, d = 1) => fmt(v * 100, d) + '%';
const $ = id => document.getElementById(id);
const sw = i => `<span class="sw" style="background:${CORES_CEN[i]}"></span>`;
const okTxt = ok => ok ? '<span class="ok">✔ OK</span>' : '<span class="fail">✖ Falha</span>';

$('chips').innerHTML = [
  `Takt ${P.takt} min`, `Jornada ${J} min/dia`, `Meta ${P.meta} elev./dia`,
  `Aquecimento ${P.dias_aquec} dias + medição ${P.dias_med} dias`, `Seed ${P.seed}`,
  `Chegadas: takt × U(0,95; 1,05)`, `BEOX-05: MTBF ${fmt(P.mtbf, 0)} / MTTR ${fmt(P.mttr, 2)} min`
].map(t => `<span class="chip">${t}</span>`).join('');

/* ===================== tabela de KPIs ===================== */
function tabelaKPI() {
  const S = CEN.map((c, i) => ({c, i, k: D.kpis[c.id], ic: D.ic[c.id]}));
  const R = [];
  const g = t => R.push({grp: t});
  const r = (l, f, d, dir, opts = {}) => R.push({l, f, d, dir, ...opts});
  g('Produção');
  r('Demanda (elevadores/dia)', s => s.c.demanda, 0, 0);
  r('Throughput médio (elevadores/dia)', s => s.k.throughput, 2, +1);
  r('IC 95% do throughput (seeds 42–46)', s => `${fmt(s.ic.lo, 2)} – ${fmt(s.ic.hi, 2)}`, null, 0);
  r('Capacidade — sistema saturado (elevadores/dia)', s => s.k.capacidade, 2, +1);
  r('Nível de serviço — dias com ≥ 10 elevadores', s => s.k.ns_meta10, 'pct', +1);
  r('Nível de serviço — dias com ≥ demanda do cenário', s => s.k.ns_demanda, 'pct', +1);
  g('Lead time (min, da chegada da obra até o kit pronto)');
  r('Kit 6N', s => s.k.lead['6N'], 0, -1);
  r('Kit 7C', s => s.k.lead['7C'], 0, -1);
  r('Kit 7D', s => s.k.lead['7D'], 0, -1);
  r('Elevador completo', s => s.k.lead.elevador, 0, -1);
  g('Estoque em processo e gargalo');
  r('WIP médio (obras simultâneas)', s => s.k.wip_medio, 1, -1);
  r('WIP máximo', s => s.k.wip_max, 0, -1);
  r('Fila BEOX-05 — média', s => s.k.fila_media['BEOX-05'], 1, -1);
  r('Fila BEOX-05 — máxima', s => s.k.fila_max['BEOX-05'], 0, -1);
  r('Quebras do BEOX-05 na medição', s => s.k.quebras, 0, -1);
  r('Gargalo — maior utilização', s => s.k.gargalos_util.join(' + '), null, 0);
  r('Gargalo — maior fila média', s => s.k.gargalo_fila, null, 0);
  g('Utilização por estação');
  EST.forEach(e => r(e.codigo + (e.maquina ? ' (máquina)' : ''), s => s.k.util[e.codigo], 'util', 0));
  r('Operadores 6N (pool)', s => s.k.util_op6n, 'util', 0, {extra: s => ` <span class="muted">(${s.c.operadores_6n} op.)</span>`});

  let h = '<thead><tr><th>KPI</th>' + S.map(s =>
    `<th>${sw(s.i)}${s.c.id} · ${s.c.nome}</th>`).join('') + '</tr></thead><tbody>';
  for (const row of R) {
    if (row.grp) { h += `<tr class="grp"><td colspan="${S.length + 1}">${row.grp}</td></tr>`; continue; }
    const vals = S.map(row.f);
    let best = null;
    if (row.dir) {
      const nums = vals.map(Number);
      best = row.dir > 0 ? Math.max(...nums) : Math.min(...nums);
    }
    h += `<tr><td>${row.l}</td>` + vals.map((v, i) => {
      let txt;
      if (row.d === null) txt = v;
      else if (row.d === 'pct') txt = pct(v, 0);
      else if (row.d === 'util') {
        txt = pct(v, 1) + (v >= 0.95 ? ' <span class="warn" title="Saturado">⚠</span>' : '')
          + `<span class="ubar" style="width:${Math.round(Math.min(v, 1) * 44)}px"></span>`;
      } else txt = fmt(v, row.d);
      if (row.extra) txt += row.extra(S[i]);
      const isBest = row.dir && Math.abs(Number(v) - best) < 1e-9;
      return `<td${isBest ? ' class="best"' : ''}>${txt}</td>`;
    }).join('') + '</tr>';
  }
  $('tblKpi').innerHTML = h + '</tbody>';

  const k1 = D.kpis.C1, k2 = D.kpis.C2, k3 = D.kpis.C3, k4 = D.kpis.C4;
  $('leitura').innerHTML = `<b>Leitura rápida.</b> No AS-IS o BEOX-05 trabalha ${pct(k1.util['BEOX-05'], 0)} do tempo, com quebras, e a fila dele cresce durante toda a medição (máx. ${fmt(k1.fila_max['BEOX-05'], 0)} kits): a capacidade fica em ${fmt(k1.capacidade, 2)} elev./dia, abaixo da meta. `
    + `No TO-BE, sem quebras, a capacidade sobe para ${fmt(k2.capacidade, 2)} elev./dia; o throughput fica em ${fmt(k2.throughput, 2)} porque a produção é puxada pela demanda (10/dia) e o lead time do elevador cai de ${fmt(k1.lead.elevador, 0)} para ${fmt(k2.lead.elevador, 0)} min. `
    + `Com demanda +20% (C3), a capacidade instalada não acompanha: o sistema entrega ${fmt(k3.throughput, 2)} elev./dia e três postos saturam ao mesmo tempo (${k3.gargalos_util.join(', ')}). O BEOX-05 continua sendo a maior fila, e BEOX-01 e BEOX-07 passam a ser os próximos gargalos. `
    + `Tirar 1 operador da 6N no TO-BE (C4) mantém o throughput (${fmt(k4.throughput, 2)}), mas o pool passa a ${pct(k4.util_op6n, 0)} de ocupação e o lead time do kit 6N sobe de ${fmt(k2.lead['6N'], 0)} para ${fmt(k4.lead['6N'], 0)} min.`;
}

/* ===================== gráficos (Plotly) ===================== */
const FONT = {family: 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif', size: 12, color: '#3b3b38'};
function lay(extra = {}) {
  const base = {
    font: FONT, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {l: 54, r: 12, t: 54, b: 44}, hovermode: 'x unified',
    legend: {orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: {size: 11.5}},
    hoverlabel: {bgcolor: '#fff', bordercolor: '#d6d4cd', font: {size: 12}},
  };
  const ax = {gridcolor: '#ecebe7', zeroline: false, linecolor: '#c9c8c2', tickfont: {color: '#52514e'}};
  return {...base, ...extra, xaxis: {...ax, ...(extra.xaxis || {})}, yaxis: {...ax, ...(extra.yaxis || {})}};
}
const CFG = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']};
const faixaAquec = [{type: 'rect', xref: 'x', yref: 'paper', x0: 0, x1: P.dias_aquec, y0: 0, y1: 1,
  fillcolor: '#efeee9', line: {width: 0}, layer: 'below'}];
const anotAquec = [{x: P.dias_aquec / 2, y: 1, xref: 'x', yref: 'paper', text: 'aquecimento', showarrow: false,
  yanchor: 'top', font: {size: 11, color: '#8a8984'}}];

function graficos() {
  if (!window.Plotly) {
    document.querySelectorAll('.chart,.hist').forEach(e =>
      e.innerHTML = '<p class="aviso">Os gráficos usam Plotly via CDN e precisam de internet. O restante da página funciona offline.</p>');
    return false;
  }
  const S = id => D.series[id];
  // 1. throughput acumulado
  const t1 = CEN.map((c, i) => ({x: S(c.id).acum_x, y: S(c.id).acum, name: `${c.id} ${c.nome}`, mode: 'lines',
    line: {color: CORES_CEN[i], width: 2}, hovertemplate: '%{y:.0f}<extra>' + c.id + '</extra>'}));
  t1.push({x: [0, P.dias_med], y: [0, P.meta * P.dias_med], name: 'Meta 10/dia', mode: 'lines',
    line: {color: '#8a8984', width: 1.5, dash: 'dash'}, hoverinfo: 'skip'});
  Plotly.newPlot('gAcum', t1, lay({xaxis: {title: 'Dia de medição', range: [0, P.dias_med]}, yaxis: {title: 'Elevadores'}}), CFG);
  // 2. produção diária
  const dias = Array.from({length: P.dias_med}, (_, i) => i + 1);
  const t2 = CEN.map((c, i) => ({x: dias, y: D.kpis[c.id].diario, name: `${c.id} ${c.nome}`, type: 'bar',
    marker: {color: CORES_CEN[i], line: {color: '#fcfcfb', width: 1}}, hovertemplate: '%{y}<extra>' + c.id + '</extra>'}));
  Plotly.newPlot('gDia', t2, lay({barmode: 'group', bargap: 0.25, bargroupgap: 0.05,
    xaxis: {title: 'Dia de medição', dtick: 1}, yaxis: {title: 'Elevadores/dia'},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: P.meta, y1: P.meta, line: {color: '#52514e', width: 1.2, dash: 'dash'}}]}), CFG);
  // 3. WIP
  const t3 = CEN.map((c, i) => ({x: S(c.id).x_dias, y: S(c.id).wip, name: `${c.id} ${c.nome}`, mode: 'lines',
    line: {color: CORES_CEN[i], width: 1.6, shape: 'hv'}, hovertemplate: '%{y}<extra>' + c.id + '</extra>'}));
  Plotly.newPlot('gWip', t3, lay({shapes: faixaAquec, annotations: anotAquec,
    xaxis: {title: 'Dia de simulação', range: [0, TF / J]}, yaxis: {title: 'Obras em processo'}}), CFG);
  // 4. fila BEOX-05
  const t4 = CEN.map((c, i) => ({x: S(c.id).x_dias, y: S(c.id).fila05, name: `${c.id} ${c.nome}`, mode: 'lines',
    line: {color: CORES_CEN[i], width: 1.6, shape: 'hv'}, hovertemplate: '%{y}<extra>' + c.id + '</extra>'}));
  Plotly.newPlot('gFila', t4, lay({shapes: faixaAquec, annotations: anotAquec,
    xaxis: {title: 'Dia de simulação', range: [0, TF / J]}, yaxis: {title: 'Kits na fila'}}), CFG);
  // 5. utilização
  const nomes = EST.map(e => e.codigo);
  const t5 = CEN.map((c, i) => ({x: nomes, y: nomes.map(n => D.kpis[c.id].util[n] * 100), name: `${c.id} ${c.nome}`, type: 'bar',
    marker: {color: CORES_CEN[i], line: {color: '#fcfcfb', width: 1}}, hovertemplate: '%{y:.1f}%<extra>' + c.id + '</extra>'}));
  Plotly.newPlot('gUtil', t5, lay({barmode: 'group', bargap: 0.22, yaxis: {title: 'Utilização (%)', range: [0, 105]},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 100, y1: 100, line: {color: '#52514e', width: 1, dash: 'dot'}}]}), CFG);
  // 6. fila média por estação
  const t6 = CEN.map((c, i) => ({x: nomes, y: nomes.map(n => D.kpis[c.id].fila_media[n]), name: `${c.id} ${c.nome}`, type: 'bar',
    marker: {color: CORES_CEN[i], line: {color: '#fcfcfb', width: 1}}, hovertemplate: '%{y:.2f}<extra>' + c.id + '</extra>'}));
  Plotly.newPlot('gFilaEst', t6, lay({barmode: 'group', bargap: 0.22, yaxis: {title: 'Kits aguardando (média)'}}), CFG);
  return true;
}

/* ===================== V&V ===================== */
function vv() {
  const V = D.vv;
  $('tblEstr').innerHTML = '<thead><tr><th>Verificação</th><th>Resultado</th><th>Critério</th><th>Status</th></tr></thead><tbody>'
    + V.estrutural.map(c => `<tr><td>${c.nome}</td><td>${c.resultado}</td><td>${c.criterio}</td><td>${okTxt(c.ok)}</td></tr>`).join('') + '</tbody>';

  $('tblAnal').innerHTML = '<thead><tr><th>Cenário</th><th>Takt</th><th>Disponib. A</th><th>Tempo de ciclo</th><th>U teórica</th><th>U esperada</th><th>U simulada</th><th>Δ</th><th>Status</th><th>Capac. teórica</th><th>Capac. simulada</th><th>Status</th></tr></thead><tbody>'
    + V.analitica.map((a, i) => `<tr><td>${sw(i)}${a.cenario} · ${a.nome}</td><td>${fmt(a.takt, 0)}</td><td>${pct(a.disp, 1)}</td><td>${fmt(a.tc, 1)} min</td>
      <td>${pct(a.u_teo, 1)}</td><td>${pct(a.u_esp, 1)}</td><td>${pct(a.u_sim, 1)}</td><td>${fmt((a.u_sim - a.u_esp) * 100, 1)} p.p.</td><td>${okTxt(a.ok_u)}</td>
      <td>${fmt(a.cap_teo, 2)}</td><td>${fmt(a.cap_sim, 2)}</td><td>${okTxt(a.ok_cap)}</td></tr>`).join('') + '</tbody>';
  $('tblAnal').insertAdjacentHTML('afterend', '<p class="note">Critérios: |U simulada − U esperada| ≤ 3 p.p.; |capacidade simulada − teórica| ≤ 3%. A capacidade simulada é a média das 5 seeds com o sistema saturado (chegadas a 75% do takt).</p>');

  $('tblLittle').innerHTML = '<thead><tr><th>Cenário</th><th>L = WIP médio</th><th>λ·W</th><th>Diferença</th><th>Regime</th></tr></thead><tbody>'
    + V.little.map((l, i) => `<tr><td>${sw(i)}${l.cenario} · ${l.nome}</td><td>${fmt(l.L, 2)}</td><td>${fmt(l.lw, 2)}</td><td>${pct(l.erro, 1)}</td>
      <td>${l.estacionario ? (Math.abs(l.erro) <= 0.05 ? '<span class="ok">estacionário ✔</span>' : '<span class="fail">estacionário ✖</span>') : '<span class="warn">fila crescente (aproximado)</span>'}</td></tr>`).join('') + '</tbody>';

  $('tblDist').innerHTML = '<thead><tr><th>Estação</th><th>Distribuição</th><th>CV base</th><th>Média teór.</th><th>Média amostral</th><th>Erro</th><th>DP teór.</th><th>DP amostral</th><th>Erro</th><th>KS D</th><th>D crítico</th><th>Status</th></tr></thead><tbody>'
    + V.dist.map(d => `<tr><td>${d.estacao}</td><td style="text-align:left">${d.dist}</td><td>${fmt(d.cv, 2)}</td>
      <td>${fmt(d.media_teo, 2)}</td><td>${fmt(d.media_amo, 2)}</td><td>${pct(d.erro_media, 2)}</td>
      <td>${fmt(d.dp_teo, 2)}</td><td>${fmt(d.dp_amo, 2)}</td><td>${pct(d.erro_dp, 2)}</td>
      <td>${fmt(d.ks, 4)}</td><td>${fmt(d.ks_crit, 4)}</td><td>${okTxt(d.ks_ok)}</td></tr>`).join('') + '</tbody>';

  $('tblJust').innerHTML = '<thead><tr><th>Estação</th><th style="text-align:left">Justificativa</th></tr></thead><tbody>'
    + V.dist.map(d => `<tr><td>${d.estacao}</td><td style="text-align:left;white-space:normal">${d.justificativa}</td></tr>`).join('') + '</tbody>';

  $('tblIC').innerHTML = '<thead><tr><th>Cenário</th>' + P.seeds.map(s => `<th>seed ${s}</th>`).join('')
    + '<th>Média</th><th>s</th><th>IC 95%</th><th>Semi-amplitude</th><th>Capacidade (IC 95%)</th></tr></thead><tbody>'
    + CEN.map((c, i) => {
      const ic = D.ic[c.id], cp = D.ic_cap[c.id];
      return `<tr${c.id === 'C1' ? ' style="background:#f6f9fe"' : ''}><td>${sw(i)}${c.id} · ${c.nome}</td>` + ic.reps.map(v => `<td>${fmt(v, 2)}</td>`).join('')
        + `<td><b>${fmt(ic.media, 2)}</b></td><td>${fmt(ic.s, 3)}</td><td>[${fmt(ic.lo, 2)}; ${fmt(ic.hi, 2)}]</td><td>±${fmt(ic.h, 2)}</td>
        <td>${fmt(cp.media, 2)} [${fmt(cp.lo, 2)}; ${fmt(cp.hi, 2)}]</td></tr>`;
    }).join('') + '</tbody>';
  const ic1 = D.ic.C1;
  const contem = ic1.lo <= 9.6 && 9.6 <= ic1.hi;
  $('tblIC').insertAdjacentHTML('afterend', `<p class="note">Cenário 1 (AS-IS): IC 95% do throughput diário = [${fmt(ic1.lo, 2)}; ${fmt(ic1.hi, 2)}] elev./dia — ${contem ? 'contém' : 'não contém'} o valor esperado de 9,6. ${D.ic.C1.hi < D.ic.C2.lo ? 'Os ICs de C1 e C2 não se sobrepõem: a diferença de throughput entre AS-IS e TO-BE é estatisticamente significativa (α = 5%).' : 'Os ICs de C1 e C2 se sobrepõem: com 5 réplicas a diferença de throughput não é conclusiva.'}</p>`);

  if (!window.Plotly) return;
  const cv = V.convergencia;
  Plotly.newPlot('gConv', [
    {x: cv.seeds, y: cv.valores, mode: 'markers', name: 'Throughput da réplica', marker: {color: '#9ec1ee', size: 8, line: {color: '#fcfcfb', width: 1.5}}, hovertemplate: 'seed %{x}: %{y:.2f}<extra></extra>'},
    {x: cv.seeds, y: cv.media_acum, mode: 'lines', name: 'Média acumulada', line: {color: CORES_CEN[0], width: 2}, hovertemplate: 'média: %{y:.3f}<extra></extra>'},
    {x: [cv.seeds[0], cv.seeds[cv.seeds.length - 1]], y: [9.6, 9.6], mode: 'lines', name: 'Esperado 9,6', line: {color: '#52514e', width: 1.2, dash: 'dash'}, hoverinfo: 'skip'},
  ], lay({hovermode: 'closest', xaxis: {title: 'Seed'}, yaxis: {title: 'Elevadores/dia'}}), CFG);

  const box = $('hists');
  V.dist.forEach((d, i) => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `<h3>${d.estacao}</h3><p>${d.dist}</p><div class="hist" id="h${i}"></div>`;
    box.appendChild(div);
    Plotly.newPlot('h' + i, [
      {x: d.hist_x, y: d.hist_y, type: 'bar', width: d.hist_w * 0.92, name: 'Amostra', marker: {color: '#b7d0f2'}, hovertemplate: '%{x:.1f} min: %{y:.4f}<extra>amostra</extra>'},
      {x: d.pdf_x, y: d.pdf_y, mode: 'lines', name: 'Teórica', line: {color: '#1c4f91', width: 2}, hovertemplate: '%{x:.1f} min: %{y:.4f}<extra>teórica</extra>'},
    ], lay({showlegend: i === 0, margin: {l: 46, r: 8, t: 30, b: 34}, hovermode: 'closest',
      xaxis: {title: {text: 'min', standoff: 4}}, yaxis: {title: ''}}), CFG);
  });
}

function premissas() {
  const itens = [
    'Tempo simulado em minutos de trabalho: cada dia tem 480 min corridos (um turno), sem paradas fora do turno.',
    'Cada obra (elevador) dispara ao mesmo tempo os 3 kits; o elevador conta como pronto quando o último kit termina.',
    'Cada posto e cada máquina atende um kit por vez, em ordem de chegada (FIFO), com fila ilimitada à frente.',
    'Linha 6N: os operadores formam um pool compartilhado. O kit ocupa o posto e precisa de um operador livre para começar. No AS-IS são 3 operadores para 4 postos, com BEOX-04 no fim da sequência (BEOX-01→02→03→04). No TO-BE o BEOX-04 sai do takt e começa junto com o BEOX-01 (ramo paralelo), e o kit 6N fica pronto quando os dois ramos terminam.',
    `BEOX-04: a duração da "parada adicional" (36% das obras) não foi informada; adotou-se Exponencial com média de ${fmt(P.parada04, 1)} min (constante PARADA_EXTRA_BEOX04 no script).`,
    'BEOX-05: Gamma(40,7; 9,0) + paradas curtas Exp(3,9) em toda operação + setup fixo de 0,2 min (média 44,8 min). No AS-IS, durante a operação, sorteia-se o tempo até a próxima falha ~ Exp(MTBF = 387 min); se a falha ocorre antes do fim, soma-se um reparo ~ Exp(MTTR = 44,35 min) e o sorteio se repete para o tempo restante. A utilização do BEOX-05 inclui o tempo em reparo.',
    'Linha 7D: após a máquina automática, o kit se divide entre BEOX-07 e BEOX-08 (em paralelo) e só fica pronto quando os dois terminam.',
    `Capacidade = throughput com o sistema saturado (chegadas a cada ${fmt(P.fator_sat * 100, 0)}% do takt, de modo que nunca falte trabalho na frente do gargalo). Throughput = elevadores completos por dia com a demanda do cenário; quando a capacidade supera a demanda, o throughput fica limitado pela própria demanda.`,
    'Gerador numpy.random.default_rng(seed) único por réplica; seed 42 é a principal; as seeds 43 a 46 completam as 5 réplicas do IC; as seeds 42 a 81 são usadas no estudo de convergência.',
    'Nível de serviço: % dos 18 dias de medição com produção diária ≥ 10 (meta) e, separadamente, ≥ demanda do cenário (12 no C3).',
  ];
  $('premissas').innerHTML = itens.map(t => `<li>${t}</li>`).join('');
}

/* ===================== ANIMAÇÃO (Canvas 2D) ===================== */
const W = 1280, H = 640, TRAVEL = 4;          // TRAVEL: minutos simulados de deslocamento visual
const BW = 140, BH = 60;                       // tamanho das caixas dos postos
const cv = $('cv'), ctx = cv.getContext('2d');
const LINHA_TOK = ['6N', '6N', '7C', '7D', '7D', '7D'];
const COR_TOK = ['#2a6fd6', '#86aee9', '#d64545', '#1f9a55', '#1f9a55', '#7fcb9c'];
const KIT = {'6N': {x: 990, y: 121}, '7C': {x: 990, y: 315}, '7D': {x: 990, y: 514}};
const ELEV = {x: 1115, y: 40, w: 150, h: 560};
const LANES = [
  {id: '6N', nome: 'LINHA 6N · Kit Poço', y0: 12, y1: 230},
  {id: '7C', nome: 'LINHA 7C · Kit Painéis/SFM', y0: 240, y1: 390},
  {id: '7D', nome: 'LINHA 7D · Kit Teto/Plataforma', y0: 400, y1: 628},
];
const ENT = {'6N': {x: 60, y: 121}, '7C': {x: 60, y: 315}, '7D': {x: 60, y: 514}};

function montarLayout(par) {
  const st = [];
  if (par) {
    st[0] = {x: 300, y: 170, q: 100}; st[1] = {x: 500, y: 170, q: 370}; st[2] = {x: 700, y: 170, q: 570};
    st[3] = {x: 500, y: 72, q: 100};
  } else {
    st[0] = {x: 245, y: 121, q: 100}; st[1] = {x: 445, y: 121, q: 315}; st[2] = {x: 645, y: 121, q: 515};
    st[3] = {x: 845, y: 121, q: 715};
  }
  st[4] = {x: 245, y: 315, q: 100}; st[5] = {x: 700, y: 315, q: 315};
  st[6] = {x: 245, y: 514, q: 100}; st[7] = {x: 700, y: 462, q: 315}; st[8] = {x: 700, y: 570, q: 315};
  st.forEach(s => { s.maxCols = Math.max(1, Math.floor((s.x - BW / 2 - 8 - s.q - 6) / 15)); });
  const R = i => ({x: st[i].x + BW / 2, y: st[i].y}), Lf = i => ({x: st[i].x - BW / 2, y: st[i].y});
  const e = (a, b, cor) => ({a, b, cor});
  const ent = k => ({x: ENT[k].x + 40, y: ENT[k].y});
  const kitL = k => ({x: KIT[k].x - 60, y: KIT[k].y}), kitR = k => ({x: KIT[k].x + 60, y: KIT[k].y});
  const c6 = COR_LINHA['6N'], c7c = COR_LINHA['7C'], c7d = COR_LINHA['7D'];
  const edges = par
    ? [e(ent('6N'), Lf(0), c6), e(ent('6N'), Lf(3), c6), e(R(0), Lf(1), c6), e(R(1), Lf(2), c6), e(R(2), kitL('6N'), c6), e(R(3), kitL('6N'), c6)]
    : [e(ent('6N'), Lf(0), c6), e(R(0), Lf(1), c6), e(R(1), Lf(2), c6), e(R(2), Lf(3), c6), e(R(3), kitL('6N'), c6)];
  edges.push(e(ent('7C'), Lf(4), c7c), e(R(4), Lf(5), c7c), e(R(5), kitL('7C'), c7c),
    e(ent('7D'), Lf(6), c7d), e(R(6), Lf(7), c7d), e(R(6), Lf(8), c7d), e(R(7), kitL('7D'), c7d), e(R(8), kitL('7D'), c7d));
  ['6N', '7C', '7D'].forEach(k => edges.push(e(kitR(k), {x: ELEV.x, y: KIT[k].y}, '#6b6a65')));
  return {st, edges};
}

// ---- preparação do log de cada cenário ----
const CACHE = {};
function prepararCenario(id) {
  if (CACHE[id]) return CACHE[id];
  const A = D.anim[id], o = A.ops, map = new Map();
  for (let i = 0; i < o.length; i += 6) {
    const j = o[i], tok = o[i + 1], key = j * 8 + tok;
    let tk = map.get(key);
    if (!tk) { tk = {j, tok, ops: []}; map.set(key, tk); }
    tk.ops.push({est: o[i + 2], tq: o[i + 3], ts: o[i + 4] < 0 ? Infinity : o[i + 4], te: o[i + 5] < 0 ? Infinity : o[i + 5]});
  }
  const toks = [...map.values()];
  for (const tk of toks) {
    tk.ops.sort((a, b) => a.tq - b.tq);
    tk.t0 = tk.ops[0].tq;
    const c = A.conc[tk.j];
    tk.c = c < 0 ? Infinity : c;
    tk.t1 = tk.tok === 3 ? tk.ops[tk.ops.length - 1].te : tk.c + TRAVEL;
  }
  const rep = [];
  for (let i = 0; i < A.rep.length; i += 2) rep.push([A.rep[i], A.rep[i + 1]]);
  const concOrd = A.conc.filter(c => c >= 0).sort((a, b) => a - b);
  const cen = CEN.find(c => c.id === id);
  CACHE[id] = {toks, rep, cheg: A.cheg, concOrd, cen, L: montarLayout(cen.beox04_paralelo)};
  return CACHE[id];
}
const contarAte = (arr, t) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= t) lo = m + 1; else hi = m; } return lo; };

// ---- primitivas de desenho ----
function rr(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function seta(a, b, cor) {
  ctx.strokeStyle = cor; ctx.fillStyle = cor; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.55;
  ctx.beginPath(); ctx.moveTo(a.x, a.y);
  if (Math.abs(a.y - b.y) < 1) ctx.lineTo(b.x - 7, b.y);
  else { const mx = (a.x + b.x) / 2; ctx.lineTo(mx, a.y); ctx.lineTo(mx, b.y); ctx.lineTo(b.x - 7, b.y); }
  ctx.stroke();
  ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(b.x - 8, b.y - 4.5); ctx.lineTo(b.x - 8, b.y + 4.5); ctx.closePath(); ctx.fill();
  ctx.globalAlpha = 1;
}
function boneco(x, y, ativo, fase) {
  // (x, y) = quadril. Capacete amarelo, cabeça, tronco, braços e pernas em traços.
  ctx.save();
  ctx.globalAlpha = ativo ? 1 : 0.22;
  const sw = ativo ? Math.sin(fase) * 4 : 0;
  ctx.strokeStyle = '#26262a'; ctx.lineWidth = 2.3; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x - 6, y + 15);              // perna esq.
  ctx.moveTo(x, y); ctx.lineTo(x + 6, y + 15);              // perna dir.
  ctx.moveTo(x, y); ctx.lineTo(x, y - 16);                  // tronco
  ctx.moveTo(x, y - 12);
  if (ativo) { ctx.lineTo(x - 11, y - 6 + sw); ctx.moveTo(x, y - 12); ctx.lineTo(x - 12, y - 10 - sw); } // braços trabalhando
  else { ctx.lineTo(x - 5, y - 1); ctx.moveTo(x, y - 12); ctx.lineTo(x + 5, y - 1); }                  // braços parados
  ctx.stroke();
  ctx.fillStyle = '#f2c9a4'; ctx.beginPath(); ctx.arc(x, y - 22, 5.2, 0, Math.PI * 2); ctx.fill(); ctx.lineWidth = 1.2; ctx.stroke();
  ctx.fillStyle = '#f5c518'; ctx.beginPath(); ctx.arc(x, y - 23.5, 6, Math.PI, 0); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = '#a88400'; ctx.lineWidth = 1; ctx.stroke();
  ctx.strokeStyle = '#d4a800'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 8.5, y - 23.2); ctx.lineTo(x + 7.5, y - 23.2); ctx.stroke();
  ctx.restore();
}
function engrenagem(x, y, ativo, ang) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(ativo ? ang : 0); ctx.globalAlpha = ativo ? 1 : 0.3;
  ctx.fillStyle = '#5b6170';
  ctx.beginPath();
  for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4; ctx.rect(-3, -15, 6, 6); ctx.rotate(Math.PI / 4); }
  ctx.fill();
  ctx.beginPath(); ctx.arc(0, 0, 11, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(0, 0, 4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}
function kitBox(x, y, cor, s = 12) {
  rr(x - s / 2, y - s / 2, s, s, 2.5);
  ctx.fillStyle = cor; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffffff'; ctx.stroke();
}
const lerp = (a, b, f) => ({x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f});
const ease = f => f < 0 ? 0 : f > 1 ? 1 : f * f * (3 - 2 * f);

// ---- estado da animação ----
const AN = {id: 'C1', t: 0, play: false, vel: 15, last: null};
function setVel() {
  const v = +$('vel').value; AN.vel = Math.pow(10, -1 + 4 * v / 1000);
  $('velTxt').textContent = (AN.vel < 1 ? fmt(AN.vel, 2) : AN.vel < 10 ? fmt(AN.vel, 1) : fmt(AN.vel, 0)) + ' min/s';
}

function desenhar() {
  const C = prepararCenario(AN.id), L = C.L, st = L.st, t = AN.t;
  const fase = performance.now() / 160;
  ctx.clearRect(0, 0, W, H);
  // faixas das linhas
  for (const ln of LANES) {
    const cor = COR_LINHA[ln.id];
    ctx.fillStyle = cor + '0f'; rr(6, ln.y0, 1094, ln.y1 - ln.y0, 10); ctx.fill();
    ctx.fillStyle = cor; rr(6, ln.y0, 5, ln.y1 - ln.y0, 2); ctx.fill();
    ctx.font = '700 12px system-ui,sans-serif'; ctx.fillStyle = cor; ctx.textAlign = 'left'; ctx.fillText(ln.nome, 20, ln.y0 + 18);
  }
  L.edges.forEach(e => seta(e.a, e.b, e.cor));

  // ---- estado dos tokens ----
  const vis = [], filas = st.map(() => []), emProc = st.map(() => null), noKit = {'6N': [], '7C': [], '7D': []};
  for (const tk of C.toks) {
    if (t < tk.t0 || t >= tk.t1) continue;
    let i = tk.ops.length - 1; while (i > 0 && tk.ops[i].tq > t) i--;
    const op = tk.ops[i];
    let s;
    if (t < op.te) s = t < op.ts ? 'fila' : 'proc';
    else if (tk.tok === 3) continue;
    else s = t < tk.c ? 'kit' : 'saida';
    const v = {tk, i, op, s};
    vis.push(v);
    if (s === 'fila') filas[op.est].push(v);
    else if (s === 'proc') emProc[op.est] = v;
    else noKit[LINHA_TOK[tk.tok]].push(v);
  }
  filas.forEach(f => f.sort((a, b) => a.op.tq - b.op.tq || a.tk.j - b.tk.j));
  Object.values(noKit).forEach(f => f.sort((a, b) => a.tk.j - b.tk.j || a.tk.tok - b.tk.tok));
  const emReparo = C.rep.some(r => t >= r[0] && t < r[1]);

  // ---- entradas ----
  ctx.textAlign = 'center';
  for (const k of ['6N', '7C', '7D']) {
    const p = ENT[k];
    rr(p.x - 40, p.y - 22, 80, 44, 8); ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = COR_LINHA[k]; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.fillStyle = '#52514e'; ctx.font = '600 11.5px system-ui,sans-serif'; ctx.fillText('Entrada', p.x, p.y - 3);
    ctx.font = '11px system-ui,sans-serif'; ctx.fillStyle = '#8a8984'; ctx.fillText('obras', p.x, p.y + 12);
  }

  // ---- postos ----
  st.forEach((p, idx) => {
    const e = EST[idx], cor = COR_LINHA[e.linha], ativo = !!emProc[idx];
    const quebra = idx === 5 && emReparo;
    const x = p.x - BW / 2, y = p.y - BH / 2;
    rr(x, y, BW, BH, 8); ctx.fillStyle = quebra ? '#fdecea' : (ativo ? '#ffffff' : '#fafaf8'); ctx.fill();
    ctx.lineWidth = quebra ? 2.5 : (ativo ? 2 : 1.2); ctx.strokeStyle = quebra ? '#d92d20' : (ativo ? cor : '#cfcdc6'); ctx.stroke();
    ctx.textAlign = 'left'; ctx.fillStyle = '#0b0b0b'; ctx.font = '700 12.5px system-ui,sans-serif';
    ctx.fillText(e.maquina ? (idx === 4 ? 'Máq. Auto 7C' : 'Máq. Auto 7D') : e.codigo, x + 8, y + 17);
    ctx.font = '11px system-ui,sans-serif'; ctx.fillStyle = '#52514e';
    ctx.fillText('μ = ' + fmt(e.media, 1) + ' min', x + 8, y + 32);
    if (ativo) {
      const op = emProc[idx].op;
      ctx.fillStyle = '#8a8984'; ctx.fillText('obra #' + emProc[idx].tk.j, x + 8, y + 46);
      const prog = isFinite(op.te) ? Math.min(1, (t - op.ts) / (op.te - op.ts)) : 0.5;
      rr(x + 6, y + BH - 7, BW - 12, 3.5, 1.5); ctx.fillStyle = '#ecebe7'; ctx.fill();
      rr(x + 6, y + BH - 7, (BW - 12) * prog, 3.5, 1.5); ctx.fillStyle = quebra ? '#d92d20' : cor; ctx.fill();
    }
    if (e.maquina) engrenagem(x + BW - 26, p.y - 2, ativo, fase / 3);
    else boneco(x + BW - 24, p.y + 6, ativo, fase);
    if (quebra) {
      ctx.fillStyle = '#d92d20'; rr(x + BW - 78, y - 12, 76, 18, 9); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '700 11px system-ui,sans-serif'; ctx.textAlign = 'center'; ctx.fillText('⚠ QUEBRA', x + BW - 40, y + 1);
    }
    // contador de fila
    const nf = filas[idx].length;
    if (nf > 0 && !e.maquina) {
      ctx.textAlign = 'right'; ctx.fillStyle = nf >= 5 ? '#b42318' : '#52514e'; ctx.font = '600 11px system-ui,sans-serif';
      ctx.fillText('fila ' + nf, x - 4, y - 4);
    }
  });

  // ---- caixas de kit e elevador ----
  for (const k of ['6N', '7C', '7D']) {
    const p = KIT[k];
    rr(p.x - 60, p.y - 36, 120, 72, 8); ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = COR_LINHA[k]; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.textAlign = 'left'; ctx.fillStyle = COR_LINHA[k]; ctx.font = '700 12px system-ui,sans-serif'; ctx.fillText('Kit ' + k, p.x - 52, p.y - 20);
    ctx.textAlign = 'right'; ctx.fillStyle = '#52514e'; ctx.font = '11px system-ui,sans-serif'; ctx.fillText(noKit[k].filter(v => v.s === 'kit').length + ' aguard.', p.x + 54, p.y - 20);
  }
  const concAte = contarAte(C.concOrd, t);
  const diaIni = Math.floor(t / J) * J;
  const hoje = concAte - contarAte(C.concOrd, diaIni - 1e-9);
  const ultima = concAte > 0 ? C.concOrd[concAte - 1] : -1e9;
  const flash = t - ultima < 6;
  rr(ELEV.x, ELEV.y, ELEV.w, ELEV.h, 10); ctx.fillStyle = flash ? '#fff8e0' : '#fff'; ctx.fill();
  ctx.strokeStyle = '#3b3b38'; ctx.lineWidth = 1.6; ctx.stroke();
  const cx = ELEV.x + ELEV.w / 2;
  ctx.textAlign = 'center'; ctx.fillStyle = '#0b0b0b'; ctx.font = '700 13px system-ui,sans-serif';
  ctx.fillText('ELEVADOR', cx, ELEV.y + 24); ctx.fillText('COMPLETO', cx, ELEV.y + 40);
  ctx.font = '700 40px system-ui,sans-serif'; ctx.fillText(concAte, cx, ELEV.y + 92);
  ctx.font = '11.5px system-ui,sans-serif'; ctx.fillStyle = '#52514e'; ctx.fillText('concluídos no total', cx, ELEV.y + 110);
  // cabine estilizada
  const by = ELEV.y + 150;
  ctx.strokeStyle = '#3b3b38'; ctx.lineWidth = 1.5; rr(cx - 34, by, 68, 96, 4); ctx.stroke();
  ctx.fillStyle = flash ? '#f5c518' : '#e9e8e3'; ctx.fillRect(cx - 30, by + 4, 28, 88); ctx.fillRect(cx + 2, by + 4, 28, 88);
  [['6N', 0], ['7C', 1], ['7D', 2]].forEach(([k, i]) => { kitBox(cx - 20 + i * 20, by + 112, COR_LINHA[k], 14); });
  ctx.fillStyle = '#52514e'; ctx.font = '11px system-ui,sans-serif'; ctx.fillText('6N + 7C + 7D', cx, by + 136);
  // produção do dia
  const meta = C.cen.demanda;
  ctx.fillStyle = '#0b0b0b'; ctx.font = '700 12px system-ui,sans-serif'; ctx.fillText('Hoje: ' + hoje + ' / ' + meta, cx, ELEV.y + 330);
  rr(ELEV.x + 14, ELEV.y + 340, ELEV.w - 28, 8, 4); ctx.fillStyle = '#ecebe7'; ctx.fill();
  rr(ELEV.x + 14, ELEV.y + 340, (ELEV.w - 28) * Math.min(1, hoje / meta), 8, 4); ctx.fillStyle = hoje >= meta ? '#1d7a3c' : '#2a78d6'; ctx.fill();
  ctx.fillStyle = '#52514e'; ctx.font = '11px system-ui,sans-serif';
  ctx.fillText('meta do cenário', cx, ELEV.y + 364);
  // mini histórico por dia
  const diaAtual = Math.floor(t / J);
  ctx.textAlign = 'left'; ctx.font = '600 11px system-ui,sans-serif'; ctx.fillStyle = '#52514e'; ctx.fillText('Por dia (últimos 8)', ELEV.x + 14, ELEV.y + 392);
  for (let k = 0; k < 8; k++) {
    const d = diaAtual - 7 + k; if (d < 0) continue;
    const n = contarAte(C.concOrd, Math.min((d + 1) * J, t) - 1e-9) - contarAte(C.concOrd, d * J - 1e-9);
    const hgt = Math.min(1, n / 13) * 110, bx = ELEV.x + 14 + k * 15.5;
    rr(bx, ELEV.y + 520 - hgt, 11, Math.max(hgt, 1), 2); ctx.fillStyle = d < P.dias_aquec ? '#c9c8c2' : (n >= meta ? '#1d7a3c' : '#2a78d6'); ctx.fill();
  }
  const ym = ELEV.y + 520 - meta / 13 * 110;
  ctx.strokeStyle = '#52514e'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(ELEV.x + 10, ym); ctx.lineTo(ELEV.x + ELEV.w - 10, ym); ctx.stroke(); ctx.setLineDash([]);
  ctx.textAlign = 'center'; ctx.fillStyle = '#8a8984'; ctx.fillText('verde = meta batida', cx, ELEV.y + 542);

  // ---- posições dos kits ----
  const posEst = idx => ({x: st[idx].x + 18, y: st[idx].y + 14});
  const posSaida = idx => ({x: st[idx].x + BW / 2, y: st[idx].y});
  const origem = v => {
    if (v.i > 0) return posSaida(v.tk.ops[v.i - 1].est);
    if (v.tk.tok === 4 || v.tk.tok === 5) return posSaida(6);
    const e = ENT[LINHA_TOK[v.tk.tok]]; return {x: e.x + 40, y: e.y};
  };
  const ocultos = st.map(() => 0);
  const slotFila = (idx, k) => {
    const s = st[idx], col = Math.floor(k / 4), row = k % 4;
    if (col >= s.maxCols) return null;
    return {x: s.x - BW / 2 - 14 - col * 15, y: s.y - 22.5 + row * 15};
  };
  const slotKit = (lin, k) => {
    const p = KIT[lin], col = k % 7, row = Math.floor(k / 7);
    if (row >= 3) return null;
    return {x: p.x - 48 + col * 16, y: p.y - 2 + row * 15};
  };
  const desenhos = [];
  filas.forEach((f, idx) => f.forEach((v, k) => {
    const alvo = slotFila(idx, k);
    if (!alvo) { ocultos[idx]++; return; }
    const fr = (t - v.op.tq) / TRAVEL;
    desenhos.push([fr < 1 ? lerp(origem(v), alvo, ease(fr)) : alvo, v]);
  }));
  emProc.forEach((v, idx) => {
    if (!v) return;
    const alvo = posEst(idx), fr = (t - v.op.tq) / TRAVEL;
    desenhos.push([fr < 1 ? lerp(origem(v), alvo, ease(fr)) : alvo, v]);
  });
  const ocultosKit = {'6N': 0, '7C': 0, '7D': 0};
  for (const lin of ['6N', '7C', '7D']) {
    noKit[lin].forEach((v, k) => {
      const alvo = slotKit(lin, k) || {x: KIT[lin].x + 40, y: KIT[lin].y + 26};
      if (v.s === 'kit') {
        if (!slotKit(lin, k)) { ocultosKit[lin]++; return; }
        const fr = (t - v.op.te) / TRAVEL;
        desenhos.push([fr < 1 ? lerp(posSaida(v.op.est), alvo, ease(fr)) : alvo, v]);
      } else {
        const fr = (t - v.tk.c) / TRAVEL, fim = {x: cx, y: by + 112};
        desenhos.push([lerp(alvo, fim, ease(fr)), v]);
      }
    });
  }
  for (const [p, v] of desenhos) kitBox(p.x, p.y, COR_TOK[v.tk.tok]);
  ctx.font = '700 11px system-ui,sans-serif'; ctx.fillStyle = '#b42318'; ctx.textAlign = 'right';
  ocultos.forEach((n, idx) => { if (n) { const s = st[idx]; ctx.fillText('+' + n, s.x - BW / 2 - 8 - s.maxCols * 15, s.y + 36); } });
  ctx.textAlign = 'center';
  Object.entries(ocultosKit).forEach(([lin, n]) => { if (n) ctx.fillText('+' + n, KIT[lin].x, KIT[lin].y + 46); });

  // ---- HUD ----
  const dia = Math.min(Math.floor(t / J) + 1, TF / J), minTurno = t >= TF ? J : t % J;
  const hh = 7 + Math.floor(minTurno / 60), mm = Math.floor(minTurno % 60);
  $('hDia').textContent = `Dia ${dia} · ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
  $('hFase').textContent = t < TA ? 'Aquecimento' : `Medição ${Math.min(dia - P.dias_aquec, P.dias_med)}/${P.dias_med}`;
  $('hConc').textContent = concAte;
  $('hWip').textContent = contarAte(C.cheg, t) - concAte;
  $('hFila').textContent = filas[5].length;
  const ops6 = [0, 1, 2, 3].filter(i => emProc[i]).length;
  $('hOps').textContent = `${ops6} / ${C.cen.operadores_6n}`;
}

function loop(ts) {
  if (AN.play) {
    if (AN.last !== null) AN.t = Math.min(TF, AN.t + (ts - AN.last) / 1000 * AN.vel);
    AN.last = ts;
    if (AN.t >= TF) { AN.play = false; $('btnPlay').textContent = '▶ Play'; }
    $('tempo').value = AN.t;
  } else AN.last = null;
  desenhar();
  requestAnimationFrame(loop);
}

function redimensionar() {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth;
  cv.width = Math.round(w * dpr); cv.height = Math.round(w * H / W * dpr);
  ctx.setTransform(cv.width / W, 0, 0, cv.height / H, 0, 0);
}

function iniciarAnimacao() {
  const sel = $('selCen');
  sel.innerHTML = CEN.map(c => `<option value="${c.id}">${c.id} · ${c.nome}</option>`).join('');
  const desc = () => { const c = CEN.find(c => c.id === AN.id); $('descCen').textContent = c.descricao; };
  sel.onchange = () => { AN.id = sel.value; desc(); };
  $('btnPlay').onclick = () => {
    if (AN.t >= TF) AN.t = 0;
    AN.play = !AN.play; $('btnPlay').textContent = AN.play ? '⏸ Pause' : '▶ Play';
  };
  $('btnReset').onclick = () => { AN.t = 0; AN.play = false; $('btnPlay').textContent = '▶ Play'; $('tempo').value = 0; };
  $('btnSkip').onclick = () => { AN.t = TA; $('tempo').value = TA; };
  $('vel').oninput = setVel; setVel();
  const tp = $('tempo'); tp.max = TF; tp.oninput = () => { AN.t = +tp.value; };
  desc();
  window.addEventListener('resize', redimensionar); redimensionar();
  requestAnimationFrame(loop);
}

/* ===================== inicialização ===================== */
tabelaKPI();
premissas();
iniciarAnimacao();
const temPlotly = graficos();
vv();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
