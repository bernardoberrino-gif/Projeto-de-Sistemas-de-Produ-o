#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eox_cad.py
==========
Modelo CAD 3D (sólidos B-Rep, pythonOCC / OpenCASCADE) dos layouts AS-IS e TO-BE
da linha de montagem de elevadores EOX, integrado à simulação DES
(``eox_simulacao.py``).

Integração com a simulação (fonte única de verdade)
---------------------------------------------------
O CAD não repete números da simulação: ele importa o módulo ``eox_simulacao``
e roda o modelo DES (seed 42) para cada cenário. Daí vêm:

* lista de estações, linha de cada estação e tempo médio (rótulos);
* configuração do cenário: nº de operadores da 6N, BEOX-04 em série/paralelo,
  quebras do BEOX-05 (que decide se existe o armário de manutenção preventiva);
* fila máxima simulada de cada estação  -> nº de posições de buffer no piso;
* máximo de kits aguardando consolidação -> tamanho das áreas de kit;
* utilização de cada estação              -> cor do andon (verde/amarelo/vermelho);
* estado do sistema no instante de WIP máximo (t*) -> onde cada kit e cada
  operador aparecem no modelo 3D (snapshot do log de eventos).

O grafo de fluxo do CAD é escrito aqui de forma independente e depois
conferido contra as precedências observadas no log da simulação (V&V).

Saídas (pasta ``cad_saida/``)
-----------------------------
* ``eox_layout_asis.step`` e ``eox_layout_tobe.step`` — montagens STEP AP214 com
  nome e cor de cada peça;
* imagens PNG das vistas (isométrica, planta, detalhes) e comparativos;
* ``eox_cad_vv.json`` e ``eox_cad_relatorio.html`` — resultados da V&V.

Uso
---
    conda install -c conda-forge pythonocc-core=7.8.1 simpy numpy pillow
    python eox_cad.py                 # modelo + V&V + STEP + imagens
    python eox_cad.py --sem-imagens   # sem renderização (ex.: servidor sem OpenGL)

A renderização usa o OffscreenRenderer do pythonOCC e precisa de um contexto
OpenGL (desktop comum ou ``xvfb-run`` em servidor). Unidades: milímetros.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- OpenCASCADE (pythonOCC) --------------------------------------------------
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                  BRepPrimAPI_MakePrism, BRepPrimAPI_MakeSphere)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCC.Core.STEPControl import STEPControl_AsIs
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHELL, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, topods
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

# --- Simulação DES: fonte única de verdade ------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eox_simulacao as sim  # noqa: E402

# =============================================================================
# 1. PARÂMETROS DO GALPÃO E DO LAYOUT (mm)
# =============================================================================
GALPAO_X, GALPAO_Y = 64_000, 42_000
ALTURA_PAREDE = 5_000
ALTURA_COLUNA = 7_000
PASSO_COLUNAS = 8_000

# Faixas (y0, y1) de cada baia e dos corredores de empilhadeira
BAIAS = {"7D": (2_000, 13_000), "7C": (16_500, 25_500), "6N": (29_000, 40_000)}
CORREDORES = [(13_000, 16_500), (25_500, 29_000)]
X_FIM_CORREDOR = 52_000
NOME_LINHA = {"6N": "Linha 6N · Kit Poço", "7C": "Linha 7C · Kit Painéis/SFM",
              "7D": "Linha 7D · Kit Teto/Plataforma"}

X_ENTRADA = 2_500            # centro do ponto de recebimento de cada linha
X_AREA_KIT = 39_000          # início das áreas de kit
X_LIMITE_LINHA = 38_500      # nenhuma estação pode passar daqui (V&V)
ZONA_CONSOLIDACAO = (53_000, 62_500)

PALETE = (1_200, 1_000, 144)          # palete PBR (mm)
PASSO_BUFFER = (1_500, 1_300)         # passo das posições de palete no piso
LINHAS_BUFFER = 3                     # posições por coluna do buffer de estação
LINHAS_AREA_KIT = 5                   # posições por coluna da área de kits
FOLGA_FLUXO = 1_000                   # folga entre blocos consecutivos
FOLGA_BUFFER_POSTO = 600              # folga entre buffer e posto
DESLOC_RAMO = 2_500                   # deslocamento em y de ramos paralelos
DIST_OPERADOR = 550                   # centro do operador até a borda da bancada

# Critérios de V&V
CORREDOR_MIN = 3_000                  # largura livre mínima p/ empilhadeira
ALCANCE_MIN, ALCANCE_MAX = 1, 150     # folga mãos do operador ativo–tampo (mm)
VEL_TRANSPORTE = 1.0                  # m/s (carrinho / rebocador)
LIMITE_TRANSPORTE_TAKT = 0.05         # transporte ≤ 5% do takt -> DES pode desprezá-lo
TOL_INTERFERENCIA = 1_000.0           # mm³ (1 cm³)

# Dimensões dos postos manuais (comprimento em x, profundidade em y)
POSTO_DIM = {
    "BEOX-01": (3_600, 1_600), "BEOX-02": (3_000, 1_500), "BEOX-03": (3_000, 1_500),
    "BEOX-04": (3_600, 1_800), "BEOX-05": (3_200, 1_800),
    "BEOX-07": (3_400, 1_800), "BEOX-08": (3_000, 1_600),
}
# Células automáticas: (rótulo, comprimento, profundidade, altura, tipo)
CELULAS = {
    "MAQ-7C": [("SG1530 · puncionadeira", 4_200, 3_000, 2_300, "puncionadeira"),
               ("EBe · dobradeira de painéis", 3_200, 2_600, 2_500, "painel"),
               ("LVD · prensa dobradeira", 3_600, 1_900, 2_800, "prensa")],
    "MAQ-7D": [("Salvagnini · dobradeira de painéis", 4_600, 3_200, 2_700, "painel"),
               ("TOX · clinching", 2_200, 2_000, 2_900, "clinch")],
}
TRANSPORTADOR = 900                   # comprimento do transportador entre máquinas

# Paleta (RGB 0–1). As cores das linhas são as mesmas da animação da simulação.
def _hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


COR = {
    "6N": _hex("#2a6fd6"), "7C": _hex("#d64545"), "7D": _hex("#1f9a55"),
    "6N_ramo": _hex("#86aee9"), "7D_ramo": _hex("#7fcb9c"),
    "piso": _hex("#d4d3ce"), "zona6N": _hex("#d7e3f5"), "zona7C": _hex("#f3dcdc"), "zona7D": _hex("#d8ede0"), "parede": _hex("#e4e2dc"), "coluna": _hex("#5c6470"),
    "faixa": _hex("#f2c40f"), "bancada_tampo": _hex("#8e949c"), "bancada_estrutura": _hex("#3a4656"),
    "painel": _hex("#9aa7b8"), "gabarito": _hex("#4b5563"), "maquina": _hex("#e8eaec"),
    "maquina_base": _hex("#3b3f45"), "maquina_vidro": _hex("#a9d3f0"), "ihm": _hex("#1f2329"),
    "palete": _hex("#b88a55"), "mp": _hex("#9ca3af"), "colete_ativo": _hex("#f07b1a"),
    "colete_ocioso": _hex("#9aa0a6"), "calca": _hex("#27324a"), "pele": _hex("#e0b48f"),
    "capacete": _hex("#f5c518"), "andon_off": _hex("#4a4d52"), "verde": _hex("#22a447"),
    "amarelo": _hex("#f2b705"), "vermelho": _hex("#e02424"), "manut": _hex("#c0392b"),
    "cabine": _hex("#c7ccd3"), "cabine_porta": _hex("#8f98a3"), "rolo": _hex("#b4b9c0"),
}

# Categorias que ocupam volume físico (entram na checagem de interferência)
CATEGORIAS_SOLIDAS = {"estrutura", "bancada", "maquina", "operador", "kit", "andon",
                      "estoque", "produto", "ferramental"}
CATEGORIAS_EQUIPAMENTO = {"bancada", "maquina", "kit", "andon", "estoque", "produto", "ferramental"}


# =============================================================================
# 2. PRIMITIVAS B-REP
# =============================================================================
def caixa(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()


def caixa_c(cx: float, cy: float, z0: float, dx: float, dy: float, dz: float) -> TopoDS_Shape:
    """Caixa centrada em (cx, cy) no plano, base em z0."""
    return caixa(cx - dx / 2, cy - dy / 2, z0, dx, dy, dz)


def cilindro(x: float, y: float, z: float, r: float, h: float,
             d: tuple[float, float, float] = (0, 0, 1)) -> TopoDS_Shape:
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), gp_Dir(*d)), r, h).Shape()


def esfera(x: float, y: float, z: float, r: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), r).Shape()


def _lista(formas) -> TopTools_ListOfShape:
    lst = TopTools_ListOfShape()
    for f in formas:
        lst.Append(f)
    return lst


def _limpar(forma: TopoDS_Shape) -> TopoDS_Shape:
    """Une faces coplanares/cocilíndricas criadas pelas booleanas (topologia limpa)."""
    u = ShapeUpgrade_UnifySameDomain(forma, True, True, True)
    u.Build()
    return u.Shape()


def unir(*formas: TopoDS_Shape) -> TopoDS_Shape:
    """União booleana (BOP) de vários sólidos em um único sólido."""
    if len(formas) == 1:
        return formas[0]
    op = BRepAlgoAPI_Fuse()
    op.SetArguments(_lista(formas[:1]))
    op.SetTools(_lista(formas[1:]))
    op.SetFuzzyValue(1e-5)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("falha na união booleana")
    return _limpar(op.Shape())


def subtrair(base: TopoDS_Shape, *ferramentas: TopoDS_Shape) -> TopoDS_Shape:
    """Subtração booleana: base − ferramentas."""
    op = BRepAlgoAPI_Cut()
    op.SetArguments(_lista([base]))
    op.SetTools(_lista(ferramentas))
    op.Build()
    if not op.IsDone():
        raise RuntimeError("falha na subtração booleana")
    return _limpar(op.Shape())


def arredondar(forma: TopoDS_Shape, raio: float, so_verticais: bool = False) -> TopoDS_Shape:
    """Arredonda arestas (todas ou só as verticais). Mantém a forma se o fillet falhar."""
    mk = BRepFilletAPI_MakeFillet(forma)
    exp = TopExp_Explorer(forma, TopAbs_EDGE)
    n = 0
    while exp.More():
        aresta = topods.Edge(exp.Current())
        if so_verticais:
            c = BRepAdaptor_Curve(aresta)
            p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
            if abs(p0.X() - p1.X()) < 1e-6 and abs(p0.Y() - p1.Y()) < 1e-6:
                mk.Add(raio, aresta)
                n += 1
        else:
            mk.Add(raio, aresta)
            n += 1
        exp.Next()
    if n == 0:
        return forma
    try:
        mk.Build()
        return mk.Shape() if mk.IsDone() else forma
    except RuntimeError:
        return forma


def prisma(pontos: list[tuple[float, float]], z0: float, h: float) -> TopoDS_Shape:
    """Extruda um polígono fechado do plano XY (perfil -> face -> prisma)."""
    poly = BRepBuilderAPI_MakePolygon()
    for x, y in pontos:
        poly.Add(gp_Pnt(x, y, z0))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, h)).Shape()


def prisma_xz(pontos: list[tuple[float, float]], y0: float, esp: float) -> TopoDS_Shape:
    """Extruda em y um perfil desenhado no plano XZ (ex.: quadro em C de prensa)."""
    poly = BRepBuilderAPI_MakePolygon()
    for x, z in pontos:
        poly.Add(gp_Pnt(x, y0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, esp, 0)).Shape()


def posicionar(forma: TopoDS_Shape, dx: float, dy: float, dz: float = 0.0,
               ang_z: float = 0.0) -> TopoDS_Shape:
    """Rotação em torno de Z (graus) seguida de translação. Reaproveita a geometria
    (instância localizada), sem copiar a B-Rep — útil para paletes e operadores."""
    rot = gp_Trsf()
    rot.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(ang_z))
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    tr.Multiply(rot)
    return forma.Moved(TopLoc_Location(tr))


def composto(formas: list[TopoDS_Shape]) -> TopoDS_Compound:
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for f in formas:
        b.Add(comp, f)
    return comp


def volume(forma: TopoDS_Shape) -> float:
    p = GProp_GProps()
    brepgprop.VolumeProperties(forma, p)
    return p.Mass()


def caixa_limite(forma: TopoDS_Shape) -> tuple[float, float, float, float, float, float]:
    b = Bnd_Box()
    brepbndlib.AddOptimal(forma, b, False, False)
    return b.Get()


def contar_solidos(forma: TopoDS_Shape) -> int:
    exp, n = TopExp_Explorer(forma, TopAbs_SOLID), 0
    while exp.More():
        n += 1
        exp.Next()
    return n


# =============================================================================
# 3. MODELO DE DADOS DO CAD
# =============================================================================
@dataclass
class Peca:
    """Uma peça do modelo: sólido (ou composto de sólidos) + metadados."""
    nome: str
    forma: TopoDS_Shape
    cor: tuple[float, float, float]
    categoria: str
    grupo: str                       # conjunto físico (bancada, palete+carga, máquina…)
    linha: str | None = None
    estacao: str | None = None
    transparencia: float = 0.0


@dataclass
class Rotulo:
    pos: tuple[float, float, float]
    texto: str
    altura: float = 15.0
    destaque: bool = False
    vistas: str = "todas"                                   # todas | planta | 3d
    pos_planta: tuple[float, float, float] | None = None    # posição alternativa na planta


@dataclass
class No:
    """Nó do grafo de fluxo posicionado no galpão."""
    id: str
    tipo: str                        # entrada | estacao | maquina | kit | consolidacao
    linha: str | None
    x: float = 0.0
    y: float = 0.0
    comp: float = 0.0
    larg: float = 0.0
    n_buffer: int = 0
    buffer_x0: float = 0.0          # x da borda esquerda do buffer
    slots: list[tuple[float, float]] = field(default_factory=list)
    lado_operador: int = -1          # -1: operador do lado y−; +1: lado y+

    @property
    def x0(self) -> float:
        return self.x - self.comp / 2

    @property
    def x1(self) -> float:
        return self.x + self.comp / 2


@dataclass
class Cena:
    cen: sim.Cenario
    nos: dict[str, No]
    arestas: list[tuple[str, str]]
    rotas: dict[tuple[str, str], list[tuple[float, float]]]
    pecas: list[Peca]
    rotulos: list[Rotulo]
    snap: dict
    kpis: dict
    modelo: sim.Modelo
    kits_max: dict[str, int]


# =============================================================================
# 4. GRAFO DE FLUXO DO CAD (definido aqui; conferido contra o log da DES)
# =============================================================================
def grafo_fluxo(cen: sim.Cenario) -> list[tuple[str, str]]:
    if cen.beox04_paralelo:      # TO-BE: BEOX-04 fora do takt, em paralelo ao BEOX-01
        a = [("ENT-6N", "BEOX-01"), ("ENT-6N", "BEOX-04"), ("BEOX-01", "BEOX-02"),
             ("BEOX-02", "BEOX-03"), ("BEOX-03", "KIT-6N"), ("BEOX-04", "KIT-6N")]
    else:                        # AS-IS: BEOX-04 no fim da sequência
        a = [("ENT-6N", "BEOX-01"), ("BEOX-01", "BEOX-02"), ("BEOX-02", "BEOX-03"),
             ("BEOX-03", "BEOX-04"), ("BEOX-04", "KIT-6N")]
    a += [("ENT-7C", "MAQ-7C"), ("MAQ-7C", "BEOX-05"), ("BEOX-05", "KIT-7C"),
          ("ENT-7D", "MAQ-7D"), ("MAQ-7D", "BEOX-07"), ("MAQ-7D", "BEOX-08"),
          ("BEOX-07", "KIT-7D"), ("BEOX-08", "KIT-7D"),
          ("KIT-6N", "CONSOLIDACAO"), ("KIT-7C", "CONSOLIDACAO"), ("KIT-7D", "CONSOLIDACAO")]
    return a


# Origem de cada "token" do log da simulação (ver eox_simulacao.py)
ORIGEM_TOKEN = {0: "ENT-6N", 1: "ENT-6N", 2: "ENT-7C", 3: "ENT-7D", 4: "MAQ-7D", 5: "MAQ-7D"}
LINHA_TOKEN = {0: "6N", 1: "6N", 2: "7C", 3: "7D", 4: "7D", 5: "7D"}


def tokens_do_log(m: sim.Modelo) -> dict[tuple[int, int], list[list]]:
    tk: dict[tuple[int, int], list[list]] = defaultdict(list)
    for reg in m.ops:
        tk[(reg[0], reg[1])].append(reg)
    for lst in tk.values():
        lst.sort(key=lambda r: r[3])
    return tk


def arestas_da_simulacao(m: sim.Modelo) -> set[tuple[str, str]]:
    """Precedências realmente percorridas pelos kits no log de eventos da DES."""
    obs: set[tuple[str, str]] = set()
    for (j, tok), ops in tokens_do_log(m).items():
        ant = ORIGEM_TOKEN[tok]
        for reg in ops:
            est = sim.ESTACOES[reg[2]]
            obs.add((ant, est))
            ant = est
        if tok != 3 and ops[-1][5] is not None:
            obs.add((ant, "KIT-" + LINHA_TOKEN[tok]))
    for c in m.conclusao:
        if c is not None:
            obs |= {("KIT-6N", "CONSOLIDACAO"), ("KIT-7C", "CONSOLIDACAO"), ("KIT-7D", "CONSOLIDACAO")}
            break
    return obs


# =============================================================================
# 5. LEITURA DO ESTADO SIMULADO (snapshot e dimensionamento)
# =============================================================================
def _inicio_area_kit(ops_linha: list[list[list]]) -> float:
    """Instante em que o primeiro ramo de um kit chega à área de kits."""
    fins = [ops[-1][5] for ops in ops_linha if ops and ops[-1][5] is not None]
    return min(fins) if fins else math.inf


def intervalos_area_kit(m: sim.Modelo) -> dict[str, list[tuple[int, float, float]]]:
    """(obra, entrada, saída) de cada palete de kit na área de kits, por linha.
    Um kit ocupa 1 posição desde que o 1º ramo termina até o elevador ser concluído."""
    tk = tokens_do_log(m)
    por_linha: dict[str, list[tuple[int, float, float]]] = {"6N": [], "7C": [], "7D": []}
    for j in range(len(m.chegada)):
        c = m.conclusao[j] if m.conclusao[j] is not None else math.inf
        for linha, toks in (("6N", (0, 1)), ("7C", (2,)), ("7D", (4, 5))):
            t0 = _inicio_area_kit([tk.get((j, t), []) for t in toks])
            if t0 < math.inf:
                por_linha[linha].append((j, t0, c))
    return por_linha


def maximo_area_kit(m: sim.Modelo) -> dict[str, int]:
    a, b = sim.T_AQUEC, sim.T_FIM
    res = {}
    for linha, ivs in intervalos_area_kit(m).items():
        eventos = sorted([(t0, 1) for _, t0, _ in ivs] + [(t1, -1) for _, _, t1 in ivs if t1 < math.inf],
                         key=lambda e: (e[0], e[1]))
        n = mx = 0
        for t, d in eventos:
            if t >= a and mx == 0:
                mx = n                      # nível já existente no início da medição
            n += d
            if a <= t < b:
                mx = max(mx, n)
        res[linha] = mx
    return res


def instante_wip_maximo(m: sim.Modelo) -> float:
    """Meio do primeiro intervalo, dentro da medição, em que o WIP atinge o máximo."""
    t = np.asarray(m.wip.t)
    v = np.asarray(m.wip.v, dtype=float)
    fim = np.append(t[1:], sim.T_FIM)
    lo, hi = np.maximum(t, sim.T_AQUEC), np.minimum(fim, sim.T_FIM)
    ok = hi > lo
    alvo = v[ok].max()
    i = np.flatnonzero(ok & (v == alvo))[0]
    return float((lo[i] + hi[i]) / 2)


def snapshot(m: sim.Modelo, t: float) -> dict:
    """Estado de cada kit no instante t, lido do log de eventos da simulação."""
    fila: dict[str, list[tuple[int, int]]] = defaultdict(list)
    proc: dict[str, tuple[int, int]] = {}
    for (j, tok), ops in tokens_do_log(m).items():
        for reg in ops:
            tq, ts, te = reg[3], reg[4], reg[5]
            if tq <= t and (te is None or t < te):
                est = sim.ESTACOES[reg[2]]
                if ts is None or t < ts:
                    fila[est].append((j, tok, tq))
                else:
                    proc[est] = (j, tok)
                break
    for est in fila:
        fila[est] = [(j, tok) for j, tok, _ in sorted(fila[est], key=lambda r: (r[2], r[0]))]
    tk = tokens_do_log(m)
    area_kit: dict[str, list[tuple[int, list[int]]]] = {"6N": [], "7C": [], "7D": []}
    for linha, ivs in intervalos_area_kit(m).items():
        toks = {"6N": (0, 1), "7C": (2,), "7D": (4, 5)}[linha]
        for j, t0, t1 in ivs:
            if t0 <= t < t1:
                presentes = [tok for tok in toks if tk.get((j, tok)) and tk[(j, tok)][-1][5] is not None
                             and tk[(j, tok)][-1][5] <= t]
                area_kit[linha].append((j, presentes))
    obras = {j for lst in fila.values() for j, _ in lst} | {j for j, _ in proc.values()} \
        | {j for lst in area_kit.values() for j, _ in lst}
    em_reparo = any(t0 <= t < t1 for t0, t1 in m.reparos)
    return {"t": t, "fila": dict(fila), "proc": proc, "area_kit": area_kit,
            "obras": sorted(obras), "wip_sim": int(m.wip.amostrar(np.array([t]))[0]),
            "reparo_05": em_reparo}


# =============================================================================
# 6. MOTOR DE LAYOUT — posiciona os nós a partir do grafo e das filas simuladas
# =============================================================================
def _profundidades(arestas: list[tuple[str, str]], linha_de: dict[str, str]) -> dict[str, int]:
    """Profundidade = maior caminho a partir da entrada da linha (ordem topológica)."""
    prof: dict[str, int] = {}
    pred = defaultdict(list)
    for a, b in arestas:
        pred[b].append(a)

    def p(n: str) -> int:
        if n not in prof:
            prof[n] = 0 if not pred[n] or n.startswith("ENT") else 1 + max(p(q) for q in pred[n])
        return prof[n]
    for n in linha_de:
        p(n)
    return prof


def _slots_buffer(x_dir: float, y: float, n: int, linhas: int) -> list[tuple[float, float]]:
    """Posições de palete à esquerda de x_dir, preenchidas a partir do posto (FIFO)."""
    px, py = PASSO_BUFFER
    rows = min(linhas, max(n, 1))
    slots = []
    for k in range(n):
        col, row = divmod(k, rows)
        slots.append((x_dir - px / 2 - col * px, y + (row - (rows - 1) / 2) * py))
    return slots


def planejar_layout(cen: sim.Cenario, kpis: dict, kits_max: dict[str, int]) -> dict[str, No]:
    arestas = grafo_fluxo(cen)
    nos: dict[str, No] = {}
    for linha, (y0, y1) in BAIAS.items():
        nos[f"ENT-{linha}"] = No(f"ENT-{linha}", "entrada", linha, X_ENTRADA, (y0 + y1) / 2, 2_000, 2_400)
        nos[f"KIT-{linha}"] = No(f"KIT-{linha}", "kit", linha)
    for est in sim.ESTACOES:
        linha = sim.LINHA_DA_ESTACAO[est]
        if est in CELULAS:
            comp = sum(m[1] for m in CELULAS[est]) + TRANSPORTADOR * (len(CELULAS[est]) - 1)
            larg = max(m[2] for m in CELULAS[est])
            nos[est] = No(est, "maquina", linha, comp=comp, larg=larg)
        else:
            nos[est] = No(est, "estacao", linha, comp=POSTO_DIM[est][0], larg=POSTO_DIM[est][1])
    nos["CONSOLIDACAO"] = No("CONSOLIDACAO", "consolidacao", None,
                             sum(ZONA_CONSOLIDACAO) / 2, GALPAO_Y / 2,
                             ZONA_CONSOLIDACAO[1] - ZONA_CONSOLIDACAO[0], 38_000)

    # deslocamento em y: sucessores de uma bifurcação vão para ramos separados
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in arestas:
        succ[a].append(b)
        pred[b].append(a)
    desloc: dict[str, float] = {}
    for a in sorted(succ):
        alvos = [b for b in succ[a] if nos[b].tipo in ("estacao", "maquina")]
        if len(alvos) > 1:
            for k, b in enumerate(sorted(alvos)):
                desloc[b] = DESLOC_RAMO * (1 - 2 * k / (len(alvos) - 1))
    linha_de = {n: nos[n].linha for n in nos if nos[n].tipo in ("estacao", "maquina", "entrada")}
    prof = _profundidades([e for e in arestas if nos[e[1]].tipo in ("estacao", "maquina")], linha_de)
    for n in sorted(linha_de, key=lambda k: prof.get(k, 0)):
        if nos[n].tipo == "entrada":
            continue
        if n not in desloc:
            desloc[n] = desloc.get(pred[n][0], 0.0) if pred[n] and nos[pred[n][0]].tipo != "entrada" else 0.0

    # x: blocos [buffer | posto] por nível de profundidade, da entrada para a área de kits
    for linha, (y0, y1) in BAIAS.items():
        yc = (y0 + y1) / 2
        x_dir = nos[f"ENT-{linha}"].x1
        niveis = defaultdict(list)
        for n, l in linha_de.items():
            if l == linha and nos[n].tipo != "entrada":
                niveis[prof[n]].append(n)
        for d in sorted(niveis):
            grupo = niveis[d]
            for n in grupo:
                nos[n].n_buffer = max(2, int(math.ceil(kpis["fila_max"][n])))
            col_buf = max(math.ceil(nos[n].n_buffer / LINHAS_BUFFER) for n in grupo)
            x_buf0 = x_dir + FOLGA_FLUXO
            x_posto0 = x_buf0 + col_buf * PASSO_BUFFER[0] + FOLGA_BUFFER_POSTO
            for n in grupo:
                no = nos[n]
                no.y = yc + desloc.get(n, 0.0)
                no.x = x_posto0 + no.comp / 2
                no.buffer_x0 = x_buf0
                no.slots = _slots_buffer(x_posto0 - FOLGA_BUFFER_POSTO, no.y, no.n_buffer, LINHAS_BUFFER)
                no.lado_operador = 1 if desloc.get(n, 0.0) < 0 and linha == "6N" else -1
            x_dir = max(nos[n].x1 for n in grupo)

        # área de kits dimensionada pelo máximo simulado de kits aguardando
        kit = nos[f"KIT-{linha}"]
        kit.n_buffer = max(4, kits_max[linha])
        cols = math.ceil(kit.n_buffer / LINHAS_AREA_KIT)
        kit.comp = cols * PASSO_BUFFER[0]
        kit.larg = LINHAS_AREA_KIT * PASSO_BUFFER[1]
        kit.x, kit.y = X_AREA_KIT + kit.comp / 2, yc
        kit.slots = [(X_AREA_KIT + PASSO_BUFFER[0] / 2 + (k // LINHAS_AREA_KIT) * PASSO_BUFFER[0],
                      yc + (k % LINHAS_AREA_KIT - (LINHAS_AREA_KIT - 1) / 2) * PASSO_BUFFER[1])
                     for k in range(kit.n_buffer)]
    return nos


def rotear(nos: dict[str, No], a: str, b: str) -> list[tuple[float, float]]:
    """Rota ortogonal (Manhattan) entre a saída de `a` e a entrada do bloco de `b`."""
    na, nb = nos[a], nos[b]
    x0, y0 = (na.x1 + 150, na.y)
    if nb.tipo == "kit":
        x1 = nb.x0 - 150
        y1 = min(max(y0, nb.y - nb.larg / 2 + 500), nb.y + nb.larg / 2 - 500)
    elif nb.tipo == "consolidacao":
        x1, y1 = nb.x0 - 150, y0
    else:
        x1, y1 = nb.buffer_x0 - 150, nb.y
    if abs(y1 - y0) < 1:
        return [(x0, y0), (x1, y1)]
    xm = max(x0, min(x0 + 800.0, x1 - 600.0))
    return [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]


def comprimento_rota(rota: list[tuple[float, float]]) -> float:
    return sum(abs(x1 - x0) + abs(y1 - y0) for (x0, y0), (x1, y1) in zip(rota, rota[1:]))


def distancia_percorrida(nos: dict[str, No], a: str, b: str, rota: list[tuple[float, float]]) -> float:
    """Percurso do kit de centro a centro: saída do nó `a` + rota + travessia do
    buffer até o centro de `b` (Manhattan)."""
    (xs, ys), (xe, ye) = rota[0], rota[-1]
    na, nb = nos[a], nos[b]
    return (abs(xs - na.x) + abs(ys - na.y) + comprimento_rota(rota)
            + abs(nb.x - xe) + (abs(nb.y - ye) if nb.tipo != "consolidacao" else 0.0))


# =============================================================================
# 7. BIBLIOTECA DE COMPONENTES (cada função devolve sólidos B-Rep)
# =============================================================================
class Biblioteca:
    """Gera (e guarda em cache) as geometrias-modelo; as cenas usam instâncias
    localizadas delas (mesma B-Rep, outra posição) — como blocos de um CAD."""

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def _memo(self, chave: str, fabrica):
        if chave not in self._cache:
            self._cache[chave] = fabrica()
        return self._cache[chave]

    # ---- palete PBR com entradas de garfo (subtração booleana) ---------------
    def palete(self) -> TopoDS_Shape:
        def f():
            lx, ly, h = PALETE
            base = caixa_c(0, 0, 0, lx, ly, h)
            garfos = [caixa_c(0, s * 260, 0, lx + 10, 300, 100) for s in (-1, 1)]
            return subtrair(base, *garfos)
        return self._memo("palete", f)

    # ---- cargas de kit (por linha e por ramo) --------------------------------
    def carga(self, tok: int) -> TopoDS_Shape:
        z0 = PALETE[2]

        def f():
            # cada ramo ocupa metade do palete, para o kit poder ficar parcial
            if tok == 0:      # kit poço: feixe de guias (metade y−)
                guias = [caixa_c(0, y, z0, 1_100, 90, 90) for y in (-420, -290, -160)]
                topo = [caixa_c(0, y, z0 + 90, 1_100, 90, 90) for y in (-355, -225)]
                return unir(*guias, *topo)
            if tok == 1:      # ramo BEOX-04: caixa de componentes de poço (metade y+)
                return arredondar(caixa_c(0, 250, z0, 900, 420, 300), 25)
            if tok == 2:      # kit painéis: pilha de chapas com ranhuras
                pilha = caixa_c(0, 0, z0, 1_100, 900, 300)
                ranh = [caixa(-560, -460, z0 + k * 50, 1_120, 920, 6) for k in range(1, 6)]
                return subtrair(pilha, *ranh)
            if tok == 4:      # ramo BEOX-07: painéis de teto (metade y−)
                return arredondar(caixa_c(0, -250, z0, 1_100, 460, 160), 15)
            if tok == 5:      # ramo BEOX-08: plataforma (metade y+)
                return arredondar(caixa_c(0, 260, z0, 1_000, 420, 120), 15)
            if tok == 3:      # chapa na máquina 7D
                return caixa_c(0, 0, 0, 1_100, 900, 40)
            raise ValueError(tok)
        return self._memo(f"carga{tok}", f)

    def materia_prima(self) -> TopoDS_Shape:
        return self._memo("mp", lambda: subtrair(caixa_c(0, 0, PALETE[2], 1_100, 900, 420),
                                                  *[caixa(-560, -460, PALETE[2] + k * 70, 1_120, 920, 5)
                                                    for k in range(1, 6)]))

    # ---- operador (boneco) ----------------------------------------------------
    def operador(self, ativo: bool) -> dict[str, TopoDS_Shape]:
        """Operador de frente para +y, pés em z=0. Quatro sólidos: pernas, tronco
        com braços (colete), cabeça com pescoço e capacete oco com aba."""
        def f():
            pernas = unir(cilindro(-110, 0, 70, 75, 790), cilindro(110, 0, 70, 75, 790),
                          caixa_c(-110, 60, 0, 130, 280, 70), caixa_c(110, 60, 0, 130, 280, 70))
            tronco = arredondar(caixa_c(0, 0, 860, 420, 240, 620), 60)
            if ativo:   # braços estendidos para a bancada
                d = (0.0, 0.75, -0.66)
                n = math.sqrt(sum(c * c for c in d))
                d = tuple(c / n for c in d)
            else:       # braços ao longo do corpo
                d = (0.0, 0.05, -0.999)
            # ombros: barra abaixo da zona arredondada do tronco (interseções só
            # entre faces planas e cilíndricas -> B-Rep robusta no STEP)
            L = 580
            bracos = [caixa_c(0, 0, 1_290, 620, 200, 110)]
            for sx in (-1, 1):
                ox, oy, oz = sx * 272, 0.0, 1_345   # braço não tangencia o tronco
                bracos.append(cilindro(ox, oy, oz, 52, L, d))
                bracos.append(esfera(ox + d[0] * (L + 30), oy + d[1] * (L + 30), oz + d[2] * (L + 30), 62))
            colete = unir(tronco, *bracos)
            cabeca = unir(cilindro(0, 0, 1_480, 55, 70), esfera(0, 0, 1_640, 115))
            cupula = subtrair(esfera(0, 0, 1_660, 132), esfera(0, 0, 1_660, 120),
                              caixa_c(0, 0, 1_300, 400, 400, 360))
            aba = subtrair(cilindro(0, 20, 1_650, 165, 16), cilindro(0, 0, 1_640, 121, 40))
            capacete = unir(cupula, aba)
            return {"pernas": pernas, "colete": colete, "cabeca": cabeca, "capacete": capacete}
        return self._memo(f"operador{int(ativo)}", f)  # type: ignore[return-value]

    # ---- coluna metálica perfil I (HEB 300) ------------------------------------
    def coluna(self) -> TopoDS_Shape:
        def f():
            b, h, tw, tf = 300.0, 300.0, 11.0, 19.0
            perfil = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, -h / 2 + tf), (tw / 2, -h / 2 + tf),
                      (tw / 2, h / 2 - tf), (b / 2, h / 2 - tf), (b / 2, h / 2), (-b / 2, h / 2),
                      (-b / 2, h / 2 - tf), (-tw / 2, h / 2 - tf), (-tw / 2, -h / 2 + tf), (-b / 2, -h / 2 + tf)]
            return unir(caixa_c(0, 0, 0, 500, 500, 25), prisma(perfil, 25, ALTURA_COLUNA - 25))
        return self._memo("coluna", f)


# =============================================================================
# 8. CONSTRUÇÃO DA CENA
# =============================================================================
class Construtor:
    def __init__(self, cen: sim.Cenario, nos: dict[str, No], kpis: dict, snap: dict,
                 bib: Biblioteca) -> None:
        self.cen, self.nos, self.kpis, self.snap, self.bib = cen, nos, kpis, snap, bib
        self.pecas: list[Peca] = []
        self.rotulos: list[Rotulo] = []

    def add(self, nome, forma, cor, categoria, grupo, linha=None, estacao=None, transp=0.0):
        self.pecas.append(Peca(nome, forma, cor, categoria, grupo, linha, estacao, transp))

    # ---------------------------------------------------------------- galpão
    def galpao(self) -> None:
        self.add("Piso industrial", caixa(0, 0, -200, GALPAO_X, GALPAO_Y, 200), COR["piso"], "piso", "galpao")
        # paredes do fundo com portas de recebimento (subtração) alinhadas às entradas
        portas = [caixa_c(100, (y0 + y1) / 2, 0, 400, 4_000, 4_200) for y0, y1 in BAIAS.values()]
        parede_x = subtrair(caixa(0, 0, 0, 200, GALPAO_Y - 200, ALTURA_PAREDE), *portas)
        self.add("Parede oeste (recebimento)", parede_x, COR["parede"], "estrutura", "parede_o")
        self.add("Parede norte", caixa(0, GALPAO_Y - 200, 0, GALPAO_X, 200, ALTURA_PAREDE),
                 COR["parede"], "estrutura", "parede_n")
        col = self.bib.coluna()
        k = 0
        for x in range(PASSO_COLUNAS, GALPAO_X, PASSO_COLUNAS):
            for y in (500, GALPAO_Y - 650):
                k += 1
                self.add(f"Coluna HEB300 C{k:02d}", posicionar(col, x, y), COR["coluna"], "estrutura", f"col{k}")
        for y in range(8_400, GALPAO_Y - 4_000, 8_400):
            for x in (550, GALPAO_X - 500):
                if any(y0 - 2_500 < y < y1 + 2_500 for y0, y1 in BAIAS.values()) and x < 1_000:
                    continue   # não bloquear as portas de recebimento
                k += 1
                self.add(f"Coluna HEB300 C{k:02d}", posicionar(col, x, y), COR["coluna"], "estrutura", f"col{k}")
        # zonas das baias (pintura de piso) e faixas amarelas dos corredores
        for linha, (y0, y1) in BAIAS.items():
            self.add(f"Zona {linha}", caixa(1_000, y0, 0, X_FIM_CORREDOR - 1_000, y1 - y0, 2),
                     COR["zona" + linha], "zona", f"zona{linha}", linha)
        for y0, y1 in CORREDORES:
            for y in (y0, y1 - 100):
                self.add(f"Faixa de corredor y={y}", caixa(1_000, y, 0, X_FIM_CORREDOR - 1_000, 100, 4),
                         COR["faixa"], "marcacao", f"faixa{y}")
        x0, x1 = ZONA_CONSOLIDACAO
        moldura = subtrair(caixa(x0, 2_000, 0, x1 - x0, 38_000, 4),
                           caixa(x0 + 120, 2_120, -1, x1 - x0 - 240, 37_760, 6))
        self.add("Zona de consolidação e expedição", moldura, COR["faixa"], "marcacao", "consol")
        for linha, (y0, y1) in BAIAS.items():
            # nome da linha no corredor/passarela logo acima da baia (área sempre livre)
            self.rotulos.append(Rotulo((6_000, y1 + 750, 50), NOME_LINHA[linha], 16, True, "planta"))

    # ------------------------------------------------------ marcações de posição
    def _posicao_piso(self, x: float, y: float, nome: str, grupo: str) -> None:
        lx, ly = PALETE[0] + 120, PALETE[1] + 120
        quadro = subtrair(caixa_c(x, y, 0, lx, ly, 4), caixa_c(x, y, -1, lx - 100, ly - 100, 6))
        self.add(nome, quadro, COR["faixa"], "buffer", grupo)

    def _palete_com_carga(self, x: float, y: float, toks: list[int], obra: int, onde: str,
                          linha: str) -> None:
        g = f"kit_{linha}_{obra}_{onde}"
        self.add(f"Palete obra #{obra} ({onde})", posicionar(self.bib.palete(), x, y), COR["palete"],
                 "kit", g, linha)
        for tok in toks:
            cor = {0: COR["6N"], 1: COR["6N_ramo"], 2: COR["7C"], 4: COR["7D"], 5: COR["7D_ramo"]}[tok]
            self.add(f"Kit {linha} obra #{obra} (token {tok})", posicionar(self.bib.carga(tok), x, y),
                     cor, "kit", g, linha)

    # ------------------------------------------------------------------ nós
    def entradas(self) -> None:
        for linha in BAIAS:
            no = self.nos[f"ENT-{linha}"]
            for k, dy in enumerate((-650, 650)):
                self._posicao_piso(no.x, no.y + dy, f"Posição recebimento {linha}-{k}", f"ent{linha}")
                g = f"mp_{linha}_{k}"
                self.add(f"Palete matéria-prima {linha}-{k}", posicionar(self.bib.palete(), no.x, no.y + dy),
                         COR["palete"], "estoque", g, linha)
                self.add(f"Matéria-prima {linha}-{k}", posicionar(self.bib.materia_prima(), no.x, no.y + dy),
                         COR["mp"], "estoque", g, linha)
            self.rotulos.append(Rotulo((no.x, no.y - 1_700, 50), f"Recebimento {linha}", 11, vistas="planta"))

    def _cor_andon(self, est: str) -> tuple[str, tuple[float, float, float]]:
        u = self.kpis["util"][est]
        if u >= 0.95:
            return "vermelho", COR["vermelho"]
        if u >= 0.85:
            return "amarelo", COR["amarelo"]
        return "verde", COR["verde"]

    def posto_manual(self, est: str) -> None:
        no = self.nos[est]
        g = f"posto_{est}"
        lado = no.lado_operador                    # lado do operador (−1: y−, +1: y+)
        L, P, H = no.comp, no.larg, 900.0
        # bancada: tampo arredondado + pés + travessa inferior (união booleana)
        tampo = arredondar(caixa_c(no.x, no.y, H - 50, L, P, 50), 12)
        pes = [caixa_c(no.x + sx * (L / 2 - 90), no.y + sy * (P / 2 - 90), 0, 60, 60, H - 50)
               for sx in (-1, 1) for sy in (-1, 1)]
        travessas = [caixa_c(no.x, no.y + sy * (P / 2 - 90), 200, L - 180, 40, 40) for sy in (-1, 1)]
        bancada = unir(tampo, *pes, *travessas)
        self.add(f"{est} · bancada", bancada, COR["bancada_estrutura"], "bancada", g, no.linha, est)
        # painel de ferramentas no lado oposto ao operador
        y_painel = no.y - lado * (P / 2 - 25)
        self.add(f"{est} · painel de ferramentas", caixa_c(no.x, y_painel, H, L - 200, 50, 1_100),
                 COR["painel"], "bancada", g, no.linha, est)
        # gabarito de montagem (dispositivo) sobre o tampo
        gab = unir(caixa_c(no.x, no.y, H, L * 0.62, P * 0.55, 30),
                   *[caixa_c(no.x + s * L * 0.28, no.y, H + 30, 80, P * 0.5, 120) for s in (-1, 1)])
        self.add(f"{est} · gabarito", gab, COR["gabarito"], "bancada", g, no.linha, est)
        # andon: cor ativa dada pela utilização simulada
        nome_cor, cor_on = self._cor_andon(est)
        ax, ay = no.x + L / 2 - 180, y_painel + lado * 120
        self.add(f"{est} · andon haste", unir(cilindro(ax, ay, H, 60, 15), cilindro(ax, ay, H + 15, 18, 700)),
                 COR["andon_off"], "andon", g, no.linha, est)
        for k, (nome, cor) in enumerate((("verde", COR["verde"]), ("amarelo", COR["amarelo"]),
                                         ("vermelho", COR["vermelho"]))):
            ligado = nome == nome_cor
            self.add(f"{est} · andon {nome}{' (aceso)' if ligado else ''}",
                     cilindro(ax, ay, H + 715 + k * 95, 48, 90), cor if ligado else COR["andon_off"],
                     "andon", g, no.linha, est)
        u = self.kpis["util"][est]
        texto = f"{est} · μ {sim.DISTRIBUICOES[est].media_teorica:.1f} min · U {u:.0%}".replace(".", ",")
        y_pl = no.y - lado * (P / 2 + 650)
        self.rotulos.append(Rotulo((no.x, no.y, 2_650), texto, 13, True, pos_planta=(no.x, y_pl, 50)))
        self.rotulos.append(Rotulo((no.x, y_pl, 50), f"{est} · U {u:.0%}", 12, True, "planta"))

    def celula(self, est: str) -> None:
        no = self.nos[est]
        g = f"celula_{est}"
        x = no.x0
        maquinas = CELULAS[est]
        for k, (nome, comp, larg, alt, tipo) in enumerate(maquinas):
            cx = x + comp / 2
            self._maquina(cx, no.y, comp, larg, alt, tipo, f"{est} · {nome}", g, no.linha, est)
            x += comp
            if k < len(maquinas) - 1:
                self._transportador(x, no.y, TRANSPORTADOR, 1_200, f"{est} · transportador {k + 1}", g, no.linha, est)
                x += TRANSPORTADOR
        u = self.kpis["util"][est]
        self.rotulos.append(Rotulo((no.x, no.y, 3_400),
                                   f"Máq. Auto {no.linha} ({len(maquinas)} máquinas) · U {u:.0%}", 13, True,
                                   pos_planta=(no.x, no.y + no.larg / 2 + 500, 50)))

    def _maquina(self, cx, cy, comp, larg, alt, tipo, nome, g, linha, est) -> None:
        base = caixa_c(cx, cy, 0, comp, larg, 200)
        self.add(f"{nome} · base", base, COR["maquina_base"], "maquina", g, linha, est)
        if tipo == "prensa":
            # quadro em C desenhado no plano XZ e extrudado (garganta da dobradeira)
            x0, x1 = cx - comp / 2, cx + comp / 2
            perfil = [(x0, 200), (x1, 200), (x1, alt), (x0, alt), (x0, 1_500), (x0 + comp * 0.45, 1_500),
                      (x0 + comp * 0.45, 1_050), (x0, 1_050)]
            corpo = prisma_xz(perfil, cy - larg / 2, larg)
            corpo = subtrair(corpo, caixa_c(cx, cy, 1_050, comp * 0.92, larg * 0.6, 450))
        else:
            corpo = arredondar(caixa_c(cx, cy, 200, comp, larg, alt - 200), 70, so_verticais=True)
            janela = caixa_c(cx, cy - larg / 2 + 60, 900, comp * 0.6, 160, alt * 0.35)
            corpo = subtrair(corpo, janela)
            vidro = caixa_c(cx, cy - larg / 2 + 120, 900, comp * 0.6, 20, alt * 0.35)
            self.add(f"{nome} · visor", vidro, COR["maquina_vidro"], "maquina", g, linha, est, transp=0.45)
        if tipo == "puncionadeira":
            corpo = unir(corpo, cilindro(cx, cy, alt, comp * 0.18, 220))       # torre de ferramentas
        if tipo == "clinch":
            corpo = unir(corpo, cilindro(cx, cy, alt, 260, 350))               # cilindro da prensa
        self.add(f"{nome} · corpo", corpo, COR["maquina"], "maquina", g, linha, est)
        faixa = caixa_c(cx, cy - larg / 2 - 1, 210 + 60, comp * 0.9, 2, 120)
        self.add(f"{nome} · faixa", faixa, COR[linha], "maquina", g, linha, est)
        # IHM (painel de comando) em pedestal à frente da máquina
        px, py = cx + comp / 2 - 400, cy - larg / 2 - 450
        self.add(f"{nome} · IHM", unir(cilindro(px, py, 0, 70, 1_100), caixa_c(px, py, 1_100, 450, 120, 380)),
                 COR["ihm"], "maquina", g, linha, est)

    def _transportador(self, x0, cy, comp, larg, nome, g, linha, est) -> None:
        h = 850.0
        trilhos = [caixa(x0, cy + s * (larg / 2) - (0 if s < 0 else 60), h - 120, comp, 60, 120) for s in (-1, 1)]
        pes = [caixa_c(x0 + dx, cy + s * (larg / 2 - 30), 0, 60, 60, h - 120)
               for dx in (100, comp - 100) for s in (-1, 1)]
        self.add(f"{nome} · estrutura", unir(*trilhos, *pes), COR["maquina_base"], "maquina", g, linha, est)
        rolos = [cilindro(x0 + 75 + k * 150, cy - larg / 2 + 60, h - 40, 30, larg - 120, (0, 1, 0))
                 for k in range(int((comp - 75) // 150))]
        self.add(f"{nome} · rolos", composto(rolos), COR["rolo"], "maquina", g, linha, est)

    # ------------------------------------------------------------- buffers e kits
    def buffers_e_kits(self) -> None:
        snap = self.snap
        for est in sim.ESTACOES:
            no = self.nos[est]
            for k, (x, y) in enumerate(no.slots):
                self._posicao_piso(x, y, f"Buffer {est} posição {k + 1}", f"buf_{est}")
            for k, (j, tok) in enumerate(snap["fila"].get(est, [])):
                x, y = no.slots[k]
                self._palete_com_carga(x, y, [tok] if tok != 3 else [], j, f"fila {est}", no.linha)
            if est in snap["proc"]:
                j, tok = snap["proc"][est]
                g = f"kit_{no.linha}_{j}_proc"
                if no.tipo == "maquina":
                    # chapa em processo: palete de alimentação à frente da 1ª máquina
                    m0 = CELULAS[est][0]
                    xm, ym = no.x0 + m0[1] / 2, no.y - m0[2] / 2 - 750
                    self.add(f"Palete obra #{j} em processo em {est}", posicionar(self.bib.palete(), xm, ym),
                             COR["palete"], "kit", g, no.linha, est)
                    carga, z = (self.bib.carga(3), PALETE[2]) if tok == 3 else (self.bib.carga(tok), 0.0)
                    self.add(f"Chapa obra #{j} em {est}", posicionar(carga, xm, ym, z),
                             COR["7D"] if tok == 3 else COR["7C"], "kit", g, no.linha, est)
                else:
                    cor = {0: COR["6N"], 1: COR["6N_ramo"], 2: COR["7C"], 4: COR["7D"], 5: COR["7D_ramo"]}[tok]
                    self.add(f"Kit obra #{j} em processo em {est}",
                             posicionar(self.bib.carga(tok), no.x, no.y, 930 - PALETE[2]),
                             cor, "kit", g, no.linha, est)
            n_fila = len(snap["fila"].get(est, []))
            if no.n_buffer:
                xs = [x for x, _ in no.slots]
                topo = no.y + min(LINHAS_BUFFER, no.n_buffer) * PASSO_BUFFER[1] / 2 + 250
                self.rotulos.append(Rotulo(((min(xs) + max(xs)) / 2, topo, 60),
                                           f"fila {n_fila} / {no.n_buffer} pos.", 11))
        for linha in BAIAS:
            no = self.nos[f"KIT-{linha}"]
            for k, (x, y) in enumerate(no.slots):
                self._posicao_piso(x, y, f"Área de kits {linha} posição {k + 1}", f"area_{linha}")
            for k, (j, toks) in enumerate(snap["area_kit"][linha]):
                x, y = no.slots[k]
                self._palete_com_carga(x, y, toks, j, "área de kits", linha)
            self.rotulos.append(Rotulo((no.x, no.y + no.larg / 2 + 350, 60),
                                       f"Área de kits {linha}: {len(snap['area_kit'][linha])}/{no.n_buffer} posições", 12, True))

    # ---------------------------------------------------------------- operadores
    def _operador(self, x: float, y: float, ang: float, ativo: bool, nome: str, linha: str, est: str | None) -> None:
        partes = self.bib.operador(ativo)
        g = f"op_{nome}"
        cores = {"pernas": COR["calca"], "colete": COR["colete_ativo"] if ativo else COR["colete_ocioso"],
                 "cabeca": COR["pele"], "capacete": COR["capacete"]}
        for parte, forma in partes.items():
            self.add(f"{nome} · {parte}", posicionar(forma, x, y, 0, ang), cores[parte], "operador", g, linha, est)

    def operadores(self) -> dict[str, int]:
        """Posiciona os operadores conforme o estado simulado em t*: a 6N usa um pool
        (ocupados nos postos em processo, livres no ponto de espera); 7C e 7D têm
        operador dedicado em cada posto manual."""
        cont = {"6N": 0, "7C": 0, "7D": 0}
        proc = self.snap["proc"]
        postos_6n = [e for e in ("BEOX-01", "BEOX-02", "BEOX-03", "BEOX-04")]
        ocupados = [e for e in postos_6n if e in proc][: self.cen.operadores_6n]
        livres = self.cen.operadores_6n - len(ocupados)
        k = 0
        for est in postos_6n:
            if est in ocupados:
                no = self.nos[est]
                y = no.y + no.lado_operador * (no.larg / 2 + DIST_OPERADOR)
                self._operador(no.x, y, 0 if no.lado_operador < 0 else 180, True, f"Operador 6N-{k + 1}", "6N", est)
                k += 1
        y0, y1 = BAIAS["6N"]
        for i in range(livres):
            x, y = 5_200 + i * 800, y0 + 900
            self._operador(x, y, 0, False, f"Operador 6N-{k + 1} (livre)", "6N", None)
            k += 1
        cont["6N"] = k
        for est, linha in (("BEOX-05", "7C"), ("BEOX-07", "7D"), ("BEOX-08", "7D")):
            no = self.nos[est]
            y = no.y + no.lado_operador * (no.larg / 2 + DIST_OPERADOR)
            ativo = est in proc
            self._operador(no.x, y, 0 if no.lado_operador < 0 else 180, ativo,
                           f"Operador {est}{'' if ativo else ' (ocioso)'}", linha, est)
            cont[linha] += 1
        if livres:
            self.rotulos.append(Rotulo((5_200 + (livres - 1) * 400, y0 + 900, 2_100),
                                       f"{livres} operador(es) livre(s) — pool 6N", 11))
        return cont

    # ------------------------------------------------- manutenção, setas, produto
    def manutencao(self) -> None:
        no = self.nos["BEOX-05"]
        x, y = no.x1 + 900, no.y + no.larg / 2 + 300
        if self.cen.quebras_05:
            self.rotulos.append(Rotulo((no.x, no.y - 2_600, 60),
                                       f"BEOX-05: {self.kpis['quebras']} quebras na medição (MTBF {sim.MTBF_05:g} min)".replace(".", ","), 12))
        else:
            arm = subtrair(caixa_c(x, y, 0, 1_000, 600, 1_900), caixa_c(x, y - 260, 120, 900, 120, 1_660))
            portas = caixa_c(x, y - 285, 120, 900, 30, 1_660)
            self.add("Armário de manutenção preventiva (TPM)", arm, COR["manut"], "ferramental", "tpm", "7C", "BEOX-05")
            self.add("Armário TPM · porta", portas, COR["painel"], "ferramental", "tpm", "7C", "BEOX-05", transp=0.3)
            self.rotulos.append(Rotulo((x, y, 2_250), "Manutenção preventiva", 11))

    def setas(self, arestas, rotas) -> None:
        for (a, b), rota in rotas.items():
            linha = self.nos[a].linha or self.nos[b].linha
            w, z0, h = 220.0, 4.0, 6.0
            segs = []
            pts = list(rota)
            # recua o último ponto para dar lugar à ponta
            (xa, ya), (xb, yb) = pts[-2], pts[-1]
            pts[-1] = (xb - 450, yb)
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
                    continue
                segs.append(caixa(min(x0, x1) - w / 2, min(y0, y1) - w / 2, z0,
                                  abs(x1 - x0) + w, abs(y1 - y0) + w, h))
            ponta = prisma([(xb - 500, yb - 300), (xb, yb), (xb - 500, yb + 300)], z0, h)
            self.add(f"Fluxo {a} → {b}", unir(*segs, ponta), COR[linha] if linha else COR["faixa"],
                     "fluxo", f"fluxo_{a}_{b}", linha)

    def produto(self) -> None:
        x = (ZONA_CONSOLIDACAO[0] + ZONA_CONSOLIDACAO[1]) / 2
        y = GALPAO_Y / 2
        # cabine de elevador: casca oca (subtração) com vão de porta e piso
        casca = subtrair(caixa_c(x, y, 300, 1_700, 1_500, 2_400), caixa_c(x, y, 360, 1_580, 1_380, 2_280),
                         caixa_c(x, y - 700, 360, 900, 200, 2_100))
        self.add("Cabine de elevador (produto de referência)", casca, COR["cabine"], "produto", "cabine")
        self.add("Cabine · base", caixa_c(x, y, 0, 1_900, 1_700, 300), COR["maquina_base"], "produto", "cabine")
        self.add("Cabine · porta", caixa_c(x - 230, y - 790, 360, 440, 30, 2_100), COR["cabine_porta"],
                 "produto", "cabine")
        for k, linha in enumerate(("6N", "7C", "7D")):
            self._posicao_piso(x + (k - 1) * 1_500, y - 3_000, f"Consolidação posição {linha}", "consol_pos")
        self.rotulos.append(Rotulo((x, y, 3_200), "Consolidação → Elevador completo", 14, True))
        self.rotulos.append(Rotulo((x, y - 4_200, 60), "3 kits (6N + 7C + 7D) = 1 elevador", 11, vistas="planta"))


# =============================================================================
# 9. MONTAGEM DE UM CENÁRIO
# =============================================================================
def montar_cena(cen: sim.Cenario, bib: Biblioteca) -> Cena:
    m = sim.Modelo(cen, sim.SEED_PRINCIPAL).rodar()
    kpis = sim.calcular_kpis(m)
    kits_max = maximo_area_kit(m)
    t_snap = instante_wip_maximo(m)
    snap = snapshot(m, t_snap)
    nos = planejar_layout(cen, kpis, kits_max)
    arestas = grafo_fluxo(cen)
    rotas = {(a, b): rotear(nos, a, b) for a, b in arestas}

    c = Construtor(cen, nos, kpis, snap, bib)
    c.galpao()
    c.entradas()
    for est in sim.ESTACOES:
        (c.celula if est in CELULAS else c.posto_manual)(est)
    c.buffers_e_kits()
    snap["operadores_cad"] = c.operadores()
    c.manutencao()
    c.setas(arestas, rotas)
    c.produto()
    dia = int(t_snap // sim.JORNADA) + 1
    minuto = t_snap % sim.JORNADA
    snap["rotulo"] = f"dia {dia}, {7 + int(minuto // 60):02d}:{int(minuto % 60):02d}"
    return Cena(cen, nos, arestas, rotas, c.pecas, c.rotulos, snap, kpis, m, kits_max)


# =============================================================================
# 10. VERIFICAÇÃO & VALIDAÇÃO DO CAD
# =============================================================================
class VV:
    def __init__(self) -> None:
        self.itens: list[dict] = []

    def add(self, grupo: str, cenario: str, nome: str, resultado: str, criterio: str, ok: bool) -> None:
        self.itens.append({"grupo": grupo, "cenario": cenario, "nome": nome,
                           "resultado": resultado, "criterio": criterio, "ok": bool(ok)})


def _solidos(forma: TopoDS_Shape) -> list[TopoDS_Shape]:
    exp, out = TopExp_Explorer(forma, TopAbs_SOLID), []
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def verificar_geometria(cena: Cena, vv: VV) -> dict:
    cid = cena.cen.nome
    G = "Verificação geométrica (B-Rep)"
    solidos = [(p, s) for p in cena.pecas for s in _solidos(p.forma)]
    invalidos = [p.nome for p, s in solidos if not BRepCheck_Analyzer(s).IsValid()]
    vv.add(G, cid, "Validade topológica/geométrica (BRepCheck_Analyzer)",
           f"{len(solidos)} sólidos, {len(invalidos)} inválidos", "0 inválidos", not invalidos)
    abertos, vol_neg = [], []
    for p, s in solidos:
        if volume(s) <= 0:
            vol_neg.append(p.nome)
        exp = TopExp_Explorer(s, TopAbs_SHELL)
        while exp.More():
            if not BRep_Tool.IsClosed(exp.Current()):
                abertos.append(p.nome)
            exp.Next()
    vv.add(G, cid, "Sólidos fechados e com volume positivo",
           f"{len(abertos)} cascas abertas, {len(vol_neg)} volumes ≤ 0", "0 e 0", not abertos and not vol_neg)

    # interferência entre conjuntos físicos distintos (filtro por caixa + booleana)
    fis = [p for p in cena.pecas if p.categoria in CATEGORIAS_SOLIDAS]
    caixas = [caixa_limite(p.forma) for p in fis]
    interf, testados = [], 0
    for i in range(len(fis)):
        a = caixas[i]
        for j in range(i + 1, len(fis)):
            if fis[i].grupo == fis[j].grupo:
                continue
            b = caixas[j]
            if a[3] <= b[0] + 1 or b[3] <= a[0] + 1 or a[4] <= b[1] + 1 or b[4] <= a[1] + 1 \
                    or a[5] <= b[2] + 1 or b[5] <= a[2] + 1:
                continue
            testados += 1
            v = volume(BRepAlgoAPI_Common(fis[i].forma, fis[j].forma).Shape())
            if v > TOL_INTERFERENCIA:
                interf.append(f"{fis[i].nome} × {fis[j].nome} ({v / 1e6:.2f} L)")
    vv.add(G, cid, "Interferência entre conjuntos (pares com volume comum > 1 cm³)",
           f"{len(interf)} de {testados} pares candidatos" + (f": {interf[:3]}" if interf else ""),
           "0 interferências", not interf)

    fora = [p.nome for p, c in zip(fis, caixas)
            if c[0] < -1 or c[1] < -1 or c[3] > GALPAO_X + 1 or c[4] > GALPAO_Y + 1]
    vv.add(G, cid, "Tudo dentro do galpão (64 × 42 m)", f"{len(fora)} peças fora", "0", not fora)

    # corredores de empilhadeira livres (volume de 2,5 m de altura)
    ocupacao = []
    equip = [p for p in cena.pecas if p.categoria in CATEGORIAS_EQUIPAMENTO | {"operador"}]
    for y0, y1 in CORREDORES:
        vol_corr = caixa(1_000, y0, 0, X_FIM_CORREDOR - 1_000, y1 - y0, 2_500)
        for p in equip:
            c = caixa_limite(p.forma)
            if c[4] > y0 and c[1] < y1 and c[0] < X_FIM_CORREDOR:
                if volume(BRepAlgoAPI_Common(vol_corr, p.forma).Shape()) > TOL_INTERFERENCIA:
                    ocupacao.append(p.nome)
    vv.add(G, cid, f"Corredores de empilhadeira livres (largura {CORREDOR_MIN / 1000:g} m+)",
           f"{len(ocupacao)} peças invadindo; larguras {[f'{(y1 - y0) / 1000:g} m' for y0, y1 in CORREDORES]}",
           f"0 peças e largura ≥ {CORREDOR_MIN / 1000:g} m",
           not ocupacao and all(y1 - y0 >= CORREDOR_MIN for y0, y1 in CORREDORES))

    fora_baia = []
    for p in equip:
        if p.linha in BAIAS and p.categoria != "produto":
            c = caixa_limite(p.forma)
            y0, y1 = BAIAS[p.linha]
            if c[1] < y0 - 1 or c[4] > y1 + 1:
                fora_baia.append(p.nome)
    vv.add(G, cid, "Equipamentos de cada linha dentro da sua baia", f"{len(fora_baia)} fora",
           "0", not fora_baia)

    # ergonomia: as mãos do operador ativo alcançam o tampo sem colidir
    folgas = []
    for p in cena.pecas:
        if p.categoria == "operador" and p.nome.endswith("colete") and p.estacao and "ocioso" not in p.nome:
            banc = next(q for q in cena.pecas if q.estacao == p.estacao and q.nome.endswith("bancada"))
            d = BRepExtrema_DistShapeShape(p.forma, banc.forma)
            folgas.append((p.estacao, d.Value()))
    ok = all(ALCANCE_MIN <= d <= ALCANCE_MAX for _, d in folgas)
    vv.add(G, cid, "Alcance das mãos do operador ativo ao tampo (BRepExtrema)",
           ", ".join(f"{e}: {d:.0f} mm" for e, d in folgas), f"{ALCANCE_MIN}–{ALCANCE_MAX} mm", ok)
    return {"n_solidos": len(solidos), "n_pecas": len(cena.pecas)}


def verificar_coerencia(cena: Cena, vv: VV) -> dict:
    cen, cid, m, snap = cena.cen, cena.cen.nome, cena.modelo, cena.snap
    L, F = "Coerência lógica CAD ↔ DES", "Coerência física CAD ↔ DES"

    est_cad = {n for n, no in cena.nos.items() if no.tipo in ("estacao", "maquina")}
    vv.add(L, cid, "Conjunto de estações", f"CAD {len(est_cad)} · DES {len(sim.ESTACOES)}",
           "conjuntos idênticos", est_cad == set(sim.ESTACOES))

    cad, obs = set(cena.arestas), arestas_da_simulacao(m)
    so_cad, so_sim = sorted(cad - obs), sorted(obs - cad)
    vv.add(L, cid, "Grafo de fluxo do CAD = precedências observadas no log da DES",
           f"{len(cad)} arestas CAD, {len(obs)} no log; só CAD: {so_cad or '—'}; só DES: {so_sim or '—'}",
           "diferença simétrica vazia", not so_cad and not so_sim)

    tk = tokens_do_log(m)
    if cen.beox04_paralelo:
        pares = [(tk[(j, 0)][0][3], tk[(j, 1)][0][3]) for j in range(len(m.chegada)) if (j, 1) in tk]
        ok = all(abs(a - b) < 1e-9 for a, b in pares)
        vv.add(L, cid, "BEOX-04 em paralelo: entra junto com o BEOX-01",
               f"{len(pares)} obras, t_fila(04) = t_fila(01) em todas" if ok else "divergência",
               "ramos paralelos no CAD e início simultâneo no log", ok and ("BEOX-03", "BEOX-04") not in cad)
    else:
        pares = [(ops[2][5], ops[3][3]) for (j, t), ops in tk.items() if t == 0 and len(ops) == 4 and ops[2][5]]
        ok = all(abs(a - b) < 1e-9 for a, b in pares)
        vv.add(L, cid, "BEOX-04 em série após BEOX-03",
               f"{len(pares)} obras com t_fila(04) = t_fim(03)", "aresta BEOX-03→04 no CAD e no log",
               ok and ("BEOX-03", "BEOX-04") in cad)

    pares78 = [(tk[(j, 4)][0][3], tk[(j, 5)][0][3]) for j in range(len(m.chegada)) if (j, 4) in tk and (j, 5) in tk]
    vv.add(L, cid, "BEOX-07 ∥ BEOX-08 após a máquina 7D",
           f"{len(pares78)} obras com início simultâneo", "mesmo predecessor e início simultâneo",
           all(abs(a - b) < 1e-9 for a, b in pares78))

    ops_cad = snap["operadores_cad"]
    esperado = {"6N": cen.operadores_6n, "7C": 1, "7D": 2}
    vv.add(L, cid, "Operadores por linha (CAD) = recursos da DES",
           f"CAD {ops_cad} · DES {esperado}", "iguais", ops_cad == esperado)
    sem_op = all(not any(p.categoria == "operador" and p.estacao == e for p in cena.pecas) for e in CELULAS)
    vv.add(L, cid, "Máquinas automáticas sem operador (DES não usa o pool nelas)",
           "nenhum operador em MAQ-7C/MAQ-7D" if sem_op else "operador em máquina", "nenhum", sem_op)
    tpm = any("TPM" in p.nome for p in cena.pecas)
    vv.add(L, cid, "Manutenção preventiva ↔ quebras do BEOX-05",
           f"quebras na DES: {'sim' if cen.quebras_05 else 'não'} · armário TPM no CAD: {'sim' if tpm else 'não'}",
           "TPM presente ⇔ sem quebras", tpm == (not cen.quebras_05))

    # ---- física ----
    falta = [f"{e}: {cena.nos[e].n_buffer} < {cena.kpis['fila_max'][e]:.0f}"
             for e in sim.ESTACOES if cena.nos[e].n_buffer < cena.kpis["fila_max"][e]]
    vv.add(F, cid, "Posições de buffer ≥ fila máxima simulada (cada estação)",
           "; ".join(f"{e} {cena.nos[e].n_buffer}≥{cena.kpis['fila_max'][e]:.0f}" for e in sim.ESTACOES),
           "todas atendidas", not falta)
    vv.add(F, cid, "Área de kits ≥ máximo de kits aguardando consolidação",
           "; ".join(f"{l} {cena.nos['KIT-' + l].n_buffer}≥{cena.kits_max[l]}" for l in BAIAS), "todas atendidas",
           all(cena.nos["KIT-" + l].n_buffer >= cena.kits_max[l] for l in BAIAS))
    kits_cad = {p.grupo for p in cena.pecas if p.categoria == "kit"}
    obras_cad = {int(g.split("_")[2]) for g in kits_cad}
    vv.add(F, cid, f"Snapshot t* ({snap['rotulo']}): obras no CAD = WIP da DES",
           f"CAD {len(obras_cad)} obras · DES WIP = {snap['wip_sim']}", "iguais", len(obras_cad) == snap["wip_sim"])
    n_fila_cad = sum(1 for p in cena.pecas if p.categoria == "kit" and "fila" in p.nome and p.nome.startswith("Palete"))
    n_fila_sim = sum(len(v) for v in snap["fila"].values())
    vv.add(F, cid, "Snapshot t*: paletes em fila no CAD = kits em fila na DES",
           f"CAD {n_fila_cad} · DES {n_fila_sim} (BEOX-05: {len(snap['fila'].get('BEOX-05', []))})",
           "iguais", n_fila_cad == n_fila_sim)
    limite = [f"{l}: {max(cena.nos[e].x1 for e in cena.nos if cena.nos[e].linha == l and cena.nos[e].tipo in ('estacao', 'maquina')) / 1000:.1f} m"
              for l in BAIAS]
    ok_lim = all(cena.nos[e].x1 <= X_LIMITE_LINHA for e in sim.ESTACOES)
    vv.add(F, cid, "Linhas cabem antes da área de kits", "fim das linhas: " + ", ".join(limite),
           f"≤ {X_LIMITE_LINHA / 1000:g} m", ok_lim)

    # transporte: comprimento das rotas medido no CAD × hipótese de transferência instantânea da DES
    comp = {e: distancia_percorrida(cena.nos, *e, r) for e, r in cena.rotas.items()}
    succ = defaultdict(list)
    for a, b in cena.arestas:
        succ[a].append(b)

    def mais_longo(n: str) -> float:
        return max((comp[(n, b)] + mais_longo(b) for b in succ[n]), default=0.0)
    caminhos = {l: mais_longo(f"ENT-{l}") for l in BAIAS}
    pior = max(caminhos.values())
    t_min = pior / 1000 / VEL_TRANSPORTE / 60
    vv.add(F, cid, "Transporte desprezível frente ao takt (hipótese da DES)",
           f"maior caminho {pior / 1000:.1f} m → {t_min:.2f} min a {VEL_TRANSPORTE:g} m/s = {t_min / cen.takt:.1%} do takt",
           f"≤ {LIMITE_TRANSPORTE_TAKT:.0%} do takt", t_min / cen.takt <= LIMITE_TRANSPORTE_TAKT)

    cls = {e: ("vermelho" if u >= .95 else "amarelo" if u >= .85 else "verde") for e, u in cena.kpis["util"].items()
           if e not in CELULAS}
    acesos = {p.estacao: p.nome.split("andon ")[1].split(" ")[0] for p in cena.pecas if "(aceso)" in p.nome}
    vv.add(F, cid, "Andon de cada posto = classe de utilização simulada",
           ", ".join(f"{e} {cena.kpis['util'][e]:.0%}→{acesos[e]}" for e in cls), "iguais", acesos == cls)

    area_wip = sum(len(no.slots) for no in cena.nos.values()) * (PASSO_BUFFER[0] * PASSO_BUFFER[1]) / 1e6
    return {"caminhos_m": {l: round(v / 1000, 1) for l, v in caminhos.items()},
            "distancia_total_m": round(sum(comp.values()) / 1000, 1),
            "transporte_pct_takt": round(t_min / cen.takt * 100, 2),
            "area_wip_m2": round(area_wip, 1),
            "posicoes_buffer": {e: cena.nos[e].n_buffer for e in sim.ESTACOES},
            "posicoes_area_kit": {l: cena.nos["KIT-" + l].n_buffer for l in BAIAS},
            "comprimento_linhas_m": {l: round(max(cena.nos[e].x1 for e in sim.ESTACOES
                                                  if sim.LINHA_DA_ESTACAO[e] == l) / 1000, 1) for l in BAIAS},
            "snapshot": {"t_min": round(snap["t"], 1), "rotulo": snap["rotulo"], "wip": snap["wip_sim"],
                         "fila_05": len(snap["fila"].get("BEOX-05", [])), "reparo_05": snap["reparo_05"]}}


# =============================================================================
# 11. EXPORTAÇÃO STEP (XCAF: montagem com nomes e cores) + releitura
# =============================================================================
def exportar_step(cena: Cena, caminho: Path, vv: VV) -> int:
    doc = TDocStd_Document("pythonocc-doc")
    st = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    raiz = st.NewShape()
    TDataStd_Name.Set(raiz, f"EOX_{cena.cen.nome}")
    n_solidos = 0
    for p in cena.pecas:
        lab = st.AddShape(p.forma, False)
        TDataStd_Name.Set(lab, p.nome)
        ct.SetColor(lab, Quantity_Color(*p.cor, Quantity_TOC_RGB), XCAFDoc_ColorSurf)
        st.AddComponent(raiz, lab, TopLoc_Location())
        n_solidos += contar_solidos(p.forma)
    st.UpdateAssemblies()
    w = STEPCAFControl_Writer()
    w.SetColorMode(True)
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    ok_w = w.Write(str(caminho)) == 1

    # releitura (round-trip) para confirmar que o arquivo é íntegro
    r = STEPCAFControl_Reader()
    r.SetColorMode(True)
    r.SetNameMode(True)
    ok_r = r.ReadFile(str(caminho)) == 1
    doc2 = TDocStd_Document("pythonocc-doc")
    r.Transfer(doc2)
    st2 = XCAFDoc_DocumentTool.ShapeTool(doc2.Main())
    seq = TDF_LabelSequence()
    st2.GetFreeShapes(seq)
    lido = composto([st2.GetShape(seq.Value(i + 1)) for i in range(seq.Length())])
    n_lido = contar_solidos(lido)
    valido = BRepCheck_Analyzer(lido).IsValid()
    tam = caminho.stat().st_size / 1e6
    vv.add("Exportação STEP", cena.cen.nome, f"{caminho.name}: escrita, releitura e contagem de sólidos",
           f"{n_solidos} sólidos escritos, {n_lido} relidos, válido={valido}, {tam:.1f} MB",
           "escrita e leitura OK, mesma contagem, B-Rep válida", ok_w and ok_r and n_lido == n_solidos and valido)
    return n_solidos


# =============================================================================
# 12. RENDERIZAÇÃO (OffscreenRenderer do pythonOCC)
# =============================================================================
class Renderizador:
    def __init__(self, largura: int = 1800, altura: int = 1100) -> None:
        from OCC.Display.OCCViewer import OffscreenRenderer
        self.r = OffscreenRenderer(screen_size=(largura, altura))
        self.view, self.ctx = self.r.View, self.r.Context
        self.view.ChangeRenderingParams().NbMsaaSamples = 8
        self.view.TriedronErase()
        self.view.SetBgGradientColors(Quantity_Color(0.98, 0.98, 0.97, Quantity_TOC_RGB),
                                      Quantity_Color(0.84, 0.86, 0.89, Quantity_TOC_RGB), 2, True)

    def mostrar(self, pecas: list[Peca], rotulos: list[Rotulo], modo: str = "3d",
                altura_rotulo: float = 1.0) -> None:
        from OCC.Core.AIS import AIS_Shape, AIS_TextLabel
        from OCC.Core.Aspect import Aspect_TODT_SUBTITLE, Aspect_TOL_SOLID
        from OCC.Core.Prs3d import Prs3d_LineAspect
        from OCC.Core.Graphic3d import (Graphic3d_HTA_CENTER, Graphic3d_MaterialAspect,
                                        Graphic3d_NameOfMaterial_Satin)
        self.ctx.RemoveAll(False)
        for p in pecas:
            ais = AIS_Shape(p.forma)
            ais.SetMaterial(Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial_Satin))
            ais.SetColor(Quantity_Color(*p.cor, Quantity_TOC_RGB))
            if p.transparencia:
                ais.SetTransparency(p.transparencia)
            dr = ais.Attributes()
            dr.SetFaceBoundaryDraw(p.categoria not in ("piso", "zona", "marcacao", "fluxo", "buffer"))
            dr.SetFaceBoundaryAspect(Prs3d_LineAspect(Quantity_Color(0.15, 0.16, 0.18, Quantity_TOC_RGB),
                                                      Aspect_TOL_SOLID, 0.6))
            self.ctx.Display(ais, 1, 0, False)
        for rt in rotulos:
            if rt.vistas not in ("todas", modo) or (modo == "planta" and rt.pos_planta and rt.vistas == "todas"
                                                    and rt.texto.startswith("BEOX")):
                continue       # na planta o posto usa o rótulo curto
            lbl = AIS_TextLabel()
            lbl.SetText(rt.texto)          # str -> TCollection_ExtendedString (UTF-8)
            lbl.SetPosition(gp_Pnt(*(rt.pos_planta if modo == "planta" and rt.pos_planta else rt.pos)))
            lbl.SetHeight(rt.altura * altura_rotulo)
            lbl.SetFont("DejaVu Sans")
            lbl.SetColor(Quantity_Color(0.05, 0.05, 0.06, Quantity_TOC_RGB))
            lbl.SetDisplayType(Aspect_TODT_SUBTITLE)
            lbl.SetColorSubTitle(Quantity_Color(1, 1, 1, Quantity_TOC_RGB) if not rt.destaque
                                 else Quantity_Color(1.0, 0.96, 0.78, Quantity_TOC_RGB))
            lbl.SetHJustification(Graphic3d_HTA_CENTER)
            self.ctx.Display(lbl, False)

    def camera(self, olho, alvo, perspectiva: bool = True, fov: float = 32.0, encaixar: bool = False) -> None:
        from OCC.Core.Graphic3d import Graphic3d_Camera
        cam = self.view.Camera()
        cam.SetProjectionType(Graphic3d_Camera.Projection_Perspective if perspectiva
                              else Graphic3d_Camera.Projection_Orthographic)
        cam.SetFOVy(fov)
        self.view.SetEye(*olho)
        self.view.SetAt(*alvo)
        self.view.SetUp(0, 0, 1)
        if encaixar:
            self.view.FitAll(0.02, False)
        self.view.ZFitAll(1.0)

    def planta(self) -> None:
        from OCC.Core.Graphic3d import Graphic3d_Camera
        from OCC.Core.V3d import V3d_Zpos
        self.view.Camera().SetProjectionType(Graphic3d_Camera.Projection_Orthographic)
        self.view.SetProj(V3d_Zpos)
        self.view.FitAll(0.01, False)
        self.view.ZFitAll(1.0)

    def salvar(self, caminho: Path) -> None:
        self.view.Redraw()
        self.view.Dump(str(caminho))


VISTAS = {
    # nome: (olho, alvo, enquadrar tudo, escala dos rótulos)
    "isometrica": ((80_000, -30_000, 40_000), (31_000, 21_000, 0), True, 1.0),
    "linha6N": ((25_000, 15_500, 13_000), (15_500, 34_500, 600), False, 1.15),
    "linha7C": ((37_000, 4_000, 12_500), (23_500, 21_000, 600), False, 1.15),
    "posto05": (None, None, False, 1.25),
}


def cena_biblioteca(bib: Biblioteca) -> tuple[list[Peca], list[Rotulo]]:
    """Prancha com os componentes paramétricos usados nos dois layouts."""
    nos = {"BEOX-05": No("BEOX-05", "estacao", "7C", x=1_800, y=0, comp=3_200, larg=1_800)}
    c = Construtor(sim.CENARIOS[1], nos, {"util": {"BEOX-05": 0.92}}, {}, bib)
    c.posto_manual("BEOX-05")
    c._operador(1_800, -900 - DIST_OPERADOR, 0, True, "Operador ativo", "7C", "BEOX-05")
    c._operador(-1_200, -900 - DIST_OPERADOR, 0, False, "Operador ocioso", "7C", None)
    c.rotulos = [Rotulo((1_800, 0, 2_600), "Posto manual: bancada, gabarito, painel, andon", 13, True),
                 Rotulo((-1_200, -1_450, 2_100), "Operador ocioso", 12)]
    x = 6_000
    for (nome, comp, larg, alt, tipo), linha in [(m, "7C") for m in CELULAS["MAQ-7C"]] + \
            [(m, "7D") for m in CELULAS["MAQ-7D"]]:
        c._maquina(x + comp / 2, 0, comp, larg, alt, tipo, nome, f"bib_{tipo}_{x}", linha, None)
        c.rotulos.append(Rotulo((x + comp / 2, 0, alt + 700), nome, 12, True))
        x += comp + 1_600
    for k, (toks, linha, nome) in enumerate([([0], "6N", "Kit 6N parcial"), ([0, 1], "6N", "Kit 6N completo"),
                                             ([2], "7C", "Kit 7C"), ([4], "7D", "Kit 7D parcial"),
                                             ([4, 5], "7D", "Kit 7D completo")]):
        xp = 1_000 + k * 2_200
        c._palete_com_carga(xp, -5_200, toks, k, "biblioteca", linha)
        c.rotulos.append(Rotulo((xp, -5_200, 1_000), nome, 11))
    c.add("Coluna HEB300", posicionar(bib.coluna(), -3_000, 1_500), COR["coluna"], "estrutura", "bib_col")
    c.rotulos.append(Rotulo((-3_000, 1_500, ALTURA_COLUNA + 300), "Coluna HEB 300 (perfil extrudado)", 11))
    c.add("Piso", caixa(-5_000, -8_000, -200, x + 5_000, 11_000, 200), COR["piso"], "piso", "bib_piso")
    return c.pecas, c.rotulos


def renderizar(cenas: dict[str, Cena], pasta: Path, bib: Biblioteca) -> list[Path]:
    rd = Renderizador()
    arquivos: list[Path] = []
    for chave, cena in cenas.items():
        for vista, (olho, alvo, encaixar, esc) in VISTAS.items():
            if vista == "posto05":
                no = cena.nos["BEOX-05"]
                alvo = (no.x - 1_500, no.y, 700)
                olho = (no.x + 6_000, no.y - 8_000, 5_000)
            rd.mostrar(cena.pecas, cena.rotulos, "3d", esc)
            rd.camera(olho, alvo, encaixar=encaixar)
            arq = pasta / f"eox_{chave}_{vista}.png"
            rd.salvar(arq)
            arquivos.append(arq)
        rd.mostrar(cena.pecas, cena.rotulos, "planta", 0.95)
        rd.planta()
        arq = pasta / f"eox_{chave}_planta.png"
        rd.salvar(arq)
        arquivos.append(arq)
    pecas, rotulos = cena_biblioteca(bib)
    rd.mostrar(pecas, rotulos, "3d", 1.1)
    rd.camera((24_000, -24_000, 13_000), (12_000, -1_500, 900), encaixar=True)
    arq = pasta / "eox_biblioteca_componentes.png"
    rd.salvar(arq)
    arquivos.append(arq)
    return arquivos


def comparativos(pasta: Path, metricas: dict) -> list[Path]:
    """Monta imagens lado a lado AS-IS | TO-BE com legenda (Pillow)."""
    from PIL import Image, ImageDraw, ImageFont
    fonte = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fonte2 = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        f1, f2 = ImageFont.truetype(fonte, 34), ImageFont.truetype(fonte2, 24)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    saida = []
    for vista in ("isometrica", "planta", "linha6N", "linha7C"):
        ims = [Image.open(pasta / f"eox_{k}_{vista}.png").convert("RGB") for k in ("asis", "tobe")]
        w, h = ims[0].size
        tela = Image.new("RGB", (2 * w + 30, h + 120), (255, 255, 255))
        d = ImageDraw.Draw(tela)
        for i, (im, k) in enumerate(zip(ims, ("asis", "tobe"))):
            x = i * (w + 30)
            tela.paste(im, (x, 120))
            mt = metricas[k]
            d.text((x + 20, 18), "AS-IS" if k == "asis" else "TO-BE", font=f1, fill=(20, 33, 61))
            d.text((x + 20, 64), f"{mt['legenda']}", font=f2, fill=(80, 80, 78))
        arq = pasta / f"eox_comparativo_{vista}.png"
        tela.save(arq, optimize=True)
        saida.append(arq)
    return saida


# =============================================================================
# 13. RELATÓRIO HTML
# =============================================================================
def relatorio_html(pasta: Path, vv: VV, metricas: dict, imagens: list[Path]) -> Path:
    from PIL import Image

    def img64(p: Path, largura: int = 1600) -> str:
        im = Image.open(p).convert("RGB")
        if im.width > largura:
            im = im.resize((largura, int(im.height * largura / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=86)
        return base64.b64encode(buf.getvalue()).decode()

    ok = sum(i["ok"] for i in vv.itens)
    linhas = []
    grupo_ant = None
    for i in vv.itens:
        if i["grupo"] != grupo_ant:
            linhas.append(f'<tr class="g"><td colspan="5">{i["grupo"]}</td></tr>')
            grupo_ant = i["grupo"]
        linhas.append(f'<tr><td>{i["cenario"]}</td><td>{i["nome"]}</td><td>{i["resultado"]}</td>'
                      f'<td>{i["criterio"]}</td><td class="{"ok" if i["ok"] else "f"}">'
                      f'{"✔ OK" if i["ok"] else "✖ Falha"}</td></tr>')
    a, t = metricas["asis"], metricas["tobe"]
    comp = [
        ("Snapshot (instante de WIP máximo)", a["snapshot"]["rotulo"], t["snapshot"]["rotulo"]),
        ("Obras em processo no snapshot", a["snapshot"]["wip"], t["snapshot"]["wip"]),
        ("Kits na fila do BEOX-05 no snapshot", a["snapshot"]["fila_05"], t["snapshot"]["fila_05"]),
        ("Posições de buffer do BEOX-05 (≥ fila máx. simulada)", a["posicoes_buffer"]["BEOX-05"], t["posicoes_buffer"]["BEOX-05"]),
        ("Posições nas áreas de kit 6N / 7C / 7D", " / ".join(map(str, a["posicoes_area_kit"].values())),
         " / ".join(map(str, t["posicoes_area_kit"].values()))),
        ("Área de piso reservada a WIP (m²)", f'{a["area_wip_m2"]:.0f}', f'{t["area_wip_m2"]:.0f}'),
        ("Comprimento das linhas 6N / 7C / 7D (m)", " / ".join(f"{v:.1f}" for v in a["comprimento_linhas_m"].values()),
         " / ".join(f"{v:.1f}" for v in t["comprimento_linhas_m"].values())),
        ("Distância de fluxo por elevador (m)", a["distancia_total_m"], t["distancia_total_m"]),
        ("Operadores 6N", a["operadores_6n"], t["operadores_6n"]),
        ("Sólidos B-Rep no modelo", a["n_solidos"], t["n_solidos"]),
    ]
    tab_comp = "".join(f"<tr><td>{n}</td><td>{x}</td><td>{y}</td></tr>" for n, x, y in comp)
    rastreio = [
        ("Estações, linhas e tempos médios", "ESTACOES, LINHA_DA_ESTACAO, DISTRIBUICOES", "nós do layout e rótulos"),
        ("Operadores 6N, BEOX-04 série/paralelo, quebras", "Cenario (C1, C2)", "nº de bonecos, topologia da 6N, armário TPM"),
        ("Fila máxima por estação", "calcular_kpis → fila_max", "nº de posições de buffer no piso"),
        ("Kits aguardando consolidação", "log de eventos (ops, conclusao)", "tamanho das áreas de kit"),
        ("Utilização por estação", "calcular_kpis → util", "cor do andon"),
        ("Estado no instante de WIP máximo", "log de eventos + série de WIP", "posição de cada palete e operador"),
        ("Precedências percorridas", "log de eventos (ops)", "conferência do grafo de fluxo"),
        ("Transferência instantânea entre postos", "hipótese do modelo DES", "rotas medidas no CAD ÷ velocidade"),
    ]
    tab_rast = "".join(f"<tr><td>{a}</td><td><code>{b}</code></td><td>{c}</td></tr>" for a, b, c in rastreio)
    figs = "".join(f'<figure><img src="data:image/jpeg;base64,{img64(p)}" alt="{p.stem}">'
                   f'<figcaption>{p.name}</figcaption></figure>' for p in imagens)
    html = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CAD 3D EOX — V&amp;V</title>
<style>body{{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:#f4f3ef;color:#0b0b0b}}
header{{background:#14213d;color:#fff;padding:26px 32px}}header h1{{margin:0 0 4px;font-size:23px}}header p{{margin:0;color:#c9d3e6}}
main{{max-width:1500px;margin:0 auto;padding:18px 24px 60px}}section{{background:#fcfcfb;border:1px solid #e2e0da;border-radius:12px;padding:18px 22px;margin:16px 0}}
h2{{margin:0 0 10px;font-size:18px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 9px;border-bottom:1px solid #ecebe7;text-align:left;vertical-align:top}}
th{{color:#52514e;font-size:12.5px}}tr.g td{{background:#f4f3ef;font-weight:650;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#52514e}}
td.ok{{color:#1d7a3c;font-weight:600;white-space:nowrap}}td.f{{color:#b42318;font-weight:600}}figure{{margin:0 0 18px}}img{{width:100%;border:1px solid #e2e0da;border-radius:8px}}
figcaption{{font-size:12px;color:#8a8984}}.kpi{{display:flex;gap:12px;flex-wrap:wrap}}.kpi div{{background:#f4f3ef;border-radius:8px;padding:8px 14px}}.kpi b{{display:block;font-size:20px}}</style></head><body>
<header><h1>Modelo CAD 3D (B-Rep · pythonOCC) — Linha EOX AS-IS × TO-BE</h1>
<p>Gerado por eox_cad.py a partir da simulação eox_simulacao.py (seed {sim.SEED_PRINCIPAL}). Unidades em mm; arquivos STEP AP214 com nomes e cores.</p></header><main>
<section><h2>Resumo da V&amp;V</h2><div class="kpi"><div><span>Verificações</span><b>{ok}/{len(vv.itens)} OK</b></div>
<div><span>Sólidos AS-IS</span><b>{a["n_solidos"]}</b></div><div><span>Sólidos TO-BE</span><b>{t["n_solidos"]}</b></div></div></section>
<section><h2>Comparativo AS-IS × TO-BE (medido no CAD)</h2><table><tr><th>Métrica</th><th>AS-IS</th><th>TO-BE</th></tr>{tab_comp}</table></section>
<section><h2>Verificação &amp; validação</h2><table><tr><th>Cenário</th><th>Verificação</th><th>Resultado</th><th>Critério</th><th>Status</th></tr>{"".join(linhas)}</table></section>
<section><h2>Rastreabilidade simulação → CAD</h2><table><tr><th>Informação</th><th>Origem na simulação</th><th>Uso no CAD</th></tr>{tab_rast}</table></section>
<section><h2>Imagens do modelamento 3D</h2>{figs}</section></main></body></html>"""
    arq = pasta / "eox_cad_relatorio.html"
    arq.write_text(html, encoding="utf-8")
    return arq


# =============================================================================
# 14. PROGRAMA PRINCIPAL
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="CAD 3D B-Rep dos layouts AS-IS e TO-BE da linha EOX")
    ap.add_argument("--saida", default="cad_saida", help="pasta de saída (relativa)")
    ap.add_argument("--sem-imagens", action="store_true", help="não renderiza as imagens")
    args = ap.parse_args()
    pasta = Path(args.saida)
    pasta.mkdir(exist_ok=True)

    t0 = time.time()
    bib = Biblioteca()
    cenarios = {"asis": sim.CENARIOS[0], "tobe": sim.CENARIOS[1]}
    cenas, metricas, vv = {}, {}, VV()
    for chave, cen in cenarios.items():
        print(f"[{cen.nome}] simulando e montando a cena…")
        cena = montar_cena(cen, bib)
        cenas[chave] = cena
        print(f"[{cen.nome}] {len(cena.pecas)} peças; V&V…")
        mg = verificar_geometria(cena, vv)
        mc = verificar_coerencia(cena, vv)
        n = exportar_step(cena, pasta / f"eox_layout_{chave}.step", vv)
        metricas[chave] = {**mg, **mc, "n_solidos_step": n, "operadores_6n": cen.operadores_6n,
                           "legenda": (f"{cen.operadores_6n} operadores na 6N · BEOX-04 "
                                       f"{'em paralelo' if cen.beox04_paralelo else 'em série'} · "
                                       f"{'quebras no BEOX-05' if cen.quebras_05 else 'manutenção preventiva'} · "
                                       f"snapshot {cena.snap['rotulo']}: WIP {cena.snap['wip_sim']}, "
                                       f"fila BEOX-05 = {len(cena.snap['fila'].get('BEOX-05', []))}")}

    imagens: list[Path] = []
    if not args.sem_imagens:
        print("Renderizando imagens…")
        imagens = renderizar(cenas, pasta, bib)
        imagens = comparativos(pasta, metricas) + imagens

    (pasta / "eox_cad_vv.json").write_text(json.dumps({"vv": vv.itens, "metricas": metricas},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    rel = relatorio_html(pasta, vv, metricas, imagens)
    ok = sum(i["ok"] for i in vv.itens)
    for i in vv.itens:
        if not i["ok"]:
            print(f"  FALHA [{i['cenario']}] {i['nome']}: {i['resultado']}")
    print(f"V&V do CAD: {ok}/{len(vv.itens)} verificações OK")
    print(f"Saídas em {pasta}/ ({rel.name}, STEP, PNG) — {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
