r"""
Oligopolio de Cournot con n empresas
====================================
Organización Industrial — demostración matemática completa.

Modelo
------
Demanda inversa lineal:         P(Q) = a - b·Q,     a, b > 0
Costos:                         C_i(q_i) = c_i · q_i  (MC constante)
Cantidad agregada:              Q = Σ_{i=1..n} q_i
Beneficio de la empresa i:      π_i(q_i, q_{-i}) = (P(Q) - c_i) · q_i

Cada empresa elige q_i tomando q_{-i} como dado (Nash en cantidades).
Este script:

  1) Deriva simbólicamente CPO, función de reacción y equilibrio.
  2) Obtiene fórmulas cerradas (caso simétrico y asimétrico).
  3) Verifica el equilibrio numéricamente por iteración de mejores
     respuestas (contracción → convergencia a Nash).
  4) Estática comparativa: n → ∞ implica P → c (competencia perfecta).
  5) Bienestar: excedente consumidor, beneficios, DWL vs. monopolio y C.P.

Autor: demo para cátedra de Organización Industrial.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
#   1.  DERIVACIÓN SIMBÓLICA
# ═══════════════════════════════════════════════════════════════════════════
def derivacion_simbolica_simetrica() -> dict:
    """Resuelve el caso simétrico c_i = c paso a paso con sympy."""
    a, b, c, n, q, qi, Q = sp.symbols("a b c n q q_i Q", positive=True)

    print("── 1) CASO SIMÉTRICO  (c_i = c ∀ i) ─────────────────────────────")
    print(f"   Demanda inversa:    P(Q) = a - b·Q")
    print(f"   Beneficio firma i:  π_i = (a - b·Q - c)·q_i\n")

    # En simetría Q = n·q y cada firma ve Q_{-i} = (n-1)·q.
    Q_tot = (n - 1) * q + qi
    pi_i = (a - b * Q_tot - c) * qi

    print("   Derivada parcial respecto a q_i (CPO):")
    foc = sp.diff(pi_i, qi)
    print(f"   ∂π_i/∂q_i = {sp.simplify(foc)}  =  0\n")

    # En el Nash simétrico imponemos q_i = q y despejamos.
    foc_sim = foc.subs(qi, q)
    q_star = sp.solve(foc_sim, q)[0]
    q_star = sp.simplify(q_star)

    Q_star = sp.simplify(n * q_star)
    P_star = sp.simplify(a - b * Q_star)
    pi_star = sp.simplify((P_star - c) * q_star)

    # Excedente del consumidor con demanda lineal: CS = ½·b·Q²
    CS_star = sp.simplify(sp.Rational(1, 2) * b * Q_star**2)
    W_star = sp.simplify(n * pi_star + CS_star)

    print("   Equilibrio de Cournot-Nash (simétrico):")
    print(f"       q*   = {q_star}")
    print(f"       Q*   = {Q_star}")
    print(f"       P*   = {P_star}")
    print(f"       π*   = {pi_star}")
    print(f"       CS*  = {CS_star}")
    print(f"       W*   = {W_star}\n")

    print("   Índice de Lerner  L = (P - c)/P  = "
          f"{sp.simplify((P_star - c) / P_star)}\n")

    # Límites: n=1 → monopolio ; n→∞ → competencia perfecta.
    P_mono = sp.simplify(P_star.subs(n, 1))
    P_lim  = sp.limit(P_star, n, sp.oo)
    print(f"   n = 1     →  P* = {P_mono}   (monopolio)")
    print(f"   n → ∞     →  P* = {P_lim}   (competencia perfecta)\n")

    return dict(q=q_star, Q=Q_star, P=P_star, pi=pi_star, CS=CS_star, W=W_star,
                symbols=(a, b, c, n))


def derivacion_simbolica_asimetrica(n_val: int = 3) -> None:
    """Resuelve simbólicamente el sistema de CPO para n firmas con costos
    distintos. Útil para mostrar que el método escala."""
    print(f"── 2) CASO ASIMÉTRICO CERRADO  (n = {n_val})  ───────────────────")
    a, b = sp.symbols("a b", positive=True)
    q = sp.symbols(f"q1:{n_val + 1}", positive=True)
    c = sp.symbols(f"c1:{n_val + 1}", positive=True)
    Q = sum(q)

    foc_sys = [sp.diff((a - b * Q - c[i]) * q[i], q[i]) for i in range(n_val)]
    sol = sp.solve(foc_sys, q, dict=True)[0]

    # Fórmula general: q_i* = (a - n·c_i + Σ_{j≠i} c_j) / ((n+1)·b)
    print("   Fórmula general demostrable por álgebra:")
    print("       q_i* = ( a - n·c_i + Σ_{j≠i} c_j ) / ( (n+1)·b )\n")
    print("   Verificación por solución del sistema (sympy):")
    for qi, expr in sol.items():
        print(f"       {qi}* = {sp.simplify(expr)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#   2.  SOLUCIONES CERRADAS (caso asimétrico, n arbitrario)
# ═══════════════════════════════════════════════════════════════════════════
def equilibrio_asimetrico(a: float, b: float, c: np.ndarray) -> dict:
    """
    Cournot-Nash con n firmas heterogéneas y demanda lineal.
        q_i* = [ a - n·c_i + Σ_{j≠i} c_j ] / [ (n+1)·b ]
    Sólo es válido cuando q_i* > 0 ∀ i (sin salida endógena).
    """
    c = np.asarray(c, dtype=float)
    n = c.size
    suma_c = c.sum()
    q = (a + suma_c - (n + 1) * c) / ((n + 1) * b)
    if np.any(q <= 0):
        raise ValueError("Alguna firma tendría q*≤0; requiere modelo con salida.")
    Q = q.sum()
    P = a - b * Q
    pi = (P - c) * q
    CS = 0.5 * b * Q ** 2
    W = pi.sum() + CS
    return dict(q=q, Q=Q, P=P, pi=pi, CS=CS, W=W)


# ═══════════════════════════════════════════════════════════════════════════
#   3.  VERIFICACIÓN NUMÉRICA POR MEJORES RESPUESTAS
# ═══════════════════════════════════════════════════════════════════════════
def mejor_respuesta(a: float, b: float, c_i: float, Q_menos_i: float) -> float:
    """BR_i(Q_{-i}) = max{ (a - c_i - b·Q_{-i}) / (2b) , 0 }"""
    return max(0.0, (a - c_i - b * Q_menos_i) / (2 * b))


def iterar_nash(a: float, b: float, c: np.ndarray,
                tol: float = 1e-10, max_iter: int = 10_000) -> tuple[np.ndarray, int]:
    """Dinámica de mejores respuestas SECUENCIAL (Gauss-Seidel): cada firma
    reoptimiza usando las cantidades más recientes de sus rivales. En
    Cournot con demanda lineal converge al único Nash desde cualquier q₀.
    (La versión simultánea/Jacobi no contrae para n ≥ 3.)"""
    q = np.zeros_like(c, dtype=float)
    for k in range(1, max_iter + 1):
        q_prev = q.copy()
        for i in range(c.size):
            q[i] = mejor_respuesta(a, b, c[i], q.sum() - q[i])
        if np.max(np.abs(q - q_prev)) < tol:
            return q, k
    raise RuntimeError("No convergió.")


# ═══════════════════════════════════════════════════════════════════════════
#   4.  ESTÁTICA COMPARATIVA + GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════
def graficos_comparativos(a: float, b: float, c: float, n_max: int = 25,
                          ruta: str = "cournot_estatica.png") -> None:
    ns = np.arange(1, n_max + 1)
    q  = (a - c) / ((ns + 1) * b)
    Q  = ns * q
    P  = (a + ns * c) / (ns + 1)
    pi = (a - c) ** 2 / ((ns + 1) ** 2 * b)
    CS = 0.5 * b * Q ** 2
    W  = ns * pi + CS
    L  = (P - c) / P                     # Lerner

    Q_cp = (a - c) / b                    # competencia perfecta
    W_cp = 0.5 * b * Q_cp ** 2
    DWL  = W_cp - W                       # pérdida de peso muerto

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("Cournot con n empresas simétricas  ·  "
                 f"a={a}, b={b}, c={c}", fontweight="bold")

    ax[0, 0].plot(ns, P, "o-", label="P*(n)")
    ax[0, 0].axhline(c, ls="--", c="gray", label="MC = c  (C.P.)")
    ax[0, 0].set(xlabel="n", ylabel="precio", title="Precio → MC cuando n→∞")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(ns, Q, "o-", label="Q*(n)")
    ax[0, 1].axhline(Q_cp, ls="--", c="gray", label="Q competitiva")
    ax[0, 1].set(xlabel="n", ylabel="cantidad", title="Cantidad agregada")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    ax[1, 0].plot(ns, L, "o-", c="crimson")
    ax[1, 0].set(xlabel="n", ylabel="(P-MC)/P",
                 title="Índice de Lerner  L = 1/(n·ε)  →  0")
    ax[1, 0].grid(alpha=.3)

    ax[1, 1].plot(ns, W,   "o-", label="Bienestar W(n)")
    ax[1, 1].plot(ns, DWL, "s-", label="DWL(n)", c="orange")
    ax[1, 1].axhline(W_cp, ls="--", c="gray", label="W competitivo")
    ax[1, 1].set(xlabel="n", ylabel="bienestar",
                 title="Bienestar y pérdida de peso muerto")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    print(f"   [gráfico guardado en {ruta}]\n")


# ═══════════════════════════════════════════════════════════════════════════
#   5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("═" * 67)
    print("  DEMOSTRACIÓN: OLIGOPOLIO DE COURNOT CON n EMPRESAS")
    print("  Organización Industrial")
    print("═" * 67, "\n")

    # ---- 1. simbólico simétrico ------------------------------------------
    derivacion_simbolica_simetrica()

    # ---- 2. simbólico asimétrico (n=3) -----------------------------------
    derivacion_simbolica_asimetrica(n_val=3)

    # ---- 3. ejemplo numérico asimétrico ----------------------------------
    a, b = 100.0, 1.0
    c = np.array([10.0, 15.0, 20.0, 25.0])        # 4 firmas heterogéneas
    print("── 3) EJEMPLO NUMÉRICO ASIMÉTRICO  (n = 4) ──────────────────────")
    print(f"   a = {a}, b = {b}, c = {c.tolist()}\n")

    eq = equilibrio_asimetrico(a, b, c)
    print("   Solución cerrada:")
    for i, (qi, pii) in enumerate(zip(eq["q"], eq["pi"]), start=1):
        print(f"     firma {i}:  q_i* = {qi:8.4f}   π_i* = {pii:10.4f}")
    print(f"     Q* = {eq['Q']:.4f}   P* = {eq['P']:.4f}   "
          f"CS* = {eq['CS']:.4f}   W* = {eq['W']:.4f}\n")

    q_num, iters = iterar_nash(a, b, c)
    err = np.max(np.abs(q_num - eq["q"]))
    print(f"   Mejores respuestas iteradas: convergió en {iters} pasos, "
          f"‖Δ‖∞ = {err:.2e}\n")

    # ---- 4. verificación de propiedades clave (asserts "matemáticos") ----
    P_sim = lambda n, c_: (a + n * c_) / (n + 1)
    assert abs(P_sim(1, 10) - (a + 10) / 2) < 1e-12       # monopolio
    assert abs(P_sim(1_000_000, 10) - 10) < 1e-3          # n→∞ ⇒ P→c
    # Lerner simétrico: (P-c)/P = 1/(n·ε) con ε = P/(b·Q)·1 = (a+nc)/(n(a-c))
    n_t, c_t = 5, 10.0
    P_t = P_sim(n_t, c_t)
    Q_t = n_t * (a - c_t) / ((n_t + 1) * b)
    eps = (P_t / (b * Q_t))
    lhs, rhs = (P_t - c_t) / P_t, 1 / (n_t * eps)
    assert abs(lhs - rhs) < 1e-12, "Lerner 1/(nε) falló"
    print("   ✓ Monopolio, competencia perfecta y Lerner L=1/(n·ε) verificados.\n")

    # ---- 5. estática comparativa gráfica ---------------------------------
    print("── 4) ESTÁTICA COMPARATIVA  (gráficos) ──────────────────────────")
    graficos_comparativos(a=100, b=1, c=10, n_max=25)


if __name__ == "__main__":
    main()
