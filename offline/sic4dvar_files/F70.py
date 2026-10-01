"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
w = 'seq'
v = 'invalid value encountered'
u = 'ignore'
t = 'interpolation: trying to fill point {} at time '
s = NotImplementedError
r = tuple
q = float
l = 'increase'
k = 'decrease'
j = 'raise'
i = 'inc'
f = 'force'
V = dict
a = 'dec'
X = int
T = abs
P = RuntimeError
O = TypeError
K = None
J = AssertionError
N = 1.0
F = isinstance
H = ''
G = len
E = range
D = True
C = 0.0
B = False
import numpy as A, logging, copy as I, warnings as Y, itertools as b, operator as d, math as W
from typing import Iterable, Tuple, Any, Literal
from copy import deepcopy as Q
from pathlib import Path
from .sic4dvar_common_methods import check_na as U, fnc_APR as x
from .sic4dvar_0_defaults import SIC4DVarLowCostDefaults as L
try:
    c = b.pairwise
except ImportError:

    def c(iterable):
        B, A = b.tee(iterable)
        next(A, K)
        return zip(B, A)

def h(value_0, value_1, cor):
    return N - A.exp(-(value_1 - value_0) / cor)

def m(value_init, value_smoothed, min_change_v_thr=L().def_float_atol, value_ref=A.nan):
    E = value_ref
    D = value_smoothed
    B = value_init
    F = A.abs(D - B)
    if F < min_change_v_thr:
        return (B, C)
    if A.isfinite(E):
        if A.abs(D - E) <= A.abs(B - E):
            return (D, F)
        else:
            return (B, C)
    return (D, F)

def n(value_0, value_1, b=H, b_min_thr=L().def_float_atol, b_max_thr=A.inf):
    C = value_1
    A = value_0
    if not b:
        return B
    if T(C - A) > b_max_thr:
        return B
    if T(C - A) < b_min_thr:
        return B
    if b == a:
        if C < A:
            return B
        return D
    if C > A:
        return B
    return D

def y(sub_value0_array, sub_base0_array, cor, min_change_v_thr=L().def_float_atol, inter_behavior=H, inter_behavior_min_thr=L().def_float_atol, inter_behavior_max_thr=A.inf, inter_only=B, check_nan=D, debug_mode=B):
    o = debug_mode
    l = inter_behavior_max_thr
    k = inter_behavior_min_thr
    j = inter_behavior
    d = min_change_v_thr
    Y = inter_only
    D = cor
    if a in j.lower():
        Z = a
    elif i in j.lower():
        Z = i
    else:
        Z = H
        if Y:
            raise O
    p = 'interchange' if Y else 'relaxation'
    B = S(sub_value0_array)
    b = S(sub_base0_array)
    e = [B, b]
    if A.any([A.ndim != 1 for A in e]):
        raise P
    if A.any([A.shape != B.shape for A in e]):
        raise P
    if check_nan:
        if A.any([A.any(A.isnan(B)) for B in e[:-1]]):
            raise P
    U = X(T(A.log10(d)))
    Q = 0
    for J, K in c(E(G(B))):
        V = f'Forward sweep for point {K}, (previous point {J}),'
        L, M = (B[J], B[K])
        R = C
        if F(D, A.ndarray):
            if D.shape[0] > 1:
                R = (D[J] + D[K]) / 2
            else:
                R = D[J]
        elif F(D, q):
            R = D
        elif F(D, X):
            R = D
        if not Y:
            W = h(b[J], b[K], R)
            f = L * (N - W) + M * W
            f, I = m(value_init=M, value_smoothed=f, min_change_v_thr=d)
            B[K] = f
            if I > Q:
                Q = I
            V += f' relax factor {W:.{U}f}, change in values {I:.{U}f},'
            L, M = (B[J], B[K])
        if n(value_0=L, value_1=M, b=Z, b_min_thr=k, b_max_thr=l):
            B[J], B[K] = (M, L)
            I = T(M - L)
            if I > Q:
                Q = I
            V += f' values interchanged to force {Z}reasing behavior, change in values {I:.{U}f}: {M:.{U}f}, {L:.{U}f}'
        elif Y:
            V += f' values not interchanged.'
        if o:
            0
    for K, J in c(reversed(E(G(B)))):
        V = f'Backward sweep for point {J}, (forward point {K}),'
        L, M = (B[J], B[K])
        R = C
        if F(D, A.ndarray):
            if D.shape[0] > 1:
                R = (D[J] + D[K]) / 2
            else:
                R = D[K]
        elif F(D, q):
            R = D
        elif F(D, X):
            R = D
        if not Y:
            W = h(b[J], b[K], R)
            g = M * (N - W) + L * W
            g, I = m(value_init=L, value_smoothed=g, min_change_v_thr=d)
            B[J] = g
            if I > Q:
                Q = I
            if I > Q:
                Q = I
            V += f' relax factor {W:.{U}f}, change in values {I:.{U}f}, '
            L, M = (B[J], B[K])
        if n(value_0=L, value_1=M, b=Z, b_min_thr=k, b_max_thr=l):
            B[J], B[K] = (M, L)
            I = T(M - L)
            if I > Q:
                Q = I
            V += f' values interchanged to force {Z}reasing behavior, change in values {I:.{U}f}: {M:.{U}f}, {L:.{U}f}'
        elif Y:
            V += f' values not interchanged.'
        if o:
            0
    return (B, Q)

def e(my_array, comp_op, check, force, remove_nan=D):
    J = force
    H = check
    G = my_array
    if H and J:
        raise O
    if not H and (not J):
        raise O
    if not F(G, A.ndarray):
        G = A.array(G)
    if G.ndim != 1:
        raise O
    C = I.deepcopy(G)
    if remove_nan:
        C = C[A.isfinite(C)]
    for M, K in c(E(C.shape[0])):
        L, N = (C[M], C[K])
        if comp_op(N, L):
            if H:
                return B
            C[K] = L
    if H:
        return D
    return C

def z(my_array, remove_nan=D):
    return e(my_array, comp_op=d.gt, check=D, force=B, remove_nan=remove_nan)

def A0(my_array, remove_nan=D):
    return e(my_array, comp_op=d.lt, check=D, force=B, remove_nan=remove_nan)

def A1(*C):
    G = K
    B = A.empty(0)
    if not F(C, r):
        C = (C,)
    for D in C:
        if D.ndim > 2:
            raise O
        E = I.deepcopy(D)
        E = S(E)
        if G is K:
            G = D.shape
            B = I.deepcopy(E)
            continue
        if D.shape != G:
            raise O
        B += E
        if A.any(A.isfinite(B)):
            B /= A.nanmin(B)
    H = A.isnan(B)
    return H

def A2(*H, drop_na_how='any', axis=(0, 1)):
    J = axis
    if F(J, X):
        J = (J,)
    if not F(H, r):
        H = (H,)
    B = A1(*H)
    C = B.shape
    K = A.full_like(B, fill_value=N, dtype=A.float32)
    K[B] = A.nan
    if not A.any(B):
        L = A.array(E(C[0]))
        if G(C) == 1:
            return (L,)
        return (L, A.array(E(C[1])))
    if A.all(B):
        return (A.array([], dtype=A.int32),) * G(C)
    if G(C) == 1:
        return ((~B).nonzero()[0],)
    D = pd.DataFrame(I.deepcopy(K))
    for M in J:
        D = D.dropna(axis=M, how=drop_na_how)
    if D.empty:
        return (A.array([], dtype=A.int32),) * G(C)
    return (A.array(D.index, dtype=A.int32), A.array(D.columns, dtype=A.int32))

def A3(value0_array, base0_array, max_iter, cor, always_run_first_iter=D, behavior=H, inter_behavior=B, inter_behavior_min_thr=L().def_float_atol, inter_behavior_max_thr=A.inf, check_behavior=j, min_change_v_thr=L().def_float_atol, plot=D, plot_title=H, clean_run=B, debug_mode=B, time_integration=B):
    w = 'print'
    v = 'warn'
    t = time_integration
    s = always_run_first_iter
    r = 'none'
    n = debug_mode
    m = min_change_v_thr
    g = plot
    W = inter_behavior
    V = behavior
    R = clean_run
    Q = check_behavior
    if n:
        R = B
    if R:
        n = B
    L = S(value0_array)
    Z = S(base0_array)
    u = [L, Z]
    if A.any([A.ndim != 1 for A in u]):
        raise P
    if A.any([A.shape != L.shape for A in u]):
        raise P
    if a in V.lower():
        U = z
        o = 'decreasing'
        W = k if W else H
    elif i in V.lower():
        U = A0
        o = 'increasing'
        W = l if W else H
    else:
        V = H
        s = D
        U = K
        o = H
        Q = r
        W = H
    if V:
        if not any([A == Q.lower() for A in [f, j, v, w, r, H]]):
            raise O
    A4 = X(T(A.log10(m)))
    N = A2(L, Z)[0]
    if not R:
        0
    h, b = ([], [])
    if not s:
        if U(L, remove_nan=D):
            if g:
                0
            return L
    if G(N) < 2:
        if not R:
            0
        return L
    F = L[N]
    if t:
        c = C
        from copy import deepcopy as p
        M = p(Z[N])
        d = C
        for J in E(1, G(M)):
            c += (F[J] + F[J - 1]) / 2 * (M[J] - M[J - 1])
            d += M[J] - M[J - 1]
        c = c / d
    else:
        c = A.mean(F)
    if not R:
        0
    if g:
        h.append(I.deepcopy(L[N]))
        b.append('Initial')
    for x in E(max_iter):
        F, A1 = y(sub_value0_array=L[N], sub_base0_array=Z[N], cor=cor, min_change_v_thr=m, inter_behavior=W, inter_only=B, inter_behavior_min_thr=inter_behavior_min_thr, inter_behavior_max_thr=inter_behavior_max_thr, check_nan=B, debug_mode=n)
        if not R:
            0
        if A1 <= m:
            q = D
        elif U is not K:
            q = D if U(F, remove_nan=B) else B
        else:
            q = B
        if g:
            h.append(I.deepcopy(F))
            b.append(f'Iteration {x + 1}')
        L[N] = F
        if q:
            break
    if t:
        from copy import deepcopy as p
        M = p(Z[N])
        e = C
        d = C
        for J in E(1, G(M)):
            e += (F[J] + F[J - 1]) / 2 * (M[J] - M[J - 1])
            d += M[J] - M[J - 1]
        e = e / d
    else:
        e = A.mean(F)
    F = F + (c - e)
    if U is not K:
        if V:
            A3 = f'values are not always {o}'
            if not Q:
                0
            elif Q.lower() == r:
                0
            elif not U(F, remove_nan=B):
                if Q.lower() == v:
                    Y.warn(A3, RuntimeWarning)
                elif Q.lower() == w:
                    0
                elif f in Q.lower():
                    F = A.sort(F)
                    if a in V.lower():
                        F = F[::-1]
                else:
                    with A.printoptions(precision=4, suppress=D):
                        0
                    raise P
    if not R:
        0
    if g:
        h.append(I.deepcopy(F))
        b.append('Final')
        helper_plot_lines(xs=[N] * G(b), ys=h, show=D, line_labels=b, title=plot_title, x_axis_title='Indexes', y_axis_title='Values', fig_width=15, fig_height=5, add_legend=D)
    L[N] = F
    return L

def M(dim, value0_array, base0_array, max_iter, cor, always_run_first_iter=D, behavior=H, inter_behavior=B, inter_behavior_min_thr=L().def_float_atol, inter_behavior_max_thr=A.inf, check_behavior=j, min_change_v_thr=L().def_float_atol, plot=B, plot_title=H, clean_run=B, debug_mode=B, time_integration=B):
    L = value0_array
    K = debug_mode
    H = clean_run
    G = dim
    if K:
        H = B
    if H:
        K = B
    C = S(L)
    J = S(base0_array)
    M = [C, J]
    if A.any([A.ndim > 2 for A in M]):
        raise P
    if A.any([A.shape != C.shape for A in M[2:]]):
        raise P
    if G > 1:
        raise s
    C = I.deepcopy(L)
    Q = D if C.ndim == J.ndim else B
    if not H:
        0
    for F in E(C.shape[G]):
        if not H:
            0
        R = C[F, :] if G == 0 else C[:, F]
        if Q:
            N = J[F, :] if G == 0 else J[:, F]
        else:
            N = J
        O = A3(value0_array=R, base0_array=N, max_iter=max_iter, cor=cor, always_run_first_iter=always_run_first_iter, behavior=behavior, inter_behavior=inter_behavior, inter_behavior_min_thr=inter_behavior_min_thr, inter_behavior_max_thr=inter_behavior_max_thr, check_behavior=check_behavior, min_change_v_thr=min_change_v_thr, plot=plot, plot_title=plot_title + f' {F}', clean_run=H, debug_mode=K, time_integration=time_integration)
        if G == 0:
            C[F, :] = O
        else:
            C[:, F] = O
    return C

def A4(node_x, z, reach_t, times_debug=A.array([])):
    I = node_x
    d = B
    I = T(I - I[0]) / 1000
    b = Q(z)
    S = C
    V = C
    W = C
    Y = C
    Z = C
    c = C
    L = C
    M = C
    X = C
    k = C
    l = C
    m = C
    n = []
    o = []
    for J in E(0, G(z[0])):
        H = []
        for F in E(0, G(z)):
            if not U(z[F, J]):
                H.append(F)
        K = G(H)
        X += K
        e = A.arange(0, G(z[0]))
        p = C
        for F in E(0, K):
            S = S + I[H[F]] * I[H[F]]
            V = V + I[H[F]]
            W = W + I[H[F]]
            Y = Y + N
            Z = Z + I[H[F]] * z[H[F], J]
            c = c + z[H[F], J]
    if X >= 1:
        if V * W - Y * S == 0:
            L = C
            M = C
        else:
            M = (Z * W - c * S) / (V * W - Y * S)
            L = (Z - V * M) / S
    i = 2.0
    j = 2.0
    X = C
    f = []
    for J in E(0, G(z[0])):
        H = []
        for F in E(0, G(z)):
            if not U(z[F, J]):
                H.append(F)
        K = G(H)
        X += K
        e = A.arange(0, G(z[0]))
        g = A.ones(K) * A.nan
        if K > 0:
            for F in E(0, K):
                O = L * I[H[F]] + M - z[H[F], J]
                g[F] = O
            f.append(g)
    h = A.concatenate(f)
    from scipy.stats import iqr
    a = iqr(h)
    P = A.nanmedian(h)
    for J in E(0, G(z[0])):
        H = []
        for F in E(0, G(z)):
            if not U(z[F, J]):
                H.append(F)
        K = G(H)
        X += K
        e = A.arange(0, G(z[0]))
        R = []
        R.append(C)
        R.append(C)
        R.append(C)
        for F in E(0, K):
            O = L * I[H[F]] + M - z[H[F], J]
            if A.abs(R[2]) < A.abs(O - P):
                R[0] = F
                R[1] = J
                R[2] = A.abs(O - P)
            if A.abs(O - P) > 1 * a:
                0
            if A.abs(O - P) > 3 * a:
                if O - P > 0:
                    b[H[F], J] = L * I[H[F]] + M + (i * a + P)
                else:
                    b[H[F], J] = L * I[H[F]] + M + (j * a + P)
    if L > 0.01:
        d = D
    return (b, d, L, M)

def A5(my_array, value):
    B = my_array
    if not F(B, A.ndarray):
        B = A.array(B)
    C = A.abs(B - value).argmin()
    D = B[C]
    return (C, D)

def A6(my_iterable):
    B = I.deepcopy(my_iterable)
    if not F(B, A.ndarray) or not F(B, A.ma.MaskedArray):
        B = A.array(B)
    else:
        B = B
    if B.ndim != 1:
        B = B.flatten()
    return B

def S(my_array):
    B = I.deepcopy(my_array)
    if B.dtype == A.int32 or B.dtype == A.int64:
        if B.size == 0:
            B = A.empty(0, dtype=A.float32)
            return B
        if F(B, A.ma.MaskedArray):
            C = A.ma.max(B)
        else:
            C = A.nanmax(B)
        if B.dtype == A.int32:
            if not A.isnan(C):
                D = 'max value of array is bigger than max allowed valued in float 32'
                if C > A.finfo(A.float32).max:
                    raise P
            B = B.astype(A.float32)
            return B
        if B.dtype == A.int64:
            if not A.isnan(C):
                D = 'max value of array is bigger than max allowed valued in float 64'
                if C > A.finfo(A.float64).max:
                    raise P
            B = B.astype(A.float64)
            return B
    if F(B, A.ma.MaskedArray):
        B = B.filled(A.nan)
    return B

def R(my_array, fill_value=K):
    C = fill_value
    B = I.deepcopy(my_array)
    if not F(B, A.ma.MaskedArray):
        B = A.ma.masked_invalid(B)
    if C is not K:
        B.set_fill_value(C)
    return B

def g(x_array, y_array):
    B, C = A.polyfit(x_array, y_array, 1)
    return (C, B)

def A7(values_in_array, base_in_array, limits=K, check_nan=D, float_atol=L().def_float_atol):
    a = 'lin'
    W = float_atol
    U = base_in_array
    T = values_in_array
    K = limits
    if any([F(B, A.ma.MaskedArray) for B in [T, U]]):
        raise s
    B, H = [A6(A) for A in [T, U]]
    if check_nan:
        if A.any(A.isnan(H)):
            raise P
    if not K:
        M = V(left=A.nan, right=A.nan)
    elif 'fill' in K.lower():
        M = V()
    elif a in K.lower():
        M = V(left=A.nan, right=A.nan)
    else:
        raise O
    D = A.isfinite(B)
    L = D.nonzero()[0]
    G, E = L[[0, -1]]
    if A.all(D):
        return B
    if A.count_nonzero(D) < 2:
        return B
    if K and a in K.lower():
        M = V(left=B[D][0], right=B[D][-1])
    b = A.interp(x=H, xp=H[D], fp=B[D], **M)
    B = b
    if not K or 'fil' in K.lower():
        return B
    if G == 0:
        if E == D.size - 1:
            return B
    if G > 0:
        B[:G] = A.nan
    if E < D.size - 1:
        if E + 1 < D.size - 1:
            B[E + 1:] = A.nan
        else:
            B[E + 1] = A.nan
    D = A.isfinite(B)
    L = D.nonzero()[0]
    G, E = L[[0, -1]]
    if G > 0:
        X = L[1]
        c = H[[G, X]]
        N = I.deepcopy(B[[G, X]])
        if A.isclose(N[0], N[1], rtol=C, atol=W):
            B[:G] = N[0]
        else:
            d, Q = g(c, N)
            B[:G] = d + Q * H[:G]
    if E < D.size - 1:
        Y = L[-2]
        e = H[[Y, E]]
        R = I.deepcopy(B[[Y, E]])
        if A.isclose(R[0], R[1], rtol=C, atol=W):
            S = R[0]
        else:
            Z, Q = g(e, R)
            if E + 1 < D.size - 1:
                S = Z + Q * H[E + 1:]
            else:
                S = Z + Q * H[E + 1]
        if E + 1 < D.size - 1:
            B[E + 1:] = S
        else:
            B[E + 1] = S
    if A.count_nonzero(A.isnan(B)) > 0:
        raise J
    return B

def o(values0_array, space0_array, clean_run=B):
    F = space0_array
    C = values0_array
    H, R = (0, 1)
    if not clean_run:
        0
    if C.ndim != 2:
        raise O
    if (C.shape[H],) != F.shape:
        T = f'space array must be a 1D array with size {C.shape[H]}, not an array with shape {F.shape}'
        raise O
    N = C.shape[0]
    E = I.deepcopy(C)
    try:
        J = D
        L = E.get_fill_value()
        E = S(E)
    except AttributeError:
        J = B
        L = K
    G = I.deepcopy(E)
    P = S(I.deepcopy(F))
    M = A.count_nonzero(A.isfinite(G), axis=0)
    Q = A.nonzero((M >= 1) & (M < G.shape[0]))[0]
    return (E, G, P, Q, N, J, L)

def p(values_out_array, space1_array, dx_max_in, dx_max_out, float_atol=L().def_float_atol):
    P = dx_max_out
    K = space1_array
    H = values_out_array
    M = A.full_like(H, fill_value=A.nan, dtype=A.float32)
    L = A.full_like(H, fill_value=A.nan, dtype=A.float32)
    Q = E(H.shape[0])
    for C in E(H.shape[1]):
        R = A.isnan(H[:, C])
        if A.all(R):
            continue
        if not A.any(R):
            M[:, C] = I.deepcopy(H[:, C])
            continue
        F = A.isfinite(H[:, C]).nonzero()[0]
        S = G(F)
        if S == 0:
            raise J
        elif S == 1:
            B = F[0]
            L[:, C] = B
            L[B, C] = A.nan
            continue
        M[:, C] = A7(values_in_array=H[:, C], base_in_array=K, limits='linear', check_nan=D, float_atol=float_atol)
        N = I.deepcopy(F)
        for B in Q:
            if B in F:
                L[B, C] = B
                continue
            if B > F[-1]:
                L[B, C] = F[-1]
                if A.abs(K[F[-1]] - K[B]) > P:
                    M[B, C] = A.nan
                continue
            if B < F[0]:
                L[B, C] = F[0]
                if A.abs(K[F[0]] - K[B]) > P:
                    M[B, C] = A.nan
                continue
            T, O = A5([A for A in N if A != B], B)
            if O < B:
                N = N[T:]
            L[B, C] = O
            if A.abs(K[O] - K[B]) > dx_max_in:
                M[B, C] = A.nan
    del Q
    return (L, M)

def A8(values0_array, space0_array, dx_max_in, dx_max_out, dw_min, float_atol=L().def_float_atol, interp_missing_nodes=B, clean_run=B, debug_mode=B):
    I = clean_run
    F = debug_mode
    if F:
        I = B
    if I:
        F = B
    E, G, e, f, l, Q, S = o(values0_array=values0_array, space0_array=space0_array, clean_run=I)
    if A.all(A.isnan(G)):
        if Q:
            E = R(E, fill_value=S)
        return E
    if A.all(A.isfinite(G)):
        if Q:
            E = R(E, fill_value=S)
        return E
    if interp_missing_nodes:
        T = A.full(G.shape[0], fill_value=D)
    else:
        T = A.count_nonzero(A.isfinite(G), axis=1) > 0
    g, K = p(values_out_array=E, space1_array=e, dx_max_in=dx_max_in, dx_max_out=dx_max_out, float_atol=float_atol)
    for H in f:
        m = t + str(H)
        L = A.isnan(G[:, H])
        if A.all(~L):
            raise J
        if A.all(L):
            raise J
        if F:
            U = A.nonzero(L)[0]
            h = A.nonzero(T)[0]
            for i in U:
                if i not in h:
                    0
        U = A.nonzero(L & T)[0]
        for M in U:
            if F:
                0
            V = X(g[M, H])
            W = G[V, H]
            if A.isnan(W):
                raise J
            c = A.nonzero(A.isfinite(K[M, :]) & A.isfinite(K[V, :]))[0]
            if c.size == 0:
                if F:
                    0
                continue
            j, O, Z = (0, C, C)
            for a in c:
                if a == H:
                    continue
                d = K[M, a]
                b = K[V, a]
                if A.isnan(d):
                    raise J
                if A.isnan(b):
                    raise J
                k = b - d
                P = (b - W) ** 2
                P = N / max(P, dw_min)
                O += k * P
                Z += P
                j += 1
            if Z > C:
                with Y.catch_warnings():
                    Y.filterwarnings(u, message=v)
                    O = O / Z
                    E[M, H] = W - O
                if F:
                    0
            elif F:
                0
    if not I:
        0
    if Q:
        E = R(E, fill_value=S)
    return E

def A9(values0_array, space0_array, weight0_array, dx_max_in, dx_max_out, dw_min, weight_exp_beta=0.01, float_atol=0.01, clean_run=B, debug_mode=B):
    V = weight0_array
    U = values0_array
    I = clean_run
    E = debug_mode
    if E:
        I = B
    if I:
        E = B
    if U.shape != V.shape:
        raise O
    D, L, f, g, m, M, N = o(values0_array=U, space0_array=space0_array, clean_run=I)
    if A.all(A.isnan(L)):
        if M:
            D = R(D, fill_value=N)
        return D
    if A.all(A.isfinite(L)):
        if M:
            D = R(D, fill_value=N)
        return D
    n, P = p(values_out_array=D, space1_array=f, dx_max_in=dx_max_in, dx_max_out=dx_max_out, float_atol=float_atol)
    G = S(V)
    P[A.isnan(G)] = A.nan
    h = A.nanmax(G) - A.nanmin(G)
    for H in g:
        q = t + str(H)
        K = A.isnan(L[:, H])
        if A.all(~K):
            raise J
        if A.all(K):
            raise J
        W = A.isfinite(G[:, H])
        if E:
            i = A.nonzero(K)[0]
            j = A.nonzero(W)[0]
            for F in i:
                if F not in j:
                    0
        k = A.nonzero(K & W)[0]
        for F in k:
            if E:
                0
            X = G[F, H]
            if A.isnan(X):
                raise J
            Z = A.isfinite(P[F, :]).nonzero()[0]
            if Z.size == 0:
                if E:
                    0
                continue
            l, a, Q = (0, C, C)
            for T in Z:
                if T == H:
                    continue
                b = P[F, T]
                c = G[F, T]
                if A.isnan(b):
                    raise J
                if A.isnan(c):
                    raise J
                d = A.abs(X - c)
                d /= h
                e = A.exp(-weight_exp_beta * d)
                a += b * e
                Q += e
                l += 1
            if Q > C:
                with Y.catch_warnings():
                    Y.filterwarnings(u, message=v)
                    D[F, H] = a / Q
                if E:
                    0
            elif E:
                0
    if not I:
        0
    if M:
        D = R(D, fill_value=N)
    return D

def AA(node_z, corx, node_x, LSMX, eps1, eps2):
    O = node_x
    D = node_z
    R = D.shape[1]
    H = D.shape[0]
    I = A.zeros(H)
    S = Q(D)
    for F in E(0, R):
        G = C
        for B in E(0, H):
            G = G + D[B, F]
        P = G / H
        J = C
        for B in E(0, H):
            J = J + (D[B, F] - P) ** 2
        J = A.sqrt(J / H)
        for B in E(1, H):
            I[B] = N - A.exp(-N * ((O[B] - O[B - 1]) / corx))
        L = C
        for M in E(0, LSMX):
            K = 0
            for B in E(H - 2, 0, -1):
                D[B, F] = D[B + 1, F] * (N - I[B + 1]) + D[B, F] * I[B + 1]
                if M > 0:
                    if D[B, F] < D[B + 1, F] - eps1:
                        G = D[B, F]
                        D[B, F] = D[B + 1, F]
                        D[B + 1, F] = G
                        K = K + 1
            for B in E(1, H):
                D[B, F] = D[B - 1, F] * (N - I[B]) + D[B, F] * I[B]
                if M > 0:
                    if D[B, F] > D[B - 1, F] + eps1:
                        G = D[B - 1, F]
                        D[B - 1, F] = D[B, F]
                        D[B, F] = G
                        K = K + 1
            G = C
            for B in E(0, H):
                G = G + (D[B, F] - S[B, F]) ** 2
            G = A.sqrt(G / H)
            if M > 0:
                if A.abs((G - L) / J) > eps2:
                    L = G
                else:
                    break
            else:
                L = G
        G = C
        for B in E(0, H):
            G = G + D[B, F]
        T = G / H
        for B in E(0, H):
            D[B, F] = D[B, F] + P - T
    return D

def Z(array, cslope, node_x):
    K = cslope
    H = node_x
    C = array
    from lib.lib_verif import check_na as L
    D = -1
    M = -1
    F = -1
    I = Q(C)
    for B in E(0, G(C)):
        if not L(C[B]):
            D = B
            I[B] = C[B]
        else:
            if B > F:
                J = -1
                if M != 0:
                    for N in E(B + 1, G(C)):
                        if not L(C[N]):
                            J = N
                            break
                        else:
                            0
            F = J
            if J == -1:
                M = 0
            if D != -1 and F != -1:
                I[B] = C[D] + (C[F] - C[D]) * (H[B] - H[D]) / (H[F] - H[D])
            elif D == -1 and F != -1:
                I[B] = C[F] + K * (H[B] - H[F])
            elif D != -1 and F == -1:
                I[B] = C[D] + K * (H[B] - H[D])
            elif D == -1 and F == -1:
                I[B] = A.nan
    return I

def AB(node_z, node_w, node_x, reach_t, corx, corx_array, cort_wse, cort_width, gnuplot_saving=B, reach_id=K, output_dir=K, run_type=w, use_large_deviations=D, start_from_downstream=B, run_preprocessing=D, run_extrapolation=D, pooling=B, regularization_profile=B, use_quantiles_for_densification=B):
    Y = run_extrapolation
    X = start_from_downstream
    W = cort_wse
    V = corx_array
    R = 'tmp_interp_values_w'
    P = 'tmp_interp_values'
    N = run_preprocessing
    L = reach_t
    J = node_x
    G = node_w
    F = node_z
    h = Q(F)
    I = {}
    I[P] = []
    I[R] = []
    U = B
    if use_large_deviations and run_type == w:
        F, U, K, i = A4(J, F, L, times_debug=L)
        j = Q(F)
        I[P].append(F)
    O = H
    if U:
        if X:
            O = k
        else:
            O = l
    elif X:
        O = l
    else:
        O = k
    I['reverse_order'] = U
    if N:
        F = M(dim=1, value0_array=F, base0_array=J, max_iter=1, cor=V, always_run_first_iter=D, behavior=O, inter_behavior=D, inter_behavior_min_thr=0.01, inter_behavior_max_thr=A.inf, check_behavior=f, min_change_v_thr=0.0001, plot=B, plot_title='Relaxation sweep in space 1 WSE without Interchange', clean_run=D, debug_mode=B, time_integration=B)
    I[P].append(F)
    if regularization_profile:
        if K == C:
            K = -1e-06
        else:
            K = K * 0.001
        if not use_quantiles_for_densification:
            from sic4dvar_functions.sic4dvar_helper_functions import compute_mean_elevation_profile as a
            c = a(F, L)
            d = Z(c, K, J)
            K = C
            e = a(G, L)
            g = Z(e, K, J)
            F[:, -1] = d
            G[:, -1] = g
        else:
            S = A.zeros((F.shape[0], 3))
            for b in E(0, F.shape[0]):
                S[b, :] = A.nanquantile(F[b, :], [0.33, 0.66, 0.99])
            for T in E(0, 3):
                S[:, T] = Z(S[:, T], K, J)
                F[:, -1 - T] = S[:, T]
    m = Q(F)
    if Y:
        F = A8(values0_array=F, space0_array=J, dx_max_in=A.inf, dx_max_out=500, dw_min=0.01, clean_run=B, debug_mode=B, interp_missing_nodes=D)
    I[P].append(F)
    if N:
        for n in E(0, 1):
            F = M(dim=1, value0_array=F, base0_array=J, max_iter=10, cor=V, always_run_first_iter=D, behavior=O, inter_behavior=D, inter_behavior_min_thr=0.01, inter_behavior_max_thr=500, check_behavior=f, min_change_v_thr=0.0001, plot=B, plot_title='Relaxation sweep in space 2 WSE With Interchange', clean_run=D, debug_mode=B, time_integration=B)
    I[P].append(F)
    o = Q(F)
    if N:
        F = M(dim=0, value0_array=F, base0_array=L, max_iter=1, cor=W, always_run_first_iter=B, behavior=H, inter_behavior=B, inter_behavior_min_thr=0.01, inter_behavior_max_thr=500, check_behavior=H, min_change_v_thr=0.0001, plot=B, plot_title='Relaxation sweep in time WSE without Interchange', clean_run=D, debug_mode=B, time_integration=B)
    I[P].append(F)
    if N:
        G = M(dim=1, value0_array=G, base0_array=J, max_iter=1, cor=corx, always_run_first_iter=D, behavior=H, inter_behavior=B, check_behavior=H, min_change_v_thr=0.01, plot=B, plot_title='Relaxation sweep in space 1 W without Interchange', clean_run=D, debug_mode=B, time_integration=B)
    I[R].append(G)
    if Y:
        G = A9(values0_array=G, space0_array=J, weight0_array=F, dx_max_in=A.inf, dx_max_out=500, dw_min=0.1, clean_run=B, debug_mode=B)
    I[R].append(G)
    if N:
        G = M(dim=1, value0_array=G, base0_array=J, max_iter=1, cor=corx, always_run_first_iter=D, behavior=H, inter_behavior=B, check_behavior=H, min_change_v_thr=0.01, plot=B, plot_title='Relaxation sweep in space 2 W without Interchange', clean_run=D, debug_mode=B, time_integration=B)
    I[R].append(G)
    if N:
        G = M(dim=0, value0_array=G, base0_array=L, max_iter=1, cor=W, always_run_first_iter=D, behavior=H, inter_behavior=B, check_behavior=H, min_change_v_thr=0.01, plot=B, plot_title='Relaxation sweep in time W without Interchange', clean_run=D, debug_mode=B, time_integration=B)
    I[R].append(G)
    return (A.ma.filled(F, A.nan), A.ma.filled(G, A.nan), I)

def AC(node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, ZB_update, Zb, t, node_x, SLOPEM1, ZM=N, KMI=N, option_recompute_area=B, sic4dvar_dict=[]):
    S = node_p
    R = node_a
    J = node_xr
    I = node_yr
    F = node_x
    L = C
    G = []
    for B in E(last_node_for_integral):
        Z, H = (A.nanmin(I[B]), A.nanmin(J[B]))
        if H == 0:
            H = N
        if t == 0:
            Zb[B] = I[B][0] + Zb2 + ZB_update * (Wmean / H)
        M = I[B][0]
        O = J[B][0]
        if Z != M:
            0
        if H != O:
            0
        if H == 0 or U(H):
            0
        if O == 0:
            0
        if option_recompute_area:
            a = J[B][~A.isnan(J[B])]
            b = I[B][~A.isnan(I)[B]]
            R[B], S[B], c, c = x([sic4dvar_dict['filtered_data']['node_z'][B, t] * ZM], a, b)
        d = R[B][t]
        e = S[B][t]
        V = O * (M - Zb[B])
        if U(V):
            0
        f = 2 * (M - Zb[B])
        P = V + d
        X = f + e
        if P < 0:
            0
        if X < 0:
            0
        Y = P / X
        G.append(Y)
        D = P * W.pow(Y, 2.0 / 3.0)
        if B == 0:
            K = D
        else:
            if K <= 0 or D <= 0:
                0
            if K > 0 and D > 0:
                L += T(F[B] - F[B - 1]) * (W.pow(K, -2) + W.pow(D, -2)) / 2.0
            K = D
    G = A.array(G)
    G = A.mean(G)
    Q = SLOPEM1[t] * ZM
    if Q / T(F[-1] - F[0]) < 1e-06:
        Q = T(F[-1] - F[0]) * 1e-06
    D = A.sqrt(Q / L) * KMI
    return (D, L, Zb, G)