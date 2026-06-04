var Tc = e => { throw TypeError(e) };
var gl = (e, t, n) => t.has(e) || Tc("Cannot " + n);
var b = (e, t, n) => (gl(e, t, "read from private field"), n ? n.call(e) : t.get(e)),
    q = (e, t, n) => t.has(e) ? Tc("Cannot add the same private member more than once") : t instanceof WeakSet ? t.add(e) : t.set(e, n),
    F = (e, t, n, r) => (gl(e, t, "write to private field"), r ? r.call(e, n) : t.set(e, n), n),
    Ee = (e, t, n) => (gl(e, t, "access private method"), n);
var ws = (e, t, n, r) => ({set _(o) { F(e, t, o, n) }, get _() { return b(e, t, r) } });

function rv(e, t) {
    for (var n = 0; n < t.length; n++) {
        const r = t[n];
        if (typeof r != "string" && !Array.isArray(r)) {
            for (const o in r)
                if (o !== "default" && !(o in e)) {
                    const s = Object.getOwnPropertyDescriptor(r, o);
                    s && Object.defineProperty(e, o, s.get ? s : { enumerable: !0, get: () => r[o] })
                }
        }
    }
    return Object.freeze(Object.defineProperty(e, Symbol.toStringTag, { value: "Module" }))
}(function() {
    const t = document.createElement("link").relList;
    if (t && t.supports && t.supports("modulepreload")) return;
    for (const o of document.querySelectorAll('link[rel="modulepreload"]')) r(o);
    new MutationObserver(o => {
        for (const s of o)
            if (s.type === "childList")
                for (const i of s.addedNodes) i.tagName === "LINK" && i.rel === "modulepreload" && r(i)
    }).observe(document, { childList: !0, subtree: !0 });

    function n(o) { const s = {}; return o.integrity && (s.integrity = o.integrity), o.referrerPolicy && (s.referrerPolicy = o.referrerPolicy), o.crossOrigin === "use-credentials" ? s.credentials = "include" : o.crossOrigin === "anonymous" ? s.credentials = "omit" : s.credentials = "same-origin", s }

    function r(o) {
        if (o.ep) return;
        o.ep = !0;
        const s = n(o);
        fetch(o.href, s)
    }
})();

function Mf(e) { return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e }
var If = { exports: {} },
    Ii = {},
    Df = { exports: {} },
    G = {};
/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var as = Symbol.for("react.element"),
    ov = Symbol.for("react.portal"),
    sv = Symbol.for("react.fragment"),
    iv = Symbol.for("react.strict_mode"),
    lv = Symbol.for("react.profiler"),
    av = Symbol.for("react.provider"),
    uv = Symbol.for("react.context"),
    cv = Symbol.for("react.forward_ref"),
    dv = Symbol.for("react.suspense"),
    fv = Symbol.for("react.memo"),
    pv = Symbol.for("react.lazy"),
    Nc = Symbol.iterator;

function hv(e) { return e === null || typeof e != "object" ? null : (e = Nc && e[Nc] || e["@@iterator"], typeof e == "function" ? e : null) }
var zf = { isMounted: function() { return !1 }, enqueueForceUpdate: function() {}, enqueueReplaceState: function() {}, enqueueSetState: function() {} },
    Ff = Object.assign,
    $f = {};

function io(e, t, n) { this.props = e, this.context = t, this.refs = $f, this.updater = n || zf }
io.prototype.isReactComponent = {};
io.prototype.setState = function(e, t) {
    if (typeof e != "object" && typeof e != "function" && e != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
    this.updater.enqueueSetState(this, e, t, "setState")
};
io.prototype.forceUpdate = function(e) { this.updater.enqueueForceUpdate(this, e, "forceUpdate") };

function Bf() {}
Bf.prototype = io.prototype;

function lu(e, t, n) { this.props = e, this.context = t, this.refs = $f, this.updater = n || zf }
var au = lu.prototype = new Bf;
au.constructor = lu;
Ff(au, io.prototype);
au.isPureReactComponent = !0;
var Rc = Array.isArray,
    Uf = Object.prototype.hasOwnProperty,
    uu = { current: null },
    Vf = { key: !0, ref: !0, __self: !0, __source: !0 };

function Wf(e, t, n) {
    var r, o = {},
        s = null,
        i = null;
    if (t != null)
        for (r in t.ref !== void 0 && (i = t.ref), t.key !== void 0 && (s = "" + t.key), t) Uf.call(t, r) && !Vf.hasOwnProperty(r) && (o[r] = t[r]);
    var l = arguments.length - 2;
    if (l === 1) o.children = n;
    else if (1 < l) {
        for (var a = Array(l), u = 0; u < l; u++) a[u] = arguments[u + 2];
        o.children = a
    }
    if (e && e.defaultProps)
        for (r in l = e.defaultProps, l) o[r] === void 0 && (o[r] = l[r]);
    return { $$typeof: as, type: e, key: s, ref: i, props: o, _owner: uu.current }
}

function mv(e, t) { return { $$typeof: as, type: e.type, key: t, ref: e.ref, props: e.props, _owner: e._owner } }

function cu(e) { return typeof e == "object" && e !== null && e.$$typeof === as }

function gv(e) { var t = { "=": "=0", ":": "=2" }; return "$" + e.replace(/[=:]/g, function(n) { return t[n] }) }
var jc = /\/+/g;

function vl(e, t) { return typeof e == "object" && e !== null && e.key != null ? gv("" + e.key) : t.toString(36) }

function Vs(e, t, n, r, o) {
    var s = typeof e;
    (s === "undefined" || s === "boolean") && (e = null);
    var i = !1;
    if (e === null) i = !0;
    else switch (s) {
        case "string":
        case "number":
            i = !0;
            break;
        case "object":
            switch (e.$$typeof) {
                case as:
                case ov:
                    i = !0
            }
    }
    if (i) return i = e, o = o(i), e = r === "" ? "." + vl(i, 0) : r, Rc(o) ? (n = "", e != null && (n = e.replace(jc, "$&/") + "/"), Vs(o, t, n, "", function(u) { return u })) : o != null && (cu(o) && (o = mv(o, n + (!o.key || i && i.key === o.key ? "" : ("" + o.key).replace(jc, "$&/") + "/") + e)), t.push(o)), 1;
    if (i = 0, r = r === "" ? "." : r + ":", Rc(e))
        for (var l = 0; l < e.length; l++) {
            s = e[l];
            var a = r + vl(s, l);
            i += Vs(s, t, n, a, o)
        } else if (a = hv(e), typeof a == "function")
            for (e = a.call(e), l = 0; !(s = e.next()).done;) s = s.value, a = r + vl(s, l++), i += Vs(s, t, n, a, o);
        else if (s === "object") throw t = String(e), Error("Objects are not valid as a React child (found: " + (t === "[object Object]" ? "object with keys {" + Object.keys(e).join(", ") + "}" : t) + "). If you meant to render a collection of children, use an array instead.");
    return i
}

function xs(e, t, n) {
    if (e == null) return e;
    var r = [],
        o = 0;
    return Vs(e, r, "", "", function(s) { return t.call(n, s, o++) }), r
}

function vv(e) {
    if (e._status === -1) {
        var t = e._result;
        t = t(), t.then(function(n) {
            (e._status === 0 || e._status === -1) && (e._status = 1, e._result = n)
        }, function(n) {
            (e._status === 0 || e._status === -1) && (e._status = 2, e._result = n)
        }), e._status === -1 && (e._status = 0, e._result = t)
    }
    if (e._status === 1) return e._result.default;
    throw e._result
}
var De = { current: null },
    Ws = { transition: null },
    yv = { ReactCurrentDispatcher: De, ReactCurrentBatchConfig: Ws, ReactCurrentOwner: uu };

function Hf() { throw Error("act(...) is not supported in production builds of React.") }
G.Children = { map: xs, forEach: function(e, t, n) { xs(e, function() { t.apply(this, arguments) }, n) }, count: function(e) { var t = 0; return xs(e, function() { t++ }), t }, toArray: function(e) { return xs(e, function(t) { return t }) || [] }, only: function(e) { if (!cu(e)) throw Error("React.Children.only expected to receive a single React element child."); return e } };
G.Component = io;
G.Fragment = sv;
G.Profiler = lv;
G.PureComponent = lu;
G.StrictMode = iv;
G.Suspense = dv;
G.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = yv;
G.act = Hf;
G.cloneElement = function(e, t, n) {
    if (e == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + e + ".");
    var r = Ff({}, e.props),
        o = e.key,
        s = e.ref,
        i = e._owner;
    if (t != null) { if (t.ref !== void 0 && (s = t.ref, i = uu.current), t.key !== void 0 && (o = "" + t.key), e.type && e.type.defaultProps) var l = e.type.defaultProps; for (a in t) Uf.call(t, a) && !Vf.hasOwnProperty(a) && (r[a] = t[a] === void 0 && l !== void 0 ? l[a] : t[a]) }
    var a = arguments.length - 2;
    if (a === 1) r.children = n;
    else if (1 < a) {
        l = Array(a);
        for (var u = 0; u < a; u++) l[u] = arguments[u + 2];
        r.children = l
    }
    return { $$typeof: as, type: e.type, key: o, ref: s, props: r, _owner: i }
};
G.createContext = function(e) { return e = { $$typeof: uv, _currentValue: e, _currentValue2: e, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null }, e.Provider = { $$typeof: av, _context: e }, e.Consumer = e };
G.createElement = Wf;
G.createFactory = function(e) { var t = Wf.bind(null, e); return t.type = e, t };
G.createRef = function() { return { current: null } };
G.forwardRef = function(e) { return { $$typeof: cv, render: e } };
G.isValidElement = cu;
G.lazy = function(e) { return { $$typeof: pv, _payload: { _status: -1, _result: e }, _init: vv } };
G.memo = function(e, t) { return { $$typeof: fv, type: e, compare: t === void 0 ? null : t } };
G.startTransition = function(e) {
    var t = Ws.transition;
    Ws.transition = {};
    try { e() } finally { Ws.transition = t }
};
G.unstable_act = Hf;
G.useCallback = function(e, t) { return De.current.useCallback(e, t) };
G.useContext = function(e) { return De.current.useContext(e) };
G.useDebugValue = function() {};
G.useDeferredValue = function(e) { return De.current.useDeferredValue(e) };
G.useEffect = function(e, t) { return De.current.useEffect(e, t) };
G.useId = function() { return De.current.useId() };
G.useImperativeHandle = function(e, t, n) { return De.current.useImperativeHandle(e, t, n) };
G.useInsertionEffect = function(e, t) { return De.current.useInsertionEffect(e, t) };
G.useLayoutEffect = function(e, t) { return De.current.useLayoutEffect(e, t) };
G.useMemo = function(e, t) { return De.current.useMemo(e, t) };
G.useReducer = function(e, t, n) { return De.current.useReducer(e, t, n) };
G.useRef = function(e) { return De.current.useRef(e) };
G.useState = function(e) { return De.current.useState(e) };
G.useSyncExternalStore = function(e, t, n) { return De.current.useSyncExternalStore(e, t, n) };
G.useTransition = function() { return De.current.useTransition() };
G.version = "18.3.1";
Df.exports = G;
var w = Df.exports;
const _ = Mf(w),
    du = rv({ __proto__: null, default: _ }, [w]);
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var wv = w,
    xv = Symbol.for("react.element"),
    Sv = Symbol.for("react.fragment"),
    Ev = Object.prototype.hasOwnProperty,
    Cv = wv.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,
    kv = { key: !0, ref: !0, __self: !0, __source: !0 };

function Qf(e, t, n) {
    var r, o = {},
        s = null,
        i = null;
    n !== void 0 && (s = "" + n), t.key !== void 0 && (s = "" + t.key), t.ref !== void 0 && (i = t.ref);
    for (r in t) Ev.call(t, r) && !kv.hasOwnProperty(r) && (o[r] = t[r]);
    if (e && e.defaultProps)
        for (r in t = e.defaultProps, t) o[r] === void 0 && (o[r] = t[r]);
    return { $$typeof: xv, type: e, key: s, ref: i, props: o, _owner: Cv.current }
}
Ii.Fragment = Sv;
Ii.jsx = Qf;
Ii.jsxs = Qf;
If.exports = Ii;
var v = If.exports,
    Kf = { exports: {} },
    et = {},
    Gf = { exports: {} },
    Yf = {};
/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
(function(e) {
    function t(T, R) {
        var M = T.length;
        T.push(R);
        e: for (; 0 < M;) {
            var W = M - 1 >>> 1,
                z = T[W];
            if (0 < o(z, R)) T[W] = R, T[M] = z, M = W;
            else break e
        }
    }

    function n(T) { return T.length === 0 ? null : T[0] }

    function r(T) {
        if (T.length === 0) return null;
        var R = T[0],
            M = T.pop();
        if (M !== R) {
            T[0] = M;
            e: for (var W = 0, z = T.length, K = z >>> 1; W < K;) {
                var X = 2 * (W + 1) - 1,
                    he = T[X],
                    Ne = X + 1,
                    J = T[Ne];
                if (0 > o(he, M)) Ne < z && 0 > o(J, he) ? (T[W] = J, T[Ne] = M, W = Ne) : (T[W] = he, T[X] = M, W = X);
                else if (Ne < z && 0 > o(J, M)) T[W] = J, T[Ne] = M, W = Ne;
                else break e
            }
        }
        return R
    }

    function o(T, R) { var M = T.sortIndex - R.sortIndex; return M !== 0 ? M : T.id - R.id }
    if (typeof performance == "object" && typeof performance.now == "function") {
        var s = performance;
        e.unstable_now = function() { return s.now() }
    } else {
        var i = Date,
            l = i.now();
        e.unstable_now = function() { return i.now() - l }
    }
    var a = [],
        u = [],
        c = 1,
        f = null,
        m = 3,
        d = !1,
        S = !1,
        y = !1,
        x = typeof setTimeout == "function" ? setTimeout : null,
        h = typeof clearTimeout == "function" ? clearTimeout : null,
        p = typeof setImmediate < "u" ? setImmediate : null;
    typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);

    function g(T) {
        for (var R = n(u); R !== null;) {
            if (R.callback === null) r(u);
            else if (R.startTime <= T) r(u), R.sortIndex = R.expirationTime, t(a, R);
            else break;
            R = n(u)
        }
    }

    function E(T) {
        if (y = !1, g(T), !S)
            if (n(a) !== null) S = !0, B(C);
            else {
                var R = n(u);
                R !== null && V(E, R.startTime - T)
            }
    }

    function C(T, R) {
        S = !1, y && (y = !1, h(N), N = -1), d = !0;
        var M = m;
        try {
            for (g(R), f = n(a); f !== null && (!(f.expirationTime > R) || T && !$());) {
                var W = f.callback;
                if (typeof W == "function") {
                    f.callback = null, m = f.priorityLevel;
                    var z = W(f.expirationTime <= R);
                    R = e.unstable_now(), typeof z == "function" ? f.callback = z : f === n(a) && r(a), g(R)
                } else r(a);
                f = n(a)
            }
            if (f !== null) var K = !0;
            else {
                var X = n(u);
                X !== null && V(E, X.startTime - R), K = !1
            }
            return K
        } finally { f = null, m = M, d = !1 }
    }
    var k = !1,
        P = null,
        N = -1,
        L = 5,
        A = -1;

    function $() { return !(e.unstable_now() - A < L) }

    function D() {
        if (P !== null) {
            var T = e.unstable_now();
            A = T;
            var R = !0;
            try { R = P(!0, T) } finally { R ? Q() : (k = !1, P = null) }
        } else k = !1
    }
    var Q;
    if (typeof p == "function") Q = function() { p(D) };
    else if (typeof MessageChannel < "u") {
        var O = new MessageChannel,
            Y = O.port2;
        O.port1.onmessage = D, Q = function() { Y.postMessage(null) }
    } else Q = function() { x(D, 0) };

    function B(T) { P = T, k || (k = !0, Q()) }

    function V(T, R) { N = x(function() { T(e.unstable_now()) }, R) }
    e.unstable_IdlePriority = 5, e.unstable_ImmediatePriority = 1, e.unstable_LowPriority = 4, e.unstable_NormalPriority = 3, e.unstable_Profiling = null, e.unstable_UserBlockingPriority = 2, e.unstable_cancelCallback = function(T) { T.callback = null }, e.unstable_continueExecution = function() { S || d || (S = !0, B(C)) }, e.unstable_forceFrameRate = function(T) { 0 > T || 125 < T ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : L = 0 < T ? Math.floor(1e3 / T) : 5 }, e.unstable_getCurrentPriorityLevel = function() { return m }, e.unstable_getFirstCallbackNode = function() { return n(a) }, e.unstable_next = function(T) {
        switch (m) {
            case 1:
            case 2:
            case 3:
                var R = 3;
                break;
            default:
                R = m
        }
        var M = m;
        m = R;
        try { return T() } finally { m = M }
    }, e.unstable_pauseExecution = function() {}, e.unstable_requestPaint = function() {}, e.unstable_runWithPriority = function(T, R) {
        switch (T) {
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
                break;
            default:
                T = 3
        }
        var M = m;
        m = T;
        try { return R() } finally { m = M }
    }, e.unstable_scheduleCallback = function(T, R, M) {
        var W = e.unstable_now();
        switch (typeof M == "object" && M !== null ? (M = M.delay, M = typeof M == "number" && 0 < M ? W + M : W) : M = W, T) {
            case 1:
                var z = -1;
                break;
            case 2:
                z = 250;
                break;
            case 5:
                z = 1073741823;
                break;
            case 4:
                z = 1e4;
                break;
            default:
                z = 5e3
        }
        return z = M + z, T = { id: c++, callback: R, priorityLevel: T, startTime: M, expirationTime: z, sortIndex: -1 }, M > W ? (T.sortIndex = M, t(u, T), n(a) === null && T === n(u) && (y ? (h(N), N = -1) : y = !0, V(E, M - W))) : (T.sortIndex = z, t(a, T), S || d || (S = !0, B(C))), T
    }, e.unstable_shouldYield = $, e.unstable_wrapCallback = function(T) {
        var R = m;
        return function() {
            var M = m;
            m = R;
            try { return T.apply(this, arguments) } finally { m = M }
        }
    }
})(Yf);
Gf.exports = Yf;
var Pv = Gf.exports;
/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var bv = w,
    Je = Pv;

function j(e) { for (var t = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, n = 1; n < arguments.length; n++) t += "&args[]=" + encodeURIComponent(arguments[n]); return "Minified React error #" + e + "; visit " + t + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings." }
var Xf = new Set,
    Fo = {};

function cr(e, t) { Xr(e, t), Xr(e + "Capture", t) }

function Xr(e, t) { for (Fo[e] = t, e = 0; e < t.length; e++) Xf.add(t[e]) }
var Kt = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"),
    ql = Object.prototype.hasOwnProperty,
    Tv = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,
    _c = {},
    Ac = {};

function Nv(e) { return ql.call(Ac, e) ? !0 : ql.call(_c, e) ? !1 : Tv.test(e) ? Ac[e] = !0 : (_c[e] = !0, !1) }

function Rv(e, t, n, r) {
    if (n !== null && n.type === 0) return !1;
    switch (typeof t) {
        case "function":
        case "symbol":
            return !0;
        case "boolean":
            return r ? !1 : n !== null ? !n.acceptsBooleans : (e = e.toLowerCase().slice(0, 5), e !== "data-" && e !== "aria-");
        default:
            return !1
    }
}

function jv(e, t, n, r) {
    if (t === null || typeof t > "u" || Rv(e, t, n, r)) return !0;
    if (r) return !1;
    if (n !== null) switch (n.type) {
        case 3:
            return !t;
        case 4:
            return t === !1;
        case 5:
            return isNaN(t);
        case 6:
            return isNaN(t) || 1 > t
    }
    return !1
}

function ze(e, t, n, r, o, s, i) { this.acceptsBooleans = t === 2 || t === 3 || t === 4, this.attributeName = r, this.attributeNamespace = o, this.mustUseProperty = n, this.propertyName = e, this.type = t, this.sanitizeURL = s, this.removeEmptyString = i }
var Pe = {};
"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e) { Pe[e] = new ze(e, 0, !1, e, null, !1, !1) });
[
    ["acceptCharset", "accept-charset"],
    ["className", "class"],
    ["htmlFor", "for"],
    ["httpEquiv", "http-equiv"]
].forEach(function(e) {
    var t = e[0];
    Pe[t] = new ze(t, 1, !1, e[1], null, !1, !1)
});
["contentEditable", "draggable", "spellCheck", "value"].forEach(function(e) { Pe[e] = new ze(e, 2, !1, e.toLowerCase(), null, !1, !1) });
["autoReverse", "externalResourcesRequired", "focusable", "preserveAlpha"].forEach(function(e) { Pe[e] = new ze(e, 2, !1, e, null, !1, !1) });
"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e) { Pe[e] = new ze(e, 3, !1, e.toLowerCase(), null, !1, !1) });
["checked", "multiple", "muted", "selected"].forEach(function(e) { Pe[e] = new ze(e, 3, !0, e, null, !1, !1) });
["capture", "download"].forEach(function(e) { Pe[e] = new ze(e, 4, !1, e, null, !1, !1) });
["cols", "rows", "size", "span"].forEach(function(e) { Pe[e] = new ze(e, 6, !1, e, null, !1, !1) });
["rowSpan", "start"].forEach(function(e) { Pe[e] = new ze(e, 5, !1, e.toLowerCase(), null, !1, !1) });
var fu = /[\-:]([a-z])/g;

function pu(e) { return e[1].toUpperCase() }
"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e) {
    var t = e.replace(fu, pu);
    Pe[t] = new ze(t, 1, !1, e, null, !1, !1)
});
"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e) {
    var t = e.replace(fu, pu);
    Pe[t] = new ze(t, 1, !1, e, "http://www.w3.org/1999/xlink", !1, !1)
});
["xml:base", "xml:lang", "xml:space"].forEach(function(e) {
    var t = e.replace(fu, pu);
    Pe[t] = new ze(t, 1, !1, e, "http://www.w3.org/XML/1998/namespace", !1, !1)
});
["tabIndex", "crossOrigin"].forEach(function(e) { Pe[e] = new ze(e, 1, !1, e.toLowerCase(), null, !1, !1) });
Pe.xlinkHref = new ze("xlinkHref", 1, !1, "xlink:href", "http://www.w3.org/1999/xlink", !0, !1);
["src", "href", "action", "formAction"].forEach(function(e) { Pe[e] = new ze(e, 1, !1, e.toLowerCase(), null, !0, !0) });

function hu(e, t, n, r) {
    var o = Pe.hasOwnProperty(t) ? Pe[t] : null;
    (o !== null ? o.type !== 0 : r || !(2 < t.length) || t[0] !== "o" && t[0] !== "O" || t[1] !== "n" && t[1] !== "N") && (jv(t, n, o, r) && (n = null), r || o === null ? Nv(t) && (n === null ? e.removeAttribute(t) : e.setAttribute(t, "" + n)) : o.mustUseProperty ? e[o.propertyName] = n === null ? o.type === 3 ? !1 : "" : n : (t = o.attributeName, r = o.attributeNamespace, n === null ? e.removeAttribute(t) : (o = o.type, n = o === 3 || o === 4 && n === !0 ? "" : "" + n, r ? e.setAttributeNS(r, t, n) : e.setAttribute(t, n))))
}
var en = bv.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,
    Ss = Symbol.for("react.element"),
    wr = Symbol.for("react.portal"),
    xr = Symbol.for("react.fragment"),
    mu = Symbol.for("react.strict_mode"),
    Zl = Symbol.for("react.profiler"),
    qf = Symbol.for("react.provider"),
    Zf = Symbol.for("react.context"),
    gu = Symbol.for("react.forward_ref"),
    Jl = Symbol.for("react.suspense"),
    ea = Symbol.for("react.suspense_list"),
    vu = Symbol.for("react.memo"),
    pn = Symbol.for("react.lazy"),
    Jf = Symbol.for("react.offscreen"),
    Oc = Symbol.iterator;

function mo(e) { return e === null || typeof e != "object" ? null : (e = Oc && e[Oc] || e["@@iterator"], typeof e == "function" ? e : null) }
var ce = Object.assign,
    yl;

function Po(e) {
    if (yl === void 0) try { throw Error() } catch (n) {
        var t = n.stack.trim().match(/\n( *(at )?)/);
        yl = t && t[1] || ""
    }
    return `
` + yl + e
}
var wl = !1;

function xl(e, t) {
    if (!e || wl) return "";
    wl = !0;
    var n = Error.prepareStackTrace;
    Error.prepareStackTrace = void 0;
    try {
        if (t)
            if (t = function() { throw Error() }, Object.defineProperty(t.prototype, "props", { set: function() { throw Error() } }), typeof Reflect == "object" && Reflect.construct) {
                try { Reflect.construct(t, []) } catch (u) { var r = u }
                Reflect.construct(e, [], t)
            } else {
                try { t.call() } catch (u) { r = u }
                e.call(t.prototype)
            }
        else {
            try { throw Error() } catch (u) { r = u }
            e()
        }
    } catch (u) {
        if (u && r && typeof u.stack == "string") {
            for (var o = u.stack.split(`
`), s = r.stack.split(`
`), i = o.length - 1, l = s.length - 1; 1 <= i && 0 <= l && o[i] !== s[l];) l--;
            for (; 1 <= i && 0 <= l; i--, l--)
                if (o[i] !== s[l]) {
                    if (i !== 1 || l !== 1)
                        do
                            if (i--, l--, 0 > l || o[i] !== s[l]) { var a = `
` + o[i].replace(" at new ", " at "); return e.displayName && a.includes("<anonymous>") && (a = a.replace("<anonymous>", e.displayName)), a }
                    while (1 <= i && 0 <= l);
                    break
                }
        }
    } finally { wl = !1, Error.prepareStackTrace = n }
    return (e = e ? e.displayName || e.name : "") ? Po(e) : ""
}

function _v(e) {
    switch (e.tag) {
        case 5:
            return Po(e.type);
        case 16:
            return Po("Lazy");
        case 13:
            return Po("Suspense");
        case 19:
            return Po("SuspenseList");
        case 0:
        case 2:
        case 15:
            return e = xl(e.type, !1), e;
        case 11:
            return e = xl(e.type.render, !1), e;
        case 1:
            return e = xl(e.type, !0), e;
        default:
            return ""
    }
}

function ta(e) {
    if (e == null) return null;
    if (typeof e == "function") return e.displayName || e.name || null;
    if (typeof e == "string") return e;
    switch (e) {
        case xr:
            return "Fragment";
        case wr:
            return "Portal";
        case Zl:
            return "Profiler";
        case mu:
            return "StrictMode";
        case Jl:
            return "Suspense";
        case ea:
            return "SuspenseList"
    }
    if (typeof e == "object") switch (e.$$typeof) {
        case Zf:
            return (e.displayName || "Context") + ".Consumer";
        case qf:
            return (e._context.displayName || "Context") + ".Provider";
        case gu:
            var t = e.render;
            return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
        case vu:
            return t = e.displayName || null, t !== null ? t : ta(e.type) || "Memo";
        case pn:
            t = e._payload, e = e._init;
            try { return ta(e(t)) } catch {}
    }
    return null
}

function Av(e) {
    var t = e.type;
    switch (e.tag) {
        case 24:
            return "Cache";
        case 9:
            return (t.displayName || "Context") + ".Consumer";
        case 10:
            return (t._context.displayName || "Context") + ".Provider";
        case 18:
            return "DehydratedFragment";
        case 11:
            return e = t.render, e = e.displayName || e.name || "", t.displayName || (e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef");
        case 7:
            return "Fragment";
        case 5:
            return t;
        case 4:
            return "Portal";
        case 3:
            return "Root";
        case 6:
            return "Text";
        case 16:
            return ta(t);
        case 8:
            return t === mu ? "StrictMode" : "Mode";
        case 22:
            return "Offscreen";
        case 12:
            return "Profiler";
        case 21:
            return "Scope";
        case 13:
            return "Suspense";
        case 19:
            return "SuspenseList";
        case 25:
            return "TracingMarker";
        case 1:
        case 0:
        case 17:
        case 2:
        case 14:
        case 15:
            if (typeof t == "function") return t.displayName || t.name || null;
            if (typeof t == "string") return t
    }
    return null
}

function Ln(e) {
    switch (typeof e) {
        case "boolean":
        case "number":
        case "string":
        case "undefined":
            return e;
        case "object":
            return e;
        default:
            return ""
    }
}

function ep(e) { var t = e.type; return (e = e.nodeName) && e.toLowerCase() === "input" && (t === "checkbox" || t === "radio") }

function Ov(e) {
    var t = ep(e) ? "checked" : "value",
        n = Object.getOwnPropertyDescriptor(e.constructor.prototype, t),
        r = "" + e[t];
    if (!e.hasOwnProperty(t) && typeof n < "u" && typeof n.get == "function" && typeof n.set == "function") {
        var o = n.get,
            s = n.set;
        return Object.defineProperty(e, t, { configurable: !0, get: function() { return o.call(this) }, set: function(i) { r = "" + i, s.call(this, i) } }), Object.defineProperty(e, t, { enumerable: n.enumerable }), { getValue: function() { return r }, setValue: function(i) { r = "" + i }, stopTracking: function() { e._valueTracker = null, delete e[t] } }
    }
}

function Es(e) { e._valueTracker || (e._valueTracker = Ov(e)) }

function tp(e) {
    if (!e) return !1;
    var t = e._valueTracker;
    if (!t) return !0;
    var n = t.getValue(),
        r = "";
    return e && (r = ep(e) ? e.checked ? "true" : "false" : e.value), e = r, e !== n ? (t.setValue(e), !0) : !1
}

function oi(e) { if (e = e || (typeof document < "u" ? document : void 0), typeof e > "u") return null; try { return e.activeElement || e.body } catch { return e.body } }

function na(e, t) { var n = t.checked; return ce({}, t, { defaultChecked: void 0, defaultValue: void 0, value: void 0, checked: n ? ? e._wrapperState.initialChecked }) }

function Lc(e, t) {
    var n = t.defaultValue == null ? "" : t.defaultValue,
        r = t.checked != null ? t.checked : t.defaultChecked;
    n = Ln(t.value != null ? t.value : n), e._wrapperState = { initialChecked: r, initialValue: n, controlled: t.type === "checkbox" || t.type === "radio" ? t.checked != null : t.value != null }
}

function np(e, t) { t = t.checked, t != null && hu(e, "checked", t, !1) }

function ra(e, t) {
    np(e, t);
    var n = Ln(t.value),
        r = t.type;
    if (n != null) r === "number" ? (n === 0 && e.value === "" || e.value != n) && (e.value = "" + n) : e.value !== "" + n && (e.value = "" + n);
    else if (r === "submit" || r === "reset") { e.removeAttribute("value"); return }
    t.hasOwnProperty("value") ? oa(e, t.type, n) : t.hasOwnProperty("defaultValue") && oa(e, t.type, Ln(t.defaultValue)), t.checked == null && t.defaultChecked != null && (e.defaultChecked = !!t.defaultChecked)
}

function Mc(e, t, n) {
    if (t.hasOwnProperty("value") || t.hasOwnProperty("defaultValue")) {
        var r = t.type;
        if (!(r !== "submit" && r !== "reset" || t.value !== void 0 && t.value !== null)) return;
        t = "" + e._wrapperState.initialValue, n || t === e.value || (e.value = t), e.defaultValue = t
    }
    n = e.name, n !== "" && (e.name = ""), e.defaultChecked = !!e._wrapperState.initialChecked, n !== "" && (e.name = n)
}

function oa(e, t, n) {
    (t !== "number" || oi(e.ownerDocument) !== e) && (n == null ? e.defaultValue = "" + e._wrapperState.initialValue : e.defaultValue !== "" + n && (e.defaultValue = "" + n))
}
var bo = Array.isArray;

function _r(e, t, n, r) {
    if (e = e.options, t) { t = {}; for (var o = 0; o < n.length; o++) t["$" + n[o]] = !0; for (n = 0; n < e.length; n++) o = t.hasOwnProperty("$" + e[n].value), e[n].selected !== o && (e[n].selected = o), o && r && (e[n].defaultSelected = !0) } else {
        for (n = "" + Ln(n), t = null, o = 0; o < e.length; o++) {
            if (e[o].value === n) { e[o].selected = !0, r && (e[o].defaultSelected = !0); return }
            t !== null || e[o].disabled || (t = e[o])
        }
        t !== null && (t.selected = !0)
    }
}

function sa(e, t) { if (t.dangerouslySetInnerHTML != null) throw Error(j(91)); return ce({}, t, { value: void 0, defaultValue: void 0, children: "" + e._wrapperState.initialValue }) }

function Ic(e, t) {
    var n = t.value;
    if (n == null) {
        if (n = t.children, t = t.defaultValue, n != null) {
            if (t != null) throw Error(j(92));
            if (bo(n)) {
                if (1 < n.length) throw Error(j(93));
                n = n[0]
            }
            t = n
        }
        t == null && (t = ""), n = t
    }
    e._wrapperState = { initialValue: Ln(n) }
}

function rp(e, t) {
    var n = Ln(t.value),
        r = Ln(t.defaultValue);
    n != null && (n = "" + n, n !== e.value && (e.value = n), t.defaultValue == null && e.defaultValue !== n && (e.defaultValue = n)), r != null && (e.defaultValue = "" + r)
}

function Dc(e) {
    var t = e.textContent;
    t === e._wrapperState.initialValue && t !== "" && t !== null && (e.value = t)
}

function op(e) {
    switch (e) {
        case "svg":
            return "http://www.w3.org/2000/svg";
        case "math":
            return "http://www.w3.org/1998/Math/MathML";
        default:
            return "http://www.w3.org/1999/xhtml"
    }
}

function ia(e, t) { return e == null || e === "http://www.w3.org/1999/xhtml" ? op(t) : e === "http://www.w3.org/2000/svg" && t === "foreignObject" ? "http://www.w3.org/1999/xhtml" : e }
var Cs, sp = function(e) { return typeof MSApp < "u" && MSApp.execUnsafeLocalFunction ? function(t, n, r, o) { MSApp.execUnsafeLocalFunction(function() { return e(t, n, r, o) }) } : e }(function(e, t) {
    if (e.namespaceURI !== "http://www.w3.org/2000/svg" || "innerHTML" in e) e.innerHTML = t;
    else { for (Cs = Cs || document.createElement("div"), Cs.innerHTML = "<svg>" + t.valueOf().toString() + "</svg>", t = Cs.firstChild; e.firstChild;) e.removeChild(e.firstChild); for (; t.firstChild;) e.appendChild(t.firstChild) }
});

function $o(e, t) {
    if (t) { var n = e.firstChild; if (n && n === e.lastChild && n.nodeType === 3) { n.nodeValue = t; return } }
    e.textContent = t
}
var Ro = { animationIterationCount: !0, aspectRatio: !0, borderImageOutset: !0, borderImageSlice: !0, borderImageWidth: !0, boxFlex: !0, boxFlexGroup: !0, boxOrdinalGroup: !0, columnCount: !0, columns: !0, flex: !0, flexGrow: !0, flexPositive: !0, flexShrink: !0, flexNegative: !0, flexOrder: !0, gridArea: !0, gridRow: !0, gridRowEnd: !0, gridRowSpan: !0, gridRowStart: !0, gridColumn: !0, gridColumnEnd: !0, gridColumnSpan: !0, gridColumnStart: !0, fontWeight: !0, lineClamp: !0, lineHeight: !0, opacity: !0, order: !0, orphans: !0, tabSize: !0, widows: !0, zIndex: !0, zoom: !0, fillOpacity: !0, floodOpacity: !0, stopOpacity: !0, strokeDasharray: !0, strokeDashoffset: !0, strokeMiterlimit: !0, strokeOpacity: !0, strokeWidth: !0 },
    Lv = ["Webkit", "ms", "Moz", "O"];
Object.keys(Ro).forEach(function(e) { Lv.forEach(function(t) { t = t + e.charAt(0).toUpperCase() + e.substring(1), Ro[t] = Ro[e] }) });

function ip(e, t, n) { return t == null || typeof t == "boolean" || t === "" ? "" : n || typeof t != "number" || t === 0 || Ro.hasOwnProperty(e) && Ro[e] ? ("" + t).trim() : t + "px" }

function lp(e, t) {
    e = e.style;
    for (var n in t)
        if (t.hasOwnProperty(n)) {
            var r = n.indexOf("--") === 0,
                o = ip(n, t[n], r);
            n === "float" && (n = "cssFloat"), r ? e.setProperty(n, o) : e[n] = o
        }
}
var Mv = ce({ menuitem: !0 }, { area: !0, base: !0, br: !0, col: !0, embed: !0, hr: !0, img: !0, input: !0, keygen: !0, link: !0, meta: !0, param: !0, source: !0, track: !0, wbr: !0 });

function la(e, t) { if (t) { if (Mv[e] && (t.children != null || t.dangerouslySetInnerHTML != null)) throw Error(j(137, e)); if (t.dangerouslySetInnerHTML != null) { if (t.children != null) throw Error(j(60)); if (typeof t.dangerouslySetInnerHTML != "object" || !("__html" in t.dangerouslySetInnerHTML)) throw Error(j(61)) } if (t.style != null && typeof t.style != "object") throw Error(j(62)) } }

function aa(e, t) {
    if (e.indexOf("-") === -1) return typeof t.is == "string";
    switch (e) {
        case "annotation-xml":
        case "color-profile":
        case "font-face":
        case "font-face-src":
        case "font-face-uri":
        case "font-face-format":
        case "font-face-name":
        case "missing-glyph":
            return !1;
        default:
            return !0
    }
}
var ua = null;

function yu(e) { return e = e.target || e.srcElement || window, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e }
var ca = null,
    Ar = null,
    Or = null;

function zc(e) {
    if (e = ds(e)) {
        if (typeof ca != "function") throw Error(j(280));
        var t = e.stateNode;
        t && (t = Bi(t), ca(e.stateNode, e.type, t))
    }
}

function ap(e) { Ar ? Or ? Or.push(e) : Or = [e] : Ar = e }

function up() {
    if (Ar) {
        var e = Ar,
            t = Or;
        if (Or = Ar = null, zc(e), t)
            for (e = 0; e < t.length; e++) zc(t[e])
    }
}

function cp(e, t) { return e(t) }

function dp() {}
var Sl = !1;

function fp(e, t, n) {
    if (Sl) return e(t, n);
    Sl = !0;
    try { return cp(e, t, n) } finally { Sl = !1, (Ar !== null || Or !== null) && (dp(), up()) }
}

function Bo(e, t) {
    var n = e.stateNode;
    if (n === null) return null;
    var r = Bi(n);
    if (r === null) return null;
    n = r[t];
    e: switch (t) {
        case "onClick":
        case "onClickCapture":
        case "onDoubleClick":
        case "onDoubleClickCapture":
        case "onMouseDown":
        case "onMouseDownCapture":
        case "onMouseMove":
        case "onMouseMoveCapture":
        case "onMouseUp":
        case "onMouseUpCapture":
        case "onMouseEnter":
            (r = !r.disabled) || (e = e.type, r = !(e === "button" || e === "input" || e === "select" || e === "textarea")), e = !r;
            break e;
        default:
            e = !1
    }
    if (e) return null;
    if (n && typeof n != "function") throw Error(j(231, t, typeof n));
    return n
}
var da = !1;
if (Kt) try {
    var go = {};
    Object.defineProperty(go, "passive", { get: function() { da = !0 } }), window.addEventListener("test", go, go), window.removeEventListener("test", go, go)
} catch { da = !1 }

function Iv(e, t, n, r, o, s, i, l, a) { var u = Array.prototype.slice.call(arguments, 3); try { t.apply(n, u) } catch (c) { this.onError(c) } }
var jo = !1,
    si = null,
    ii = !1,
    fa = null,
    Dv = { onError: function(e) { jo = !0, si = e } };

function zv(e, t, n, r, o, s, i, l, a) { jo = !1, si = null, Iv.apply(Dv, arguments) }

function Fv(e, t, n, r, o, s, i, l, a) {
    if (zv.apply(this, arguments), jo) {
        if (jo) {
            var u = si;
            jo = !1, si = null
        } else throw Error(j(198));
        ii || (ii = !0, fa = u)
    }
}

function dr(e) {
    var t = e,
        n = e;
    if (e.alternate)
        for (; t.return;) t = t.return;
    else {
        e = t;
        do t = e, t.flags & 4098 && (n = t.return), e = t.return; while (e)
    }
    return t.tag === 3 ? n : null
}

function pp(e) { if (e.tag === 13) { var t = e.memoizedState; if (t === null && (e = e.alternate, e !== null && (t = e.memoizedState)), t !== null) return t.dehydrated } return null }

function Fc(e) { if (dr(e) !== e) throw Error(j(188)) }

function $v(e) {
    var t = e.alternate;
    if (!t) { if (t = dr(e), t === null) throw Error(j(188)); return t !== e ? null : e }
    for (var n = e, r = t;;) {
        var o = n.return;
        if (o === null) break;
        var s = o.alternate;
        if (s === null) { if (r = o.return, r !== null) { n = r; continue } break }
        if (o.child === s.child) {
            for (s = o.child; s;) {
                if (s === n) return Fc(o), e;
                if (s === r) return Fc(o), t;
                s = s.sibling
            }
            throw Error(j(188))
        }
        if (n.return !== r.return) n = o, r = s;
        else {
            for (var i = !1, l = o.child; l;) {
                if (l === n) { i = !0, n = o, r = s; break }
                if (l === r) { i = !0, r = o, n = s; break }
                l = l.sibling
            }
            if (!i) {
                for (l = s.child; l;) {
                    if (l === n) { i = !0, n = s, r = o; break }
                    if (l === r) { i = !0, r = s, n = o; break }
                    l = l.sibling
                }
                if (!i) throw Error(j(189))
            }
        }
        if (n.alternate !== r) throw Error(j(190))
    }
    if (n.tag !== 3) throw Error(j(188));
    return n.stateNode.current === n ? e : t
}

function hp(e) { return e = $v(e), e !== null ? mp(e) : null }

function mp(e) {
    if (e.tag === 5 || e.tag === 6) return e;
    for (e = e.child; e !== null;) {
        var t = mp(e);
        if (t !== null) return t;
        e = e.sibling
    }
    return null
}
var gp = Je.unstable_scheduleCallback,
    $c = Je.unstable_cancelCallback,
    Bv = Je.unstable_shouldYield,
    Uv = Je.unstable_requestPaint,
    pe = Je.unstable_now,
    Vv = Je.unstable_getCurrentPriorityLevel,
    wu = Je.unstable_ImmediatePriority,
    vp = Je.unstable_UserBlockingPriority,
    li = Je.unstable_NormalPriority,
    Wv = Je.unstable_LowPriority,
    yp = Je.unstable_IdlePriority,
    Di = null,
    Lt = null;

function Hv(e) { if (Lt && typeof Lt.onCommitFiberRoot == "function") try { Lt.onCommitFiberRoot(Di, e, void 0, (e.current.flags & 128) === 128) } catch {} }
var xt = Math.clz32 ? Math.clz32 : Gv,
    Qv = Math.log,
    Kv = Math.LN2;

function Gv(e) { return e >>>= 0, e === 0 ? 32 : 31 - (Qv(e) / Kv | 0) | 0 }
var ks = 64,
    Ps = 4194304;

function To(e) {
    switch (e & -e) {
        case 1:
            return 1;
        case 2:
            return 2;
        case 4:
            return 4;
        case 8:
            return 8;
        case 16:
            return 16;
        case 32:
            return 32;
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
            return e & 4194240;
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
        case 67108864:
            return e & 130023424;
        case 134217728:
            return 134217728;
        case 268435456:
            return 268435456;
        case 536870912:
            return 536870912;
        case 1073741824:
            return 1073741824;
        default:
            return e
    }
}

function ai(e, t) {
    var n = e.pendingLanes;
    if (n === 0) return 0;
    var r = 0,
        o = e.suspendedLanes,
        s = e.pingedLanes,
        i = n & 268435455;
    if (i !== 0) {
        var l = i & ~o;
        l !== 0 ? r = To(l) : (s &= i, s !== 0 && (r = To(s)))
    } else i = n & ~o, i !== 0 ? r = To(i) : s !== 0 && (r = To(s));
    if (r === 0) return 0;
    if (t !== 0 && t !== r && !(t & o) && (o = r & -r, s = t & -t, o >= s || o === 16 && (s & 4194240) !== 0)) return t;
    if (r & 4 && (r |= n & 16), t = e.entangledLanes, t !== 0)
        for (e = e.entanglements, t &= r; 0 < t;) n = 31 - xt(t), o = 1 << n, r |= e[n], t &= ~o;
    return r
}

function Yv(e, t) {
    switch (e) {
        case 1:
        case 2:
        case 4:
            return t + 250;
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
            return t + 5e3;
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
        case 67108864:
            return -1;
        case 134217728:
        case 268435456:
        case 536870912:
        case 1073741824:
            return -1;
        default:
            return -1
    }
}

function Xv(e, t) {
    for (var n = e.suspendedLanes, r = e.pingedLanes, o = e.expirationTimes, s = e.pendingLanes; 0 < s;) {
        var i = 31 - xt(s),
            l = 1 << i,
            a = o[i];
        a === -1 ? (!(l & n) || l & r) && (o[i] = Yv(l, t)) : a <= t && (e.expiredLanes |= l), s &= ~l
    }
}

function pa(e) { return e = e.pendingLanes & -1073741825, e !== 0 ? e : e & 1073741824 ? 1073741824 : 0 }

function wp() { var e = ks; return ks <<= 1, !(ks & 4194240) && (ks = 64), e }

function El(e) { for (var t = [], n = 0; 31 > n; n++) t.push(e); return t }

function us(e, t, n) { e.pendingLanes |= t, t !== 536870912 && (e.suspendedLanes = 0, e.pingedLanes = 0), e = e.eventTimes, t = 31 - xt(t), e[t] = n }

function qv(e, t) {
    var n = e.pendingLanes & ~t;
    e.pendingLanes = t, e.suspendedLanes = 0, e.pingedLanes = 0, e.expiredLanes &= t, e.mutableReadLanes &= t, e.entangledLanes &= t, t = e.entanglements;
    var r = e.eventTimes;
    for (e = e.expirationTimes; 0 < n;) {
        var o = 31 - xt(n),
            s = 1 << o;
        t[o] = 0, r[o] = -1, e[o] = -1, n &= ~s
    }
}

function xu(e, t) {
    var n = e.entangledLanes |= t;
    for (e = e.entanglements; n;) {
        var r = 31 - xt(n),
            o = 1 << r;
        o & t | e[r] & t && (e[r] |= t), n &= ~o
    }
}
var ee = 0;

function xp(e) { return e &= -e, 1 < e ? 4 < e ? e & 268435455 ? 16 : 536870912 : 4 : 1 }
var Sp, Su, Ep, Cp, kp, ha = !1,
    bs = [],
    bn = null,
    Tn = null,
    Nn = null,
    Uo = new Map,
    Vo = new Map,
    mn = [],
    Zv = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");

function Bc(e, t) {
    switch (e) {
        case "focusin":
        case "focusout":
            bn = null;
            break;
        case "dragenter":
        case "dragleave":
            Tn = null;
            break;
        case "mouseover":
        case "mouseout":
            Nn = null;
            break;
        case "pointerover":
        case "pointerout":
            Uo.delete(t.pointerId);
            break;
        case "gotpointercapture":
        case "lostpointercapture":
            Vo.delete(t.pointerId)
    }
}

function vo(e, t, n, r, o, s) { return e === null || e.nativeEvent !== s ? (e = { blockedOn: t, domEventName: n, eventSystemFlags: r, nativeEvent: s, targetContainers: [o] }, t !== null && (t = ds(t), t !== null && Su(t)), e) : (e.eventSystemFlags |= r, t = e.targetContainers, o !== null && t.indexOf(o) === -1 && t.push(o), e) }

function Jv(e, t, n, r, o) {
    switch (t) {
        case "focusin":
            return bn = vo(bn, e, t, n, r, o), !0;
        case "dragenter":
            return Tn = vo(Tn, e, t, n, r, o), !0;
        case "mouseover":
            return Nn = vo(Nn, e, t, n, r, o), !0;
        case "pointerover":
            var s = o.pointerId;
            return Uo.set(s, vo(Uo.get(s) || null, e, t, n, r, o)), !0;
        case "gotpointercapture":
            return s = o.pointerId, Vo.set(s, vo(Vo.get(s) || null, e, t, n, r, o)), !0
    }
    return !1
}

function Pp(e) {
    var t = Kn(e.target);
    if (t !== null) { var n = dr(t); if (n !== null) { if (t = n.tag, t === 13) { if (t = pp(n), t !== null) { e.blockedOn = t, kp(e.priority, function() { Ep(n) }); return } } else if (t === 3 && n.stateNode.current.memoizedState.isDehydrated) { e.blockedOn = n.tag === 3 ? n.stateNode.containerInfo : null; return } } }
    e.blockedOn = null
}

function Hs(e) {
    if (e.blockedOn !== null) return !1;
    for (var t = e.targetContainers; 0 < t.length;) {
        var n = ma(e.domEventName, e.eventSystemFlags, t[0], e.nativeEvent);
        if (n === null) {
            n = e.nativeEvent;
            var r = new n.constructor(n.type, n);
            ua = r, n.target.dispatchEvent(r), ua = null
        } else return t = ds(n), t !== null && Su(t), e.blockedOn = n, !1;
        t.shift()
    }
    return !0
}

function Uc(e, t, n) { Hs(e) && n.delete(t) }

function ey() { ha = !1, bn !== null && Hs(bn) && (bn = null), Tn !== null && Hs(Tn) && (Tn = null), Nn !== null && Hs(Nn) && (Nn = null), Uo.forEach(Uc), Vo.forEach(Uc) }

function yo(e, t) { e.blockedOn === t && (e.blockedOn = null, ha || (ha = !0, Je.unstable_scheduleCallback(Je.unstable_NormalPriority, ey))) }

function Wo(e) {
    function t(o) { return yo(o, e) }
    if (0 < bs.length) {
        yo(bs[0], e);
        for (var n = 1; n < bs.length; n++) {
            var r = bs[n];
            r.blockedOn === e && (r.blockedOn = null)
        }
    }
    for (bn !== null && yo(bn, e), Tn !== null && yo(Tn, e), Nn !== null && yo(Nn, e), Uo.forEach(t), Vo.forEach(t), n = 0; n < mn.length; n++) r = mn[n], r.blockedOn === e && (r.blockedOn = null);
    for (; 0 < mn.length && (n = mn[0], n.blockedOn === null);) Pp(n), n.blockedOn === null && mn.shift()
}
var Lr = en.ReactCurrentBatchConfig,
    ui = !0;

function ty(e, t, n, r) {
    var o = ee,
        s = Lr.transition;
    Lr.transition = null;
    try { ee = 1, Eu(e, t, n, r) } finally { ee = o, Lr.transition = s }
}

function ny(e, t, n, r) {
    var o = ee,
        s = Lr.transition;
    Lr.transition = null;
    try { ee = 4, Eu(e, t, n, r) } finally { ee = o, Lr.transition = s }
}

function Eu(e, t, n, r) {
    if (ui) {
        var o = ma(e, t, n, r);
        if (o === null) Al(e, t, r, ci, n), Bc(e, r);
        else if (Jv(o, e, t, n, r)) r.stopPropagation();
        else if (Bc(e, r), t & 4 && -1 < Zv.indexOf(e)) {
            for (; o !== null;) {
                var s = ds(o);
                if (s !== null && Sp(s), s = ma(e, t, n, r), s === null && Al(e, t, r, ci, n), s === o) break;
                o = s
            }
            o !== null && r.stopPropagation()
        } else Al(e, t, r, null, n)
    }
}
var ci = null;

function ma(e, t, n, r) {
    if (ci = null, e = yu(r), e = Kn(e), e !== null)
        if (t = dr(e), t === null) e = null;
        else if (n = t.tag, n === 13) {
        if (e = pp(t), e !== null) return e;
        e = null
    } else if (n === 3) {
        if (t.stateNode.current.memoizedState.isDehydrated) return t.tag === 3 ? t.stateNode.containerInfo : null;
        e = null
    } else t !== e && (e = null);
    return ci = e, null
}

function bp(e) {
    switch (e) {
        case "cancel":
        case "click":
        case "close":
        case "contextmenu":
        case "copy":
        case "cut":
        case "auxclick":
        case "dblclick":
        case "dragend":
        case "dragstart":
        case "drop":
        case "focusin":
        case "focusout":
        case "input":
        case "invalid":
        case "keydown":
        case "keypress":
        case "keyup":
        case "mousedown":
        case "mouseup":
        case "paste":
        case "pause":
        case "play":
        case "pointercancel":
        case "pointerdown":
        case "pointerup":
        case "ratechange":
        case "reset":
        case "resize":
        case "seeked":
        case "submit":
        case "touchcancel":
        case "touchend":
        case "touchstart":
        case "volumechange":
        case "change":
        case "selectionchange":
        case "textInput":
        case "compositionstart":
        case "compositionend":
        case "compositionupdate":
        case "beforeblur":
        case "afterblur":
        case "beforeinput":
        case "blur":
        case "fullscreenchange":
        case "focus":
        case "hashchange":
        case "popstate":
        case "select":
        case "selectstart":
            return 1;
        case "drag":
        case "dragenter":
        case "dragexit":
        case "dragleave":
        case "dragover":
        case "mousemove":
        case "mouseout":
        case "mouseover":
        case "pointermove":
        case "pointerout":
        case "pointerover":
        case "scroll":
        case "toggle":
        case "touchmove":
        case "wheel":
        case "mouseenter":
        case "mouseleave":
        case "pointerenter":
        case "pointerleave":
            return 4;
        case "message":
            switch (Vv()) {
                case wu:
                    return 1;
                case vp:
                    return 4;
                case li:
                case Wv:
                    return 16;
                case yp:
                    return 536870912;
                default:
                    return 16
            }
        default:
            return 16
    }
}
var Cn = null,
    Cu = null,
    Qs = null;

function Tp() {
    if (Qs) return Qs;
    var e, t = Cu,
        n = t.length,
        r, o = "value" in Cn ? Cn.value : Cn.textContent,
        s = o.length;
    for (e = 0; e < n && t[e] === o[e]; e++);
    var i = n - e;
    for (r = 1; r <= i && t[n - r] === o[s - r]; r++);
    return Qs = o.slice(e, 1 < r ? 1 - r : void 0)
}

function Ks(e) { var t = e.keyCode; return "charCode" in e ? (e = e.charCode, e === 0 && t === 13 && (e = 13)) : e = t, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0 }

function Ts() { return !0 }

function Vc() { return !1 }

function tt(e) {
    function t(n, r, o, s, i) { this._reactName = n, this._targetInst = o, this.type = r, this.nativeEvent = s, this.target = i, this.currentTarget = null; for (var l in e) e.hasOwnProperty(l) && (n = e[l], this[l] = n ? n(s) : s[l]); return this.isDefaultPrevented = (s.defaultPrevented != null ? s.defaultPrevented : s.returnValue === !1) ? Ts : Vc, this.isPropagationStopped = Vc, this }
    return ce(t.prototype, {
        preventDefault: function() {
            this.defaultPrevented = !0;
            var n = this.nativeEvent;
            n && (n.preventDefault ? n.preventDefault() : typeof n.returnValue != "unknown" && (n.returnValue = !1), this.isDefaultPrevented = Ts)
        },
        stopPropagation: function() {
            var n = this.nativeEvent;
            n && (n.stopPropagation ? n.stopPropagation() : typeof n.cancelBubble != "unknown" && (n.cancelBubble = !0), this.isPropagationStopped = Ts)
        },
        persist: function() {},
        isPersistent: Ts
    }), t
}
var lo = { eventPhase: 0, bubbles: 0, cancelable: 0, timeStamp: function(e) { return e.timeStamp || Date.now() }, defaultPrevented: 0, isTrusted: 0 },
    ku = tt(lo),
    cs = ce({}, lo, { view: 0, detail: 0 }),
    ry = tt(cs),
    Cl, kl, wo, zi = ce({}, cs, { screenX: 0, screenY: 0, clientX: 0, clientY: 0, pageX: 0, pageY: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, getModifierState: Pu, button: 0, buttons: 0, relatedTarget: function(e) { return e.relatedTarget === void 0 ? e.fromElement === e.srcElement ? e.toElement : e.fromElement : e.relatedTarget }, movementX: function(e) { return "movementX" in e ? e.movementX : (e !== wo && (wo && e.type === "mousemove" ? (Cl = e.screenX - wo.screenX, kl = e.screenY - wo.screenY) : kl = Cl = 0, wo = e), Cl) }, movementY: function(e) { return "movementY" in e ? e.movementY : kl } }),
    Wc = tt(zi),
    oy = ce({}, zi, { dataTransfer: 0 }),
    sy = tt(oy),
    iy = ce({}, cs, { relatedTarget: 0 }),
    Pl = tt(iy),
    ly = ce({}, lo, { animationName: 0, elapsedTime: 0, pseudoElement: 0 }),
    ay = tt(ly),
    uy = ce({}, lo, { clipboardData: function(e) { return "clipboardData" in e ? e.clipboardData : window.clipboardData } }),
    cy = tt(uy),
    dy = ce({}, lo, { data: 0 }),
    Hc = tt(dy),
    fy = { Esc: "Escape", Spacebar: " ", Left: "ArrowLeft", Up: "ArrowUp", Right: "ArrowRight", Down: "ArrowDown", Del: "Delete", Win: "OS", Menu: "ContextMenu", Apps: "ContextMenu", Scroll: "ScrollLock", MozPrintableKey: "Unidentified" },
    py = { 8: "Backspace", 9: "Tab", 12: "Clear", 13: "Enter", 16: "Shift", 17: "Control", 18: "Alt", 19: "Pause", 20: "CapsLock", 27: "Escape", 32: " ", 33: "PageUp", 34: "PageDown", 35: "End", 36: "Home", 37: "ArrowLeft", 38: "ArrowUp", 39: "ArrowRight", 40: "ArrowDown", 45: "Insert", 46: "Delete", 112: "F1", 113: "F2", 114: "F3", 115: "F4", 116: "F5", 117: "F6", 118: "F7", 119: "F8", 120: "F9", 121: "F10", 122: "F11", 123: "F12", 144: "NumLock", 145: "ScrollLock", 224: "Meta" },
    hy = { Alt: "altKey", Control: "ctrlKey", Meta: "metaKey", Shift: "shiftKey" };

function my(e) { var t = this.nativeEvent; return t.getModifierState ? t.getModifierState(e) : (e = hy[e]) ? !!t[e] : !1 }

function Pu() { return my }
var gy = ce({}, cs, { key: function(e) { if (e.key) { var t = fy[e.key] || e.key; if (t !== "Unidentified") return t } return e.type === "keypress" ? (e = Ks(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? py[e.keyCode] || "Unidentified" : "" }, code: 0, location: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, repeat: 0, locale: 0, getModifierState: Pu, charCode: function(e) { return e.type === "keypress" ? Ks(e) : 0 }, keyCode: function(e) { return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0 }, which: function(e) { return e.type === "keypress" ? Ks(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0 } }),
    vy = tt(gy),
    yy = ce({}, zi, { pointerId: 0, width: 0, height: 0, pressure: 0, tangentialPressure: 0, tiltX: 0, tiltY: 0, twist: 0, pointerType: 0, isPrimary: 0 }),
    Qc = tt(yy),
    wy = ce({}, cs, { touches: 0, targetTouches: 0, changedTouches: 0, altKey: 0, metaKey: 0, ctrlKey: 0, shiftKey: 0, getModifierState: Pu }),
    xy = tt(wy),
    Sy = ce({}, lo, { propertyName: 0, elapsedTime: 0, pseudoElement: 0 }),
    Ey = tt(Sy),
    Cy = ce({}, zi, { deltaX: function(e) { return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0 }, deltaY: function(e) { return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0 }, deltaZ: 0, deltaMode: 0 }),
    ky = tt(Cy),
    Py = [9, 13, 27, 32],
    bu = Kt && "CompositionEvent" in window,
    _o = null;
Kt && "documentMode" in document && (_o = document.documentMode);
var by = Kt && "TextEvent" in window && !_o,
    Np = Kt && (!bu || _o && 8 < _o && 11 >= _o),
    Kc = " ",
    Gc = !1;

function Rp(e, t) {
    switch (e) {
        case "keyup":
            return Py.indexOf(t.keyCode) !== -1;
        case "keydown":
            return t.keyCode !== 229;
        case "keypress":
        case "mousedown":
        case "focusout":
            return !0;
        default:
            return !1
    }
}

function jp(e) { return e = e.detail, typeof e == "object" && "data" in e ? e.data : null }
var Sr = !1;

function Ty(e, t) {
    switch (e) {
        case "compositionend":
            return jp(t);
        case "keypress":
            return t.which !== 32 ? null : (Gc = !0, Kc);
        case "textInput":
            return e = t.data, e === Kc && Gc ? null : e;
        default:
            return null
    }
}

function Ny(e, t) {
    if (Sr) return e === "compositionend" || !bu && Rp(e, t) ? (e = Tp(), Qs = Cu = Cn = null, Sr = !1, e) : null;
    switch (e) {
        case "paste":
            return null;
        case "keypress":
            if (!(t.ctrlKey || t.altKey || t.metaKey) || t.ctrlKey && t.altKey) { if (t.char && 1 < t.char.length) return t.char; if (t.which) return String.fromCharCode(t.which) }
            return null;
        case "compositionend":
            return Np && t.locale !== "ko" ? null : t.data;
        default:
            return null
    }
}
var Ry = { color: !0, date: !0, datetime: !0, "datetime-local": !0, email: !0, month: !0, number: !0, password: !0, range: !0, search: !0, tel: !0, text: !0, time: !0, url: !0, week: !0 };

function Yc(e) { var t = e && e.nodeName && e.nodeName.toLowerCase(); return t === "input" ? !!Ry[e.type] : t === "textarea" }

function _p(e, t, n, r) { ap(r), t = di(t, "onChange"), 0 < t.length && (n = new ku("onChange", "change", null, n, r), e.push({ event: n, listeners: t })) }
var Ao = null,
    Ho = null;

function jy(e) { Up(e, 0) }

function Fi(e) { var t = kr(e); if (tp(t)) return e }

function _y(e, t) { if (e === "change") return t }
var Ap = !1;
if (Kt) {
    var bl;
    if (Kt) {
        var Tl = "oninput" in document;
        if (!Tl) {
            var Xc = document.createElement("div");
            Xc.setAttribute("oninput", "return;"), Tl = typeof Xc.oninput == "function"
        }
        bl = Tl
    } else bl = !1;
    Ap = bl && (!document.documentMode || 9 < document.documentMode)
}

function qc() { Ao && (Ao.detachEvent("onpropertychange", Op), Ho = Ao = null) }

function Op(e) {
    if (e.propertyName === "value" && Fi(Ho)) {
        var t = [];
        _p(t, Ho, e, yu(e)), fp(jy, t)
    }
}

function Ay(e, t, n) { e === "focusin" ? (qc(), Ao = t, Ho = n, Ao.attachEvent("onpropertychange", Op)) : e === "focusout" && qc() }

function Oy(e) { if (e === "selectionchange" || e === "keyup" || e === "keydown") return Fi(Ho) }

function Ly(e, t) { if (e === "click") return Fi(t) }

function My(e, t) { if (e === "input" || e === "change") return Fi(t) }

function Iy(e, t) { return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t }
var Et = typeof Object.is == "function" ? Object.is : Iy;

function Qo(e, t) {
    if (Et(e, t)) return !0;
    if (typeof e != "object" || e === null || typeof t != "object" || t === null) return !1;
    var n = Object.keys(e),
        r = Object.keys(t);
    if (n.length !== r.length) return !1;
    for (r = 0; r < n.length; r++) { var o = n[r]; if (!ql.call(t, o) || !Et(e[o], t[o])) return !1 }
    return !0
}

function Zc(e) { for (; e && e.firstChild;) e = e.firstChild; return e }

function Jc(e, t) {
    var n = Zc(e);
    e = 0;
    for (var r; n;) {
        if (n.nodeType === 3) {
            if (r = e + n.textContent.length, e <= t && r >= t) return { node: n, offset: t - e };
            e = r
        }
        e: {
            for (; n;) {
                if (n.nextSibling) { n = n.nextSibling; break e }
                n = n.parentNode
            }
            n = void 0
        }
        n = Zc(n)
    }
}

function Lp(e, t) { return e && t ? e === t ? !0 : e && e.nodeType === 3 ? !1 : t && t.nodeType === 3 ? Lp(e, t.parentNode) : "contains" in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : !1 : !1 }

function Mp() {
    for (var e = window, t = oi(); t instanceof e.HTMLIFrameElement;) {
        try { var n = typeof t.contentWindow.location.href == "string" } catch { n = !1 }
        if (n) e = t.contentWindow;
        else break;
        t = oi(e.document)
    }
    return t
}

function Tu(e) { var t = e && e.nodeName && e.nodeName.toLowerCase(); return t && (t === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || t === "textarea" || e.contentEditable === "true") }

function Dy(e) {
    var t = Mp(),
        n = e.focusedElem,
        r = e.selectionRange;
    if (t !== n && n && n.ownerDocument && Lp(n.ownerDocument.documentElement, n)) {
        if (r !== null && Tu(n)) {
            if (t = r.start, e = r.end, e === void 0 && (e = t), "selectionStart" in n) n.selectionStart = t, n.selectionEnd = Math.min(e, n.value.length);
            else if (e = (t = n.ownerDocument || document) && t.defaultView || window, e.getSelection) {
                e = e.getSelection();
                var o = n.textContent.length,
                    s = Math.min(r.start, o);
                r = r.end === void 0 ? s : Math.min(r.end, o), !e.extend && s > r && (o = r, r = s, s = o), o = Jc(n, s);
                var i = Jc(n, r);
                o && i && (e.rangeCount !== 1 || e.anchorNode !== o.node || e.anchorOffset !== o.offset || e.focusNode !== i.node || e.focusOffset !== i.offset) && (t = t.createRange(), t.setStart(o.node, o.offset), e.removeAllRanges(), s > r ? (e.addRange(t), e.extend(i.node, i.offset)) : (t.setEnd(i.node, i.offset), e.addRange(t)))
            }
        }
        for (t = [], e = n; e = e.parentNode;) e.nodeType === 1 && t.push({ element: e, left: e.scrollLeft, top: e.scrollTop });
        for (typeof n.focus == "function" && n.focus(), n = 0; n < t.length; n++) e = t[n], e.element.scrollLeft = e.left, e.element.scrollTop = e.top
    }
}
var zy = Kt && "documentMode" in document && 11 >= document.documentMode,
    Er = null,
    ga = null,
    Oo = null,
    va = !1;

function ed(e, t, n) {
    var r = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
    va || Er == null || Er !== oi(r) || (r = Er, "selectionStart" in r && Tu(r) ? r = { start: r.selectionStart, end: r.selectionEnd } : (r = (r.ownerDocument && r.ownerDocument.defaultView || window).getSelection(), r = { anchorNode: r.anchorNode, anchorOffset: r.anchorOffset, focusNode: r.focusNode, focusOffset: r.focusOffset }), Oo && Qo(Oo, r) || (Oo = r, r = di(ga, "onSelect"), 0 < r.length && (t = new ku("onSelect", "select", null, t, n), e.push({ event: t, listeners: r }), t.target = Er)))
}

function Ns(e, t) { var n = {}; return n[e.toLowerCase()] = t.toLowerCase(), n["Webkit" + e] = "webkit" + t, n["Moz" + e] = "moz" + t, n }
var Cr = { animationend: Ns("Animation", "AnimationEnd"), animationiteration: Ns("Animation", "AnimationIteration"), animationstart: Ns("Animation", "AnimationStart"), transitionend: Ns("Transition", "TransitionEnd") },
    Nl = {},
    Ip = {};
Kt && (Ip = document.createElement("div").style, "AnimationEvent" in window || (delete Cr.animationend.animation, delete Cr.animationiteration.animation, delete Cr.animationstart.animation), "TransitionEvent" in window || delete Cr.transitionend.transition);

function $i(e) {
    if (Nl[e]) return Nl[e];
    if (!Cr[e]) return e;
    var t = Cr[e],
        n;
    for (n in t)
        if (t.hasOwnProperty(n) && n in Ip) return Nl[e] = t[n];
    return e
}
var Dp = $i("animationend"),
    zp = $i("animationiteration"),
    Fp = $i("animationstart"),
    $p = $i("transitionend"),
    Bp = new Map,
    td = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");

function zn(e, t) { Bp.set(e, t), cr(t, [e]) }
for (var Rl = 0; Rl < td.length; Rl++) {
    var jl = td[Rl],
        Fy = jl.toLowerCase(),
        $y = jl[0].toUpperCase() + jl.slice(1);
    zn(Fy, "on" + $y)
}
zn(Dp, "onAnimationEnd");
zn(zp, "onAnimationIteration");
zn(Fp, "onAnimationStart");
zn("dblclick", "onDoubleClick");
zn("focusin", "onFocus");
zn("focusout", "onBlur");
zn($p, "onTransitionEnd");
Xr("onMouseEnter", ["mouseout", "mouseover"]);
Xr("onMouseLeave", ["mouseout", "mouseover"]);
Xr("onPointerEnter", ["pointerout", "pointerover"]);
Xr("onPointerLeave", ["pointerout", "pointerover"]);
cr("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
cr("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
cr("onBeforeInput", ["compositionend", "keypress", "textInput", "paste"]);
cr("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
cr("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
cr("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
var No = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),
    By = new Set("cancel close invalid load scroll toggle".split(" ").concat(No));

function nd(e, t, n) {
    var r = e.type || "unknown-event";
    e.currentTarget = n, Fv(r, t, void 0, e), e.currentTarget = null
}

function Up(e, t) {
    t = (t & 4) !== 0;
    for (var n = 0; n < e.length; n++) {
        var r = e[n],
            o = r.event;
        r = r.listeners;
        e: {
            var s = void 0;
            if (t)
                for (var i = r.length - 1; 0 <= i; i--) {
                    var l = r[i],
                        a = l.instance,
                        u = l.currentTarget;
                    if (l = l.listener, a !== s && o.isPropagationStopped()) break e;
                    nd(o, l, u), s = a
                } else
                    for (i = 0; i < r.length; i++) {
                        if (l = r[i], a = l.instance, u = l.currentTarget, l = l.listener, a !== s && o.isPropagationStopped()) break e;
                        nd(o, l, u), s = a
                    }
        }
    }
    if (ii) throw e = fa, ii = !1, fa = null, e
}

function oe(e, t) {
    var n = t[Ea];
    n === void 0 && (n = t[Ea] = new Set);
    var r = e + "__bubble";
    n.has(r) || (Vp(t, e, 2, !1), n.add(r))
}

function _l(e, t, n) {
    var r = 0;
    t && (r |= 4), Vp(n, e, r, t)
}
var Rs = "_reactListening" + Math.random().toString(36).slice(2);

function Ko(e) {
    if (!e[Rs]) {
        e[Rs] = !0, Xf.forEach(function(n) { n !== "selectionchange" && (By.has(n) || _l(n, !1, e), _l(n, !0, e)) });
        var t = e.nodeType === 9 ? e : e.ownerDocument;
        t === null || t[Rs] || (t[Rs] = !0, _l("selectionchange", !1, t))
    }
}

function Vp(e, t, n, r) {
    switch (bp(t)) {
        case 1:
            var o = ty;
            break;
        case 4:
            o = ny;
            break;
        default:
            o = Eu
    }
    n = o.bind(null, t, n, e), o = void 0, !da || t !== "touchstart" && t !== "touchmove" && t !== "wheel" || (o = !0), r ? o !== void 0 ? e.addEventListener(t, n, { capture: !0, passive: o }) : e.addEventListener(t, n, !0) : o !== void 0 ? e.addEventListener(t, n, { passive: o }) : e.addEventListener(t, n, !1)
}

function Al(e, t, n, r, o) {
    var s = r;
    if (!(t & 1) && !(t & 2) && r !== null) e: for (;;) {
        if (r === null) return;
        var i = r.tag;
        if (i === 3 || i === 4) {
            var l = r.stateNode.containerInfo;
            if (l === o || l.nodeType === 8 && l.parentNode === o) break;
            if (i === 4)
                for (i = r.return; i !== null;) {
                    var a = i.tag;
                    if ((a === 3 || a === 4) && (a = i.stateNode.containerInfo, a === o || a.nodeType === 8 && a.parentNode === o)) return;
                    i = i.return
                }
            for (; l !== null;) {
                if (i = Kn(l), i === null) return;
                if (a = i.tag, a === 5 || a === 6) { r = s = i; continue e }
                l = l.parentNode
            }
        }
        r = r.return
    }
    fp(function() {
        var u = s,
            c = yu(n),
            f = [];
        e: {
            var m = Bp.get(e);
            if (m !== void 0) {
                var d = ku,
                    S = e;
                switch (e) {
                    case "keypress":
                        if (Ks(n) === 0) break e;
                    case "keydown":
                    case "keyup":
                        d = vy;
                        break;
                    case "focusin":
                        S = "focus", d = Pl;
                        break;
                    case "focusout":
                        S = "blur", d = Pl;
                        break;
                    case "beforeblur":
                    case "afterblur":
                        d = Pl;
                        break;
                    case "click":
                        if (n.button === 2) break e;
                    case "auxclick":
                    case "dblclick":
                    case "mousedown":
                    case "mousemove":
                    case "mouseup":
                    case "mouseout":
                    case "mouseover":
                    case "contextmenu":
                        d = Wc;
                        break;
                    case "drag":
                    case "dragend":
                    case "dragenter":
                    case "dragexit":
                    case "dragleave":
                    case "dragover":
                    case "dragstart":
                    case "drop":
                        d = sy;
                        break;
                    case "touchcancel":
                    case "touchend":
                    case "touchmove":
                    case "touchstart":
                        d = xy;
                        break;
                    case Dp:
                    case zp:
                    case Fp:
                        d = ay;
                        break;
                    case $p:
                        d = Ey;
                        break;
                    case "scroll":
                        d = ry;
                        break;
                    case "wheel":
                        d = ky;
                        break;
                    case "copy":
                    case "cut":
                    case "paste":
                        d = cy;
                        break;
                    case "gotpointercapture":
                    case "lostpointercapture":
                    case "pointercancel":
                    case "pointerdown":
                    case "pointermove":
                    case "pointerout":
                    case "pointerover":
                    case "pointerup":
                        d = Qc
                }
                var y = (t & 4) !== 0,
                    x = !y && e === "scroll",
                    h = y ? m !== null ? m + "Capture" : null : m;
                y = [];
                for (var p = u, g; p !== null;) {
                    g = p;
                    var E = g.stateNode;
                    if (g.tag === 5 && E !== null && (g = E, h !== null && (E = Bo(p, h), E != null && y.push(Go(p, E, g)))), x) break;
                    p = p.return
                }
                0 < y.length && (m = new d(m, S, null, n, c), f.push({ event: m, listeners: y }))
            }
        }
        if (!(t & 7)) {
            e: {
                if (m = e === "mouseover" || e === "pointerover", d = e === "mouseout" || e === "pointerout", m && n !== ua && (S = n.relatedTarget || n.fromElement) && (Kn(S) || S[Gt])) break e;
                if ((d || m) && (m = c.window === c ? c : (m = c.ownerDocument) ? m.defaultView || m.parentWindow : window, d ? (S = n.relatedTarget || n.toElement, d = u, S = S ? Kn(S) : null, S !== null && (x = dr(S), S !== x || S.tag !== 5 && S.tag !== 6) && (S = null)) : (d = null, S = u), d !== S)) {
                    if (y = Wc, E = "onMouseLeave", h = "onMouseEnter", p = "mouse", (e === "pointerout" || e === "pointerover") && (y = Qc, E = "onPointerLeave", h = "onPointerEnter", p = "pointer"), x = d == null ? m : kr(d), g = S == null ? m : kr(S), m = new y(E, p + "leave", d, n, c), m.target = x, m.relatedTarget = g, E = null, Kn(c) === u && (y = new y(h, p + "enter", S, n, c), y.target = g, y.relatedTarget = x, E = y), x = E, d && S) t: {
                        for (y = d, h = S, p = 0, g = y; g; g = yr(g)) p++;
                        for (g = 0, E = h; E; E = yr(E)) g++;
                        for (; 0 < p - g;) y = yr(y),
                        p--;
                        for (; 0 < g - p;) h = yr(h),
                        g--;
                        for (; p--;) {
                            if (y === h || h !== null && y === h.alternate) break t;
                            y = yr(y), h = yr(h)
                        }
                        y = null
                    }
                    else y = null;
                    d !== null && rd(f, m, d, y, !1), S !== null && x !== null && rd(f, x, S, y, !0)
                }
            }
            e: {
                if (m = u ? kr(u) : window, d = m.nodeName && m.nodeName.toLowerCase(), d === "select" || d === "input" && m.type === "file") var C = _y;
                else if (Yc(m))
                    if (Ap) C = My;
                    else { C = Oy; var k = Ay }
                else(d = m.nodeName) && d.toLowerCase() === "input" && (m.type === "checkbox" || m.type === "radio") && (C = Ly);
                if (C && (C = C(e, u))) { _p(f, C, n, c); break e }
                k && k(e, m, u),
                e === "focusout" && (k = m._wrapperState) && k.controlled && m.type === "number" && oa(m, "number", m.value)
            }
            switch (k = u ? kr(u) : window, e) {
                case "focusin":
                    (Yc(k) || k.contentEditable === "true") && (Er = k, ga = u, Oo = null);
                    break;
                case "focusout":
                    Oo = ga = Er = null;
                    break;
                case "mousedown":
                    va = !0;
                    break;
                case "contextmenu":
                case "mouseup":
                case "dragend":
                    va = !1, ed(f, n, c);
                    break;
                case "selectionchange":
                    if (zy) break;
                case "keydown":
                case "keyup":
                    ed(f, n, c)
            }
            var P;
            if (bu) e: {
                switch (e) {
                    case "compositionstart":
                        var N = "onCompositionStart";
                        break e;
                    case "compositionend":
                        N = "onCompositionEnd";
                        break e;
                    case "compositionupdate":
                        N = "onCompositionUpdate";
                        break e
                }
                N = void 0
            }
            else Sr ? Rp(e, n) && (N = "onCompositionEnd") : e === "keydown" && n.keyCode === 229 && (N = "onCompositionStart");N && (Np && n.locale !== "ko" && (Sr || N !== "onCompositionStart" ? N === "onCompositionEnd" && Sr && (P = Tp()) : (Cn = c, Cu = "value" in Cn ? Cn.value : Cn.textContent, Sr = !0)), k = di(u, N), 0 < k.length && (N = new Hc(N, e, null, n, c), f.push({ event: N, listeners: k }), P ? N.data = P : (P = jp(n), P !== null && (N.data = P)))),
            (P = by ? Ty(e, n) : Ny(e, n)) && (u = di(u, "onBeforeInput"), 0 < u.length && (c = new Hc("onBeforeInput", "beforeinput", null, n, c), f.push({ event: c, listeners: u }), c.data = P))
        }
        Up(f, t)
    })
}

function Go(e, t, n) { return { instance: e, listener: t, currentTarget: n } }

function di(e, t) {
    for (var n = t + "Capture", r = []; e !== null;) {
        var o = e,
            s = o.stateNode;
        o.tag === 5 && s !== null && (o = s, s = Bo(e, n), s != null && r.unshift(Go(e, s, o)), s = Bo(e, t), s != null && r.push(Go(e, s, o))), e = e.return
    }
    return r
}

function yr(e) {
    if (e === null) return null;
    do e = e.return; while (e && e.tag !== 5);
    return e || null
}

function rd(e, t, n, r, o) {
    for (var s = t._reactName, i = []; n !== null && n !== r;) {
        var l = n,
            a = l.alternate,
            u = l.stateNode;
        if (a !== null && a === r) break;
        l.tag === 5 && u !== null && (l = u, o ? (a = Bo(n, s), a != null && i.unshift(Go(n, a, l))) : o || (a = Bo(n, s), a != null && i.push(Go(n, a, l)))), n = n.return
    }
    i.length !== 0 && e.push({ event: t, listeners: i })
}
var Uy = /\r\n?/g,
    Vy = /\u0000|\uFFFD/g;

function od(e) { return (typeof e == "string" ? e : "" + e).replace(Uy, `
`).replace(Vy, "") }

function js(e, t, n) { if (t = od(t), od(e) !== t && n) throw Error(j(425)) }

function fi() {}
var ya = null,
    wa = null;

function xa(e, t) { return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null }
var Sa = typeof setTimeout == "function" ? setTimeout : void 0,
    Wy = typeof clearTimeout == "function" ? clearTimeout : void 0,
    sd = typeof Promise == "function" ? Promise : void 0,
    Hy = typeof queueMicrotask == "function" ? queueMicrotask : typeof sd < "u" ? function(e) { return sd.resolve(null).then(e).catch(Qy) } : Sa;

function Qy(e) { setTimeout(function() { throw e }) }

function Ol(e, t) {
    var n = t,
        r = 0;
    do {
        var o = n.nextSibling;
        if (e.removeChild(n), o && o.nodeType === 8)
            if (n = o.data, n === "/$") {
                if (r === 0) { e.removeChild(o), Wo(t); return }
                r--
            } else n !== "$" && n !== "$?" && n !== "$!" || r++;
        n = o
    } while (n);
    Wo(t)
}

function Rn(e) { for (; e != null; e = e.nextSibling) { var t = e.nodeType; if (t === 1 || t === 3) break; if (t === 8) { if (t = e.data, t === "$" || t === "$!" || t === "$?") break; if (t === "/$") return null } } return e }

function id(e) {
    e = e.previousSibling;
    for (var t = 0; e;) {
        if (e.nodeType === 8) {
            var n = e.data;
            if (n === "$" || n === "$!" || n === "$?") {
                if (t === 0) return e;
                t--
            } else n === "/$" && t++
        }
        e = e.previousSibling
    }
    return null
}
var ao = Math.random().toString(36).slice(2),
    At = "__reactFiber$" + ao,
    Yo = "__reactProps$" + ao,
    Gt = "__reactContainer$" + ao,
    Ea = "__reactEvents$" + ao,
    Ky = "__reactListeners$" + ao,
    Gy = "__reactHandles$" + ao;

function Kn(e) {
    var t = e[At];
    if (t) return t;
    for (var n = e.parentNode; n;) {
        if (t = n[Gt] || n[At]) {
            if (n = t.alternate, t.child !== null || n !== null && n.child !== null)
                for (e = id(e); e !== null;) {
                    if (n = e[At]) return n;
                    e = id(e)
                }
            return t
        }
        e = n, n = e.parentNode
    }
    return null
}

function ds(e) { return e = e[At] || e[Gt], !e || e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3 ? null : e }

function kr(e) { if (e.tag === 5 || e.tag === 6) return e.stateNode; throw Error(j(33)) }

function Bi(e) { return e[Yo] || null }
var Ca = [],
    Pr = -1;

function Fn(e) { return { current: e } }

function se(e) { 0 > Pr || (e.current = Ca[Pr], Ca[Pr] = null, Pr--) }

function ne(e, t) { Pr++, Ca[Pr] = e.current, e.current = t }
var Mn = {},
    Oe = Fn(Mn),
    Ue = Fn(!1),
    or = Mn;

function qr(e, t) {
    var n = e.type.contextTypes;
    if (!n) return Mn;
    var r = e.stateNode;
    if (r && r.__reactInternalMemoizedUnmaskedChildContext === t) return r.__reactInternalMemoizedMaskedChildContext;
    var o = {},
        s;
    for (s in n) o[s] = t[s];
    return r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = t, e.__reactInternalMemoizedMaskedChildContext = o), o
}

function Ve(e) { return e = e.childContextTypes, e != null }

function pi() { se(Ue), se(Oe) }

function ld(e, t, n) {
    if (Oe.current !== Mn) throw Error(j(168));
    ne(Oe, t), ne(Ue, n)
}

function Wp(e, t, n) {
    var r = e.stateNode;
    if (t = t.childContextTypes, typeof r.getChildContext != "function") return n;
    r = r.getChildContext();
    for (var o in r)
        if (!(o in t)) throw Error(j(108, Av(e) || "Unknown", o));
    return ce({}, n, r)
}

function hi(e) { return e = (e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext || Mn, or = Oe.current, ne(Oe, e), ne(Ue, Ue.current), !0 }

function ad(e, t, n) {
    var r = e.stateNode;
    if (!r) throw Error(j(169));
    n ? (e = Wp(e, t, or), r.__reactInternalMemoizedMergedChildContext = e, se(Ue), se(Oe), ne(Oe, e)) : se(Ue), ne(Ue, n)
}
var Bt = null,
    Ui = !1,
    Ll = !1;

function Hp(e) { Bt === null ? Bt = [e] : Bt.push(e) }

function Yy(e) { Ui = !0, Hp(e) }

function $n() {
    if (!Ll && Bt !== null) {
        Ll = !0;
        var e = 0,
            t = ee;
        try {
            var n = Bt;
            for (ee = 1; e < n.length; e++) {
                var r = n[e];
                do r = r(!0); while (r !== null)
            }
            Bt = null, Ui = !1
        } catch (o) { throw Bt !== null && (Bt = Bt.slice(e + 1)), gp(wu, $n), o } finally { ee = t, Ll = !1 }
    }
    return null
}
var br = [],
    Tr = 0,
    mi = null,
    gi = 0,
    ot = [],
    st = 0,
    sr = null,
    Vt = 1,
    Wt = "";

function Hn(e, t) { br[Tr++] = gi, br[Tr++] = mi, mi = e, gi = t }

function Qp(e, t, n) {
    ot[st++] = Vt, ot[st++] = Wt, ot[st++] = sr, sr = e;
    var r = Vt;
    e = Wt;
    var o = 32 - xt(r) - 1;
    r &= ~(1 << o), n += 1;
    var s = 32 - xt(t) + o;
    if (30 < s) {
        var i = o - o % 5;
        s = (r & (1 << i) - 1).toString(32), r >>= i, o -= i, Vt = 1 << 32 - xt(t) + o | n << o | r, Wt = s + e
    } else Vt = 1 << s | n << o | r, Wt = e
}

function Nu(e) { e.return !== null && (Hn(e, 1), Qp(e, 1, 0)) }

function Ru(e) { for (; e === mi;) mi = br[--Tr], br[Tr] = null, gi = br[--Tr], br[Tr] = null; for (; e === sr;) sr = ot[--st], ot[st] = null, Wt = ot[--st], ot[st] = null, Vt = ot[--st], ot[st] = null }
var qe = null,
    Xe = null,
    le = !1,
    wt = null;

function Kp(e, t) {
    var n = it(5, null, null, 0);
    n.elementType = "DELETED", n.stateNode = t, n.return = e, t = e.deletions, t === null ? (e.deletions = [n], e.flags |= 16) : t.push(n)
}

function ud(e, t) {
    switch (e.tag) {
        case 5:
            var n = e.type;
            return t = t.nodeType !== 1 || n.toLowerCase() !== t.nodeName.toLowerCase() ? null : t, t !== null ? (e.stateNode = t, qe = e, Xe = Rn(t.firstChild), !0) : !1;
        case 6:
            return t = e.pendingProps === "" || t.nodeType !== 3 ? null : t, t !== null ? (e.stateNode = t, qe = e, Xe = null, !0) : !1;
        case 13:
            return t = t.nodeType !== 8 ? null : t, t !== null ? (n = sr !== null ? { id: Vt, overflow: Wt } : null, e.memoizedState = { dehydrated: t, treeContext: n, retryLane: 1073741824 }, n = it(18, null, null, 0), n.stateNode = t, n.return = e, e.child = n, qe = e, Xe = null, !0) : !1;
        default:
            return !1
    }
}

function ka(e) { return (e.mode & 1) !== 0 && (e.flags & 128) === 0 }

function Pa(e) {
    if (le) {
        var t = Xe;
        if (t) {
            var n = t;
            if (!ud(e, t)) {
                if (ka(e)) throw Error(j(418));
                t = Rn(n.nextSibling);
                var r = qe;
                t && ud(e, t) ? Kp(r, n) : (e.flags = e.flags & -4097 | 2, le = !1, qe = e)
            }
        } else {
            if (ka(e)) throw Error(j(418));
            e.flags = e.flags & -4097 | 2, le = !1, qe = e
        }
    }
}

function cd(e) {
    for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13;) e = e.return;
    qe = e
}

function _s(e) {
    if (e !== qe) return !1;
    if (!le) return cd(e), le = !0, !1;
    var t;
    if ((t = e.tag !== 3) && !(t = e.tag !== 5) && (t = e.type, t = t !== "head" && t !== "body" && !xa(e.type, e.memoizedProps)), t && (t = Xe)) { if (ka(e)) throw Gp(), Error(j(418)); for (; t;) Kp(e, t), t = Rn(t.nextSibling) }
    if (cd(e), e.tag === 13) {
        if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e) throw Error(j(317));
        e: {
            for (e = e.nextSibling, t = 0; e;) {
                if (e.nodeType === 8) {
                    var n = e.data;
                    if (n === "/$") {
                        if (t === 0) { Xe = Rn(e.nextSibling); break e }
                        t--
                    } else n !== "$" && n !== "$!" && n !== "$?" || t++
                }
                e = e.nextSibling
            }
            Xe = null
        }
    } else Xe = qe ? Rn(e.stateNode.nextSibling) : null;
    return !0
}

function Gp() { for (var e = Xe; e;) e = Rn(e.nextSibling) }

function Zr() { Xe = qe = null, le = !1 }

function ju(e) { wt === null ? wt = [e] : wt.push(e) }
var Xy = en.ReactCurrentBatchConfig;

function xo(e, t, n) {
    if (e = n.ref, e !== null && typeof e != "function" && typeof e != "object") {
        if (n._owner) {
            if (n = n._owner, n) { if (n.tag !== 1) throw Error(j(309)); var r = n.stateNode }
            if (!r) throw Error(j(147, e));
            var o = r,
                s = "" + e;
            return t !== null && t.ref !== null && typeof t.ref == "function" && t.ref._stringRef === s ? t.ref : (t = function(i) {
                var l = o.refs;
                i === null ? delete l[s] : l[s] = i
            }, t._stringRef = s, t)
        }
        if (typeof e != "string") throw Error(j(284));
        if (!n._owner) throw Error(j(290, e))
    }
    return e
}

function As(e, t) { throw e = Object.prototype.toString.call(t), Error(j(31, e === "[object Object]" ? "object with keys {" + Object.keys(t).join(", ") + "}" : e)) }

function dd(e) { var t = e._init; return t(e._payload) }

function Yp(e) {
    function t(h, p) {
        if (e) {
            var g = h.deletions;
            g === null ? (h.deletions = [p], h.flags |= 16) : g.push(p)
        }
    }

    function n(h, p) { if (!e) return null; for (; p !== null;) t(h, p), p = p.sibling; return null }

    function r(h, p) { for (h = new Map; p !== null;) p.key !== null ? h.set(p.key, p) : h.set(p.index, p), p = p.sibling; return h }

    function o(h, p) { return h = On(h, p), h.index = 0, h.sibling = null, h }

    function s(h, p, g) { return h.index = g, e ? (g = h.alternate, g !== null ? (g = g.index, g < p ? (h.flags |= 2, p) : g) : (h.flags |= 2, p)) : (h.flags |= 1048576, p) }

    function i(h) { return e && h.alternate === null && (h.flags |= 2), h }

    function l(h, p, g, E) { return p === null || p.tag !== 6 ? (p = Bl(g, h.mode, E), p.return = h, p) : (p = o(p, g), p.return = h, p) }

    function a(h, p, g, E) { var C = g.type; return C === xr ? c(h, p, g.props.children, E, g.key) : p !== null && (p.elementType === C || typeof C == "object" && C !== null && C.$$typeof === pn && dd(C) === p.type) ? (E = o(p, g.props), E.ref = xo(h, p, g), E.return = h, E) : (E = ei(g.type, g.key, g.props, null, h.mode, E), E.ref = xo(h, p, g), E.return = h, E) }

    function u(h, p, g, E) { return p === null || p.tag !== 4 || p.stateNode.containerInfo !== g.containerInfo || p.stateNode.implementation !== g.implementation ? (p = Ul(g, h.mode, E), p.return = h, p) : (p = o(p, g.children || []), p.return = h, p) }

    function c(h, p, g, E, C) { return p === null || p.tag !== 7 ? (p = rr(g, h.mode, E, C), p.return = h, p) : (p = o(p, g), p.return = h, p) }

    function f(h, p, g) {
        if (typeof p == "string" && p !== "" || typeof p == "number") return p = Bl("" + p, h.mode, g), p.return = h, p;
        if (typeof p == "object" && p !== null) {
            switch (p.$$typeof) {
                case Ss:
                    return g = ei(p.type, p.key, p.props, null, h.mode, g), g.ref = xo(h, null, p), g.return = h, g;
                case wr:
                    return p = Ul(p, h.mode, g), p.return = h, p;
                case pn:
                    var E = p._init;
                    return f(h, E(p._payload), g)
            }
            if (bo(p) || mo(p)) return p = rr(p, h.mode, g, null), p.return = h, p;
            As(h, p)
        }
        return null
    }

    function m(h, p, g, E) {
        var C = p !== null ? p.key : null;
        if (typeof g == "string" && g !== "" || typeof g == "number") return C !== null ? null : l(h, p, "" + g, E);
        if (typeof g == "object" && g !== null) {
            switch (g.$$typeof) {
                case Ss:
                    return g.key === C ? a(h, p, g, E) : null;
                case wr:
                    return g.key === C ? u(h, p, g, E) : null;
                case pn:
                    return C = g._init, m(h, p, C(g._payload), E)
            }
            if (bo(g) || mo(g)) return C !== null ? null : c(h, p, g, E, null);
            As(h, g)
        }
        return null
    }

    function d(h, p, g, E, C) {
        if (typeof E == "string" && E !== "" || typeof E == "number") return h = h.get(g) || null, l(p, h, "" + E, C);
        if (typeof E == "object" && E !== null) {
            switch (E.$$typeof) {
                case Ss:
                    return h = h.get(E.key === null ? g : E.key) || null, a(p, h, E, C);
                case wr:
                    return h = h.get(E.key === null ? g : E.key) || null, u(p, h, E, C);
                case pn:
                    var k = E._init;
                    return d(h, p, g, k(E._payload), C)
            }
            if (bo(E) || mo(E)) return h = h.get(g) || null, c(p, h, E, C, null);
            As(p, E)
        }
        return null
    }

    function S(h, p, g, E) {
        for (var C = null, k = null, P = p, N = p = 0, L = null; P !== null && N < g.length; N++) {
            P.index > N ? (L = P, P = null) : L = P.sibling;
            var A = m(h, P, g[N], E);
            if (A === null) { P === null && (P = L); break }
            e && P && A.alternate === null && t(h, P), p = s(A, p, N), k === null ? C = A : k.sibling = A, k = A, P = L
        }
        if (N === g.length) return n(h, P), le && Hn(h, N), C;
        if (P === null) { for (; N < g.length; N++) P = f(h, g[N], E), P !== null && (p = s(P, p, N), k === null ? C = P : k.sibling = P, k = P); return le && Hn(h, N), C }
        for (P = r(h, P); N < g.length; N++) L = d(P, h, N, g[N], E), L !== null && (e && L.alternate !== null && P.delete(L.key === null ? N : L.key), p = s(L, p, N), k === null ? C = L : k.sibling = L, k = L);
        return e && P.forEach(function($) { return t(h, $) }), le && Hn(h, N), C
    }

    function y(h, p, g, E) {
        var C = mo(g);
        if (typeof C != "function") throw Error(j(150));
        if (g = C.call(g), g == null) throw Error(j(151));
        for (var k = C = null, P = p, N = p = 0, L = null, A = g.next(); P !== null && !A.done; N++, A = g.next()) {
            P.index > N ? (L = P, P = null) : L = P.sibling;
            var $ = m(h, P, A.value, E);
            if ($ === null) { P === null && (P = L); break }
            e && P && $.alternate === null && t(h, P), p = s($, p, N), k === null ? C = $ : k.sibling = $, k = $, P = L
        }
        if (A.done) return n(h, P), le && Hn(h, N), C;
        if (P === null) { for (; !A.done; N++, A = g.next()) A = f(h, A.value, E), A !== null && (p = s(A, p, N), k === null ? C = A : k.sibling = A, k = A); return le && Hn(h, N), C }
        for (P = r(h, P); !A.done; N++, A = g.next()) A = d(P, h, N, A.value, E), A !== null && (e && A.alternate !== null && P.delete(A.key === null ? N : A.key), p = s(A, p, N), k === null ? C = A : k.sibling = A, k = A);
        return e && P.forEach(function(D) { return t(h, D) }), le && Hn(h, N), C
    }

    function x(h, p, g, E) {
        if (typeof g == "object" && g !== null && g.type === xr && g.key === null && (g = g.props.children), typeof g == "object" && g !== null) {
            switch (g.$$typeof) {
                case Ss:
                    e: {
                        for (var C = g.key, k = p; k !== null;) {
                            if (k.key === C) {
                                if (C = g.type, C === xr) { if (k.tag === 7) { n(h, k.sibling), p = o(k, g.props.children), p.return = h, h = p; break e } } else if (k.elementType === C || typeof C == "object" && C !== null && C.$$typeof === pn && dd(C) === k.type) { n(h, k.sibling), p = o(k, g.props), p.ref = xo(h, k, g), p.return = h, h = p; break e }
                                n(h, k);
                                break
                            } else t(h, k);
                            k = k.sibling
                        }
                        g.type === xr ? (p = rr(g.props.children, h.mode, E, g.key), p.return = h, h = p) : (E = ei(g.type, g.key, g.props, null, h.mode, E), E.ref = xo(h, p, g), E.return = h, h = E)
                    }
                    return i(h);
                case wr:
                    e: {
                        for (k = g.key; p !== null;) {
                            if (p.key === k)
                                if (p.tag === 4 && p.stateNode.containerInfo === g.containerInfo && p.stateNode.implementation === g.implementation) { n(h, p.sibling), p = o(p, g.children || []), p.return = h, h = p; break e } else { n(h, p); break }
                            else t(h, p);
                            p = p.sibling
                        }
                        p = Ul(g, h.mode, E),
                        p.return = h,
                        h = p
                    }
                    return i(h);
                case pn:
                    return k = g._init, x(h, p, k(g._payload), E)
            }
            if (bo(g)) return S(h, p, g, E);
            if (mo(g)) return y(h, p, g, E);
            As(h, g)
        }
        return typeof g == "string" && g !== "" || typeof g == "number" ? (g = "" + g, p !== null && p.tag === 6 ? (n(h, p.sibling), p = o(p, g), p.return = h, h = p) : (n(h, p), p = Bl(g, h.mode, E), p.return = h, h = p), i(h)) : n(h, p)
    }
    return x
}
var Jr = Yp(!0),
    Xp = Yp(!1),
    vi = Fn(null),
    yi = null,
    Nr = null,
    _u = null;

function Au() { _u = Nr = yi = null }

function Ou(e) {
    var t = vi.current;
    se(vi), e._currentValue = t
}

function ba(e, t, n) {
    for (; e !== null;) {
        var r = e.alternate;
        if ((e.childLanes & t) !== t ? (e.childLanes |= t, r !== null && (r.childLanes |= t)) : r !== null && (r.childLanes & t) !== t && (r.childLanes |= t), e === n) break;
        e = e.return
    }
}

function Mr(e, t) { yi = e, _u = Nr = null, e = e.dependencies, e !== null && e.firstContext !== null && (e.lanes & t && (Be = !0), e.firstContext = null) }

function at(e) {
    var t = e._currentValue;
    if (_u !== e)
        if (e = { context: e, memoizedValue: t, next: null }, Nr === null) {
            if (yi === null) throw Error(j(308));
            Nr = e, yi.dependencies = { lanes: 0, firstContext: e }
        } else Nr = Nr.next = e;
    return t
}
var Gn = null;

function Lu(e) { Gn === null ? Gn = [e] : Gn.push(e) }

function qp(e, t, n, r) { var o = t.interleaved; return o === null ? (n.next = n, Lu(t)) : (n.next = o.next, o.next = n), t.interleaved = n, Yt(e, r) }

function Yt(e, t) { e.lanes |= t; var n = e.alternate; for (n !== null && (n.lanes |= t), n = e, e = e.return; e !== null;) e.childLanes |= t, n = e.alternate, n !== null && (n.childLanes |= t), n = e, e = e.return; return n.tag === 3 ? n.stateNode : null }
var hn = !1;

function Mu(e) { e.updateQueue = { baseState: e.memoizedState, firstBaseUpdate: null, lastBaseUpdate: null, shared: { pending: null, interleaved: null, lanes: 0 }, effects: null } }

function Zp(e, t) { e = e.updateQueue, t.updateQueue === e && (t.updateQueue = { baseState: e.baseState, firstBaseUpdate: e.firstBaseUpdate, lastBaseUpdate: e.lastBaseUpdate, shared: e.shared, effects: e.effects }) }

function Qt(e, t) { return { eventTime: e, lane: t, tag: 0, payload: null, callback: null, next: null } }

function jn(e, t, n) { var r = e.updateQueue; if (r === null) return null; if (r = r.shared, Z & 2) { var o = r.pending; return o === null ? t.next = t : (t.next = o.next, o.next = t), r.pending = t, Yt(e, n) } return o = r.interleaved, o === null ? (t.next = t, Lu(r)) : (t.next = o.next, o.next = t), r.interleaved = t, Yt(e, n) }

function Gs(e, t, n) {
    if (t = t.updateQueue, t !== null && (t = t.shared, (n & 4194240) !== 0)) {
        var r = t.lanes;
        r &= e.pendingLanes, n |= r, t.lanes = n, xu(e, n)
    }
}

function fd(e, t) {
    var n = e.updateQueue,
        r = e.alternate;
    if (r !== null && (r = r.updateQueue, n === r)) {
        var o = null,
            s = null;
        if (n = n.firstBaseUpdate, n !== null) {
            do {
                var i = { eventTime: n.eventTime, lane: n.lane, tag: n.tag, payload: n.payload, callback: n.callback, next: null };
                s === null ? o = s = i : s = s.next = i, n = n.next
            } while (n !== null);
            s === null ? o = s = t : s = s.next = t
        } else o = s = t;
        n = { baseState: r.baseState, firstBaseUpdate: o, lastBaseUpdate: s, shared: r.shared, effects: r.effects }, e.updateQueue = n;
        return
    }
    e = n.lastBaseUpdate, e === null ? n.firstBaseUpdate = t : e.next = t, n.lastBaseUpdate = t
}

function wi(e, t, n, r) {
    var o = e.updateQueue;
    hn = !1;
    var s = o.firstBaseUpdate,
        i = o.lastBaseUpdate,
        l = o.shared.pending;
    if (l !== null) {
        o.shared.pending = null;
        var a = l,
            u = a.next;
        a.next = null, i === null ? s = u : i.next = u, i = a;
        var c = e.alternate;
        c !== null && (c = c.updateQueue, l = c.lastBaseUpdate, l !== i && (l === null ? c.firstBaseUpdate = u : l.next = u, c.lastBaseUpdate = a))
    }
    if (s !== null) {
        var f = o.baseState;
        i = 0, c = u = a = null, l = s;
        do {
            var m = l.lane,
                d = l.eventTime;
            if ((r & m) === m) {
                c !== null && (c = c.next = { eventTime: d, lane: 0, tag: l.tag, payload: l.payload, callback: l.callback, next: null });
                e: {
                    var S = e,
                        y = l;
                    switch (m = t, d = n, y.tag) {
                        case 1:
                            if (S = y.payload, typeof S == "function") { f = S.call(d, f, m); break e }
                            f = S;
                            break e;
                        case 3:
                            S.flags = S.flags & -65537 | 128;
                        case 0:
                            if (S = y.payload, m = typeof S == "function" ? S.call(d, f, m) : S, m == null) break e;
                            f = ce({}, f, m);
                            break e;
                        case 2:
                            hn = !0
                    }
                }
                l.callback !== null && l.lane !== 0 && (e.flags |= 64, m = o.effects, m === null ? o.effects = [l] : m.push(l))
            } else d = { eventTime: d, lane: m, tag: l.tag, payload: l.payload, callback: l.callback, next: null }, c === null ? (u = c = d, a = f) : c = c.next = d, i |= m;
            if (l = l.next, l === null) {
                if (l = o.shared.pending, l === null) break;
                m = l, l = m.next, m.next = null, o.lastBaseUpdate = m, o.shared.pending = null
            }
        } while (!0);
        if (c === null && (a = f), o.baseState = a, o.firstBaseUpdate = u, o.lastBaseUpdate = c, t = o.shared.interleaved, t !== null) {
            o = t;
            do i |= o.lane, o = o.next; while (o !== t)
        } else s === null && (o.shared.lanes = 0);
        lr |= i, e.lanes = i, e.memoizedState = f
    }
}

function pd(e, t, n) {
    if (e = t.effects, t.effects = null, e !== null)
        for (t = 0; t < e.length; t++) {
            var r = e[t],
                o = r.callback;
            if (o !== null) {
                if (r.callback = null, r = n, typeof o != "function") throw Error(j(191, o));
                o.call(r)
            }
        }
}
var fs = {},
    Mt = Fn(fs),
    Xo = Fn(fs),
    qo = Fn(fs);

function Yn(e) { if (e === fs) throw Error(j(174)); return e }

function Iu(e, t) {
    switch (ne(qo, t), ne(Xo, e), ne(Mt, fs), e = t.nodeType, e) {
        case 9:
        case 11:
            t = (t = t.documentElement) ? t.namespaceURI : ia(null, "");
            break;
        default:
            e = e === 8 ? t.parentNode : t, t = e.namespaceURI || null, e = e.tagName, t = ia(t, e)
    }
    se(Mt), ne(Mt, t)
}

function eo() { se(Mt), se(Xo), se(qo) }

function Jp(e) {
    Yn(qo.current);
    var t = Yn(Mt.current),
        n = ia(t, e.type);
    t !== n && (ne(Xo, e), ne(Mt, n))
}

function Du(e) { Xo.current === e && (se(Mt), se(Xo)) }
var ae = Fn(0);

function xi(e) {
    for (var t = e; t !== null;) {
        if (t.tag === 13) { var n = t.memoizedState; if (n !== null && (n = n.dehydrated, n === null || n.data === "$?" || n.data === "$!")) return t } else if (t.tag === 19 && t.memoizedProps.revealOrder !== void 0) { if (t.flags & 128) return t } else if (t.child !== null) { t.child.return = t, t = t.child; continue }
        if (t === e) break;
        for (; t.sibling === null;) {
            if (t.return === null || t.return === e) return null;
            t = t.return
        }
        t.sibling.return = t.return, t = t.sibling
    }
    return null
}
var Ml = [];

function zu() {
    for (var e = 0; e < Ml.length; e++) Ml[e]._workInProgressVersionPrimary = null;
    Ml.length = 0
}
var Ys = en.ReactCurrentDispatcher,
    Il = en.ReactCurrentBatchConfig,
    ir = 0,
    ue = null,
    ge = null,
    xe = null,
    Si = !1,
    Lo = !1,
    Zo = 0,
    qy = 0;

function Re() { throw Error(j(321)) }

function Fu(e, t) {
    if (t === null) return !1;
    for (var n = 0; n < t.length && n < e.length; n++)
        if (!Et(e[n], t[n])) return !1;
    return !0
}

function $u(e, t, n, r, o, s) {
    if (ir = s, ue = t, t.memoizedState = null, t.updateQueue = null, t.lanes = 0, Ys.current = e === null || e.memoizedState === null ? t0 : n0, e = n(r, o), Lo) {
        s = 0;
        do {
            if (Lo = !1, Zo = 0, 25 <= s) throw Error(j(301));
            s += 1, xe = ge = null, t.updateQueue = null, Ys.current = r0, e = n(r, o)
        } while (Lo)
    }
    if (Ys.current = Ei, t = ge !== null && ge.next !== null, ir = 0, xe = ge = ue = null, Si = !1, t) throw Error(j(300));
    return e
}

function Bu() { var e = Zo !== 0; return Zo = 0, e }

function Nt() { var e = { memoizedState: null, baseState: null, baseQueue: null, queue: null, next: null }; return xe === null ? ue.memoizedState = xe = e : xe = xe.next = e, xe }

function ut() {
    if (ge === null) {
        var e = ue.alternate;
        e = e !== null ? e.memoizedState : null
    } else e = ge.next;
    var t = xe === null ? ue.memoizedState : xe.next;
    if (t !== null) xe = t, ge = e;
    else {
        if (e === null) throw Error(j(310));
        ge = e, e = { memoizedState: ge.memoizedState, baseState: ge.baseState, baseQueue: ge.baseQueue, queue: ge.queue, next: null }, xe === null ? ue.memoizedState = xe = e : xe = xe.next = e
    }
    return xe
}

function Jo(e, t) { return typeof t == "function" ? t(e) : t }

function Dl(e) {
    var t = ut(),
        n = t.queue;
    if (n === null) throw Error(j(311));
    n.lastRenderedReducer = e;
    var r = ge,
        o = r.baseQueue,
        s = n.pending;
    if (s !== null) {
        if (o !== null) {
            var i = o.next;
            o.next = s.next, s.next = i
        }
        r.baseQueue = o = s, n.pending = null
    }
    if (o !== null) {
        s = o.next, r = r.baseState;
        var l = i = null,
            a = null,
            u = s;
        do {
            var c = u.lane;
            if ((ir & c) === c) a !== null && (a = a.next = { lane: 0, action: u.action, hasEagerState: u.hasEagerState, eagerState: u.eagerState, next: null }), r = u.hasEagerState ? u.eagerState : e(r, u.action);
            else {
                var f = { lane: c, action: u.action, hasEagerState: u.hasEagerState, eagerState: u.eagerState, next: null };
                a === null ? (l = a = f, i = r) : a = a.next = f, ue.lanes |= c, lr |= c
            }
            u = u.next
        } while (u !== null && u !== s);
        a === null ? i = r : a.next = l, Et(r, t.memoizedState) || (Be = !0), t.memoizedState = r, t.baseState = i, t.baseQueue = a, n.lastRenderedState = r
    }
    if (e = n.interleaved, e !== null) {
        o = e;
        do s = o.lane, ue.lanes |= s, lr |= s, o = o.next; while (o !== e)
    } else o === null && (n.lanes = 0);
    return [t.memoizedState, n.dispatch]
}

function zl(e) {
    var t = ut(),
        n = t.queue;
    if (n === null) throw Error(j(311));
    n.lastRenderedReducer = e;
    var r = n.dispatch,
        o = n.pending,
        s = t.memoizedState;
    if (o !== null) {
        n.pending = null;
        var i = o = o.next;
        do s = e(s, i.action), i = i.next; while (i !== o);
        Et(s, t.memoizedState) || (Be = !0), t.memoizedState = s, t.baseQueue === null && (t.baseState = s), n.lastRenderedState = s
    }
    return [s, r]
}

function eh() {}

function th(e, t) {
    var n = ue,
        r = ut(),
        o = t(),
        s = !Et(r.memoizedState, o);
    if (s && (r.memoizedState = o, Be = !0), r = r.queue, Uu(oh.bind(null, n, r, e), [e]), r.getSnapshot !== t || s || xe !== null && xe.memoizedState.tag & 1) {
        if (n.flags |= 2048, es(9, rh.bind(null, n, r, o, t), void 0, null), Se === null) throw Error(j(349));
        ir & 30 || nh(n, t, o)
    }
    return o
}

function nh(e, t, n) { e.flags |= 16384, e = { getSnapshot: t, value: n }, t = ue.updateQueue, t === null ? (t = { lastEffect: null, stores: null }, ue.updateQueue = t, t.stores = [e]) : (n = t.stores, n === null ? t.stores = [e] : n.push(e)) }

function rh(e, t, n, r) { t.value = n, t.getSnapshot = r, sh(t) && ih(e) }

function oh(e, t, n) { return n(function() { sh(t) && ih(e) }) }

function sh(e) {
    var t = e.getSnapshot;
    e = e.value;
    try { var n = t(); return !Et(e, n) } catch { return !0 }
}

function ih(e) {
    var t = Yt(e, 1);
    t !== null && St(t, e, 1, -1)
}

function hd(e) { var t = Nt(); return typeof e == "function" && (e = e()), t.memoizedState = t.baseState = e, e = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: Jo, lastRenderedState: e }, t.queue = e, e = e.dispatch = e0.bind(null, ue, e), [t.memoizedState, e] }

function es(e, t, n, r) { return e = { tag: e, create: t, destroy: n, deps: r, next: null }, t = ue.updateQueue, t === null ? (t = { lastEffect: null, stores: null }, ue.updateQueue = t, t.lastEffect = e.next = e) : (n = t.lastEffect, n === null ? t.lastEffect = e.next = e : (r = n.next, n.next = e, e.next = r, t.lastEffect = e)), e }

function lh() { return ut().memoizedState }

function Xs(e, t, n, r) {
    var o = Nt();
    ue.flags |= e, o.memoizedState = es(1 | t, n, void 0, r === void 0 ? null : r)
}

function Vi(e, t, n, r) {
    var o = ut();
    r = r === void 0 ? null : r;
    var s = void 0;
    if (ge !== null) { var i = ge.memoizedState; if (s = i.destroy, r !== null && Fu(r, i.deps)) { o.memoizedState = es(t, n, s, r); return } }
    ue.flags |= e, o.memoizedState = es(1 | t, n, s, r)
}

function md(e, t) { return Xs(8390656, 8, e, t) }

function Uu(e, t) { return Vi(2048, 8, e, t) }

function ah(e, t) { return Vi(4, 2, e, t) }

function uh(e, t) { return Vi(4, 4, e, t) }

function ch(e, t) {
    if (typeof t == "function") return e = e(), t(e),
        function() { t(null) };
    if (t != null) return e = e(), t.current = e,
        function() { t.current = null }
}

function dh(e, t, n) { return n = n != null ? n.concat([e]) : null, Vi(4, 4, ch.bind(null, t, e), n) }

function Vu() {}

function fh(e, t) {
    var n = ut();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Fu(t, r[1]) ? r[0] : (n.memoizedState = [e, t], e)
}

function ph(e, t) {
    var n = ut();
    t = t === void 0 ? null : t;
    var r = n.memoizedState;
    return r !== null && t !== null && Fu(t, r[1]) ? r[0] : (e = e(), n.memoizedState = [e, t], e)
}

function hh(e, t, n) { return ir & 21 ? (Et(n, t) || (n = wp(), ue.lanes |= n, lr |= n, e.baseState = !0), t) : (e.baseState && (e.baseState = !1, Be = !0), e.memoizedState = n) }

function Zy(e, t) {
    var n = ee;
    ee = n !== 0 && 4 > n ? n : 4, e(!0);
    var r = Il.transition;
    Il.transition = {};
    try { e(!1), t() } finally { ee = n, Il.transition = r }
}

function mh() { return ut().memoizedState }

function Jy(e, t, n) {
    var r = An(e);
    if (n = { lane: r, action: n, hasEagerState: !1, eagerState: null, next: null }, gh(e)) vh(t, n);
    else if (n = qp(e, t, n, r), n !== null) {
        var o = Ie();
        St(n, e, r, o), yh(n, t, r)
    }
}

function e0(e, t, n) {
    var r = An(e),
        o = { lane: r, action: n, hasEagerState: !1, eagerState: null, next: null };
    if (gh(e)) vh(t, o);
    else {
        var s = e.alternate;
        if (e.lanes === 0 && (s === null || s.lanes === 0) && (s = t.lastRenderedReducer, s !== null)) try {
            var i = t.lastRenderedState,
                l = s(i, n);
            if (o.hasEagerState = !0, o.eagerState = l, Et(l, i)) {
                var a = t.interleaved;
                a === null ? (o.next = o, Lu(t)) : (o.next = a.next, a.next = o), t.interleaved = o;
                return
            }
        } catch {} finally {}
        n = qp(e, t, o, r), n !== null && (o = Ie(), St(n, e, r, o), yh(n, t, r))
    }
}

function gh(e) { var t = e.alternate; return e === ue || t !== null && t === ue }

function vh(e, t) {
    Lo = Si = !0;
    var n = e.pending;
    n === null ? t.next = t : (t.next = n.next, n.next = t), e.pending = t
}

function yh(e, t, n) {
    if (n & 4194240) {
        var r = t.lanes;
        r &= e.pendingLanes, n |= r, t.lanes = n, xu(e, n)
    }
}
var Ei = { readContext: at, useCallback: Re, useContext: Re, useEffect: Re, useImperativeHandle: Re, useInsertionEffect: Re, useLayoutEffect: Re, useMemo: Re, useReducer: Re, useRef: Re, useState: Re, useDebugValue: Re, useDeferredValue: Re, useTransition: Re, useMutableSource: Re, useSyncExternalStore: Re, useId: Re, unstable_isNewReconciler: !1 },
    t0 = {
        readContext: at,
        useCallback: function(e, t) { return Nt().memoizedState = [e, t === void 0 ? null : t], e },
        useContext: at,
        useEffect: md,
        useImperativeHandle: function(e, t, n) { return n = n != null ? n.concat([e]) : null, Xs(4194308, 4, ch.bind(null, t, e), n) },
        useLayoutEffect: function(e, t) { return Xs(4194308, 4, e, t) },
        useInsertionEffect: function(e, t) { return Xs(4, 2, e, t) },
        useMemo: function(e, t) { var n = Nt(); return t = t === void 0 ? null : t, e = e(), n.memoizedState = [e, t], e },
        useReducer: function(e, t, n) { var r = Nt(); return t = n !== void 0 ? n(t) : t, r.memoizedState = r.baseState = t, e = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: e, lastRenderedState: t }, r.queue = e, e = e.dispatch = Jy.bind(null, ue, e), [r.memoizedState, e] },
        useRef: function(e) { var t = Nt(); return e = { current: e }, t.memoizedState = e },
        useState: hd,
        useDebugValue: Vu,
        useDeferredValue: function(e) { return Nt().memoizedState = e },
        useTransition: function() {
            var e = hd(!1),
                t = e[0];
            return e = Zy.bind(null, e[1]), Nt().memoizedState = e, [t, e]
        },
        useMutableSource: function() {},
        useSyncExternalStore: function(e, t, n) {
            var r = ue,
                o = Nt();
            if (le) {
                if (n === void 0) throw Error(j(407));
                n = n()
            } else {
                if (n = t(), Se === null) throw Error(j(349));
                ir & 30 || nh(r, t, n)
            }
            o.memoizedState = n;
            var s = { value: n, getSnapshot: t };
            return o.queue = s, md(oh.bind(null, r, s, e), [e]), r.flags |= 2048, es(9, rh.bind(null, r, s, n, t), void 0, null), n
        },
        useId: function() {
            var e = Nt(),
                t = Se.identifierPrefix;
            if (le) {
                var n = Wt,
                    r = Vt;
                n = (r & ~(1 << 32 - xt(r) - 1)).toString(32) + n, t = ":" + t + "R" + n, n = Zo++, 0 < n && (t += "H" + n.toString(32)), t += ":"
            } else n = qy++, t = ":" + t + "r" + n.toString(32) + ":";
            return e.memoizedState = t
        },
        unstable_isNewReconciler: !1
    },
    n0 = {
        readContext: at,
        useCallback: fh,
        useContext: at,
        useEffect: Uu,
        useImperativeHandle: dh,
        useInsertionEffect: ah,
        useLayoutEffect: uh,
        useMemo: ph,
        useReducer: Dl,
        useRef: lh,
        useState: function() { return Dl(Jo) },
        useDebugValue: Vu,
        useDeferredValue: function(e) { var t = ut(); return hh(t, ge.memoizedState, e) },
        useTransition: function() {
            var e = Dl(Jo)[0],
                t = ut().memoizedState;
            return [e, t]
        },
        useMutableSource: eh,
        useSyncExternalStore: th,
        useId: mh,
        unstable_isNewReconciler: !1
    },
    r0 = {
        readContext: at,
        useCallback: fh,
        useContext: at,
        useEffect: Uu,
        useImperativeHandle: dh,
        useInsertionEffect: ah,
        useLayoutEffect: uh,
        useMemo: ph,
        useReducer: zl,
        useRef: lh,
        useState: function() { return zl(Jo) },
        useDebugValue: Vu,
        useDeferredValue: function(e) { var t = ut(); return ge === null ? t.memoizedState = e : hh(t, ge.memoizedState, e) },
        useTransition: function() {
            var e = zl(Jo)[0],
                t = ut().memoizedState;
            return [e, t]
        },
        useMutableSource: eh,
        useSyncExternalStore: th,
        useId: mh,
        unstable_isNewReconciler: !1
    };

function mt(e, t) { if (e && e.defaultProps) { t = ce({}, t), e = e.defaultProps; for (var n in e) t[n] === void 0 && (t[n] = e[n]); return t } return t }

function Ta(e, t, n, r) { t = e.memoizedState, n = n(r, t), n = n == null ? t : ce({}, t, n), e.memoizedState = n, e.lanes === 0 && (e.updateQueue.baseState = n) }
var Wi = {
    isMounted: function(e) { return (e = e._reactInternals) ? dr(e) === e : !1 },
    enqueueSetState: function(e, t, n) {
        e = e._reactInternals;
        var r = Ie(),
            o = An(e),
            s = Qt(r, o);
        s.payload = t, n != null && (s.callback = n), t = jn(e, s, o), t !== null && (St(t, e, o, r), Gs(t, e, o))
    },
    enqueueReplaceState: function(e, t, n) {
        e = e._reactInternals;
        var r = Ie(),
            o = An(e),
            s = Qt(r, o);
        s.tag = 1, s.payload = t, n != null && (s.callback = n), t = jn(e, s, o), t !== null && (St(t, e, o, r), Gs(t, e, o))
    },
    enqueueForceUpdate: function(e, t) {
        e = e._reactInternals;
        var n = Ie(),
            r = An(e),
            o = Qt(n, r);
        o.tag = 2, t != null && (o.callback = t), t = jn(e, o, r), t !== null && (St(t, e, r, n), Gs(t, e, r))
    }
};

function gd(e, t, n, r, o, s, i) { return e = e.stateNode, typeof e.shouldComponentUpdate == "function" ? e.shouldComponentUpdate(r, s, i) : t.prototype && t.prototype.isPureReactComponent ? !Qo(n, r) || !Qo(o, s) : !0 }

function wh(e, t, n) {
    var r = !1,
        o = Mn,
        s = t.contextType;
    return typeof s == "object" && s !== null ? s = at(s) : (o = Ve(t) ? or : Oe.current, r = t.contextTypes, s = (r = r != null) ? qr(e, o) : Mn), t = new t(n, s), e.memoizedState = t.state !== null && t.state !== void 0 ? t.state : null, t.updater = Wi, e.stateNode = t, t._reactInternals = e, r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = o, e.__reactInternalMemoizedMaskedChildContext = s), t
}

function vd(e, t, n, r) { e = t.state, typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(n, r), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(n, r), t.state !== e && Wi.enqueueReplaceState(t, t.state, null) }

function Na(e, t, n, r) {
    var o = e.stateNode;
    o.props = n, o.state = e.memoizedState, o.refs = {}, Mu(e);
    var s = t.contextType;
    typeof s == "object" && s !== null ? o.context = at(s) : (s = Ve(t) ? or : Oe.current, o.context = qr(e, s)), o.state = e.memoizedState, s = t.getDerivedStateFromProps, typeof s == "function" && (Ta(e, t, s, n), o.state = e.memoizedState), typeof t.getDerivedStateFromProps == "function" || typeof o.getSnapshotBeforeUpdate == "function" || typeof o.UNSAFE_componentWillMount != "function" && typeof o.componentWillMount != "function" || (t = o.state, typeof o.componentWillMount == "function" && o.componentWillMount(), typeof o.UNSAFE_componentWillMount == "function" && o.UNSAFE_componentWillMount(), t !== o.state && Wi.enqueueReplaceState(o, o.state, null), wi(e, n, o, r), o.state = e.memoizedState), typeof o.componentDidMount == "function" && (e.flags |= 4194308)
}

function to(e, t) {
    try {
        var n = "",
            r = t;
        do n += _v(r), r = r.return; while (r);
        var o = n
    } catch (s) { o = `
Error generating stack: ` + s.message + `
` + s.stack }
    return { value: e, source: t, stack: o, digest: null }
}

function Fl(e, t, n) { return { value: e, source: null, stack: n ? ? null, digest: t ? ? null } }

function Ra(e, t) { try { console.error(t.value) } catch (n) { setTimeout(function() { throw n }) } }
var o0 = typeof WeakMap == "function" ? WeakMap : Map;

function xh(e, t, n) { n = Qt(-1, n), n.tag = 3, n.payload = { element: null }; var r = t.value; return n.callback = function() { ki || (ki = !0, Fa = r), Ra(e, t) }, n }

function Sh(e, t, n) {
    n = Qt(-1, n), n.tag = 3;
    var r = e.type.getDerivedStateFromError;
    if (typeof r == "function") {
        var o = t.value;
        n.payload = function() { return r(o) }, n.callback = function() { Ra(e, t) }
    }
    var s = e.stateNode;
    return s !== null && typeof s.componentDidCatch == "function" && (n.callback = function() {
        Ra(e, t), typeof r != "function" && (_n === null ? _n = new Set([this]) : _n.add(this));
        var i = t.stack;
        this.componentDidCatch(t.value, { componentStack: i !== null ? i : "" })
    }), n
}

function yd(e, t, n) {
    var r = e.pingCache;
    if (r === null) {
        r = e.pingCache = new o0;
        var o = new Set;
        r.set(t, o)
    } else o = r.get(t), o === void 0 && (o = new Set, r.set(t, o));
    o.has(n) || (o.add(n), e = y0.bind(null, e, t, n), t.then(e, e))
}

function wd(e) {
    do {
        var t;
        if ((t = e.tag === 13) && (t = e.memoizedState, t = t !== null ? t.dehydrated !== null : !0), t) return e;
        e = e.return
    } while (e !== null);
    return null
}

function xd(e, t, n, r, o) { return e.mode & 1 ? (e.flags |= 65536, e.lanes = o, e) : (e === t ? e.flags |= 65536 : (e.flags |= 128, n.flags |= 131072, n.flags &= -52805, n.tag === 1 && (n.alternate === null ? n.tag = 17 : (t = Qt(-1, 1), t.tag = 2, jn(n, t, 1))), n.lanes |= 1), e) }
var s0 = en.ReactCurrentOwner,
    Be = !1;

function Le(e, t, n, r) { t.child = e === null ? Xp(t, null, n, r) : Jr(t, e.child, n, r) }

function Sd(e, t, n, r, o) { n = n.render; var s = t.ref; return Mr(t, o), r = $u(e, t, n, r, s, o), n = Bu(), e !== null && !Be ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~o, Xt(e, t, o)) : (le && n && Nu(t), t.flags |= 1, Le(e, t, r, o), t.child) }

function Ed(e, t, n, r, o) { if (e === null) { var s = n.type; return typeof s == "function" && !qu(s) && s.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0 ? (t.tag = 15, t.type = s, Eh(e, t, s, r, o)) : (e = ei(n.type, null, r, t, t.mode, o), e.ref = t.ref, e.return = t, t.child = e) } if (s = e.child, !(e.lanes & o)) { var i = s.memoizedProps; if (n = n.compare, n = n !== null ? n : Qo, n(i, r) && e.ref === t.ref) return Xt(e, t, o) } return t.flags |= 1, e = On(s, r), e.ref = t.ref, e.return = t, t.child = e }

function Eh(e, t, n, r, o) {
    if (e !== null) {
        var s = e.memoizedProps;
        if (Qo(s, r) && e.ref === t.ref)
            if (Be = !1, t.pendingProps = r = s, (e.lanes & o) !== 0) e.flags & 131072 && (Be = !0);
            else return t.lanes = e.lanes, Xt(e, t, o)
    }
    return ja(e, t, n, r, o)
}

function Ch(e, t, n) {
    var r = t.pendingProps,
        o = r.children,
        s = e !== null ? e.memoizedState : null;
    if (r.mode === "hidden")
        if (!(t.mode & 1)) t.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, ne(jr, Ke), Ke |= n;
        else {
            if (!(n & 1073741824)) return e = s !== null ? s.baseLanes | n : n, t.lanes = t.childLanes = 1073741824, t.memoizedState = { baseLanes: e, cachePool: null, transitions: null }, t.updateQueue = null, ne(jr, Ke), Ke |= e, null;
            t.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, r = s !== null ? s.baseLanes : n, ne(jr, Ke), Ke |= r
        }
    else s !== null ? (r = s.baseLanes | n, t.memoizedState = null) : r = n, ne(jr, Ke), Ke |= r;
    return Le(e, t, o, n), t.child
}

function kh(e, t) {
    var n = t.ref;
    (e === null && n !== null || e !== null && e.ref !== n) && (t.flags |= 512, t.flags |= 2097152)
}

function ja(e, t, n, r, o) { var s = Ve(n) ? or : Oe.current; return s = qr(t, s), Mr(t, o), n = $u(e, t, n, r, s, o), r = Bu(), e !== null && !Be ? (t.updateQueue = e.updateQueue, t.flags &= -2053, e.lanes &= ~o, Xt(e, t, o)) : (le && r && Nu(t), t.flags |= 1, Le(e, t, n, o), t.child) }

function Cd(e, t, n, r, o) {
    if (Ve(n)) {
        var s = !0;
        hi(t)
    } else s = !1;
    if (Mr(t, o), t.stateNode === null) qs(e, t), wh(t, n, r), Na(t, n, r, o), r = !0;
    else if (e === null) {
        var i = t.stateNode,
            l = t.memoizedProps;
        i.props = l;
        var a = i.context,
            u = n.contextType;
        typeof u == "object" && u !== null ? u = at(u) : (u = Ve(n) ? or : Oe.current, u = qr(t, u));
        var c = n.getDerivedStateFromProps,
            f = typeof c == "function" || typeof i.getSnapshotBeforeUpdate == "function";
        f || typeof i.UNSAFE_componentWillReceiveProps != "function" && typeof i.componentWillReceiveProps != "function" || (l !== r || a !== u) && vd(t, i, r, u), hn = !1;
        var m = t.memoizedState;
        i.state = m, wi(t, r, i, o), a = t.memoizedState, l !== r || m !== a || Ue.current || hn ? (typeof c == "function" && (Ta(t, n, c, r), a = t.memoizedState), (l = hn || gd(t, n, l, r, m, a, u)) ? (f || typeof i.UNSAFE_componentWillMount != "function" && typeof i.componentWillMount != "function" || (typeof i.componentWillMount == "function" && i.componentWillMount(), typeof i.UNSAFE_componentWillMount == "function" && i.UNSAFE_componentWillMount()), typeof i.componentDidMount == "function" && (t.flags |= 4194308)) : (typeof i.componentDidMount == "function" && (t.flags |= 4194308), t.memoizedProps = r, t.memoizedState = a), i.props = r, i.state = a, i.context = u, r = l) : (typeof i.componentDidMount == "function" && (t.flags |= 4194308), r = !1)
    } else {
        i = t.stateNode, Zp(e, t), l = t.memoizedProps, u = t.type === t.elementType ? l : mt(t.type, l), i.props = u, f = t.pendingProps, m = i.context, a = n.contextType, typeof a == "object" && a !== null ? a = at(a) : (a = Ve(n) ? or : Oe.current, a = qr(t, a));
        var d = n.getDerivedStateFromProps;
        (c = typeof d == "function" || typeof i.getSnapshotBeforeUpdate == "function") || typeof i.UNSAFE_componentWillReceiveProps != "function" && typeof i.componentWillReceiveProps != "function" || (l !== f || m !== a) && vd(t, i, r, a), hn = !1, m = t.memoizedState, i.state = m, wi(t, r, i, o);
        var S = t.memoizedState;
        l !== f || m !== S || Ue.current || hn ? (typeof d == "function" && (Ta(t, n, d, r), S = t.memoizedState), (u = hn || gd(t, n, u, r, m, S, a) || !1) ? (c || typeof i.UNSAFE_componentWillUpdate != "function" && typeof i.componentWillUpdate != "function" || (typeof i.componentWillUpdate == "function" && i.componentWillUpdate(r, S, a), typeof i.UNSAFE_componentWillUpdate == "function" && i.UNSAFE_componentWillUpdate(r, S, a)), typeof i.componentDidUpdate == "function" && (t.flags |= 4), typeof i.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof i.componentDidUpdate != "function" || l === e.memoizedProps && m === e.memoizedState || (t.flags |= 4), typeof i.getSnapshotBeforeUpdate != "function" || l === e.memoizedProps && m === e.memoizedState || (t.flags |= 1024), t.memoizedProps = r, t.memoizedState = S), i.props = r, i.state = S, i.context = a, r = u) : (typeof i.componentDidUpdate != "function" || l === e.memoizedProps && m === e.memoizedState || (t.flags |= 4), typeof i.getSnapshotBeforeUpdate != "function" || l === e.memoizedProps && m === e.memoizedState || (t.flags |= 1024), r = !1)
    }
    return _a(e, t, n, r, s, o)
}

function _a(e, t, n, r, o, s) {
    kh(e, t);
    var i = (t.flags & 128) !== 0;
    if (!r && !i) return o && ad(t, n, !1), Xt(e, t, s);
    r = t.stateNode, s0.current = t;
    var l = i && typeof n.getDerivedStateFromError != "function" ? null : r.render();
    return t.flags |= 1, e !== null && i ? (t.child = Jr(t, e.child, null, s), t.child = Jr(t, null, l, s)) : Le(e, t, l, s), t.memoizedState = r.state, o && ad(t, n, !0), t.child
}

function Ph(e) {
    var t = e.stateNode;
    t.pendingContext ? ld(e, t.pendingContext, t.pendingContext !== t.context) : t.context && ld(e, t.context, !1), Iu(e, t.containerInfo)
}

function kd(e, t, n, r, o) { return Zr(), ju(o), t.flags |= 256, Le(e, t, n, r), t.child }
var Aa = { dehydrated: null, treeContext: null, retryLane: 0 };

function Oa(e) { return { baseLanes: e, cachePool: null, transitions: null } }

function bh(e, t, n) {
    var r = t.pendingProps,
        o = ae.current,
        s = !1,
        i = (t.flags & 128) !== 0,
        l;
    if ((l = i) || (l = e !== null && e.memoizedState === null ? !1 : (o & 2) !== 0), l ? (s = !0, t.flags &= -129) : (e === null || e.memoizedState !== null) && (o |= 1), ne(ae, o & 1), e === null) return Pa(t), e = t.memoizedState, e !== null && (e = e.dehydrated, e !== null) ? (t.mode & 1 ? e.data === "$!" ? t.lanes = 8 : t.lanes = 1073741824 : t.lanes = 1, null) : (i = r.children, e = r.fallback, s ? (r = t.mode, s = t.child, i = { mode: "hidden", children: i }, !(r & 1) && s !== null ? (s.childLanes = 0, s.pendingProps = i) : s = Ki(i, r, 0, null), e = rr(e, r, n, null), s.return = t, e.return = t, s.sibling = e, t.child = s, t.child.memoizedState = Oa(n), t.memoizedState = Aa, e) : Wu(t, i));
    if (o = e.memoizedState, o !== null && (l = o.dehydrated, l !== null)) return i0(e, t, i, r, l, o, n);
    if (s) { s = r.fallback, i = t.mode, o = e.child, l = o.sibling; var a = { mode: "hidden", children: r.children }; return !(i & 1) && t.child !== o ? (r = t.child, r.childLanes = 0, r.pendingProps = a, t.deletions = null) : (r = On(o, a), r.subtreeFlags = o.subtreeFlags & 14680064), l !== null ? s = On(l, s) : (s = rr(s, i, n, null), s.flags |= 2), s.return = t, r.return = t, r.sibling = s, t.child = r, r = s, s = t.child, i = e.child.memoizedState, i = i === null ? Oa(n) : { baseLanes: i.baseLanes | n, cachePool: null, transitions: i.transitions }, s.memoizedState = i, s.childLanes = e.childLanes & ~n, t.memoizedState = Aa, r }
    return s = e.child, e = s.sibling, r = On(s, { mode: "visible", children: r.children }), !(t.mode & 1) && (r.lanes = n), r.return = t, r.sibling = null, e !== null && (n = t.deletions, n === null ? (t.deletions = [e], t.flags |= 16) : n.push(e)), t.child = r, t.memoizedState = null, r
}

function Wu(e, t) { return t = Ki({ mode: "visible", children: t }, e.mode, 0, null), t.return = e, e.child = t }

function Os(e, t, n, r) { return r !== null && ju(r), Jr(t, e.child, null, n), e = Wu(t, t.pendingProps.children), e.flags |= 2, t.memoizedState = null, e }

function i0(e, t, n, r, o, s, i) {
    if (n) return t.flags & 256 ? (t.flags &= -257, r = Fl(Error(j(422))), Os(e, t, i, r)) : t.memoizedState !== null ? (t.child = e.child, t.flags |= 128, null) : (s = r.fallback, o = t.mode, r = Ki({ mode: "visible", children: r.children }, o, 0, null), s = rr(s, o, i, null), s.flags |= 2, r.return = t, s.return = t, r.sibling = s, t.child = r, t.mode & 1 && Jr(t, e.child, null, i), t.child.memoizedState = Oa(i), t.memoizedState = Aa, s);
    if (!(t.mode & 1)) return Os(e, t, i, null);
    if (o.data === "$!") { if (r = o.nextSibling && o.nextSibling.dataset, r) var l = r.dgst; return r = l, s = Error(j(419)), r = Fl(s, r, void 0), Os(e, t, i, r) }
    if (l = (i & e.childLanes) !== 0, Be || l) {
        if (r = Se, r !== null) {
            switch (i & -i) {
                case 4:
                    o = 2;
                    break;
                case 16:
                    o = 8;
                    break;
                case 64:
                case 128:
                case 256:
                case 512:
                case 1024:
                case 2048:
                case 4096:
                case 8192:
                case 16384:
                case 32768:
                case 65536:
                case 131072:
                case 262144:
                case 524288:
                case 1048576:
                case 2097152:
                case 4194304:
                case 8388608:
                case 16777216:
                case 33554432:
                case 67108864:
                    o = 32;
                    break;
                case 536870912:
                    o = 268435456;
                    break;
                default:
                    o = 0
            }
            o = o & (r.suspendedLanes | i) ? 0 : o, o !== 0 && o !== s.retryLane && (s.retryLane = o, Yt(e, o), St(r, e, o, -1))
        }
        return Xu(), r = Fl(Error(j(421))), Os(e, t, i, r)
    }
    return o.data === "$?" ? (t.flags |= 128, t.child = e.child, t = w0.bind(null, e), o._reactRetry = t, null) : (e = s.treeContext, Xe = Rn(o.nextSibling), qe = t, le = !0, wt = null, e !== null && (ot[st++] = Vt, ot[st++] = Wt, ot[st++] = sr, Vt = e.id, Wt = e.overflow, sr = t), t = Wu(t, r.children), t.flags |= 4096, t)
}

function Pd(e, t, n) {
    e.lanes |= t;
    var r = e.alternate;
    r !== null && (r.lanes |= t), ba(e.return, t, n)
}

function $l(e, t, n, r, o) {
    var s = e.memoizedState;
    s === null ? e.memoizedState = { isBackwards: t, rendering: null, renderingStartTime: 0, last: r, tail: n, tailMode: o } : (s.isBackwards = t, s.rendering = null, s.renderingStartTime = 0, s.last = r, s.tail = n, s.tailMode = o)
}

function Th(e, t, n) {
    var r = t.pendingProps,
        o = r.revealOrder,
        s = r.tail;
    if (Le(e, t, r.children, n), r = ae.current, r & 2) r = r & 1 | 2, t.flags |= 128;
    else {
        if (e !== null && e.flags & 128) e: for (e = t.child; e !== null;) {
            if (e.tag === 13) e.memoizedState !== null && Pd(e, n, t);
            else if (e.tag === 19) Pd(e, n, t);
            else if (e.child !== null) { e.child.return = e, e = e.child; continue }
            if (e === t) break e;
            for (; e.sibling === null;) {
                if (e.return === null || e.return === t) break e;
                e = e.return
            }
            e.sibling.return = e.return, e = e.sibling
        }
        r &= 1
    }
    if (ne(ae, r), !(t.mode & 1)) t.memoizedState = null;
    else switch (o) {
        case "forwards":
            for (n = t.child, o = null; n !== null;) e = n.alternate, e !== null && xi(e) === null && (o = n), n = n.sibling;
            n = o, n === null ? (o = t.child, t.child = null) : (o = n.sibling, n.sibling = null), $l(t, !1, o, n, s);
            break;
        case "backwards":
            for (n = null, o = t.child, t.child = null; o !== null;) {
                if (e = o.alternate, e !== null && xi(e) === null) { t.child = o; break }
                e = o.sibling, o.sibling = n, n = o, o = e
            }
            $l(t, !0, n, null, s);
            break;
        case "together":
            $l(t, !1, null, null, void 0);
            break;
        default:
            t.memoizedState = null
    }
    return t.child
}

function qs(e, t) {!(t.mode & 1) && e !== null && (e.alternate = null, t.alternate = null, t.flags |= 2) }

function Xt(e, t, n) {
    if (e !== null && (t.dependencies = e.dependencies), lr |= t.lanes, !(n & t.childLanes)) return null;
    if (e !== null && t.child !== e.child) throw Error(j(153));
    if (t.child !== null) {
        for (e = t.child, n = On(e, e.pendingProps), t.child = n, n.return = t; e.sibling !== null;) e = e.sibling, n = n.sibling = On(e, e.pendingProps), n.return = t;
        n.sibling = null
    }
    return t.child
}

function l0(e, t, n) {
    switch (t.tag) {
        case 3:
            Ph(t), Zr();
            break;
        case 5:
            Jp(t);
            break;
        case 1:
            Ve(t.type) && hi(t);
            break;
        case 4:
            Iu(t, t.stateNode.containerInfo);
            break;
        case 10:
            var r = t.type._context,
                o = t.memoizedProps.value;
            ne(vi, r._currentValue), r._currentValue = o;
            break;
        case 13:
            if (r = t.memoizedState, r !== null) return r.dehydrated !== null ? (ne(ae, ae.current & 1), t.flags |= 128, null) : n & t.child.childLanes ? bh(e, t, n) : (ne(ae, ae.current & 1), e = Xt(e, t, n), e !== null ? e.sibling : null);
            ne(ae, ae.current & 1);
            break;
        case 19:
            if (r = (n & t.childLanes) !== 0, e.flags & 128) {
                if (r) return Th(e, t, n);
                t.flags |= 128
            }
            if (o = t.memoizedState, o !== null && (o.rendering = null, o.tail = null, o.lastEffect = null), ne(ae, ae.current), r) break;
            return null;
        case 22:
        case 23:
            return t.lanes = 0, Ch(e, t, n)
    }
    return Xt(e, t, n)
}
var Nh, La, Rh, jh;
Nh = function(e, t) {
    for (var n = t.child; n !== null;) {
        if (n.tag === 5 || n.tag === 6) e.appendChild(n.stateNode);
        else if (n.tag !== 4 && n.child !== null) { n.child.return = n, n = n.child; continue }
        if (n === t) break;
        for (; n.sibling === null;) {
            if (n.return === null || n.return === t) return;
            n = n.return
        }
        n.sibling.return = n.return, n = n.sibling
    }
};
La = function() {};
Rh = function(e, t, n, r) {
    var o = e.memoizedProps;
    if (o !== r) {
        e = t.stateNode, Yn(Mt.current);
        var s = null;
        switch (n) {
            case "input":
                o = na(e, o), r = na(e, r), s = [];
                break;
            case "select":
                o = ce({}, o, { value: void 0 }), r = ce({}, r, { value: void 0 }), s = [];
                break;
            case "textarea":
                o = sa(e, o), r = sa(e, r), s = [];
                break;
            default:
                typeof o.onClick != "function" && typeof r.onClick == "function" && (e.onclick = fi)
        }
        la(n, r);
        var i;
        n = null;
        for (u in o)
            if (!r.hasOwnProperty(u) && o.hasOwnProperty(u) && o[u] != null)
                if (u === "style") { var l = o[u]; for (i in l) l.hasOwnProperty(i) && (n || (n = {}), n[i] = "") } else u !== "dangerouslySetInnerHTML" && u !== "children" && u !== "suppressContentEditableWarning" && u !== "suppressHydrationWarning" && u !== "autoFocus" && (Fo.hasOwnProperty(u) ? s || (s = []) : (s = s || []).push(u, null));
        for (u in r) {
            var a = r[u];
            if (l = o != null ? o[u] : void 0, r.hasOwnProperty(u) && a !== l && (a != null || l != null))
                if (u === "style")
                    if (l) { for (i in l) !l.hasOwnProperty(i) || a && a.hasOwnProperty(i) || (n || (n = {}), n[i] = ""); for (i in a) a.hasOwnProperty(i) && l[i] !== a[i] && (n || (n = {}), n[i] = a[i]) } else n || (s || (s = []), s.push(u, n)), n = a;
            else u === "dangerouslySetInnerHTML" ? (a = a ? a.__html : void 0, l = l ? l.__html : void 0, a != null && l !== a && (s = s || []).push(u, a)) : u === "children" ? typeof a != "string" && typeof a != "number" || (s = s || []).push(u, "" + a) : u !== "suppressContentEditableWarning" && u !== "suppressHydrationWarning" && (Fo.hasOwnProperty(u) ? (a != null && u === "onScroll" && oe("scroll", e), s || l === a || (s = [])) : (s = s || []).push(u, a))
        }
        n && (s = s || []).push("style", n);
        var u = s;
        (t.updateQueue = u) && (t.flags |= 4)
    }
};
jh = function(e, t, n, r) { n !== r && (t.flags |= 4) };

function So(e, t) {
    if (!le) switch (e.tailMode) {
        case "hidden":
            t = e.tail;
            for (var n = null; t !== null;) t.alternate !== null && (n = t), t = t.sibling;
            n === null ? e.tail = null : n.sibling = null;
            break;
        case "collapsed":
            n = e.tail;
            for (var r = null; n !== null;) n.alternate !== null && (r = n), n = n.sibling;
            r === null ? t || e.tail === null ? e.tail = null : e.tail.sibling = null : r.sibling = null
    }
}

function je(e) {
    var t = e.alternate !== null && e.alternate.child === e.child,
        n = 0,
        r = 0;
    if (t)
        for (var o = e.child; o !== null;) n |= o.lanes | o.childLanes, r |= o.subtreeFlags & 14680064, r |= o.flags & 14680064, o.return = e, o = o.sibling;
    else
        for (o = e.child; o !== null;) n |= o.lanes | o.childLanes, r |= o.subtreeFlags, r |= o.flags, o.return = e, o = o.sibling;
    return e.subtreeFlags |= r, e.childLanes = n, t
}

function a0(e, t, n) {
    var r = t.pendingProps;
    switch (Ru(t), t.tag) {
        case 2:
        case 16:
        case 15:
        case 0:
        case 11:
        case 7:
        case 8:
        case 12:
        case 9:
        case 14:
            return je(t), null;
        case 1:
            return Ve(t.type) && pi(), je(t), null;
        case 3:
            return r = t.stateNode, eo(), se(Ue), se(Oe), zu(), r.pendingContext && (r.context = r.pendingContext, r.pendingContext = null), (e === null || e.child === null) && (_s(t) ? t.flags |= 4 : e === null || e.memoizedState.isDehydrated && !(t.flags & 256) || (t.flags |= 1024, wt !== null && (Ua(wt), wt = null))), La(e, t), je(t), null;
        case 5:
            Du(t);
            var o = Yn(qo.current);
            if (n = t.type, e !== null && t.stateNode != null) Rh(e, t, n, r, o), e.ref !== t.ref && (t.flags |= 512, t.flags |= 2097152);
            else {
                if (!r) { if (t.stateNode === null) throw Error(j(166)); return je(t), null }
                if (e = Yn(Mt.current), _s(t)) {
                    r = t.stateNode, n = t.type;
                    var s = t.memoizedProps;
                    switch (r[At] = t, r[Yo] = s, e = (t.mode & 1) !== 0, n) {
                        case "dialog":
                            oe("cancel", r), oe("close", r);
                            break;
                        case "iframe":
                        case "object":
                        case "embed":
                            oe("load", r);
                            break;
                        case "video":
                        case "audio":
                            for (o = 0; o < No.length; o++) oe(No[o], r);
                            break;
                        case "source":
                            oe("error", r);
                            break;
                        case "img":
                        case "image":
                        case "link":
                            oe("error", r), oe("load", r);
                            break;
                        case "details":
                            oe("toggle", r);
                            break;
                        case "input":
                            Lc(r, s), oe("invalid", r);
                            break;
                        case "select":
                            r._wrapperState = { wasMultiple: !!s.multiple }, oe("invalid", r);
                            break;
                        case "textarea":
                            Ic(r, s), oe("invalid", r)
                    }
                    la(n, s), o = null;
                    for (var i in s)
                        if (s.hasOwnProperty(i)) {
                            var l = s[i];
                            i === "children" ? typeof l == "string" ? r.textContent !== l && (s.suppressHydrationWarning !== !0 && js(r.textContent, l, e), o = ["children", l]) : typeof l == "number" && r.textContent !== "" + l && (s.suppressHydrationWarning !== !0 && js(r.textContent, l, e), o = ["children", "" + l]) : Fo.hasOwnProperty(i) && l != null && i === "onScroll" && oe("scroll", r)
                        }
                    switch (n) {
                        case "input":
                            Es(r), Mc(r, s, !0);
                            break;
                        case "textarea":
                            Es(r), Dc(r);
                            break;
                        case "select":
                        case "option":
                            break;
                        default:
                            typeof s.onClick == "function" && (r.onclick = fi)
                    }
                    r = o, t.updateQueue = r, r !== null && (t.flags |= 4)
                } else {
                    i = o.nodeType === 9 ? o : o.ownerDocument, e === "http://www.w3.org/1999/xhtml" && (e = op(n)), e === "http://www.w3.org/1999/xhtml" ? n === "script" ? (e = i.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild)) : typeof r.is == "string" ? e = i.createElement(n, { is: r.is }) : (e = i.createElement(n), n === "select" && (i = e, r.multiple ? i.multiple = !0 : r.size && (i.size = r.size))) : e = i.createElementNS(e, n), e[At] = t, e[Yo] = r, Nh(e, t, !1, !1), t.stateNode = e;
                    e: {
                        switch (i = aa(n, r), n) {
                            case "dialog":
                                oe("cancel", e), oe("close", e), o = r;
                                break;
                            case "iframe":
                            case "object":
                            case "embed":
                                oe("load", e), o = r;
                                break;
                            case "video":
                            case "audio":
                                for (o = 0; o < No.length; o++) oe(No[o], e);
                                o = r;
                                break;
                            case "source":
                                oe("error", e), o = r;
                                break;
                            case "img":
                            case "image":
                            case "link":
                                oe("error", e), oe("load", e), o = r;
                                break;
                            case "details":
                                oe("toggle", e), o = r;
                                break;
                            case "input":
                                Lc(e, r), o = na(e, r), oe("invalid", e);
                                break;
                            case "option":
                                o = r;
                                break;
                            case "select":
                                e._wrapperState = { wasMultiple: !!r.multiple }, o = ce({}, r, { value: void 0 }), oe("invalid", e);
                                break;
                            case "textarea":
                                Ic(e, r), o = sa(e, r), oe("invalid", e);
                                break;
                            default:
                                o = r
                        }
                        la(n, o),
                        l = o;
                        for (s in l)
                            if (l.hasOwnProperty(s)) {
                                var a = l[s];
                                s === "style" ? lp(e, a) : s === "dangerouslySetInnerHTML" ? (a = a ? a.__html : void 0, a != null && sp(e, a)) : s === "children" ? typeof a == "string" ? (n !== "textarea" || a !== "") && $o(e, a) : typeof a == "number" && $o(e, "" + a) : s !== "suppressContentEditableWarning" && s !== "suppressHydrationWarning" && s !== "autoFocus" && (Fo.hasOwnProperty(s) ? a != null && s === "onScroll" && oe("scroll", e) : a != null && hu(e, s, a, i))
                            }
                        switch (n) {
                            case "input":
                                Es(e), Mc(e, r, !1);
                                break;
                            case "textarea":
                                Es(e), Dc(e);
                                break;
                            case "option":
                                r.value != null && e.setAttribute("value", "" + Ln(r.value));
                                break;
                            case "select":
                                e.multiple = !!r.multiple, s = r.value, s != null ? _r(e, !!r.multiple, s, !1) : r.defaultValue != null && _r(e, !!r.multiple, r.defaultValue, !0);
                                break;
                            default:
                                typeof o.onClick == "function" && (e.onclick = fi)
                        }
                        switch (n) {
                            case "button":
                            case "input":
                            case "select":
                            case "textarea":
                                r = !!r.autoFocus;
                                break e;
                            case "img":
                                r = !0;
                                break e;
                            default:
                                r = !1
                        }
                    }
                    r && (t.flags |= 4)
                }
                t.ref !== null && (t.flags |= 512, t.flags |= 2097152)
            }
            return je(t), null;
        case 6:
            if (e && t.stateNode != null) jh(e, t, e.memoizedProps, r);
            else {
                if (typeof r != "string" && t.stateNode === null) throw Error(j(166));
                if (n = Yn(qo.current), Yn(Mt.current), _s(t)) {
                    if (r = t.stateNode, n = t.memoizedProps, r[At] = t, (s = r.nodeValue !== n) && (e = qe, e !== null)) switch (e.tag) {
                        case 3:
                            js(r.nodeValue, n, (e.mode & 1) !== 0);
                            break;
                        case 5:
                            e.memoizedProps.suppressHydrationWarning !== !0 && js(r.nodeValue, n, (e.mode & 1) !== 0)
                    }
                    s && (t.flags |= 4)
                } else r = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(r), r[At] = t, t.stateNode = r
            }
            return je(t), null;
        case 13:
            if (se(ae), r = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
                if (le && Xe !== null && t.mode & 1 && !(t.flags & 128)) Gp(), Zr(), t.flags |= 98560, s = !1;
                else if (s = _s(t), r !== null && r.dehydrated !== null) {
                    if (e === null) {
                        if (!s) throw Error(j(318));
                        if (s = t.memoizedState, s = s !== null ? s.dehydrated : null, !s) throw Error(j(317));
                        s[At] = t
                    } else Zr(), !(t.flags & 128) && (t.memoizedState = null), t.flags |= 4;
                    je(t), s = !1
                } else wt !== null && (Ua(wt), wt = null), s = !0;
                if (!s) return t.flags & 65536 ? t : null
            }
            return t.flags & 128 ? (t.lanes = n, t) : (r = r !== null, r !== (e !== null && e.memoizedState !== null) && r && (t.child.flags |= 8192, t.mode & 1 && (e === null || ae.current & 1 ? ye === 0 && (ye = 3) : Xu())), t.updateQueue !== null && (t.flags |= 4), je(t), null);
        case 4:
            return eo(), La(e, t), e === null && Ko(t.stateNode.containerInfo), je(t), null;
        case 10:
            return Ou(t.type._context), je(t), null;
        case 17:
            return Ve(t.type) && pi(), je(t), null;
        case 19:
            if (se(ae), s = t.memoizedState, s === null) return je(t), null;
            if (r = (t.flags & 128) !== 0, i = s.rendering, i === null)
                if (r) So(s, !1);
                else {
                    if (ye !== 0 || e !== null && e.flags & 128)
                        for (e = t.child; e !== null;) {
                            if (i = xi(e), i !== null) { for (t.flags |= 128, So(s, !1), r = i.updateQueue, r !== null && (t.updateQueue = r, t.flags |= 4), t.subtreeFlags = 0, r = n, n = t.child; n !== null;) s = n, e = r, s.flags &= 14680066, i = s.alternate, i === null ? (s.childLanes = 0, s.lanes = e, s.child = null, s.subtreeFlags = 0, s.memoizedProps = null, s.memoizedState = null, s.updateQueue = null, s.dependencies = null, s.stateNode = null) : (s.childLanes = i.childLanes, s.lanes = i.lanes, s.child = i.child, s.subtreeFlags = 0, s.deletions = null, s.memoizedProps = i.memoizedProps, s.memoizedState = i.memoizedState, s.updateQueue = i.updateQueue, s.type = i.type, e = i.dependencies, s.dependencies = e === null ? null : { lanes: e.lanes, firstContext: e.firstContext }), n = n.sibling; return ne(ae, ae.current & 1 | 2), t.child }
                            e = e.sibling
                        }
                    s.tail !== null && pe() > no && (t.flags |= 128, r = !0, So(s, !1), t.lanes = 4194304)
                }
            else {
                if (!r)
                    if (e = xi(i), e !== null) { if (t.flags |= 128, r = !0, n = e.updateQueue, n !== null && (t.updateQueue = n, t.flags |= 4), So(s, !0), s.tail === null && s.tailMode === "hidden" && !i.alternate && !le) return je(t), null } else 2 * pe() - s.renderingStartTime > no && n !== 1073741824 && (t.flags |= 128, r = !0, So(s, !1), t.lanes = 4194304);
                s.isBackwards ? (i.sibling = t.child, t.child = i) : (n = s.last, n !== null ? n.sibling = i : t.child = i, s.last = i)
            }
            return s.tail !== null ? (t = s.tail, s.rendering = t, s.tail = t.sibling, s.renderingStartTime = pe(), t.sibling = null, n = ae.current, ne(ae, r ? n & 1 | 2 : n & 1), t) : (je(t), null);
        case 22:
        case 23:
            return Yu(), r = t.memoizedState !== null, e !== null && e.memoizedState !== null !== r && (t.flags |= 8192), r && t.mode & 1 ? Ke & 1073741824 && (je(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : je(t), null;
        case 24:
            return null;
        case 25:
            return null
    }
    throw Error(j(156, t.tag))
}

function u0(e, t) {
    switch (Ru(t), t.tag) {
        case 1:
            return Ve(t.type) && pi(), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
        case 3:
            return eo(), se(Ue), se(Oe), zu(), e = t.flags, e & 65536 && !(e & 128) ? (t.flags = e & -65537 | 128, t) : null;
        case 5:
            return Du(t), null;
        case 13:
            if (se(ae), e = t.memoizedState, e !== null && e.dehydrated !== null) {
                if (t.alternate === null) throw Error(j(340));
                Zr()
            }
            return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, t) : null;
        case 19:
            return se(ae), null;
        case 4:
            return eo(), null;
        case 10:
            return Ou(t.type._context), null;
        case 22:
        case 23:
            return Yu(), null;
        case 24:
            return null;
        default:
            return null
    }
}
var Ls = !1,
    Ae = !1,
    c0 = typeof WeakSet == "function" ? WeakSet : Set,
    I = null;

function Rr(e, t) {
    var n = e.ref;
    if (n !== null)
        if (typeof n == "function") try { n(null) } catch (r) { fe(e, t, r) } else n.current = null
}

function Ma(e, t, n) { try { n() } catch (r) { fe(e, t, r) } }
var bd = !1;

function d0(e, t) {
    if (ya = ui, e = Mp(), Tu(e)) {
        if ("selectionStart" in e) var n = { start: e.selectionStart, end: e.selectionEnd };
        else e: {
            n = (n = e.ownerDocument) && n.defaultView || window;
            var r = n.getSelection && n.getSelection();
            if (r && r.rangeCount !== 0) {
                n = r.anchorNode;
                var o = r.anchorOffset,
                    s = r.focusNode;
                r = r.focusOffset;
                try { n.nodeType, s.nodeType } catch { n = null; break e }
                var i = 0,
                    l = -1,
                    a = -1,
                    u = 0,
                    c = 0,
                    f = e,
                    m = null;
                t: for (;;) {
                    for (var d; f !== n || o !== 0 && f.nodeType !== 3 || (l = i + o), f !== s || r !== 0 && f.nodeType !== 3 || (a = i + r), f.nodeType === 3 && (i += f.nodeValue.length), (d = f.firstChild) !== null;) m = f, f = d;
                    for (;;) {
                        if (f === e) break t;
                        if (m === n && ++u === o && (l = i), m === s && ++c === r && (a = i), (d = f.nextSibling) !== null) break;
                        f = m, m = f.parentNode
                    }
                    f = d
                }
                n = l === -1 || a === -1 ? null : { start: l, end: a }
            } else n = null
        }
        n = n || { start: 0, end: 0 }
    } else n = null;
    for (wa = { focusedElem: e, selectionRange: n }, ui = !1, I = t; I !== null;)
        if (t = I, e = t.child, (t.subtreeFlags & 1028) !== 0 && e !== null) e.return = t, I = e;
        else
            for (; I !== null;) {
                t = I;
                try {
                    var S = t.alternate;
                    if (t.flags & 1024) switch (t.tag) {
                        case 0:
                        case 11:
                        case 15:
                            break;
                        case 1:
                            if (S !== null) {
                                var y = S.memoizedProps,
                                    x = S.memoizedState,
                                    h = t.stateNode,
                                    p = h.getSnapshotBeforeUpdate(t.elementType === t.type ? y : mt(t.type, y), x);
                                h.__reactInternalSnapshotBeforeUpdate = p
                            }
                            break;
                        case 3:
                            var g = t.stateNode.containerInfo;
                            g.nodeType === 1 ? g.textContent = "" : g.nodeType === 9 && g.documentElement && g.removeChild(g.documentElement);
                            break;
                        case 5:
                        case 6:
                        case 4:
                        case 17:
                            break;
                        default:
                            throw Error(j(163))
                    }
                } catch (E) { fe(t, t.return, E) }
                if (e = t.sibling, e !== null) { e.return = t.return, I = e; break }
                I = t.return
            }
    return S = bd, bd = !1, S
}

function Mo(e, t, n) {
    var r = t.updateQueue;
    if (r = r !== null ? r.lastEffect : null, r !== null) {
        var o = r = r.next;
        do {
            if ((o.tag & e) === e) {
                var s = o.destroy;
                o.destroy = void 0, s !== void 0 && Ma(t, n, s)
            }
            o = o.next
        } while (o !== r)
    }
}

function Hi(e, t) {
    if (t = t.updateQueue, t = t !== null ? t.lastEffect : null, t !== null) {
        var n = t = t.next;
        do {
            if ((n.tag & e) === e) {
                var r = n.create;
                n.destroy = r()
            }
            n = n.next
        } while (n !== t)
    }
}

function Ia(e) {
    var t = e.ref;
    if (t !== null) {
        var n = e.stateNode;
        switch (e.tag) {
            case 5:
                e = n;
                break;
            default:
                e = n
        }
        typeof t == "function" ? t(e) : t.current = e
    }
}

function _h(e) {
    var t = e.alternate;
    t !== null && (e.alternate = null, _h(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && (delete t[At], delete t[Yo], delete t[Ea], delete t[Ky], delete t[Gy])), e.stateNode = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null
}

function Ah(e) { return e.tag === 5 || e.tag === 3 || e.tag === 4 }

function Td(e) {
    e: for (;;) {
        for (; e.sibling === null;) {
            if (e.return === null || Ah(e.return)) return null;
            e = e.return
        }
        for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18;) {
            if (e.flags & 2 || e.child === null || e.tag === 4) continue e;
            e.child.return = e, e = e.child
        }
        if (!(e.flags & 2)) return e.stateNode
    }
}

function Da(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.nodeType === 8 ? n.parentNode.insertBefore(e, t) : n.insertBefore(e, t) : (n.nodeType === 8 ? (t = n.parentNode, t.insertBefore(e, n)) : (t = n, t.appendChild(e)), n = n._reactRootContainer, n != null || t.onclick !== null || (t.onclick = fi));
    else if (r !== 4 && (e = e.child, e !== null))
        for (Da(e, t, n), e = e.sibling; e !== null;) Da(e, t, n), e = e.sibling
}

function za(e, t, n) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, t ? n.insertBefore(e, t) : n.appendChild(e);
    else if (r !== 4 && (e = e.child, e !== null))
        for (za(e, t, n), e = e.sibling; e !== null;) za(e, t, n), e = e.sibling
}
var Ce = null,
    yt = !1;

function un(e, t, n) { for (n = n.child; n !== null;) Oh(e, t, n), n = n.sibling }

function Oh(e, t, n) {
    if (Lt && typeof Lt.onCommitFiberUnmount == "function") try { Lt.onCommitFiberUnmount(Di, n) } catch {}
    switch (n.tag) {
        case 5:
            Ae || Rr(n, t);
        case 6:
            var r = Ce,
                o = yt;
            Ce = null, un(e, t, n), Ce = r, yt = o, Ce !== null && (yt ? (e = Ce, n = n.stateNode, e.nodeType === 8 ? e.parentNode.removeChild(n) : e.removeChild(n)) : Ce.removeChild(n.stateNode));
            break;
        case 18:
            Ce !== null && (yt ? (e = Ce, n = n.stateNode, e.nodeType === 8 ? Ol(e.parentNode, n) : e.nodeType === 1 && Ol(e, n), Wo(e)) : Ol(Ce, n.stateNode));
            break;
        case 4:
            r = Ce, o = yt, Ce = n.stateNode.containerInfo, yt = !0, un(e, t, n), Ce = r, yt = o;
            break;
        case 0:
        case 11:
        case 14:
        case 15:
            if (!Ae && (r = n.updateQueue, r !== null && (r = r.lastEffect, r !== null))) {
                o = r = r.next;
                do {
                    var s = o,
                        i = s.destroy;
                    s = s.tag, i !== void 0 && (s & 2 || s & 4) && Ma(n, t, i), o = o.next
                } while (o !== r)
            }
            un(e, t, n);
            break;
        case 1:
            if (!Ae && (Rr(n, t), r = n.stateNode, typeof r.componentWillUnmount == "function")) try { r.props = n.memoizedProps, r.state = n.memoizedState, r.componentWillUnmount() } catch (l) { fe(n, t, l) }
            un(e, t, n);
            break;
        case 21:
            un(e, t, n);
            break;
        case 22:
            n.mode & 1 ? (Ae = (r = Ae) || n.memoizedState !== null, un(e, t, n), Ae = r) : un(e, t, n);
            break;
        default:
            un(e, t, n)
    }
}

function Nd(e) {
    var t = e.updateQueue;
    if (t !== null) {
        e.updateQueue = null;
        var n = e.stateNode;
        n === null && (n = e.stateNode = new c0), t.forEach(function(r) {
            var o = x0.bind(null, e, r);
            n.has(r) || (n.add(r), r.then(o, o))
        })
    }
}

function pt(e, t) {
    var n = t.deletions;
    if (n !== null)
        for (var r = 0; r < n.length; r++) {
            var o = n[r];
            try {
                var s = e,
                    i = t,
                    l = i;
                e: for (; l !== null;) {
                    switch (l.tag) {
                        case 5:
                            Ce = l.stateNode, yt = !1;
                            break e;
                        case 3:
                            Ce = l.stateNode.containerInfo, yt = !0;
                            break e;
                        case 4:
                            Ce = l.stateNode.containerInfo, yt = !0;
                            break e
                    }
                    l = l.return
                }
                if (Ce === null) throw Error(j(160));
                Oh(s, i, o), Ce = null, yt = !1;
                var a = o.alternate;
                a !== null && (a.return = null), o.return = null
            } catch (u) { fe(o, t, u) }
        }
    if (t.subtreeFlags & 12854)
        for (t = t.child; t !== null;) Lh(t, e), t = t.sibling
}

function Lh(e, t) {
    var n = e.alternate,
        r = e.flags;
    switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
            if (pt(t, e), Tt(e), r & 4) { try { Mo(3, e, e.return), Hi(3, e) } catch (y) { fe(e, e.return, y) } try { Mo(5, e, e.return) } catch (y) { fe(e, e.return, y) } }
            break;
        case 1:
            pt(t, e), Tt(e), r & 512 && n !== null && Rr(n, n.return);
            break;
        case 5:
            if (pt(t, e), Tt(e), r & 512 && n !== null && Rr(n, n.return), e.flags & 32) { var o = e.stateNode; try { $o(o, "") } catch (y) { fe(e, e.return, y) } }
            if (r & 4 && (o = e.stateNode, o != null)) {
                var s = e.memoizedProps,
                    i = n !== null ? n.memoizedProps : s,
                    l = e.type,
                    a = e.updateQueue;
                if (e.updateQueue = null, a !== null) try {
                    l === "input" && s.type === "radio" && s.name != null && np(o, s), aa(l, i);
                    var u = aa(l, s);
                    for (i = 0; i < a.length; i += 2) {
                        var c = a[i],
                            f = a[i + 1];
                        c === "style" ? lp(o, f) : c === "dangerouslySetInnerHTML" ? sp(o, f) : c === "children" ? $o(o, f) : hu(o, c, f, u)
                    }
                    switch (l) {
                        case "input":
                            ra(o, s);
                            break;
                        case "textarea":
                            rp(o, s);
                            break;
                        case "select":
                            var m = o._wrapperState.wasMultiple;
                            o._wrapperState.wasMultiple = !!s.multiple;
                            var d = s.value;
                            d != null ? _r(o, !!s.multiple, d, !1) : m !== !!s.multiple && (s.defaultValue != null ? _r(o, !!s.multiple, s.defaultValue, !0) : _r(o, !!s.multiple, s.multiple ? [] : "", !1))
                    }
                    o[Yo] = s
                } catch (y) { fe(e, e.return, y) }
            }
            break;
        case 6:
            if (pt(t, e), Tt(e), r & 4) {
                if (e.stateNode === null) throw Error(j(162));
                o = e.stateNode, s = e.memoizedProps;
                try { o.nodeValue = s } catch (y) { fe(e, e.return, y) }
            }
            break;
        case 3:
            if (pt(t, e), Tt(e), r & 4 && n !== null && n.memoizedState.isDehydrated) try { Wo(t.containerInfo) } catch (y) { fe(e, e.return, y) }
            break;
        case 4:
            pt(t, e), Tt(e);
            break;
        case 13:
            pt(t, e), Tt(e), o = e.child, o.flags & 8192 && (s = o.memoizedState !== null, o.stateNode.isHidden = s, !s || o.alternate !== null && o.alternate.memoizedState !== null || (Ku = pe())), r & 4 && Nd(e);
            break;
        case 22:
            if (c = n !== null && n.memoizedState !== null, e.mode & 1 ? (Ae = (u = Ae) || c, pt(t, e), Ae = u) : pt(t, e), Tt(e), r & 8192) {
                if (u = e.memoizedState !== null, (e.stateNode.isHidden = u) && !c && e.mode & 1)
                    for (I = e, c = e.child; c !== null;) {
                        for (f = I = c; I !== null;) {
                            switch (m = I, d = m.child, m.tag) {
                                case 0:
                                case 11:
                                case 14:
                                case 15:
                                    Mo(4, m, m.return);
                                    break;
                                case 1:
                                    Rr(m, m.return);
                                    var S = m.stateNode;
                                    if (typeof S.componentWillUnmount == "function") { r = m, n = m.return; try { t = r, S.props = t.memoizedProps, S.state = t.memoizedState, S.componentWillUnmount() } catch (y) { fe(r, n, y) } }
                                    break;
                                case 5:
                                    Rr(m, m.return);
                                    break;
                                case 22:
                                    if (m.memoizedState !== null) { jd(f); continue }
                            }
                            d !== null ? (d.return = m, I = d) : jd(f)
                        }
                        c = c.sibling
                    }
                e: for (c = null, f = e;;) {
                    if (f.tag === 5) { if (c === null) { c = f; try { o = f.stateNode, u ? (s = o.style, typeof s.setProperty == "function" ? s.setProperty("display", "none", "important") : s.display = "none") : (l = f.stateNode, a = f.memoizedProps.style, i = a != null && a.hasOwnProperty("display") ? a.display : null, l.style.display = ip("display", i)) } catch (y) { fe(e, e.return, y) } } } else if (f.tag === 6) { if (c === null) try { f.stateNode.nodeValue = u ? "" : f.memoizedProps } catch (y) { fe(e, e.return, y) } } else if ((f.tag !== 22 && f.tag !== 23 || f.memoizedState === null || f === e) && f.child !== null) { f.child.return = f, f = f.child; continue }
                    if (f === e) break e;
                    for (; f.sibling === null;) {
                        if (f.return === null || f.return === e) break e;
                        c === f && (c = null), f = f.return
                    }
                    c === f && (c = null), f.sibling.return = f.return, f = f.sibling
                }
            }
            break;
        case 19:
            pt(t, e), Tt(e), r & 4 && Nd(e);
            break;
        case 21:
            break;
        default:
            pt(t, e), Tt(e)
    }
}

function Tt(e) {
    var t = e.flags;
    if (t & 2) {
        try {
            e: {
                for (var n = e.return; n !== null;) {
                    if (Ah(n)) { var r = n; break e }
                    n = n.return
                }
                throw Error(j(160))
            }
            switch (r.tag) {
                case 5:
                    var o = r.stateNode;
                    r.flags & 32 && ($o(o, ""), r.flags &= -33);
                    var s = Td(e);
                    za(e, s, o);
                    break;
                case 3:
                case 4:
                    var i = r.stateNode.containerInfo,
                        l = Td(e);
                    Da(e, l, i);
                    break;
                default:
                    throw Error(j(161))
            }
        }
        catch (a) { fe(e, e.return, a) }
        e.flags &= -3
    }
    t & 4096 && (e.flags &= -4097)
}

function f0(e, t, n) { I = e, Mh(e) }

function Mh(e, t, n) {
    for (var r = (e.mode & 1) !== 0; I !== null;) {
        var o = I,
            s = o.child;
        if (o.tag === 22 && r) {
            var i = o.memoizedState !== null || Ls;
            if (!i) {
                var l = o.alternate,
                    a = l !== null && l.memoizedState !== null || Ae;
                l = Ls;
                var u = Ae;
                if (Ls = i, (Ae = a) && !u)
                    for (I = o; I !== null;) i = I, a = i.child, i.tag === 22 && i.memoizedState !== null ? _d(o) : a !== null ? (a.return = i, I = a) : _d(o);
                for (; s !== null;) I = s, Mh(s), s = s.sibling;
                I = o, Ls = l, Ae = u
            }
            Rd(e)
        } else o.subtreeFlags & 8772 && s !== null ? (s.return = o, I = s) : Rd(e)
    }
}

function Rd(e) {
    for (; I !== null;) {
        var t = I;
        if (t.flags & 8772) {
            var n = t.alternate;
            try {
                if (t.flags & 8772) switch (t.tag) {
                    case 0:
                    case 11:
                    case 15:
                        Ae || Hi(5, t);
                        break;
                    case 1:
                        var r = t.stateNode;
                        if (t.flags & 4 && !Ae)
                            if (n === null) r.componentDidMount();
                            else {
                                var o = t.elementType === t.type ? n.memoizedProps : mt(t.type, n.memoizedProps);
                                r.componentDidUpdate(o, n.memoizedState, r.__reactInternalSnapshotBeforeUpdate)
                            }
                        var s = t.updateQueue;
                        s !== null && pd(t, s, r);
                        break;
                    case 3:
                        var i = t.updateQueue;
                        if (i !== null) {
                            if (n = null, t.child !== null) switch (t.child.tag) {
                                case 5:
                                    n = t.child.stateNode;
                                    break;
                                case 1:
                                    n = t.child.stateNode
                            }
                            pd(t, i, n)
                        }
                        break;
                    case 5:
                        var l = t.stateNode;
                        if (n === null && t.flags & 4) {
                            n = l;
                            var a = t.memoizedProps;
                            switch (t.type) {
                                case "button":
                                case "input":
                                case "select":
                                case "textarea":
                                    a.autoFocus && n.focus();
                                    break;
                                case "img":
                                    a.src && (n.src = a.src)
                            }
                        }
                        break;
                    case 6:
                        break;
                    case 4:
                        break;
                    case 12:
                        break;
                    case 13:
                        if (t.memoizedState === null) {
                            var u = t.alternate;
                            if (u !== null) {
                                var c = u.memoizedState;
                                if (c !== null) {
                                    var f = c.dehydrated;
                                    f !== null && Wo(f)
                                }
                            }
                        }
                        break;
                    case 19:
                    case 17:
                    case 21:
                    case 22:
                    case 23:
                    case 25:
                        break;
                    default:
                        throw Error(j(163))
                }
                Ae || t.flags & 512 && Ia(t)
            } catch (m) { fe(t, t.return, m) }
        }
        if (t === e) { I = null; break }
        if (n = t.sibling, n !== null) { n.return = t.return, I = n; break }
        I = t.return
    }
}

function jd(e) {
    for (; I !== null;) {
        var t = I;
        if (t === e) { I = null; break }
        var n = t.sibling;
        if (n !== null) { n.return = t.return, I = n; break }
        I = t.return
    }
}

function _d(e) {
    for (; I !== null;) {
        var t = I;
        try {
            switch (t.tag) {
                case 0:
                case 11:
                case 15:
                    var n = t.return;
                    try { Hi(4, t) } catch (a) { fe(t, n, a) }
                    break;
                case 1:
                    var r = t.stateNode;
                    if (typeof r.componentDidMount == "function") { var o = t.return; try { r.componentDidMount() } catch (a) { fe(t, o, a) } }
                    var s = t.return;
                    try { Ia(t) } catch (a) { fe(t, s, a) }
                    break;
                case 5:
                    var i = t.return;
                    try { Ia(t) } catch (a) { fe(t, i, a) }
            }
        } catch (a) { fe(t, t.return, a) }
        if (t === e) { I = null; break }
        var l = t.sibling;
        if (l !== null) { l.return = t.return, I = l; break }
        I = t.return
    }
}
var p0 = Math.ceil,
    Ci = en.ReactCurrentDispatcher,
    Hu = en.ReactCurrentOwner,
    lt = en.ReactCurrentBatchConfig,
    Z = 0,
    Se = null,
    me = null,
    ke = 0,
    Ke = 0,
    jr = Fn(0),
    ye = 0,
    ts = null,
    lr = 0,
    Qi = 0,
    Qu = 0,
    Io = null,
    $e = null,
    Ku = 0,
    no = 1 / 0,
    $t = null,
    ki = !1,
    Fa = null,
    _n = null,
    Ms = !1,
    kn = null,
    Pi = 0,
    Do = 0,
    $a = null,
    Zs = -1,
    Js = 0;

function Ie() { return Z & 6 ? pe() : Zs !== -1 ? Zs : Zs = pe() }

function An(e) { return e.mode & 1 ? Z & 2 && ke !== 0 ? ke & -ke : Xy.transition !== null ? (Js === 0 && (Js = wp()), Js) : (e = ee, e !== 0 || (e = window.event, e = e === void 0 ? 16 : bp(e.type)), e) : 1 }

function St(e, t, n, r) {
    if (50 < Do) throw Do = 0, $a = null, Error(j(185));
    us(e, n, r), (!(Z & 2) || e !== Se) && (e === Se && (!(Z & 2) && (Qi |= n), ye === 4 && gn(e, ke)), We(e, r), n === 1 && Z === 0 && !(t.mode & 1) && (no = pe() + 500, Ui && $n()))
}

function We(e, t) {
    var n = e.callbackNode;
    Xv(e, t);
    var r = ai(e, e === Se ? ke : 0);
    if (r === 0) n !== null && $c(n), e.callbackNode = null, e.callbackPriority = 0;
    else if (t = r & -r, e.callbackPriority !== t) {
        if (n != null && $c(n), t === 1) e.tag === 0 ? Yy(Ad.bind(null, e)) : Hp(Ad.bind(null, e)), Hy(function() {!(Z & 6) && $n() }), n = null;
        else {
            switch (xp(r)) {
                case 1:
                    n = wu;
                    break;
                case 4:
                    n = vp;
                    break;
                case 16:
                    n = li;
                    break;
                case 536870912:
                    n = yp;
                    break;
                default:
                    n = li
            }
            n = Vh(n, Ih.bind(null, e))
        }
        e.callbackPriority = t, e.callbackNode = n
    }
}

function Ih(e, t) {
    if (Zs = -1, Js = 0, Z & 6) throw Error(j(327));
    var n = e.callbackNode;
    if (Ir() && e.callbackNode !== n) return null;
    var r = ai(e, e === Se ? ke : 0);
    if (r === 0) return null;
    if (r & 30 || r & e.expiredLanes || t) t = bi(e, r);
    else {
        t = r;
        var o = Z;
        Z |= 2;
        var s = zh();
        (Se !== e || ke !== t) && ($t = null, no = pe() + 500, nr(e, t));
        do try { g0(); break } catch (l) { Dh(e, l) }
        while (!0);
        Au(), Ci.current = s, Z = o, me !== null ? t = 0 : (Se = null, ke = 0, t = ye)
    }
    if (t !== 0) {
        if (t === 2 && (o = pa(e), o !== 0 && (r = o, t = Ba(e, o))), t === 1) throw n = ts, nr(e, 0), gn(e, r), We(e, pe()), n;
        if (t === 6) gn(e, r);
        else {
            if (o = e.current.alternate, !(r & 30) && !h0(o) && (t = bi(e, r), t === 2 && (s = pa(e), s !== 0 && (r = s, t = Ba(e, s))), t === 1)) throw n = ts, nr(e, 0), gn(e, r), We(e, pe()), n;
            switch (e.finishedWork = o, e.finishedLanes = r, t) {
                case 0:
                case 1:
                    throw Error(j(345));
                case 2:
                    Qn(e, $e, $t);
                    break;
                case 3:
                    if (gn(e, r), (r & 130023424) === r && (t = Ku + 500 - pe(), 10 < t)) {
                        if (ai(e, 0) !== 0) break;
                        if (o = e.suspendedLanes, (o & r) !== r) { Ie(), e.pingedLanes |= e.suspendedLanes & o; break }
                        e.timeoutHandle = Sa(Qn.bind(null, e, $e, $t), t);
                        break
                    }
                    Qn(e, $e, $t);
                    break;
                case 4:
                    if (gn(e, r), (r & 4194240) === r) break;
                    for (t = e.eventTimes, o = -1; 0 < r;) {
                        var i = 31 - xt(r);
                        s = 1 << i, i = t[i], i > o && (o = i), r &= ~s
                    }
                    if (r = o, r = pe() - r, r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * p0(r / 1960)) - r, 10 < r) { e.timeoutHandle = Sa(Qn.bind(null, e, $e, $t), r); break }
                    Qn(e, $e, $t);
                    break;
                case 5:
                    Qn(e, $e, $t);
                    break;
                default:
                    throw Error(j(329))
            }
        }
    }
    return We(e, pe()), e.callbackNode === n ? Ih.bind(null, e) : null
}

function Ba(e, t) { var n = Io; return e.current.memoizedState.isDehydrated && (nr(e, t).flags |= 256), e = bi(e, t), e !== 2 && (t = $e, $e = n, t !== null && Ua(t)), e }

function Ua(e) { $e === null ? $e = e : $e.push.apply($e, e) }

function h0(e) {
    for (var t = e;;) {
        if (t.flags & 16384) {
            var n = t.updateQueue;
            if (n !== null && (n = n.stores, n !== null))
                for (var r = 0; r < n.length; r++) {
                    var o = n[r],
                        s = o.getSnapshot;
                    o = o.value;
                    try { if (!Et(s(), o)) return !1 } catch { return !1 }
                }
        }
        if (n = t.child, t.subtreeFlags & 16384 && n !== null) n.return = t, t = n;
        else {
            if (t === e) break;
            for (; t.sibling === null;) {
                if (t.return === null || t.return === e) return !0;
                t = t.return
            }
            t.sibling.return = t.return, t = t.sibling
        }
    }
    return !0
}

function gn(e, t) {
    for (t &= ~Qu, t &= ~Qi, e.suspendedLanes |= t, e.pingedLanes &= ~t, e = e.expirationTimes; 0 < t;) {
        var n = 31 - xt(t),
            r = 1 << n;
        e[n] = -1, t &= ~r
    }
}

function Ad(e) {
    if (Z & 6) throw Error(j(327));
    Ir();
    var t = ai(e, 0);
    if (!(t & 1)) return We(e, pe()), null;
    var n = bi(e, t);
    if (e.tag !== 0 && n === 2) {
        var r = pa(e);
        r !== 0 && (t = r, n = Ba(e, r))
    }
    if (n === 1) throw n = ts, nr(e, 0), gn(e, t), We(e, pe()), n;
    if (n === 6) throw Error(j(345));
    return e.finishedWork = e.current.alternate, e.finishedLanes = t, Qn(e, $e, $t), We(e, pe()), null
}

function Gu(e, t) {
    var n = Z;
    Z |= 1;
    try { return e(t) } finally { Z = n, Z === 0 && (no = pe() + 500, Ui && $n()) }
}

function ar(e) {
    kn !== null && kn.tag === 0 && !(Z & 6) && Ir();
    var t = Z;
    Z |= 1;
    var n = lt.transition,
        r = ee;
    try { if (lt.transition = null, ee = 1, e) return e() } finally { ee = r, lt.transition = n, Z = t, !(Z & 6) && $n() }
}

function Yu() { Ke = jr.current, se(jr) }

function nr(e, t) {
    e.finishedWork = null, e.finishedLanes = 0;
    var n = e.timeoutHandle;
    if (n !== -1 && (e.timeoutHandle = -1, Wy(n)), me !== null)
        for (n = me.return; n !== null;) {
            var r = n;
            switch (Ru(r), r.tag) {
                case 1:
                    r = r.type.childContextTypes, r != null && pi();
                    break;
                case 3:
                    eo(), se(Ue), se(Oe), zu();
                    break;
                case 5:
                    Du(r);
                    break;
                case 4:
                    eo();
                    break;
                case 13:
                    se(ae);
                    break;
                case 19:
                    se(ae);
                    break;
                case 10:
                    Ou(r.type._context);
                    break;
                case 22:
                case 23:
                    Yu()
            }
            n = n.return
        }
    if (Se = e, me = e = On(e.current, null), ke = Ke = t, ye = 0, ts = null, Qu = Qi = lr = 0, $e = Io = null, Gn !== null) {
        for (t = 0; t < Gn.length; t++)
            if (n = Gn[t], r = n.interleaved, r !== null) {
                n.interleaved = null;
                var o = r.next,
                    s = n.pending;
                if (s !== null) {
                    var i = s.next;
                    s.next = o, r.next = i
                }
                n.pending = r
            }
        Gn = null
    }
    return e
}

function Dh(e, t) {
    do {
        var n = me;
        try {
            if (Au(), Ys.current = Ei, Si) {
                for (var r = ue.memoizedState; r !== null;) {
                    var o = r.queue;
                    o !== null && (o.pending = null), r = r.next
                }
                Si = !1
            }
            if (ir = 0, xe = ge = ue = null, Lo = !1, Zo = 0, Hu.current = null, n === null || n.return === null) { ye = 1, ts = t, me = null; break }
            e: {
                var s = e,
                    i = n.return,
                    l = n,
                    a = t;
                if (t = ke, l.flags |= 32768, a !== null && typeof a == "object" && typeof a.then == "function") {
                    var u = a,
                        c = l,
                        f = c.tag;
                    if (!(c.mode & 1) && (f === 0 || f === 11 || f === 15)) {
                        var m = c.alternate;
                        m ? (c.updateQueue = m.updateQueue, c.memoizedState = m.memoizedState, c.lanes = m.lanes) : (c.updateQueue = null, c.memoizedState = null)
                    }
                    var d = wd(i);
                    if (d !== null) {
                        d.flags &= -257, xd(d, i, l, s, t), d.mode & 1 && yd(s, u, t), t = d, a = u;
                        var S = t.updateQueue;
                        if (S === null) {
                            var y = new Set;
                            y.add(a), t.updateQueue = y
                        } else S.add(a);
                        break e
                    } else {
                        if (!(t & 1)) { yd(s, u, t), Xu(); break e }
                        a = Error(j(426))
                    }
                } else if (le && l.mode & 1) { var x = wd(i); if (x !== null) {!(x.flags & 65536) && (x.flags |= 256), xd(x, i, l, s, t), ju(to(a, l)); break e } }
                s = a = to(a, l),
                ye !== 4 && (ye = 2),
                Io === null ? Io = [s] : Io.push(s),
                s = i;do {
                    switch (s.tag) {
                        case 3:
                            s.flags |= 65536, t &= -t, s.lanes |= t;
                            var h = xh(s, a, t);
                            fd(s, h);
                            break e;
                        case 1:
                            l = a;
                            var p = s.type,
                                g = s.stateNode;
                            if (!(s.flags & 128) && (typeof p.getDerivedStateFromError == "function" || g !== null && typeof g.componentDidCatch == "function" && (_n === null || !_n.has(g)))) {
                                s.flags |= 65536, t &= -t, s.lanes |= t;
                                var E = Sh(s, l, t);
                                fd(s, E);
                                break e
                            }
                    }
                    s = s.return
                } while (s !== null)
            }
            $h(n)
        } catch (C) { t = C, me === n && n !== null && (me = n = n.return); continue }
        break
    } while (!0)
}

function zh() { var e = Ci.current; return Ci.current = Ei, e === null ? Ei : e }

function Xu() {
    (ye === 0 || ye === 3 || ye === 2) && (ye = 4), Se === null || !(lr & 268435455) && !(Qi & 268435455) || gn(Se, ke)
}

function bi(e, t) {
    var n = Z;
    Z |= 2;
    var r = zh();
    (Se !== e || ke !== t) && ($t = null, nr(e, t));
    do try { m0(); break } catch (o) { Dh(e, o) }
    while (!0);
    if (Au(), Z = n, Ci.current = r, me !== null) throw Error(j(261));
    return Se = null, ke = 0, ye
}

function m0() { for (; me !== null;) Fh(me) }

function g0() { for (; me !== null && !Bv();) Fh(me) }

function Fh(e) {
    var t = Uh(e.alternate, e, Ke);
    e.memoizedProps = e.pendingProps, t === null ? $h(e) : me = t, Hu.current = null
}

function $h(e) {
    var t = e;
    do {
        var n = t.alternate;
        if (e = t.return, t.flags & 32768) {
            if (n = u0(n, t), n !== null) { n.flags &= 32767, me = n; return }
            if (e !== null) e.flags |= 32768, e.subtreeFlags = 0, e.deletions = null;
            else { ye = 6, me = null; return }
        } else if (n = a0(n, t, Ke), n !== null) { me = n; return }
        if (t = t.sibling, t !== null) { me = t; return }
        me = t = e
    } while (t !== null);
    ye === 0 && (ye = 5)
}

function Qn(e, t, n) {
    var r = ee,
        o = lt.transition;
    try { lt.transition = null, ee = 1, v0(e, t, n, r) } finally { lt.transition = o, ee = r }
    return null
}

function v0(e, t, n, r) {
    do Ir(); while (kn !== null);
    if (Z & 6) throw Error(j(327));
    n = e.finishedWork;
    var o = e.finishedLanes;
    if (n === null) return null;
    if (e.finishedWork = null, e.finishedLanes = 0, n === e.current) throw Error(j(177));
    e.callbackNode = null, e.callbackPriority = 0;
    var s = n.lanes | n.childLanes;
    if (qv(e, s), e === Se && (me = Se = null, ke = 0), !(n.subtreeFlags & 2064) && !(n.flags & 2064) || Ms || (Ms = !0, Vh(li, function() { return Ir(), null })), s = (n.flags & 15990) !== 0, n.subtreeFlags & 15990 || s) {
        s = lt.transition, lt.transition = null;
        var i = ee;
        ee = 1;
        var l = Z;
        Z |= 4, Hu.current = null, d0(e, n), Lh(n, e), Dy(wa), ui = !!ya, wa = ya = null, e.current = n, f0(n), Uv(), Z = l, ee = i, lt.transition = s
    } else e.current = n;
    if (Ms && (Ms = !1, kn = e, Pi = o), s = e.pendingLanes, s === 0 && (_n = null), Hv(n.stateNode), We(e, pe()), t !== null)
        for (r = e.onRecoverableError, n = 0; n < t.length; n++) o = t[n], r(o.value, { componentStack: o.stack, digest: o.digest });
    if (ki) throw ki = !1, e = Fa, Fa = null, e;
    return Pi & 1 && e.tag !== 0 && Ir(), s = e.pendingLanes, s & 1 ? e === $a ? Do++ : (Do = 0, $a = e) : Do = 0, $n(), null
}

function Ir() {
    if (kn !== null) {
        var e = xp(Pi),
            t = lt.transition,
            n = ee;
        try {
            if (lt.transition = null, ee = 16 > e ? 16 : e, kn === null) var r = !1;
            else {
                if (e = kn, kn = null, Pi = 0, Z & 6) throw Error(j(331));
                var o = Z;
                for (Z |= 4, I = e.current; I !== null;) {
                    var s = I,
                        i = s.child;
                    if (I.flags & 16) {
                        var l = s.deletions;
                        if (l !== null) {
                            for (var a = 0; a < l.length; a++) {
                                var u = l[a];
                                for (I = u; I !== null;) {
                                    var c = I;
                                    switch (c.tag) {
                                        case 0:
                                        case 11:
                                        case 15:
                                            Mo(8, c, s)
                                    }
                                    var f = c.child;
                                    if (f !== null) f.return = c, I = f;
                                    else
                                        for (; I !== null;) {
                                            c = I;
                                            var m = c.sibling,
                                                d = c.return;
                                            if (_h(c), c === u) { I = null; break }
                                            if (m !== null) { m.return = d, I = m; break }
                                            I = d
                                        }
                                }
                            }
                            var S = s.alternate;
                            if (S !== null) {
                                var y = S.child;
                                if (y !== null) {
                                    S.child = null;
                                    do {
                                        var x = y.sibling;
                                        y.sibling = null, y = x
                                    } while (y !== null)
                                }
                            }
                            I = s
                        }
                    }
                    if (s.subtreeFlags & 2064 && i !== null) i.return = s, I = i;
                    else e: for (; I !== null;) {
                        if (s = I, s.flags & 2048) switch (s.tag) {
                            case 0:
                            case 11:
                            case 15:
                                Mo(9, s, s.return)
                        }
                        var h = s.sibling;
                        if (h !== null) { h.return = s.return, I = h; break e }
                        I = s.return
                    }
                }
                var p = e.current;
                for (I = p; I !== null;) {
                    i = I;
                    var g = i.child;
                    if (i.subtreeFlags & 2064 && g !== null) g.return = i, I = g;
                    else e: for (i = p; I !== null;) {
                        if (l = I, l.flags & 2048) try {
                            switch (l.tag) {
                                case 0:
                                case 11:
                                case 15:
                                    Hi(9, l)
                            }
                        } catch (C) { fe(l, l.return, C) }
                        if (l === i) { I = null; break e }
                        var E = l.sibling;
                        if (E !== null) { E.return = l.return, I = E; break e }
                        I = l.return
                    }
                }
                if (Z = o, $n(), Lt && typeof Lt.onPostCommitFiberRoot == "function") try { Lt.onPostCommitFiberRoot(Di, e) } catch {}
                r = !0
            }
            return r
        } finally { ee = n, lt.transition = t }
    }
    return !1
}

function Od(e, t, n) { t = to(n, t), t = xh(e, t, 1), e = jn(e, t, 1), t = Ie(), e !== null && (us(e, 1, t), We(e, t)) }

function fe(e, t, n) {
    if (e.tag === 3) Od(e, e, n);
    else
        for (; t !== null;) {
            if (t.tag === 3) { Od(t, e, n); break } else if (t.tag === 1) { var r = t.stateNode; if (typeof t.type.getDerivedStateFromError == "function" || typeof r.componentDidCatch == "function" && (_n === null || !_n.has(r))) { e = to(n, e), e = Sh(t, e, 1), t = jn(t, e, 1), e = Ie(), t !== null && (us(t, 1, e), We(t, e)); break } }
            t = t.return
        }
}

function y0(e, t, n) {
    var r = e.pingCache;
    r !== null && r.delete(t), t = Ie(), e.pingedLanes |= e.suspendedLanes & n, Se === e && (ke & n) === n && (ye === 4 || ye === 3 && (ke & 130023424) === ke && 500 > pe() - Ku ? nr(e, 0) : Qu |= n), We(e, t)
}

function Bh(e, t) {
    t === 0 && (e.mode & 1 ? (t = Ps, Ps <<= 1, !(Ps & 130023424) && (Ps = 4194304)) : t = 1);
    var n = Ie();
    e = Yt(e, t), e !== null && (us(e, t, n), We(e, n))
}

function w0(e) {
    var t = e.memoizedState,
        n = 0;
    t !== null && (n = t.retryLane), Bh(e, n)
}

function x0(e, t) {
    var n = 0;
    switch (e.tag) {
        case 13:
            var r = e.stateNode,
                o = e.memoizedState;
            o !== null && (n = o.retryLane);
            break;
        case 19:
            r = e.stateNode;
            break;
        default:
            throw Error(j(314))
    }
    r !== null && r.delete(t), Bh(e, n)
}
var Uh;
Uh = function(e, t, n) {
    if (e !== null)
        if (e.memoizedProps !== t.pendingProps || Ue.current) Be = !0;
        else {
            if (!(e.lanes & n) && !(t.flags & 128)) return Be = !1, l0(e, t, n);
            Be = !!(e.flags & 131072)
        }
    else Be = !1, le && t.flags & 1048576 && Qp(t, gi, t.index);
    switch (t.lanes = 0, t.tag) {
        case 2:
            var r = t.type;
            qs(e, t), e = t.pendingProps;
            var o = qr(t, Oe.current);
            Mr(t, n), o = $u(null, t, r, e, o, n);
            var s = Bu();
            return t.flags |= 1, typeof o == "object" && o !== null && typeof o.render == "function" && o.$$typeof === void 0 ? (t.tag = 1, t.memoizedState = null, t.updateQueue = null, Ve(r) ? (s = !0, hi(t)) : s = !1, t.memoizedState = o.state !== null && o.state !== void 0 ? o.state : null, Mu(t), o.updater = Wi, t.stateNode = o, o._reactInternals = t, Na(t, r, e, n), t = _a(null, t, r, !0, s, n)) : (t.tag = 0, le && s && Nu(t), Le(null, t, o, n), t = t.child), t;
        case 16:
            r = t.elementType;
            e: {
                switch (qs(e, t), e = t.pendingProps, o = r._init, r = o(r._payload), t.type = r, o = t.tag = E0(r), e = mt(r, e), o) {
                    case 0:
                        t = ja(null, t, r, e, n);
                        break e;
                    case 1:
                        t = Cd(null, t, r, e, n);
                        break e;
                    case 11:
                        t = Sd(null, t, r, e, n);
                        break e;
                    case 14:
                        t = Ed(null, t, r, mt(r.type, e), n);
                        break e
                }
                throw Error(j(306, r, ""))
            }
            return t;
        case 0:
            return r = t.type, o = t.pendingProps, o = t.elementType === r ? o : mt(r, o), ja(e, t, r, o, n);
        case 1:
            return r = t.type, o = t.pendingProps, o = t.elementType === r ? o : mt(r, o), Cd(e, t, r, o, n);
        case 3:
            e: {
                if (Ph(t), e === null) throw Error(j(387));r = t.pendingProps,
                s = t.memoizedState,
                o = s.element,
                Zp(e, t),
                wi(t, r, null, n);
                var i = t.memoizedState;
                if (r = i.element, s.isDehydrated)
                    if (s = { element: r, isDehydrated: !1, cache: i.cache, pendingSuspenseBoundaries: i.pendingSuspenseBoundaries, transitions: i.transitions }, t.updateQueue.baseState = s, t.memoizedState = s, t.flags & 256) { o = to(Error(j(423)), t), t = kd(e, t, r, n, o); break e } else if (r !== o) { o = to(Error(j(424)), t), t = kd(e, t, r, n, o); break e } else
                    for (Xe = Rn(t.stateNode.containerInfo.firstChild), qe = t, le = !0, wt = null, n = Xp(t, null, r, n), t.child = n; n;) n.flags = n.flags & -3 | 4096, n = n.sibling;
                else {
                    if (Zr(), r === o) { t = Xt(e, t, n); break e }
                    Le(e, t, r, n)
                }
                t = t.child
            }
            return t;
        case 5:
            return Jp(t), e === null && Pa(t), r = t.type, o = t.pendingProps, s = e !== null ? e.memoizedProps : null, i = o.children, xa(r, o) ? i = null : s !== null && xa(r, s) && (t.flags |= 32), kh(e, t), Le(e, t, i, n), t.child;
        case 6:
            return e === null && Pa(t), null;
        case 13:
            return bh(e, t, n);
        case 4:
            return Iu(t, t.stateNode.containerInfo), r = t.pendingProps, e === null ? t.child = Jr(t, null, r, n) : Le(e, t, r, n), t.child;
        case 11:
            return r = t.type, o = t.pendingProps, o = t.elementType === r ? o : mt(r, o), Sd(e, t, r, o, n);
        case 7:
            return Le(e, t, t.pendingProps, n), t.child;
        case 8:
            return Le(e, t, t.pendingProps.children, n), t.child;
        case 12:
            return Le(e, t, t.pendingProps.children, n), t.child;
        case 10:
            e: {
                if (r = t.type._context, o = t.pendingProps, s = t.memoizedProps, i = o.value, ne(vi, r._currentValue), r._currentValue = i, s !== null)
                    if (Et(s.value, i)) { if (s.children === o.children && !Ue.current) { t = Xt(e, t, n); break e } } else
                        for (s = t.child, s !== null && (s.return = t); s !== null;) {
                            var l = s.dependencies;
                            if (l !== null) {
                                i = s.child;
                                for (var a = l.firstContext; a !== null;) {
                                    if (a.context === r) {
                                        if (s.tag === 1) {
                                            a = Qt(-1, n & -n), a.tag = 2;
                                            var u = s.updateQueue;
                                            if (u !== null) {
                                                u = u.shared;
                                                var c = u.pending;
                                                c === null ? a.next = a : (a.next = c.next, c.next = a), u.pending = a
                                            }
                                        }
                                        s.lanes |= n, a = s.alternate, a !== null && (a.lanes |= n), ba(s.return, n, t), l.lanes |= n;
                                        break
                                    }
                                    a = a.next
                                }
                            } else if (s.tag === 10) i = s.type === t.type ? null : s.child;
                            else if (s.tag === 18) {
                                if (i = s.return, i === null) throw Error(j(341));
                                i.lanes |= n, l = i.alternate, l !== null && (l.lanes |= n), ba(i, n, t), i = s.sibling
                            } else i = s.child;
                            if (i !== null) i.return = s;
                            else
                                for (i = s; i !== null;) {
                                    if (i === t) { i = null; break }
                                    if (s = i.sibling, s !== null) { s.return = i.return, i = s; break }
                                    i = i.return
                                }
                            s = i
                        }
                Le(e, t, o.children, n),
                t = t.child
            }
            return t;
        case 9:
            return o = t.type, r = t.pendingProps.children, Mr(t, n), o = at(o), r = r(o), t.flags |= 1, Le(e, t, r, n), t.child;
        case 14:
            return r = t.type, o = mt(r, t.pendingProps), o = mt(r.type, o), Ed(e, t, r, o, n);
        case 15:
            return Eh(e, t, t.type, t.pendingProps, n);
        case 17:
            return r = t.type, o = t.pendingProps, o = t.elementType === r ? o : mt(r, o), qs(e, t), t.tag = 1, Ve(r) ? (e = !0, hi(t)) : e = !1, Mr(t, n), wh(t, r, o), Na(t, r, o, n), _a(null, t, r, !0, e, n);
        case 19:
            return Th(e, t, n);
        case 22:
            return Ch(e, t, n)
    }
    throw Error(j(156, t.tag))
};

function Vh(e, t) { return gp(e, t) }

function S0(e, t, n, r) { this.tag = e, this.key = n, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = r, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null }

function it(e, t, n, r) { return new S0(e, t, n, r) }

function qu(e) { return e = e.prototype, !(!e || !e.isReactComponent) }

function E0(e) { if (typeof e == "function") return qu(e) ? 1 : 0; if (e != null) { if (e = e.$$typeof, e === gu) return 11; if (e === vu) return 14 } return 2 }

function On(e, t) { var n = e.alternate; return n === null ? (n = it(e.tag, t, e.key, e.mode), n.elementType = e.elementType, n.type = e.type, n.stateNode = e.stateNode, n.alternate = e, e.alternate = n) : (n.pendingProps = t, n.type = e.type, n.flags = 0, n.subtreeFlags = 0, n.deletions = null), n.flags = e.flags & 14680064, n.childLanes = e.childLanes, n.lanes = e.lanes, n.child = e.child, n.memoizedProps = e.memoizedProps, n.memoizedState = e.memoizedState, n.updateQueue = e.updateQueue, t = e.dependencies, n.dependencies = t === null ? null : { lanes: t.lanes, firstContext: t.firstContext }, n.sibling = e.sibling, n.index = e.index, n.ref = e.ref, n }

function ei(e, t, n, r, o, s) {
    var i = 2;
    if (r = e, typeof e == "function") qu(e) && (i = 1);
    else if (typeof e == "string") i = 5;
    else e: switch (e) {
        case xr:
            return rr(n.children, o, s, t);
        case mu:
            i = 8, o |= 8;
            break;
        case Zl:
            return e = it(12, n, t, o | 2), e.elementType = Zl, e.lanes = s, e;
        case Jl:
            return e = it(13, n, t, o), e.elementType = Jl, e.lanes = s, e;
        case ea:
            return e = it(19, n, t, o), e.elementType = ea, e.lanes = s, e;
        case Jf:
            return Ki(n, o, s, t);
        default:
            if (typeof e == "object" && e !== null) switch (e.$$typeof) {
                case qf:
                    i = 10;
                    break e;
                case Zf:
                    i = 9;
                    break e;
                case gu:
                    i = 11;
                    break e;
                case vu:
                    i = 14;
                    break e;
                case pn:
                    i = 16, r = null;
                    break e
            }
            throw Error(j(130, e == null ? e : typeof e, ""))
    }
    return t = it(i, n, t, o), t.elementType = e, t.type = r, t.lanes = s, t
}

function rr(e, t, n, r) { return e = it(7, e, r, t), e.lanes = n, e }

function Ki(e, t, n, r) { return e = it(22, e, r, t), e.elementType = Jf, e.lanes = n, e.stateNode = { isHidden: !1 }, e }

function Bl(e, t, n) { return e = it(6, e, null, t), e.lanes = n, e }

function Ul(e, t, n) { return t = it(4, e.children !== null ? e.children : [], e.key, t), t.lanes = n, t.stateNode = { containerInfo: e.containerInfo, pendingChildren: null, implementation: e.implementation }, t }

function C0(e, t, n, r, o) { this.tag = t, this.containerInfo = e, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = El(0), this.expirationTimes = El(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = El(0), this.identifierPrefix = r, this.onRecoverableError = o, this.mutableSourceEagerHydrationData = null }

function Zu(e, t, n, r, o, s, i, l, a) { return e = new C0(e, t, n, l, a), t === 1 ? (t = 1, s === !0 && (t |= 8)) : t = 0, s = it(3, null, null, t), e.current = s, s.stateNode = e, s.memoizedState = { element: r, isDehydrated: n, cache: null, transitions: null, pendingSuspenseBoundaries: null }, Mu(s), e }

function k0(e, t, n) { var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null; return { $$typeof: wr, key: r == null ? null : "" + r, children: e, containerInfo: t, implementation: n } }

function Wh(e) {
    if (!e) return Mn;
    e = e._reactInternals;
    e: {
        if (dr(e) !== e || e.tag !== 1) throw Error(j(170));
        var t = e;do {
            switch (t.tag) {
                case 3:
                    t = t.stateNode.context;
                    break e;
                case 1:
                    if (Ve(t.type)) { t = t.stateNode.__reactInternalMemoizedMergedChildContext; break e }
            }
            t = t.return
        } while (t !== null);
        throw Error(j(171))
    }
    if (e.tag === 1) { var n = e.type; if (Ve(n)) return Wp(e, n, t) }
    return t
}

function Hh(e, t, n, r, o, s, i, l, a) { return e = Zu(n, r, !0, e, o, s, i, l, a), e.context = Wh(null), n = e.current, r = Ie(), o = An(n), s = Qt(r, o), s.callback = t ? ? null, jn(n, s, o), e.current.lanes = o, us(e, o, r), We(e, r), e }

function Gi(e, t, n, r) {
    var o = t.current,
        s = Ie(),
        i = An(o);
    return n = Wh(n), t.context === null ? t.context = n : t.pendingContext = n, t = Qt(s, i), t.payload = { element: e }, r = r === void 0 ? null : r, r !== null && (t.callback = r), e = jn(o, t, i), e !== null && (St(e, o, i, s), Gs(e, o, i)), i
}

function Ti(e) {
    if (e = e.current, !e.child) return null;
    switch (e.child.tag) {
        case 5:
            return e.child.stateNode;
        default:
            return e.child.stateNode
    }
}

function Ld(e, t) {
    if (e = e.memoizedState, e !== null && e.dehydrated !== null) {
        var n = e.retryLane;
        e.retryLane = n !== 0 && n < t ? n : t
    }
}

function Ju(e, t) { Ld(e, t), (e = e.alternate) && Ld(e, t) }

function P0() { return null }
var Qh = typeof reportError == "function" ? reportError : function(e) { console.error(e) };

function ec(e) { this._internalRoot = e }
Yi.prototype.render = ec.prototype.render = function(e) {
    var t = this._internalRoot;
    if (t === null) throw Error(j(409));
    Gi(e, t, null, null)
};
Yi.prototype.unmount = ec.prototype.unmount = function() {
    var e = this._internalRoot;
    if (e !== null) {
        this._internalRoot = null;
        var t = e.containerInfo;
        ar(function() { Gi(null, e, null, null) }), t[Gt] = null
    }
};

function Yi(e) { this._internalRoot = e }
Yi.prototype.unstable_scheduleHydration = function(e) {
    if (e) {
        var t = Cp();
        e = { blockedOn: null, target: e, priority: t };
        for (var n = 0; n < mn.length && t !== 0 && t < mn[n].priority; n++);
        mn.splice(n, 0, e), n === 0 && Pp(e)
    }
};

function tc(e) { return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11) }

function Xi(e) { return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11 && (e.nodeType !== 8 || e.nodeValue !== " react-mount-point-unstable ")) }

function Md() {}

function b0(e, t, n, r, o) {
    if (o) {
        if (typeof r == "function") {
            var s = r;
            r = function() {
                var u = Ti(i);
                s.call(u)
            }
        }
        var i = Hh(t, r, e, 0, null, !1, !1, "", Md);
        return e._reactRootContainer = i, e[Gt] = i.current, Ko(e.nodeType === 8 ? e.parentNode : e), ar(), i
    }
    for (; o = e.lastChild;) e.removeChild(o);
    if (typeof r == "function") {
        var l = r;
        r = function() {
            var u = Ti(a);
            l.call(u)
        }
    }
    var a = Zu(e, 0, !1, null, null, !1, !1, "", Md);
    return e._reactRootContainer = a, e[Gt] = a.current, Ko(e.nodeType === 8 ? e.parentNode : e), ar(function() { Gi(t, a, n, r) }), a
}

function qi(e, t, n, r, o) {
    var s = n._reactRootContainer;
    if (s) {
        var i = s;
        if (typeof o == "function") {
            var l = o;
            o = function() {
                var a = Ti(i);
                l.call(a)
            }
        }
        Gi(t, i, e, o)
    } else i = b0(n, t, e, o, r);
    return Ti(i)
}
Sp = function(e) {
    switch (e.tag) {
        case 3:
            var t = e.stateNode;
            if (t.current.memoizedState.isDehydrated) {
                var n = To(t.pendingLanes);
                n !== 0 && (xu(t, n | 1), We(t, pe()), !(Z & 6) && (no = pe() + 500, $n()))
            }
            break;
        case 13:
            ar(function() {
                var r = Yt(e, 1);
                if (r !== null) {
                    var o = Ie();
                    St(r, e, 1, o)
                }
            }), Ju(e, 1)
    }
};
Su = function(e) {
    if (e.tag === 13) {
        var t = Yt(e, 134217728);
        if (t !== null) {
            var n = Ie();
            St(t, e, 134217728, n)
        }
        Ju(e, 134217728)
    }
};
Ep = function(e) {
    if (e.tag === 13) {
        var t = An(e),
            n = Yt(e, t);
        if (n !== null) {
            var r = Ie();
            St(n, e, t, r)
        }
        Ju(e, t)
    }
};
Cp = function() { return ee };
kp = function(e, t) { var n = ee; try { return ee = e, t() } finally { ee = n } };
ca = function(e, t, n) {
    switch (t) {
        case "input":
            if (ra(e, n), t = n.name, n.type === "radio" && t != null) {
                for (n = e; n.parentNode;) n = n.parentNode;
                for (n = n.querySelectorAll("input[name=" + JSON.stringify("" + t) + '][type="radio"]'), t = 0; t < n.length; t++) {
                    var r = n[t];
                    if (r !== e && r.form === e.form) {
                        var o = Bi(r);
                        if (!o) throw Error(j(90));
                        tp(r), ra(r, o)
                    }
                }
            }
            break;
        case "textarea":
            rp(e, n);
            break;
        case "select":
            t = n.value, t != null && _r(e, !!n.multiple, t, !1)
    }
};
cp = Gu;
dp = ar;
var T0 = { usingClientEntryPoint: !1, Events: [ds, kr, Bi, ap, up, Gu] },
    Eo = { findFiberByHostInstance: Kn, bundleType: 0, version: "18.3.1", rendererPackageName: "react-dom" },
    N0 = { bundleType: Eo.bundleType, version: Eo.version, rendererPackageName: Eo.rendererPackageName, rendererConfig: Eo.rendererConfig, overrideHookState: null, overrideHookStateDeletePath: null, overrideHookStateRenamePath: null, overrideProps: null, overridePropsDeletePath: null, overridePropsRenamePath: null, setErrorHandler: null, setSuspenseHandler: null, scheduleUpdate: null, currentDispatcherRef: en.ReactCurrentDispatcher, findHostInstanceByFiber: function(e) { return e = hp(e), e === null ? null : e.stateNode }, findFiberByHostInstance: Eo.findFiberByHostInstance || P0, findHostInstancesForRefresh: null, scheduleRefresh: null, scheduleRoot: null, setRefreshHandler: null, getCurrentFiber: null, reconcilerVersion: "18.3.1-next-f1338f8080-20240426" };
if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") { var Is = __REACT_DEVTOOLS_GLOBAL_HOOK__; if (!Is.isDisabled && Is.supportsFiber) try { Di = Is.inject(N0), Lt = Is } catch {} }
et.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = T0;
et.createPortal = function(e, t) { var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null; if (!tc(t)) throw Error(j(200)); return k0(e, t, null, n) };
et.createRoot = function(e, t) {
    if (!tc(e)) throw Error(j(299));
    var n = !1,
        r = "",
        o = Qh;
    return t != null && (t.unstable_strictMode === !0 && (n = !0), t.identifierPrefix !== void 0 && (r = t.identifierPrefix), t.onRecoverableError !== void 0 && (o = t.onRecoverableError)), t = Zu(e, 1, !1, null, null, n, !1, r, o), e[Gt] = t.current, Ko(e.nodeType === 8 ? e.parentNode : e), new ec(t)
};
et.findDOMNode = function(e) { if (e == null) return null; if (e.nodeType === 1) return e; var t = e._reactInternals; if (t === void 0) throw typeof e.render == "function" ? Error(j(188)) : (e = Object.keys(e).join(","), Error(j(268, e))); return e = hp(t), e = e === null ? null : e.stateNode, e };
et.flushSync = function(e) { return ar(e) };
et.hydrate = function(e, t, n) { if (!Xi(t)) throw Error(j(200)); return qi(null, e, t, !0, n) };
et.hydrateRoot = function(e, t, n) {
    if (!tc(e)) throw Error(j(405));
    var r = n != null && n.hydratedSources || null,
        o = !1,
        s = "",
        i = Qh;
    if (n != null && (n.unstable_strictMode === !0 && (o = !0), n.identifierPrefix !== void 0 && (s = n.identifierPrefix), n.onRecoverableError !== void 0 && (i = n.onRecoverableError)), t = Hh(t, null, e, 1, n ? ? null, o, !1, s, i), e[Gt] = t.current, Ko(e), r)
        for (e = 0; e < r.length; e++) n = r[e], o = n._getVersion, o = o(n._source), t.mutableSourceEagerHydrationData == null ? t.mutableSourceEagerHydrationData = [n, o] : t.mutableSourceEagerHydrationData.push(n, o);
    return new Yi(t)
};
et.render = function(e, t, n) { if (!Xi(t)) throw Error(j(200)); return qi(null, e, t, !1, n) };
et.unmountComponentAtNode = function(e) { if (!Xi(e)) throw Error(j(40)); return e._reactRootContainer ? (ar(function() { qi(null, null, e, !1, function() { e._reactRootContainer = null, e[Gt] = null }) }), !0) : !1 };
et.unstable_batchedUpdates = Gu;
et.unstable_renderSubtreeIntoContainer = function(e, t, n, r) { if (!Xi(n)) throw Error(j(200)); if (e == null || e._reactInternals === void 0) throw Error(j(38)); return qi(e, t, n, !1, r) };
et.version = "18.3.1-next-f1338f8080-20240426";

function Kh() { if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) try { __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Kh) } catch (e) { console.error(e) } }
Kh(), Kf.exports = et;
var ps = Kf.exports;
const Gh = Mf(ps);
var Yh, Id = ps;
Yh = Id.createRoot, Id.hydrateRoot;
const R0 = 1,
    j0 = 1e6,
    Ht = { ADD_TOAST: "ADD_TOAST", UPDATE_TOAST: "UPDATE_TOAST", DISMISS_TOAST: "DISMISS_TOAST", REMOVE_TOAST: "REMOVE_TOAST" };
let Vl = 0;

function _0() { return Vl = (Vl + 1) % Number.MAX_SAFE_INTEGER, Vl.toString() }
const Wl = new Map,
    Dd = e => {
        if (Wl.has(e)) return;
        const t = setTimeout(() => { Wl.delete(e), zo({ type: Ht.REMOVE_TOAST, toastId: e }) }, j0);
        Wl.set(e, t)
    },
    A0 = (e, t) => {
        switch (t.type) {
            case Ht.ADD_TOAST:
                return {...e, toasts: [t.toast, ...e.toasts].slice(0, R0) };
            case Ht.UPDATE_TOAST:
                return {...e, toasts: e.toasts.map(n => n.id === t.toast.id ? {...n, ...t.toast } : n) };
            case Ht.DISMISS_TOAST:
                { const { toastId: n } = t; return n ? Dd(n) : e.toasts.forEach(r => { Dd(r.id) }), {...e, toasts: e.toasts.map(r => n === void 0 || r.id === n ? {...r, open: !1 } : r) } }
            case Ht.REMOVE_TOAST:
                return t.toastId === void 0 ? {...e, toasts: [] } : {...e, toasts: e.toasts.filter(n => n.id !== t.toastId) };
            default:
                return e
        }
    },
    ti = [];
let ni = { toasts: [] };

function zo(e) { ni = A0(ni, e), ti.forEach(t => { t(ni) }) }

function O0(e) {
    const t = _0(),
        n = o => zo({ type: Ht.UPDATE_TOAST, toast: {...o, id: t } }),
        r = () => zo({ type: Ht.DISMISS_TOAST, toastId: t });
    return zo({ type: Ht.ADD_TOAST, toast: {...e, id: t, open: !0, onOpenChange: o => { o || r() } } }), { id: t, dismiss: r, update: n }
}

function Xh() {
    const [e, t] = w.useState(ni);
    return w.useEffect(() => (ti.push(t), () => {
        const n = ti.indexOf(t);
        n > -1 && ti.splice(n, 1)
    }), [e]), {...e, toast: O0, dismiss: n => zo({ type: Ht.DISMISS_TOAST, toastId: n }) }
}

function ve(e, t, { checkForDefaultPrevented: n = !0 } = {}) { return function(o) { if (e == null || e(o), n === !1 || !o.defaultPrevented) return t == null ? void 0 : t(o) } }

function zd(e, t) {
    if (typeof e == "function") return e(t);
    e != null && (e.current = t)
}

function Zi(...e) {
    return t => {
        let n = !1;
        const r = e.map(o => { const s = zd(o, t); return !n && typeof s == "function" && (n = !0), s });
        if (n) return () => {
            for (let o = 0; o < r.length; o++) {
                const s = r[o];
                typeof s == "function" ? s() : zd(e[o], null)
            }
        }
    }
}

function Ct(...e) { return w.useCallback(Zi(...e), e) }

function Ji(e, t = []) {
    let n = [];

    function r(s, i) {
        const l = w.createContext(i),
            a = n.length;
        n = [...n, i];
        const u = f => { var h; const { scope: m, children: d, ...S } = f, y = ((h = m == null ? void 0 : m[e]) == null ? void 0 : h[a]) || l, x = w.useMemo(() => S, Object.values(S)); return v.jsx(y.Provider, { value: x, children: d }) };
        u.displayName = s + "Provider";

        function c(f, m) {
            var y;
            const d = ((y = m == null ? void 0 : m[e]) == null ? void 0 : y[a]) || l,
                S = w.useContext(d);
            if (S) return S;
            if (i !== void 0) return i;
            throw new Error(`\`${f}\` must be used within \`${s}\``)
        }
        return [u, c]
    }
    const o = () => {
        const s = n.map(i => w.createContext(i));
        return function(l) {
            const a = (l == null ? void 0 : l[e]) || s;
            return w.useMemo(() => ({
                [`__scope${e}`]: {...l, [e]: a }
            }), [l, a])
        }
    };
    return o.scopeName = e, [r, L0(o, ...t)]
}

function L0(...e) {
    const t = e[0];
    if (e.length === 1) return t;
    const n = () => {
        const r = e.map(o => ({ useScope: o(), scopeName: o.scopeName }));
        return function(s) {
            const i = r.reduce((l, { useScope: a, scopeName: u }) => { const f = a(s)[`__scope${u}`]; return {...l, ...f } }, {});
            return w.useMemo(() => ({
                [`__scope${t.scopeName}`]: i
            }), [i])
        }
    };
    return n.scopeName = t.scopeName, n
}

function Fd(e) {
    const t = M0(e),
        n = w.forwardRef((r, o) => {
            const { children: s, ...i } = r, l = w.Children.toArray(s), a = l.find(D0);
            if (a) {
                const u = a.props.children,
                    c = l.map(f => f === a ? w.Children.count(u) > 1 ? w.Children.only(null) : w.isValidElement(u) ? u.props.children : null : f);
                return v.jsx(t, {...i, ref: o, children: w.isValidElement(u) ? w.cloneElement(u, void 0, c) : null })
            }
            return v.jsx(t, {...i, ref: o, children: s })
        });
    return n.displayName = `${e}.Slot`, n
}

function M0(e) {
    const t = w.forwardRef((n, r) => {
        const { children: o, ...s } = n;
        if (w.isValidElement(o)) {
            const i = F0(o),
                l = z0(s, o.props);
            return o.type !== w.Fragment && (l.ref = r ? Zi(r, i) : i), w.cloneElement(o, l)
        }
        return w.Children.count(o) > 1 ? w.Children.only(null) : null
    });
    return t.displayName = `${e}.SlotClone`, t
}
var I0 = Symbol("radix.slottable");

function D0(e) { return w.isValidElement(e) && typeof e.type == "function" && "__radixId" in e.type && e.type.__radixId === I0 }

function z0(e, t) {
    const n = {...t };
    for (const r in t) {
        const o = e[r],
            s = t[r];
        /^on[A-Z]/.test(r) ? o && s ? n[r] = (...l) => { const a = s(...l); return o(...l), a } : o && (n[r] = o) : r === "style" ? n[r] = {...o, ...s } : r === "className" && (n[r] = [o, s].filter(Boolean).join(" "))
    }
    return {...e, ...n }
}

function F0(e) {
    var r, o;
    let t = (r = Object.getOwnPropertyDescriptor(e.props, "ref")) == null ? void 0 : r.get,
        n = t && "isReactWarning" in t && t.isReactWarning;
    return n ? e.ref : (t = (o = Object.getOwnPropertyDescriptor(e, "ref")) == null ? void 0 : o.get, n = t && "isReactWarning" in t && t.isReactWarning, n ? e.props.ref : e.props.ref || e.ref)
}

function $0(e) {
    const t = e + "CollectionProvider",
        [n, r] = Ji(t),
        [o, s] = n(t, { collectionRef: { current: null }, itemMap: new Map }),
        i = y => { const { scope: x, children: h } = y, p = _.useRef(null), g = _.useRef(new Map).current; return v.jsx(o, { scope: x, itemMap: g, collectionRef: p, children: h }) };
    i.displayName = t;
    const l = e + "CollectionSlot",
        a = Fd(l),
        u = _.forwardRef((y, x) => { const { scope: h, children: p } = y, g = s(l, h), E = Ct(x, g.collectionRef); return v.jsx(a, { ref: E, children: p }) });
    u.displayName = l;
    const c = e + "CollectionItemSlot",
        f = "data-radix-collection-item",
        m = Fd(c),
        d = _.forwardRef((y, x) => {
            const { scope: h, children: p, ...g } = y, E = _.useRef(null), C = Ct(x, E), k = s(c, h);
            return _.useEffect(() => (k.itemMap.set(E, { ref: E, ...g }), () => void k.itemMap.delete(E))), v.jsx(m, {
                [f]: "",
                ref: C,
                children: p
            })
        });
    d.displayName = c;

    function S(y) { const x = s(e + "CollectionConsumer", y); return _.useCallback(() => { const p = x.collectionRef.current; if (!p) return []; const g = Array.from(p.querySelectorAll(`[${f}]`)); return Array.from(x.itemMap.values()).sort((k, P) => g.indexOf(k.ref.current) - g.indexOf(P.ref.current)) }, [x.collectionRef, x.itemMap]) }
    return [{ Provider: i, Slot: u, ItemSlot: d }, S, r]
}

function B0(e) {
    const t = U0(e),
        n = w.forwardRef((r, o) => {
            const { children: s, ...i } = r, l = w.Children.toArray(s), a = l.find(W0);
            if (a) {
                const u = a.props.children,
                    c = l.map(f => f === a ? w.Children.count(u) > 1 ? w.Children.only(null) : w.isValidElement(u) ? u.props.children : null : f);
                return v.jsx(t, {...i, ref: o, children: w.isValidElement(u) ? w.cloneElement(u, void 0, c) : null })
            }
            return v.jsx(t, {...i, ref: o, children: s })
        });
    return n.displayName = `${e}.Slot`, n
}

function U0(e) {
    const t = w.forwardRef((n, r) => {
        const { children: o, ...s } = n;
        if (w.isValidElement(o)) {
            const i = Q0(o),
                l = H0(s, o.props);
            return o.type !== w.Fragment && (l.ref = r ? Zi(r, i) : i), w.cloneElement(o, l)
        }
        return w.Children.count(o) > 1 ? w.Children.only(null) : null
    });
    return t.displayName = `${e}.SlotClone`, t
}
var V0 = Symbol("radix.slottable");

function W0(e) { return w.isValidElement(e) && typeof e.type == "function" && "__radixId" in e.type && e.type.__radixId === V0 }

function H0(e, t) {
    const n = {...t };
    for (const r in t) {
        const o = e[r],
            s = t[r];
        /^on[A-Z]/.test(r) ? o && s ? n[r] = (...l) => { const a = s(...l); return o(...l), a } : o && (n[r] = o) : r === "style" ? n[r] = {...o, ...s } : r === "className" && (n[r] = [o, s].filter(Boolean).join(" "))
    }
    return {...e, ...n }
}

function Q0(e) {
    var r, o;
    let t = (r = Object.getOwnPropertyDescriptor(e.props, "ref")) == null ? void 0 : r.get,
        n = t && "isReactWarning" in t && t.isReactWarning;
    return n ? e.ref : (t = (o = Object.getOwnPropertyDescriptor(e, "ref")) == null ? void 0 : o.get, n = t && "isReactWarning" in t && t.isReactWarning, n ? e.props.ref : e.props.ref || e.ref)
}
var K0 = ["a", "button", "div", "form", "h2", "h3", "img", "input", "label", "li", "nav", "ol", "p", "select", "span", "svg", "ul"],
    Qe = K0.reduce((e, t) => {
        const n = B0(`Primitive.${t}`),
            r = w.forwardRef((o, s) => { const { asChild: i, ...l } = o, a = i ? n : t; return typeof window < "u" && (window[Symbol.for("radix-ui")] = !0), v.jsx(a, {...l, ref: s }) });
        return r.displayName = `Primitive.${t}`, {...e, [t]: r }
    }, {});

function qh(e, t) { e && ps.flushSync(() => e.dispatchEvent(t)) }

function qt(e) { const t = w.useRef(e); return w.useEffect(() => { t.current = e }), w.useMemo(() => (...n) => { var r; return (r = t.current) == null ? void 0 : r.call(t, ...n) }, []) }

function G0(e, t = globalThis == null ? void 0 : globalThis.document) {
    const n = qt(e);
    w.useEffect(() => { const r = o => { o.key === "Escape" && n(o) }; return t.addEventListener("keydown", r, { capture: !0 }), () => t.removeEventListener("keydown", r, { capture: !0 }) }, [n, t])
}
var Y0 = "DismissableLayer",
    Va = "dismissableLayer.update",
    X0 = "dismissableLayer.pointerDownOutside",
    q0 = "dismissableLayer.focusOutside",
    $d, Zh = w.createContext({ layers: new Set, layersWithOutsidePointerEventsDisabled: new Set, branches: new Set }),
    nc = w.forwardRef((e, t) => {
        const { disableOutsidePointerEvents: n = !1, onEscapeKeyDown: r, onPointerDownOutside: o, onFocusOutside: s, onInteractOutside: i, onDismiss: l, ...a } = e, u = w.useContext(Zh), [c, f] = w.useState(null), m = (c == null ? void 0 : c.ownerDocument) ? ? (globalThis == null ? void 0 : globalThis.document), [, d] = w.useState({}), S = Ct(t, P => f(P)), y = Array.from(u.layers), [x] = [...u.layersWithOutsidePointerEventsDisabled].slice(-1), h = y.indexOf(x), p = c ? y.indexOf(c) : -1, g = u.layersWithOutsidePointerEventsDisabled.size > 0, E = p >= h, C = J0(P => {
            const N = P.target,
                L = [...u.branches].some(A => A.contains(N));
            !E || L || (o == null || o(P), i == null || i(P), P.defaultPrevented || l == null || l())
        }, m), k = ew(P => {
            const N = P.target;
            [...u.branches].some(A => A.contains(N)) || (s == null || s(P), i == null || i(P), P.defaultPrevented || l == null || l())
        }, m);
        return G0(P => { p === u.layers.size - 1 && (r == null || r(P), !P.defaultPrevented && l && (P.preventDefault(), l())) }, m), w.useEffect(() => { if (c) return n && (u.layersWithOutsidePointerEventsDisabled.size === 0 && ($d = m.body.style.pointerEvents, m.body.style.pointerEvents = "none"), u.layersWithOutsidePointerEventsDisabled.add(c)), u.layers.add(c), Bd(), () => { n && u.layersWithOutsidePointerEventsDisabled.size === 1 && (m.body.style.pointerEvents = $d) } }, [c, m, n, u]), w.useEffect(() => () => { c && (u.layers.delete(c), u.layersWithOutsidePointerEventsDisabled.delete(c), Bd()) }, [c, u]), w.useEffect(() => { const P = () => d({}); return document.addEventListener(Va, P), () => document.removeEventListener(Va, P) }, []), v.jsx(Qe.div, {...a, ref: S, style: { pointerEvents: g ? E ? "auto" : "none" : void 0, ...e.style }, onFocusCapture: ve(e.onFocusCapture, k.onFocusCapture), onBlurCapture: ve(e.onBlurCapture, k.onBlurCapture), onPointerDownCapture: ve(e.onPointerDownCapture, C.onPointerDownCapture) })
    });
nc.displayName = Y0;
var Z0 = "DismissableLayerBranch",
    Jh = w.forwardRef((e, t) => {
        const n = w.useContext(Zh),
            r = w.useRef(null),
            o = Ct(t, r);
        return w.useEffect(() => { const s = r.current; if (s) return n.branches.add(s), () => { n.branches.delete(s) } }, [n.branches]), v.jsx(Qe.div, {...e, ref: o })
    });
Jh.displayName = Z0;

function J0(e, t = globalThis == null ? void 0 : globalThis.document) {
    const n = qt(e),
        r = w.useRef(!1),
        o = w.useRef(() => {});
    return w.useEffect(() => {
        const s = l => {
                if (l.target && !r.current) {
                    let a = function() { em(X0, n, u, { discrete: !0 }) };
                    const u = { originalEvent: l };
                    l.pointerType === "touch" ? (t.removeEventListener("click", o.current), o.current = a, t.addEventListener("click", o.current, { once: !0 })) : a()
                } else t.removeEventListener("click", o.current);
                r.current = !1
            },
            i = window.setTimeout(() => { t.addEventListener("pointerdown", s) }, 0);
        return () => { window.clearTimeout(i), t.removeEventListener("pointerdown", s), t.removeEventListener("click", o.current) }
    }, [t, n]), { onPointerDownCapture: () => r.current = !0 }
}

function ew(e, t = globalThis == null ? void 0 : globalThis.document) {
    const n = qt(e),
        r = w.useRef(!1);
    return w.useEffect(() => { const o = s => { s.target && !r.current && em(q0, n, { originalEvent: s }, { discrete: !1 }) }; return t.addEventListener("focusin", o), () => t.removeEventListener("focusin", o) }, [t, n]), { onFocusCapture: () => r.current = !0, onBlurCapture: () => r.current = !1 }
}

function Bd() {
    const e = new CustomEvent(Va);
    document.dispatchEvent(e)
}

function em(e, t, n, { discrete: r }) {
    const o = n.originalEvent.target,
        s = new CustomEvent(e, { bubbles: !1, cancelable: !0, detail: n });
    t && o.addEventListener(e, t, { once: !0 }), r ? qh(o, s) : o.dispatchEvent(s)
}
var tw = nc,
    nw = Jh,
    kt = globalThis != null && globalThis.document ? w.useLayoutEffect : () => {},
    rw = "Portal",
    tm = w.forwardRef((e, t) => {
        var l;
        const { container: n, ...r } = e, [o, s] = w.useState(!1);
        kt(() => s(!0), []);
        const i = n || o && ((l = globalThis == null ? void 0 : globalThis.document) == null ? void 0 : l.body);
        return i ? Gh.createPortal(v.jsx(Qe.div, {...r, ref: t }), i) : null
    });
tm.displayName = rw;

function ow(e, t) { return w.useReducer((n, r) => t[n][r] ? ? n, e) }
var rc = e => { const { present: t, children: n } = e, r = sw(t), o = typeof n == "function" ? n({ present: r.isPresent }) : w.Children.only(n), s = Ct(r.ref, iw(o)); return typeof n == "function" || r.isPresent ? w.cloneElement(o, { ref: s }) : null };
rc.displayName = "Presence";

function sw(e) {
    const [t, n] = w.useState(), r = w.useRef(null), o = w.useRef(e), s = w.useRef("none"), i = e ? "mounted" : "unmounted", [l, a] = ow(i, { mounted: { UNMOUNT: "unmounted", ANIMATION_OUT: "unmountSuspended" }, unmountSuspended: { MOUNT: "mounted", ANIMATION_END: "unmounted" }, unmounted: { MOUNT: "mounted" } });
    return w.useEffect(() => {
        const u = Ds(r.current);
        s.current = l === "mounted" ? u : "none"
    }, [l]), kt(() => {
        const u = r.current,
            c = o.current;
        if (c !== e) {
            const m = s.current,
                d = Ds(u);
            e ? a("MOUNT") : d === "none" || (u == null ? void 0 : u.display) === "none" ? a("UNMOUNT") : a(c && m !== d ? "ANIMATION_OUT" : "UNMOUNT"), o.current = e
        }
    }, [e, a]), kt(() => {
        if (t) {
            let u;
            const c = t.ownerDocument.defaultView ? ? window,
                f = d => {
                    const y = Ds(r.current).includes(CSS.escape(d.animationName));
                    if (d.target === t && y && (a("ANIMATION_END"), !o.current)) {
                        const x = t.style.animationFillMode;
                        t.style.animationFillMode = "forwards", u = c.setTimeout(() => { t.style.animationFillMode === "forwards" && (t.style.animationFillMode = x) })
                    }
                },
                m = d => { d.target === t && (s.current = Ds(r.current)) };
            return t.addEventListener("animationstart", m), t.addEventListener("animationcancel", f), t.addEventListener("animationend", f), () => { c.clearTimeout(u), t.removeEventListener("animationstart", m), t.removeEventListener("animationcancel", f), t.removeEventListener("animationend", f) }
        } else a("ANIMATION_END")
    }, [t, a]), { isPresent: ["mounted", "unmountSuspended"].includes(l), ref: w.useCallback(u => { r.current = u ? getComputedStyle(u) : null, n(u) }, []) }
}

function Ds(e) { return (e == null ? void 0 : e.animationName) || "none" }

function iw(e) {
    var r, o;
    let t = (r = Object.getOwnPropertyDescriptor(e.props, "ref")) == null ? void 0 : r.get,
        n = t && "isReactWarning" in t && t.isReactWarning;
    return n ? e.ref : (t = (o = Object.getOwnPropertyDescriptor(e, "ref")) == null ? void 0 : o.get, n = t && "isReactWarning" in t && t.isReactWarning, n ? e.props.ref : e.props.ref || e.ref)
}
var lw = du[" useInsertionEffect ".trim().toString()] || kt;

function aw({ prop: e, defaultProp: t, onChange: n = () => {}, caller: r }) {
    const [o, s, i] = uw({ defaultProp: t, onChange: n }), l = e !== void 0, a = l ? e : o; {
        const c = w.useRef(e !== void 0);
        w.useEffect(() => {
            const f = c.current;
            f !== l && console.warn(`${r} is changing from ${f?"controlled":"uncontrolled"} to ${l?"controlled":"uncontrolled"}. Components should not switch from controlled to uncontrolled (or vice versa). Decide between using a controlled or uncontrolled value for the lifetime of the component.`), c.current = l
        }, [l, r])
    }
    const u = w.useCallback(c => {
        var f;
        if (l) {
            const m = cw(c) ? c(e) : c;
            m !== e && ((f = i.current) == null || f.call(i, m))
        } else s(c)
    }, [l, e, s, i]);
    return [a, u]
}

function uw({ defaultProp: e, onChange: t }) {
    const [n, r] = w.useState(e), o = w.useRef(n), s = w.useRef(t);
    return lw(() => { s.current = t }, [t]), w.useEffect(() => {
        var i;
        o.current !== n && ((i = s.current) == null || i.call(s, n), o.current = n)
    }, [n, o]), [n, r, s]
}

function cw(e) { return typeof e == "function" }
var dw = Object.freeze({ position: "absolute", border: 0, width: 1, height: 1, padding: 0, margin: -1, overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", wordWrap: "normal" }),
    fw = "VisuallyHidden",
    el = w.forwardRef((e, t) => v.jsx(Qe.span, {...e, ref: t, style: {...dw, ...e.style } }));
el.displayName = fw;
var pw = el,
    oc = "ToastProvider",
    [sc, hw, mw] = $0("Toast"),
    [nm] = Ji("Toast", [mw]),
    [gw, tl] = nm(oc),
    rm = e => { const { __scopeToast: t, label: n = "Notification", duration: r = 5e3, swipeDirection: o = "right", swipeThreshold: s = 50, children: i } = e, [l, a] = w.useState(null), [u, c] = w.useState(0), f = w.useRef(!1), m = w.useRef(!1); return n.trim() || console.error(`Invalid prop \`label\` supplied to \`${oc}\`. Expected non-empty \`string\`.`), v.jsx(sc.Provider, { scope: t, children: v.jsx(gw, { scope: t, label: n, duration: r, swipeDirection: o, swipeThreshold: s, toastCount: u, viewport: l, onViewportChange: a, onToastAdd: w.useCallback(() => c(d => d + 1), []), onToastRemove: w.useCallback(() => c(d => d - 1), []), isFocusedToastEscapeKeyDownRef: f, isClosePausedRef: m, children: i }) }) };
rm.displayName = oc;
var om = "ToastViewport",
    vw = ["F8"],
    Wa = "toast.viewportPause",
    Ha = "toast.viewportResume",
    sm = w.forwardRef((e, t) => {
        const { __scopeToast: n, hotkey: r = vw, label: o = "Notifications ({hotkey})", ...s } = e, i = tl(om, n), l = hw(n), a = w.useRef(null), u = w.useRef(null), c = w.useRef(null), f = w.useRef(null), m = Ct(t, f, i.onViewportChange), d = r.join("+").replace(/Key/g, "").replace(/Digit/g, ""), S = i.toastCount > 0;
        w.useEffect(() => {
            const x = h => {
                var g;
                r.length !== 0 && r.every(E => h[E] || h.code === E) && ((g = f.current) == null || g.focus())
            };
            return document.addEventListener("keydown", x), () => document.removeEventListener("keydown", x)
        }, [r]), w.useEffect(() => {
            const x = a.current,
                h = f.current;
            if (S && x && h) {
                const p = () => {
                        if (!i.isClosePausedRef.current) {
                            const k = new CustomEvent(Wa);
                            h.dispatchEvent(k), i.isClosePausedRef.current = !0
                        }
                    },
                    g = () => {
                        if (i.isClosePausedRef.current) {
                            const k = new CustomEvent(Ha);
                            h.dispatchEvent(k), i.isClosePausedRef.current = !1
                        }
                    },
                    E = k => {!x.contains(k.relatedTarget) && g() },
                    C = () => { x.contains(document.activeElement) || g() };
                return x.addEventListener("focusin", p), x.addEventListener("focusout", E), x.addEventListener("pointermove", p), x.addEventListener("pointerleave", C), window.addEventListener("blur", p), window.addEventListener("focus", g), () => { x.removeEventListener("focusin", p), x.removeEventListener("focusout", E), x.removeEventListener("pointermove", p), x.removeEventListener("pointerleave", C), window.removeEventListener("blur", p), window.removeEventListener("focus", g) }
            }
        }, [S, i.isClosePausedRef]);
        const y = w.useCallback(({ tabbingDirection: x }) => {
            const p = l().map(g => {
                const E = g.ref.current,
                    C = [E, ...jw(E)];
                return x === "forwards" ? C : C.reverse()
            });
            return (x === "forwards" ? p.reverse() : p).flat()
        }, [l]);
        return w.useEffect(() => {
            const x = f.current;
            if (x) {
                const h = p => {
                    var C, k, P;
                    const g = p.altKey || p.ctrlKey || p.metaKey;
                    if (p.key === "Tab" && !g) {
                        const N = document.activeElement,
                            L = p.shiftKey;
                        if (p.target === x && L) {
                            (C = u.current) == null || C.focus();
                            return
                        }
                        const D = y({ tabbingDirection: L ? "backwards" : "forwards" }),
                            Q = D.findIndex(O => O === N);
                        Hl(D.slice(Q + 1)) ? p.preventDefault() : L ? (k = u.current) == null || k.focus() : (P = c.current) == null || P.focus()
                    }
                };
                return x.addEventListener("keydown", h), () => x.removeEventListener("keydown", h)
            }
        }, [l, y]), v.jsxs(nw, {
            ref: a,
            role: "region",
            "aria-label": o.replace("{hotkey}", d),
            tabIndex: -1,
            style: { pointerEvents: S ? void 0 : "none" },
            children: [S && v.jsx(Qa, {
                ref: u,
                onFocusFromOutsideViewport: () => {
                    const x = y({ tabbingDirection: "forwards" });
                    Hl(x)
                }
            }), v.jsx(sc.Slot, { scope: n, children: v.jsx(Qe.ol, { tabIndex: -1, ...s, ref: m }) }), S && v.jsx(Qa, {
                ref: c,
                onFocusFromOutsideViewport: () => {
                    const x = y({ tabbingDirection: "backwards" });
                    Hl(x)
                }
            })]
        })
    });
sm.displayName = om;
var im = "ToastFocusProxy",
    Qa = w.forwardRef((e, t) => { const { __scopeToast: n, onFocusFromOutsideViewport: r, ...o } = e, s = tl(im, n); return v.jsx(el, { tabIndex: 0, ...o, ref: t, style: { position: "fixed" }, onFocus: i => { var u; const l = i.relatedTarget;!((u = s.viewport) != null && u.contains(l)) && r() } }) });
Qa.displayName = im;
var hs = "Toast",
    yw = "toast.swipeStart",
    ww = "toast.swipeMove",
    xw = "toast.swipeCancel",
    Sw = "toast.swipeEnd",
    lm = w.forwardRef((e, t) => {
        const { forceMount: n, open: r, defaultOpen: o, onOpenChange: s, ...i } = e, [l, a] = aw({ prop: r, defaultProp: o ? ? !0, onChange: s, caller: hs });
        return v.jsx(rc, {
            present: n || l,
            children: v.jsx(kw, {
                open: l,
                ...i,
                ref: t,
                onClose: () => a(!1),
                onPause: qt(e.onPause),
                onResume: qt(e.onResume),
                onSwipeStart: ve(e.onSwipeStart, u => { u.currentTarget.setAttribute("data-swipe", "start") }),
                onSwipeMove: ve(e.onSwipeMove, u => {
                    const { x: c, y: f } = u.detail.delta;
                    u.currentTarget.setAttribute("data-swipe", "move"), u.currentTarget.style.setProperty("--radix-toast-swipe-move-x", `${c}px`), u.currentTarget.style.setProperty("--radix-toast-swipe-move-y", `${f}px`)
                }),
                onSwipeCancel: ve(e.onSwipeCancel, u => { u.currentTarget.setAttribute("data-swipe", "cancel"), u.currentTarget.style.removeProperty("--radix-toast-swipe-move-x"), u.currentTarget.style.removeProperty("--radix-toast-swipe-move-y"), u.currentTarget.style.removeProperty("--radix-toast-swipe-end-x"), u.currentTarget.style.removeProperty("--radix-toast-swipe-end-y") }),
                onSwipeEnd: ve(e.onSwipeEnd, u => {
                    const { x: c, y: f } = u.detail.delta;
                    u.currentTarget.setAttribute("data-swipe", "end"), u.currentTarget.style.removeProperty("--radix-toast-swipe-move-x"), u.currentTarget.style.removeProperty("--radix-toast-swipe-move-y"), u.currentTarget.style.setProperty("--radix-toast-swipe-end-x", `${c}px`), u.currentTarget.style.setProperty("--radix-toast-swipe-end-y", `${f}px`), a(!1)
                })
            })
        })
    });
lm.displayName = hs;
var [Ew, Cw] = nm(hs, { onClose() {} }), kw = w.forwardRef((e, t) => {
    const { __scopeToast: n, type: r = "foreground", duration: o, open: s, onClose: i, onEscapeKeyDown: l, onPause: a, onResume: u, onSwipeStart: c, onSwipeMove: f, onSwipeCancel: m, onSwipeEnd: d, ...S } = e, y = tl(hs, n), [x, h] = w.useState(null), p = Ct(t, O => h(O)), g = w.useRef(null), E = w.useRef(null), C = o || y.duration, k = w.useRef(0), P = w.useRef(C), N = w.useRef(0), { onToastAdd: L, onToastRemove: A } = y, $ = qt(() => {
        var Y;
        (x == null ? void 0 : x.contains(document.activeElement)) && ((Y = y.viewport) == null || Y.focus()), i()
    }), D = w.useCallback(O => {!O || O === 1 / 0 || (window.clearTimeout(N.current), k.current = new Date().getTime(), N.current = window.setTimeout($, O)) }, [$]);
    w.useEffect(() => {
        const O = y.viewport;
        if (O) {
            const Y = () => { D(P.current), u == null || u() },
                B = () => {
                    const V = new Date().getTime() - k.current;
                    P.current = P.current - V, window.clearTimeout(N.current), a == null || a()
                };
            return O.addEventListener(Wa, B), O.addEventListener(Ha, Y), () => { O.removeEventListener(Wa, B), O.removeEventListener(Ha, Y) }
        }
    }, [y.viewport, C, a, u, D]), w.useEffect(() => { s && !y.isClosePausedRef.current && D(C) }, [s, C, y.isClosePausedRef, D]), w.useEffect(() => (L(), () => A()), [L, A]);
    const Q = w.useMemo(() => x ? hm(x) : null, [x]);
    return y.viewport ? v.jsxs(v.Fragment, {
        children: [Q && v.jsx(Pw, { __scopeToast: n, role: "status", "aria-live": r === "foreground" ? "assertive" : "polite", children: Q }), v.jsx(Ew, {
            scope: n,
            onClose: $,
            children: ps.createPortal(v.jsx(sc.ItemSlot, {
                scope: n,
                children: v.jsx(tw, {
                    asChild: !0,
                    onEscapeKeyDown: ve(l, () => { y.isFocusedToastEscapeKeyDownRef.current || $(), y.isFocusedToastEscapeKeyDownRef.current = !1 }),
                    children: v.jsx(Qe.li, {
                        tabIndex: 0,
                        "data-state": s ? "open" : "closed",
                        "data-swipe-direction": y.swipeDirection,
                        ...S,
                        ref: p,
                        style: { userSelect: "none", touchAction: "none", ...e.style },
                        onKeyDown: ve(e.onKeyDown, O => { O.key === "Escape" && (l == null || l(O.nativeEvent), O.nativeEvent.defaultPrevented || (y.isFocusedToastEscapeKeyDownRef.current = !0, $())) }),
                        onPointerDown: ve(e.onPointerDown, O => { O.button === 0 && (g.current = { x: O.clientX, y: O.clientY }) }),
                        onPointerMove: ve(e.onPointerMove, O => {
                            if (!g.current) return;
                            const Y = O.clientX - g.current.x,
                                B = O.clientY - g.current.y,
                                V = !!E.current,
                                T = ["left", "right"].includes(y.swipeDirection),
                                R = ["left", "up"].includes(y.swipeDirection) ? Math.min : Math.max,
                                M = T ? R(0, Y) : 0,
                                W = T ? 0 : R(0, B),
                                z = O.pointerType === "touch" ? 10 : 2,
                                K = { x: M, y: W },
                                X = { originalEvent: O, delta: K };
                            V ? (E.current = K, zs(ww, f, X, { discrete: !1 })) : Ud(K, y.swipeDirection, z) ? (E.current = K, zs(yw, c, X, { discrete: !1 }), O.target.setPointerCapture(O.pointerId)) : (Math.abs(Y) > z || Math.abs(B) > z) && (g.current = null)
                        }),
                        onPointerUp: ve(e.onPointerUp, O => {
                            const Y = E.current,
                                B = O.target;
                            if (B.hasPointerCapture(O.pointerId) && B.releasePointerCapture(O.pointerId), E.current = null, g.current = null, Y) {
                                const V = O.currentTarget,
                                    T = { originalEvent: O, delta: Y };
                                Ud(Y, y.swipeDirection, y.swipeThreshold) ? zs(Sw, d, T, { discrete: !0 }) : zs(xw, m, T, { discrete: !0 }), V.addEventListener("click", R => R.preventDefault(), { once: !0 })
                            }
                        })
                    })
                })
            }), y.viewport)
        })]
    }) : null
}), Pw = e => { const { __scopeToast: t, children: n, ...r } = e, o = tl(hs, t), [s, i] = w.useState(!1), [l, a] = w.useState(!1); return Nw(() => i(!0)), w.useEffect(() => { const u = window.setTimeout(() => a(!0), 1e3); return () => window.clearTimeout(u) }, []), l ? null : v.jsx(tm, { asChild: !0, children: v.jsx(el, {...r, children: s && v.jsxs(v.Fragment, { children: [o.label, " ", n] }) }) }) }, bw = "ToastTitle", am = w.forwardRef((e, t) => { const { __scopeToast: n, ...r } = e; return v.jsx(Qe.div, {...r, ref: t }) });
am.displayName = bw;
var Tw = "ToastDescription",
    um = w.forwardRef((e, t) => { const { __scopeToast: n, ...r } = e; return v.jsx(Qe.div, {...r, ref: t }) });
um.displayName = Tw;
var cm = "ToastAction",
    dm = w.forwardRef((e, t) => { const { altText: n, ...r } = e; return n.trim() ? v.jsx(pm, { altText: n, asChild: !0, children: v.jsx(ic, {...r, ref: t }) }) : (console.error(`Invalid prop \`altText\` supplied to \`${cm}\`. Expected non-empty \`string\`.`), null) });
dm.displayName = cm;
var fm = "ToastClose",
    ic = w.forwardRef((e, t) => { const { __scopeToast: n, ...r } = e, o = Cw(fm, n); return v.jsx(pm, { asChild: !0, children: v.jsx(Qe.button, { type: "button", ...r, ref: t, onClick: ve(e.onClick, o.onClose) }) }) });
ic.displayName = fm;
var pm = w.forwardRef((e, t) => { const { __scopeToast: n, altText: r, ...o } = e; return v.jsx(Qe.div, { "data-radix-toast-announce-exclude": "", "data-radix-toast-announce-alt": r || void 0, ...o, ref: t }) });

function hm(e) {
    const t = [];
    return Array.from(e.childNodes).forEach(r => {
        if (r.nodeType === r.TEXT_NODE && r.textContent && t.push(r.textContent), Rw(r)) {
            const o = r.ariaHidden || r.hidden || r.style.display === "none",
                s = r.dataset.radixToastAnnounceExclude === "";
            if (!o)
                if (s) {
                    const i = r.dataset.radixToastAnnounceAlt;
                    i && t.push(i)
                } else t.push(...hm(r))
        }
    }), t
}

function zs(e, t, n, { discrete: r }) {
    const o = n.originalEvent.currentTarget,
        s = new CustomEvent(e, { bubbles: !0, cancelable: !0, detail: n });
    t && o.addEventListener(e, t, { once: !0 }), r ? qh(o, s) : o.dispatchEvent(s)
}
var Ud = (e, t, n = 0) => {
    const r = Math.abs(e.x),
        o = Math.abs(e.y),
        s = r > o;
    return t === "left" || t === "right" ? s && r > n : !s && o > n
};

function Nw(e = () => {}) {
    const t = qt(e);
    kt(() => {
        let n = 0,
            r = 0;
        return n = window.requestAnimationFrame(() => r = window.requestAnimationFrame(t)), () => { window.cancelAnimationFrame(n), window.cancelAnimationFrame(r) }
    }, [t])
}

function Rw(e) { return e.nodeType === e.ELEMENT_NODE }

function jw(e) {
    const t = [],
        n = document.createTreeWalker(e, NodeFilter.SHOW_ELEMENT, { acceptNode: r => { const o = r.tagName === "INPUT" && r.type === "hidden"; return r.disabled || r.hidden || o ? NodeFilter.FILTER_SKIP : r.tabIndex >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP } });
    for (; n.nextNode();) t.push(n.currentNode);
    return t
}

function Hl(e) { const t = document.activeElement; return e.some(n => n === t ? !0 : (n.focus(), document.activeElement !== t)) }
var _w = rm,
    mm = sm,
    gm = lm,
    vm = am,
    ym = um,
    wm = dm,
    xm = ic;
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Aw = e => e.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase(),
    Sm = (...e) => e.filter((t, n, r) => !!t && t.trim() !== "" && r.indexOf(t) === n).join(" ").trim();
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
var Ow = { xmlns: "http://www.w3.org/2000/svg", width: 24, height: 24, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Lw = w.forwardRef(({ color: e = "currentColor", size: t = 24, strokeWidth: n = 2, absoluteStrokeWidth: r, className: o = "", children: s, iconNode: i, ...l }, a) => w.createElement("svg", { ref: a, ...Ow, width: t, height: t, stroke: e, strokeWidth: r ? Number(n) * 24 / Number(t) : n, className: Sm("lucide", o), ...l }, [...i.map(([u, c]) => w.createElement(u, c)), ...Array.isArray(s) ? s : [s]]));
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const be = (e, t) => { const n = w.forwardRef(({ className: r, ...o }, s) => w.createElement(Lw, { ref: s, iconNode: t, className: Sm(`lucide-${Aw(e)}`, r), ...o })); return n.displayName = `${e}`, n };
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Mw = be("ArrowRight", [
    ["path", { d: "M5 12h14", key: "1ays0h" }],
    ["path", { d: "m12 5 7 7-7 7", key: "xquz4c" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Iw = be("Camera", [
    ["path", { d: "M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z", key: "1tc9qg" }],
    ["circle", { cx: "12", cy: "13", r: "3", key: "1vg3eu" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Dw = be("Check", [
    ["path", { d: "M20 6 9 17l-5-5", key: "1gmf2c" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const zw = be("CircleAlert", [
    ["circle", { cx: "12", cy: "12", r: "10", key: "1mglay" }],
    ["line", { x1: "12", x2: "12", y1: "8", y2: "12", key: "1pkeuh" }],
    ["line", { x1: "12", x2: "12.01", y1: "16", y2: "16", key: "4dfq90" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Fw = be("CircleCheckBig", [
    ["path", { d: "M21.801 10A10 10 0 1 1 17 3.335", key: "yps3ct" }],
    ["path", { d: "m9 11 3 3L22 4", key: "1pflzl" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const $w = be("Clock", [
    ["circle", { cx: "12", cy: "12", r: "10", key: "1mglay" }],
    ["polyline", { points: "12 6 12 12 16 14", key: "68esgv" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Bw = be("Github", [
    ["path", { d: "M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4", key: "tonef" }],
    ["path", { d: "M9 18c-4.51 2-5-2-7-2", key: "9comsn" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Uw = be("Image", [
    ["rect", { width: "18", height: "18", x: "3", y: "3", rx: "2", ry: "2", key: "1m3agn" }],
    ["circle", { cx: "9", cy: "9", r: "2", key: "af1f0g" }],
    ["path", { d: "m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21", key: "1xmnt7" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Vw = be("Instagram", [
    ["rect", { width: "20", height: "20", x: "2", y: "2", rx: "5", ry: "5", key: "2e1cvw" }],
    ["path", { d: "M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z", key: "9exkf1" }],
    ["line", { x1: "17.5", x2: "17.51", y1: "6.5", y2: "6.5", key: "r4j83e" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Em = be("Leaf", [
    ["path", { d: "M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z", key: "nnexq3" }],
    ["path", { d: "M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12", key: "mt58a7" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Ww = be("Search", [
    ["circle", { cx: "11", cy: "11", r: "8", key: "4ej97u" }],
    ["path", { d: "m21 21-4.3-4.3", key: "1qie3q" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Hw = be("ShieldCheck", [
    ["path", { d: "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z", key: "oel41y" }],
    ["path", { d: "m9 12 2 2 4-4", key: "dzmm74" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Qw = be("Sparkles", [
    ["path", { d: "M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z", key: "4pj2yx" }],
    ["path", { d: "M20 3v4", key: "1olli1" }],
    ["path", { d: "M22 5h-4", key: "1gvqau" }],
    ["path", { d: "M4 17v2", key: "vumght" }],
    ["path", { d: "M5 18H3", key: "zchphs" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Kw = be("TrendingUp", [
    ["polyline", { points: "22 7 13.5 15.5 8.5 10.5 2 17", key: "126l90" }],
    ["polyline", { points: "16 7 22 7 22 13", key: "kwv8wd" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Gw = be("Twitter", [
    ["path", { d: "M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z", key: "pff0z6" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Yw = be("Upload", [
    ["path", { d: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", key: "ih7n3h" }],
    ["polyline", { points: "17 8 12 3 7 8", key: "t8dd8p" }],
    ["line", { x1: "12", x2: "12", y1: "3", y2: "15", key: "widbto" }]
]);
/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Xw = be("X", [
    ["path", { d: "M18 6 6 18", key: "1bl5f8" }],
    ["path", { d: "m6 6 12 12", key: "d8bk6v" }]
]);

function Cm(e) {
    var t, n, r = "";
    if (typeof e == "string" || typeof e == "number") r += e;
    else if (typeof e == "object")
        if (Array.isArray(e)) { var o = e.length; for (t = 0; t < o; t++) e[t] && (n = Cm(e[t])) && (r && (r += " "), r += n) } else
            for (n in e) e[n] && (r && (r += " "), r += n);
    return r
}

function km() { for (var e, t, n = 0, r = "", o = arguments.length; n < o; n++)(e = arguments[n]) && (t = Cm(e)) && (r && (r += " "), r += t); return r }
const lc = "-",
    qw = e => {
        const t = Jw(e),
            { conflictingClassGroups: n, conflictingClassGroupModifiers: r } = e;
        return { getClassGroupId: i => { const l = i.split(lc); return l[0] === "" && l.length !== 1 && l.shift(), Pm(l, t) || Zw(i) }, getConflictingClassGroupIds: (i, l) => { const a = n[i] || []; return l && r[i] ? [...a, ...r[i]] : a } }
    },
    Pm = (e, t) => {
        var i;
        if (e.length === 0) return t.classGroupId;
        const n = e[0],
            r = t.nextPart.get(n),
            o = r ? Pm(e.slice(1), r) : void 0;
        if (o) return o;
        if (t.validators.length === 0) return;
        const s = e.join(lc);
        return (i = t.validators.find(({ validator: l }) => l(s))) == null ? void 0 : i.classGroupId
    },
    Vd = /^\[(.+)\]$/,
    Zw = e => {
        if (Vd.test(e)) {
            const t = Vd.exec(e)[1],
                n = t == null ? void 0 : t.substring(0, t.indexOf(":"));
            if (n) return "arbitrary.." + n
        }
    },
    Jw = e => { const { theme: t, prefix: n } = e, r = { nextPart: new Map, validators: [] }; return tx(Object.entries(e.classGroups), n).forEach(([s, i]) => { Ka(i, r, s, t) }), r },
    Ka = (e, t, n, r) => {
        e.forEach(o => {
            if (typeof o == "string") {
                const s = o === "" ? t : Wd(t, o);
                s.classGroupId = n;
                return
            }
            if (typeof o == "function") {
                if (ex(o)) { Ka(o(r), t, n, r); return }
                t.validators.push({ validator: o, classGroupId: n });
                return
            }
            Object.entries(o).forEach(([s, i]) => { Ka(i, Wd(t, s), n, r) })
        })
    },
    Wd = (e, t) => { let n = e; return t.split(lc).forEach(r => { n.nextPart.has(r) || n.nextPart.set(r, { nextPart: new Map, validators: [] }), n = n.nextPart.get(r) }), n },
    ex = e => e.isThemeGetter,
    tx = (e, t) => t ? e.map(([n, r]) => { const o = r.map(s => typeof s == "string" ? t + s : typeof s == "object" ? Object.fromEntries(Object.entries(s).map(([i, l]) => [t + i, l])) : s); return [n, o] }) : e,
    nx = e => {
        if (e < 1) return { get: () => {}, set: () => {} };
        let t = 0,
            n = new Map,
            r = new Map;
        const o = (s, i) => { n.set(s, i), t++, t > e && (t = 0, r = n, n = new Map) };
        return {get(s) { let i = n.get(s); if (i !== void 0) return i; if ((i = r.get(s)) !== void 0) return o(s, i), i }, set(s, i) { n.has(s) ? n.set(s, i) : o(s, i) } }
    },
    bm = "!",
    rx = e => {
        const { separator: t, experimentalParseClassName: n } = e, r = t.length === 1, o = t[0], s = t.length, i = l => {
            const a = [];
            let u = 0,
                c = 0,
                f;
            for (let x = 0; x < l.length; x++) {
                let h = l[x];
                if (u === 0) { if (h === o && (r || l.slice(x, x + s) === t)) { a.push(l.slice(c, x)), c = x + s; continue } if (h === "/") { f = x; continue } }
                h === "[" ? u++ : h === "]" && u--
            }
            const m = a.length === 0 ? l : l.substring(c),
                d = m.startsWith(bm),
                S = d ? m.substring(1) : m,
                y = f && f > c ? f - c : void 0;
            return { modifiers: a, hasImportantModifier: d, baseClassName: S, maybePostfixModifierPosition: y }
        };
        return n ? l => n({ className: l, parseClassName: i }) : i
    },
    ox = e => { if (e.length <= 1) return e; const t = []; let n = []; return e.forEach(r => { r[0] === "[" ? (t.push(...n.sort(), r), n = []) : n.push(r) }), t.push(...n.sort()), t },
    sx = e => ({ cache: nx(e.cacheSize), parseClassName: rx(e), ...qw(e) }),
    ix = /\s+/,
    lx = (e, t) => {
        const { parseClassName: n, getClassGroupId: r, getConflictingClassGroupIds: o } = t, s = [], i = e.trim().split(ix);
        let l = "";
        for (let a = i.length - 1; a >= 0; a -= 1) {
            const u = i[a],
                { modifiers: c, hasImportantModifier: f, baseClassName: m, maybePostfixModifierPosition: d } = n(u);
            let S = !!d,
                y = r(S ? m.substring(0, d) : m);
            if (!y) {
                if (!S) { l = u + (l.length > 0 ? " " + l : l); continue }
                if (y = r(m), !y) { l = u + (l.length > 0 ? " " + l : l); continue }
                S = !1
            }
            const x = ox(c).join(":"),
                h = f ? x + bm : x,
                p = h + y;
            if (s.includes(p)) continue;
            s.push(p);
            const g = o(y, S);
            for (let E = 0; E < g.length; ++E) {
                const C = g[E];
                s.push(h + C)
            }
            l = u + (l.length > 0 ? " " + l : l)
        }
        return l
    };

function ax() {
    let e = 0,
        t, n, r = "";
    for (; e < arguments.length;)(t = arguments[e++]) && (n = Tm(t)) && (r && (r += " "), r += n);
    return r
}
const Tm = e => { if (typeof e == "string") return e; let t, n = ""; for (let r = 0; r < e.length; r++) e[r] && (t = Tm(e[r])) && (n && (n += " "), n += t); return n };

function ux(e, ...t) {
    let n, r, o, s = i;

    function i(a) { const u = t.reduce((c, f) => f(c), e()); return n = sx(u), r = n.cache.get, o = n.cache.set, s = l, l(a) }

    function l(a) { const u = r(a); if (u) return u; const c = lx(a, n); return o(a, c), c }
    return function() { return s(ax.apply(null, arguments)) }
}
const re = e => { const t = n => n[e] || []; return t.isThemeGetter = !0, t },
    Nm = /^\[(?:([a-z-]+):)?(.+)\]$/i,
    cx = /^\d+\/\d+$/,
    dx = new Set(["px", "full", "screen"]),
    fx = /^(\d+(\.\d+)?)?(xs|sm|md|lg|xl)$/,
    px = /\d+(%|px|r?em|[sdl]?v([hwib]|min|max)|pt|pc|in|cm|mm|cap|ch|ex|r?lh|cq(w|h|i|b|min|max))|\b(calc|min|max|clamp)\(.+\)|^0$/,
    hx = /^(rgba?|hsla?|hwb|(ok)?(lab|lch)|color-mix)\(.+\)$/,
    mx = /^(inset_)?-?((\d+)?\.?(\d+)[a-z]+|0)_-?((\d+)?\.?(\d+)[a-z]+|0)/,
    gx = /^(url|image|image-set|cross-fade|element|(repeating-)?(linear|radial|conic)-gradient)\(.+\)$/,
    zt = e => Dr(e) || dx.has(e) || cx.test(e),
    cn = e => uo(e, "length", kx),
    Dr = e => !!e && !Number.isNaN(Number(e)),
    Ql = e => uo(e, "number", Dr),
    Co = e => !!e && Number.isInteger(Number(e)),
    vx = e => e.endsWith("%") && Dr(e.slice(0, -1)),
    H = e => Nm.test(e),
    dn = e => fx.test(e),
    yx = new Set(["length", "size", "percentage"]),
    wx = e => uo(e, yx, Rm),
    xx = e => uo(e, "position", Rm),
    Sx = new Set(["image", "url"]),
    Ex = e => uo(e, Sx, bx),
    Cx = e => uo(e, "", Px),
    ko = () => !0,
    uo = (e, t, n) => { const r = Nm.exec(e); return r ? r[1] ? typeof t == "string" ? r[1] === t : t.has(r[1]) : n(r[2]) : !1 },
    kx = e => px.test(e) && !hx.test(e),
    Rm = () => !1,
    Px = e => mx.test(e),
    bx = e => gx.test(e),
    Tx = () => {
        const e = re("colors"),
            t = re("spacing"),
            n = re("blur"),
            r = re("brightness"),
            o = re("borderColor"),
            s = re("borderRadius"),
            i = re("borderSpacing"),
            l = re("borderWidth"),
            a = re("contrast"),
            u = re("grayscale"),
            c = re("hueRotate"),
            f = re("invert"),
            m = re("gap"),
            d = re("gradientColorStops"),
            S = re("gradientColorStopPositions"),
            y = re("inset"),
            x = re("margin"),
            h = re("opacity"),
            p = re("padding"),
            g = re("saturate"),
            E = re("scale"),
            C = re("sepia"),
            k = re("skew"),
            P = re("space"),
            N = re("translate"),
            L = () => ["auto", "contain", "none"],
            A = () => ["auto", "hidden", "clip", "visible", "scroll"],
            $ = () => ["auto", H, t],
            D = () => [H, t],
            Q = () => ["", zt, cn],
            O = () => ["auto", Dr, H],
            Y = () => ["bottom", "center", "left", "left-bottom", "left-top", "right", "right-bottom", "right-top", "top"],
            B = () => ["solid", "dashed", "dotted", "double", "none"],
            V = () => ["normal", "multiply", "screen", "overlay", "darken", "lighten", "color-dodge", "color-burn", "hard-light", "soft-light", "difference", "exclusion", "hue", "saturation", "color", "luminosity"],
            T = () => ["start", "end", "center", "between", "around", "evenly", "stretch"],
            R = () => ["", "0", H],
            M = () => ["auto", "avoid", "all", "avoid-page", "page", "left", "right", "column"],
            W = () => [Dr, H];
        return { cacheSize: 500, separator: ":", theme: { colors: [ko], spacing: [zt, cn], blur: ["none", "", dn, H], brightness: W(), borderColor: [e], borderRadius: ["none", "", "full", dn, H], borderSpacing: D(), borderWidth: Q(), contrast: W(), grayscale: R(), hueRotate: W(), invert: R(), gap: D(), gradientColorStops: [e], gradientColorStopPositions: [vx, cn], inset: $(), margin: $(), opacity: W(), padding: D(), saturate: W(), scale: W(), sepia: R(), skew: W(), space: D(), translate: D() }, classGroups: { aspect: [{ aspect: ["auto", "square", "video", H] }], container: ["container"], columns: [{ columns: [dn] }], "break-after": [{ "break-after": M() }], "break-before": [{ "break-before": M() }], "break-inside": [{ "break-inside": ["auto", "avoid", "avoid-page", "avoid-column"] }], "box-decoration": [{ "box-decoration": ["slice", "clone"] }], box: [{ box: ["border", "content"] }], display: ["block", "inline-block", "inline", "flex", "inline-flex", "table", "inline-table", "table-caption", "table-cell", "table-column", "table-column-group", "table-footer-group", "table-header-group", "table-row-group", "table-row", "flow-root", "grid", "inline-grid", "contents", "list-item", "hidden"], float: [{ float: ["right", "left", "none", "start", "end"] }], clear: [{ clear: ["left", "right", "both", "none", "start", "end"] }], isolation: ["isolate", "isolation-auto"], "object-fit": [{ object: ["contain", "cover", "fill", "none", "scale-down"] }], "object-position": [{ object: [...Y(), H] }], overflow: [{ overflow: A() }], "overflow-x": [{ "overflow-x": A() }], "overflow-y": [{ "overflow-y": A() }], overscroll: [{ overscroll: L() }], "overscroll-x": [{ "overscroll-x": L() }], "overscroll-y": [{ "overscroll-y": L() }], position: ["static", "fixed", "absolute", "relative", "sticky"], inset: [{ inset: [y] }], "inset-x": [{ "inset-x": [y] }], "inset-y": [{ "inset-y": [y] }], start: [{ start: [y] }], end: [{ end: [y] }], top: [{ top: [y] }], right: [{ right: [y] }], bottom: [{ bottom: [y] }], left: [{ left: [y] }], visibility: ["visible", "invisible", "collapse"], z: [{ z: ["auto", Co, H] }], basis: [{ basis: $() }], "flex-direction": [{ flex: ["row", "row-reverse", "col", "col-reverse"] }], "flex-wrap": [{ flex: ["wrap", "wrap-reverse", "nowrap"] }], flex: [{ flex: ["1", "auto", "initial", "none", H] }], grow: [{ grow: R() }], shrink: [{ shrink: R() }], order: [{ order: ["first", "last", "none", Co, H] }], "grid-cols": [{ "grid-cols": [ko] }], "col-start-end": [{ col: ["auto", { span: ["full", Co, H] }, H] }], "col-start": [{ "col-start": O() }], "col-end": [{ "col-end": O() }], "grid-rows": [{ "grid-rows": [ko] }], "row-start-end": [{ row: ["auto", { span: [Co, H] }, H] }], "row-start": [{ "row-start": O() }], "row-end": [{ "row-end": O() }], "grid-flow": [{ "grid-flow": ["row", "col", "dense", "row-dense", "col-dense"] }], "auto-cols": [{ "auto-cols": ["auto", "min", "max", "fr", H] }], "auto-rows": [{ "auto-rows": ["auto", "min", "max", "fr", H] }], gap: [{ gap: [m] }], "gap-x": [{ "gap-x": [m] }], "gap-y": [{ "gap-y": [m] }], "justify-content": [{ justify: ["normal", ...T()] }], "justify-items": [{ "justify-items": ["start", "end", "center", "stretch"] }], "justify-self": [{ "justify-self": ["auto", "start", "end", "center", "stretch"] }], "align-content": [{ content: ["normal", ...T(), "baseline"] }], "align-items": [{ items: ["start", "end", "center", "baseline", "stretch"] }], "align-self": [{ self: ["auto", "start", "end", "center", "stretch", "baseline"] }], "place-content": [{ "place-content": [...T(), "baseline"] }], "place-items": [{ "place-items": ["start", "end", "center", "baseline", "stretch"] }], "place-self": [{ "place-self": ["auto", "start", "end", "center", "stretch"] }], p: [{ p: [p] }], px: [{ px: [p] }], py: [{ py: [p] }], ps: [{ ps: [p] }], pe: [{ pe: [p] }], pt: [{ pt: [p] }], pr: [{ pr: [p] }], pb: [{ pb: [p] }], pl: [{ pl: [p] }], m: [{ m: [x] }], mx: [{ mx: [x] }], my: [{ my: [x] }], ms: [{ ms: [x] }], me: [{ me: [x] }], mt: [{ mt: [x] }], mr: [{ mr: [x] }], mb: [{ mb: [x] }], ml: [{ ml: [x] }], "space-x": [{ "space-x": [P] }], "space-x-reverse": ["space-x-reverse"], "space-y": [{ "space-y": [P] }], "space-y-reverse": ["space-y-reverse"], w: [{ w: ["auto", "min", "max", "fit", "svw", "lvw", "dvw", H, t] }], "min-w": [{ "min-w": [H, t, "min", "max", "fit"] }], "max-w": [{ "max-w": [H, t, "none", "full", "min", "max", "fit", "prose", { screen: [dn] }, dn] }], h: [{ h: [H, t, "auto", "min", "max", "fit", "svh", "lvh", "dvh"] }], "min-h": [{ "min-h": [H, t, "min", "max", "fit", "svh", "lvh", "dvh"] }], "max-h": [{ "max-h": [H, t, "min", "max", "fit", "svh", "lvh", "dvh"] }], size: [{ size: [H, t, "auto", "min", "max", "fit"] }], "font-size": [{ text: ["base", dn, cn] }], "font-smoothing": ["antialiased", "subpixel-antialiased"], "font-style": ["italic", "not-italic"], "font-weight": [{ font: ["thin", "extralight", "light", "normal", "medium", "semibold", "bold", "extrabold", "black", Ql] }], "font-family": [{ font: [ko] }], "fvn-normal": ["normal-nums"], "fvn-ordinal": ["ordinal"], "fvn-slashed-zero": ["slashed-zero"], "fvn-figure": ["lining-nums", "oldstyle-nums"], "fvn-spacing": ["proportional-nums", "tabular-nums"], "fvn-fraction": ["diagonal-fractions", "stacked-fractions"], tracking: [{ tracking: ["tighter", "tight", "normal", "wide", "wider", "widest", H] }], "line-clamp": [{ "line-clamp": ["none", Dr, Ql] }], leading: [{ leading: ["none", "tight", "snug", "normal", "relaxed", "loose", zt, H] }], "list-image": [{ "list-image": ["none", H] }], "list-style-type": [{ list: ["none", "disc", "decimal", H] }], "list-style-position": [{ list: ["inside", "outside"] }], "placeholder-color": [{ placeholder: [e] }], "placeholder-opacity": [{ "placeholder-opacity": [h] }], "text-alignment": [{ text: ["left", "center", "right", "justify", "start", "end"] }], "text-color": [{ text: [e] }], "text-opacity": [{ "text-opacity": [h] }], "text-decoration": ["underline", "overline", "line-through", "no-underline"], "text-decoration-style": [{ decoration: [...B(), "wavy"] }], "text-decoration-thickness": [{ decoration: ["auto", "from-font", zt, cn] }], "underline-offset": [{ "underline-offset": ["auto", zt, H] }], "text-decoration-color": [{ decoration: [e] }], "text-transform": ["uppercase", "lowercase", "capitalize", "normal-case"], "text-overflow": ["truncate", "text-ellipsis", "text-clip"], "text-wrap": [{ text: ["wrap", "nowrap", "balance", "pretty"] }], indent: [{ indent: D() }], "vertical-align": [{ align: ["baseline", "top", "middle", "bottom", "text-top", "text-bottom", "sub", "super", H] }], whitespace: [{ whitespace: ["normal", "nowrap", "pre", "pre-line", "pre-wrap", "break-spaces"] }], break: [{ break: ["normal", "words", "all", "keep"] }], hyphens: [{ hyphens: ["none", "manual", "auto"] }], content: [{ content: ["none", H] }], "bg-attachment": [{ bg: ["fixed", "local", "scroll"] }], "bg-clip": [{ "bg-clip": ["border", "padding", "content", "text"] }], "bg-opacity": [{ "bg-opacity": [h] }], "bg-origin": [{ "bg-origin": ["border", "padding", "content"] }], "bg-position": [{ bg: [...Y(), xx] }], "bg-repeat": [{ bg: ["no-repeat", { repeat: ["", "x", "y", "round", "space"] }] }], "bg-size": [{ bg: ["auto", "cover", "contain", wx] }], "bg-image": [{ bg: ["none", { "gradient-to": ["t", "tr", "r", "br", "b", "bl", "l", "tl"] }, Ex] }], "bg-color": [{ bg: [e] }], "gradient-from-pos": [{ from: [S] }], "gradient-via-pos": [{ via: [S] }], "gradient-to-pos": [{ to: [S] }], "gradient-from": [{ from: [d] }], "gradient-via": [{ via: [d] }], "gradient-to": [{ to: [d] }], rounded: [{ rounded: [s] }], "rounded-s": [{ "rounded-s": [s] }], "rounded-e": [{ "rounded-e": [s] }], "rounded-t": [{ "rounded-t": [s] }], "rounded-r": [{ "rounded-r": [s] }], "rounded-b": [{ "rounded-b": [s] }], "rounded-l": [{ "rounded-l": [s] }], "rounded-ss": [{ "rounded-ss": [s] }], "rounded-se": [{ "rounded-se": [s] }], "rounded-ee": [{ "rounded-ee": [s] }], "rounded-es": [{ "rounded-es": [s] }], "rounded-tl": [{ "rounded-tl": [s] }], "rounded-tr": [{ "rounded-tr": [s] }], "rounded-br": [{ "rounded-br": [s] }], "rounded-bl": [{ "rounded-bl": [s] }], "border-w": [{ border: [l] }], "border-w-x": [{ "border-x": [l] }], "border-w-y": [{ "border-y": [l] }], "border-w-s": [{ "border-s": [l] }], "border-w-e": [{ "border-e": [l] }], "border-w-t": [{ "border-t": [l] }], "border-w-r": [{ "border-r": [l] }], "border-w-b": [{ "border-b": [l] }], "border-w-l": [{ "border-l": [l] }], "border-opacity": [{ "border-opacity": [h] }], "border-style": [{ border: [...B(), "hidden"] }], "divide-x": [{ "divide-x": [l] }], "divide-x-reverse": ["divide-x-reverse"], "divide-y": [{ "divide-y": [l] }], "divide-y-reverse": ["divide-y-reverse"], "divide-opacity": [{ "divide-opacity": [h] }], "divide-style": [{ divide: B() }], "border-color": [{ border: [o] }], "border-color-x": [{ "border-x": [o] }], "border-color-y": [{ "border-y": [o] }], "border-color-s": [{ "border-s": [o] }], "border-color-e": [{ "border-e": [o] }], "border-color-t": [{ "border-t": [o] }], "border-color-r": [{ "border-r": [o] }], "border-color-b": [{ "border-b": [o] }], "border-color-l": [{ "border-l": [o] }], "divide-color": [{ divide: [o] }], "outline-style": [{ outline: ["", ...B()] }], "outline-offset": [{ "outline-offset": [zt, H] }], "outline-w": [{ outline: [zt, cn] }], "outline-color": [{ outline: [e] }], "ring-w": [{ ring: Q() }], "ring-w-inset": ["ring-inset"], "ring-color": [{ ring: [e] }], "ring-opacity": [{ "ring-opacity": [h] }], "ring-offset-w": [{ "ring-offset": [zt, cn] }], "ring-offset-color": [{ "ring-offset": [e] }], shadow: [{ shadow: ["", "inner", "none", dn, Cx] }], "shadow-color": [{ shadow: [ko] }], opacity: [{ opacity: [h] }], "mix-blend": [{ "mix-blend": [...V(), "plus-lighter", "plus-darker"] }], "bg-blend": [{ "bg-blend": V() }], filter: [{ filter: ["", "none"] }], blur: [{ blur: [n] }], brightness: [{ brightness: [r] }], contrast: [{ contrast: [a] }], "drop-shadow": [{ "drop-shadow": ["", "none", dn, H] }], grayscale: [{ grayscale: [u] }], "hue-rotate": [{ "hue-rotate": [c] }], invert: [{ invert: [f] }], saturate: [{ saturate: [g] }], sepia: [{ sepia: [C] }], "backdrop-filter": [{ "backdrop-filter": ["", "none"] }], "backdrop-blur": [{ "backdrop-blur": [n] }], "backdrop-brightness": [{ "backdrop-brightness": [r] }], "backdrop-contrast": [{ "backdrop-contrast": [a] }], "backdrop-grayscale": [{ "backdrop-grayscale": [u] }], "backdrop-hue-rotate": [{ "backdrop-hue-rotate": [c] }], "backdrop-invert": [{ "backdrop-invert": [f] }], "backdrop-opacity": [{ "backdrop-opacity": [h] }], "backdrop-saturate": [{ "backdrop-saturate": [g] }], "backdrop-sepia": [{ "backdrop-sepia": [C] }], "border-collapse": [{ border: ["collapse", "separate"] }], "border-spacing": [{ "border-spacing": [i] }], "border-spacing-x": [{ "border-spacing-x": [i] }], "border-spacing-y": [{ "border-spacing-y": [i] }], "table-layout": [{ table: ["auto", "fixed"] }], caption: [{ caption: ["top", "bottom"] }], transition: [{ transition: ["none", "all", "", "colors", "opacity", "shadow", "transform", H] }], duration: [{ duration: W() }], ease: [{ ease: ["linear", "in", "out", "in-out", H] }], delay: [{ delay: W() }], animate: [{ animate: ["none", "spin", "ping", "pulse", "bounce", H] }], transform: [{ transform: ["", "gpu", "none"] }], scale: [{ scale: [E] }], "scale-x": [{ "scale-x": [E] }], "scale-y": [{ "scale-y": [E] }], rotate: [{ rotate: [Co, H] }], "translate-x": [{ "translate-x": [N] }], "translate-y": [{ "translate-y": [N] }], "skew-x": [{ "skew-x": [k] }], "skew-y": [{ "skew-y": [k] }], "transform-origin": [{ origin: ["center", "top", "top-right", "right", "bottom-right", "bottom", "bottom-left", "left", "top-left", H] }], accent: [{ accent: ["auto", e] }], appearance: [{ appearance: ["none", "auto"] }], cursor: [{ cursor: ["auto", "default", "pointer", "wait", "text", "move", "help", "not-allowed", "none", "context-menu", "progress", "cell", "crosshair", "vertical-text", "alias", "copy", "no-drop", "grab", "grabbing", "all-scroll", "col-resize", "row-resize", "n-resize", "e-resize", "s-resize", "w-resize", "ne-resize", "nw-resize", "se-resize", "sw-resize", "ew-resize", "ns-resize", "nesw-resize", "nwse-resize", "zoom-in", "zoom-out", H] }], "caret-color": [{ caret: [e] }], "pointer-events": [{ "pointer-events": ["none", "auto"] }], resize: [{ resize: ["none", "y", "x", ""] }], "scroll-behavior": [{ scroll: ["auto", "smooth"] }], "scroll-m": [{ "scroll-m": D() }], "scroll-mx": [{ "scroll-mx": D() }], "scroll-my": [{ "scroll-my": D() }], "scroll-ms": [{ "scroll-ms": D() }], "scroll-me": [{ "scroll-me": D() }], "scroll-mt": [{ "scroll-mt": D() }], "scroll-mr": [{ "scroll-mr": D() }], "scroll-mb": [{ "scroll-mb": D() }], "scroll-ml": [{ "scroll-ml": D() }], "scroll-p": [{ "scroll-p": D() }], "scroll-px": [{ "scroll-px": D() }], "scroll-py": [{ "scroll-py": D() }], "scroll-ps": [{ "scroll-ps": D() }], "scroll-pe": [{ "scroll-pe": D() }], "scroll-pt": [{ "scroll-pt": D() }], "scroll-pr": [{ "scroll-pr": D() }], "scroll-pb": [{ "scroll-pb": D() }], "scroll-pl": [{ "scroll-pl": D() }], "snap-align": [{ snap: ["start", "end", "center", "align-none"] }], "snap-stop": [{ snap: ["normal", "always"] }], "snap-type": [{ snap: ["none", "x", "y", "both"] }], "snap-strictness": [{ snap: ["mandatory", "proximity"] }], touch: [{ touch: ["auto", "none", "manipulation"] }], "touch-x": [{ "touch-pan": ["x", "left", "right"] }], "touch-y": [{ "touch-pan": ["y", "up", "down"] }], "touch-pz": ["touch-pinch-zoom"], select: [{ select: ["none", "text", "all", "auto"] }], "will-change": [{ "will-change": ["auto", "scroll", "contents", "transform", H] }], fill: [{ fill: [e, "none"] }], "stroke-w": [{ stroke: [zt, cn, Ql] }], stroke: [{ stroke: [e, "none"] }], sr: ["sr-only", "not-sr-only"], "forced-color-adjust": [{ "forced-color-adjust": ["auto", "none"] }] }, conflictingClassGroups: { overflow: ["overflow-x", "overflow-y"], overscroll: ["overscroll-x", "overscroll-y"], inset: ["inset-x", "inset-y", "start", "end", "top", "right", "bottom", "left"], "inset-x": ["right", "left"], "inset-y": ["top", "bottom"], flex: ["basis", "grow", "shrink"], gap: ["gap-x", "gap-y"], p: ["px", "py", "ps", "pe", "pt", "pr", "pb", "pl"], px: ["pr", "pl"], py: ["pt", "pb"], m: ["mx", "my", "ms", "me", "mt", "mr", "mb", "ml"], mx: ["mr", "ml"], my: ["mt", "mb"], size: ["w", "h"], "font-size": ["leading"], "fvn-normal": ["fvn-ordinal", "fvn-slashed-zero", "fvn-figure", "fvn-spacing", "fvn-fraction"], "fvn-ordinal": ["fvn-normal"], "fvn-slashed-zero": ["fvn-normal"], "fvn-figure": ["fvn-normal"], "fvn-spacing": ["fvn-normal"], "fvn-fraction": ["fvn-normal"], "line-clamp": ["display", "overflow"], rounded: ["rounded-s", "rounded-e", "rounded-t", "rounded-r", "rounded-b", "rounded-l", "rounded-ss", "rounded-se", "rounded-ee", "rounded-es", "rounded-tl", "rounded-tr", "rounded-br", "rounded-bl"], "rounded-s": ["rounded-ss", "rounded-es"], "rounded-e": ["rounded-se", "rounded-ee"], "rounded-t": ["rounded-tl", "rounded-tr"], "rounded-r": ["rounded-tr", "rounded-br"], "rounded-b": ["rounded-br", "rounded-bl"], "rounded-l": ["rounded-tl", "rounded-bl"], "border-spacing": ["border-spacing-x", "border-spacing-y"], "border-w": ["border-w-s", "border-w-e", "border-w-t", "border-w-r", "border-w-b", "border-w-l"], "border-w-x": ["border-w-r", "border-w-l"], "border-w-y": ["border-w-t", "border-w-b"], "border-color": ["border-color-s", "border-color-e", "border-color-t", "border-color-r", "border-color-b", "border-color-l"], "border-color-x": ["border-color-r", "border-color-l"], "border-color-y": ["border-color-t", "border-color-b"], "scroll-m": ["scroll-mx", "scroll-my", "scroll-ms", "scroll-me", "scroll-mt", "scroll-mr", "scroll-mb", "scroll-ml"], "scroll-mx": ["scroll-mr", "scroll-ml"], "scroll-my": ["scroll-mt", "scroll-mb"], "scroll-p": ["scroll-px", "scroll-py", "scroll-ps", "scroll-pe", "scroll-pt", "scroll-pr", "scroll-pb", "scroll-pl"], "scroll-px": ["scroll-pr", "scroll-pl"], "scroll-py": ["scroll-pt", "scroll-pb"], touch: ["touch-x", "touch-y", "touch-pz"], "touch-x": ["touch"], "touch-y": ["touch"], "touch-pz": ["touch"] }, conflictingClassGroupModifiers: { "font-size": ["leading"] } }
    },
    Nx = ux(Tx);

function Te(...e) { return Nx(km(e)) }
const Hd = e => typeof e == "boolean" ? `${e}` : e === 0 ? "0" : e,
    Qd = km,
    jm = (e, t) => n => {
        var r;
        if ((t == null ? void 0 : t.variants) == null) return Qd(e, n == null ? void 0 : n.class, n == null ? void 0 : n.className);
        const { variants: o, defaultVariants: s } = t, i = Object.keys(o).map(u => {
            const c = n == null ? void 0 : n[u],
                f = s == null ? void 0 : s[u];
            if (c === null) return null;
            const m = Hd(c) || Hd(f);
            return o[u][m]
        }), l = n && Object.entries(n).reduce((u, c) => { let [f, m] = c; return m === void 0 || (u[f] = m), u }, {}), a = t == null || (r = t.compoundVariants) === null || r === void 0 ? void 0 : r.reduce((u, c) => { let { class: f, className: m, ...d } = c; return Object.entries(d).every(S => { let [y, x] = S; return Array.isArray(x) ? x.includes({...s, ...l }[y]) : {...s, ...l }[y] === x }) ? [...u, f, m] : u }, []);
        return Qd(e, i, a, n == null ? void 0 : n.class, n == null ? void 0 : n.className)
    },
    Rx = _w,
    _m = _.forwardRef(({ className: e, ...t }, n) => v.jsx(mm, { ref: n, className: Te("fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]", e), ...t }));
_m.displayName = mm.displayName;
const jx = jm("group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border p-6 pr-8 shadow-lg transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full", { variants: { variant: { default: "border bg-background text-foreground", destructive: "destructive group border-destructive bg-destructive text-destructive-foreground" } }, defaultVariants: { variant: "default" } }),
    Am = _.forwardRef(({ className: e, variant: t, ...n }, r) => v.jsx(gm, { ref: r, className: Te(jx({ variant: t }), e), ...n }));
Am.displayName = gm.displayName;
const _x = _.forwardRef(({ className: e, ...t }, n) => v.jsx(wm, { ref: n, className: Te("inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-sm font-medium ring-offset-background transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 group-[.destructive]:border-muted/40 group-[.destructive]:hover:border-destructive/30 group-[.destructive]:hover:bg-destructive group-[.destructive]:hover:text-destructive-foreground group-[.destructive]:focus:ring-destructive", e), ...t }));
_x.displayName = wm.displayName;
const Om = _.forwardRef(({ className: e, ...t }, n) => v.jsx(xm, { ref: n, className: Te("absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100 group-[.destructive]:text-red-300 group-[.destructive]:hover:text-red-50 group-[.destructive]:focus:ring-red-400 group-[.destructive]:focus:ring-offset-red-600", e), "toast-close": "", ...t, children: v.jsx(Xw, { className: "h-4 w-4" }) }));
Om.displayName = xm.displayName;
const Lm = _.forwardRef(({ className: e, ...t }, n) => v.jsx(vm, { ref: n, className: Te("text-sm font-semibold", e), ...t }));
Lm.displayName = vm.displayName;
const Mm = _.forwardRef(({ className: e, ...t }, n) => v.jsx(ym, { ref: n, className: Te("text-sm opacity-90", e), ...t }));
Mm.displayName = ym.displayName;

function Ax() { const { toasts: e } = Xh(); return v.jsxs(Rx, { children: [e.map(function({ id: t, title: n, description: r, action: o, ...s }) { return v.jsxs(Am, {...s, children: [v.jsxs("div", { className: "grid gap-1", children: [n && v.jsx(Lm, { children: n }), r && v.jsx(Mm, { children: r })] }), o, v.jsx(Om, {})] }, t) }), v.jsx(_m, {})] }) }
var Kd = ["light", "dark"],
    Ox = "(prefers-color-scheme: dark)",
    Lx = w.createContext(void 0),
    Mx = { setTheme: e => {}, themes: [] },
    Ix = () => { var e; return (e = w.useContext(Lx)) != null ? e : Mx };
w.memo(({ forcedTheme: e, storageKey: t, attribute: n, enableSystem: r, enableColorScheme: o, defaultTheme: s, value: i, attrs: l, nonce: a }) => {
            let u = s === "system",
                c = n === "class" ? `var d=document.documentElement,c=d.classList;${`c.remove(${l.map(S=>`'${S}'`).join(",")})`};`:`var d=document.documentElement,n='${n}',s='setAttribute';`,f=o?Kd.includes(s)&&s?`if(e==='light'||e==='dark'||!e)d.style.colorScheme=e||'${s}'`:"if(e==='light'||e==='dark')d.style.colorScheme=e":"",m=(S,y=!1,x=!0)=>{let h=i?i[S]:S,p=y?S+"|| ''":`'${h}'`,g="";return o&&x&&!y&&Kd.includes(S)&&(g+=`d.style.colorScheme = '${S}';`),n==="class"?y||h?g+=`c.add(${p})`:g+="null":h&&(g+=`d[s](n,${p})`),g},d=e?`!function(){${c}${m(e)}}()`:r?`!function(){try{${c}var e=localStorage.getItem('${t}');if('system'===e||(!e&&${u})){var t='${Ox}',m=window.matchMedia(t);if(m.media!==t||m.matches){${m("dark")}}else{${m("light")}}}else if(e){${i?`var x=${JSON.stringify(i)};`:""}${m(i?"x[e]":"e",!0)}}${u?"":"else{"+m(s,!1,!1)+"}"}${f}}catch(e){}}()`:`!function(){try{${c}var e=localStorage.getItem('${t}');if(e){${i?`var x=${JSON.stringify(i)};`:""}${m(i?"x[e]":"e",!0)}}else{${m(s,!1,!1)};}${f}}catch(t){}}();`;return w.createElement("script",{nonce:a,dangerouslySetInnerHTML:{__html:d}})});var Dx=e=>{switch(e){case"success":return $x;case"info":return Ux;case"warning":return Bx;case"error":return Vx;default:return null}},zx=Array(12).fill(0),Fx=({visible:e,className:t})=>_.createElement("div",{className:["sonner-loading-wrapper",t].filter(Boolean).join(" "),"data-visible":e},_.createElement("div",{className:"sonner-spinner"},zx.map((n,r)=>_.createElement("div",{className:"sonner-loading-bar",key:`spinner-bar-${r}`})))),$x=_.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 20 20",fill:"currentColor",height:"20",width:"20"},_.createElement("path",{fillRule:"evenodd",d:"M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z",clipRule:"evenodd"})),Bx=_.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 24 24",fill:"currentColor",height:"20",width:"20"},_.createElement("path",{fillRule:"evenodd",d:"M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003zM12 8.25a.75.75 0 01.75.75v3.75a.75.75 0 01-1.5 0V9a.75.75 0 01.75-.75zm0 8.25a.75.75 0 100-1.5.75.75 0 000 1.5z",clipRule:"evenodd"})),Ux=_.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 20 20",fill:"currentColor",height:"20",width:"20"},_.createElement("path",{fillRule:"evenodd",d:"M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z",clipRule:"evenodd"})),Vx=_.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 20 20",fill:"currentColor",height:"20",width:"20"},_.createElement("path",{fillRule:"evenodd",d:"M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z",clipRule:"evenodd"})),Wx=_.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"12",viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:"1.5",strokeLinecap:"round",strokeLinejoin:"round"},_.createElement("line",{x1:"18",y1:"6",x2:"6",y2:"18"}),_.createElement("line",{x1:"6",y1:"6",x2:"18",y2:"18"})),Hx=()=>{let[e,t]=_.useState(document.hidden);return _.useEffect(()=>{let n=()=>{t(document.hidden)};return document.addEventListener("visibilitychange",n),()=>window.removeEventListener("visibilitychange",n)},[]),e},Ga=1,Qx=class{constructor(){this.subscribe=e=>(this.subscribers.push(e),()=>{let t=this.subscribers.indexOf(e);this.subscribers.splice(t,1)}),this.publish=e=>{this.subscribers.forEach(t=>t(e))},this.addToast=e=>{this.publish(e),this.toasts=[...this.toasts,e]},this.create=e=>{var t;let{message:n,...r}=e,o=typeof(e==null?void 0:e.id)=="number"||((t=e.id)==null?void 0:t.length)>0?e.id:Ga++,s=this.toasts.find(l=>l.id===o),i=e.dismissible===void 0?!0:e.dismissible;return this.dismissedToasts.has(o)&&this.dismissedToasts.delete(o),s?this.toasts=this.toasts.map(l=>l.id===o?(this.publish({...l,...e,id:o,title:n}),{...l,...e,id:o,dismissible:i,title:n}):l):this.addToast({title:n,...r,dismissible:i,id:o}),o},this.dismiss=e=>(this.dismissedToasts.add(e),e||this.toasts.forEach(t=>{this.subscribers.forEach(n=>n({id:t.id,dismiss:!0}))}),this.subscribers.forEach(t=>t({id:e,dismiss:!0})),e),this.message=(e,t)=>this.create({...t,message:e}),this.error=(e,t)=>this.create({...t,message:e,type:"error"}),this.success=(e,t)=>this.create({...t,type:"success",message:e}),this.info=(e,t)=>this.create({...t,type:"info",message:e}),this.warning=(e,t)=>this.create({...t,type:"warning",message:e}),this.loading=(e,t)=>this.create({...t,type:"loading",message:e}),this.promise=(e,t)=>{if(!t)return;let n;t.loading!==void 0&&(n=this.create({...t,promise:e,type:"loading",message:t.loading,description:typeof t.description!="function"?t.description:void 0}));let r=e instanceof Promise?e:e(),o=n!==void 0,s,i=r.then(async a=>{if(s=["resolve",a],_.isValidElement(a))o=!1,this.create({id:n,type:"default",message:a});else if(Gx(a)&&!a.ok){o=!1;let u=typeof t.error=="function"?await t.error(`HTTP error! status: ${a.status}`):t.error,c=typeof t.description=="function"?await t.description(`HTTP error! status: ${a.status}`):t.description;this.create({id:n,type:"error",message:u,description:c})}else if(t.success!==void 0){o=!1;let u=typeof t.success=="function"?await t.success(a):t.success,c=typeof t.description=="function"?await t.description(a):t.description;this.create({id:n,type:"success",message:u,description:c})}}).catch(async a=>{if(s=["reject",a],t.error!==void 0){o=!1;let u=typeof t.error=="function"?await t.error(a):t.error,c=typeof t.description=="function"?await t.description(a):t.description;this.create({id:n,type:"error",message:u,description:c})}}).finally(()=>{var a;o&&(this.dismiss(n),n=void 0),(a=t.finally)==null||a.call(t)}),l=()=>new Promise((a,u)=>i.then(()=>s[0]==="reject"?u(s[1]):a(s[1])).catch(u));return typeof n!="string"&&typeof n!="number"?{unwrap:l}:Object.assign(n,{unwrap:l})},this.custom=(e,t)=>{let n=(t==null?void 0:t.id)||Ga++;return this.create({jsx:e(n),id:n,...t}),n},this.getActiveToasts=()=>this.toasts.filter(e=>!this.dismissedToasts.has(e.id)),this.subscribers=[],this.toasts=[],this.dismissedToasts=new Set}},Fe=new Qx,Kx=(e,t)=>{let n=(t==null?void 0:t.id)||Ga++;return Fe.addToast({title:e,...t,id:n}),n},Gx=e=>e&&typeof e=="object"&&"ok"in e&&typeof e.ok=="boolean"&&"status"in e&&typeof e.status=="number",Yx=Kx,Xx=()=>Fe.toasts,qx=()=>Fe.getActiveToasts();Object.assign(Yx,{success:Fe.success,info:Fe.info,warning:Fe.warning,error:Fe.error,custom:Fe.custom,message:Fe.message,promise:Fe.promise,dismiss:Fe.dismiss,loading:Fe.loading},{getHistory:Xx,getToasts:qx});function Zx(e,{insertAt:t}={}){if(typeof document>"u")return;let n=document.head||document.getElementsByTagName("head")[0],r=document.createElement("style");r.type="text/css",t==="top"&&n.firstChild?n.insertBefore(r,n.firstChild):n.appendChild(r),r.styleSheet?r.styleSheet.cssText=e:r.appendChild(document.createTextNode(e))}Zx(`:where(html[dir="ltr"]),:where([data-sonner-toaster][dir="ltr"]){--toast-icon-margin-start: -3px;--toast-icon-margin-end: 4px;--toast-svg-margin-start: -1px;--toast-svg-margin-end: 0px;--toast-button-margin-start: auto;--toast-button-margin-end: 0;--toast-close-button-start: 0;--toast-close-button-end: unset;--toast-close-button-transform: translate(-35%, -35%)}:where(html[dir="rtl"]),:where([data-sonner-toaster][dir="rtl"]){--toast-icon-margin-start: 4px;--toast-icon-margin-end: -3px;--toast-svg-margin-start: 0px;--toast-svg-margin-end: -1px;--toast-button-margin-start: 0;--toast-button-margin-end: auto;--toast-close-button-start: unset;--toast-close-button-end: 0;--toast-close-button-transform: translate(35%, -35%)}:where([data-sonner-toaster]){position:fixed;width:var(--width);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji;--gray1: hsl(0, 0%, 99%);--gray2: hsl(0, 0%, 97.3%);--gray3: hsl(0, 0%, 95.1%);--gray4: hsl(0, 0%, 93%);--gray5: hsl(0, 0%, 90.9%);--gray6: hsl(0, 0%, 88.7%);--gray7: hsl(0, 0%, 85.8%);--gray8: hsl(0, 0%, 78%);--gray9: hsl(0, 0%, 56.1%);--gray10: hsl(0, 0%, 52.3%);--gray11: hsl(0, 0%, 43.5%);--gray12: hsl(0, 0%, 9%);--border-radius: 8px;box-sizing:border-box;padding:0;margin:0;list-style:none;outline:none;z-index:999999999;transition:transform .4s ease}:where([data-sonner-toaster][data-lifted="true"]){transform:translateY(-10px)}@media (hover: none) and (pointer: coarse){:where([data-sonner-toaster][data-lifted="true"]){transform:none}}:where([data-sonner-toaster][data-x-position="right"]){right:var(--offset-right)}:where([data-sonner-toaster][data-x-position="left"]){left:var(--offset-left)}:where([data-sonner-toaster][data-x-position="center"]){left:50%;transform:translate(-50%)}:where([data-sonner-toaster][data-y-position="top"]){top:var(--offset-top)}:where([data-sonner-toaster][data-y-position="bottom"]){bottom:var(--offset-bottom)}:where([data-sonner-toast]){--y: translateY(100%);--lift-amount: calc(var(--lift) * var(--gap));z-index:var(--z-index);position:absolute;opacity:0;transform:var(--y);filter:blur(0);touch-action:none;transition:transform .4s,opacity .4s,height .4s,box-shadow .2s;box-sizing:border-box;outline:none;overflow-wrap:anywhere}:where([data-sonner-toast][data-styled="true"]){padding:16px;background:var(--normal-bg);border:1px solid var(--normal-border);color:var(--normal-text);border-radius:var(--border-radius);box-shadow:0 4px 12px #0000001a;width:var(--width);font-size:13px;display:flex;align-items:center;gap:6px}:where([data-sonner-toast]:focus-visible){box-shadow:0 4px 12px #0000001a,0 0 0 2px #0003}:where([data-sonner-toast][data-y-position="top"]){top:0;--y: translateY(-100%);--lift: 1;--lift-amount: calc(1 * var(--gap))}:where([data-sonner-toast][data-y-position="bottom"]){bottom:0;--y: translateY(100%);--lift: -1;--lift-amount: calc(var(--lift) * var(--gap))}:where([data-sonner-toast]) :where([data-description]){font-weight:400;line-height:1.4;color:inherit}:where([data-sonner-toast]) :where([data-title]){font-weight:500;line-height:1.5;color:inherit}:where([data-sonner-toast]) :where([data-icon]){display:flex;height:16px;width:16px;position:relative;justify-content:flex-start;align-items:center;flex-shrink:0;margin-left:var(--toast-icon-margin-start);margin-right:var(--toast-icon-margin-end)}:where([data-sonner-toast][data-promise="true"]) :where([data-icon])>svg{opacity:0;transform:scale(.8);transform-origin:center;animation:sonner-fade-in .3s ease forwards}:where([data-sonner-toast]) :where([data-icon])>*{flex-shrink:0}:where([data-sonner-toast]) :where([data-icon]) svg{margin-left:var(--toast-svg-margin-start);margin-right:var(--toast-svg-margin-end)}:where([data-sonner-toast]) :where([data-content]){display:flex;flex-direction:column;gap:2px}[data-sonner-toast][data-styled=true] [data-button]{border-radius:4px;padding-left:8px;padding-right:8px;height:24px;font-size:12px;color:var(--normal-bg);background:var(--normal-text);margin-left:var(--toast-button-margin-start);margin-right:var(--toast-button-margin-end);border:none;cursor:pointer;outline:none;display:flex;align-items:center;flex-shrink:0;transition:opacity .4s,box-shadow .2s}:where([data-sonner-toast]) :where([data-button]):focus-visible{box-shadow:0 0 0 2px #0006}:where([data-sonner-toast]) :where([data-button]):first-of-type{margin-left:var(--toast-button-margin-start);margin-right:var(--toast-button-margin-end)}:where([data-sonner-toast]) :where([data-cancel]){color:var(--normal-text);background:rgba(0,0,0,.08)}:where([data-sonner-toast][data-theme="dark"]) :where([data-cancel]){background:rgba(255,255,255,.3)}:where([data-sonner-toast]) :where([data-close-button]){position:absolute;left:var(--toast-close-button-start);right:var(--toast-close-button-end);top:0;height:20px;width:20px;display:flex;justify-content:center;align-items:center;padding:0;color:var(--gray12);border:1px solid var(--gray4);transform:var(--toast-close-button-transform);border-radius:50%;cursor:pointer;z-index:1;transition:opacity .1s,background .2s,border-color .2s}[data-sonner-toast] [data-close-button]{background:var(--gray1)}:where([data-sonner-toast]) :where([data-close-button]):focus-visible{box-shadow:0 4px 12px #0000001a,0 0 0 2px #0003}:where([data-sonner-toast]) :where([data-disabled="true"]){cursor:not-allowed}:where([data-sonner-toast]):hover :where([data-close-button]):hover{background:var(--gray2);border-color:var(--gray5)}:where([data-sonner-toast][data-swiping="true"]):before{content:"";position:absolute;left:-50%;right:-50%;height:100%;z-index:-1}:where([data-sonner-toast][data-y-position="top"][data-swiping="true"]):before{bottom:50%;transform:scaleY(3) translateY(50%)}:where([data-sonner-toast][data-y-position="bottom"][data-swiping="true"]):before{top:50%;transform:scaleY(3) translateY(-50%)}:where([data-sonner-toast][data-swiping="false"][data-removed="true"]):before{content:"";position:absolute;inset:0;transform:scaleY(2)}:where([data-sonner-toast]):after{content:"";position:absolute;left:0;height:calc(var(--gap) + 1px);bottom:100%;width:100%}:where([data-sonner-toast][data-mounted="true"]){--y: translateY(0);opacity:1}:where([data-sonner-toast][data-expanded="false"][data-front="false"]){--scale: var(--toasts-before) * .05 + 1;--y: translateY(calc(var(--lift-amount) * var(--toasts-before))) scale(calc(-1 * var(--scale)));height:var(--front-toast-height)}:where([data-sonner-toast])>*{transition:opacity .4s}:where([data-sonner-toast][data-expanded="false"][data-front="false"][data-styled="true"])>*{opacity:0}:where([data-sonner-toast][data-visible="false"]){opacity:0;pointer-events:none}:where([data-sonner-toast][data-mounted="true"][data-expanded="true"]){--y: translateY(calc(var(--lift) * var(--offset)));height:var(--initial-height)}:where([data-sonner-toast][data-removed="true"][data-front="true"][data-swipe-out="false"]){--y: translateY(calc(var(--lift) * -100%));opacity:0}:where([data-sonner-toast][data-removed="true"][data-front="false"][data-swipe-out="false"][data-expanded="true"]){--y: translateY(calc(var(--lift) * var(--offset) + var(--lift) * -100%));opacity:0}:where([data-sonner-toast][data-removed="true"][data-front="false"][data-swipe-out="false"][data-expanded="false"]){--y: translateY(40%);opacity:0;transition:transform .5s,opacity .2s}:where([data-sonner-toast][data-removed="true"][data-front="false"]):before{height:calc(var(--initial-height) + 20%)}[data-sonner-toast][data-swiping=true]{transform:var(--y) translateY(var(--swipe-amount-y, 0px)) translate(var(--swipe-amount-x, 0px));transition:none}[data-sonner-toast][data-swiped=true]{user-select:none}[data-sonner-toast][data-swipe-out=true][data-y-position=bottom],[data-sonner-toast][data-swipe-out=true][data-y-position=top]{animation-duration:.2s;animation-timing-function:ease-out;animation-fill-mode:forwards}[data-sonner-toast][data-swipe-out=true][data-swipe-direction=left]{animation-name:swipe-out-left}[data-sonner-toast][data-swipe-out=true][data-swipe-direction=right]{animation-name:swipe-out-right}[data-sonner-toast][data-swipe-out=true][data-swipe-direction=up]{animation-name:swipe-out-up}[data-sonner-toast][data-swipe-out=true][data-swipe-direction=down]{animation-name:swipe-out-down}@keyframes swipe-out-left{0%{transform:var(--y) translate(var(--swipe-amount-x));opacity:1}to{transform:var(--y) translate(calc(var(--swipe-amount-x) - 100%));opacity:0}}@keyframes swipe-out-right{0%{transform:var(--y) translate(var(--swipe-amount-x));opacity:1}to{transform:var(--y) translate(calc(var(--swipe-amount-x) + 100%));opacity:0}}@keyframes swipe-out-up{0%{transform:var(--y) translateY(var(--swipe-amount-y));opacity:1}to{transform:var(--y) translateY(calc(var(--swipe-amount-y) - 100%));opacity:0}}@keyframes swipe-out-down{0%{transform:var(--y) translateY(var(--swipe-amount-y));opacity:1}to{transform:var(--y) translateY(calc(var(--swipe-amount-y) + 100%));opacity:0}}@media (max-width: 600px){[data-sonner-toaster]{position:fixed;right:var(--mobile-offset-right);left:var(--mobile-offset-left);width:100%}[data-sonner-toaster][dir=rtl]{left:calc(var(--mobile-offset-left) * -1)}[data-sonner-toaster] [data-sonner-toast]{left:0;right:0;width:calc(100% - var(--mobile-offset-left) * 2)}[data-sonner-toaster][data-x-position=left]{left:var(--mobile-offset-left)}[data-sonner-toaster][data-y-position=bottom]{bottom:var(--mobile-offset-bottom)}[data-sonner-toaster][data-y-position=top]{top:var(--mobile-offset-top)}[data-sonner-toaster][data-x-position=center]{left:var(--mobile-offset-left);right:var(--mobile-offset-right);transform:none}}[data-sonner-toaster][data-theme=light]{--normal-bg: #fff;--normal-border: var(--gray4);--normal-text: var(--gray12);--success-bg: hsl(143, 85%, 96%);--success-border: hsl(145, 92%, 91%);--success-text: hsl(140, 100%, 27%);--info-bg: hsl(208, 100%, 97%);--info-border: hsl(221, 91%, 91%);--info-text: hsl(210, 92%, 45%);--warning-bg: hsl(49, 100%, 97%);--warning-border: hsl(49, 91%, 91%);--warning-text: hsl(31, 92%, 45%);--error-bg: hsl(359, 100%, 97%);--error-border: hsl(359, 100%, 94%);--error-text: hsl(360, 100%, 45%)}[data-sonner-toaster][data-theme=light] [data-sonner-toast][data-invert=true]{--normal-bg: #000;--normal-border: hsl(0, 0%, 20%);--normal-text: var(--gray1)}[data-sonner-toaster][data-theme=dark] [data-sonner-toast][data-invert=true]{--normal-bg: #fff;--normal-border: var(--gray3);--normal-text: var(--gray12)}[data-sonner-toaster][data-theme=dark]{--normal-bg: #000;--normal-bg-hover: hsl(0, 0%, 12%);--normal-border: hsl(0, 0%, 20%);--normal-border-hover: hsl(0, 0%, 25%);--normal-text: var(--gray1);--success-bg: hsl(150, 100%, 6%);--success-border: hsl(147, 100%, 12%);--success-text: hsl(150, 86%, 65%);--info-bg: hsl(215, 100%, 6%);--info-border: hsl(223, 100%, 12%);--info-text: hsl(216, 87%, 65%);--warning-bg: hsl(64, 100%, 6%);--warning-border: hsl(60, 100%, 12%);--warning-text: hsl(46, 87%, 65%);--error-bg: hsl(358, 76%, 10%);--error-border: hsl(357, 89%, 16%);--error-text: hsl(358, 100%, 81%)}[data-sonner-toaster][data-theme=dark] [data-sonner-toast] [data-close-button]{background:var(--normal-bg);border-color:var(--normal-border);color:var(--normal-text)}[data-sonner-toaster][data-theme=dark] [data-sonner-toast] [data-close-button]:hover{background:var(--normal-bg-hover);border-color:var(--normal-border-hover)}[data-rich-colors=true][data-sonner-toast][data-type=success],[data-rich-colors=true][data-sonner-toast][data-type=success] [data-close-button]{background:var(--success-bg);border-color:var(--success-border);color:var(--success-text)}[data-rich-colors=true][data-sonner-toast][data-type=info],[data-rich-colors=true][data-sonner-toast][data-type=info] [data-close-button]{background:var(--info-bg);border-color:var(--info-border);color:var(--info-text)}[data-rich-colors=true][data-sonner-toast][data-type=warning],[data-rich-colors=true][data-sonner-toast][data-type=warning] [data-close-button]{background:var(--warning-bg);border-color:var(--warning-border);color:var(--warning-text)}[data-rich-colors=true][data-sonner-toast][data-type=error],[data-rich-colors=true][data-sonner-toast][data-type=error] [data-close-button]{background:var(--error-bg);border-color:var(--error-border);color:var(--error-text)}.sonner-loading-wrapper{--size: 16px;height:var(--size);width:var(--size);position:absolute;inset:0;z-index:10}.sonner-loading-wrapper[data-visible=false]{transform-origin:center;animation:sonner-fade-out .2s ease forwards}.sonner-spinner{position:relative;top:50%;left:50%;height:var(--size);width:var(--size)}.sonner-loading-bar{animation:sonner-spin 1.2s linear infinite;background:var(--gray11);border-radius:6px;height:8%;left:-10%;position:absolute;top:-3.9%;width:24%}.sonner-loading-bar:nth-child(1){animation-delay:-1.2s;transform:rotate(.0001deg) translate(146%)}.sonner-loading-bar:nth-child(2){animation-delay:-1.1s;transform:rotate(30deg) translate(146%)}.sonner-loading-bar:nth-child(3){animation-delay:-1s;transform:rotate(60deg) translate(146%)}.sonner-loading-bar:nth-child(4){animation-delay:-.9s;transform:rotate(90deg) translate(146%)}.sonner-loading-bar:nth-child(5){animation-delay:-.8s;transform:rotate(120deg) translate(146%)}.sonner-loading-bar:nth-child(6){animation-delay:-.7s;transform:rotate(150deg) translate(146%)}.sonner-loading-bar:nth-child(7){animation-delay:-.6s;transform:rotate(180deg) translate(146%)}.sonner-loading-bar:nth-child(8){animation-delay:-.5s;transform:rotate(210deg) translate(146%)}.sonner-loading-bar:nth-child(9){animation-delay:-.4s;transform:rotate(240deg) translate(146%)}.sonner-loading-bar:nth-child(10){animation-delay:-.3s;transform:rotate(270deg) translate(146%)}.sonner-loading-bar:nth-child(11){animation-delay:-.2s;transform:rotate(300deg) translate(146%)}.sonner-loading-bar:nth-child(12){animation-delay:-.1s;transform:rotate(330deg) translate(146%)}@keyframes sonner-fade-in{0%{opacity:0;transform:scale(.8)}to{opacity:1;transform:scale(1)}}@keyframes sonner-fade-out{0%{opacity:1;transform:scale(1)}to{opacity:0;transform:scale(.8)}}@keyframes sonner-spin{0%{opacity:1}to{opacity:.15}}@media (prefers-reduced-motion){[data-sonner-toast],[data-sonner-toast]>*,.sonner-loading-bar{transition:none!important;animation:none!important}}.sonner-loader{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);transform-origin:center;transition:opacity .2s,transform .2s}.sonner-loader[data-visible=false]{opacity:0;transform:scale(.8) translate(-50%,-50%)}
`);function Fs(e){return e.label!==void 0}var Jx=3,e1="32px",t1="16px",Gd=4e3,n1=356,r1=14,o1=20,s1=200;function ht(...e){return e.filter(Boolean).join(" ")}function i1(e){let[t,n]=e.split("-"),r=[];return t&&r.push(t),n&&r.push(n),r}var l1=e=>{var t,n,r,o,s,i,l,a,u,c,f;let{invert:m,toast:d,unstyled:S,interacting:y,setHeights:x,visibleToasts:h,heights:p,index:g,toasts:E,expanded:C,removeToast:k,defaultRichColors:P,closeButton:N,style:L,cancelButtonStyle:A,actionButtonStyle:$,className:D="",descriptionClassName:Q="",duration:O,position:Y,gap:B,loadingIcon:V,expandByDefault:T,classNames:R,icons:M,closeButtonAriaLabel:W="Close toast",pauseWhenPageIsHidden:z}=e,[K,X]=_.useState(null),[he,Ne]=_.useState(null),[J,fr]=_.useState(!1),[nn,Bn]=_.useState(!1),[rn,pr]=_.useState(!1),[on,gs]=_.useState(!1),[fl,vs]=_.useState(!1),[pl,po]=_.useState(0),[hr,Sc]=_.useState(0),ho=_.useRef(d.duration||O||Gd),Ec=_.useRef(null),Un=_.useRef(null),Gg=g===0,Yg=g+1<=h,nt=d.type,mr=d.dismissible!==!1,Xg=d.className||"",qg=d.descriptionClassName||"",ys=_.useMemo(()=>p.findIndex(U=>U.toastId===d.id)||0,[p,d.id]),Zg=_.useMemo(()=>{var U;return(U=d.closeButton)!=null?U:N},[d.closeButton,N]),Cc=_.useMemo(()=>d.duration||O||Gd,[d.duration,O]),hl=_.useRef(0),gr=_.useRef(0),kc=_.useRef(0),vr=_.useRef(null),[Jg,ev]=Y.split("-"),Pc=_.useMemo(()=>p.reduce((U,te,ie)=>ie>=ys?U:U+te.height,0),[p,ys]),bc=Hx(),tv=d.invert||m,ml=nt==="loading";gr.current=_.useMemo(()=>ys*B+Pc,[ys,Pc]),_.useEffect(()=>{ho.current=Cc},[Cc]),_.useEffect(()=>{fr(!0)},[]),_.useEffect(()=>{let U=Un.current;if(U){let te=U.getBoundingClientRect().height;return Sc(te),x(ie=>[{toastId:d.id,height:te,position:d.position},...ie]),()=>x(ie=>ie.filter(ct=>ct.toastId!==d.id))}},[x,d.id]),_.useLayoutEffect(()=>{if(!J)return;let U=Un.current,te=U.style.height;U.style.height="auto";let ie=U.getBoundingClientRect().height;U.style.height=te,Sc(ie),x(ct=>ct.find(dt=>dt.toastId===d.id)?ct.map(dt=>dt.toastId===d.id?{...dt,height:ie}:dt):[{toastId:d.id,height:ie,position:d.position},...ct])},[J,d.title,d.description,x,d.id]);let sn=_.useCallback(()=>{Bn(!0),po(gr.current),x(U=>U.filter(te=>te.toastId!==d.id)),setTimeout(()=>{k(d)},s1)},[d,k,x,gr]);_.useEffect(()=>{if(d.promise&&nt==="loading"||d.duration===1/0||d.type==="loading")return;let U;return C||y||z&&bc?(()=>{if(kc.current<hl.current){let te=new Date().getTime()-hl.current;ho.current=ho.current-te}kc.current=new Date().getTime()})():ho.current!==1/0&&(hl.current=new Date().getTime(),U=setTimeout(()=>{var te;(te=d.onAutoClose)==null||te.call(d,d),sn()},ho.current)),()=>clearTimeout(U)},[C,y,d,nt,z,bc,sn]),_.useEffect(()=>{d.delete&&sn()},[sn,d.delete]);function nv(){var U,te,ie;return M!=null&&M.loading?_.createElement("div",{className:ht(R==null?void 0:R.loader,(U=d==null?void 0:d.classNames)==null?void 0:U.loader,"sonner-loader"),"data-visible":nt==="loading"},M.loading):V?_.createElement("div",{className:ht(R==null?void 0:R.loader,(te=d==null?void 0:d.classNames)==null?void 0:te.loader,"sonner-loader"),"data-visible":nt==="loading"},V):_.createElement(Fx,{className:ht(R==null?void 0:R.loader,(ie=d==null?void 0:d.classNames)==null?void 0:ie.loader),visible:nt==="loading"})}return _.createElement("li",{tabIndex:0,ref:Un,className:ht(D,Xg,R==null?void 0:R.toast,(t=d==null?void 0:d.classNames)==null?void 0:t.toast,R==null?void 0:R.default,R==null?void 0:R[nt],(n=d==null?void 0:d.classNames)==null?void 0:n[nt]),"data-sonner-toast":"","data-rich-colors":(r=d.richColors)!=null?r:P,"data-styled":!(d.jsx||d.unstyled||S),"data-mounted":J,"data-promise":!!d.promise,"data-swiped":fl,"data-removed":nn,"data-visible":Yg,"data-y-position":Jg,"data-x-position":ev,"data-index":g,"data-front":Gg,"data-swiping":rn,"data-dismissible":mr,"data-type":nt,"data-invert":tv,"data-swipe-out":on,"data-swipe-direction":he,"data-expanded":!!(C||T&&J),style:{"--index":g,"--toasts-before":g,"--z-index":E.length-g,"--offset":`${nn?pl:gr.current}px`,"--initial-height":T?"auto":`${hr}px`,...L,...d.style},onDragEnd:()=>{pr(!1),X(null),vr.current=null},onPointerDown:U=>{ml||!mr||(Ec.current=new Date,po(gr.current),U.target.setPointerCapture(U.pointerId),U.target.tagName!=="BUTTON"&&(pr(!0),vr.current={x:U.clientX,y:U.clientY}))},onPointerUp:()=>{var U,te,ie,ct;if(on||!mr)return;vr.current=null;let dt=Number(((U=Un.current)==null?void 0:U.style.getPropertyValue("--swipe-amount-x").replace("px",""))||0),ln=Number(((te=Un.current)==null?void 0:te.style.getPropertyValue("--swipe-amount-y").replace("px",""))||0),Vn=new Date().getTime()-((ie=Ec.current)==null?void 0:ie.getTime()),ft=K==="x"?dt:ln,an=Math.abs(ft)/Vn;if(Math.abs(ft)>=o1||an>.11){po(gr.current),(ct=d.onDismiss)==null||ct.call(d,d),Ne(K==="x"?dt>0?"right":"left":ln>0?"down":"up"),sn(),gs(!0),vs(!1);return}pr(!1),X(null)},onPointerMove:U=>{var te,ie,ct,dt;if(!vr.current||!mr||((te=window.getSelection())==null?void 0:te.toString().length)>0)return;let ln=U.clientY-vr.current.y,Vn=U.clientX-vr.current.x,ft=(ie=e.swipeDirections)!=null?ie:i1(Y);!K&&(Math.abs(Vn)>1||Math.abs(ln)>1)&&X(Math.abs(Vn)>Math.abs(ln)?"x":"y");let an={x:0,y:0};K==="y"?(ft.includes("top")||ft.includes("bottom"))&&(ft.includes("top")&&ln<0||ft.includes("bottom")&&ln>0)&&(an.y=ln):K==="x"&&(ft.includes("left")||ft.includes("right"))&&(ft.includes("left")&&Vn<0||ft.includes("right")&&Vn>0)&&(an.x=Vn),(Math.abs(an.x)>0||Math.abs(an.y)>0)&&vs(!0),(ct=Un.current)==null||ct.style.setProperty("--swipe-amount-x",`${an.x}px`),(dt=Un.current)==null||dt.style.setProperty("--swipe-amount-y",`${an.y}px`)}},Zg&&!d.jsx?_.createElement("button",{"aria-label":W,"data-disabled":ml,"data-close-button":!0,onClick:ml||!mr?()=>{}:()=>{var U;sn(),(U=d.onDismiss)==null||U.call(d,d)},className:ht(R==null?void 0:R.closeButton,(o=d==null?void 0:d.classNames)==null?void 0:o.closeButton)},(s=M==null?void 0:M.close)!=null?s:Wx):null,d.jsx||w.isValidElement(d.title)?d.jsx?d.jsx:typeof d.title=="function"?d.title():d.title:_.createElement(_.Fragment,null,nt||d.icon||d.promise?_.createElement("div",{"data-icon":"",className:ht(R==null?void 0:R.icon,(i=d==null?void 0:d.classNames)==null?void 0:i.icon)},d.promise||d.type==="loading"&&!d.icon?d.icon||nv():null,d.type!=="loading"?d.icon||(M==null?void 0:M[nt])||Dx(nt):null):null,_.createElement("div",{"data-content":"",className:ht(R==null?void 0:R.content,(l=d==null?void 0:d.classNames)==null?void 0:l.content)},_.createElement("div",{"data-title":"",className:ht(R==null?void 0:R.title,(a=d==null?void 0:d.classNames)==null?void 0:a.title)},typeof d.title=="function"?d.title():d.title),d.description?_.createElement("div",{"data-description":"",className:ht(Q,qg,R==null?void 0:R.description,(u=d==null?void 0:d.classNames)==null?void 0:u.description)},typeof d.description=="function"?d.description():d.description):null),w.isValidElement(d.cancel)?d.cancel:d.cancel&&Fs(d.cancel)?_.createElement("button",{"data-button":!0,"data-cancel":!0,style:d.cancelButtonStyle||A,onClick:U=>{var te,ie;Fs(d.cancel)&&mr&&((ie=(te=d.cancel).onClick)==null||ie.call(te,U),sn())},className:ht(R==null?void 0:R.cancelButton,(c=d==null?void 0:d.classNames)==null?void 0:c.cancelButton)},d.cancel.label):null,w.isValidElement(d.action)?d.action:d.action&&Fs(d.action)?_.createElement("button",{"data-button":!0,"data-action":!0,style:d.actionButtonStyle||$,onClick:U=>{var te,ie;Fs(d.action)&&((ie=(te=d.action).onClick)==null||ie.call(te,U),!U.defaultPrevented&&sn())},className:ht(R==null?void 0:R.actionButton,(f=d==null?void 0:d.classNames)==null?void 0:f.actionButton)},d.action.label):null))};function Yd(){if(typeof window>"u"||typeof document>"u")return"ltr";let e=document.documentElement.getAttribute("dir");return e==="auto"||!e?window.getComputedStyle(document.documentElement).direction:e}function a1(e,t){let n={};return[e,t].forEach((r,o)=>{let s=o===1,i=s?"--mobile-offset":"--offset",l=s?t1:e1;function a(u){["top","right","bottom","left"].forEach(c=>{n[`${i}-${c}`]=typeof u=="number"?`${u}px`:u})}typeof r=="number"||typeof r=="string"?a(r):typeof r=="object"?["top","right","bottom","left"].forEach(u=>{r[u]===void 0?n[`${i}-${u}`]=l:n[`${i}-${u}`]=typeof r[u]=="number"?`${r[u]}px`:r[u]}):a(l)}),n}var u1=w.forwardRef(function(e,t){let{invert:n,position:r="bottom-right",hotkey:o=["altKey","KeyT"],expand:s,closeButton:i,className:l,offset:a,mobileOffset:u,theme:c="light",richColors:f,duration:m,style:d,visibleToasts:S=Jx,toastOptions:y,dir:x=Yd(),gap:h=r1,loadingIcon:p,icons:g,containerAriaLabel:E="Notifications",pauseWhenPageIsHidden:C}=e,[k,P]=_.useState([]),N=_.useMemo(()=>Array.from(new Set([r].concat(k.filter(z=>z.position).map(z=>z.position)))),[k,r]),[L,A]=_.useState([]),[$,D]=_.useState(!1),[Q,O]=_.useState(!1),[Y,B]=_.useState(c!=="system"?c:typeof window<"u"&&window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"),V=_.useRef(null),T=o.join("+").replace(/Key/g,"").replace(/Digit/g,""),R=_.useRef(null),M=_.useRef(!1),W=_.useCallback(z=>{P(K=>{var X;return(X=K.find(he=>he.id===z.id))!=null&&X.delete||Fe.dismiss(z.id),K.filter(({id:he})=>he!==z.id)})},[]);return _.useEffect(()=>Fe.subscribe(z=>{if(z.dismiss){P(K=>K.map(X=>X.id===z.id?{...X,delete:!0}:X));return}setTimeout(()=>{Gh.flushSync(()=>{P(K=>{let X=K.findIndex(he=>he.id===z.id);return X!==-1?[...K.slice(0,X),{...K[X],...z},...K.slice(X+1)]:[z,...K]})})})}),[]),_.useEffect(()=>{if(c!=="system"){B(c);return}if(c==="system"&&(window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches?B("dark"):B("light")),typeof window>"u")return;let z=window.matchMedia("(prefers-color-scheme: dark)");try{z.addEventListener("change",({matches:K})=>{B(K?"dark":"light")})}catch{z.addListener(({matches:X})=>{try{B(X?"dark":"light")}catch(he){console.error(he)}})}},[c]),_.useEffect(()=>{k.length<=1&&D(!1)},[k]),_.useEffect(()=>{let z=K=>{var X,he;o.every(Ne=>K[Ne]||K.code===Ne)&&(D(!0),(X=V.current)==null||X.focus()),K.code==="Escape"&&(document.activeElement===V.current||(he=V.current)!=null&&he.contains(document.activeElement))&&D(!1)};return document.addEventListener("keydown",z),()=>document.removeEventListener("keydown",z)},[o]),_.useEffect(()=>{if(V.current)return()=>{R.current&&(R.current.focus({preventScroll:!0}),R.current=null,M.current=!1)}},[V.current]),_.createElement("section",{ref:t,"aria-label":`${E} ${T}`,tabIndex:-1,"aria-live":"polite","aria-relevant":"additions text","aria-atomic":"false",suppressHydrationWarning:!0},N.map((z,K)=>{var X;let[he,Ne]=z.split("-");return k.length?_.createElement("ol",{key:z,dir:x==="auto"?Yd():x,tabIndex:-1,ref:V,className:l,"data-sonner-toaster":!0,"data-theme":Y,"data-y-position":he,"data-lifted":$&&k.length>1&&!s,"data-x-position":Ne,style:{"--front-toast-height":`${((X=L[0])==null?void 0:X.height)||0}px`,"--width":`${n1}px`,"--gap":`${h}px`,...d,...a1(a,u)},onBlur:J=>{M.current&&!J.currentTarget.contains(J.relatedTarget)&&(M.current=!1,R.current&&(R.current.focus({preventScroll:!0}),R.current=null))},onFocus:J=>{J.target instanceof HTMLElement&&J.target.dataset.dismissible==="false"||M.current||(M.current=!0,R.current=J.relatedTarget)},onMouseEnter:()=>D(!0),onMouseMove:()=>D(!0),onMouseLeave:()=>{Q||D(!1)},onDragEnd:()=>D(!1),onPointerDown:J=>{J.target instanceof HTMLElement&&J.target.dataset.dismissible==="false"||O(!0)},onPointerUp:()=>O(!1)},k.filter(J=>!J.position&&K===0||J.position===z).map((J,fr)=>{var nn,Bn;return _.createElement(l1,{key:J.id,icons:g,index:fr,toast:J,defaultRichColors:f,duration:(nn=y==null?void 0:y.duration)!=null?nn:m,className:y==null?void 0:y.className,descriptionClassName:y==null?void 0:y.descriptionClassName,invert:n,visibleToasts:S,closeButton:(Bn=y==null?void 0:y.closeButton)!=null?Bn:i,interacting:Q,position:z,style:y==null?void 0:y.style,unstyled:y==null?void 0:y.unstyled,classNames:y==null?void 0:y.classNames,cancelButtonStyle:y==null?void 0:y.cancelButtonStyle,actionButtonStyle:y==null?void 0:y.actionButtonStyle,removeToast:W,toasts:k.filter(rn=>rn.position==J.position),heights:L.filter(rn=>rn.position==J.position),setHeights:A,expandByDefault:s,gap:h,loadingIcon:p,expanded:$,pauseWhenPageIsHidden:C,swipeDirections:e.swipeDirections})})):null}))});const c1=e=>{const{theme:t="system"}=Ix();return v.jsx(u1,{theme:t,className:"toaster group",toastOptions:{classNames:{toast:"group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg",description:"group-[.toast]:text-muted-foreground",actionButton:"group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",cancelButton:"group-[.toast]:bg-muted group-[.toast]:text-muted-foreground"}},...e})},d1=["top","right","bottom","left"],In=Math.min,Ye=Math.max,Ni=Math.round,$s=Math.floor,It=e=>({x:e,y:e}),f1={left:"right",right:"left",bottom:"top",top:"bottom"};function Ya(e,t,n){return Ye(e,In(t,n))}function Zt(e,t){return typeof e=="function"?e(t):e}function Jt(e){return e.split("-")[0]}function co(e){return e.split("-")[1]}function ac(e){return e==="x"?"y":"x"}function uc(e){return e==="y"?"height":"width"}function Ot(e){const t=e[0];return t==="t"||t==="b"?"y":"x"}function cc(e){return ac(Ot(e))}function p1(e,t,n){n===void 0&&(n=!1);const r=co(e),o=cc(e),s=uc(o);let i=o==="x"?r===(n?"end":"start")?"right":"left":r==="start"?"bottom":"top";return t.reference[s]>t.floating[s]&&(i=Ri(i)),[i,Ri(i)]}function h1(e){const t=Ri(e);return[Xa(e),t,Xa(t)]}function Xa(e){return e.includes("start")?e.replace("start","end"):e.replace("end","start")}const Xd=["left","right"],qd=["right","left"],m1=["top","bottom"],g1=["bottom","top"];function v1(e,t,n){switch(e){case"top":case"bottom":return n?t?qd:Xd:t?Xd:qd;case"left":case"right":return t?m1:g1;default:return[]}}function y1(e,t,n,r){const o=co(e);let s=v1(Jt(e),n==="start",r);return o&&(s=s.map(i=>i+"-"+o),t&&(s=s.concat(s.map(Xa)))),s}function Ri(e){const t=Jt(e);return f1[t]+e.slice(t.length)}function w1(e){return{top:0,right:0,bottom:0,left:0,...e}}function Im(e){return typeof e!="number"?w1(e):{top:e,right:e,bottom:e,left:e}}function ji(e){const{x:t,y:n,width:r,height:o}=e;return{width:r,height:o,top:n,left:t,right:t+r,bottom:n+o,x:t,y:n}}function Zd(e,t,n){let{reference:r,floating:o}=e;const s=Ot(t),i=cc(t),l=uc(i),a=Jt(t),u=s==="y",c=r.x+r.width/2-o.width/2,f=r.y+r.height/2-o.height/2,m=r[l]/2-o[l]/2;let d;switch(a){case"top":d={x:c,y:r.y-o.height};break;case"bottom":d={x:c,y:r.y+r.height};break;case"right":d={x:r.x+r.width,y:f};break;case"left":d={x:r.x-o.width,y:f};break;default:d={x:r.x,y:r.y}}switch(co(t)){case"start":d[i]-=m*(n&&u?-1:1);break;case"end":d[i]+=m*(n&&u?-1:1);break}return d}async function x1(e,t){var n;t===void 0&&(t={});const{x:r,y:o,platform:s,rects:i,elements:l,strategy:a}=e,{boundary:u="clippingAncestors",rootBoundary:c="viewport",elementContext:f="floating",altBoundary:m=!1,padding:d=0}=Zt(t,e),S=Im(d),x=l[m?f==="floating"?"reference":"floating":f],h=ji(await s.getClippingRect({element:(n=await(s.isElement==null?void 0:s.isElement(x)))==null||n?x:x.contextElement||await(s.getDocumentElement==null?void 0:s.getDocumentElement(l.floating)),boundary:u,rootBoundary:c,strategy:a})),p=f==="floating"?{x:r,y:o,width:i.floating.width,height:i.floating.height}:i.reference,g=await(s.getOffsetParent==null?void 0:s.getOffsetParent(l.floating)),E=await(s.isElement==null?void 0:s.isElement(g))?await(s.getScale==null?void 0:s.getScale(g))||{x:1,y:1}:{x:1,y:1},C=ji(s.convertOffsetParentRelativeRectToViewportRelativeRect?await s.convertOffsetParentRelativeRectToViewportRelativeRect({elements:l,rect:p,offsetParent:g,strategy:a}):p);return{top:(h.top-C.top+S.top)/E.y,bottom:(C.bottom-h.bottom+S.bottom)/E.y,left:(h.left-C.left+S.left)/E.x,right:(C.right-h.right+S.right)/E.x}}const S1=50,E1=async(e,t,n)=>{const{placement:r="bottom",strategy:o="absolute",middleware:s=[],platform:i}=n,l=i.detectOverflow?i:{...i,detectOverflow:x1},a=await(i.isRTL==null?void 0:i.isRTL(t));let u=await i.getElementRects({reference:e,floating:t,strategy:o}),{x:c,y:f}=Zd(u,r,a),m=r,d=0;const S={};for(let y=0;y<s.length;y++){const x=s[y];if(!x)continue;const{name:h,fn:p}=x,{x:g,y:E,data:C,reset:k}=await p({x:c,y:f,initialPlacement:r,placement:m,strategy:o,middlewareData:S,rects:u,platform:l,elements:{reference:e,floating:t}});c=g??c,f=E??f,S[h]={...S[h],...C},k&&d<S1&&(d++,typeof k=="object"&&(k.placement&&(m=k.placement),k.rects&&(u=k.rects===!0?await i.getElementRects({reference:e,floating:t,strategy:o}):k.rects),{x:c,y:f}=Zd(u,m,a)),y=-1)}return{x:c,y:f,placement:m,strategy:o,middlewareData:S}},C1=e=>({name:"arrow",options:e,async fn(t){const{x:n,y:r,placement:o,rects:s,platform:i,elements:l,middlewareData:a}=t,{element:u,padding:c=0}=Zt(e,t)||{};if(u==null)return{};const f=Im(c),m={x:n,y:r},d=cc(o),S=uc(d),y=await i.getDimensions(u),x=d==="y",h=x?"top":"left",p=x?"bottom":"right",g=x?"clientHeight":"clientWidth",E=s.reference[S]+s.reference[d]-m[d]-s.floating[S],C=m[d]-s.reference[d],k=await(i.getOffsetParent==null?void 0:i.getOffsetParent(u));let P=k?k[g]:0;(!P||!await(i.isElement==null?void 0:i.isElement(k)))&&(P=l.floating[g]||s.floating[S]);const N=E/2-C/2,L=P/2-y[S]/2-1,A=In(f[h],L),$=In(f[p],L),D=A,Q=P-y[S]-$,O=P/2-y[S]/2+N,Y=Ya(D,O,Q),B=!a.arrow&&co(o)!=null&&O!==Y&&s.reference[S]/2-(O<D?A:$)-y[S]/2<0,V=B?O<D?O-D:O-Q:0;return{[d]:m[d]+V,data:{[d]:Y,centerOffset:O-Y-V,...B&&{alignmentOffset:V}},reset:B}}}),k1=function(e){return e===void 0&&(e={}),{name:"flip",options:e,async fn(t){var n,r;const{placement:o,middlewareData:s,rects:i,initialPlacement:l,platform:a,elements:u}=t,{mainAxis:c=!0,crossAxis:f=!0,fallbackPlacements:m,fallbackStrategy:d="bestFit",fallbackAxisSideDirection:S="none",flipAlignment:y=!0,...x}=Zt(e,t);if((n=s.arrow)!=null&&n.alignmentOffset)return{};const h=Jt(o),p=Ot(l),g=Jt(l)===l,E=await(a.isRTL==null?void 0:a.isRTL(u.floating)),C=m||(g||!y?[Ri(l)]:h1(l)),k=S!=="none";!m&&k&&C.push(...y1(l,y,S,E));const P=[l,...C],N=await a.detectOverflow(t,x),L=[];let A=((r=s.flip)==null?void 0:r.overflows)||[];if(c&&L.push(N[h]),f){const O=p1(o,i,E);L.push(N[O[0]],N[O[1]])}if(A=[...A,{placement:o,overflows:L}],!L.every(O=>O<=0)){var $,D;const O=((($=s.flip)==null?void 0:$.index)||0)+1,Y=P[O];if(Y&&(!(f==="alignment"?p!==Ot(Y):!1)||A.every(T=>Ot(T.placement)===p?T.overflows[0]>0:!0)))return{data:{index:O,overflows:A},reset:{placement:Y}};let B=(D=A.filter(V=>V.overflows[0]<=0).sort((V,T)=>V.overflows[1]-T.overflows[1])[0])==null?void 0:D.placement;if(!B)switch(d){case"bestFit":{var Q;const V=(Q=A.filter(T=>{if(k){const R=Ot(T.placement);return R===p||R==="y"}return!0}).map(T=>[T.placement,T.overflows.filter(R=>R>0).reduce((R,M)=>R+M,0)]).sort((T,R)=>T[1]-R[1])[0])==null?void 0:Q[0];V&&(B=V);break}case"initialPlacement":B=l;break}if(o!==B)return{reset:{placement:B}}}return{}}}};function Jd(e,t){return{top:e.top-t.height,right:e.right-t.width,bottom:e.bottom-t.height,left:e.left-t.width}}function ef(e){return d1.some(t=>e[t]>=0)}const P1=function(e){return e===void 0&&(e={}),{name:"hide",options:e,async fn(t){const{rects:n,platform:r}=t,{strategy:o="referenceHidden",...s}=Zt(e,t);switch(o){case"referenceHidden":{const i=await r.detectOverflow(t,{...s,elementContext:"reference"}),l=Jd(i,n.reference);return{data:{referenceHiddenOffsets:l,referenceHidden:ef(l)}}}case"escaped":{const i=await r.detectOverflow(t,{...s,altBoundary:!0}),l=Jd(i,n.floating);return{data:{escapedOffsets:l,escaped:ef(l)}}}default:return{}}}}},Dm=new Set(["left","top"]);async function b1(e,t){const{placement:n,platform:r,elements:o}=e,s=await(r.isRTL==null?void 0:r.isRTL(o.floating)),i=Jt(n),l=co(n),a=Ot(n)==="y",u=Dm.has(i)?-1:1,c=s&&a?-1:1,f=Zt(t,e);let{mainAxis:m,crossAxis:d,alignmentAxis:S}=typeof f=="number"?{mainAxis:f,crossAxis:0,alignmentAxis:null}:{mainAxis:f.mainAxis||0,crossAxis:f.crossAxis||0,alignmentAxis:f.alignmentAxis};return l&&typeof S=="number"&&(d=l==="end"?S*-1:S),a?{x:d*c,y:m*u}:{x:m*u,y:d*c}}const T1=function(e){return e===void 0&&(e=0),{name:"offset",options:e,async fn(t){var n,r;const{x:o,y:s,placement:i,middlewareData:l}=t,a=await b1(t,e);return i===((n=l.offset)==null?void 0:n.placement)&&(r=l.arrow)!=null&&r.alignmentOffset?{}:{x:o+a.x,y:s+a.y,data:{...a,placement:i}}}}},N1=function(e){return e===void 0&&(e={}),{name:"shift",options:e,async fn(t){const{x:n,y:r,placement:o,platform:s}=t,{mainAxis:i=!0,crossAxis:l=!1,limiter:a={fn:h=>{let{x:p,y:g}=h;return{x:p,y:g}}},...u}=Zt(e,t),c={x:n,y:r},f=await s.detectOverflow(t,u),m=Ot(Jt(o)),d=ac(m);let S=c[d],y=c[m];if(i){const h=d==="y"?"top":"left",p=d==="y"?"bottom":"right",g=S+f[h],E=S-f[p];S=Ya(g,S,E)}if(l){const h=m==="y"?"top":"left",p=m==="y"?"bottom":"right",g=y+f[h],E=y-f[p];y=Ya(g,y,E)}const x=a.fn({...t,[d]:S,[m]:y});return{...x,data:{x:x.x-n,y:x.y-r,enabled:{[d]:i,[m]:l}}}}}},R1=function(e){return e===void 0&&(e={}),{options:e,fn(t){const{x:n,y:r,placement:o,rects:s,middlewareData:i}=t,{offset:l=0,mainAxis:a=!0,crossAxis:u=!0}=Zt(e,t),c={x:n,y:r},f=Ot(o),m=ac(f);let d=c[m],S=c[f];const y=Zt(l,t),x=typeof y=="number"?{mainAxis:y,crossAxis:0}:{mainAxis:0,crossAxis:0,...y};if(a){const g=m==="y"?"height":"width",E=s.reference[m]-s.floating[g]+x.mainAxis,C=s.reference[m]+s.reference[g]-x.mainAxis;d<E?d=E:d>C&&(d=C)}if(u){var h,p;const g=m==="y"?"width":"height",E=Dm.has(Jt(o)),C=s.reference[f]-s.floating[g]+(E&&((h=i.offset)==null?void 0:h[f])||0)+(E?0:x.crossAxis),k=s.reference[f]+s.reference[g]+(E?0:((p=i.offset)==null?void 0:p[f])||0)-(E?x.crossAxis:0);S<C?S=C:S>k&&(S=k)}return{[m]:d,[f]:S}}}},j1=function(e){return e===void 0&&(e={}),{name:"size",options:e,async fn(t){var n,r;const{placement:o,rects:s,platform:i,elements:l}=t,{apply:a=()=>{},...u}=Zt(e,t),c=await i.detectOverflow(t,u),f=Jt(o),m=co(o),d=Ot(o)==="y",{width:S,height:y}=s.floating;let x,h;f==="top"||f==="bottom"?(x=f,h=m===(await(i.isRTL==null?void 0:i.isRTL(l.floating))?"start":"end")?"left":"right"):(h=f,x=m==="end"?"top":"bottom");const p=y-c.top-c.bottom,g=S-c.left-c.right,E=In(y-c[x],p),C=In(S-c[h],g),k=!t.middlewareData.shift;let P=E,N=C;if((n=t.middlewareData.shift)!=null&&n.enabled.x&&(N=g),(r=t.middlewareData.shift)!=null&&r.enabled.y&&(P=p),k&&!m){const A=Ye(c.left,0),$=Ye(c.right,0),D=Ye(c.top,0),Q=Ye(c.bottom,0);d?N=S-2*(A!==0||$!==0?A+$:Ye(c.left,c.right)):P=y-2*(D!==0||Q!==0?D+Q:Ye(c.top,c.bottom))}await a({...t,availableWidth:N,availableHeight:P});const L=await i.getDimensions(l.floating);return S!==L.width||y!==L.height?{reset:{rects:!0}}:{}}}};function nl(){return typeof window<"u"}function fo(e){return zm(e)?(e.nodeName||"").toLowerCase():"#document"}function Ze(e){var t;return(e==null||(t=e.ownerDocument)==null?void 0:t.defaultView)||window}function Dt(e){var t;return(t=(zm(e)?e.ownerDocument:e.document)||window.document)==null?void 0:t.documentElement}function zm(e){return nl()?e instanceof Node||e instanceof Ze(e).Node:!1}function Pt(e){return nl()?e instanceof Element||e instanceof Ze(e).Element:!1}function tn(e){return nl()?e instanceof HTMLElement||e instanceof Ze(e).HTMLElement:!1}function tf(e){return!nl()||typeof ShadowRoot>"u"?!1:e instanceof ShadowRoot||e instanceof Ze(e).ShadowRoot}function ms(e){const{overflow:t,overflowX:n,overflowY:r,display:o}=bt(e);return/auto|scroll|overlay|hidden|clip/.test(t+r+n)&&o!=="inline"&&o!=="contents"}function _1(e){return/^(table|td|th)$/.test(fo(e))}function rl(e){try{if(e.matches(":popover-open"))return!0}catch{}try{return e.matches(":modal")}catch{return!1}}const A1=/transform|translate|scale|rotate|perspective|filter/,O1=/paint|layout|strict|content/,Wn=e=>!!e&&e!=="none";let Kl;function dc(e){const t=Pt(e)?bt(e):e;return Wn(t.transform)||Wn(t.translate)||Wn(t.scale)||Wn(t.rotate)||Wn(t.perspective)||!fc()&&(Wn(t.backdropFilter)||Wn(t.filter))||A1.test(t.willChange||"")||O1.test(t.contain||"")}function L1(e){let t=Dn(e);for(;tn(t)&&!ro(t);){if(dc(t))return t;if(rl(t))return null;t=Dn(t)}return null}function fc(){return Kl==null&&(Kl=typeof CSS<"u"&&CSS.supports&&CSS.supports("-webkit-backdrop-filter","none")),Kl}function ro(e){return/^(html|body|#document)$/.test(fo(e))}function bt(e){return Ze(e).getComputedStyle(e)}function ol(e){return Pt(e)?{scrollLeft:e.scrollLeft,scrollTop:e.scrollTop}:{scrollLeft:e.scrollX,scrollTop:e.scrollY}}function Dn(e){if(fo(e)==="html")return e;const t=e.assignedSlot||e.parentNode||tf(e)&&e.host||Dt(e);return tf(t)?t.host:t}function Fm(e){const t=Dn(e);return ro(t)?e.ownerDocument?e.ownerDocument.body:e.body:tn(t)&&ms(t)?t:Fm(t)}function ns(e,t,n){var r;t===void 0&&(t=[]),n===void 0&&(n=!0);const o=Fm(e),s=o===((r=e.ownerDocument)==null?void 0:r.body),i=Ze(o);if(s){const l=qa(i);return t.concat(i,i.visualViewport||[],ms(o)?o:[],l&&n?ns(l):[])}else return t.concat(o,ns(o,[],n))}function qa(e){return e.parent&&Object.getPrototypeOf(e.parent)?e.frameElement:null}function $m(e){const t=bt(e);let n=parseFloat(t.width)||0,r=parseFloat(t.height)||0;const o=tn(e),s=o?e.offsetWidth:n,i=o?e.offsetHeight:r,l=Ni(n)!==s||Ni(r)!==i;return l&&(n=s,r=i),{width:n,height:r,$:l}}function pc(e){return Pt(e)?e:e.contextElement}function zr(e){const t=pc(e);if(!tn(t))return It(1);const n=t.getBoundingClientRect(),{width:r,height:o,$:s}=$m(t);let i=(s?Ni(n.width):n.width)/r,l=(s?Ni(n.height):n.height)/o;return(!i||!Number.isFinite(i))&&(i=1),(!l||!Number.isFinite(l))&&(l=1),{x:i,y:l}}const M1=It(0);function Bm(e){const t=Ze(e);return!fc()||!t.visualViewport?M1:{x:t.visualViewport.offsetLeft,y:t.visualViewport.offsetTop}}function I1(e,t,n){return t===void 0&&(t=!1),!n||t&&n!==Ze(e)?!1:t}function ur(e,t,n,r){t===void 0&&(t=!1),n===void 0&&(n=!1);const o=e.getBoundingClientRect(),s=pc(e);let i=It(1);t&&(r?Pt(r)&&(i=zr(r)):i=zr(e));const l=I1(s,n,r)?Bm(s):It(0);let a=(o.left+l.x)/i.x,u=(o.top+l.y)/i.y,c=o.width/i.x,f=o.height/i.y;if(s){const m=Ze(s),d=r&&Pt(r)?Ze(r):r;let S=m,y=qa(S);for(;y&&r&&d!==S;){const x=zr(y),h=y.getBoundingClientRect(),p=bt(y),g=h.left+(y.clientLeft+parseFloat(p.paddingLeft))*x.x,E=h.top+(y.clientTop+parseFloat(p.paddingTop))*x.y;a*=x.x,u*=x.y,c*=x.x,f*=x.y,a+=g,u+=E,S=Ze(y),y=qa(S)}}return ji({width:c,height:f,x:a,y:u})}function sl(e,t){const n=ol(e).scrollLeft;return t?t.left+n:ur(Dt(e)).left+n}function Um(e,t){const n=e.getBoundingClientRect(),r=n.left+t.scrollLeft-sl(e,n),o=n.top+t.scrollTop;return{x:r,y:o}}function D1(e){let{elements:t,rect:n,offsetParent:r,strategy:o}=e;const s=o==="fixed",i=Dt(r),l=t?rl(t.floating):!1;if(r===i||l&&s)return n;let a={scrollLeft:0,scrollTop:0},u=It(1);const c=It(0),f=tn(r);if((f||!f&&!s)&&((fo(r)!=="body"||ms(i))&&(a=ol(r)),f)){const d=ur(r);u=zr(r),c.x=d.x+r.clientLeft,c.y=d.y+r.clientTop}const m=i&&!f&&!s?Um(i,a):It(0);return{width:n.width*u.x,height:n.height*u.y,x:n.x*u.x-a.scrollLeft*u.x+c.x+m.x,y:n.y*u.y-a.scrollTop*u.y+c.y+m.y}}function z1(e){return Array.from(e.getClientRects())}function F1(e){const t=Dt(e),n=ol(e),r=e.ownerDocument.body,o=Ye(t.scrollWidth,t.clientWidth,r.scrollWidth,r.clientWidth),s=Ye(t.scrollHeight,t.clientHeight,r.scrollHeight,r.clientHeight);let i=-n.scrollLeft+sl(e);const l=-n.scrollTop;return bt(r).direction==="rtl"&&(i+=Ye(t.clientWidth,r.clientWidth)-o),{width:o,height:s,x:i,y:l}}const nf=25;function $1(e,t){const n=Ze(e),r=Dt(e),o=n.visualViewport;let s=r.clientWidth,i=r.clientHeight,l=0,a=0;if(o){s=o.width,i=o.height;const c=fc();(!c||c&&t==="fixed")&&(l=o.offsetLeft,a=o.offsetTop)}const u=sl(r);if(u<=0){const c=r.ownerDocument,f=c.body,m=getComputedStyle(f),d=c.compatMode==="CSS1Compat"&&parseFloat(m.marginLeft)+parseFloat(m.marginRight)||0,S=Math.abs(r.clientWidth-f.clientWidth-d);S<=nf&&(s-=S)}else u<=nf&&(s+=u);return{width:s,height:i,x:l,y:a}}function B1(e,t){const n=ur(e,!0,t==="fixed"),r=n.top+e.clientTop,o=n.left+e.clientLeft,s=tn(e)?zr(e):It(1),i=e.clientWidth*s.x,l=e.clientHeight*s.y,a=o*s.x,u=r*s.y;return{width:i,height:l,x:a,y:u}}function rf(e,t,n){let r;if(t==="viewport")r=$1(e,n);else if(t==="document")r=F1(Dt(e));else if(Pt(t))r=B1(t,n);else{const o=Bm(e);r={x:t.x-o.x,y:t.y-o.y,width:t.width,height:t.height}}return ji(r)}function Vm(e,t){const n=Dn(e);return n===t||!Pt(n)||ro(n)?!1:bt(n).position==="fixed"||Vm(n,t)}function U1(e,t){const n=t.get(e);if(n)return n;let r=ns(e,[],!1).filter(l=>Pt(l)&&fo(l)!=="body"),o=null;const s=bt(e).position==="fixed";let i=s?Dn(e):e;for(;Pt(i)&&!ro(i);){const l=bt(i),a=dc(i);!a&&l.position==="fixed"&&(o=null),(s?!a&&!o:!a&&l.position==="static"&&!!o&&(o.position==="absolute"||o.position==="fixed")||ms(i)&&!a&&Vm(e,i))?r=r.filter(c=>c!==i):o=l,i=Dn(i)}return t.set(e,r),r}function V1(e){let{element:t,boundary:n,rootBoundary:r,strategy:o}=e;const i=[...n==="clippingAncestors"?rl(t)?[]:U1(t,this._c):[].concat(n),r],l=rf(t,i[0],o);let a=l.top,u=l.right,c=l.bottom,f=l.left;for(let m=1;m<i.length;m++){const d=rf(t,i[m],o);a=Ye(d.top,a),u=In(d.right,u),c=In(d.bottom,c),f=Ye(d.left,f)}return{width:u-f,height:c-a,x:f,y:a}}function W1(e){const{width:t,height:n}=$m(e);return{width:t,height:n}}function H1(e,t,n){const r=tn(t),o=Dt(t),s=n==="fixed",i=ur(e,!0,s,t);let l={scrollLeft:0,scrollTop:0};const a=It(0);function u(){a.x=sl(o)}if(r||!r&&!s)if((fo(t)!=="body"||ms(o))&&(l=ol(t)),r){const d=ur(t,!0,s,t);a.x=d.x+t.clientLeft,a.y=d.y+t.clientTop}else o&&u();s&&!r&&o&&u();const c=o&&!r&&!s?Um(o,l):It(0),f=i.left+l.scrollLeft-a.x-c.x,m=i.top+l.scrollTop-a.y-c.y;return{x:f,y:m,width:i.width,height:i.height}}function Gl(e){return bt(e).position==="static"}function of(e,t){if(!tn(e)||bt(e).position==="fixed")return null;if(t)return t(e);let n=e.offsetParent;return Dt(e)===n&&(n=n.ownerDocument.body),n}function Wm(e,t){const n=Ze(e);if(rl(e))return n;if(!tn(e)){let o=Dn(e);for(;o&&!ro(o);){if(Pt(o)&&!Gl(o))return o;o=Dn(o)}return n}let r=of(e,t);for(;r&&_1(r)&&Gl(r);)r=of(r,t);return r&&ro(r)&&Gl(r)&&!dc(r)?n:r||L1(e)||n}const Q1=async function(e){const t=this.getOffsetParent||Wm,n=this.getDimensions,r=await n(e.floating);return{reference:H1(e.reference,await t(e.floating),e.strategy),floating:{x:0,y:0,width:r.width,height:r.height}}};function K1(e){return bt(e).direction==="rtl"}const G1={convertOffsetParentRelativeRectToViewportRelativeRect:D1,getDocumentElement:Dt,getClippingRect:V1,getOffsetParent:Wm,getElementRects:Q1,getClientRects:z1,getDimensions:W1,getScale:zr,isElement:Pt,isRTL:K1};function Hm(e,t){return e.x===t.x&&e.y===t.y&&e.width===t.width&&e.height===t.height}function Y1(e,t){let n=null,r;const o=Dt(e);function s(){var l;clearTimeout(r),(l=n)==null||l.disconnect(),n=null}function i(l,a){l===void 0&&(l=!1),a===void 0&&(a=1),s();const u=e.getBoundingClientRect(),{left:c,top:f,width:m,height:d}=u;if(l||t(),!m||!d)return;const S=$s(f),y=$s(o.clientWidth-(c+m)),x=$s(o.clientHeight-(f+d)),h=$s(c),g={rootMargin:-S+"px "+-y+"px "+-x+"px "+-h+"px",threshold:Ye(0,In(1,a))||1};let E=!0;function C(k){const P=k[0].intersectionRatio;if(P!==a){if(!E)return i();P?i(!1,P):r=setTimeout(()=>{i(!1,1e-7)},1e3)}P===1&&!Hm(u,e.getBoundingClientRect())&&i(),E=!1}try{n=new IntersectionObserver(C,{...g,root:o.ownerDocument})}catch{n=new IntersectionObserver(C,g)}n.observe(e)}return i(!0),s}function X1(e,t,n,r){r===void 0&&(r={});const{ancestorScroll:o=!0,ancestorResize:s=!0,elementResize:i=typeof ResizeObserver=="function",layoutShift:l=typeof IntersectionObserver=="function",animationFrame:a=!1}=r,u=pc(e),c=o||s?[...u?ns(u):[],...t?ns(t):[]]:[];c.forEach(h=>{o&&h.addEventListener("scroll",n,{passive:!0}),s&&h.addEventListener("resize",n)});const f=u&&l?Y1(u,n):null;let m=-1,d=null;i&&(d=new ResizeObserver(h=>{let[p]=h;p&&p.target===u&&d&&t&&(d.unobserve(t),cancelAnimationFrame(m),m=requestAnimationFrame(()=>{var g;(g=d)==null||g.observe(t)})),n()}),u&&!a&&d.observe(u),t&&d.observe(t));let S,y=a?ur(e):null;a&&x();function x(){const h=ur(e);y&&!Hm(y,h)&&n(),y=h,S=requestAnimationFrame(x)}return n(),()=>{var h;c.forEach(p=>{o&&p.removeEventListener("scroll",n),s&&p.removeEventListener("resize",n)}),f==null||f(),(h=d)==null||h.disconnect(),d=null,a&&cancelAnimationFrame(S)}}const q1=T1,Z1=N1,J1=k1,eS=j1,tS=P1,sf=C1,nS=R1,rS=(e,t,n)=>{const r=new Map,o={platform:G1,...n},s={...o.platform,_c:r};return E1(e,t,{...o,platform:s})};var oS=typeof document<"u",sS=function(){},ri=oS?w.useLayoutEffect:sS;function _i(e,t){if(e===t)return!0;if(typeof e!=typeof t)return!1;if(typeof e=="function"&&e.toString()===t.toString())return!0;let n,r,o;if(e&&t&&typeof e=="object"){if(Array.isArray(e)){if(n=e.length,n!==t.length)return!1;for(r=n;r--!==0;)if(!_i(e[r],t[r]))return!1;return!0}if(o=Object.keys(e),n=o.length,n!==Object.keys(t).length)return!1;for(r=n;r--!==0;)if(!{}.hasOwnProperty.call(t,o[r]))return!1;for(r=n;r--!==0;){const s=o[r];if(!(s==="_owner"&&e.$$typeof)&&!_i(e[s],t[s]))return!1}return!0}return e!==e&&t!==t}function Qm(e){return typeof window>"u"?1:(e.ownerDocument.defaultView||window).devicePixelRatio||1}function lf(e,t){const n=Qm(e);return Math.round(t*n)/n}function Yl(e){const t=w.useRef(e);return ri(()=>{t.current=e}),t}function iS(e){e===void 0&&(e={});const{placement:t="bottom",strategy:n="absolute",middleware:r=[],platform:o,elements:{reference:s,floating:i}={},transform:l=!0,whileElementsMounted:a,open:u}=e,[c,f]=w.useState({x:0,y:0,strategy:n,placement:t,middlewareData:{},isPositioned:!1}),[m,d]=w.useState(r);_i(m,r)||d(r);const[S,y]=w.useState(null),[x,h]=w.useState(null),p=w.useCallback(T=>{T!==k.current&&(k.current=T,y(T))},[]),g=w.useCallback(T=>{T!==P.current&&(P.current=T,h(T))},[]),E=s||S,C=i||x,k=w.useRef(null),P=w.useRef(null),N=w.useRef(c),L=a!=null,A=Yl(a),$=Yl(o),D=Yl(u),Q=w.useCallback(()=>{if(!k.current||!P.current)return;const T={placement:t,strategy:n,middleware:m};$.current&&(T.platform=$.current),rS(k.current,P.current,T).then(R=>{const M={...R,isPositioned:D.current!==!1};O.current&&!_i(N.current,M)&&(N.current=M,ps.flushSync(()=>{f(M)}))})},[m,t,n,$,D]);ri(()=>{u===!1&&N.current.isPositioned&&(N.current.isPositioned=!1,f(T=>({...T,isPositioned:!1})))},[u]);const O=w.useRef(!1);ri(()=>(O.current=!0,()=>{O.current=!1}),[]),ri(()=>{if(E&&(k.current=E),C&&(P.current=C),E&&C){if(A.current)return A.current(E,C,Q);Q()}},[E,C,Q,A,L]);const Y=w.useMemo(()=>({reference:k,floating:P,setReference:p,setFloating:g}),[p,g]),B=w.useMemo(()=>({reference:E,floating:C}),[E,C]),V=w.useMemo(()=>{const T={position:n,left:0,top:0};if(!B.floating)return T;const R=lf(B.floating,c.x),M=lf(B.floating,c.y);return l?{...T,transform:"translate("+R+"px, "+M+"px)",...Qm(B.floating)>=1.5&&{willChange:"transform"}}:{position:n,left:R,top:M}},[n,l,B.floating,c.x,c.y]);return w.useMemo(()=>({...c,update:Q,refs:Y,elements:B,floatingStyles:V}),[c,Q,Y,B,V])}const lS=e=>{function t(n){return{}.hasOwnProperty.call(n,"current")}return{name:"arrow",options:e,fn(n){const{element:r,padding:o}=typeof e=="function"?e(n):e;return r&&t(r)?r.current!=null?sf({element:r.current,padding:o}).fn(n):{}:r?sf({element:r,padding:o}).fn(n):{}}}},aS=(e,t)=>{const n=q1(e);return{name:n.name,fn:n.fn,options:[e,t]}},uS=(e,t)=>{const n=Z1(e);return{name:n.name,fn:n.fn,options:[e,t]}},cS=(e,t)=>({fn:nS(e).fn,options:[e,t]}),dS=(e,t)=>{const n=J1(e);return{name:n.name,fn:n.fn,options:[e,t]}},fS=(e,t)=>{const n=eS(e);return{name:n.name,fn:n.fn,options:[e,t]}},pS=(e,t)=>{const n=tS(e);return{name:n.name,fn:n.fn,options:[e,t]}},hS=(e,t)=>{const n=lS(e);return{name:n.name,fn:n.fn,options:[e,t]}};var mS="Arrow",Km=w.forwardRef((e,t)=>{const{children:n,width:r=10,height:o=5,...s}=e;return v.jsx(Qe.svg,{...s,ref:t,width:r,height:o,viewBox:"0 0 30 10",preserveAspectRatio:"none",children:e.asChild?n:v.jsx("polygon",{points:"0,0 30,0 15,10"})})});Km.displayName=mS;var gS=Km;function vS(e){const[t,n]=w.useState(void 0);return kt(()=>{if(e){n({width:e.offsetWidth,height:e.offsetHeight});const r=new ResizeObserver(o=>{if(!Array.isArray(o)||!o.length)return;const s=o[0];let i,l;if("borderBoxSize"in s){const a=s.borderBoxSize,u=Array.isArray(a)?a[0]:a;i=u.inlineSize,l=u.blockSize}else i=e.offsetWidth,l=e.offsetHeight;n({width:i,height:l})});return r.observe(e,{box:"border-box"}),()=>r.unobserve(e)}else n(void 0)},[e]),t}var Gm="Popper",[Ym,Xm]=Ji(Gm),[s2,qm]=Ym(Gm),Zm="PopperAnchor",Jm=w.forwardRef((e,t)=>{const{__scopePopper:n,virtualRef:r,...o}=e,s=qm(Zm,n),i=w.useRef(null),l=Ct(t,i),a=w.useRef(null);return w.useEffect(()=>{const u=a.current;a.current=(r==null?void 0:r.current)||i.current,u!==a.current&&s.onAnchorChange(a.current)}),r?null:v.jsx(Qe.div,{...o,ref:l})});Jm.displayName=Zm;var hc="PopperContent",[yS,wS]=Ym(hc),eg=w.forwardRef((e,t)=>{var J,fr,nn,Bn,rn,pr;const{__scopePopper:n,side:r="bottom",sideOffset:o=0,align:s="center",alignOffset:i=0,arrowPadding:l=0,avoidCollisions:a=!0,collisionBoundary:u=[],collisionPadding:c=0,sticky:f="partial",hideWhenDetached:m=!1,updatePositionStrategy:d="optimized",onPlaced:S,...y}=e,x=qm(hc,n),[h,p]=w.useState(null),g=Ct(t,on=>p(on)),[E,C]=w.useState(null),k=vS(E),P=(k==null?void 0:k.width)??0,N=(k==null?void 0:k.height)??0,L=r+(s!=="center"?"-"+s:""),A=typeof c=="number"?c:{top:0,right:0,bottom:0,left:0,...c},$=Array.isArray(u)?u:[u],D=$.length>0,Q={padding:A,boundary:$.filter(SS),altBoundary:D},{refs:O,floatingStyles:Y,placement:B,isPositioned:V,middlewareData:T}=iS({strategy:"fixed",placement:L,whileElementsMounted:(...on)=>X1(...on,{animationFrame:d==="always"}),elements:{reference:x.anchor},middleware:[aS({mainAxis:o+N,alignmentAxis:i}),a&&uS({mainAxis:!0,crossAxis:!1,limiter:f==="partial"?cS():void 0,...Q}),a&&dS({...Q}),fS({...Q,apply:({elements:on,rects:gs,availableWidth:fl,availableHeight:vs})=>{const{width:pl,height:po}=gs.reference,hr=on.floating.style;hr.setProperty("--radix-popper-available-width",`${fl}px`),hr.setProperty("--radix-popper-available-height",`${vs}px`),hr.setProperty("--radix-popper-anchor-width",`${pl}px`),hr.setProperty("--radix-popper-anchor-height",`${po}px`)}}),E&&hS({element:E,padding:l}),ES({arrowWidth:P,arrowHeight:N}),m&&pS({strategy:"referenceHidden",...Q})]}),[R,M]=rg(B),W=qt(S);kt(()=>{V&&(W==null||W())},[V,W]);const z=(J=T.arrow)==null?void 0:J.x,K=(fr=T.arrow)==null?void 0:fr.y,X=((nn=T.arrow)==null?void 0:nn.centerOffset)!==0,[he,Ne]=w.useState();return kt(()=>{h&&Ne(window.getComputedStyle(h).zIndex)},[h]),v.jsx("div",{ref:O.setFloating,"data-radix-popper-content-wrapper":"",style:{...Y,transform:V?Y.transform:"translate(0, -200%)",minWidth:"max-content",zIndex:he,"--radix-popper-transform-origin":[(Bn=T.transformOrigin)==null?void 0:Bn.x,(rn=T.transformOrigin)==null?void 0:rn.y].join(" "),...((pr=T.hide)==null?void 0:pr.referenceHidden)&&{visibility:"hidden",pointerEvents:"none"}},dir:e.dir,children:v.jsx(yS,{scope:n,placedSide:R,onArrowChange:C,arrowX:z,arrowY:K,shouldHideArrow:X,children:v.jsx(Qe.div,{"data-side":R,"data-align":M,...y,ref:g,style:{...y.style,animation:V?void 0:"none"}})})})});eg.displayName=hc;var tg="PopperArrow",xS={top:"bottom",right:"left",bottom:"top",left:"right"},ng=w.forwardRef(function(t,n){const{__scopePopper:r,...o}=t,s=wS(tg,r),i=xS[s.placedSide];return v.jsx("span",{ref:s.onArrowChange,style:{position:"absolute",left:s.arrowX,top:s.arrowY,[i]:0,transformOrigin:{top:"",right:"0 0",bottom:"center 0",left:"100% 0"}[s.placedSide],transform:{top:"translateY(100%)",right:"translateY(50%) rotate(90deg) translateX(-50%)",bottom:"rotate(180deg)",left:"translateY(50%) rotate(-90deg) translateX(50%)"}[s.placedSide],visibility:s.shouldHideArrow?"hidden":void 0},children:v.jsx(gS,{...o,ref:n,style:{...o.style,display:"block"}})})});ng.displayName=tg;function SS(e){return e!==null}var ES=e=>({name:"transformOrigin",options:e,fn(t){var x,h,p;const{placement:n,rects:r,middlewareData:o}=t,i=((x=o.arrow)==null?void 0:x.centerOffset)!==0,l=i?0:e.arrowWidth,a=i?0:e.arrowHeight,[u,c]=rg(n),f={start:"0%",center:"50%",end:"100%"}[c],m=(((h=o.arrow)==null?void 0:h.x)??0)+l/2,d=(((p=o.arrow)==null?void 0:p.y)??0)+a/2;let S="",y="";return u==="bottom"?(S=i?f:`${m}px`,y=`${-a}px`):u==="top"?(S=i?f:`${m}px`,y=`${r.floating.height+a}px`):u==="right"?(S=`${-a}px`,y=i?f:`${d}px`):u==="left"&&(S=`${r.floating.width+a}px`,y=i?f:`${d}px`),{data:{x:S,y}}}});function rg(e){const[t,n="center"]=e.split("-");return[t,n]}var CS=Jm,kS=eg,PS=ng,bS=Symbol("radix.slottable");function TS(e){const t=({children:n})=>v.jsx(v.Fragment,{children:n});return t.displayName=`${e}.Slottable`,t.__radixId=bS,t}var[il]=Ji("Tooltip",[Xm]),mc=Xm(),og="TooltipProvider",NS=700,af="tooltip.open",[RS,sg]=il(og),ig=e=>{const{__scopeTooltip:t,delayDuration:n=NS,skipDelayDuration:r=300,disableHoverableContent:o=!1,children:s}=e,i=w.useRef(!0),l=w.useRef(!1),a=w.useRef(0);return w.useEffect(()=>{const u=a.current;return()=>window.clearTimeout(u)},[]),v.jsx(RS,{scope:t,isOpenDelayedRef:i,delayDuration:n,onOpen:w.useCallback(()=>{window.clearTimeout(a.current),i.current=!1},[]),onClose:w.useCallback(()=>{window.clearTimeout(a.current),a.current=window.setTimeout(()=>i.current=!0,r)},[r]),isPointerInTransitRef:l,onPointerInTransitChange:w.useCallback(u=>{l.current=u},[]),disableHoverableContent:o,children:s})};ig.displayName=og;var lg="Tooltip",[i2,ll]=il(lg),Za="TooltipTrigger",jS=w.forwardRef((e,t)=>{const{__scopeTooltip:n,...r}=e,o=ll(Za,n),s=sg(Za,n),i=mc(n),l=w.useRef(null),a=Ct(t,l,o.onTriggerChange),u=w.useRef(!1),c=w.useRef(!1),f=w.useCallback(()=>u.current=!1,[]);return w.useEffect(()=>()=>document.removeEventListener("pointerup",f),[f]),v.jsx(CS,{asChild:!0,...i,children:v.jsx(Qe.button,{"aria-describedby":o.open?o.contentId:void 0,"data-state":o.stateAttribute,...r,ref:a,onPointerMove:ve(e.onPointerMove,m=>{m.pointerType!=="touch"&&!c.current&&!s.isPointerInTransitRef.current&&(o.onTriggerEnter(),c.current=!0)}),onPointerLeave:ve(e.onPointerLeave,()=>{o.onTriggerLeave(),c.current=!1}),onPointerDown:ve(e.onPointerDown,()=>{o.open&&o.onClose(),u.current=!0,document.addEventListener("pointerup",f,{once:!0})}),onFocus:ve(e.onFocus,()=>{u.current||o.onOpen()}),onBlur:ve(e.onBlur,o.onClose),onClick:ve(e.onClick,o.onClose)})})});jS.displayName=Za;var _S="TooltipPortal",[l2,AS]=il(_S,{forceMount:void 0}),oo="TooltipContent",ag=w.forwardRef((e,t)=>{const n=AS(oo,e.__scopeTooltip),{forceMount:r=n.forceMount,side:o="top",...s}=e,i=ll(oo,e.__scopeTooltip);return v.jsx(rc,{present:r||i.open,children:i.disableHoverableContent?v.jsx(ug,{side:o,...s,ref:t}):v.jsx(OS,{side:o,...s,ref:t})})}),OS=w.forwardRef((e,t)=>{const n=ll(oo,e.__scopeTooltip),r=sg(oo,e.__scopeTooltip),o=w.useRef(null),s=Ct(t,o),[i,l]=w.useState(null),{trigger:a,onClose:u}=n,c=o.current,{onPointerInTransitChange:f}=r,m=w.useCallback(()=>{l(null),f(!1)},[f]),d=w.useCallback((S,y)=>{const x=S.currentTarget,h={x:S.clientX,y:S.clientY},p=zS(h,x.getBoundingClientRect()),g=FS(h,p),E=$S(y.getBoundingClientRect()),C=US([...g,...E]);l(C),f(!0)},[f]);return w.useEffect(()=>()=>m(),[m]),w.useEffect(()=>{if(a&&c){const S=x=>d(x,c),y=x=>d(x,a);return a.addEventListener("pointerleave",S),c.addEventListener("pointerleave",y),()=>{a.removeEventListener("pointerleave",S),c.removeEventListener("pointerleave",y)}}},[a,c,d,m]),w.useEffect(()=>{if(i){const S=y=>{const x=y.target,h={x:y.clientX,y:y.clientY},p=(a==null?void 0:a.contains(x))||(c==null?void 0:c.contains(x)),g=!BS(h,i);p?m():g&&(m(),u())};return document.addEventListener("pointermove",S),()=>document.removeEventListener("pointermove",S)}},[a,c,i,u,m]),v.jsx(ug,{...e,ref:s})}),[LS,MS]=il(lg,{isInside:!1}),IS=TS("TooltipContent"),ug=w.forwardRef((e,t)=>{const{__scopeTooltip:n,children:r,"aria-label":o,onEscapeKeyDown:s,onPointerDownOutside:i,...l}=e,a=ll(oo,n),u=mc(n),{onClose:c}=a;return w.useEffect(()=>(document.addEventListener(af,c),()=>document.removeEventListener(af,c)),[c]),w.useEffect(()=>{if(a.trigger){const f=m=>{const d=m.target;d!=null&&d.contains(a.trigger)&&c()};return window.addEventListener("scroll",f,{capture:!0}),()=>window.removeEventListener("scroll",f,{capture:!0})}},[a.trigger,c]),v.jsx(nc,{asChild:!0,disableOutsidePointerEvents:!1,onEscapeKeyDown:s,onPointerDownOutside:i,onFocusOutside:f=>f.preventDefault(),onDismiss:c,children:v.jsxs(kS,{"data-state":a.stateAttribute,...u,...l,ref:t,style:{...l.style,"--radix-tooltip-content-transform-origin":"var(--radix-popper-transform-origin)","--radix-tooltip-content-available-width":"var(--radix-popper-available-width)","--radix-tooltip-content-available-height":"var(--radix-popper-available-height)","--radix-tooltip-trigger-width":"var(--radix-popper-anchor-width)","--radix-tooltip-trigger-height":"var(--radix-popper-anchor-height)"},children:[v.jsx(IS,{children:r}),v.jsx(LS,{scope:n,isInside:!0,children:v.jsx(pw,{id:a.contentId,role:"tooltip",children:o||r})})]})})});ag.displayName=oo;var cg="TooltipArrow",DS=w.forwardRef((e,t)=>{const{__scopeTooltip:n,...r}=e,o=mc(n);return MS(cg,n).isInside?null:v.jsx(PS,{...o,...r,ref:t})});DS.displayName=cg;function zS(e,t){const n=Math.abs(t.top-e.y),r=Math.abs(t.bottom-e.y),o=Math.abs(t.right-e.x),s=Math.abs(t.left-e.x);switch(Math.min(n,r,o,s)){case s:return"left";case o:return"right";case n:return"top";case r:return"bottom";default:throw new Error("unreachable")}}function FS(e,t,n=5){const r=[];switch(t){case"top":r.push({x:e.x-n,y:e.y+n},{x:e.x+n,y:e.y+n});break;case"bottom":r.push({x:e.x-n,y:e.y-n},{x:e.x+n,y:e.y-n});break;case"left":r.push({x:e.x+n,y:e.y-n},{x:e.x+n,y:e.y+n});break;case"right":r.push({x:e.x-n,y:e.y-n},{x:e.x-n,y:e.y+n});break}return r}function $S(e){const{top:t,right:n,bottom:r,left:o}=e;return[{x:o,y:t},{x:n,y:t},{x:n,y:r},{x:o,y:r}]}function BS(e,t){const{x:n,y:r}=e;let o=!1;for(let s=0,i=t.length-1;s<t.length;i=s++){const l=t[s],a=t[i],u=l.x,c=l.y,f=a.x,m=a.y;c>r!=m>r&&n<(f-u)*(r-c)/(m-c)+u&&(o=!o)}return o}function US(e){const t=e.slice();return t.sort((n,r)=>n.x<r.x?-1:n.x>r.x?1:n.y<r.y?-1:n.y>r.y?1:0),VS(t)}function VS(e){if(e.length<=1)return e.slice();const t=[];for(let r=0;r<e.length;r++){const o=e[r];for(;t.length>=2;){const s=t[t.length-1],i=t[t.length-2];if((s.x-i.x)*(o.y-i.y)>=(s.y-i.y)*(o.x-i.x))t.pop();else break}t.push(o)}t.pop();const n=[];for(let r=e.length-1;r>=0;r--){const o=e[r];for(;n.length>=2;){const s=n[n.length-1],i=n[n.length-2];if((s.x-i.x)*(o.y-i.y)>=(s.y-i.y)*(o.x-i.x))n.pop();else break}n.push(o)}return n.pop(),t.length===1&&n.length===1&&t[0].x===n[0].x&&t[0].y===n[0].y?t:t.concat(n)}var WS=ig,dg=ag;const HS=WS,QS=w.forwardRef(({className:e,sideOffset:t=4,...n},r)=>v.jsx(dg,{ref:r,sideOffset:t,className:Te("z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",e),...n}));QS.displayName=dg.displayName;var al=class{constructor(){this.listeners=new Set,this.subscribe=this.subscribe.bind(this)}subscribe(e){return this.listeners.add(e),this.onSubscribe(),()=>{this.listeners.delete(e),this.onUnsubscribe()}}hasListeners(){return this.listeners.size>0}onSubscribe(){}onUnsubscribe(){}},Xn,vn,Br,bf,KS=(bf=class extends al{constructor(){super();q(this,Xn);q(this,vn);q(this,Br);F(this,Br,t=>{if(typeof window<"u"&&window.addEventListener){const n=()=>t();return window.addEventListener("visibilitychange",n,!1),()=>{window.removeEventListener("visibilitychange",n)}}})}onSubscribe(){b(this,vn)||this.setEventListener(b(this,Br))}onUnsubscribe(){var t;this.hasListeners()||((t=b(this,vn))==null||t.call(this),F(this,vn,void 0))}setEventListener(t){var n;F(this,Br,t),(n=b(this,vn))==null||n.call(this),F(this,vn,t(r=>{typeof r=="boolean"?this.setFocused(r):this.onFocus()}))}setFocused(t){b(this,Xn)!==t&&(F(this,Xn,t),this.onFocus())}onFocus(){const t=this.isFocused();this.listeners.forEach(n=>{n(t)})}isFocused(){var t;return typeof b(this,Xn)=="boolean"?b(this,Xn):((t=globalThis.document)==null?void 0:t.visibilityState)!=="hidden"}},Xn=new WeakMap,vn=new WeakMap,Br=new WeakMap,bf),fg=new KS,GS={setTimeout:(e,t)=>setTimeout(e,t),clearTimeout:e=>clearTimeout(e),setInterval:(e,t)=>setInterval(e,t),clearInterval:e=>clearInterval(e)},yn,iu,Tf,YS=(Tf=class{constructor(){q(this,yn,GS);q(this,iu,!1)}setTimeoutProvider(e){F(this,yn,e)}setTimeout(e,t){return b(this,yn).setTimeout(e,t)}clearTimeout(e){b(this,yn).clearTimeout(e)}setInterval(e,t){return b(this,yn).setInterval(e,t)}clearInterval(e){b(this,yn).clearInterval(e)}},yn=new WeakMap,iu=new WeakMap,Tf),Ja=new YS;function XS(e){setTimeout(e,0)}var qS=typeof window>"u"||"Deno"in globalThis;function gt(){}function ZS(e,t){return typeof e=="function"?e(t):e}function JS(e){return typeof e=="number"&&e>=0&&e!==1/0}function eE(e,t){return Math.max(e+(t||0)-Date.now(),0)}function eu(e,t){return typeof e=="function"?e(t):e}function tE(e,t){return typeof e=="function"?e(t):e}function uf(e,t){const{type:n="all",exact:r,fetchStatus:o,predicate:s,queryKey:i,stale:l}=e;if(i){if(r){if(t.queryHash!==gc(i,t.options))return!1}else if(!os(t.queryKey,i))return!1}if(n!=="all"){const a=t.isActive();if(n==="active"&&!a||n==="inactive"&&a)return!1}return!(typeof l=="boolean"&&t.isStale()!==l||o&&o!==t.state.fetchStatus||s&&!s(t))}function cf(e,t){const{exact:n,status:r,predicate:o,mutationKey:s}=e;if(s){if(!t.options.mutationKey)return!1;if(n){if(rs(t.options.mutationKey)!==rs(s))return!1}else if(!os(t.options.mutationKey,s))return!1}return!(r&&t.state.status!==r||o&&!o(t))}function gc(e,t){return((t==null?void 0:t.queryKeyHashFn)||rs)(e)}function rs(e){return JSON.stringify(e,(t,n)=>tu(n)?Object.keys(n).sort().reduce((r,o)=>(r[o]=n[o],r),{}):n)}function os(e,t){return e===t?!0:typeof e!=typeof t?!1:e&&t&&typeof e=="object"&&typeof t=="object"?Object.keys(t).every(n=>os(e[n],t[n])):!1}var nE=Object.prototype.hasOwnProperty;function pg(e,t,n=0){if(e===t)return e;if(n>500)return t;const r=df(e)&&df(t);if(!r&&!(tu(e)&&tu(t)))return t;const s=(r?e:Object.keys(e)).length,i=r?t:Object.keys(t),l=i.length,a=r?new Array(l):{};let u=0;for(let c=0;c<l;c++){const f=r?c:i[c],m=e[f],d=t[f];if(m===d){a[f]=m,(r?c<s:nE.call(e,f))&&u++;continue}if(m===null||d===null||typeof m!="object"||typeof d!="object"){a[f]=d;continue}const S=pg(m,d,n+1);a[f]=S,S===m&&u++}return s===l&&u===s?e:a}function df(e){return Array.isArray(e)&&e.length===Object.keys(e).length}function tu(e){if(!ff(e))return!1;const t=e.constructor;if(t===void 0)return!0;const n=t.prototype;return!(!ff(n)||!n.hasOwnProperty("isPrototypeOf")||Object.getPrototypeOf(e)!==Object.prototype)}function ff(e){return Object.prototype.toString.call(e)==="[object Object]"}function rE(e){return new Promise(t=>{Ja.setTimeout(t,e)})}function oE(e,t,n){return typeof n.structuralSharing=="function"?n.structuralSharing(e,t):n.structuralSharing!==!1?pg(e,t):t}function sE(e,t,n=0){const r=[...e,t];return n&&r.length>n?r.slice(1):r}function iE(e,t,n=0){const r=[t,...e];return n&&r.length>n?r.slice(0,-1):r}var vc=Symbol();function hg(e,t){return!e.queryFn&&(t!=null&&t.initialPromise)?()=>t.initialPromise:!e.queryFn||e.queryFn===vc?()=>Promise.reject(new Error(`Missing queryFn: '${e.queryHash}'`)):e.queryFn}function lE(e,t,n){let r=!1,o;return Object.defineProperty(e,"signal",{enumerable:!0,get:()=>(o??(o=t()),r||(r=!0,o.aborted?n():o.addEventListener("abort",n,{once:!0})),o)}),e}var mg=(()=>{let e=()=>qS;return{isServer(){return e()},setIsServer(t){e=t}}})();function aE(){let e,t;const n=new Promise((o,s)=>{e=o,t=s});n.status="pending",n.catch(()=>{});function r(o){Object.assign(n,o),delete n.resolve,delete n.reject}return n.resolve=o=>{r({status:"fulfilled",value:o}),e(o)},n.reject=o=>{r({status:"rejected",reason:o}),t(o)},n}var uE=XS;function cE(){let e=[],t=0,n=l=>{l()},r=l=>{l()},o=uE;const s=l=>{t?e.push(l):o(()=>{n(l)})},i=()=>{const l=e;e=[],l.length&&o(()=>{r(()=>{l.forEach(a=>{n(a)})})})};return{batch:l=>{let a;t++;try{a=l()}finally{t--,t||i()}return a},batchCalls:l=>(...a)=>{s(()=>{l(...a)})},schedule:s,setNotifyFunction:l=>{n=l},setBatchNotifyFunction:l=>{r=l},setScheduler:l=>{o=l}}}var Me=cE(),Ur,wn,Vr,Nf,dE=(Nf=class extends al{constructor(){super();q(this,Ur,!0);q(this,wn);q(this,Vr);F(this,Vr,t=>{if(typeof window<"u"&&window.addEventListener){const n=()=>t(!0),r=()=>t(!1);return window.addEventListener("online",n,!1),window.addEventListener("offline",r,!1),()=>{window.removeEventListener("online",n),window.removeEventListener("offline",r)}}})}onSubscribe(){b(this,wn)||this.setEventListener(b(this,Vr))}onUnsubscribe(){var t;this.hasListeners()||((t=b(this,wn))==null||t.call(this),F(this,wn,void 0))}setEventListener(t){var n;F(this,Vr,t),(n=b(this,wn))==null||n.call(this),F(this,wn,t(this.setOnline.bind(this)))}setOnline(t){b(this,Ur)!==t&&(F(this,Ur,t),this.listeners.forEach(r=>{r(t)}))}isOnline(){return b(this,Ur)}},Ur=new WeakMap,wn=new WeakMap,Vr=new WeakMap,Nf),Ai=new dE;function fE(e){return Math.min(1e3*2**e,3e4)}function gg(e){return(e??"online")==="online"?Ai.isOnline():!0}var nu=class extends Error{constructor(e){super("CancelledError"),this.revert=e==null?void 0:e.revert,this.silent=e==null?void 0:e.silent}};function vg(e){let t=!1,n=0,r;const o=aE(),s=()=>o.status!=="pending",i=y=>{var x;if(!s()){const h=new nu(y);m(h),(x=e.onCancel)==null||x.call(e,h)}},l=()=>{t=!0},a=()=>{t=!1},u=()=>fg.isFocused()&&(e.networkMode==="always"||Ai.isOnline())&&e.canRun(),c=()=>gg(e.networkMode)&&e.canRun(),f=y=>{s()||(r==null||r(),o.resolve(y))},m=y=>{s()||(r==null||r(),o.reject(y))},d=()=>new Promise(y=>{var x;r=h=>{(s()||u())&&y(h)},(x=e.onPause)==null||x.call(e)}).then(()=>{var y;r=void 0,s()||(y=e.onContinue)==null||y.call(e)}),S=()=>{if(s())return;let y;const x=n===0?e.initialPromise:void 0;try{y=x??e.fn()}catch(h){y=Promise.reject(h)}Promise.resolve(y).then(f).catch(h=>{var k;if(s())return;const p=e.retry??(mg.isServer()?0:3),g=e.retryDelay??fE,E=typeof g=="function"?g(n,h):g,C=p===!0||typeof p=="number"&&n<p||typeof p=="function"&&p(n,h);if(t||!C){m(h);return}n++,(k=e.onFail)==null||k.call(e,n,h),rE(E).then(()=>u()?void 0:d()).then(()=>{t?m(h):S()})})};return{promise:o,status:()=>o.status,cancel:i,continue:()=>(r==null||r(),o),cancelRetry:l,continueRetry:a,canStart:c,start:()=>(c()?S():d().then(S),o)}}var qn,Rf,yg=(Rf=class{constructor(){q(this,qn)}destroy(){this.clearGcTimeout()}scheduleGc(){this.clearGcTimeout(),JS(this.gcTime)&&F(this,qn,Ja.setTimeout(()=>{this.optionalRemove()},this.gcTime))}updateGcTime(e){this.gcTime=Math.max(this.gcTime||0,e??(mg.isServer()?1/0:5*60*1e3))}clearGcTimeout(){b(this,qn)!==void 0&&(Ja.clearTimeout(b(this,qn)),F(this,qn,void 0))}},qn=new WeakMap,Rf);function pE(e){return{onFetch:(t,n)=>{var c,f,m,d,S;const r=t.options,o=(m=(f=(c=t.fetchOptions)==null?void 0:c.meta)==null?void 0:f.fetchMore)==null?void 0:m.direction,s=((d=t.state.data)==null?void 0:d.pages)||[],i=((S=t.state.data)==null?void 0:S.pageParams)||[];let l={pages:[],pageParams:[]},a=0;const u=async()=>{let y=!1;const x=g=>{lE(g,()=>t.signal,()=>y=!0)},h=hg(t.options,t.fetchOptions),p=async(g,E,C)=>{if(y)return Promise.reject(t.signal.reason);if(E==null&&g.pages.length)return Promise.resolve(g);const P=(()=>{const $={client:t.client,queryKey:t.queryKey,pageParam:E,direction:C?"backward":"forward",meta:t.options.meta};return x($),$})(),N=await h(P),{maxPages:L}=t.options,A=C?iE:sE;return{pages:A(g.pages,N,L),pageParams:A(g.pageParams,E,L)}};if(o&&s.length){const g=o==="backward",E=g?hE:pf,C={pages:s,pageParams:i},k=E(r,C);l=await p(C,k,g)}else{const g=e??s.length;do{const E=a===0?i[0]??r.initialPageParam:pf(r,l);if(a>0&&E==null)break;l=await p(l,E),a++}while(a<g)}return l};t.options.persister?t.fetchFn=()=>{var y,x;return(x=(y=t.options).persister)==null?void 0:x.call(y,u,{client:t.client,queryKey:t.queryKey,meta:t.options.meta,signal:t.signal},n)}:t.fetchFn=u}}}function pf(e,{pages:t,pageParams:n}){const r=t.length-1;return t.length>0?e.getNextPageParam(t[r],t,n[r],n):void 0}function hE(e,{pages:t,pageParams:n}){var r;return t.length>0?(r=e.getPreviousPageParam)==null?void 0:r.call(e,t[0],t,n[0],n):void 0}var Wr,Zn,Hr,rt,Jn,we,ss,er,Ge,wg,Ft,jf,mE=(jf=class extends yg{constructor(t){super();q(this,Ge);q(this,Wr);q(this,Zn);q(this,Hr);q(this,rt);q(this,Jn);q(this,we);q(this,ss);q(this,er);F(this,er,!1),F(this,ss,t.defaultOptions),this.setOptions(t.options),this.observers=[],F(this,Jn,t.client),F(this,rt,b(this,Jn).getQueryCache()),this.queryKey=t.queryKey,this.queryHash=t.queryHash,F(this,Zn,mf(this.options)),this.state=t.state??b(this,Zn),this.scheduleGc()}get meta(){return this.options.meta}get queryType(){return b(this,Wr)}get promise(){var t;return(t=b(this,we))==null?void 0:t.promise}setOptions(t){if(this.options={...b(this,ss),...t},t!=null&&t._type&&F(this,Wr,t._type),this.updateGcTime(this.options.gcTime),this.state&&this.state.data===void 0){const n=mf(this.options);n.data!==void 0&&(this.setState(hf(n.data,n.dataUpdatedAt)),F(this,Zn,n))}}optionalRemove(){!this.observers.length&&this.state.fetchStatus==="idle"&&b(this,rt).remove(this)}setData(t,n){const r=oE(this.state.data,t,this.options);return Ee(this,Ge,Ft).call(this,{data:r,type:"success",dataUpdatedAt:n==null?void 0:n.updatedAt,manual:n==null?void 0:n.manual}),r}setState(t){Ee(this,Ge,Ft).call(this,{type:"setState",state:t})}cancel(t){var r,o;const n=(r=b(this,we))==null?void 0:r.promise;return(o=b(this,we))==null||o.cancel(t),n?n.then(gt).catch(gt):Promise.resolve()}destroy(){super.destroy(),this.cancel({silent:!0})}get resetState(){return b(this,Zn)}reset(){this.destroy(),this.setState(this.resetState)}isActive(){return this.observers.some(t=>tE(t.options.enabled,this)!==!1)}isDisabled(){return this.getObserversCount()>0?!this.isActive():this.options.queryFn===vc||!this.isFetched()}isFetched(){return this.state.dataUpdateCount+this.state.errorUpdateCount>0}isStatic(){return this.getObserversCount()>0?this.observers.some(t=>eu(t.options.staleTime,this)==="static"):!1}isStale(){return this.getObserversCount()>0?this.observers.some(t=>t.getCurrentResult().isStale):this.state.data===void 0||this.state.isInvalidated}isStaleByTime(t=0){return this.state.data===void 0?!0:t==="static"?!1:this.state.isInvalidated?!0:!eE(this.state.dataUpdatedAt,t)}onFocus(){var n;const t=this.observers.find(r=>r.shouldFetchOnWindowFocus());t==null||t.refetch({cancelRefetch:!1}),(n=b(this,we))==null||n.continue()}onOnline(){var n;const t=this.observers.find(r=>r.shouldFetchOnReconnect());t==null||t.refetch({cancelRefetch:!1}),(n=b(this,we))==null||n.continue()}addObserver(t){this.observers.includes(t)||(this.observers.push(t),this.clearGcTimeout(),b(this,rt).notify({type:"observerAdded",query:this,observer:t}))}removeObserver(t){this.observers.includes(t)&&(this.observers=this.observers.filter(n=>n!==t),this.observers.length||(b(this,we)&&(b(this,er)||Ee(this,Ge,wg).call(this)?b(this,we).cancel({revert:!0}):b(this,we).cancelRetry()),this.scheduleGc()),b(this,rt).notify({type:"observerRemoved",query:this,observer:t}))}getObserversCount(){return this.observers.length}invalidate(){this.state.isInvalidated||Ee(this,Ge,Ft).call(this,{type:"invalidate"})}async fetch(t,n){var u,c,f,m,d,S,y,x,h,p,g;if(this.state.fetchStatus!=="idle"&&((u=b(this,we))==null?void 0:u.status())!=="rejected"){if(this.state.data!==void 0&&(n!=null&&n.cancelRefetch))this.cancel({silent:!0});else if(b(this,we))return b(this,we).continueRetry(),b(this,we).promise}if(t&&this.setOptions(t),!this.options.queryFn){const E=this.observers.find(C=>C.options.queryFn);E&&this.setOptions(E.options)}const r=new AbortController,o=E=>{Object.defineProperty(E,"signal",{enumerable:!0,get:()=>(F(this,er,!0),r.signal)})},s=()=>{const E=hg(this.options,n),k=(()=>{const P={client:b(this,Jn),queryKey:this.queryKey,meta:this.meta};return o(P),P})();return F(this,er,!1),this.options.persister?this.options.persister(E,k,this):E(k)},l=(()=>{const E={fetchOptions:n,options:this.options,queryKey:this.queryKey,client:b(this,Jn),state:this.state,fetchFn:s};return o(E),E})(),a=b(this,Wr)==="infinite"?pE(this.options.pages):this.options.behavior;a==null||a.onFetch(l,this),F(this,Hr,this.state),(this.state.fetchStatus==="idle"||this.state.fetchMeta!==((c=l.fetchOptions)==null?void 0:c.meta))&&Ee(this,Ge,Ft).call(this,{type:"fetch",meta:(f=l.fetchOptions)==null?void 0:f.meta}),F(this,we,vg({initialPromise:n==null?void 0:n.initialPromise,fn:l.fetchFn,onCancel:E=>{E instanceof nu&&E.revert&&this.setState({...b(this,Hr),fetchStatus:"idle"}),r.abort()},onFail:(E,C)=>{Ee(this,Ge,Ft).call(this,{type:"failed",failureCount:E,error:C})},onPause:()=>{Ee(this,Ge,Ft).call(this,{type:"pause"})},onContinue:()=>{Ee(this,Ge,Ft).call(this,{type:"continue"})},retry:l.options.retry,retryDelay:l.options.retryDelay,networkMode:l.options.networkMode,canRun:()=>!0}));try{const E=await b(this,we).start();if(E===void 0)throw new Error(`${this.queryHash} data is undefined`);return this.setData(E),(d=(m=b(this,rt).config).onSuccess)==null||d.call(m,E,this),(y=(S=b(this,rt).config).onSettled)==null||y.call(S,E,this.state.error,this),E}catch(E){if(E instanceof nu){if(E.silent)return b(this,we).promise;if(E.revert){if(this.state.data===void 0)throw E;return this.state.data}}throw Ee(this,Ge,Ft).call(this,{type:"error",error:E}),(h=(x=b(this,rt).config).onError)==null||h.call(x,E,this),(g=(p=b(this,rt).config).onSettled)==null||g.call(p,this.state.data,E,this),E}finally{this.scheduleGc()}}},Wr=new WeakMap,Zn=new WeakMap,Hr=new WeakMap,rt=new WeakMap,Jn=new WeakMap,we=new WeakMap,ss=new WeakMap,er=new WeakMap,Ge=new WeakSet,wg=function(){return this.state.fetchStatus==="paused"&&this.state.status==="pending"},Ft=function(t){const n=r=>{switch(t.type){case"failed":return{...r,fetchFailureCount:t.failureCount,fetchFailureReason:t.error};case"pause":return{...r,fetchStatus:"paused"};case"continue":return{...r,fetchStatus:"fetching"};case"fetch":return{...r,...gE(r.data,this.options),fetchMeta:t.meta??null};case"success":const o={...r,...hf(t.data,t.dataUpdatedAt),dataUpdateCount:r.dataUpdateCount+1,...!t.manual&&{fetchStatus:"idle",fetchFailureCount:0,fetchFailureReason:null}};return F(this,Hr,t.manual?o:void 0),o;case"error":const s=t.error;return{...r,error:s,errorUpdateCount:r.errorUpdateCount+1,errorUpdatedAt:Date.now(),fetchFailureCount:r.fetchFailureCount+1,fetchFailureReason:s,fetchStatus:"idle",status:"error",isInvalidated:!0};case"invalidate":return{...r,isInvalidated:!0};case"setState":return{...r,...t.state}}};this.state=n(this.state),Me.batch(()=>{this.observers.forEach(r=>{r.onQueryUpdate()}),b(this,rt).notify({query:this,type:"updated",action:t})})},jf);function gE(e,t){return{fetchFailureCount:0,fetchFailureReason:null,fetchStatus:gg(t.networkMode)?"fetching":"paused",...e===void 0&&{error:null,status:"pending"}}}function hf(e,t){return{data:e,dataUpdatedAt:t??Date.now(),error:null,isInvalidated:!1,status:"success"}}function mf(e){const t=typeof e.initialData=="function"?e.initialData():e.initialData,n=t!==void 0,r=n?typeof e.initialDataUpdatedAt=="function"?e.initialDataUpdatedAt():e.initialDataUpdatedAt:0;return{data:t,dataUpdateCount:0,dataUpdatedAt:n?r??Date.now():0,error:null,errorUpdateCount:0,errorUpdatedAt:0,fetchFailureCount:0,fetchFailureReason:null,fetchMeta:null,isInvalidated:!1,status:n?"success":"pending",fetchStatus:"idle"}}var is,Rt,_e,tr,jt,fn,_f,vE=(_f=class extends yg{constructor(t){super();q(this,jt);q(this,is);q(this,Rt);q(this,_e);q(this,tr);F(this,is,t.client),this.mutationId=t.mutationId,F(this,_e,t.mutationCache),F(this,Rt,[]),this.state=t.state||yE(),this.setOptions(t.options),this.scheduleGc()}setOptions(t){this.options=t,this.updateGcTime(this.options.gcTime)}get meta(){return this.options.meta}addObserver(t){b(this,Rt).includes(t)||(b(this,Rt).push(t),this.clearGcTimeout(),b(this,_e).notify({type:"observerAdded",mutation:this,observer:t}))}removeObserver(t){F(this,Rt,b(this,Rt).filter(n=>n!==t)),this.scheduleGc(),b(this,_e).notify({type:"observerRemoved",mutation:this,observer:t})}optionalRemove(){b(this,Rt).length||(this.state.status==="pending"?this.scheduleGc():b(this,_e).remove(this))}continue(){var t;return((t=b(this,tr))==null?void 0:t.continue())??this.execute(this.state.variables)}async execute(t){var i,l,a,u,c,f,m,d,S,y,x,h,p,g,E,C,k,P;const n=()=>{Ee(this,jt,fn).call(this,{type:"continue"})},r={client:b(this,is),meta:this.options.meta,mutationKey:this.options.mutationKey};F(this,tr,vg({fn:()=>this.options.mutationFn?this.options.mutationFn(t,r):Promise.reject(new Error("No mutationFn found")),onFail:(N,L)=>{Ee(this,jt,fn).call(this,{type:"failed",failureCount:N,error:L})},onPause:()=>{Ee(this,jt,fn).call(this,{type:"pause"})},onContinue:n,retry:this.options.retry??0,retryDelay:this.options.retryDelay,networkMode:this.options.networkMode,canRun:()=>b(this,_e).canRun(this)}));const o=this.state.status==="pending",s=!b(this,tr).canStart();try{if(o)n();else{Ee(this,jt,fn).call(this,{type:"pending",variables:t,isPaused:s}),b(this,_e).config.onMutate&&await b(this,_e).config.onMutate(t,this,r);const L=await((l=(i=this.options).onMutate)==null?void 0:l.call(i,t,r));L!==this.state.context&&Ee(this,jt,fn).call(this,{type:"pending",context:L,variables:t,isPaused:s})}const N=await b(this,tr).start();return await((u=(a=b(this,_e).config).onSuccess)==null?void 0:u.call(a,N,t,this.state.context,this,r)),await((f=(c=this.options).onSuccess)==null?void 0:f.call(c,N,t,this.state.context,r)),await((d=(m=b(this,_e).config).onSettled)==null?void 0:d.call(m,N,null,this.state.variables,this.state.context,this,r)),await((y=(S=this.options).onSettled)==null?void 0:y.call(S,N,null,t,this.state.context,r)),Ee(this,jt,fn).call(this,{type:"success",data:N}),N}catch(N){try{await((h=(x=b(this,_e).config).onError)==null?void 0:h.call(x,N,t,this.state.context,this,r))}catch(L){Promise.reject(L)}try{await((g=(p=this.options).onError)==null?void 0:g.call(p,N,t,this.state.context,r))}catch(L){Promise.reject(L)}try{await((C=(E=b(this,_e).config).onSettled)==null?void 0:C.call(E,void 0,N,this.state.variables,this.state.context,this,r))}catch(L){Promise.reject(L)}try{await((P=(k=this.options).onSettled)==null?void 0:P.call(k,void 0,N,t,this.state.context,r))}catch(L){Promise.reject(L)}throw Ee(this,jt,fn).call(this,{type:"error",error:N}),N}finally{b(this,_e).runNext(this)}}},is=new WeakMap,Rt=new WeakMap,_e=new WeakMap,tr=new WeakMap,jt=new WeakSet,fn=function(t){const n=r=>{switch(t.type){case"failed":return{...r,failureCount:t.failureCount,failureReason:t.error};case"pause":return{...r,isPaused:!0};case"continue":return{...r,isPaused:!1};case"pending":return{...r,context:t.context,data:void 0,failureCount:0,failureReason:null,error:null,isPaused:t.isPaused,status:"pending",variables:t.variables,submittedAt:Date.now()};case"success":return{...r,data:t.data,failureCount:0,failureReason:null,error:null,status:"success",isPaused:!1};case"error":return{...r,data:void 0,error:t.error,failureCount:r.failureCount+1,failureReason:t.error,isPaused:!1,status:"error"}}};this.state=n(this.state),Me.batch(()=>{b(this,Rt).forEach(r=>{r.onMutationUpdate(t)}),b(this,_e).notify({mutation:this,type:"updated",action:t})})},_f);function yE(){return{context:void 0,data:void 0,error:null,failureCount:0,failureReason:null,isPaused:!1,status:"idle",variables:void 0,submittedAt:0}}var Ut,vt,ls,Af,wE=(Af=class extends al{constructor(t={}){super();q(this,Ut);q(this,vt);q(this,ls);this.config=t,F(this,Ut,new Set),F(this,vt,new Map),F(this,ls,0)}build(t,n,r){const o=new vE({client:t,mutationCache:this,mutationId:++ws(this,ls)._,options:t.defaultMutationOptions(n),state:r});return this.add(o),o}add(t){b(this,Ut).add(t);const n=Bs(t);if(typeof n=="string"){const r=b(this,vt).get(n);r?r.push(t):b(this,vt).set(n,[t])}this.notify({type:"added",mutation:t})}remove(t){if(b(this,Ut).delete(t)){const n=Bs(t);if(typeof n=="string"){const r=b(this,vt).get(n);if(r)if(r.length>1){const o=r.indexOf(t);o!==-1&&r.splice(o,1)}else r[0]===t&&b(this,vt).delete(n)}}this.notify({type:"removed",mutation:t})}canRun(t){const n=Bs(t);if(typeof n=="string"){const r=b(this,vt).get(n),o=r==null?void 0:r.find(s=>s.state.status==="pending");return!o||o===t}else return!0}runNext(t){var r;const n=Bs(t);if(typeof n=="string"){const o=(r=b(this,vt).get(n))==null?void 0:r.find(s=>s!==t&&s.state.isPaused);return(o==null?void 0:o.continue())??Promise.resolve()}else return Promise.resolve()}clear(){Me.batch(()=>{b(this,Ut).forEach(t=>{this.notify({type:"removed",mutation:t})}),b(this,Ut).clear(),b(this,vt).clear()})}getAll(){return Array.from(b(this,Ut))}find(t){const n={exact:!0,...t};return this.getAll().find(r=>cf(n,r))}findAll(t={}){return this.getAll().filter(n=>cf(t,n))}notify(t){Me.batch(()=>{this.listeners.forEach(n=>{n(t)})})}resumePausedMutations(){const t=this.getAll().filter(n=>n.state.isPaused);return Me.batch(()=>Promise.all(t.map(n=>n.continue().catch(gt))))}},Ut=new WeakMap,vt=new WeakMap,ls=new WeakMap,Af);function Bs(e){var t;return(t=e.options.scope)==null?void 0:t.id}var _t,Of,xE=(Of=class extends al{constructor(t={}){super();q(this,_t);this.config=t,F(this,_t,new Map)}build(t,n,r){const o=n.queryKey,s=n.queryHash??gc(o,n);let i=this.get(s);return i||(i=new mE({client:t,queryKey:o,queryHash:s,options:t.defaultQueryOptions(n),state:r,defaultOptions:t.getQueryDefaults(o)}),this.add(i)),i}add(t){b(this,_t).has(t.queryHash)||(b(this,_t).set(t.queryHash,t),this.notify({type:"added",query:t}))}remove(t){const n=b(this,_t).get(t.queryHash);n&&(t.destroy(),n===t&&b(this,_t).delete(t.queryHash),this.notify({type:"removed",query:t}))}clear(){Me.batch(()=>{this.getAll().forEach(t=>{this.remove(t)})})}get(t){return b(this,_t).get(t)}getAll(){return[...b(this,_t).values()]}find(t){const n={exact:!0,...t};return this.getAll().find(r=>uf(n,r))}findAll(t={}){const n=this.getAll();return Object.keys(t).length>0?n.filter(r=>uf(t,r)):n}notify(t){Me.batch(()=>{this.listeners.forEach(n=>{n(t)})})}onFocus(){Me.batch(()=>{this.getAll().forEach(t=>{t.onFocus()})})}onOnline(){Me.batch(()=>{this.getAll().forEach(t=>{t.onOnline()})})}},_t=new WeakMap,Of),de,xn,Sn,Qr,Kr,En,Gr,Yr,Lf,SE=(Lf=class{constructor(e={}){q(this,de);q(this,xn);q(this,Sn);q(this,Qr);q(this,Kr);q(this,En);q(this,Gr);q(this,Yr);F(this,de,e.queryCache||new xE),F(this,xn,e.mutationCache||new wE),F(this,Sn,e.defaultOptions||{}),F(this,Qr,new Map),F(this,Kr,new Map),F(this,En,0)}mount(){ws(this,En)._++,b(this,En)===1&&(F(this,Gr,fg.subscribe(async e=>{e&&(await this.resumePausedMutations(),b(this,de).onFocus())})),F(this,Yr,Ai.subscribe(async e=>{e&&(await this.resumePausedMutations(),b(this,de).onOnline())})))}unmount(){var e,t;ws(this,En)._--,b(this,En)===0&&((e=b(this,Gr))==null||e.call(this),F(this,Gr,void 0),(t=b(this,Yr))==null||t.call(this),F(this,Yr,void 0))}isFetching(e){return b(this,de).findAll({...e,fetchStatus:"fetching"}).length}isMutating(e){return b(this,xn).findAll({...e,status:"pending"}).length}getQueryData(e){var n;const t=this.defaultQueryOptions({queryKey:e});return(n=b(this,de).get(t.queryHash))==null?void 0:n.state.data}ensureQueryData(e){const t=this.defaultQueryOptions(e),n=b(this,de).build(this,t),r=n.state.data;return r===void 0?this.fetchQuery(e):(e.revalidateIfStale&&n.isStaleByTime(eu(t.staleTime,n))&&this.prefetchQuery(t),Promise.resolve(r))}getQueriesData(e){return b(this,de).findAll(e).map(({queryKey:t,state:n})=>{const r=n.data;return[t,r]})}setQueryData(e,t,n){const r=this.defaultQueryOptions({queryKey:e}),o=b(this,de).get(r.queryHash),s=o==null?void 0:o.state.data,i=ZS(t,s);if(i!==void 0)return b(this,de).build(this,r).setData(i,{...n,manual:!0})}setQueriesData(e,t,n){return Me.batch(()=>b(this,de).findAll(e).map(({queryKey:r})=>[r,this.setQueryData(r,t,n)]))}getQueryState(e){var n;const t=this.defaultQueryOptions({queryKey:e});return(n=b(this,de).get(t.queryHash))==null?void 0:n.state}removeQueries(e){const t=b(this,de);Me.batch(()=>{t.findAll(e).forEach(n=>{t.remove(n)})})}resetQueries(e,t){const n=b(this,de);return Me.batch(()=>(n.findAll(e).forEach(r=>{r.reset()}),this.refetchQueries({type:"active",...e},t)))}cancelQueries(e,t={}){const n={revert:!0,...t},r=Me.batch(()=>b(this,de).findAll(e).map(o=>o.cancel(n)));return Promise.all(r).then(gt).catch(gt)}invalidateQueries(e,t={}){return Me.batch(()=>(b(this,de).findAll(e).forEach(n=>{n.invalidate()}),(e==null?void 0:e.refetchType)==="none"?Promise.resolve():this.refetchQueries({...e,type:(e==null?void 0:e.refetchType)??(e==null?void 0:e.type)??"active"},t)))}refetchQueries(e,t={}){const n={...t,cancelRefetch:t.cancelRefetch??!0},r=Me.batch(()=>b(this,de).findAll(e).filter(o=>!o.isDisabled()&&!o.isStatic()).map(o=>{let s=o.fetch(void 0,n);return n.throwOnError||(s=s.catch(gt)),o.state.fetchStatus==="paused"?Promise.resolve():s}));return Promise.all(r).then(gt)}fetchQuery(e){const t=this.defaultQueryOptions(e);t.retry===void 0&&(t.retry=!1);const n=b(this,de).build(this,t);return n.isStaleByTime(eu(t.staleTime,n))?n.fetch(t):Promise.resolve(n.state.data)}prefetchQuery(e){return this.fetchQuery(e).then(gt).catch(gt)}fetchInfiniteQuery(e){return e._type="infinite",this.fetchQuery(e)}prefetchInfiniteQuery(e){return this.fetchInfiniteQuery(e).then(gt).catch(gt)}ensureInfiniteQueryData(e){return e._type="infinite",this.ensureQueryData(e)}resumePausedMutations(){return Ai.isOnline()?b(this,xn).resumePausedMutations():Promise.resolve()}getQueryCache(){return b(this,de)}getMutationCache(){return b(this,xn)}getDefaultOptions(){return b(this,Sn)}setDefaultOptions(e){F(this,Sn,e)}setQueryDefaults(e,t){b(this,Qr).set(rs(e),{queryKey:e,defaultOptions:t})}getQueryDefaults(e){const t=[...b(this,Qr).values()],n={};return t.forEach(r=>{os(e,r.queryKey)&&Object.assign(n,r.defaultOptions)}),n}setMutationDefaults(e,t){b(this,Kr).set(rs(e),{mutationKey:e,defaultOptions:t})}getMutationDefaults(e){const t=[...b(this,Kr).values()],n={};return t.forEach(r=>{os(e,r.mutationKey)&&Object.assign(n,r.defaultOptions)}),n}defaultQueryOptions(e){if(e._defaulted)return e;const t={...b(this,Sn).queries,...this.getQueryDefaults(e.queryKey),...e,_defaulted:!0};return t.queryHash||(t.queryHash=gc(t.queryKey,t)),t.refetchOnReconnect===void 0&&(t.refetchOnReconnect=t.networkMode!=="always"),t.throwOnError===void 0&&(t.throwOnError=!!t.suspense),!t.networkMode&&t.persister&&(t.networkMode="offlineFirst"),t.queryFn===vc&&(t.enabled=!1),t}defaultMutationOptions(e){return e!=null&&e._defaulted?e:{...b(this,Sn).mutations,...(e==null?void 0:e.mutationKey)&&this.getMutationDefaults(e.mutationKey),...e,_defaulted:!0}}clear(){b(this,de).clear(),b(this,xn).clear()}},de=new WeakMap,xn=new WeakMap,Sn=new WeakMap,Qr=new WeakMap,Kr=new WeakMap,En=new WeakMap,Gr=new WeakMap,Yr=new WeakMap,Lf),EE=w.createContext(void 0),CE=({client:e,children:t})=>(w.useEffect(()=>(e.mount(),()=>{e.unmount()}),[e]),v.jsx(EE.Provider,{value:e,children:t}));/**
 * @remix-run/router v1.23.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Oi(){return Oi=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)({}).hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Oi.apply(null,arguments)}var Pn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(Pn||(Pn={}));const gf="popstate";function kE(e){e===void 0&&(e={});function t(r,o){let{pathname:s,search:i,hash:l}=r.location;return ru("",{pathname:s,search:i,hash:l},o.state&&o.state.usr||null,o.state&&o.state.key||"default")}function n(r,o){return typeof o=="string"?o:Sg(o)}return bE(t,n,null,e)}function He(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function xg(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function PE(){return Math.random().toString(36).substr(2,8)}function vf(e,t){return{usr:e.state,key:e.key,idx:t}}function ru(e,t,n,r){return n===void 0&&(n=null),Oi({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?ul(t):t,{state:n,key:t&&t.key||r||PE()})}function Sg(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function ul(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function bE(e,t,n,r){r===void 0&&(r={});let{window:o=document.defaultView,v5Compat:s=!1}=r,i=o.history,l=Pn.Pop,a=null,u=c();u==null&&(u=0,i.replaceState(Oi({},i.state,{idx:u}),""));function c(){return(i.state||{idx:null}).idx}function f(){l=Pn.Pop;let x=c(),h=x==null?null:x-u;u=x,a&&a({action:l,location:y.location,delta:h})}function m(x,h){l=Pn.Push;let p=ru(y.location,x,h);u=c()+1;let g=vf(p,u),E=y.createHref(p);try{i.pushState(g,"",E)}catch(C){if(C instanceof DOMException&&C.name==="DataCloneError")throw C;o.location.assign(E)}s&&a&&a({action:l,location:y.location,delta:1})}function d(x,h){l=Pn.Replace;let p=ru(y.location,x,h);u=c();let g=vf(p,u),E=y.createHref(p);i.replaceState(g,"",E),s&&a&&a({action:l,location:y.location,delta:0})}function S(x){let h=o.location.origin!=="null"?o.location.origin:o.location.href,p=typeof x=="string"?x:Sg(x);return p=p.replace(/ $/,"%20"),He(h,"No window.location.(origin|href) available to create URL for href: "+p),new URL(p,h)}let y={get action(){return l},get location(){return e(o,i)},listen(x){if(a)throw new Error("A history only accepts one active listener");return o.addEventListener(gf,f),a=x,()=>{o.removeEventListener(gf,f),a=null}},createHref(x){return t(o,x)},createURL:S,encodeLocation(x){let h=S(x);return{pathname:h.pathname,search:h.search,hash:h.hash}},push:m,replace:d,go(x){return i.go(x)}};return y}var yf;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(yf||(yf={}));function TE(e,t,n){return n===void 0&&(n="/"),NE(e,t,n)}function NE(e,t,n,r){let o=typeof t=="string"?ul(t):t,s=kg(o.pathname||"/",n);if(s==null)return null;let i=Eg(e);RE(i);let l=null,a=BE(s);for(let u=0;l==null&&u<i.length;++u)l=zE(i[u],a);return l}function Eg(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let o=(s,i,l)=>{let a={relativePath:l===void 0?s.path||"":l,caseSensitive:s.caseSensitive===!0,childrenIndex:i,route:s};a.relativePath.startsWith("/")&&(He(a.relativePath.startsWith(r),'Absolute route path "'+a.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),a.relativePath=a.relativePath.slice(r.length));let u=Fr([r,a.relativePath]),c=n.concat(a);s.children&&s.children.length>0&&(He(s.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),Eg(s.children,t,c,u)),!(s.path==null&&!s.index)&&t.push({path:u,score:IE(u,s.index),routesMeta:c})};return e.forEach((s,i)=>{var l;if(s.path===""||!((l=s.path)!=null&&l.includes("?")))o(s,i);else for(let a of Cg(s.path))o(s,i,a)}),t}function Cg(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,o=n.endsWith("?"),s=n.replace(/\?$/,"");if(r.length===0)return o?[s,""]:[s];let i=Cg(r.join("/")),l=[];return l.push(...i.map(a=>a===""?s:[s,a].join("/"))),o&&l.push(...i),l.map(a=>e.startsWith("/")&&a===""?"/":a)}function RE(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:DE(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const jE=/^:[\w-]+$/,_E=3,AE=2,OE=1,LE=10,ME=-2,wf=e=>e==="*";function IE(e,t){let n=e.split("/"),r=n.length;return n.some(wf)&&(r+=ME),t&&(r+=AE),n.filter(o=>!wf(o)).reduce((o,s)=>o+(jE.test(s)?_E:s===""?OE:LE),r)}function DE(e,t){return e.length===t.length&&e.slice(0,-1).every((r,o)=>r===t[o])?e[e.length-1]-t[t.length-1]:0}function zE(e,t,n){let{routesMeta:r}=e,o={},s="/",i=[];for(let l=0;l<r.length;++l){let a=r[l],u=l===r.length-1,c=s==="/"?t:t.slice(s.length)||"/",f=FE({path:a.relativePath,caseSensitive:a.caseSensitive,end:u},c),m=a.route;if(!f)return null;Object.assign(o,f.params),i.push({params:o,pathname:Fr([s,f.pathname]),pathnameBase:VE(Fr([s,f.pathnameBase])),route:m}),f.pathnameBase!=="/"&&(s=Fr([s,f.pathnameBase]))}return i}function FE(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=$E(e.path,e.caseSensitive,e.end),o=t.match(n);if(!o)return null;let s=o[0],i=s.replace(/(.)\/+$/,"$1"),l=o.slice(1);return{params:r.reduce((u,c,f)=>{let{paramName:m,isOptional:d}=c;if(m==="*"){let y=l[f]||"";i=s.slice(0,s.length-y.length).replace(/(.)\/+$/,"$1")}const S=l[f];return d&&!S?u[m]=void 0:u[m]=(S||"").replace(/%2F/g,"/"),u},{}),pathname:s,pathnameBase:i,pattern:e}}function $E(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),xg(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],o="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(i,l,a)=>(r.push({paramName:l,isOptional:a!=null}),a?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),o+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?o+="\\/*$":e!==""&&e!=="/"&&(o+="(?:(?=\\/|$))"),[new RegExp(o,t?void 0:"i"),r]}function BE(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return xg(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function kg(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const UE=e=>e.replace(/\/\/+/g,"/"),Fr=e=>UE(e.join("/")),VE=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/");function WE(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Pg=["post","put","patch","delete"];new Set(Pg);const HE=["get",...Pg];new Set(HE);/**
 * React Router v6.30.4
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Li(){return Li=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)({}).hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Li.apply(null,arguments)}const QE=w.createContext(null),KE=w.createContext(null),bg=w.createContext(null),cl=w.createContext(null),dl=w.createContext({outlet:null,matches:[],isDataRoute:!1}),Tg=w.createContext(null);function yc(){return w.useContext(cl)!=null}function Ng(){return yc()||He(!1),w.useContext(cl).location}function GE(e,t){return YE(e,t)}function YE(e,t,n,r){yc()||He(!1);let{navigator:o}=w.useContext(bg),{matches:s}=w.useContext(dl),i=s[s.length-1],l=i?i.params:{};i&&i.pathname;let a=i?i.pathnameBase:"/";i&&i.route;let u=Ng(),c;if(t){var f;let x=typeof t=="string"?ul(t):t;a==="/"||(f=x.pathname)!=null&&f.startsWith(a)||He(!1),c=x}else c=u;let m=c.pathname||"/",d=m;if(a!=="/"){let x=a.replace(/^\//,"").split("/");d="/"+m.replace(/^\//,"").split("/").slice(x.length).join("/")}let S=TE(e,{pathname:d}),y=eC(S&&S.map(x=>Object.assign({},x,{params:Object.assign({},l,x.params),pathname:Fr([a,o.encodeLocation?o.encodeLocation(x.pathname).pathname:x.pathname]),pathnameBase:x.pathnameBase==="/"?a:Fr([a,o.encodeLocation?o.encodeLocation(x.pathnameBase).pathname:x.pathnameBase])})),s,n,r);return t&&y?w.createElement(cl.Provider,{value:{location:Li({pathname:"/",search:"",hash:"",state:null,key:"default"},c),navigationType:Pn.Pop}},y):y}function XE(){let e=oC(),t=WE(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,o={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return w.createElement(w.Fragment,null,w.createElement("h2",null,"Unexpected Application Error!"),w.createElement("h3",{style:{fontStyle:"italic"}},t),n?w.createElement("pre",{style:o},n):null,null)}const qE=w.createElement(XE,null);class ZE extends w.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?w.createElement(dl.Provider,{value:this.props.routeContext},w.createElement(Tg.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function JE(e){let{routeContext:t,match:n,children:r}=e,o=w.useContext(QE);return o&&o.static&&o.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(o.staticContext._deepestRenderedBoundaryId=n.route.id),w.createElement(dl.Provider,{value:t},r)}function eC(e,t,n,r){var o;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var s;if(!n)return null;if(n.errors)e=n.matches;else if((s=r)!=null&&s.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let i=e,l=(o=n)==null?void 0:o.errors;if(l!=null){let c=i.findIndex(f=>f.route.id&&(l==null?void 0:l[f.route.id])!==void 0);c>=0||He(!1),i=i.slice(0,Math.min(i.length,c+1))}let a=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let c=0;c<i.length;c++){let f=i[c];if((f.route.HydrateFallback||f.route.hydrateFallbackElement)&&(u=c),f.route.id){let{loaderData:m,errors:d}=n,S=f.route.loader&&m[f.route.id]===void 0&&(!d||d[f.route.id]===void 0);if(f.route.lazy||S){a=!0,u>=0?i=i.slice(0,u+1):i=[i[0]];break}}}return i.reduceRight((c,f,m)=>{let d,S=!1,y=null,x=null;n&&(d=l&&f.route.id?l[f.route.id]:void 0,y=f.route.errorElement||qE,a&&(u<0&&m===0?(sC("route-fallback"),S=!0,x=null):u===m&&(S=!0,x=f.route.hydrateFallbackElement||null)));let h=t.concat(i.slice(0,m+1)),p=()=>{let g;return d?g=y:S?g=x:f.route.Component?g=w.createElement(f.route.Component,null):f.route.element?g=f.route.element:g=c,w.createElement(JE,{match:f,routeContext:{outlet:c,matches:h,isDataRoute:n!=null},children:g})};return n&&(f.route.ErrorBoundary||f.route.errorElement||m===0)?w.createElement(ZE,{location:n.location,revalidation:n.revalidation,component:y,error:d,children:p(),routeContext:{outlet:null,matches:h,isDataRoute:!0}}):p()},null)}var Rg=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Rg||{});function tC(e){let t=w.useContext(KE);return t||He(!1),t}function nC(e){let t=w.useContext(dl);return t||He(!1),t}function rC(e){let t=nC(),n=t.matches[t.matches.length-1];return n.route.id||He(!1),n.route.id}function oC(){var e;let t=w.useContext(Tg),n=tC(Rg.UseRouteError),r=rC();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}const xf={};function sC(e,t,n){xf[e]||(xf[e]=!0)}function iC(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function ou(e){He(!1)}function lC(e){let{basename:t="/",children:n=null,location:r,navigationType:o=Pn.Pop,navigator:s,static:i=!1,future:l}=e;yc()&&He(!1);let a=t.replace(/^\/*/,"/"),u=w.useMemo(()=>({basename:a,navigator:s,static:i,future:Li({v7_relativeSplatPath:!1},l)}),[a,l,s,i]);typeof r=="string"&&(r=ul(r));let{pathname:c="/",search:f="",hash:m="",state:d=null,key:S="default"}=r,y=w.useMemo(()=>{let x=kg(c,a);return x==null?null:{location:{pathname:x,search:f,hash:m,state:d,key:S},navigationType:o}},[a,c,f,m,d,S,o]);return y==null?null:w.createElement(bg.Provider,{value:u},w.createElement(cl.Provider,{children:n,value:y}))}function aC(e){let{children:t,location:n}=e;return GE(su(t),n)}new Promise(()=>{});function su(e,t){t===void 0&&(t=[]);let n=[];return w.Children.forEach(e,(r,o)=>{if(!w.isValidElement(r))return;let s=[...t,o];if(r.type===w.Fragment){n.push.apply(n,su(r.props.children,s));return}r.type!==ou&&He(!1),!r.props.index||!r.props.children||He(!1);let i={id:r.props.id||s.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(i.children=su(r.props.children,s)),n.push(i)}),n}/**
 * React Router DOM v6.30.4
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */const uC="6";try{window.__reactRouterVersion=uC}catch{}const cC="startTransition",Sf=du[cC];function dC(e){let{basename:t,children:n,future:r,window:o}=e,s=w.useRef();s.current==null&&(s.current=kE({window:o,v5Compat:!0}));let i=s.current,[l,a]=w.useState({action:i.action,location:i.location}),{v7_startTransition:u}=r||{},c=w.useCallback(f=>{u&&Sf?Sf(()=>a(f)):a(f)},[a,u]);return w.useLayoutEffect(()=>i.listen(c),[i,c]),w.useEffect(()=>iC(r),[r]),w.createElement(lC,{basename:t,children:n,location:l.location,navigationType:l.action,navigator:i,future:r})}var Ef;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(Ef||(Ef={}));var Cf;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(Cf||(Cf={}));const fC=()=>v.jsx("header",{className:"w-full py-4 px-4 sm:px-6 lg:px-8 border-b",children:v.jsxs("div",{className:"max-w-7xl mx-auto flex justify-between items-center",children:[v.jsxs("div",{className:"flex items-center space-x-2",children:[v.jsx(Em,{className:"w-8 h-8 text-leaf"}),v.jsx("span",{className:"text-xl font-bold",children:"PaddyGuard"})]}),v.jsxs("nav",{className:"hidden md:flex items-center space-x-8",children:[v.jsx("a",{href:"#how-it-works",className:"text-gray-600 hover:text-leaf transition-colors",children:"How It Works"}),v.jsx("a",{href:"#benefits",className:"text-gray-600 hover:text-leaf transition-colors",children:"Benefits"}),v.jsx("a",{href:"#try-now",className:"text-gray-600 hover:text-leaf transition-colors",children:"Try Now"})]}),v.jsx("div",{children:v.jsx("a",{href:"#try-now",className:"bg-leaf hover:bg-leaf-dark text-white px-4 py-2 rounded-md transition-colors",children:"Get Started"})})]})});var pC=Symbol.for("react.lazy"),Mi=du[" use ".trim().toString()];function hC(e){return typeof e=="object"&&e!==null&&"then"in e}function jg(e){return e!=null&&typeof e=="object"&&"$$typeof"in e&&e.$$typeof===pC&&"_payload"in e&&hC(e._payload)}function _g(e){const t=gC(e),n=w.forwardRef((r,o)=>{let{children:s,...i}=r;jg(s)&&typeof Mi=="function"&&(s=Mi(s._payload));const l=w.Children.toArray(s),a=l.find(yC);if(a){const u=a.props.children,c=l.map(f=>f===a?w.Children.count(u)>1?w.Children.only(null):w.isValidElement(u)?u.props.children:null:f);return v.jsx(t,{...i,ref:o,children:w.isValidElement(u)?w.cloneElement(u,void 0,c):null})}return v.jsx(t,{...i,ref:o,children:s})});return n.displayName=`${e}.Slot`,n}var mC=_g("Slot");function gC(e){const t=w.forwardRef((n,r)=>{let{children:o,...s}=n;if(jg(o)&&typeof Mi=="function"&&(o=Mi(o._payload)),w.isValidElement(o)){const i=xC(o),l=wC(s,o.props);return o.type!==w.Fragment&&(l.ref=r?Zi(r,i):i),w.cloneElement(o,l)}return w.Children.count(o)>1?w.Children.only(null):null});return t.displayName=`${e}.SlotClone`,t}var vC=Symbol("radix.slottable");function yC(e){return w.isValidElement(e)&&typeof e.type=="function"&&"__radixId"in e.type&&e.type.__radixId===vC}function wC(e,t){const n={...t};for(const r in t){const o=e[r],s=t[r];/^on[A-Z]/.test(r)?o&&s?n[r]=(...l)=>{const a=s(...l);return o(...l),a}:o&&(n[r]=o):r==="style"?n[r]={...o,...s}:r==="className"&&(n[r]=[o,s].filter(Boolean).join(" "))}return{...e,...n}}function xC(e){var r,o;let t=(r=Object.getOwnPropertyDescriptor(e.props,"ref"))==null?void 0:r.get,n=t&&"isReactWarning"in t&&t.isReactWarning;return n?e.ref:(t=(o=Object.getOwnPropertyDescriptor(e,"ref"))==null?void 0:o.get,n=t&&"isReactWarning"in t&&t.isReactWarning,n?e.props.ref:e.props.ref||e.ref)}const SC=jm("inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",{variants:{variant:{default:"bg-primary text-primary-foreground hover:bg-primary/90",destructive:"bg-destructive text-destructive-foreground hover:bg-destructive/90",outline:"border border-input bg-background hover:bg-accent hover:text-accent-foreground",secondary:"bg-secondary text-secondary-foreground hover:bg-secondary/80",ghost:"hover:bg-accent hover:text-accent-foreground",link:"text-primary underline-offset-4 hover:underline"},size:{default:"h-10 px-4 py-2",sm:"h-9 rounded-md px-3",lg:"h-11 rounded-md px-8",icon:"h-10 w-10"}},defaultVariants:{variant:"default",size:"default"}}),$r=w.forwardRef(({className:e,variant:t,size:n,asChild:r=!1,...o},s)=>{const i=r?mC:"button";return v.jsx(i,{className:Te(SC({variant:t,size:n,className:e})),ref:s,...o})});$r.displayName="Button";const EC=()=>v.jsx("section",{className:"py-16 md:py-24 px-4",children:v.jsxs("div",{className:"max-w-7xl mx-auto grid md:grid-cols-2 gap-12 items-center",children:[v.jsxs("div",{className:"space-y-6",children:[v.jsxs("h1",{className:"text-4xl md:text-5xl lg:text-6xl font-bold leading-tight",children:["Detect Rice Diseases ",v.jsx("span",{className:"text-leaf",children:"Instantly"})]}),v.jsx("p",{className:"text-xl text-gray-600 md:pr-12",children:"Upload a photo of your plant's leaves and get accurate disease diagnosis powered by AI. Save your plants before it's too late."}),v.jsxs("div",{className:"pt-4 flex flex-col sm:flex-row gap-4",children:[v.jsx($r,{size:"lg",className:"bg-leaf hover:bg-leaf-dark",children:v.jsxs("a",{href:"#try-now",className:"flex items-center",children:["Try Now",v.jsx(Mw,{className:"ml-2 h-4 w-4"})]})}),v.jsx($r,{variant:"outline",size:"lg",children:"Learn More"})]})]}),v.jsxs("div",{className:"relative",children:[v.jsx("div",{className:"absolute -inset-0.5 bg-gradient-to-r from-leaf to-leaf-light rounded-2xl blur opacity-30"}),v.jsx("div",{className:"relative overflow-hidden rounded-2xl shadow-xl animate-float",children:v.jsx("img",{src:"https://images.unsplash.com/photo-1728895604559-a4e16081504e?q=80&w=2874&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",alt:"Healthy and unhealthy plant leaves",className:"w-full h-auto object-cover rounded-2xl shadow-lg"})})]})]})}),CC=[{icon:v.jsx(Iw,{className:"w-12 h-12 text-leaf"}),title:"Take a Photo",description:"Snap a clear picture of the affected leaf or Rice part showing visible symptoms."},{icon:v.jsx(Ww,{className:"w-12 h-12 text-leaf"}),title:"AI Analysis",description:"Our powerful AI analyzes the image, comparing it against thousands of known Rice diseases."},{icon:v.jsx(Fw,{className:"w-12 h-12 text-leaf"}),title:"Get Results",description:"Receive a detailed diagnosis with suggested treatments and prevention tips."}],kC=()=>v.jsx("section",{id:"how-it-works",className:"py-16 bg-gray-50 leaf-pattern-bg",children:v.jsxs("div",{className:"max-w-7xl mx-auto px-4",children:[v.jsxs("div",{className:"text-center mb-16",children:[v.jsx("h2",{className:"text-3xl md:text-4xl font-bold mb-4",children:"How It Works"}),v.jsx("p",{className:"text-xl text-gray-600 max-w-3xl mx-auto",children:"Identifying Rice diseases has never been easier. Three simple steps to diagnose and save your rice plants."})]}),v.jsx("div",{className:"grid md:grid-cols-3 gap-8",children:CC.map((e,t)=>v.jsxs("div",{className:"leaf-card p-8 flex flex-col items-center text-center",children:[v.jsx("div",{className:"mb-6 p-4 rounded-full bg-leaf/10",children:e.icon}),v.jsx("h3",{className:"text-xl font-semibold mb-3",children:e.title}),v.jsx("p",{className:"text-gray-600",children:e.description}),v.jsxs("div",{className:"mt-6 text-leaf font-bold",children:["Step ",t+1]})]},t))})]})}),PC=[{icon:v.jsx($w,{className:"w-8 h-8 text-leaf"}),title:"Save Time",description:"Get instant results without waiting for lab tests or expert consultations."},{icon:v.jsx(Qw,{className:"w-8 h-8 text-leaf"}),title:"High Accuracy",description:"Our AI is trained on millions of images for precise diagnosis of 50+ plant diseases."},{icon:v.jsx(Kw,{className:"w-8 h-8 text-leaf"}),title:"Improve Yield",description:"Early detection helps prevent crop loss and increases your harvest yield."},{icon:v.jsx(Hw,{className:"w-8 h-8 text-leaf"}),title:"Reduce Pesticide Use",description:"Targeted treatment recommendations help minimize unnecessary chemical use."}],bC=()=>v.jsx("section",{id:"benefits",className:"py-16 px-4",children:v.jsxs("div",{className:"max-w-7xl mx-auto",children:[v.jsxs("div",{className:"text-center mb-16",children:[v.jsx("h2",{className:"text-3xl md:text-4xl font-bold mb-4",children:"Why Choose PaddyGuard"}),v.jsx("p",{className:"text-xl text-gray-600 max-w-3xl mx-auto",children:"Our advanced Rice disease detection offers numerous advantages for gardeners, farmers and rice enthusiasts."})]}),v.jsx("div",{className:"grid md:grid-cols-2 gap-8",children:PC.map((e,t)=>v.jsxs("div",{className:"p-8 border rounded-lg hover:border-leaf transition-colors flex items-start space-x-5",children:[v.jsx("div",{className:"p-3 rounded-full bg-leaf/10 flex-shrink-0",children:e.icon}),v.jsxs("div",{children:[v.jsx("h3",{className:"text-xl font-semibold mb-2",children:e.title}),v.jsx("p",{className:"text-gray-600",children:e.description})]})]},t))})]})}),Us={"Bacterial Leaf Blight":`• Avoid water stagnation (खेत में पानी जमा न होने दें)
• Use balanced fertilizers, avoid excess nitrogen (संतुलित खाद दें, ज्यादा यूरिया न डालें)
• Spray copper-based bactericide (कॉपर आधारित दवा का छिड़काव करें)
• Remove infected leaves (रोगी पत्तों को हटा दें)`,"Brown Spot":`• Improve soil fertility (मिट्टी की उर्वरता बढ़ाएं)
• Maintain proper irrigation (पानी की सही मात्रा रखें)
• Spray Mancozeb fungicide (Mancozeb दवा का छिड़काव करें)
• Provide proper nutrition (पौधों को पोषण दें)`,"Leaf Blast":`• Spray Tricyclazole fungicide (Tricyclazole दवा का छिड़काव करें)
• Maintain plant spacing (पौधों के बीच दूरी रखें)
• Avoid excess moisture (ज्यादा नमी से बचें)
• Remove infected leaves (रोगी पत्तों को हटा दें)`,Healthy:`• Crop is healthy (फसल स्वस्थ है ✅)
• Maintain regular irrigation (नियमित पानी दें)
• Apply balanced fertilizers (संतुलित खाद दें)
• Monitor regularly (समय-समय पर जांच करें)`},TC=e=>e?e.includes("Blast")?Us["Leaf Blast"]:e.includes("Brown")?Us["Brown Spot"]:e.includes("Bacterial")?Us["Bacterial Leaf Blight"]:e.includes("Healthy")?Us.Healthy:"No solution available":"No solution available",NC=()=>{const[e,t]=w.useState(null),[n,r]=w.useState(!1),[o,s]=w.useState(!1),[i,l]=w.useState(null),{toast:a}=Xh(),u=w.useRef(null),c=x=>{x.preventDefault(),r(!0)},f=()=>{r(!1)},m=x=>{x.preventDefault(),r(!1),x.dataTransfer.files&&x.dataTransfer.files[0]&&S(x.dataTransfer.files[0])},d=x=>{x.target.files&&x.target.files[0]&&S(x.target.files[0])},S=x=>{if(!x.type.match("image.*")){a({title:"Invalid file type",description:"Please upload an image file (JPEG, PNG, etc.)",variant:"destructive"});return}const h=new FileReader;h.onload=p=>{var g;(g=p.target)!=null&&g.result&&(t(p.target.result),l(null))},h.readAsDataURL(x)},y=async()=>{if(e){s(!0),l(null);try{const x=new FormData,p=await(await fetch(e)).blob();x.append("file",p,"leaf-image.jpg");const E=await(await fetch("https://rice-leaf-disease-detection-waco.onrender.com/predict"),{method:"POST",body:x})).json();if(E.prediction){const C=E.prediction.replace(/_/g," ").replace(/\s+/g," ").trim();l(C),a({title:"Analysis Complete",description:`Prediction: ${E.prediction}`})}else throw new Error("No prediction received")}catch(x){console.error("Error analyzing image:",x),a({title:"Error",description:"Something went wrong during image analysis.",variant:"destructive"})}finally{s(!1)}}};return v.jsx("section",{id:"try-now",className:"py-16 bg-gradient-to-b from-white to-gray-50 px-4",children:v.jsxs("div",{className:"max-w-4xl mx-auto",children:[v.jsxs("div",{className:"text-center mb-12",children:[v.jsx("h2",{className:"text-3xl md:text-4xl font-bold mb-4",children:"Try PaddyGuard Now"}),v.jsx("p",{className:"text-xl text-gray-600 max-w-3xl mx-auto",children:"Upload a photo of your plant's leaves to get an instant diagnosis. It's free and no registration required."})]}),v.jsx("div",{className:"bg-white rounded-xl shadow-lg p-6 md:p-8",children:e?v.jsxs("div",{className:"space-y-8",children:[v.jsxs("div",{className:"relative rounded-lg overflow-hidden",children:[v.jsx("img",{src:e,alt:"Uploaded leaf",className:"w-full h-auto"}),v.jsx($r,{variant:"outline",size:"sm",className:"absolute top-4 right-4",onClick:()=>t(null),children:"Change Image"})]}),i?v.jsx("div",{className:"bg-leaf/10 border border-leaf p-6 rounded-lg",children:v.jsxs("div",{className:"flex items-start space-x-4",children:[v.jsx(Dw,{className:"w-6 h-6 text-leaf mt-1 flex-shrink-0"}),v.jsxs("div",{children:[v.jsx("h3",{className:"text-xl font-semibold mb-2",children:"Diagnosis Result"}),v.jsxs("p",{className:"text-gray-700",children:[v.jsx("strong",{children:"Disease:"})," ",i]}),v.jsxs("p",{className:"text-gray-700 mt-2",children:[v.jsx("strong",{children:"Solution:"})," ",TC(i)]})]})]})}):v.jsx("div",{className:"flex justify-center",children:v.jsx($r,{size:"lg",className:"bg-leaf hover:bg-leaf-dark",onClick:y,disabled:o,children:o?v.jsxs(v.Fragment,{children:["Analyzing...",v.jsx("div",{className:"ml-2 animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"})]}):v.jsx(v.Fragment,{children:"Analyze Image"})})})]}):v.jsxs("div",{className:`leaf-image-upload border-2 border-dashed rounded-lg p-6 text-center transition-all ${n?"border-leaf bg-leaf/5":"border-gray-300"}`,onDragOver:c,onDragLeave:f,onDrop:m,children:[v.jsx("input",{type:"file",ref:u,className:"hidden",accept:"image/*",onChange:d}),v.jsx(Yw,{className:"w-16 h-16 text-leaf mb-4 mx-auto"}),v.jsx("h3",{className:"text-xl font-semibold mb-2",children:"Upload Your Leaf Image"}),v.jsx("p",{className:"text-gray-500 mb-4",children:"Drag and drop an image here, or click the button below"}),v.jsxs($r,{type:"button",className:"bg-leaf hover:bg-leaf-dark",onClick:()=>{var x;return(x=u.current)==null?void 0:x.click()},children:[v.jsx(Uw,{className:"mr-2 h-4 w-4"})," Select Image"]})]})}),v.jsx("div",{className:"mt-10 flex justify-center",children:v.jsxs("div",{className:"flex items-center text-sm text-gray-500",children:[v.jsx(zw,{className:"w-4 h-4 mr-2"}),v.jsx("p",{children:"For demonstration purposes only. In a real app, analysis would be performed by our AI."})]})})]})})};function RC(e,t=[]){let n=[];function r(s,i){const l=w.createContext(i);l.displayName=s+"Context";const a=n.length;n=[...n,i];const u=f=>{var h;const{scope:m,children:d,...S}=f,y=((h=m==null?void 0:m[e])==null?void 0:h[a])||l,x=w.useMemo(()=>S,Object.values(S));return v.jsx(y.Provider,{value:x,children:d})};u.displayName=s+"Provider";function c(f,m){var y;const d=((y=m==null?void 0:m[e])==null?void 0:y[a])||l,S=w.useContext(d);if(S)return S;if(i!==void 0)return i;throw new Error(`\`${f}\` must be used within \`${s}\``)}return[u,c]}const o=()=>{const s=n.map(i=>w.createContext(i));return function(l){const a=(l==null?void 0:l[e])||s;return w.useMemo(()=>({[`__scope${e}`]:{...l,[e]:a}}),[l,a])}};return o.scopeName=e,[r,jC(o,...t)]}function jC(...e){const t=e[0];if(e.length===1)return t;const n=()=>{const r=e.map(o=>({useScope:o(),scopeName:o.scopeName}));return function(s){const i=r.reduce((l,{useScope:a,scopeName:u})=>{const f=a(s)[`__scope${u}`];return{...l,...f}},{});return w.useMemo(()=>({[`__scope${t.scopeName}`]:i}),[i])}};return n.scopeName=t.scopeName,n}var _C=["a","button","div","form","h2","h3","img","input","label","li","nav","ol","p","select","span","svg","ul"],wc=_C.reduce((e,t)=>{const n=_g(`Primitive.${t}`),r=w.forwardRef((o,s)=>{const{asChild:i,...l}=o,a=i?n:t;return typeof window<"u"&&(window[Symbol.for("radix-ui")]=!0),v.jsx(a,{...l,ref:s})});return r.displayName=`Primitive.${t}`,{...e,[t]:r}},{}),Ag={exports:{}},Og={};/**
 * @license React
 * use-sync-external-store-shim.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var so=w;function AC(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var OC=typeof Object.is=="function"?Object.is:AC,LC=so.useState,MC=so.useEffect,IC=so.useLayoutEffect,DC=so.useDebugValue;function zC(e,t){var n=t(),r=LC({inst:{value:n,getSnapshot:t}}),o=r[0].inst,s=r[1];return IC(function(){o.value=n,o.getSnapshot=t,Xl(o)&&s({inst:o})},[e,n,t]),MC(function(){return Xl(o)&&s({inst:o}),e(function(){Xl(o)&&s({inst:o})})},[e]),DC(n),n}function Xl(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!OC(e,n)}catch{return!0}}function FC(e,t){return t()}var $C=typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"?FC:zC;Og.useSyncExternalStore=so.useSyncExternalStore!==void 0?so.useSyncExternalStore:$C;Ag.exports=Og;var BC=Ag.exports;function UC(){return BC.useSyncExternalStore(VC,()=>!0,()=>!1)}function VC(){return()=>{}}var xc="Avatar",[WC]=RC(xc),[HC,Lg]=WC(xc),Mg=w.forwardRef((e,t)=>{const{__scopeAvatar:n,...r}=e,[o,s]=w.useState("idle");return v.jsx(HC,{scope:n,imageLoadingStatus:o,onImageLoadingStatusChange:s,children:v.jsx(wc.span,{...r,ref:t})})});Mg.displayName=xc;var Ig="AvatarImage",Dg=w.forwardRef((e,t)=>{const{__scopeAvatar:n,src:r,onLoadingStatusChange:o=()=>{},...s}=e,i=Lg(Ig,n),l=QC(r,s),a=qt(u=>{o(u),i.onImageLoadingStatusChange(u)});return kt(()=>{l!=="idle"&&a(l)},[l,a]),l==="loaded"?v.jsx(wc.img,{...s,ref:t,src:r}):null});Dg.displayName=Ig;var zg="AvatarFallback",Fg=w.forwardRef((e,t)=>{const{__scopeAvatar:n,delayMs:r,...o}=e,s=Lg(zg,n),[i,l]=w.useState(r===void 0);return w.useEffect(()=>{if(r!==void 0){const a=window.setTimeout(()=>l(!0),r);return()=>window.clearTimeout(a)}},[r]),i&&s.imageLoadingStatus!=="loaded"?v.jsx(wc.span,{...o,ref:t}):null});Fg.displayName=zg;function kf(e,t){return e?t?(e.src!==t&&(e.src=t),e.complete&&e.naturalWidth>0?"loaded":"loading"):"error":"idle"}function QC(e,{referrerPolicy:t,crossOrigin:n}){const r=UC(),o=w.useRef(null),s=r?(o.current||(o.current=new window.Image),o.current):null,[i,l]=w.useState(()=>kf(s,e));return kt(()=>{l(kf(s,e))},[s,e]),kt(()=>{const a=f=>()=>{l(f)};if(!s)return;const u=a("loaded"),c=a("error");return s.addEventListener("load",u),s.addEventListener("error",c),t&&(s.referrerPolicy=t),typeof n=="string"&&(s.crossOrigin=n),()=>{s.removeEventListener("load",u),s.removeEventListener("error",c)}},[s,n,t]),i}var $g=Mg,Bg=Dg,Ug=Fg;const Vg=w.forwardRef(({className:e,...t},n)=>v.jsx($g,{ref:n,className:Te("relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full",e),...t}));Vg.displayName=$g.displayName;const Wg=w.forwardRef(({className:e,...t},n)=>v.jsx(Bg,{ref:n,className:Te("aspect-square h-full w-full",e),...t}));Wg.displayName=Bg.displayName;const Hg=w.forwardRef(({className:e,...t},n)=>v.jsx(Ug,{ref:n,className:Te("flex h-full w-full items-center justify-center rounded-full bg-muted",e),...t}));Hg.displayName=Ug.displayName;const Qg=w.forwardRef(({className:e,...t},n)=>v.jsx("div",{ref:n,className:Te("rounded-lg border bg-card text-card-foreground shadow-sm",e),...t}));Qg.displayName="Card";const KC=w.forwardRef(({className:e,...t},n)=>v.jsx("div",{ref:n,className:Te("flex flex-col space-y-1.5 p-6",e),...t}));KC.displayName="CardHeader";const GC=w.forwardRef(({className:e,...t},n)=>v.jsx("h3",{ref:n,className:Te("text-2xl font-semibold leading-none tracking-tight",e),...t}));GC.displayName="CardTitle";const YC=w.forwardRef(({className:e,...t},n)=>v.jsx("p",{ref:n,className:Te("text-sm text-muted-foreground",e),...t}));YC.displayName="CardDescription";const Kg=w.forwardRef(({className:e,...t},n)=>v.jsx("div",{ref:n,className:Te("p-6 pt-0",e),...t}));Kg.displayName="CardContent";const XC=w.forwardRef(({className:e,...t},n)=>v.jsx("div",{ref:n,className:Te("flex items-center p-6 pt-0",e),...t}));XC.displayName="CardFooter";const qC=[{quote:"PaddyGuard saved my rice crop! The app identified early blight before it spread to my entire garden.",name:"Sarah Johnson",role:"Home Gardener",avatar:"SJ"},{quote:"As a commercial farmer, time is money. This tool helps me spot issues early and take targeted action.",name:"Michael Rodriguez",role:"Organic Farmer",avatar:"MR"},{quote:"I've tried several plant apps, but PaddyGuard is the most accurate. It correctly identified a rare fungus.",name:"Emma Chen",role:"Plant Enthusiast",avatar:"EC"}],ZC=()=>v.jsx("section",{className:"py-16 bg-gray-50 px-4",children:v.jsxs("div",{className:"max-w-7xl mx-auto",children:[v.jsxs("div",{className:"text-center mb-16",children:[v.jsx("h2",{className:"text-3xl md:text-4xl font-bold mb-4",children:"What Users Say"}),v.jsx("p",{className:"text-xl text-gray-600 max-w-3xl mx-auto",children:"Join thousands of satisfied gardeners and farmers who trust PaddyGuard for their plant care."})]}),v.jsx("div",{className:"grid md:grid-cols-3 gap-8",children:qC.map((e,t)=>v.jsx(Qg,{className:"leaf-card",children:v.jsx(Kg,{className:"pt-6",children:v.jsxs("div",{className:"flex flex-col h-full",children:[v.jsxs("blockquote",{className:"text-gray-700 mb-6 flex-grow",children:['"',e.quote,'"']}),v.jsxs("div",{className:"flex items-center",children:[v.jsxs(Vg,{className:"h-10 w-10 mr-4 border-2 border-leaf/20",children:[v.jsx(Wg,{src:"",alt:e.name}),v.jsx(Hg,{className:"bg-leaf/10 text-leaf",children:e.avatar})]}),v.jsxs("div",{children:[v.jsx("div",{className:"font-semibold",children:e.name}),v.jsx("div",{className:"text-sm text-gray-500",children:e.role})]})]})]})})},t))})]})}),JC=()=>v.jsx("footer",{className:"bg-gray-900 text-white py-12 px-4",children:v.jsxs("div",{className:"max-w-7xl mx-auto",children:[v.jsxs("div",{className:"grid grid-cols-1 md:grid-cols-4 gap-12",children:[v.jsxs("div",{className:"space-y-4",children:[v.jsxs("div",{className:"flex items-center space-x-2",children:[v.jsx(Em,{className:"w-8 h-8 text-leaf"}),v.jsx("span",{className:"text-xl font-bold",children:"PaddyGuard"})]}),v.jsx("p",{className:"text-gray-400 pr-4",children:"Empowering gardeners and farmers with AI-powered rice disease detection."}),v.jsxs("div",{className:"flex space-x-4 pt-2",children:[v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:v.jsx(Gw,{className:"w-5 h-5"})}),v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:v.jsx(Vw,{className:"w-5 h-5"})}),v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:v.jsx(Bw,{className:"w-5 h-5"})})]})]}),v.jsxs("div",{children:[v.jsx("h3",{className:"font-semibold text-lg mb-4",children:"Product"}),v.jsxs("ul",{className:"space-y-2",children:[v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Features"})}),v.jsx("li",{children:v.jsx("a",{href:"#how-it-works",className:"text-gray-400 hover:text-white transition-colors",children:"How It Works"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Pricing"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"FAQ"})})]})]}),v.jsxs("div",{children:[v.jsx("h3",{className:"font-semibold text-lg mb-4",children:"Resources"}),v.jsxs("ul",{className:"space-y-2",children:[v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Rice Care Guide"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Disease Library"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Blog"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Community"})})]})]}),v.jsxs("div",{children:[v.jsx("h3",{className:"font-semibold text-lg mb-4",children:"Company"}),v.jsxs("ul",{className:"space-y-2",children:[v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"About Us"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Contact"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Privacy Policy"})}),v.jsx("li",{children:v.jsx("a",{href:"#",className:"text-gray-400 hover:text-white transition-colors",children:"Terms of Service"})})]})]})]}),v.jsx("div",{className:"border-t border-gray-800 mt-12 pt-8 text-center text-gray-400",children:v.jsxs("p",{children:["© ",new Date().getFullYear()," PaddyGuard. All rights reserved."]})})]})}),e2=()=>v.jsxs("div",{className:"min-h-screen flex flex-col",children:[v.jsx(fC,{}),v.jsxs("main",{className:"flex-grow",children:[v.jsx(EC,{}),v.jsx(NC,{}),v.jsx(kC,{}),v.jsx(bC,{}),v.jsx(ZC,{})]}),v.jsx(JC,{})]}),t2=()=>{const e=Ng();return w.useEffect(()=>{console.error("404 Error: User attempted to access non-existent route:",e.pathname)},[e.pathname]),v.jsx("div",{className:"min-h-screen flex items-center justify-center bg-gray-100",children:v.jsxs("div",{className:"text-center",children:[v.jsx("h1",{className:"text-4xl font-bold mb-4",children:"404"}),v.jsx("p",{className:"text-xl text-gray-600 mb-4",children:"Oops! Page not found"}),v.jsx("a",{href:"/",className:"text-blue-500 hover:text-blue-700 underline",children:"Return to Home"})]})})},n2=new SE,r2=()=>v.jsx(CE,{client:n2,children:v.jsxs(HS,{children:[v.jsx(Ax,{}),v.jsx(c1,{}),v.jsx(dC,{children:v.jsxs(aC,{children:[v.jsx(ou,{path:"/",element:v.jsx(e2,{})}),v.jsx(ou,{path:"*",element:v.jsx(t2,{})})]})})]})}),Pf=document.getElementById("root");Pf&&Yh(Pf).render(v.jsx(r2,{}));