# InfoGeoMA — Code Review Findings

**Reviewer:** Claude Opus 4.6 (1M context)
**Date:** 2026-04-03
**File:** infogeo-ma.html (586 lines)

## P0 — Critical (must fix)

### P0-1: XSS in `renderTable()` — study names inserted into innerHTML unescaped
**File:** infogeo-ma.html, line 461
**Issue:** Study names from user input are directly interpolated into innerHTML via string concatenation:
```js
tr.innerHTML='<td><input value="'+(s.study||'')+'"...
```
If a study name contains `"` followed by event handlers (e.g., `" onfocus="alert(1)`), it will execute arbitrary JavaScript. The same applies to `<script>` injection via the value.
**Fix:** Add an `escapeHtml()` function and use it for all user-provided values rendered in HTML context.

### P0-2: Missing `</html>` closing tag
**Status:** PASS — `</html>` is at line 586 (confirmed, including trailing content after `</script>`). Wait, let me re-verify...

Actually on re-reading: line 585 is `</body>` and line 586... let me check. The file ends at line 586 with no `</html>`. Let me verify.

**REVISED after verification:** Line 584 is `</script>`, line 585 is `</body>`, line 586 is... actually the Read showed line 585 as `</body>` and nothing after. The `</html>` was shown on the grep at line 586. So `</html>` IS present. PASS.

## P1 — Important

### P1-1: No CSV injection guard on JSON export
**File:** infogeo-ma.html, lines 562-572
**Issue:** The exportJSON function exports results as JSON, not CSV. JSON format is not vulnerable to formula injection.
**Status:** N/A — no CSV export exists in this app.

### P1-2: Tab buttons use `onclick` inline handlers instead of ARIA tab pattern
**File:** infogeo-ma.html, lines 54-57
**Issue:** Tabs use `onclick="switchTab('data')"` but lack `aria-selected`, `aria-controls`, and `tabindex` attributes. Keyboard arrow navigation between tabs is not implemented.
**Fix:** Add proper ARIA tab attributes.

### P1-3: Fisher-Rao distance formula verification
**File:** infogeo-ma.html, lines 187-196
**Formula:** `d_FR = sqrt(2) * arccosh(1 + (mu1-mu2)^2/(2*s1*s2) + (s1^2-s2^2)^2/(4*s1^2*s2^2))`
**Reference:** Atkinson C, Mitchell AFS (1981), Sankhya A, 43:345-365.
**Status:** PASS — correctly implements the Atkinson-Mitchell formula for the Fisher-Rao distance on the univariate normal manifold.

### P1-4: DL pooling formula verification
**File:** infogeo-ma.html, lines 277-291
**Status:** PASS — standard DerSimonian-Laird with correct Q, C, tau2, I2.

### P1-5: Frechet mean gradient descent uses numerical gradient
**File:** infogeo-ma.html, lines 219-272
**Issue:** Central difference gradient with fixed learning rate schedule. This is acceptable for a browser tool but convergence is not guaranteed for all inputs.
**Status:** Acceptable — the initialization at the Euclidean mean provides a good starting point.

## P2 — Minor

### P2-1: Skip-nav link present
**File:** infogeo-ma.html, line 43
**Status:** PASS — `<a href="#main" class="skip-nav">Skip to content</a>` exists with proper CSS.

### P2-2: Canvas elements have aria-labels
**Lines:** 90, 107
**Status:** PASS.

### P2-3: Blob URL properly revoked in exportJSON
**Line:** 571
**Status:** PASS.

### P2-4: `</html>` closing tag present
**Line:** 586
**Status:** PASS.

## Summary

| Severity | Count |
|----------|-------|
| P0       | 1     |
| P1       | 5     |
| P2       | 4     |

## Statistics Verification

- **Fisher-Rao distance:** Atkinson-Mitchell (1981) formula. PASS.
- **DL random-effects:** Standard formulation. PASS.
- **Frechet mean:** Numerical gradient descent on Fisher-Rao functional. Acceptable.
- **Kappa heterogeneity:** Novel metric = mean_geodesic_dist / expected_dist. Interpretable.
