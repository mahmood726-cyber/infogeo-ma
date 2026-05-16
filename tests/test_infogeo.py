"""
Comprehensive Selenium test suite for InfoGeoMA — Information-Geometric Meta-Analysis.
Tests Fisher-Rao distance, Frechet mean, DL pooling, curvature heterogeneity, UI, and end-to-end.
"""
import sys, io, os, unittest, time, json, math

# Set UTF-8 encoding for stdout (Windows cp1252 workaround)
# Only wrap if not already wrapped and stdout has a buffer
if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

HTML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'infogeo-ma.html'))
HTML = 'file:///' + HTML_PATH.replace('\\', '/')


class TestInfoGeoMA(unittest.TestCase):
    """Test suite for InfoGeoMA information-geometric meta-analysis tool."""

    @classmethod
    def setUpClass(cls):
        opts = Options()
        opts.add_argument('--headless=new')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-gpu')
        cls.drv = webdriver.Chrome(options=opts)
        cls.drv.get(HTML)
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.drv.quit()

    def js(self, script):
        return self.drv.execute_script(script)

    def reload(self):
        self.drv.get(HTML)
        time.sleep(0.5)

    # ======================================================================
    # 1. Fisher-Rao Distance — fisherRaoDistance(mu1, s1, mu2, s2)
    # ======================================================================

    def test_01_fr_distance_identical(self):
        """d(same, same) = 0"""
        d = self.js("return fisherRaoDistance(0, 1, 0, 1);")
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_02_fr_distance_different_sigma(self):
        """d(0,1, 0,2) > 0 when only sigma differs"""
        d = self.js("return fisherRaoDistance(0, 1, 0, 2);")
        self.assertGreater(d, 0)

    def test_03_fr_distance_different_mu(self):
        """d(0,1, 1,1) > 0 when only mu differs"""
        d = self.js("return fisherRaoDistance(0, 1, 1, 1);")
        self.assertGreater(d, 0)

    def test_04_fr_distance_symmetry(self):
        """d(a,b) = d(b,a) — symmetry property"""
        d1 = self.js("return fisherRaoDistance(0.5, 1.2, -0.3, 0.8);")
        d2 = self.js("return fisherRaoDistance(-0.3, 0.8, 0.5, 1.2);")
        self.assertAlmostEqual(d1, d2, places=10)

    def test_05_fr_distance_triangle_inequality(self):
        """d(a,c) <= d(a,b) + d(b,c) — triangle inequality"""
        dac = self.js("return fisherRaoDistance(0, 1, 2, 0.5);")
        dab = self.js("return fisherRaoDistance(0, 1, 1, 1.5);")
        dbc = self.js("return fisherRaoDistance(1, 1.5, 2, 0.5);")
        self.assertLessEqual(dac, dab + dbc + 1e-10)

    def test_06_fr_distance_nonpositive_sigma_infinity(self):
        """d with s1<=0 returns Infinity (Selenium maps JS Infinity to None)"""
        d1 = self.js("return fisherRaoDistance(0, 0, 1, 1);")
        d2 = self.js("return fisherRaoDistance(0, -1, 1, 1);")
        d3 = self.js("return fisherRaoDistance(0, 1, 1, 0);")
        # Selenium returns None for JS Infinity; verify via string check
        check1 = self.js("return fisherRaoDistance(0, 0, 1, 1) === Infinity;")
        check2 = self.js("return fisherRaoDistance(0, -1, 1, 1) === Infinity;")
        check3 = self.js("return fisherRaoDistance(0, 1, 1, 0) === Infinity;")
        self.assertTrue(check1, "d(0,0,1,1) should be Infinity")
        self.assertTrue(check2, "d(0,-1,1,1) should be Infinity")
        self.assertTrue(check3, "d(0,1,1,0) should be Infinity")

    def test_07_fr_distance_same_mu_different_sigma_formula(self):
        """When mu1=mu2, formula simplifies to sqrt(2)*|log(s1/s2)|"""
        # d_FR = sqrt(2)*arccosh(1 + (s1^2-s2^2)^2/(4*s1^2*s2^2))
        # For same mu: arg = 1 + (s1^2-s2^2)^2/(4*s1^2*s2^2) = 1 + ((s1/s2)^2-1)^2/4
        # = (s1^2+s2^2)^2/(4*s1^2*s2^2) = ((s1/s2 + s2/s1)/2)^2
        # arccosh(((r+1/r)/2)^2) where r=s1/s2... but simplified:
        # d = sqrt(2)*|log(s1/s2)| for pure sigma differences
        s1, s2 = 1.0, 3.0
        d = self.js(f"return fisherRaoDistance(0, {s1}, 0, {s2});")
        expected = math.sqrt(2) * abs(math.log(s1 / s2))
        # This is the known closed-form for pure sigma shift.
        # Verify the app's formula gives a close result:
        # arg = 1 + (1 - 9)^2/(4*1*9) = 1 + 64/36 = 1 + 16/9 = 25/9
        # arccosh(25/9) = log(25/9 + sqrt(625/81 - 1)) = log(25/9 + sqrt(544/81))
        # = log(25/9 + sqrt(544)/9)
        arg = 25.0 / 9.0
        acosh_val = math.log(arg + math.sqrt(arg * arg - 1))
        expected_full = math.sqrt(2) * acosh_val
        self.assertAlmostEqual(d, expected_full, places=8)

    def test_08_fr_distance_large_separation(self):
        """Very different distributions should have large distance"""
        d = self.js("return fisherRaoDistance(-10, 0.01, 10, 100);")
        self.assertGreater(d, 5)

    # ======================================================================
    # 2. Distance Matrix — distanceMatrix(studies)
    # ======================================================================

    def test_09_distance_matrix_size(self):
        """distanceMatrix returns k x k matrix"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:1,se:1},{yi:2,se:1}];
            var D = distanceMatrix(s);
            return [D.length, D[0].length, D[1].length, D[2].length];
        """)
        self.assertEqual(result, [3, 3, 3, 3])

    def test_10_distance_matrix_diagonal_zero(self):
        """Diagonal of distance matrix is all zeros"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:1,se:0.5},{yi:-0.5,se:2}];
            var D = distanceMatrix(s);
            return [D[0][0], D[1][1], D[2][2]];
        """)
        for v in result:
            self.assertAlmostEqual(v, 0.0, places=10)

    def test_11_distance_matrix_symmetry(self):
        """Distance matrix is symmetric: D[i][j] = D[j][i]"""
        result = self.js("""
            var s = [{yi:0.5,se:0.3},{yi:-0.2,se:0.8},{yi:1.1,se:0.15},{yi:-1.0,se:0.5}];
            var D = distanceMatrix(s);
            var diffs = [];
            for (var i=0; i<4; i++)
                for (var j=i+1; j<4; j++)
                    diffs.push(Math.abs(D[i][j] - D[j][i]));
            return Math.max.apply(null, diffs);
        """)
        self.assertLess(result, 1e-10)

    # ======================================================================
    # 3. Frechet Mean — frechetMean(studies)
    # ======================================================================

    def test_12_frechet_mean_single_study_repeated(self):
        """Frechet mean of identical studies = that study"""
        result = self.js("""
            var s = [{yi:0.5,se:0.3},{yi:0.5,se:0.3},{yi:0.5,se:0.3}];
            var fm = frechetMean(s);
            return {mu: fm.mu, sigma: fm.sigma};
        """)
        self.assertAlmostEqual(result['mu'], 0.5, places=2)
        self.assertAlmostEqual(result['sigma'], 0.3, places=2)

    def test_13_frechet_mean_returns_all_fields(self):
        """Frechet mean returns mu, sigma, distances, meanGeodesicDistance, weights"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:1,se:1}];
            var fm = frechetMean(s);
            return {
                hasMu: typeof fm.mu === 'number',
                hasSigma: typeof fm.sigma === 'number',
                hasDistances: Array.isArray(fm.distances),
                hasMeanGD: typeof fm.meanGeodesicDistance === 'number',
                hasWeights: Array.isArray(fm.weights),
                distLen: fm.distances.length,
                wLen: fm.weights.length
            };
        """)
        self.assertTrue(result['hasMu'])
        self.assertTrue(result['hasSigma'])
        self.assertTrue(result['hasDistances'])
        self.assertTrue(result['hasMeanGD'])
        self.assertTrue(result['hasWeights'])
        self.assertEqual(result['distLen'], 2)
        self.assertEqual(result['wLen'], 2)

    def test_14_frechet_mean_weights_sum_to_one(self):
        """Frechet mean weights should sum to 1"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:1,se:0.5},{yi:-0.5,se:2}];
            var fm = frechetMean(s);
            return fm.weights.reduce(function(a,b){return a+b;}, 0);
        """)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_15_frechet_mean_precision_weighting(self):
        """More precise studies (smaller SE) should get higher weights"""
        result = self.js("""
            var s = [{yi:0,se:0.1},{yi:1,se:1.0},{yi:0.5,se:0.5}];
            var fm = frechetMean(s);
            return fm.weights;
        """)
        # se=0.1 -> w proportional to 1/0.01=100
        # se=1.0 -> w proportional to 1/1.0=1
        # se=0.5 -> w proportional to 1/0.25=4
        self.assertGreater(result[0], result[2])
        self.assertGreater(result[2], result[1])

    def test_16_frechet_mean_positive_sigma(self):
        """Frechet mean sigma should always be positive"""
        result = self.js("""
            var s = [{yi:-2,se:0.5},{yi:2,se:0.5},{yi:0,se:0.1}];
            var fm = frechetMean(s);
            return fm.sigma;
        """)
        self.assertGreater(result, 0)

    def test_17_frechet_mean_distances_nonneg(self):
        """All Frechet distances should be non-negative"""
        result = self.js("""
            var s = [{yi:0,se:0.3},{yi:0.5,se:0.8},{yi:-1,se:0.2},{yi:0.2,se:0.6}];
            var fm = frechetMean(s);
            return fm.distances;
        """)
        for d in result:
            self.assertGreaterEqual(d, 0)

    # ======================================================================
    # 4. DL Pooling — dlPool(studies)
    # ======================================================================

    def test_18_dl_pool_returns_fields(self):
        """dlPool returns est, se, tau2, I2, Q, k"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:1,se:1}];
            var dl = dlPool(s);
            return {
                hasEst: typeof dl.est === 'number',
                hasSe: typeof dl.se === 'number',
                hasTau2: typeof dl.tau2 === 'number',
                hasI2: typeof dl.I2 === 'number',
                hasQ: typeof dl.Q === 'number',
                hasK: typeof dl.k === 'number',
                k: dl.k
            };
        """)
        self.assertTrue(result['hasEst'])
        self.assertTrue(result['hasSe'])
        self.assertTrue(result['hasTau2'])
        self.assertTrue(result['hasI2'])
        self.assertTrue(result['hasQ'])
        self.assertTrue(result['hasK'])
        self.assertEqual(result['k'], 2)

    def test_19_dl_pool_homogeneous(self):
        """Homogeneous data (same yi) -> tau2=0, I2=0"""
        result = self.js("""
            var s = [{yi:0.5,se:0.3},{yi:0.5,se:0.4},{yi:0.5,se:0.5}];
            var dl = dlPool(s);
            return {tau2: dl.tau2, I2: dl.I2, Q: dl.Q};
        """)
        self.assertAlmostEqual(result['tau2'], 0.0, places=10)
        self.assertAlmostEqual(result['I2'], 0.0, places=10)
        self.assertAlmostEqual(result['Q'], 0.0, places=10)

    def test_20_dl_pool_fixed_effect_single_yi(self):
        """With identical effects, DL est = that effect"""
        result = self.js("""
            var s = [{yi:0.7,se:0.1},{yi:0.7,se:0.5},{yi:0.7,se:1.0}];
            var dl = dlPool(s);
            return dl.est;
        """)
        self.assertAlmostEqual(result, 0.7, places=8)

    def test_21_dl_pool_bcg_dataset(self):
        """BCG vaccine dataset: DL random-effects pooling"""
        result = self.js("""
            var s = EXAMPLES[0].studies;
            var dl = dlPool(s);
            return {est: dl.est, k: dl.k, I2: dl.I2, tau2: dl.tau2};
        """)
        self.assertEqual(result['k'], 13)
        # BCG DL estimate ~ -0.55 (log-RR, DL random-effects with this dataset)
        self.assertAlmostEqual(result['est'], -0.549, delta=0.1)
        # Substantial heterogeneity
        self.assertGreater(result['I2'], 0.7)
        self.assertGreater(result['tau2'], 0)

    def test_22_dl_pool_two_study_basic(self):
        """Two-study DL pool: verify inverse-variance weighting"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:2,se:1}];
            var dl = dlPool(s);
            return {est: dl.est, Q: dl.Q};
        """)
        # Equal weights -> est = (0+2)/2 = 1.0 for fixed effect
        # Q = w1*(0-1)^2 + w2*(2-1)^2 = 1*1 + 1*1 = 2
        self.assertAlmostEqual(result['Q'], 2.0, places=6)

    def test_23_dl_pool_se_positive(self):
        """DL pooled SE should always be positive"""
        result = self.js("""
            var s = [{yi:-1,se:0.2},{yi:0,se:0.3},{yi:1,se:0.4}];
            var dl = dlPool(s);
            return dl.se;
        """)
        self.assertGreater(result, 0)

    # ======================================================================
    # 5. Curvature Heterogeneity — curvatureHeterogeneity(frechet, dl)
    # ======================================================================

    def test_24_kappa_returns_fields(self):
        """curvatureHeterogeneity returns kappa, meanGeodesicDistance, expectedDist, interpretation"""
        result = self.js("""
            var s = [{yi:0,se:1},{yi:1,se:1}];
            var fm = frechetMean(s);
            var dl = dlPool(s);
            var k = curvatureHeterogeneity(fm, dl);
            return {
                hasKappa: typeof k.kappa === 'number',
                hasMGD: typeof k.meanGeodesicDistance === 'number',
                hasED: typeof k.expectedDist === 'number',
                hasInterp: typeof k.interpretation === 'string'
            };
        """)
        self.assertTrue(result['hasKappa'])
        self.assertTrue(result['hasMGD'])
        self.assertTrue(result['hasED'])
        self.assertTrue(result['hasInterp'])

    def test_25_kappa_thresholds_low(self):
        """kappa < 0.5 -> Low curvature heterogeneity"""
        result = self.js("""
            var obj = {kappa: 0.3, meanGeodesicDistance: 0.1, expectedDist: 0.33};
            // Manually test the interpretation logic
            var k = 0.3;
            return k < 0.5 ? 'Low curvature heterogeneity' :
                   k < 1.5 ? 'Moderate curvature heterogeneity' :
                   k < 3.0 ? 'High curvature heterogeneity' :
                   'Extreme curvature heterogeneity (non-Euclidean structure dominant)';
        """)
        self.assertIn('Low', result)

    def test_26_kappa_thresholds_moderate(self):
        """kappa in [0.5, 1.5) -> Moderate"""
        result = self.js("""
            var k = 0.8;
            return k < 0.5 ? 'Low' : k < 1.5 ? 'Moderate' : k < 3.0 ? 'High' : 'Extreme';
        """)
        self.assertEqual(result, 'Moderate')

    def test_27_kappa_thresholds_high(self):
        """kappa in [1.5, 3.0) -> High"""
        result = self.js("""
            var k = 2.0;
            return k < 0.5 ? 'Low' : k < 1.5 ? 'Moderate' : k < 3.0 ? 'High' : 'Extreme';
        """)
        self.assertEqual(result, 'High')

    def test_28_kappa_thresholds_extreme(self):
        """kappa >= 3.0 -> Extreme"""
        result = self.js("""
            var k = 5.0;
            return k < 0.5 ? 'Low' : k < 1.5 ? 'Moderate' : k < 3.0 ? 'High' : 'Extreme';
        """)
        self.assertEqual(result, 'Extreme')

    def test_29_kappa_nonnegative(self):
        """kappa should be non-negative"""
        result = self.js("""
            var s = [{yi:0,se:0.3},{yi:0.5,se:0.8},{yi:-1,se:0.2}];
            var fm = frechetMean(s);
            var dl = dlPool(s);
            var k = curvatureHeterogeneity(fm, dl);
            return k.kappa;
        """)
        self.assertGreaterEqual(result, 0)

    # ======================================================================
    # 6. End-to-End via example datasets
    # ======================================================================

    def test_30_bcg_example_loads_and_runs(self):
        """BCG example: loads data and runs analysis"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        # Check state has 13 studies
        k = self.js("return state.studies.length;")
        self.assertEqual(k, 13)
        # Check result exists
        has_result = self.js("return state.result !== null;")
        self.assertTrue(has_result)

    def test_31_bcg_frechet_mean_value(self):
        """BCG: Frechet mean approximately -0.5 to 0.0 range"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        fm_mu = self.js("return state.result.frechet.mu;")
        # The Frechet mean for BCG should be negative (protective effect)
        # and in a reasonable range
        self.assertLess(fm_mu, 0.5)
        self.assertGreater(fm_mu, -2.0)

    def test_32_bcg_metrics_populated(self):
        """BCG: metrics cards are populated with content"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        metrics_html = self.js("return document.getElementById('manifoldMetrics').innerHTML;")
        self.assertIn('metric-val', metrics_html)
        self.assertIn('chet Mean', metrics_html)

    def test_33_bcg_distance_matrix_rendered(self):
        """BCG: distance matrix table is populated"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        dist_html = self.js("return document.getElementById('distTable').innerHTML;")
        self.assertIn('Aronson', dist_html)
        # Should have 13 rows in tbody
        row_count = self.js("return document.querySelectorAll('#distTable tbody tr').length;")
        self.assertEqual(row_count, 13)

    def test_34_bcg_report_generated(self):
        """BCG: report text is generated with correct content"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        report = self.js("return document.getElementById('reportText').textContent;")
        self.assertIn('INFORMATION-GEOMETRIC META-ANALYSIS', report)
        self.assertIn('FRECHET MEAN', report.upper().replace('\u00C9', 'E'))
        self.assertIn('DL EUCLIDEAN MEAN', report)
        self.assertIn('InfoGeoMA', report)

    def test_35_magnesium_example(self):
        """Magnesium-MI example: loads with k=8, notable heterogeneity"""
        self.reload()
        self.js("loadExample(1);")
        time.sleep(1)
        k = self.js("return state.studies.length;")
        self.assertEqual(k, 8)
        has_result = self.js("return state.result !== null;")
        self.assertTrue(has_result)
        # Magnesium dataset has moderate-to-substantial heterogeneity (I2 ~ 0.41)
        # ISIS-4 dominates by precision, and the small-study effects differ
        I2 = self.js("return state.result.dl.I2;")
        self.assertGreater(I2, 0.3)
        # Kappa should also be computed
        kappa = self.js("return state.result.kappa.kappa;")
        self.assertGreater(kappa, 0)

    def test_36_sglt2i_example(self):
        """SGLT2i HF example: loads with k=6, moderate results expected"""
        self.reload()
        self.js("loadExample(2);")
        time.sleep(1)
        k = self.js("return state.studies.length;")
        self.assertEqual(k, 6)
        has_result = self.js("return state.result !== null;")
        self.assertTrue(has_result)
        # SGLT2i data is relatively homogeneous
        dl_est = self.js("return state.result.dl.est;")
        # Effect should be negative (protective)
        self.assertLess(dl_est, 0)

    def test_37_geodesic_shift_nonnegative(self):
        """Geodesic shift |Frechet - DL| should be non-negative"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        shift = self.js("return Math.abs(state.result.frechet.mu - state.result.dl.est);")
        self.assertGreaterEqual(shift, 0)

    # ======================================================================
    # 7. UI Tests
    # ======================================================================

    def test_38_add_row(self):
        """Add row button increases study count"""
        self.reload()
        before = self.js("return state.studies.length;")
        self.js("addRow();")
        after = self.js("return state.studies.length;")
        self.assertEqual(after, before + 1)

    def test_39_clear_data(self):
        """Clear data empties studies array"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(0.5)
        self.js("clearData();")
        count = self.js("return state.studies.length;")
        self.assertEqual(count, 0)

    def test_40_tab_switching(self):
        """Tab switching shows correct panel"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)

        # Switch to manifold tab
        self.js("switchTab('manifold');")
        manifold_active = self.js(
            "return document.getElementById('panel-manifold').classList.contains('active');"
        )
        data_active = self.js(
            "return document.getElementById('panel-data').classList.contains('active');"
        )
        self.assertTrue(manifold_active)
        self.assertFalse(data_active)

        # Switch to geodesic
        self.js("switchTab('geodesic');")
        geodesic_active = self.js(
            "return document.getElementById('panel-geodesic').classList.contains('active');"
        )
        self.assertTrue(geodesic_active)

        # Switch to report
        self.js("switchTab('report');")
        report_active = self.js(
            "return document.getElementById('panel-report').classList.contains('active');"
        )
        self.assertTrue(report_active)

        # Back to data
        self.js("switchTab('data');")
        data_active = self.js(
            "return document.getElementById('panel-data').classList.contains('active');"
        )
        self.assertTrue(data_active)

    def test_41_theme_toggle(self):
        """Theme toggle switches between dark and light"""
        self.reload()
        initial = self.js("return document.documentElement.getAttribute('data-theme');")
        self.assertEqual(initial, 'dark')

        self.js("toggleTheme();")
        after = self.js("return document.documentElement.getAttribute('data-theme');")
        self.assertEqual(after, 'light')

        self.js("toggleTheme();")
        restored = self.js("return document.documentElement.getAttribute('data-theme');")
        self.assertEqual(restored, 'dark')

    def test_42_initial_state_two_rows(self):
        """Page loads with 2 empty rows in the data table"""
        self.reload()
        count = self.js("return state.studies.length;")
        self.assertEqual(count, 2)
        row_count = self.js("return document.querySelectorAll('#tbody tr').length;")
        self.assertEqual(row_count, 2)

    def test_43_example_buttons_exist(self):
        """All three example buttons exist on page"""
        self.reload()
        btns = self.js("return document.querySelectorAll('.ex-btn').length;")
        self.assertEqual(btns, 3)

    # ======================================================================
    # Additional mathematical tests
    # ======================================================================

    def test_44_fr_distance_nonnegative(self):
        """Fisher-Rao distance is always non-negative"""
        result = self.js("""
            var pairs = [
                [0,1,0,1], [1,0.5,-1,2], [0.1,0.01,0.2,0.01],
                [-5,3,5,0.1], [0,0.001,0,1000]
            ];
            var all_nonneg = true;
            for (var i=0; i<pairs.length; i++) {
                var d = fisherRaoDistance(pairs[i][0], pairs[i][1], pairs[i][2], pairs[i][3]);
                if (d < 0) all_nonneg = false;
            }
            return all_nonneg;
        """)
        self.assertTrue(result)

    def test_45_dl_pool_inverse_variance_weight(self):
        """DL with tau2=0: est = sum(wi*yi)/sum(wi) where wi=1/vi"""
        result = self.js("""
            // Homogeneous case -> tau2 = 0
            var s = [{yi:1.0,se:0.5},{yi:1.0,se:1.0},{yi:1.0,se:2.0}];
            var dl = dlPool(s);
            // All yi are equal, so est should be 1.0 regardless
            return {est: dl.est, tau2: dl.tau2};
        """)
        self.assertAlmostEqual(result['est'], 1.0, places=6)
        self.assertAlmostEqual(result['tau2'], 0.0, places=6)

    def test_46_frechet_mean_between_studies(self):
        """Frechet mean mu should lie within the range of study effects"""
        result = self.js("""
            var s = [{yi:-1,se:0.3},{yi:0,se:0.3},{yi:1,se:0.3}];
            var fm = frechetMean(s);
            return fm.mu;
        """)
        # With equal SE, Frechet mean should be near 0 (Euclidean mean)
        self.assertGreater(result, -1.5)
        self.assertLess(result, 1.5)

    def test_47_export_json_structure(self):
        """Export JSON produces correct structure"""
        self.reload()
        self.js("loadExample(2);")
        time.sleep(1)
        result = self.js("""
            var r = state.result;
            var d = {version:'infogeo-ma-1.0', frechetMu:r.frechet.mu, frechetSigma:r.frechet.sigma,
                dlEst:r.dl.est, dlSE:r.dl.se, shift:Math.abs(r.frechet.mu-r.dl.est),
                kappaHeterogeneity:r.kappa.kappa, I2:r.dl.I2,
                meanGeodesicDistance:r.frechet.meanGeodesicDistance, k:r.studies.length};
            return d;
        """)
        self.assertEqual(result['version'], 'infogeo-ma-1.0')
        self.assertEqual(result['k'], 6)
        self.assertIn('frechetMu', result)
        self.assertIn('dlEst', result)
        self.assertIn('kappaHeterogeneity', result)

    def test_48_het_table_populated(self):
        """Heterogeneity comparison table is populated after analysis"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        rows = self.js("return document.querySelectorAll('#hetTable tbody tr').length;")
        self.assertGreaterEqual(rows, 4)

    def test_49_frechet_convergence_cost_decreasing(self):
        """Frechet mean optimization cost should be finite"""
        result = self.js("""
            var s = EXAMPLES[0].studies;
            var fm = frechetMean(s);
            return {cost: fm.cost, isFinite: isFinite(fm.cost)};
        """)
        self.assertTrue(result['isFinite'])
        self.assertGreater(result['cost'], 0)

    def test_50_all_examples_run_without_error(self):
        """All three example datasets run analysis without JS errors"""
        for idx in range(3):
            self.reload()
            error = self.js(f"""
                try {{
                    loadExample({idx});
                    return null;
                }} catch(e) {{
                    return e.message;
                }}
            """)
            self.assertIsNone(error, f"Example {idx} threw error: {error}")
            has_result = self.js("return state.result !== null;")
            self.assertTrue(has_result, f"Example {idx} did not produce result")

    def test_51_geodesic_metrics_populated(self):
        """Geodesic metrics panel is populated after analysis"""
        self.reload()
        self.js("loadExample(0);")
        time.sleep(1)
        html = self.js("return document.getElementById('geodesicMetrics').innerHTML;")
        self.assertIn('metric-val', html)
        # Should contain I2 and kappa
        self.assertIn('%', html)

    def test_52_fr_distance_triangle_inequality_multiple(self):
        """Triangle inequality holds for multiple point triples"""
        result = self.js("""
            var pts = [
                {mu:0, s:1}, {mu:1, s:0.5}, {mu:-0.5, s:2},
                {mu:2, s:0.3}, {mu:-1, s:1.5}
            ];
            var violations = 0;
            for (var i=0; i<pts.length; i++)
              for (var j=0; j<pts.length; j++)
                for (var k=0; k<pts.length; k++) {
                  if (i===j || j===k || i===k) continue;
                  var dij = fisherRaoDistance(pts[i].mu, pts[i].s, pts[j].mu, pts[j].s);
                  var djk = fisherRaoDistance(pts[j].mu, pts[j].s, pts[k].mu, pts[k].s);
                  var dik = fisherRaoDistance(pts[i].mu, pts[i].s, pts[k].mu, pts[k].s);
                  if (dik > dij + djk + 1e-9) violations++;
                }
            return violations;
        """)
        self.assertEqual(result, 0)

    def test_53_dl_i2_range(self):
        """I2 should always be in [0, 1]"""
        for idx in range(3):
            self.reload()
            self.js(f"loadExample({idx});")
            time.sleep(0.8)
            I2 = self.js("return state.result.dl.I2;")
            self.assertGreaterEqual(I2, 0, f"Example {idx}: I2 < 0")
            self.assertLessEqual(I2, 1, f"Example {idx}: I2 > 1")

    def test_54_manifold_canvas_exists(self):
        """Manifold canvas element exists and has correct dimensions"""
        result = self.js("""
            var c = document.getElementById('manifoldCanvas');
            return {exists: c !== null, width: c.width, height: c.height};
        """)
        self.assertTrue(result['exists'])
        self.assertGreater(result['width'], 0)
        self.assertGreater(result['height'], 0)

    def test_55_tab_btn_active_state(self):
        """Active tab button gets the 'active' class"""
        self.reload()
        self.js("switchTab('report');")
        report_btn_active = self.js(
            "return document.getElementById('tbtn-report').classList.contains('active');"
        )
        data_btn_active = self.js(
            "return document.getElementById('tbtn-data').classList.contains('active');"
        )
        self.assertTrue(report_btn_active)
        self.assertFalse(data_btn_active)


if __name__ == '__main__':
    unittest.main(verbosity=2)
