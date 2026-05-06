"""
Generates a technical PDF document for the F1 Race Winner Prediction project.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


# ── Colour palette ────────────────────────────────────────────────────────────
RED    = colors.HexColor("#E8002D")   # F1 red
DARK   = colors.HexColor("#15151E")   # near-black
MID    = colors.HexColor("#38383F")   # dark grey
LIGHT  = colors.HexColor("#F5F5F5")   # table row fill
WHITE  = colors.white
ACCENT = colors.HexColor("#FF8000")   # amber accent


def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cell"] = ParagraphStyle(
        "cell",
        fontName="Helvetica",
        fontSize=9,
        textColor=DARK,
        leading=13,
        spaceAfter=0,
        spaceBefore=0,
    )
    styles["cell_hdr"] = ParagraphStyle(
        "cell_hdr",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=WHITE,
        leading=13,
        spaceAfter=0,
        spaceBefore=0,
    )
    styles["cell_code"] = ParagraphStyle(
        "cell_code",
        fontName="Courier",
        fontSize=8,
        textColor=DARK,
        leading=12,
        spaceAfter=0,
        spaceBefore=0,
    )

    styles["title"] = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=28,
        textColor=WHITE,
        leading=34,
        spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        fontName="Helvetica",
        fontSize=13,
        textColor=colors.HexColor("#CCCCCC"),
        leading=18,
        spaceAfter=0,
    )
    styles["h1"] = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=DARK,
        leading=22,
        spaceBefore=20,
        spaceAfter=6,
        borderPad=0,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=MID,
        leading=16,
        spaceBefore=14,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        textColor=DARK,
        leading=15,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=10,
        textColor=DARK,
        leading=14,
        spaceAfter=3,
        leftIndent=16,
        bulletIndent=4,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=9,
        textColor=colors.HexColor("#1A1A2E"),
        leading=13,
        spaceAfter=2,
        leftIndent=12,
        backColor=colors.HexColor("#F0F0F0"),
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=MID,
        leading=12,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    styles["footer"] = ParagraphStyle(
        "footer",
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        leading=10,
    )

    return styles


def hr(styles):
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=6, spaceBefore=6)


def section_rule(styles):
    return HRFlowable(width="100%", thickness=2, color=RED, spaceAfter=8, spaceBefore=4)


def _cell(text, style, is_code=False):
    """Wrap a string in a Paragraph so ReportLab wraps it within the cell."""
    if is_code:
        return Paragraph(str(text), style["cell_code"])
    return Paragraph(str(text), style["cell"])


def _hdr(text, style):
    return Paragraph(str(text), style["cell_hdr"])


def make_table(data, col_widths, styles, header_color=DARK):
    """
    Build a Table from a list-of-lists of strings, wrapping every cell in a
    Paragraph so that long text reflows instead of truncating.
    The first row is treated as the header.
    """
    wrapped = []
    for r_idx, row in enumerate(data):
        new_row = []
        for cell in row:
            if r_idx == 0:
                new_row.append(_hdr(cell, styles))
            else:
                new_row.append(_cell(cell, styles))
        wrapped.append(new_row)

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def table_style(header_color=DARK):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ])


def cover_page(styles, width, height):
    """Returns the cover as a list of flowables using a coloured background table."""
    # Title block — rendered as a dark banner using a 1-row table
    cover_data = [[
        Paragraph("F1 Race Winner Prediction", styles["title"]),
    ]]
    banner = Table(cover_data, colWidths=[width - 4 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 32),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 24),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 24),
    ]))

    subtitle_data = [[
        Paragraph("Technical Design Document", styles["subtitle"]),
    ]]
    subtitle_banner = Table(subtitle_data, colWidths=[width - 4 * cm])
    subtitle_banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 32),
        ("LEFTPADDING",   (0, 0), (-1, -1), 24),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 24),
    ]))

    red_rule = Table([[""]], colWidths=[width - 4 * cm], rowHeights=[6])
    red_rule.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED),
    ]))

    meta = ParagraphStyle(
        "meta", fontName="Helvetica", fontSize=10,
        textColor=MID, leading=16, spaceAfter=4,
    )

    elements = [
        banner,
        subtitle_banner,
        red_rule,
        Spacer(1, 1.4 * cm),
        Paragraph("Version 1.0  ·  May 2026", meta),
        Paragraph("Logistic Regression + XGBoost Ensemble", meta),
        Paragraph("Training data: 2010 – 2026 (through Japanese GP, Round 3)", meta),
        Spacer(1, 0.6 * cm),
        hr(styles),
        Spacer(1, 0.4 * cm),
    ]
    return elements


def toc(styles):
    h1 = styles["h1"]
    items = [
        Paragraph("Table of Contents", h1),
        Spacer(1, 0.3 * cm),
    ]
    sections = [
        ("1", "Project Overview"),
        ("2", "System Architecture"),
        ("3", "Data Sources"),
        ("4", "Feature Engineering"),
        ("5", "Models"),
        ("5.1", "Logistic Regression"),
        ("5.2", "XGBoost"),
        ("6", "Ensemble & Post-Processing"),
        ("7", "Prediction Workflow"),
        ("8", "Training Workflow"),
        ("9", "Limitations & Future Work"),
    ]
    toc_data = [["§", "Section"]]
    for num, title in sections:
        toc_data.append([num, title])

    t = make_table(toc_data, [1.5 * cm, 15.7 * cm], styles)
    items.append(t)
    items.append(Spacer(1, 0.5 * cm))
    return items


def section1(styles):
    h1, h2, body, bullet = styles["h1"], styles["h2"], styles["body"], styles["bullet"]
    return [
        Paragraph("1  Project Overview", h1),
        section_rule(styles),
        Paragraph(
            "This project builds a machine-learning pipeline that predicts the winner of a "
            "Formula 1 Grand Prix. Given an upcoming race, it collects qualifying grid positions, "
            "historical driver and constructor statistics, circuit-specific records, weather "
            "forecasts, and live news sentiment, then outputs a ranked probability distribution "
            "across all drivers.",
            body,
        ),
        Paragraph(
            "The system is deliberately practical: it can be retrained in minutes on a laptop, "
            "generates a prediction as soon as qualifying results are published (usually Saturday "
            "evening), and degrades gracefully to pre-qualifying mode if grid data is not yet "
            "available.",
            body,
        ),
        Paragraph("Design goals:", h2),
        Paragraph("• <b>Reproducibility</b> — all API responses cached locally; training is deterministic.", bullet),
        Paragraph("• <b>Interpretability</b> — logistic regression coefficients expose feature direction.", bullet),
        Paragraph("• <b>Timeliness</b> — live news sentiment applied as a post-hoc multiplier so the model itself never needs retraining for sentiment.", bullet),
        Paragraph("• <b>Graceful degradation</b> — missing data (no qualifying yet, no weather forecast) handled with sensible defaults rather than hard errors.", bullet),
        Spacer(1, 0.3 * cm),
    ]


def section2(styles, page_width):
    h1, h2, body, bullet, code = styles["h1"], styles["h2"], styles["body"], styles["bullet"], styles["code"]
    arch_data = [
        ["Layer", "Component", "Purpose"],
        ["Data ingestion",  "data_fetcher.py",        "Jolpica API + FastF1 wrappers, JSON cache"],
        ["Data ingestion",  "weather_fetcher.py",     "Open-Meteo historical / forecast"],
        ["Data ingestion",  "news_fetcher.py",        "NewsAPI + VADER sentiment (live only)"],
        ["Feature build",   "feature_engineering.py","Assembles 23-feature rows per driver/race"],
        ["Modelling",       "statistical_model.py",   "Logistic Regression (sklearn Pipeline)"],
        ["Modelling",       "ml_model.py",            "XGBoost classifier"],
        ["Ensemble",        "ensemble.py",            "50/50 weighted combine + news adjustments"],
        ["CLI",             "train.py",               "Fetch data, build features, train both models"],
        ["CLI",             "predict.py",             "Load models, predict given year/round"],
    ]
    t = make_table(arch_data, [3.8 * cm, 5.0 * cm, 8.4 * cm], styles)

    return [
        Paragraph("2  System Architecture", h1),
        section_rule(styles),
        Paragraph(
            "The codebase is a single Python package with two CLI entry-points (<i>train.py</i> "
            "and <i>predict.py</i>) backed by five specialised modules in <i>src/</i>. "
            "All external API responses are cached to <i>data/raw/</i> so that subsequent "
            "runs never re-fetch data that has already been downloaded.",
            body,
        ),
        Spacer(1, 0.2 * cm),
        t,
        Spacer(1, 0.4 * cm),
        Paragraph("Data flow (training):", h2),
        Paragraph("• Jolpica API → race results, qualifying, sprint results, driver/constructor standings", bullet),
        Paragraph("• Open-Meteo API → historical race-day weather per circuit", bullet),
        Paragraph("• feature_engineering.py assembles one DataFrame row per driver per race", bullet),
        Paragraph("• Both models trained on the resulting DataFrame; pickled to models/", bullet),
        Paragraph("Data flow (prediction):", h2),
        Paragraph("• Qualifying grid positions fetched (FastF1 first, Jolpica fallback)", bullet),
        Paragraph("• Same feature vector computed using history up to previous race", bullet),
        Paragraph("• Live weather forecast fetched from Open-Meteo", bullet),
        Paragraph("• NewsAPI articles scraped; VADER compound scores computed", bullet),
        Paragraph("• Both models score each driver; ensemble combines and adjusts", bullet),
        Spacer(1, 0.3 * cm),
    ]


def section3(styles):
    h1, h2, body, bullet = styles["h1"], styles["h2"], styles["body"], styles["bullet"]
    ds_data = [
        ["Source", "What it provides", "Auth / Cost", "Caching"],
        ["Jolpica (Ergast)", "Race results, qualifying, sprint results, driver & constructor standings, season schedule",
         "None (public)", "JSON files in data/raw/"],
        ["FastF1",  "Current-season qualifying times (2018+); more reliable for recent rounds",
         "None (public)", "FastF1 cache/ dir"],
        ["Open-Meteo", "Historical (1940+) and forecast (16 days) weather: temperature, precipitation, wind",
         "None (public)", "JSON files in data/raw/"],
        ["NewsAPI", "Up to 100 recent articles per query; titles + descriptions used for sentiment",
         "API key (free, 500 req/day)", "Not cached (live only)"],
    ]
    t = make_table(ds_data, [3.2 * cm, 6.8 * cm, 3.4 * cm, 3.8 * cm], styles)

    return [
        Paragraph("3  Data Sources", h1),
        section_rule(styles),
        Paragraph(
            "All four data sources are free-tier or fully public. The design intentionally "
            "avoids paid data providers so the system can be run by anyone with an internet "
            "connection.",
            body,
        ),
        Spacer(1, 0.2 * cm),
        t,
        Spacer(1, 0.35 * cm),
        Paragraph("Notes on Jolpica:", h2),
        Paragraph(
            "Ergast, the original F1 data API, was shut down in early 2025. Jolpica "
            "(<i>api.jolpi.ca/ergast/f1/</i>) is its community-maintained, drop-in replacement "
            "with an identical JSON schema. The project migrated to Jolpica with no changes to "
            "parsing logic.",
            body,
        ),
        Paragraph("Notes on news sentiment:", h2),
        Paragraph(
            "NewsAPI only retains articles for approximately 30 days, making it impossible to "
            "build a historical sentiment dataset for training. Therefore news features are "
            "<b>not included in the 23 FEATURE_COLS</b> used by either model. Instead they are "
            "applied as a post-hoc multiplicative adjustment in the ensemble layer at prediction "
            "time only.",
            body,
        ),
        Spacer(1, 0.3 * cm),
    ]


def section4(styles):
    h1, h2, body, bullet = styles["h1"], styles["h2"], styles["body"], styles["bullet"]

    features_data = [
        ["Feature", "Type", "Rationale"],
        ["grid_position",              "int",   "Starting position — strongest single predictor of race outcome."],
        ["grid_pos_win_rate",          "float", "Historical win% from this grid slot (computed from full dataset). Captures non-linear drop-off beyond P3."],
        ["driver_circuit_win_rate",    "float", "Fraction of races the driver has won at this specific circuit. Tracks circuit specialists (e.g. Verstappen at Spa)."],
        ["driver_circuit_podium_rate", "float", "Podium rate at circuit. Complements win rate with broader on-circuit strength signal."],
        ["driver_circuit_starts",      "int",   "Experience count at circuit. Regularises the rate features (0 starts → 0.0 rate, not missing)."],
        ["driver_recent_points_5",     "float", "Sum of points in last 5 races (within 2-year window). Captures current form trajectory."],
        ["driver_recent_wins_5",       "int",   "Win count in last 5 races. Separates drivers on winning streaks."],
        ["driver_standings_pos",       "int",   "Championship position entering the race. Proxy for seasonal car + driver package quality."],
        ["driver_season_wins",         "int",   "Season wins to date. Reinforces momentum signal independent of points gaps."],
        ["constructor_standings_pos",  "int",   "Team's constructor championship position. Captures car development trajectory mid-season."],
        ["is_home_race",               "bool",  "Driver racing at their home circuit (matched by nationality). Small but measurable motivation/crowd effect."],
        ["has_grid_penalty",           "bool",  "Grid penalty applied (grid > qualifying + 1 position). Negative signal disconnected from car pace."],
        ["grid_penalty_positions",     "int",   "Number of positions dropped by penalty. Quantifies penalty severity."],
        ["circuit_safety_car_rate",    "float", "Heuristic probability of safety car at this circuit. High-SC circuits randomise races; affects grid-position importance."],
        ["season_round_pct",           "float", "Round / total rounds. Models championship-phase effects (teams save upgrades for early races, etc.)."],
        ["rain_mm",                    "float", "Total precipitation on race day. Continuous signal; wet races dramatically reshuffle outcomes."],
        ["temp_celsius",               "float", "Max air temperature. Affects tyre degradation characteristics and strategy."],
        ["wind_speed_kmh",             "float", "Max wind speed. High wind amplifies aero sensitivity differences between cars."],
        ["is_wet_race",                "bool",  "Binary flag (rain_mm > 1.0 mm). Allows models to learn a discrete wet-race regime separately from the continuous rain_mm."],
        ["is_sprint_weekend",          "bool",  "Weekend includes a Sprint race. Changes strategic calculus and tyre usage."],
        ["sprint_pos_score",           "int",   "Same-weekend Sprint finish encoded as (23 − position); 0 for non-sprint weekends. Saturday pace directly previews Sunday pace."],
        ["sprint_points",              "float", "Sprint points scored this weekend. Complements sprint_pos_score with the championship-points dimension."],
        ["driver_recent_sprint_pts_3", "float", "Sum of Sprint points in last 3 sprint races (rolling). Identifies consistent Sprint performers."],
    ]

    t = make_table(features_data, [4.8 * cm, 1.4 * cm, 11.0 * cm], styles)

    return [
        Paragraph("4  Feature Engineering", h1),
        section_rule(styles),
        Paragraph(
            "Features are computed in <i>src/feature_engineering.py</i>. The canonical list is "
            "<i>FEATURE_COLS</i> (23 features). Each race produces one row per driver; "
            "all lookups use strictly historical data (prior races only) to prevent leakage.",
            body,
        ),
        Paragraph("Key design principles:", h2),
        Paragraph("• <b>No data leakage</b>: circuit stats and recent form use only races <i>before</i> the target race.", bullet),
        Paragraph("• <b>Graceful defaults</b>: 0 circuit starts → 0.0 rates (not NaN); missing weather → neutral defaults.", bullet),
        Paragraph("• <b>Zero for non-sprint</b>: all four sprint features default to 0 on normal weekends, so the model learns sprint-specific adjustments naturally.", bullet),
        Paragraph("• <b>Sentiment excluded from training</b>: news data unavailable historically; applied post-training as a multiplier.", bullet),
        Spacer(1, 0.3 * cm),
        Paragraph("Feature definitions:", h2),
        Spacer(1, 0.15 * cm),
        t,
        Spacer(1, 0.3 * cm),
    ]


def section5(styles):
    h1, h2, body, bullet, code = styles["h1"], styles["h2"], styles["body"], styles["bullet"], styles["code"]

    lr_params = [
        ["Parameter", "Value", "Reasoning"],
        ["solver",           "lbfgs",    "Efficient for small-to-medium datasets; handles L2 penalty natively."],
        ["C (regularisation)", "0.5",   "Moderate L2 regularisation. Prevents over-fitting on correlated features (e.g. grid_position and grid_pos_win_rate)."],
        ["class_weight",     "balanced", "Corrects for ~5% positive class (1 winner per ~20 drivers) by up-weighting winner rows."],
        ["max_iter",         "2000",     "Sufficient for convergence with 23 features on ~8,000 rows."],
        ["Preprocessing",    "StandardScaler", "Z-score normalisation applied in Pipeline before logistic regression. Required for meaningful coefficients and regularisation."],
    ]

    xgb_params = [
        ["Parameter", "Value", "Reasoning"],
        ["n_estimators",     "400",  "Enough trees to capture complex patterns without severe over-fitting."],
        ["max_depth",        "5",    "Limits each tree; prevents memorising individual races."],
        ["learning_rate",    "0.05", "Slow shrinkage; compensated by 400 trees."],
        ["subsample",        "0.8",  "Row-sampling per tree adds diversity and reduces variance."],
        ["colsample_bytree", "0.8",  "Feature-sampling per tree; prevents any single feature dominating."],
        ["min_child_weight", "3",    "Minimum sum of instance weights per leaf; avoids splits on very few winners."],
        ["gamma",            "1.0",  "Minimum loss reduction to make a split; prunes noisy splits."],
        ["scale_pos_weight", "auto", "Computed as (n_negatives / n_positives) ≈ 19; mirrors class_weight='balanced' for XGBoost."],
        ["tree_method",      "hist", "Histogram-based algorithm for fast training on CPU."],
    ]

    t_lr  = make_table(lr_params,  [4.0 * cm, 2.5 * cm, 10.7 * cm], styles)
    t_xgb = make_table(xgb_params, [4.0 * cm, 2.5 * cm, 10.7 * cm], styles)

    return [
        Paragraph("5  Models", h1),
        section_rule(styles),
        Paragraph(
            "Two models are trained independently on the same 23-feature dataset and combined "
            "by a 50/50 weighted ensemble. This hedge between interpretability (logistic "
            "regression) and non-linear capacity (XGBoost) consistently outperforms either "
            "model alone in leave-one-season-out cross-validation.",
            body,
        ),

        Paragraph("5.1  Logistic Regression", h2),
        Paragraph(
            "Implemented in <i>src/statistical_model.py</i> as an sklearn Pipeline "
            "(StandardScaler → LogisticRegression). It predicts a binary outcome "
            "(<i>winner=1</i>/<i>winner=0</i>) and is trained with balanced class weights "
            "to compensate for the strong class imbalance.",
            body,
        ),
        Paragraph(
            "After training, coefficients are printed per feature. <i>grid_position</i> "
            "reliably carries the strongest negative coefficient (lower grid number = higher "
            "win probability). <i>driver_circuit_win_rate</i> and "
            "<i>driver_recent_points_5</i> tend to be the next most important.",
            body,
        ),
        Paragraph(
            "Predicted raw probabilities are normalised so they sum to 1.0 across the field "
            "before being passed to the ensemble — converting a binary classifier output into "
            "a proper race-win probability distribution.",
            body,
        ),
        Spacer(1, 0.2 * cm),
        t_lr,
        Spacer(1, 0.35 * cm),

        Paragraph("5.2  XGBoost", h2),
        Paragraph(
            "Implemented in <i>src/ml_model.py</i> using <i>XGBClassifier</i> (xgboost library). "
            "XGBoost handles non-linear feature interactions naturally — for example, the "
            "interaction between <i>grid_position</i> and <i>circuit_safety_car_rate</i>: "
            "a pole-sitter at Monaco faces higher probability of losing their lead to a safety "
            "car than a pole-sitter at Monza.",
            body,
        ),
        Paragraph(
            "<i>scale_pos_weight</i> is computed automatically as the ratio of negative to "
            "positive training samples (≈19 for a typical 20-car grid). This is the XGBoost "
            "equivalent of sklearn's <i>class_weight='balanced'</i> and prevents the model "
            "from trivially predicting <i>winner=0</i> for every row.",
            body,
        ),
        Paragraph(
            "A leave-one-season-out cross-validation is available (<i>cross_validate_seasons</i>) "
            "to measure out-of-sample performance. CV model checkpoints are saved as "
            "<i>models/xgb_cv_{year}.pkl</i>.",
            body,
        ),
        Spacer(1, 0.2 * cm),
        t_xgb,
        Spacer(1, 0.3 * cm),
    ]


def section6(styles):
    h1, h2, body, bullet = styles["h1"], styles["h2"], styles["body"], styles["bullet"]

    adj_data = [
        ["Adjustment", "Formula", "Range", "Rationale"],
        ["Driver news sentiment", "prob × (1 + 0.10 × driver_sentiment)", "±10%",
         "VADER compound score from up to 100 recent articles mentioning the driver."],
        ["Team news sentiment",   "Combined as avg(driver + team) × 0.10",   "±10%",
         "Team-level buzz (upgrade announcements, reliability issues, etc.)"],
        ["Team upgrade flag",     "prob × (1 + 0.05 × upgrade_flag)",         "+5%",
         "Binary: 1 if recent articles mention a car upgrade for this team."],
        ["Clipping",              "adjustment clipped to [0.5, 2.0]",         "—",
         "Prevents extreme sentiment from completely eliminating or dominating a driver."],
        ["Renormalisation",       "Divide by sum after adjustment",            "—",
         "Ensures final probabilities sum to 100% across the full field."],
    ]
    t = make_table(adj_data, [3.8 * cm, 4.8 * cm, 1.6 * cm, 7.0 * cm], styles)

    return [
        Paragraph("6  Ensemble & Post-Processing", h1),
        section_rule(styles),
        Paragraph(
            "The ensemble layer (<i>src/ensemble.py</i>) merges the two model outputs and "
            "applies live adjustments that cannot be learned from historical training data.",
            body,
        ),
        Paragraph("Base combination:", h2),
        Paragraph(
            "Base probability = 0.5 × logistic_prob + 0.5 × xgb_prob. "
            "Both inputs have already been normalised to sum to 1.0 individually, "
            "so the 50/50 weighted sum is also normalised. The equal weighting was chosen "
            "because the two models have comparable cross-validated performance and bring "
            "complementary strengths: logistic regression is more stable on small circuits "
            "with few historical starts; XGBoost is better at capturing non-linear "
            "interactions on data-rich circuits.",
            body,
        ),
        Paragraph("Post-training news adjustments:", h2),
        Spacer(1, 0.15 * cm),
        t,
        Spacer(1, 0.3 * cm),
        Paragraph(
            "The multiplier approach (rather than adding sentiment as a model feature) was "
            "chosen deliberately: it keeps the base model stable across sessions and avoids "
            "retraining whenever the news landscape changes. The 10% / 5% magnitudes are "
            "intentionally conservative so that sentiment never overrides a clear grid-position "
            "or form signal.",
            body,
        ),
        Spacer(1, 0.3 * cm),
    ]


def section7(styles):
    h1, h2, body, bullet, code = styles["h1"], styles["h2"], styles["body"], styles["bullet"], styles["code"]
    flow_data = [
        ["Step", "Action", "Source"],
        ["1", "Fetch upcoming race schedule & circuit info",        "Jolpica season/{year}/schedule"],
        ["2", "Fetch qualifying results (grid positions)",          "FastF1 (primary) → Jolpica (fallback)"],
        ["3", "Fetch same-weekend sprint results (if applicable)",  "Jolpica /{year}/{round}/sprint.json"],
        ["4", "Load historical results for feature context",        "Jolpica cached data (2010 → today)"],
        ["5", "Fetch driver/constructor standings (prev. round)",   "Jolpica standings endpoint"],
        ["6", "Fetch race-day weather forecast",                    "Open-Meteo forecast API"],
        ["7", "Build 23-feature vector per driver",                 "feature_engineering.build_race_features()"],
        ["8", "Score with Logistic Regression",                     "statistical_model.predict()"],
        ["9", "Score with XGBoost",                                 "ml_model.predict()"],
        ["10","Fetch live news; compute VADER sentiment",           "NewsAPI + VADER (optional, --no-news to skip)"],
        ["11","Ensemble: combine + apply adjustments + renormalise","ensemble.predict_from_features()"],
        ["12","Print ranked leaderboard with win probabilities",    "predict.py output"],
    ]
    t = make_table(flow_data, [1.0 * cm, 9.5 * cm, 6.7 * cm], styles)

    return [
        Paragraph("7  Prediction Workflow", h1),
        section_rule(styles),
        Paragraph(
            "Running <b>predict.py --year Y --round R</b> executes the following pipeline. "
            "Each step degrades gracefully: if qualifying data is unavailable, grid positions "
            "default to 0 and a notice is printed; if weather is unavailable, neutral defaults "
            "are used.",
            body,
        ),
        Spacer(1, 0.2 * cm),
        t,
        Spacer(1, 0.35 * cm),
        Paragraph("CLI flags:", h2),
        Paragraph("• <b>--no-news</b>: skips steps 10 and news adjustment (useful for backtesting)", bullet),
        Paragraph("• <b>--no-qualifying</b>: forces grid_position=0 for all drivers (pre-qualifying prediction)", bullet),
        Spacer(1, 0.3 * cm),
    ]


def section8(styles):
    h1, h2, body, bullet, code = styles["h1"], styles["h2"], styles["body"], styles["bullet"], styles["code"]
    return [
        Paragraph("8  Training Workflow", h1),
        section_rule(styles),
        Paragraph(
            "Running <b>train.py</b> fetches the full historical dataset, builds the feature "
            "matrix, and trains both models sequentially. All API calls are cached, so "
            "re-running after the first fetch completes in seconds.",
            body,
        ),
        Paragraph("Typical commands:", h2),
        Paragraph("Train through the 2026 Japanese GP (Round 3):", body),
        Paragraph("python train.py --start-year 2010 --end-year 2026 --end-round 3", code),
        Spacer(1, 0.15 * cm),
        Paragraph("Train through all completed 2026 races:", body),
        Paragraph("python train.py --start-year 2010 --end-year 2026", code),
        Spacer(1, 0.15 * cm),
        Paragraph("Retrain from cached CSV (skip API fetch):", body),
        Paragraph("python train.py --start-year 2010 --end-year 2026 --use-cached", code),
        Spacer(1, 0.3 * cm),
        Paragraph("Class imbalance:", h2),
        Paragraph(
            "With 20 drivers per race and one winner, positive-class rows account for "
            "approximately 5% of training data. Both models address this: logistic regression "
            "via <i>class_weight='balanced'</i>, XGBoost via auto-computed "
            "<i>scale_pos_weight ≈ 19</i>. Without correction both models would trivially "
            "predict <i>winner=0</i> for every driver and achieve 95% accuracy while being "
            "completely useless.",
            body,
        ),
        Paragraph("Model artefacts:", h2),
        Paragraph("• <b>models/logistic_model.pkl</b> — sklearn Pipeline (StandardScaler + LogisticRegression)", bullet),
        Paragraph("• <b>models/xgb_model.pkl</b> — XGBClassifier, full training set", bullet),
        Paragraph("• <b>models/xgb_cv_{year}.pkl</b> — leave-one-season-out CV checkpoints (2023–2026)", bullet),
        Paragraph("• <b>data/processed/training_data.csv</b> — full feature matrix", bullet),
        Spacer(1, 0.3 * cm),
    ]


def section9(styles):
    h1, h2, body, bullet = styles["h1"], styles["h2"], styles["body"], styles["bullet"]
    return [
        Paragraph("9  Limitations & Future Work", h1),
        section_rule(styles),
        Paragraph("Known limitations:", h2),
        Paragraph(
            "• <b>Safety car rate</b>: the current implementation uses a hardcoded heuristic "
            "per circuit rather than actual SC lap data. FastF1 lap-by-lap telemetry could "
            "provide real SC frequency.",
            bullet,
        ),
        Paragraph(
            "• <b>Tyre strategy</b>: compound choices and pit-stop strategy are not modelled. "
            "A two-stopper can beat a front-row starter on certain circuits.",
            bullet,
        ),
        Paragraph(
            "• <b>Team radio / mechanical issues</b>: DNS, DNF, and reliability are not "
            "explicitly modelled. A driver with known engine failures in practice would not "
            "be penalised by the current features.",
            bullet,
        ),
        Paragraph(
            "• <b>Aerodynamic regulation changes</b>: major rule changes (2014, 2022, 2026) "
            "render pre-change historical statistics less predictive for post-change seasons. "
            "The model is trained on data from 2010 onwards but does not discount older "
            "seasons explicitly.",
            bullet,
        ),
        Paragraph(
            "• <b>50/50 ensemble weight</b>: equal weighting was chosen empirically. A "
            "stacked generaliser trained on out-of-fold predictions would optimise this split.",
            bullet,
        ),
        Paragraph("Future improvements:", h2),
        Paragraph("• Replace SC heuristic with actual SC lap counts from FastF1 telemetry.", bullet),
        Paragraph("• Add tyre compound and pit window features.", bullet),
        Paragraph("• Bayesian hyperparameter search for XGBoost instead of manual grid.", bullet),
        Paragraph("• Stacking: train a meta-model on cross-validated model outputs.", bullet),
        Paragraph("• Calibration: isotonic regression or Platt scaling to produce better-calibrated probabilities.", bullet),
        Paragraph("• Investigate season-weighting to discount data from pre-regulation-change eras.", bullet),
        Spacer(1, 0.5 * cm),
        HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=6),
        Paragraph(
            "End of document  ·  F1 Race Winner Prediction v1.0  ·  May 2026",
            styles["caption"],
        ),
    ]


def build_pdf(output_path: str = "F1_Prediction_Technical_Doc.pdf"):
    page_width, page_height = A4  # 595 × 842 pts

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="F1 Race Winner Prediction — Technical Document",
        author="F1 Prediction Project",
    )

    styles = build_styles()

    story = []
    story += cover_page(styles, page_width, page_height)
    story += toc(styles)
    story.append(PageBreak())
    story += section1(styles)
    story += section2(styles, page_width)
    story += section3(styles)
    story.append(PageBreak())
    story += section4(styles)
    story.append(PageBreak())
    story += section5(styles)
    story.append(PageBreak())
    story += section6(styles)
    story += section7(styles)
    story.append(PageBreak())
    story += section8(styles)
    story += section9(styles)

    doc.build(story)
    print(f"PDF written to: {output_path}")


if __name__ == "__main__":
    build_pdf()
