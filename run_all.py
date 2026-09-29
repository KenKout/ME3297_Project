"""Chạy lại TOÀN BỘ phân tích và tạo mọi bảng, hình dùng trong báo cáo.

Cách chạy:
    python get_data.py        # chạy 1 lần: giải nén CSV vào data/raw/
    python run_all.py         # ghi kết quả ra outputs/tables/*.csv và outputs/figures/*.png

Các bước chính (khớp với các mục của báo cáo):
    1. Đọc và audit dữ liệu                       -> báo cáo mục 4 (Data)
    2. Chia train/test THEO THỜI GIAN             -> mục 5, 7
    3. Cross-validation kiểu rolling-origin       -> mục 7, 8
    4. Định nghĩa chính sách ra quyết định        -> mục 7
    5. Huấn luyện lại trên toàn bộ train, đánh giá trên test -> mục 8
    6. Diễn giải mô hình                          -> mục 9 (Interpretation)
    7. Khuyến nghị, độ nhạy chi phí, kịch bản đổi ngày hứa -> mục 10 (Recommendation)
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # backend không cần màn hình: chỉ lưu hình ra file PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.data_prep import FEATURES, TARGET, audit, build_orders, load_raw
from src.modeling import (BASELINES, FIXED_RULES, NON_PROBABILISTIC, SEED, CostModel,
                          best_threshold, ev_flag, evaluate, flag_top, make_models,
                          make_preprocessor, scores)

# Tắt cảnh báo "chưa hội tụ" của MLP/LinearSVC: model vẫn dùng được, cảnh báo chỉ làm rối log.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
# ---------------------------------------------------------------- tham số chung
SPLIT_DATE = pd.Timestamp("2017-07-01")  # đơn trước ngày này -> train, từ ngày này -> test
N_FOLDS = 5                # số fold của TimeSeriesSplit
ALERT_RATE = 0.35          # năng lực cố định để so sánh: khoảng bằng tỷ lệ flag của luật First/Second Class
N_BOOT = 200               # số lần bootstrap để tính khoảng tin cậy 95% trên tập test
RULE = "Rule: + Same Day cut-off"     # luật cố định mạnh nhất
LOOKUP = "Lookup: late rate by mode"  # baseline xác suất mạnh nhất (bảng tra)
COST = CostModel()         # $10/đơn flag, thiệt hại = 10% giá trị đơn, tránh được 50%

# Từ điển dữ liệu (data dictionary) cho mọi cột được dùng: brief §6.1 yêu cầu.
# Mỗi mục: tên cột -> (kiểu dữ liệu, mô tả). Được xuất ra data_dictionary.csv.

DATA_DICTIONARY = {
    "shipping_mode": ("categorical", "Shipping mode chosen at order (Standard/Second/First/Same Day)"),
    "market": ("categorical", "Destination market (5 values)"),
    "order_region": ("categorical", "Destination region (23 values)"),
    "customer_segment": ("categorical", "Consumer / Corporate / Home Office"),
    "payment_type": ("categorical", "Transaction type: DEBIT, TRANSFER, PAYMENT, CASH"),
    "main_department": ("categorical", "Department of the order's highest-value line"),
    "order_country": ("categorical, target-encoded", "Destination country"),
    "order_city": ("categorical, target-encoded", "Destination city"),
    "main_category": ("categorical, target-encoded", "Product category of the highest-value line"),
    "scheduled_days": ("numeric", "Promised shipping days (fixed by mode: 0/1/2/4)"),
    "n_lines": ("numeric", "Number of order lines"),
    "total_qty": ("numeric", "Total units ordered"),
    "order_value": ("numeric, USD", "Sum of line Sales (before discount)"),
    "total_discount": ("numeric, USD", "Sum of line discounts"),
    "mean_discount_rate": ("numeric", "Mean line discount rate"),
    "max_unit_price": ("numeric, USD", "Highest unit price in the order"),
    "n_categories": ("numeric", "Distinct product categories in the order"),
    "order_profit": ("numeric, USD", "Sum of 'Benefit per order' over lines"),
    "profit_margin": ("numeric", "order_profit / order_value"),
    "weekday": ("numeric", "Day of week of order (0 = Monday)"),
    "hour": ("numeric", "Hour of day of order"),
    "month": ("numeric", "Month of order"),
    "cust_prior_orders": ("numeric", "Customer's earlier orders whose outcome was already known"),
    "cust_prior_late_rate": ("numeric", "Smoothed late share of those orders"),
    "city_prior_orders": ("numeric", "Earlier orders to the same city with known outcome"),
    "city_prior_late_rate": ("numeric", "Smoothed late share of those orders"),
    "citymode_prior_orders": ("numeric", "Same, for city x shipping mode"),
    "citymode_prior_late_rate": ("numeric", "Smoothed late share of those orders"),
    "late": ("binary target", "Late_delivery_risk: 1 if real shipping days > scheduled days"),
}


def savefig(name: str) -> None:
    """Lưu hình hiện tại vào outputs/figures/<name>.png rồi đóng lại để giải phóng bộ nhớ."""
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.close()


def section(title: str) -> None:
    """In tiêu đề cho từng bước để dễ theo dõi log."""
    print(f"\n=== {title} ===", flush=True)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {}   # các con số chính, cuối cùng ghi ra outputs/summary.json

    # ------------------------------------------------------------ 1. data audit
    # Đọc CSV, tạo các bảng audit (kích thước, giá trị thiếu, ngoại lai, leakage,
    # kiểm tra tính thật của dữ liệu), rồi gộp về bảng mỗi dòng = một đơn hàng.
    section("1. Load and audit")
    raw = load_raw()
    for name, table in audit(raw).items():
        table.to_csv(TAB / f"audit_{name}.csv")
        print(f"\n[{name}]\n{table.to_string()}")

    orders = build_orders(raw)
    print(f"\nOrder-level table: {orders.shape}, late share {orders[TARGET].mean():.3f}")
    # Xuất data dictionary kèm số giá trị thiếu và số giá trị khác nhau của từng feature.
    pd.DataFrame([{"feature": k, "type": v[0], "description": v[1],
                   "n_missing": int(orders[k].isna().sum()),
                   "n_unique": int(orders[k].nunique())}
                  for k, v in DATA_DICTIONARY.items()]).to_csv(TAB / "data_dictionary.csv", index=False)

    # Hình 1: số đơn mỗi tháng (cột) và tỷ lệ trễ (đường). Dùng để kiểm tra xu hướng
    # theo thời gian và đánh dấu điểm chia train/test.
    monthly = orders.set_index("order_dt").resample("MS")[TARGET].agg(["size", "mean"])
    monthly.to_csv(TAB / "monthly_volume_late_rate.csv")
    fig, ax1 = plt.subplots(figsize=(9, 3.8))
    ax1.bar(monthly.index, monthly["size"], width=20, color="#9ecae1", label="Orders")
    ax1.set_ylabel("Orders per month")
    ax1.set_xlabel("Order month")
    ax2 = ax1.twinx()   # trục y thứ hai (bên phải) cho tỷ lệ trễ
    ax2.plot(monthly.index, monthly["mean"], color="#d62728", marker=".", label="Late share")
    ax2.set_ylabel("Share of orders delivered late")
    ax2.set_ylim(0, 1)
    ax1.axvline(SPLIT_DATE, color="k", ls="--", lw=1)
    ax1.text(SPLIT_DATE, ax1.get_ylim()[1] * 0.95, " train | test", va="top")
    ax1.set_title("Monthly order volume and late-delivery share (non-cancelled orders)")
    savefig("01_monthly_volume_late_rate")

    # Hình 2: tỷ lệ trễ theo shipping mode. Đây là phát hiện quan trọng nhất của EDA.
    by_mode = orders.groupby("shipping_mode")[TARGET].agg(["size", "mean"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(by_mode.index, by_mode["mean"], color="#6baed6")
    for i, (n, m) in enumerate(zip(by_mode["size"], by_mode["mean"])):
        ax.text(m + 0.01, i, f"{m:.0%}  (n={n:,})", va="center")
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Share of orders delivered late")
    ax.set_title("Late-delivery share by shipping mode")
    savefig("02_late_rate_by_mode")

    # ------------------------------------------------------------ 2. split
    # Chia THEO THỜI GIAN, không xáo trộn: mô hình học từ quá khứ và được kiểm tra trên
    # tương lai, đúng như khi triển khai thật. Xáo trộn dữ liệu có ngày tháng là lỗi
    # leakage (brief §10, ghi chú về Rossmann).
    section("2. Time-based split")
    train = orders[orders["order_dt"] < SPLIT_DATE].reset_index(drop=True)
    test = orders[orders["order_dt"] >= SPLIT_DATE].reset_index(drop=True)
    # Chốt chặn an toàn: nếu ai đó lỡ thêm cột kết quả hoặc mã đơn vào FEATURES,
    # chương trình dừng ngay. Order Id bị cấm vì trong dữ liệu này nó quyết định số ngày giao.
    leaked = set(FEATURES) & {"outcome_real_days", "late", "Order Id"}
    assert not leaked, f"post-outcome or ID columns in features: {leaked}"
    # X = feature, y = nhãn (0/1), v = giá trị đơn (USD, dùng cho mô hình chi phí).
    X_tr, y_tr, v_tr = train[FEATURES], train[TARGET].to_numpy(), train["order_value"].to_numpy()
    X_te, y_te, v_te = test[FEATURES], test[TARGET].to_numpy(), test["order_value"].to_numpy()
    # Số tuần của giai đoạn test: dùng để đổi kết quả ra "mỗi tuần" cho người quản lý dễ hình dung.
    test_weeks = (test["order_dt"].max() - test["order_dt"].min()).days / 7
    split_info = pd.DataFrame({
        "set": ["train", "test"],
        "from": [train["order_dt"].min(), test["order_dt"].min()],
        "to": [train["order_dt"].max(), test["order_dt"].max()],
        "orders": [len(train), len(test)],
        "late_share": [y_tr.mean(), y_te.mean()],
    })
    split_info.to_csv(TAB / "split.csv", index=False)
    print(split_info.to_string(index=False))
    summary["test_weeks"] = test_weeks
    summary["test_orders_per_week"] = len(test) / test_weeks

    # Hình 2b: tỷ lệ trễ của đơn Same Day theo giờ đặt (CHỈ dùng dữ liệu train).
    # Hình này là cơ sở cho luật "cut-off 12 giờ trưa".
    sd = train[train["shipping_mode"] == "Same Day"]
    late_by_hour = sd.groupby("hour")[TARGET].mean()
    late_by_hour.to_csv(TAB / "train_same_day_late_by_hour.csv")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(late_by_hour.index, late_by_hour.values, color="#6baed6")
    ax.set_xlabel("Hour of day the order was placed")
    ax.set_ylabel("Share delivered late")
    ax.set_title("Same Day orders: late share by order hour (training period)")
    savefig("02b_same_day_late_by_hour")

    # Phân tích "đổi ngày hứa" (chỉ mang tính mô tả, có dùng kết quả thực tế nên KHÔNG
    # bao giờ là feature): nếu hứa p ngày thay vì số ngày hiện tại, thì bao nhiêu % đơn
    # của mỗi mode sẽ bị trễ?
    req_rows = []
    for mode, g in train.groupby("shipping_mode"):
        real = g["outcome_real_days"]
        row = {"shipping_mode": mode, "current_promise_days": int(g["scheduled_days"].iloc[0]),
               "orders_train": len(g), "real_days_mean": real.mean(),
               "real_days_min": real.min(), "real_days_max": real.max()}
        for p in range(0, 7):
            row[f"late_share_if_promise_{p}d"] = (real > p).mean()
        req_rows.append(row)
    requote = pd.DataFrame(req_rows)
    requote.to_csv(TAB / "promise_requote_train.csv", index=False)
    print("\nLate share by promised days (training period):\n" + requote.round(3).to_string(index=False))

    # ------------------------------------------------------------ 3. CV
    # TimeSeriesSplit: fold k huấn luyện trên mọi dữ liệu TRƯỚC khối k và kiểm tra trên
    # khối k (cửa sổ huấn luyện mở rộng dần). Không bao giờ dùng dữ liệu tương lai để
    # dự đoán quá khứ. Ta báo cáo cả mean và SD qua 5 fold (brief §6.5).
    section("3. Rolling-origin cross-validation on the training period")
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    models = make_models()
    # fold_rows: metric của từng (model, fold).
    # oof: điểm "out-of-fold" của mỗi model, tức điểm cho mỗi đơn train được dự đoán
    #      bởi một model KHÔNG được huấn luyện trên đơn đó. Dùng để chọn ngưỡng cho SVM.
    fold_rows, oof = [], {}
    for name, model in models.items():
        oof[name] = np.full(len(train), np.nan)
        t = time.time()
        for k, (i_fit, i_val) in enumerate(tscv.split(X_tr), start=1):
            # clone() tạo bản sao CHƯA huấn luyện -> mỗi fold bắt đầu từ đầu, kể cả
            # bộ tiền xử lý trong Pipeline (không rò thông tin giữa các fold).
            m = clone(model).fit(X_tr.iloc[i_fit], y_tr[i_fit])
            s = scores(m, X_tr.iloc[i_val])
            oof[name][i_val] = s
            fold_rows.append({"model": name, "fold": k, "n_fit": len(i_fit), "n_val": len(i_val),
                              **evaluate(s, y_tr[i_val], v_tr[i_val], COST, ALERT_RATE)})
        print(f"  {name:<28s} done in {time.time() - t:5.1f}s", flush=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(TAB / "cv_folds.csv", index=False)
    metric_cols = [c for c in folds.columns if c not in ("model", "fold", "n_fit", "n_val")]
    # Tổng hợp: mean và độ lệch chuẩn (std) của mỗi metric qua 5 fold.
    cv = folds.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    cv.columns = [f"{a}_{b}" for a, b in cv.columns]
    cv.to_csv(TAB / "cv_summary.csv")
    print(cv[["pr_auc_mean", "pr_auc_std", "pr_auc_tiebroken_mean", "pr_auc_tiebroken_std",
              "roc_auc_mean", "roc_auc_std",
              f"recall@{ALERT_RATE:.0%}_mean", f"recall@{ALERT_RATE:.0%}_std"]].round(4).to_string())

    # Hình 3: boxplot PR-AUC qua các fold, cho thấy độ phân tán chứ không chỉ một con số.
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = list(models)
    ax.boxplot([folds.loc[folds["model"] == n, "pr_auc_tiebroken"] for n in names], showmeans=True)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=25, ha="right")
    ax.set_ylabel("PR-AUC (average precision,\nties broken at random)")
    ax.set_title(f"PR-AUC across {N_FOLDS} rolling-origin folds (training period)")
    savefig("03_cv_pr_auc_boxplot")

    # Thử nhiều độ sâu cây (một bước tinh chỉnh nhỏ, có ghi lại): cây sâu hơn có PR-AUC
    # cao hơn một chút, nhưng khó giải thích hơn và dễ overfit hơn.
    depth_rows = []
    for depth in [2, 3, 4, 6, 8, 12, None]:
        pipe = Pipeline([("prep", make_preprocessor(scale=False)),
                         ("clf", DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50,
                                                        random_state=SEED))])
        vals = []
        for i_fit, i_val in tscv.split(X_tr):
            m = clone(pipe).fit(X_tr.iloc[i_fit], y_tr[i_fit])
            vals.append(evaluate(scores(m, X_tr.iloc[i_val]), y_tr[i_val], v_tr[i_val],
                                 COST, ALERT_RATE)["pr_auc"])
        # ddof=1: độ lệch chuẩn mẫu, cùng cách tính với pandas .std() ở bảng CV.
        depth_rows.append({"max_depth": depth or "none", "pr_auc_mean": np.mean(vals),
                           "pr_auc_std": np.std(vals, ddof=1)})
    pd.DataFrame(depth_rows).to_csv(TAB / "tree_depth_sweep.csv", index=False)
    print("\nTree depth sweep:\n" + pd.DataFrame(depth_rows).round(4).to_string(index=False))

    # ------------------------------------------------------------ 4. policies
    section("4. Decision policies")
    # Biến điểm số thành QUYẾT ĐỊNH flag / không flag:
    #   - Luật cố định: flag đúng những đơn luật chọn.
    #   - "Majority class": flag tất cả (tương đương "đơn nào cũng coi là rủi ro").
    #   - Mô hình có xác suất (kể cả bảng tra): chính sách giá trị kỳ vọng ev_flag.
    #   - SVM (điểm không phải xác suất): ngưỡng tối ưu tìm trên điểm out-of-fold của train.
    # valid: các đơn train có điểm OOF. Khối đầu tiên của TimeSeriesSplit chỉ dùng để
    # huấn luyện nên không có điểm OOF.
    valid = ~np.isnan(oof[names[0]])

    def policy(name, s, value, cost=COST):
        """Trả về mảng bool: đơn nào được flag theo chính sách của model `name`."""
        if name in FIXED_RULES:
            return s >= 0.5
        if name == "Majority class":
            return np.ones(len(s), bool)
        if name in NON_PROBABILISTIC:
            return s >= best_threshold(oof[name][valid], y_tr[valid], v_tr[valid], cost)
        return ev_flag(s, value, cost)

    # Đánh giá các chính sách trên dữ liệu OOF của train (để so sánh, chưa đụng tới test).
    oof_policy = pd.DataFrame([
        {"model": n, **evaluate(oof[n][valid], y_tr[valid], v_tr[valid], COST, ALERT_RATE,
                                policy(n, oof[n][valid], v_tr[valid]))} for n in names
    ]).set_index("model")
    oof_policy.to_csv(TAB / "oof_policy_results.csv")
    print(oof_policy[["alert_rate@policy", "recall@policy", "precision@policy",
                      "saving_per_order@policy", "brier"]].round(4).to_string())

    # ------------------------------------------------------------ 5. test
    # Huấn luyện lại mỗi model trên TOÀN BỘ train, rồi đánh giá MỘT LẦN trên tập test.
    # Tập test không được dùng cho bất kỳ lựa chọn nào trước đó (ngưỡng, siêu tham số, cut-off).
    section("5. Final fit on full training period, evaluation on held-out test period")
    fitted, test_scores, test_rows = {}, {}, []
    # Bootstrap: lấy mẫu có hoàn lại 200 lần trên tập test để ước lượng khoảng tin cậy 95%.
    # Mọi model dùng CHUNG các mẫu bootstrap -> so sánh công bằng.
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]
    for name, model in models.items():
        m = clone(model).fit(X_tr, y_tr)
        fitted[name] = m
        s = scores(m, X_te)
        test_scores[name] = s
        flag = policy(name, s, v_te)
        res = evaluate(s, y_te, v_te, COST, ALERT_RATE, flag)
        boot = [evaluate(s[b], y_te[b], v_te[b], COST, ALERT_RATE, flag[b]) for b in boot_idx]
        for key in ("pr_auc", "pr_auc_tiebroken", "roc_auc", "saving_per_order@policy"):
            # Khoảng tin cậy 95% = phân vị 2.5% và 97.5% của phân phối bootstrap.
            lo, hi = np.percentile([b[key] for b in boot], [2.5, 97.5])
            res[f"{key}_ci_low"], res[f"{key}_ci_high"] = lo, hi
        test_rows.append({"model": name, **res})
    test_tab = pd.DataFrame(test_rows).set_index("model")
    test_tab.to_csv(TAB / "test_results.csv")
    print(test_tab.round(4).to_string())

    # Chọn "mô hình tốt nhất" theo PR-AUC trung bình trên CV (KHÔNG theo kết quả test,
    # nếu không sẽ là chọn mô hình dựa trên tập test).
    candidates = [n for n in names if n not in BASELINES]
    best = max(candidates, key=lambda n: cv.loc[n, "pr_auc_tiebroken_mean"])
    summary["best_model_by_cv"] = best
    print(f"\nBest model by mean CV PR-AUC (tie-broken): {best}")

    # Mô hình có thêm giá trị gì so với luật hay không? Chỉ có thể thấy được khi xét
    # BÊN TRONG từng shipping mode, vì giữa các mode thì luật đã phân biệt được rồi.
    # ROC-AUC khoảng 0.5 nghĩa là trong mode đó mô hình không hơn đoán mò.
    # (First Class trễ 100% nên không tính được AUC -> NaN.)
    sub_rows = []
    for mode in sorted(test["shipping_mode"].unique()):
        mask = (test["shipping_mode"] == mode).to_numpy()
        row = {"shipping_mode": mode, "orders": int(mask.sum()), "late_share": y_te[mask].mean()}
        for n in ["Logistic regression", "Decision tree", "Random forest", "Gradient boosting",
                  "SVM (RBF, Nystroem)", "Neural network (MLP)"]:
            if len(np.unique(y_te[mask])) > 1:
                row[f"roc_auc | {n}"] = evaluate(test_scores[n][mask], y_te[mask], v_te[mask],
                                                 COST, ALERT_RATE)["roc_auc"]
            else:
                row[f"roc_auc | {n}"] = np.nan
        sub_rows.append(row)
    within = pd.DataFrame(sub_rows)
    within.to_csv(TAB / "test_within_mode_auc.csv", index=False)
    print("\nWithin-mode ROC-AUC on test (0.5 = no better than chance):\n"
          + within.round(3).to_string(index=False))

    # Kiểm tra drift: chia test thành 2 giai đoạn. Từ 10/2017 danh mục sản phẩm thu hẹp
    # và giá trị đơn giảm mạnh; xem chất lượng mô hình và số tiền tiết kiệm có giữ được không.
    drift_rows = []
    late_start = pd.Timestamp("2017-10-01")
    for label, mask in [("2017-07..09", test["order_dt"] < late_start),
                        ("2017-10..2018-01", test["order_dt"] >= late_start)]:
        mask = mask.to_numpy()
        for n in [best, LOOKUP, RULE]:
            s = test_scores[n][mask]
            drift_rows.append({"period": label, "model": n, "orders": int(mask.sum()),
                               "median_order_value": float(np.median(v_te[mask])),
                               **evaluate(s, y_te[mask], v_te[mask], COST, ALERT_RATE,
                                          policy(n, s, v_te[mask]))})
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(TAB / "test_drift_check.csv", index=False)
    print("\nDrift check:\n" + drift[["period", "model", "orders", "median_order_value", "pr_auc", "roc_auc",
                                      "pr_auc_tiebroken", "saving_per_order@policy"]].round(4).to_string(index=False))

    # Hình 4: đường Precision-Recall trên tập test cho mọi phương pháp.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n in names:
        if n == "Majority class":
            continue
        p, r, _ = precision_recall_curve(y_te, test_scores[n])
        ax.plot(r, p, label=f"{n} (AP={test_tab.loc[n, 'pr_auc_tiebroken']:.3f})",
                ls="--" if n in BASELINES else "-")
    ax.axhline(y_te.mean(), color="grey", ls=":", label=f"Majority class (AP={y_te.mean():.3f})")
    ax.set_xlabel("Recall (share of late orders caught)")
    ax.set_ylabel("Precision (share of flagged orders that are late)")
    ax.set_title("Precision-recall curves, test period 2017-07 to 2018-01")
    ax.legend(fontsize=7, loc="lower left")
    savefig("04_test_pr_curves")

    # Hình 5: tiền tiết kiệm mỗi tuần theo tỷ lệ flag. Trả lời câu hỏi trong brief §4:
    # "ở tỷ lệ cảnh báo nào thì chi phí kiểm tra vượt quá lợi ích bắt được đơn trễ?"
    # Đơn được xếp hạng theo thiệt hại kỳ vọng P(trễ) x giá trị đơn.
    rates = np.linspace(0, 1, 101)
    curve = pd.DataFrame({"alert_rate": rates})
    for n in [best, LOOKUP, RULE, "Rule: First/Second Class"]:
        curve[n] = [COST.saving(flag_top(test_scores[n] * v_te, r), y_te, v_te) / test_weeks for r in rates]
    curve.to_csv(TAB / "saving_vs_alert_rate.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for n in curve.columns[1:]:
        ax.plot(100 * rates, curve[n], label=n, ls="--" if n in BASELINES else "-")
    # Đường dọc: tỷ lệ flag của luật First/Second Class, để tham chiếu.
    rule_rate = (test["shipping_mode"].isin(["First Class", "Second Class"])).mean()
    ax.axvline(100 * rule_rate, color="grey", ls=":", lw=1)
    ax.set_xlabel("Alert rate (% of orders flagged)")
    ax.set_ylabel("Net saving vs. no action (USD per week)")
    ax.set_title("Weekly net saving by alert rate, test period\n"
                 f"(orders ranked by P(late) x value; ${COST.action_cost:.0f}/flag, "
                 f"miss = {COST.miss_fraction:.0%} of value, "
                 f"{COST.effectiveness:.0%} avoided)")
    ax.legend(fontsize=8)
    savefig("05_saving_vs_alert_rate")

    # ------------------------------------------------------------ 6. interpretation
    # Permutation importance: xáo trộn ngẫu nhiên một feature trên tập test rồi đo PR-AUC
    # giảm bao nhiêu. Giảm nhiều = mô hình phụ thuộc nhiều vào feature đó. Cách này áp dụng
    # được cho mọi model và tính trên CỘT GỐC (trước one-hot) nên dễ đọc.
    section("6. Interpretation")
    perm_rows = []
    # dict.fromkeys(...) loại bỏ tên trùng nhưng giữ nguyên thứ tự (khi best là Decision tree).
    for n in dict.fromkeys([best, "Decision tree", "Gradient boosting"]):
        pi = permutation_importance(fitted[n], X_te, y_te, scoring="average_precision",
                                    n_repeats=5, random_state=SEED, n_jobs=1)
        for f, mu, sd in zip(FEATURES, pi.importances_mean, pi.importances_std):
            perm_rows.append({"model": n, "feature": f, "importance_mean": mu, "importance_std": sd})
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(TAB / "permutation_importance.csv", index=False)
    top = perm[perm["model"] == best].sort_values("importance_mean", ascending=False).head(12)
    print(f"Permutation importance ({best}, drop in test PR-AUC):\n"
          + top[["feature", "importance_mean", "importance_std"]].round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    top = top.iloc[::-1]   # đảo ngược để feature quan trọng nhất nằm trên cùng của biểu đồ
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#6baed6")
    ax.set_xlabel("Drop in test PR-AUC when feature is shuffled")
    ax.set_title(f"Permutation importance, {best}")
    savefig("06_permutation_importance")

    # Hình 7: cây quyết định độ sâu 3 để minh hoạ logic của mô hình một cách dễ đọc.
    # set_params(clf__max_depth=3): cú pháp "<tên bước>__<tham số>" của Pipeline.
    shallow = clone(models["Decision tree"]).set_params(clf__max_depth=3).fit(X_tr, y_tr)
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(shallow.named_steps["clf"], feature_names=shallow.named_steps["prep"].get_feature_names_out(),
              class_names=["on time", "late"], filled=True, impurity=False, proportion=True,
              fontsize=7, ax=ax)
    ax.set_title("Decision tree, depth 3 (fitted on training period)")
    savefig("07_decision_tree_depth3")

    # Hình 8: Partial dependence cho biết xác suất trễ trung bình thay đổi thế nào khi
    # một feature thay đổi (các feature khác giữ nguyên). Dùng mẫu 3.000 đơn cho nhanh.
    pdp_feats = ["hour", "order_value", "citymode_prior_late_rate", "mean_discount_rate"]
    fig, axes = plt.subplots(1, len(pdp_feats), figsize=(13, 3.4), sharey=True)
    sample = X_te.sample(n=min(3000, len(X_te)), random_state=SEED)
    PartialDependenceDisplay.from_estimator(fitted[best], sample, pdp_feats, ax=axes,
                                            grid_resolution=20, random_state=SEED)
    for a in np.ravel(axes):
        a.set_ylabel("Partial dependence: P(late)")
    fig.suptitle(f"Partial dependence, {best} (test sample of 3,000 orders)")
    savefig("08_partial_dependence")

    # ------------------------------------------------------------ 7. recommendation
    # Đổi kết quả sang đơn vị người quản lý hiểu được: mỗi tuần flag bao nhiêu đơn,
    # bắt được bao nhiêu đơn trễ, bao nhiêu báo động giả, tiết kiệm bao nhiêu USD.
    section("7. Recommendation figures and cost sensitivity")
    rec_rows = []
    for n in [best, LOOKUP, RULE, "Rule: First/Second Class", "Majority class"]:
        flag = policy(n, test_scores[n], v_te)
        rec_rows.append({
            "policy": n + (" (flag all)" if n == "Majority class" else ""),
            "alert_rate": flag.mean(),
            "orders_per_week": len(test) / test_weeks,
            "flagged_per_week": flag.sum() / test_weeks,
            "late_caught_per_week": (flag & (y_te == 1)).sum() / test_weeks,
            "late_missed_per_week": (~flag & (y_te == 1)).sum() / test_weeks,
            "false_alarms_per_week": (flag & (y_te == 0)).sum() / test_weeks,
            "net_saving_usd_per_week": COST.saving(flag, y_te, v_te) / test_weeks,
        })
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(TAB / "recommendation.csv", index=False)
    print(rec.round(1).to_string(index=False))

    # Phân tích độ nhạy: 4 x 3 x 3 = 36 bộ giả định chi phí. Kiểm tra xem kết luận
    # ("mô hình không hơn bảng tra") có đúng với mọi bộ giả định hay chỉ với bộ mặc định.
    sens_rows = []
    for ac in [2.0, 5.0, 10.0, 20.0]:
        for mf in [0.05, 0.10, 0.20]:
            for ef in [0.3, 0.5, 0.7]:
                c = CostModel(ac, mf, ef)
                row = {"action_cost": ac, "miss_fraction": mf, "effectiveness": ef}
                for n, tag in [(best, "model"), (LOOKUP, "lookup"), (RULE, "rule")]:
                    flag = policy(n, test_scores[n], v_te, c)
                    row[f"{tag}_alert_rate"] = flag.mean()
                    row[f"{tag}_saving_per_week"] = c.saving(flag, y_te, v_te) / test_weeks
                row["flag_all_saving_per_week"] = c.saving(np.ones(len(y_te), bool), y_te, v_te) / test_weeks
                row["model_minus_lookup_per_week"] = row["model_saving_per_week"] - row["lookup_saving_per_week"]
                row["model_minus_rule_per_week"] = row["model_saving_per_week"] - row["rule_saving_per_week"]
                sens_rows.append(row)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(TAB / "cost_sensitivity.csv", index=False)
    print(sens.round(2).to_string(index=False))

    # Kịch bản ĐỔI NGÀY HỨA trên tập test: thay vì (hoặc trước khi) flag từng đơn, ta sửa
    # chính lời hứa giao hàng. Đơn trễ = số ngày thực tế > số ngày hứa.
    # Giờ cut-off lấy từ luật đã học trên train, không học lại trên test.
    cutoff = fitted[RULE].cutoff_hour_
    summary["same_day_cutoff_hour_learned_on_train"] = cutoff
    mode, real = test["shipping_mode"], test["outcome_real_days"]
    after_cutoff = (mode == "Same Day") & (test["hour"] >= cutoff)

    def promise(first=1, second=2, standard=4, same_day_cutoff=False):
        """Số ngày hứa cho từng đơn theo một kịch bản. Mặc định = lời hứa hiện tại.
        same_day_cutoff=True: đơn Same Day đặt sau giờ cut-off được hứa giao ngày hôm sau (1 ngày)."""
        p = mode.map({"First Class": first, "Second Class": second,
                      "Standard Class": standard, "Same Day": 0}).astype(float)
        if same_day_cutoff:
            p[after_cutoff] = 1
        return p

    scenarios = {
        "Current promises (0/1/2/4 days)": promise(),
        f"A: Same Day after {cutoff}:00 promised next day": promise(same_day_cutoff=True),
        "B: A + First Class promised 2 days": promise(first=2, same_day_cutoff=True),
        "C: B + Second Class promised 4 days": promise(first=2, second=4, same_day_cutoff=True),
        "D: C + Second/Standard promised 6 days": promise(first=2, second=6, standard=6,
                                                          same_day_cutoff=True),
    }
    # Kịch bản cộng dồn A -> D. Đánh đổi: hứa dài hơn thì ít trễ hơn, nhưng có thể mất khách
    # (tác động đó KHÔNG có trong dữ liệu, cần nêu trong phần Limitations).
    scen_rows = []
    for label, p in scenarios.items():
        late = (real > p).to_numpy()
        scen_rows.append({"scenario": label, "late_share": late.mean(),
                          "late_orders_per_week": late.sum() / test_weeks,
                          "mean_promised_days": p.mean(),
                          "late_cost_usd_per_week": (COST.miss_fraction * v_te[late]).sum() / test_weeks})
    scen = pd.DataFrame(scen_rows)
    scen.to_csv(TAB / "promise_scenarios_test.csv", index=False)
    print("\nPromise scenarios (test period):\n" + scen.round(2).to_string(index=False))

    # Ghi các con số chính ra summary.json để dễ trích dẫn vào báo cáo.
    summary.update({
        "cv_pr_auc_tiebroken_mean_sd": {n: [round(cv.loc[n, "pr_auc_tiebroken_mean"], 4),
                                            round(cv.loc[n, "pr_auc_tiebroken_std"], 4)] for n in names},
        "test_pr_auc_tiebroken": {n: round(test_tab.loc[n, "pr_auc_tiebroken"], 4) for n in names},
        "test_saving_usd_per_week": {r["policy"]: round(r["net_saving_usd_per_week"], 1)
                                     for r in rec_rows},
        "runtime_s": round(time.time() - t0, 1),
    })
    (ROOT / "outputs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {summary['runtime_s']}s. Outputs in {ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
