"""Định nghĩa mô hình, baseline, các chỉ số đánh giá (metric) và mô hình chi phí.

Cấu trúc file:
    1. Hai baseline tự viết: ShippingModeRule (luật) và ModeLateRateTable (bảng tra)
    2. make_preprocessor(): toàn bộ tiền xử lý đặt trong ColumnTransformer
    3. make_models(): 4 baseline + 6 mô hình học máy
    4. CostModel: quy đổi dự đoán thành tiền
    5. Các hàm chính sách (policy) và metric: flag_top, best_threshold, ev_flag,
       tiebroken_ap, evaluate
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data_prep import CAT_HIGH, CAT_LOW, NUMERIC

SEED = 42                                   # cố định seed để kết quả tái lập được
RULE_MODES = ("First Class", "Second Class")  # hai mode mà "luật hiện hành" flag


class ShippingModeRule(ClassifierMixin, BaseEstimator):
    """Baseline dạng LUẬT (mô phỏng "cách làm hiện tại" của doanh nghiệp).

    - Mặc định: flag mọi đơn First Class hoặc Second Class.
    - Nếu same_day_cutoff=True: flag thêm các đơn Same Day đặt từ "giờ cut-off" trở đi.
      Giờ cut-off được HỌC trong fit() từ dữ liệu train: là giờ nhỏ nhất mà tỷ lệ trễ
      của đơn Same Day vượt 50%. Học trong fit() (chứ không gán cứng 12) để trong
      cross-validation, mỗi fold chỉ dùng dữ liệu của chính fold đó -> không leakage.

    Kế thừa ClassifierMixin + BaseEstimator để dùng được như một model sklearn
    (clone, fit, predict_proba...) trong cùng vòng lặp với các model khác.
    """

    def __init__(self, same_day_cutoff: bool = False):
        # Quy ước sklearn: __init__ chỉ lưu tham số, không tính toán gì.
        self.same_day_cutoff = same_day_cutoff

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])   # sklearn yêu cầu thuộc tính classes_
        self.cutoff_hour_ = None           # dấu "_" ở cuối = thuộc tính học được khi fit
        if self.same_day_cutoff:
            sd = (X["shipping_mode"] == "Same Day").to_numpy()        # mask các đơn Same Day
            # Tỷ lệ trễ của đơn Same Day theo từng giờ đặt hàng.
            late_by_hour = pd.Series(np.asarray(y)[sd]).groupby(X["hour"].to_numpy()[sd]).mean()
            above = late_by_hour[late_by_hour > 0.5]
            # Giờ cut-off = giờ sớm nhất mà đa số đơn bị trễ.
            self.cutoff_hour_ = int(above.index.min()) if len(above) else None
        return self

    def predict_proba(self, X):
        # Luật chỉ cho điểm 0 hoặc 1 (không phải xác suất thật).
        flag = X["shipping_mode"].isin(RULE_MODES)
        if self.cutoff_hour_ is not None:
            flag |= (X["shipping_mode"] == "Same Day") & (X["hour"] >= self.cutoff_hour_)
        p = flag.astype(float).to_numpy()
        # Trả về 2 cột [P(không trễ), P(trễ)] đúng chuẩn sklearn.
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ModeLateRateTable(ShippingModeRule):
    """Baseline dạng BẢNG TRA xác suất (không phải mô hình học máy).

    Chia đơn thành 5 "ô": First, Second, Standard, Same Day (trước cut-off),
    Same Day (sau cut-off). Xác suất trễ dự đoán = tỷ lệ trễ lịch sử của ô đó
    trong dữ liệu train.

    Đây là baseline QUAN TRỌNG NHẤT của project: nếu các mô hình ML không vượt được
    bảng tra đơn giản này thì chúng không mang lại giá trị gì thêm.
    """

    def __init__(self):
        super().__init__(same_day_cutoff=True)

    def _cell(self, X):
        """Gán mỗi đơn vào một ô của bảng tra."""
        cell = X["shipping_mode"].astype(str).to_numpy().copy()
        if self.cutoff_hour_ is not None:
            late_sd = ((X["shipping_mode"] == "Same Day") & (X["hour"] >= self.cutoff_hour_)).to_numpy()
            cell[late_sd] = "Same Day (after cut-off)"
        return cell

    def fit(self, X, y):
        super().fit(X, y)                  # học giờ cut-off trước (từ lớp cha)
        # Tỷ lệ trễ trung bình của từng ô.
        rates = pd.Series(np.asarray(y, float)).groupby(self._cell(X)).mean()
        self.rates_ = rates.to_dict()
        self.default_ = float(np.mean(y))  # dự phòng cho ô chưa từng xuất hiện trong train
        return self

    def predict_proba(self, X):
        p = np.array([self.rates_.get(c, self.default_) for c in self._cell(X)])
        return np.column_stack([1 - p, p])


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Toàn bộ bước tiền xử lý "có học" (fitted) nằm ở đây.

    Vì nằm trong Pipeline, bộ tiền xử lý được fit LẠI trên phần train của TỪNG fold:
    median dùng để điền giá trị thiếu, mean/std của scaler, bảng target encoding...
    đều không bao giờ nhìn thấy dữ liệu validation/test. Đây chính là yêu cầu
    "scaler/imputer không được fit trên toàn bộ dữ liệu" trong brief §11.

    scale=False dùng cho mô hình cây: cây không bị ảnh hưởng bởi thang đo, và khi
    không chuẩn hoá thì ngưỡng trên hình vẽ cây vẫn đọc được (ví dụ "hour <= 11.5"
    thay vì "hour <= -0.02").
    """
    # Nhánh cho biến số: điền giá trị thiếu bằng median, rồi (tuỳ chọn) chuẩn hoá về
    # mean 0, std 1. SVM, MLP, Logistic regression cần chuẩn hoá.
    num = (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) if scale
           else SimpleImputer(strategy="median"))
    return ColumnTransformer([
        # One-hot cho biến phân loại ít giá trị. Giá trị xuất hiện < 20 lần được gộp
        # vào nhóm "infrequent"; giá trị lạ ở tập test cũng rơi vào nhóm này thay vì báo lỗi.
        ("cat_low", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  sparse_output=False), CAT_LOW),
        # Target encoding cho biến NHIỀU giá trị: thay mỗi thành phố/quốc gia/ngành hàng
        # bằng tỷ lệ trễ trung bình (đã làm mượt) của nó.
        # Điểm quan trọng: TargetEncoder dùng cross-fitting nội bộ (cv=KFold 5 phần):
        # mã hoá của một dòng train được tính từ 4 phần còn lại, nên KHÔNG dùng nhãn
        # của chính dòng đó -> tránh leakage từ target.
        ("cat_high", TargetEncoder(target_type="binary",
                                   cv=KFold(5, shuffle=True, random_state=SEED)), CAT_HIGH),
        ("num", num, NUMERIC),
    ], verbose_feature_names_out=False)


def make_models() -> dict[str, object]:
    """Trả về dict {tên: estimator} gồm 4 baseline và 6 mô hình.

    Brief yêu cầu ít nhất 3 họ mô hình, trong đó ít nhất 2 họ được dạy ở tuần 6–13.
    Ở đây có: cây quyết định (Decision tree, Random forest, Gradient boosting),
    SVM, mạng nơ-ron (MLP), cộng thêm Logistic regression làm mô hình tham chiếu.
    Baseline không được tính là một họ mô hình.
    """
    def pipe(clf, scale=True):
        # Ghép tiền xử lý + bộ phân loại thành một Pipeline duy nhất.
        return Pipeline([("prep", make_preprocessor(scale)), ("clf", clf)])

    return {
        # --- Baseline -------------------------------------------------------
        # Dự đoán tỷ lệ lớp chung cho mọi đơn -> không xếp hạng được gì (ROC-AUC = 0.5).
        "Majority class": DummyClassifier(strategy="prior"),
        "Rule: First/Second Class": ShippingModeRule(),
        "Rule: + Same Day cut-off": ShippingModeRule(same_day_cutoff=True),
        "Lookup: late rate by mode": ModeLateRateTable(),
        # --- Mô hình ---------------------------------------------------------
        # Mô hình tuyến tính, dễ giải thích, dùng làm mốc tham chiếu.
        "Logistic regression": pipe(LogisticRegression(max_iter=2000, C=1.0)),
        # Cây quyết định: giới hạn độ sâu 6 và mỗi lá tối thiểu 50 đơn để tránh overfit.
        # (Bảng tree_depth_sweep.csv cho thấy ảnh hưởng của độ sâu.)
        "Decision tree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                     random_state=SEED), scale=False),
        # Random forest: 300 cây, mỗi lần tách chỉ xét sqrt(số feature) -> giảm phương sai.
        "Random forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                     max_features="sqrt", n_jobs=-1,
                                                     random_state=SEED), scale=False),
        # Gradient boosting (dạng histogram, nhanh): các cây nông được xây nối tiếp nhau,
        # cây sau sửa lỗi của cây trước. early_stopping tự dừng khi không cải thiện nữa.
        "Gradient boosting": pipe(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                 max_leaf_nodes=31,
                                                                 early_stopping=True,
                                                                 random_state=SEED), scale=False),
        # SVM với kernel RBF. SVC chính xác có độ phức tạp khoảng O(n^2) nên không khả thi
        # với khoảng 50 nghìn đơn. Thay vào đó dùng Nystroem để XẤP XỈ không gian đặc trưng
        # của kernel RBF (600 thành phần), rồi huấn luyện SVM tuyến tính trên đó.
        # Lưu ý: LinearSVC trả về "khoảng cách tới siêu phẳng" chứ không phải xác suất.
        "SVM (RBF, Nystroem)": pipe(make_pipeline(
            Nystroem(kernel="rbf", gamma=0.02, n_components=600, random_state=SEED),
            LinearSVC(C=0.1, random_state=SEED))),
        # Mạng nơ-ron 2 lớp ẩn (64 và 32 nơ-ron). alpha = hệ số phạt L2;
        # early_stopping giữ lại 10% dữ liệu train để dừng sớm khi bắt đầu overfit.
        "Neural network (MLP)": pipe(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                   learning_rate_init=1e-3, max_iter=200,
                                                   early_stopping=True, random_state=SEED)),
    }


# Nhóm tên dùng trong run_all.py.
BASELINES = ("Majority class", "Rule: First/Second Class", "Rule: + Same Day cut-off",
             "Lookup: late rate by mode")
# Luật cố định: flag đúng những đơn mà luật chọn, không có ngưỡng nào để chỉnh.
FIXED_RULES = ("Rule: First/Second Class", "Rule: + Same Day cut-off")
# Mô hình có điểm KHÔNG phải xác suất -> không dùng được chính sách kỳ vọng (ev_flag);
# thay vào đó chọn ngưỡng tối ưu trên điểm out-of-fold của tập train.
NON_PROBABILISTIC = ("SVM (RBF, Nystroem)",)


def scores(model, X) -> np.ndarray:
    """Lấy điểm rủi ro của mỗi đơn: xác suất trễ nếu model có predict_proba,
    nếu không thì lấy decision_function (trường hợp SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ---------------------------------------------------------------- mô hình chi phí
@dataclass(frozen=True)
class CostModel:
    """Quy đổi quyết định flag / không flag thành tiền (USD). Đây là các GIẢ ĐỊNH,
    và run_all.py có phân tích độ nhạy (cost_sensitivity.csv) cho nhiều bộ giá trị.

    action_cost   : chi phí xử lý MỘT đơn bị flag (ưu tiên đóng gói, nâng hạng vận
                    chuyển, gọi điện báo trước cho khách). Mặc định $10.
    miss_fraction : thiệt hại của một đơn trễ KHÔNG được xử lý, tính bằng % giá trị đơn
                    (bồi thường thiện chí, khách liên hệ lại, nguy cơ mất khách). Mặc định 10%.
    effectiveness : tỷ lệ thiệt hại tránh được nếu đơn trễ đã được flag trước. Mặc định 50%.

    Với mỗi đơn:
        flag,  đơn không trễ (báo động giả) : tốn action_cost
        flag,  đơn trễ (bắt đúng)           : action_cost + (1 - effectiveness) * thiệt hại
        không flag, đơn trễ (bỏ sót)        : toàn bộ thiệt hại
        không flag, đơn không trễ           : 0
    """
    action_cost: float = 10.0
    miss_fraction: float = 0.10
    effectiveness: float = 0.5

    def total_cost(self, flag, y, value) -> float:
        """Tổng chi phí của một chính sách flag trên một tập đơn hàng."""
        flag = np.asarray(flag, bool)
        y = np.asarray(y, bool)
        loss = self.miss_fraction * np.asarray(value, float)   # thiệt hại nếu đơn trễ bị bỏ sót
        return (flag.sum() * self.action_cost                    # chi phí xử lý các đơn bị flag
                + loss[y & ~flag].sum()                          # đơn trễ bị bỏ sót
                + (1 - self.effectiveness) * loss[y & flag].sum())  # đơn trễ đã bắt, chỉ còn một phần thiệt hại

    def saving(self, flag, y, value) -> float:
        """Số tiền tiết kiệm so với "không làm gì" (không flag đơn nào). Số dương = có lợi."""
        return self.total_cost(np.zeros(len(y), bool), y, value) - self.total_cost(flag, y, value)


def flag_top(s: np.ndarray, rate: float, tiebreak=None) -> np.ndarray:
    """Chính sách "năng lực cố định": flag `rate` (ví dụ 35%) số đơn có điểm cao nhất.

    Mô phỏng tình huống bộ phận CSKH chỉ xử lý được một số đơn nhất định mỗi tuần.
    Khi nhiều đơn có điểm BẰNG NHAU (luật, bảng tra), ưu tiên đơn có `tiebreak` lớn
    hơn (ví dụ giá trị đơn cao hơn); nếu không có tiebreak thì theo thứ tự dòng.
    """
    k = int(round(rate * len(s)))       # số đơn được flag
    flag = np.zeros(len(s), bool)
    if k > 0:
        # np.lexsort sắp xếp theo khoá CUỐI CÙNG trước: ở đây là -s (điểm giảm dần),
        # sau đó mới tới -tiebreak.
        keys = (-s,) if tiebreak is None else (-np.asarray(tiebreak, float), -s)
        flag[np.lexsort(keys)[:k]] = True
    return flag


def best_threshold(s, y, value, cost: CostModel, grid=None) -> float:
    """Tìm ngưỡng điểm cho số tiền tiết kiệm lớn nhất.

    Chỉ được gọi trên điểm OUT-OF-FOLD của tập train (không bao giờ trên tập test),
    để ngưỡng được chọn mà không nhìn thấy dữ liệu test.
    Lưới thử là 201 phân vị của điểm số.
    """
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 201))) if grid is None else grid
    savings = [cost.saving(s >= t, y, value) for t in grid]
    return float(grid[int(np.argmax(savings))])


def ev_flag(p, value, cost: CostModel) -> np.ndarray:
    """Chính sách GIÁ TRỊ KỲ VỌNG (expected value) cho mô hình có xác suất.

    Flag một đơn khi thiệt hại kỳ vọng tránh được lớn hơn chi phí xử lý:
        P(trễ) * effectiveness * miss_fraction * giá trị đơn  >  action_cost

    Ưu điểm: không cần dò ngưỡng. Đơn giá trị thấp tự động không được flag, vì chi phí
    xử lý cao hơn lợi ích. Chính sách này đòi hỏi xác suất phải được hiệu chỉnh tốt
    (calibrated); Brier score trong bảng kết quả giúp kiểm tra điều đó.
    """
    return np.asarray(p) * cost.effectiveness * cost.miss_fraction * np.asarray(value) > cost.action_cost


def tiebroken_ap(y, s, n_rep: int = 5) -> float:
    """Average precision (PR-AUC), với các điểm BẰNG NHAU được xếp thứ tự ngẫu nhiên
    (lấy trung bình của n_rep lần).

    Vì sao cần hàm này? Hàm average_precision_score của sklearn tính cả một khối điểm
    bằng nhau bằng precision gộp của khối đó. Cách tính này làm THIỆT cho các baseline
    chỉ có vài giá trị điểm (luật, bảng tra) so với mô hình có điểm liên tục. Thực tế
    kiểm tra cho thấy: bảng tra có AP chuẩn 0.805, nhưng khi phá hoà ngẫu nhiên thì AP
    lên 0.859, ngang Gradient boosting. Phá hoà ngẫu nhiên đưa mọi phương pháp về cùng
    một mặt bằng; với điểm liên tục (không có hoà) kết quả không đổi.
    """
    s = np.asarray(s, float)
    if len(np.unique(s)) == len(s):          # không có điểm trùng -> tính như bình thường
        return average_precision_score(y, s)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_rep):
        ranks = np.empty(len(s))
        # Sắp theo điểm s; nếu bằng điểm thì theo một số ngẫu nhiên -> thứ hạng không trùng.
        ranks[np.lexsort((rng.random(len(s)), s))] = np.arange(len(s))
        vals.append(average_precision_score(y, ranks))
    return float(np.mean(vals))


def evaluate(s, y, value, cost: CostModel, alert_rate: float, flag=None) -> dict:
    """Tính toàn bộ metric cho một bộ điểm `s`.

    Nhóm 1, không phụ thuộc ngưỡng (đo khả năng XẾP HẠNG đơn rủi ro):
        pr_auc, pr_auc_tiebroken, roc_auc, brier (chỉ khi s là xác suất)
    Nhóm 2, năng lực cố định (flag top alert_rate % đơn):
        recall@35%, precision@35%, saving_per_order@35%
    Nhóm 3, chỉ tính khi truyền `flag` (kết quả của một chính sách cụ thể):
        alert_rate, recall, precision, f1, accuracy, saving_per_order

    Vì sao không dùng accuracy làm metric chính? Brief §6 và §11: accuracy không phản
    ánh chi phí khác nhau của báo động giả và bỏ sót. Ở đây accuracy chỉ được báo cáo
    kèm các metric khác.
    """
    y = np.asarray(y)
    flag_k = flag_top(s, alert_rate, value)
    tp_k = (flag_k & (y == 1)).sum()               # số đơn trễ bắt được trong top-k
    tag = f"{alert_rate:.0%}"
    out = {
        "pr_auc": average_precision_score(y, s),
        "pr_auc_tiebroken": tiebroken_ap(y, s),
        # ROC-AUC không xác định khi mọi điểm bằng nhau -> gán 0.5 (ngang đoán mò).
        "roc_auc": roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5,
        f"recall@{tag}": tp_k / max(y.sum(), 1),   # max(...,1) để tránh chia cho 0
        f"precision@{tag}": tp_k / max(flag_k.sum(), 1),
        f"saving_per_order@{tag}": cost.saving(flag_k, y, value) / len(y),
        # Brier = sai số bình phương trung bình của xác suất (càng thấp càng tốt).
        # Không tính cho SVM vì điểm của SVM không phải xác suất.
        "brier": brier_score_loss(y, s) if s.min() >= 0 and s.max() <= 1 else np.nan,
    }
    if flag is not None:
        flag = np.asarray(flag, bool)
        out.update({
            "alert_rate@policy": flag.mean(),                                    # % đơn bị flag
            "recall@policy": (flag & (y == 1)).sum() / max(y.sum(), 1),          # % đơn trễ bắt được
            "precision@policy": (flag & (y == 1)).sum() / max(flag.sum(), 1),    # % flag là đúng
            "f1@policy": f1_score(y, flag, zero_division=0),
            "accuracy@policy": (flag == y).mean(),
            "saving_per_order@policy": cost.saving(flag, y, value) / len(y),
        })
    return out
