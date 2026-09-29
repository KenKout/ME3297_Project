"""Đọc, kiểm tra chất lượng (audit) và biến đổi dữ liệu DataCo thành bảng MỖI DÒNG = MỘT ĐƠN HÀNG.

Nguyên tắc quan trọng nhất của file này (bài kiểm tra "data leakage" trong brief §4):
    Mọi đặc trưng (feature) tạo ra ở đây phải là thông tin ĐÃ BIẾT tại thời điểm ra quyết định,
    tức là lúc "order release": khách đã đặt và thanh toán, nhưng hàng CHƯA được gửi đi.

Những cột chỉ biết SAU khi giao hàng (ví dụ số ngày giao thực tế) bị liệt kê trong
`POST_OUTCOME_COLS` và tuyệt đối không được đưa vào mô hình.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Thư mục gốc của project (file này nằm trong src/, nên lùi lên 1 cấp).
ROOT = Path(__file__).resolve().parents[1]
# Đường dẫn tới file CSV gốc; file này do get_data.py giải nén ra.
RAW_CSV = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"

# ---------------------------------------------------------------------------
# Nhóm cột bị loại bỏ và LÝ DO loại bỏ (cần ghi vào báo cáo mục 5).
# ---------------------------------------------------------------------------

# Các cột chỉ biết SAU khi lô hàng đã được giao -> không bao giờ được dùng làm feature.
# Nếu dùng, mô hình sẽ "nhìn thấy đáp án" và cho kết quả đẹp giả tạo.
POST_OUTCOME_COLS = [
    "Days for shipping (real)",       # chính là kết quả: nhãn trễ = (ngày thực tế > ngày hứa)
    "Delivery Status",                # được suy ra trực tiếp từ kết quả giao hàng
    "shipping date (DateOrders)",     # trong ~97% số dòng = ngày đặt + số ngày giao thực tế -> lộ đáp án
    "Order Status",                   # trạng thái cuối cùng, được gán sau khi sự việc xảy ra
]
# Dữ liệu cá nhân (PII), không có giá trị cho mô hình -> bỏ ngay khi đọc.
# (Hiện các cột này đơn giản là không được chọn vào FEATURES bên dưới.)
PII_COLS = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode",
]
# Cột rỗng hoặc gần như rỗng (Product Description rỗng 100%, Order Zipcode rỗng 86%),
# hoặc là đường link ảnh sản phẩm, không mang thông tin vận hành.
USELESS_COLS = ["Product Description", "Order Zipcode", "Product Image"]

# Các thuộc tính ở cấp ĐƠN HÀNG: mọi dòng (order line) thuộc cùng một đơn phải có
# cùng giá trị. Hàm build_orders() sẽ kiểm tra điều này; nếu sai thì báo lỗi ngay,
# vì khi đó việc gộp về một dòng/đơn sẽ làm mất thông tin.
# Key = tên cột gốc trong CSV, value = tên cột mới (ngắn gọn, snake_case).
ORDER_CONSTANT = {
    "Shipping Mode": "shipping_mode",               # phương thức vận chuyển khách chọn
    "Days for shipment (scheduled)": "scheduled_days",  # số ngày đã hứa với khách
    "Market": "market",                             # thị trường đích (5 giá trị)
    "Order Region": "order_region",                 # vùng đích (23 giá trị)
    "Order Country": "order_country",               # quốc gia đích
    "Order City": "order_city",                     # thành phố đích
    "Customer Segment": "customer_segment",         # Consumer / Corporate / Home Office
    "Customer Id": "customer_id",                   # mã khách (chỉ dùng để tính lịch sử)
    "Type": "payment_type",                         # hình thức thanh toán
}

# Tham số cho đặc trưng lịch sử "tỷ lệ trễ trước đây" (xem build_orders):
HISTORY_PRIOR = 0.5      # giá trị "trung lập" khi chưa có lịch sử (50% trễ)
HISTORY_WEIGHT = 5.0     # coi như có sẵn 5 đơn ảo mang tỷ lệ prior -> làm mượt (smoothing)
# Nhãn trễ/không trễ của một đơn chỉ được "biết" 1 ngày SAU khi hàng thực sự tới.
# Dùng để quyết định đơn cũ nào đã có kết quả tại thời điểm đơn mới được đặt.
LABEL_DELAY = pd.Timedelta(days=1)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Đọc file CSV gốc và chuẩn hoá cơ bản.

    - encoding="latin-1": file gốc có ký tự đặc biệt (tên thành phố tiếng Tây Ban Nha...),
      đọc bằng UTF-8 mặc định sẽ bị lỗi.
    - Tên cột gốc có khoảng trắng thừa ở cuối (ví dụ "Days for shipping (real)     "),
      nên cần .str.strip() để gọi tên cột cho đúng.
    - Tạo sẵn 2 cột thời gian kiểu datetime để các bước sau tính toán.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python get_data.py` first.")
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"])        # thời điểm đặt hàng
    df["shipping_dt"] = pd.to_datetime(df["shipping date (DateOrders)"])  # chỉ dùng để audit leakage
    return df


def audit(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tạo các bảng đánh giá chất lượng dữ liệu cho báo cáo mục 4 (Data).

    Trả về dict {tên bảng: DataFrame}; run_all.py sẽ lưu mỗi bảng thành
    outputs/tables/audit_<tên>.csv.
    """
    out: dict[str, pd.DataFrame] = {}

    # 1) Kích thước dữ liệu: số dòng, số cột, số đơn, số khách, số sản phẩm, khoảng thời gian.
    #    (df.shape[1] - 2 vì ta đã thêm 2 cột order_dt, shipping_dt trong load_raw.)
    out["shape"] = pd.DataFrame({
        "item": ["rows (order lines)", "columns", "orders", "customers", "products",
                 "first order", "last order"],
        "value": [len(df), df.shape[1] - 2, df["Order Id"].nunique(), df["Customer Id"].nunique(),
                  df["Product Card Id"].nunique(), df["order_dt"].min(), df["order_dt"].max()],
    })

    # 2) Giá trị thiếu: chỉ liệt kê các cột có ít nhất 1 giá trị thiếu, kèm phần trăm.
    miss = df.drop(columns=["order_dt", "shipping_dt"]).isna().sum()
    out["missing"] = (miss[miss > 0].rename("n_missing").to_frame()
                      .assign(pct=lambda t: (100 * t["n_missing"] / len(df)).round(2)))

    # 3) Ngoại lai (outlier) theo quy tắc IQR: giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    #    Ta chỉ ĐẾM để báo cáo, không xoá: giá trị lớn (đơn hàng đắt, lỗ nặng) vẫn là
    #    dữ liệu thật và có ý nghĩa kinh doanh.
    num_cols = ["Sales", "Order Item Quantity", "Order Item Discount Rate",
                "Order Item Product Price", "Benefit per order"]
    rows = []
    for c in num_cols:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])   # tứ phân vị thứ nhất và thứ ba
        iqr = q3 - q1                        # khoảng tứ phân vị
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append({"column": c, "min": s.min(), "median": s.median(), "max": s.max(),
                     "iqr_outliers": n_out, "pct_outliers": round(100 * n_out / len(s), 2)})
    out["outliers"] = pd.DataFrame(rows)

    # 4) Phân phối trạng thái giao hàng (bao gồm cả đơn bị huỷ).
    out["delivery_status"] = df["Delivery Status"].value_counts().rename("n_lines").to_frame()

    # 5) Tỷ lệ trễ theo phương thức vận chuyển (đã bỏ đơn huỷ). Đây là bảng cho thấy
    #    shipping mode gần như quyết định nhãn (First Class trễ 100%).
    out["late_by_mode"] = (df[df["Delivery Status"] != "Shipping canceled"]
                           .groupby("Shipping Mode")
                           .agg(n_lines=("Late_delivery_risk", "size"),
                                late_rate=("Late_delivery_risk", "mean"),
                                scheduled_days=("Days for shipment (scheduled)", "first")))

    # 6) Bằng chứng về leakage:
    #    - gap = số ngày giữa "shipping date" và ngày đặt. Nếu gap == số ngày giao thực tế
    #      thì cột "shipping date" thực chất là NGÀY GIAO TỚI, tức là chứa đáp án.
    #    - label_rule: kiểm tra nhãn Late_delivery_risk có đúng bằng (thực tế > hứa) không.
    gap = (df["shipping_dt"] - df["order_dt"]).dt.days
    label_rule = df["Days for shipping (real)"] > df["Days for shipment (scheduled)"]
    not_cancel = df["Delivery Status"] != "Shipping canceled"
    out["leakage_checks"] = pd.DataFrame({
        "check": ["shipping date - order date == real shipping days",
                  "label == (real days > scheduled days), non-cancelled lines"],
        "share_of_rows": [(gap == df["Days for shipping (real)"]).mean(),
                          (label_rule[not_cancel] == df.loc[not_cancel, "Late_delivery_risk"].astype(bool)).mean()],
    })

    # 7) Kiểm tra tính "thật" của dữ liệu: kết quả giao hàng có phải là một HÀM TẤT ĐỊNH
    #    của những trường không có ý nghĩa vận hành (như mã đơn hàng) hay không?
    #    Nếu tỷ lệ khớp = 100% thì kết quả gần như chắc chắn được sinh bằng thuật toán,
    #    và ta phải nói rõ điều này trong phần Limitations của báo cáo.
    o = df[not_cancel].drop_duplicates("Order Id")          # mỗi đơn lấy 1 dòng
    std_sec = o["Shipping Mode"].isin(["Standard Class", "Second Class"])
    same_day = o["Shipping Mode"] == "Same Day"
    first = o["Shipping Mode"] == "First Class"
    # Khoảng cách (phút) giữa hai đơn liên tiếp theo thời gian.
    gaps = o.sort_values("order_dt")["order_dt"].diff().dt.total_seconds().div(60).dropna()
    out["realism_checks"] = pd.DataFrame({
        "check": [
            "Standard/Second Class: real days == (Order Id - 1) mod 5 + 2",
            "First Class: real days == 2",
            "Same Day: late == (order hour >= 12)",
            "Gap between consecutive orders is a multiple of 21 minutes",
        ],
        "share_of_orders": [
            # Số ngày giao lặp lại theo chu kỳ 2,3,4,5,6 theo mã đơn -> dấu hiệu dữ liệu giả lập.
            ((o["Order Id"] - 1) % 5 + 2 == o["Days for shipping (real)"])[std_sec].mean(),
            # First Class luôn giao đúng 2 ngày, trong khi hứa 1 ngày -> luôn trễ.
            (o["Days for shipping (real)"] == 2)[first].mean(),
            # Same Day: đặt sau 12 giờ trưa thì trễ, trước 12 giờ thì đúng hạn.
            ((o["order_dt"].dt.hour >= 12).astype(int) == o["Late_delivery_risk"])[same_day].mean(),
            # Thời điểm đặt hàng cách nhau đúng bội số của 21 phút -> không giống hành vi thật.
            (gaps.round() % 21 == 0).mean(),
        ],
    })
    return out


def _prior_history(times: np.ndarray, avail: np.ndarray, y: np.ndarray, keys: np.ndarray):
    """Tính lịch sử "không nhìn trước tương lai" cho từng đơn hàng.

    Với mỗi đơn i thuộc nhóm keys[i] (ví dụ cùng khách hàng hoặc cùng thành phố):
        n_prior[i] = số đơn CÙNG NHÓM mà kết quả đã được biết TRƯỚC thời điểm times[i]
        n_late[i]  = trong số đó, bao nhiêu đơn bị trễ

    Tham số (tất cả là mảng numpy cùng độ dài, thời gian dạng số nguyên nanosecond):
        times : thời điểm đặt của mỗi đơn (lúc cần ra quyết định)
        avail : thời điểm kết quả của mỗi đơn trở nên "được biết" (= ngày giao + 1 ngày)
        y     : nhãn 0/1 (1 = trễ)
        keys  : nhóm của mỗi đơn

    Vì sao dùng `avail` chứ không dùng `times`? Một đơn đặt hôm qua có thể CHƯA giao xong,
    nên hôm nay ta chưa biết nó có trễ hay không. Nếu đếm nó vào lịch sử thì là leakage.

    Cách làm (nhanh, không cần vòng lặp lồng nhau):
        1. Sắp xếp theo nhóm, tìm vị trí bắt đầu/kết thúc của từng nhóm.
        2. Trong mỗi nhóm, sắp xếp các đơn theo `avail` và tính tổng tích luỹ số đơn trễ.
        3. Dùng searchsorted: với mỗi thời điểm đặt `times[i]`, đếm số đơn có
           avail <= times[i]. Đó chính là số đơn đã biết kết quả.
    """
    n_prior = np.zeros(len(times))
    n_late = np.zeros(len(times))
    order = np.argsort(keys, kind="stable")      # chỉ số sau khi sắp xếp theo nhóm
    k_sorted = keys[order]
    # bounds: các vị trí mà giá trị nhóm thay đổi -> ranh giới giữa các nhóm.
    bounds = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        idx = order[s:e]                          # chỉ số (trong bảng gốc) của các đơn thuộc nhóm này
        a_order = np.argsort(avail[idx])          # sắp xếp các đơn trong nhóm theo thời điểm biết kết quả
        a_sorted = avail[idx][a_order]
        # cum_late[j] = số đơn trễ trong j đơn đầu tiên (theo thứ tự biết kết quả).
        cum_late = np.r_[0, np.cumsum(y[idx][a_order])]
        # pos[i] = số đơn trong nhóm có avail <= times[i] (side="right" -> tính cả dấu bằng).
        pos = np.searchsorted(a_sorted, times[idx], side="right")
        n_prior[idx] = pos
        n_late[idx] = cum_late[pos]
    return n_prior, n_late


def build_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng mỗi dòng = một đơn hàng (không bị huỷ), gồm các feature ở thời điểm
    order release và nhãn `late`.

    Vì sao gộp về cấp đơn hàng? Quyết định "có flag đơn này không" được đưa ra cho
    cả đơn, không phải từng dòng sản phẩm. Hơn nữa mọi dòng trong một đơn đều có
    cùng nhãn, nên nếu để ở cấp dòng thì các đơn nhiều dòng sẽ bị đếm lặp và CV
    có thể để dòng của cùng một đơn nằm ở cả train lẫn validation.
    """
    # Bỏ đơn bị huỷ / nghi gian lận (7.754 dòng): những đơn này không bao giờ được giao,
    # nên không có quyết định "rủi ro giao trễ" nào cho chúng.
    lines = df[df["Delivery Status"] != "Shipping canceled"].copy()

    # Kiểm tra giả định: các thuộc tính cấp đơn phải giống nhau trên mọi dòng của đơn.
    g = lines.groupby("Order Id")
    const = g[list(ORDER_CONSTANT)].nunique()
    bad = const.columns[(const > 1).any()].tolist()
    if bad:
        raise ValueError(f"Columns not constant within an order: {bad}")
    # Kiểm tra: không có đơn nào vừa có dòng trễ vừa có dòng đúng hạn.
    if (g["Late_delivery_risk"].nunique() > 1).any():
        raise ValueError("Some orders mix late and on-time lines")

    # "Ngành hàng chính" của đơn = ngành của dòng có doanh thu (Sales) lớn nhất.
    main_line = lines.loc[g["Sales"].idxmax(), ["Order Id", "Department Name", "Category Name"]]

    # Gộp các dòng thành một dòng/đơn. Mỗi cột mới = (cột gốc, cách gộp).
    orders = g.agg(
        order_dt=("order_dt", "first"),
        # Số ngày giao thực tế: là KẾT QUẢ, chỉ giữ lại để (1) tính thời điểm biết nhãn
        # cho các đặc trưng lịch sử, và (2) phân tích mô tả "đổi ngày hứa". KHÔNG phải feature.
        real_days=("Days for shipping (real)", "first"),
        late=("Late_delivery_risk", "first"),               # nhãn: 1 = trễ, 0 = không trễ
        n_lines=("Order Item Id", "size"),                  # số dòng sản phẩm trong đơn
        total_qty=("Order Item Quantity", "sum"),           # tổng số lượng
        order_value=("Sales", "sum"),                       # giá trị đơn (USD)
        total_discount=("Order Item Discount", "sum"),      # tổng tiền giảm giá
        mean_discount_rate=("Order Item Discount Rate", "mean"),  # tỷ lệ giảm giá trung bình
        max_unit_price=("Order Item Product Price", "max"), # đơn giá cao nhất trong đơn
        n_categories=("Category Id", "nunique"),            # số ngành hàng khác nhau
        order_profit=("Benefit per order", "sum"),          # lợi nhuận của đơn
        # Các thuộc tính cấp đơn: lấy giá trị đầu tiên (đã kiểm tra là giống nhau ở trên).
        **{v: (k, "first") for k, v in ORDER_CONSTANT.items()},
    ).reset_index()
    orders = orders.merge(
        main_line.rename(columns={"Department Name": "main_department",
                                  "Category Name": "main_category"}),
        on="Order Id")

    # Các feature dẫn xuất, đều biết ngay khi đặt hàng.
    orders["profit_margin"] = orders["order_profit"] / orders["order_value"]  # biên lợi nhuận
    orders["weekday"] = orders["order_dt"].dt.dayofweek   # thứ trong tuần (0 = thứ Hai)
    orders["hour"] = orders["order_dt"].dt.hour           # giờ đặt hàng (quan trọng với Same Day)
    orders["month"] = orders["order_dt"].dt.month         # tháng (mùa vụ)
    # Khoá ghép "thành phố | phương thức" để tính lịch sử trễ theo tuyến + mode.
    orders["city_mode"] = orders["order_city"] + " | " + orders["shipping_mode"]

    # --- Đặc trưng lịch sử (chống leakage) ---------------------------------------
    # Sắp xếp theo thời gian, rồi đổi thời gian sang số nguyên (nanosecond) để so sánh nhanh.
    orders = orders.sort_values("order_dt").reset_index(drop=True)
    t = orders["order_dt"].values.astype("datetime64[ns]").astype(np.int64)
    # Thời điểm nhãn của mỗi đơn được biết = ngày đặt + số ngày giao thực tế + 1 ngày.
    avail = (orders["order_dt"] + pd.to_timedelta(orders["real_days"], unit="D") + LABEL_DELAY)
    avail = avail.values.astype("datetime64[ns]").astype(np.int64)
    y = orders["late"].values.astype(float)
    # Tính cho 3 loại nhóm: theo khách hàng, theo thành phố, theo (thành phố, mode).
    for key, name in [("customer_id", "cust"), ("order_city", "city"), ("city_mode", "citymode")]:
        n, late = _prior_history(t, avail, y, orders[key].astype(str).values)
        orders[f"{name}_prior_orders"] = n
        # Tỷ lệ trễ được làm mượt (additive smoothing):
        #   (số đơn trễ + 5 * 0.5) / (số đơn + 5)
        # -> nhóm chưa có lịch sử nhận 0.5; nhóm có ít đơn không bị kéo về 0 hoặc 1 quá mạnh.
        orders[f"{name}_prior_late_rate"] = (late + HISTORY_WEIGHT * HISTORY_PRIOR) / (n + HISTORY_WEIGHT)

    # Đổi tên real_days thành outcome_real_days để nhắc rằng đây là KẾT QUẢ:
    # chỉ dùng cho phân tích mô tả "đổi ngày hứa", KHÔNG nằm trong FEATURES
    # (run_all.py có assert kiểm tra điều này).
    orders = orders.rename(columns={"real_days": "outcome_real_days"}).drop(columns=["city_mode"])
    # Ép kiểu các cột số về float: PartialDependenceDisplay của sklearn không nhận cột kiểu int.
    orders[NUMERIC] = orders[NUMERIC].astype(float)
    return orders


# ---------------------------------------------------------------------------
# Danh sách feature được đưa vào mô hình, chia theo cách mã hoá (xem modeling.py).
# ---------------------------------------------------------------------------
# Biến phân loại ít giá trị -> One-hot encoding.
CAT_LOW = ["shipping_mode", "market", "order_region", "customer_segment", "payment_type",
           "main_department"]
# Biến phân loại NHIỀU giá trị (hàng trăm/nghìn thành phố) -> Target encoding,
# vì one-hot sẽ tạo quá nhiều cột thưa.
CAT_HIGH = ["order_country", "order_city", "main_category"]
# Biến số.
NUMERIC = ["scheduled_days", "n_lines", "total_qty", "order_value", "total_discount",
           "mean_discount_rate", "max_unit_price", "n_categories", "order_profit", "profit_margin",
           "weekday", "hour", "month",
           "cust_prior_orders", "cust_prior_late_rate",
           "city_prior_orders", "city_prior_late_rate",
           "citymode_prior_orders", "citymode_prior_late_rate"]
FEATURES = CAT_LOW + CAT_HIGH + NUMERIC   # toàn bộ feature đầu vào
TARGET = "late"                           # biến mục tiêu
