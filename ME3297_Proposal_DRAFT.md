# ME3297 Project Proposal (CP1): Flagging DataCo orders at risk of late delivery at order release

**Section:** L\_\_ · **Team:** \_\_ · **Route:** A (approved dataset no. 1)

| Member | Student ID | Lead area |
|---|---|---|
| [Name 1] | [ID] | Problem framing, cost model, recommendation |
| [Name 2] | [ID] | Data audit, cleaning, leakage control |
| [Name 3] | [ID] | Feature engineering, pipeline, validation design |
| [Name 4] | [ID] | Tree-based models, interpretation |
| [Name 5] | [ID] | SVM / ANN models, results and presentation |

## 1. Dataset

**DataCo Smart Supply Chain for Big Data Analysis.** Constante, F., Silva, F., & Pereira, A. (2019). *Mendeley Data*, V5. https://doi.org/10.17632/8gx2fvg2k6.5 · Licence: **CC BY 4.0** (we cite it as the publisher requires).

We use `DataCoSupplyChainDataset.csv`. Our first audit found **180,519 order lines × 53 columns**, which is **65,752 orders** from 20,652 customers across 118 products, ordered between 01/2015 and 01/2018. Each order carries one label: none of the orders mix late and on-time lines, so we can model at the order level. Missing values are few and sit in columns we will not use: `Product Description` is 100% empty, `Order Zipcode` is 86% empty, and `Customer Lname` and `Customer Zipcode` are missing in fewer than 10 rows. We drop the PII fields (email, password, name, street) on load.

## 2. The decision (brief §4)

- **Decision:** Should this order be treated as high-risk before it leaves the warehouse? A high-risk order is either expedited (a mode upgrade or priority picking) or its customer is contacted in advance with a realistic date.
- **Decision-maker:** The logistics or order-fulfilment planner, supported by customer service.
- **Moment:** Order release, i.e. after the order is placed and paid and before it is dispatched. At this point we know the order contents, price, discount, customer, destination, the chosen shipping mode and its scheduled lead time. We do not yet know the actual transit time or the final status.
- **Target:** `Late_delivery_risk` (1 = late). We checked it against the other columns, and for every non-cancelled order it is exactly `Days for shipping (real) > Days for shipment (scheduled)`. We remove the 7,754 cancelled lines, because a cancelled order is never delivered and so has no delivery-risk decision. That leaves 62,897 orders, of which **57.3% are late**. The classes are mildly imbalanced, and the positive class is the majority.

**Leakage test.** The following fields are only known after the outcome, so we exclude them: `Days for shipping (real)`, `Delivery Status`, `Order Status` (which includes CANCELED/SUSPECTED\_FRAUD, set after the fact), and `shipping date (DateOrders)`. That last one looks harmless, but in 97% of rows it equals order date + real shipping days, which means it actually encodes the outcome. Customer-history features, such as a customer's past late rate, will be computed only from orders placed *before* the current one.

**Candidate features:** shipping mode, scheduled days, market, order region, country and city, customer segment, department and category, product price, quantity, discount rate, order value, number of lines, payment type, calendar features (weekday, month, hour), and prior-order count and prior late rate per customer and per destination.

**Baselines the model must beat:**
1. *Majority class* (flag everything as late): accuracy 57%, but it gives no ranking at all.
2. *Current-practice rule* (flag every First and Second Class order). On all orders this gives precision 0.89 and recall 0.54, flags 35% of orders, and reaches 70% accuracy. Shipping mode is a very strong signal, since 100% of non-cancelled First Class orders are late. So the real question is whether a model adds value *beyond* this rule, especially inside Standard Class (40% late), which carries most of the volume.

**Cost of errors:** A **false alarm** costs an unnecessary upgrade or a call, which we estimate at roughly $2–10 per order. A **missed late order** costs goodwill compensation, repeat contacts and churn risk. We take this as about 10% of order value, and the median order is $510, so a miss costs around $50. A miss is therefore about 5–25× more expensive than a false alarm. We will state these figures as assumptions and test how sensitive the results are to them.

## 3. Planned methods

- **Pipeline:** All imputation, encoding (one-hot for low-cardinality columns, target or frequency encoding for city and country inside CV folds) and scaling will sit inside a scikit-learn `Pipeline`/`ColumnTransformer`.
- **Split:** We split by time, never shuffling. Training covers orders before 2017-07-01 (≈49.8k orders) and the test set covers 2017-07 to 2018-01 (≈13.1k orders). Validation uses `TimeSeriesSplit` (5 folds) on the training set, and we report mean ± SD across folds.
- **Models (≥3 families, at least 2 from Weeks 6–13):** (1) a **decision tree** and a tree ensemble (Random Forest / gradient boosting), (2) an **SVM** (linear and RBF, the latter trained on a documented sample), and (3) an **ANN** (MLP). Logistic regression serves as a transparent reference. As an optional extra, we may run **k-means clustering** of destination × mode lanes to describe where lateness concentrates.
- **Metrics:** Our main metrics are PR-AUC and **recall at a fixed alert capacity** (e.g. the top 20% or 35% of orders flagged), plus an **expected-cost curve** over the decision threshold using the costs in §2. We also report F1 and ROC-AUC. Accuracy will only ever appear next to these metrics.
- **Interpretation:** We will use permutation importance, tree structure and partial-dependence plots, and check whether the model's drivers make supply-chain sense (mode vs. scheduled lead time, region, order size).
- **Recommendation output:** For the recommended threshold we will report orders flagged per week, late orders caught per week and the net cost saved against the rule baseline. For scale, the base volume is about 420 orders per week in 2017.

## 4. Risk note

(1) The delivery outcome looks machine-generated. In our audit, for every Standard and Second Class order the real transit days equal (Order Id − 1) mod 5 + 2. Every First Class order takes exactly 2 days, and a Same Day order is late exactly when it is placed after 12:00. So no order-level feature can explain lateness within a mode except the Same Day hour, and we will report the model's gain over a mode-based lookup table honestly, even if it is close to zero. (2) The data drifts from 10/2017. The active catalogue shrinks from about 55 to 10 products and the median order value falls from about $600 to about $85, so cost-based results for the test period depend strongly on order value.
