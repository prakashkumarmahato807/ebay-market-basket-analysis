import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# setting a nice style for all charts
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
 
print("=" * 60)
print("  eBay MARKET BASKET ANALYSIS - STARTING")
print("=" * 60)

# ============================================================
# TASK 1: DATA CLEANING AND PREPARATION 
# ============================================================
 
print("\n--- TASK 1: DATA CLEANING AND PREPARATION ---")
 
# loading the dataset
df = pd.read_csv(r"C:\Users\Prakash\Desktop\eBay_Project\eBay (1).csv")
print(f"Original dataset shape: {df.shape}")
 
# step 1: remove trailing spaces from column names
# Rating_Accuracy and Personalized_Recommendation_Frequency had trailing spaces
df.columns = df.columns.str.strip()
print("Column names cleaned - trailing spaces removed")

# step 2: check and remove duplicate columns
# Personalized_Recommendation_Frequency appeared twice - keeping the string version
# the duplicate had integer values, original had text values
if df.columns.tolist().count("Personalized_Recommendation_Frequency") > 1:
    # keeping only the first occurrence (text version)
    df = df.loc[:, ~df.columns.duplicated()]
    print("Duplicate column 'Personalized_Recommendation_Frequency' removed")
 
print(f"Shape after removing duplicates: {df.shape}")
 
# step 3: handle missing values in Product_Search_Method
# 161 values were missing - filling with most common value (mode)
missing_before = df["Product_Search_Method"].isnull().sum()
most_common = df["Product_Search_Method"].mode()[0]
df["Product_Search_Method"].fillna(most_common, inplace=True)
print(f"Filled {missing_before} missing values in Product_Search_Method with '{most_common}'")
 
# step 4: remove duplicate rows
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Removed {duplicates_before} duplicate rows")

# step 5: standardize Gender column (fixing inconsistent entries)
df["Gender"] = df["Gender"].str.strip().str.title()
print("Gender column standardized")
 
# step 6: standardize Purchase_Frequency column
df["Purchase_Frequency"] = df["Purchase_Frequency"].str.strip()
 
# step 7: convert numeric columns to proper numeric type
# Customer_Reviews_Importance and Shopping_Satisfaction should be numbers
df["Customer_Reviews_Importance"] = pd.to_numeric(df["Customer_Reviews_Importance"], errors="coerce")
df["Shopping_Satisfaction"] = pd.to_numeric(df["Shopping_Satisfaction"], errors="coerce")
df["Rating_Accuracy"] = pd.to_numeric(df["Rating_Accuracy"], errors="coerce")
df["Recommendation_Helpfulness"] = df["Recommendation_Helpfulness"].map({"Yes": 1, "Sometimes": 0.5, "No": 0})
print("Numeric columns converted to proper types")
 
# step 8: fix age outliers - age 3 is not realistic for eBay shopper
print(f"\nAge range before cleaning: {df['age'].min()} to {df['age'].max()}")
df = df[df["age"] >= 13]  # keeping only 13+ age (realistic eBay user)
df.reset_index(drop=True, inplace=True)
print(f"Age range after cleaning: {df['age'].min()} to {df['age'].max()}")
 
print(f"\nFinal clean dataset shape: {df.shape}")
print("Task 1 DONE!")

# ============================================================
# TASK 2: DESCRIPTIVE BEHAVIOR ANALYSIS 
# ============================================================
 
print("\n--- TASK 2: DESCRIPTIVE BEHAVIOR ANALYSIS ---")
 
# --- 2.1: Customer Demographics ---
 
print("\n>> Age Statistics:")
print(f"   Mean Age   : {df['age'].mean():.1f} years")
print(f"   Median Age : {df['age'].median():.1f} years")
print(f"   Min Age    : {df['age'].min()} years")
print(f"   Max Age    : {df['age'].max()} years")
 
print("\n>> Gender Distribution:")
gender_counts = df["Gender"].value_counts()
for g, c in gender_counts.items():
    print(f"   {g}: {c} ({c/len(df)*100:.1f}%)")
 
print("\n>> Purchase Frequency Distribution:")
pf_counts = df["Purchase_Frequency"].value_counts()
for p, c in pf_counts.items():
    print(f"   {p}: {c}")
 
 # --- 2.2: Popular Product Categories ---
print("\n>> Most Popular Product Categories:")
# each row can have multiple categories separated by semicolon
all_categories = []
for cats in df["Purchase_Categories"].dropna():
    for cat in cats.split(";"):
        all_categories.append(cat.strip())
 
from collections import Counter
cat_counts = Counter(all_categories)
top_categories = cat_counts.most_common(10)
print("   Top 10 categories:")
for cat, count in top_categories:
    print(f"   {cat}: {count}")
 
# --- 2.3: Cart Abandonment Analysis ---
print("\n>> Top Cart Abandonment Factors:")
abandon_counts = df["Cart_Abandonment_Factors"].value_counts()
for factor, count in abandon_counts.head(5).items():
    print(f"   {factor}: {count}")
 
 # --- 2.4: Satisfaction Statistics ---
print("\n>> Satisfaction & Rating Statistics:")
print(f"   Mean Shopping Satisfaction    : {df['Shopping_Satisfaction'].mean():.2f} / 5")
print(f"   Median Shopping Satisfaction  : {df['Shopping_Satisfaction'].median():.1f} / 5")
print(f"   Mean Rating Accuracy          : {df['Rating_Accuracy'].mean():.2f} / 5")
print(f"   Mean Customer Reviews Import  : {df['Customer_Reviews_Importance'].mean():.2f} / 5")
 
print("\nTask 2 DONE!")


# ============================================================
# TASK 3: CUSTOMER SEGMENTATION AND PROFILING 
# ============================================================
 
print("\n--- TASK 3: CUSTOMER SEGMENTATION ---")
 
# mapping purchase frequency to numbers for clustering
freq_map = {
    "Multiple times a day": 5,
    "Multiple times a week": 4,
    "Once a week": 3,
    "Few times a month": 2,
    "Once a month": 1,
    "Rarely": 0
}
 
df["Purchase_Freq_Num"] = df["Purchase_Frequency"].map(freq_map)
# filling unmapped values with median
df["Purchase_Freq_Num"].fillna(df["Purchase_Freq_Num"].median(), inplace=True)
 
# using Shopping_Satisfaction and Purchase_Freq_Num for segmentation
segmentation_data = df[["Purchase_Freq_Num", "Shopping_Satisfaction",
                          "Customer_Reviews_Importance", "Rating_Accuracy"]].copy()
segmentation_data.fillna(segmentation_data.median(), inplace=True)
 
 # scaling data before clustering (important for K-Means)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)
 
# applying K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_data)
 
# labeling clusters based on their characteristics
cluster_summary = df.groupby("Cluster").agg({
    "Purchase_Freq_Num": "mean",
    "Shopping_Satisfaction": "mean"
}).reset_index()
 
print("\n>> Cluster Summary:")
print(cluster_summary)
 
 # assigning meaningful names to clusters
cluster_labels = {}
for _, row in cluster_summary.iterrows():
    if row["Purchase_Freq_Num"] >= cluster_summary["Purchase_Freq_Num"].median() and \
       row["Shopping_Satisfaction"] >= cluster_summary["Shopping_Satisfaction"].median():
        cluster_labels[row["Cluster"]] = "Frequent Buyers"
    elif row["Shopping_Satisfaction"] < cluster_summary["Shopping_Satisfaction"].quantile(0.4):
        cluster_labels[row["Cluster"]] = "At-Risk Customers"
    else:
        cluster_labels[row["Cluster"]] = "Occasional Shoppers"
 
df["Customer_Segment"] = df["Cluster"].map(cluster_labels)
 
print("\n>> Customer Segment Distribution:")
seg_counts = df["Customer_Segment"].value_counts()
for seg, count in seg_counts.items():
    print(f"   {seg}: {count} customers ({count/len(df)*100:.1f}%)")
 
print("\n>> Segment Profiles:")
seg_profile = df.groupby("Customer_Segment").agg({
    "age": "mean",
    "Shopping_Satisfaction": "mean",
    "Purchase_Freq_Num": "mean",
    "Rating_Accuracy": "mean"
}).round(2)
print(seg_profile)
 
print("\nTask 3 DONE!")

# ============================================================
# TASK 4: RECOMMENDATION AND REVIEW INSIGHTS (10 Marks)
# ============================================================
 
print("\n--- TASK 4: RECOMMENDATION AND REVIEW INSIGHTS ---")
 
# relationship between recommendation helpfulness and satisfaction
print("\n>> Recommendation Helpfulness vs Shopping Satisfaction:")
rec_sat = df.groupby("Recommendation_Helpfulness")["Shopping_Satisfaction"].mean()
print(rec_sat)
 
# review reliability impact
print("\n>> Review Reliability Distribution:")
print(df["Review_Reliability"].value_counts())
 
# review helpfulness impact
print("\n>> Review Helpfulness Distribution:")
print(df["Review_Helpfulness"].value_counts())
 
# customers who trust recommendations
rec_freq = df["Personalized_Recommendation_Frequency"].value_counts()
print("\n>> How Often Customers See Personalized Recommendations:")
print(rec_freq)
 
 # correlation between key metrics
print("\n>> Correlation between key metrics:")
corr_cols = ["Shopping_Satisfaction", "Rating_Accuracy",
             "Customer_Reviews_Importance", "Recommendation_Helpfulness"]
corr_data = df[corr_cols].dropna()
print(corr_data.corr().round(3))
 
print("\nTask 4 DONE!")

# ============================================================
# TASK 5: VISUALIZATION AND REPORTING 
# ============================================================
 
print("\n--- TASK 5: CREATING VISUALIZATIONS ---")
 
# --- Chart 1: Gender Distribution Pie Chart ---
fig, ax = plt.subplots(figsize=(8, 6))
gender_data = df["Gender"].value_counts()
colors = ["#4da6ff", "#ff6b9d", "#a8e6cf", "#ffd93d"]
ax.pie(gender_data.values, labels=gender_data.index, autopct="%1.1f%%",
       colors=colors, startangle=90, textprops={"fontsize": 12})
ax.set_title("Gender Distribution of eBay Customers", pad=20)
plt.tight_layout()
plt.savefig("chart1_gender_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 1 saved: Gender Distribution")

# --- Chart 2: Purchase Frequency Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))
order = ["Multiple times a day", "Multiple times a week", "Once a week",
         "Few times a month", "Once a month", "Rarely"]
pf_ordered = df["Purchase_Frequency"].value_counts().reindex(order, fill_value=0)
bars = ax.bar(range(len(pf_ordered)), pf_ordered.values, color="#4da6ff", edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(pf_ordered)))
ax.set_xticklabels(pf_ordered.index, rotation=30, ha="right", fontsize=11)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.set_title("Purchase Frequency Distribution", fontsize=14, fontweight="bold")
for bar, val in zip(bars, pf_ordered.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=10)
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("chart2_purchase_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved: Purchase Frequency")
 
 # --- Chart 3: Top 10 Product Categories Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 7))
top10 = dict(top_categories)
colors_grad = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
bars = ax.barh(list(top10.keys()), list(top10.values()),
               color=colors_grad, edgecolor="white")
ax.set_xlabel("Number of Customers", fontsize=12)
ax.set_title("Top 10 Most Popular Purchase Categories", fontsize=14, fontweight="bold")
for bar, val in zip(bars, top10.values()):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig("chart3_product_categories.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 3 saved: Product Categories")

# --- Chart 4: Shopping Satisfaction Distribution ---
fig, ax = plt.subplots(figsize=(8, 6))
sat_counts = df["Shopping_Satisfaction"].value_counts().sort_index()
colors_sat = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#4dabf7"]
bars = ax.bar(sat_counts.index, sat_counts.values,
              color=colors_sat[:len(sat_counts)], edgecolor="white", linewidth=0.5, width=0.6)
ax.set_xlabel("Satisfaction Level (1=Low, 5=High)", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.set_title("Shopping Satisfaction Distribution", fontsize=14, fontweight="bold")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1\n(Very Low)", "2\n(Low)", "3\n(Medium)", "4\n(High)", "5\n(Very High)"])
for bar, val in zip(bars, sat_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=10)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig("chart4_satisfaction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved: Shopping Satisfaction")
 
 # --- Chart 5: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df[["Shopping_Satisfaction", "Rating_Accuracy",
                   "Customer_Reviews_Importance", "Recommendation_Helpfulness",
                   "Purchase_Freq_Num", "age"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues",
            ax=ax, linewidths=0.5, square=True,
            annot_kws={"size": 11})
ax.set_title("Correlation Between Key Behavioral Metrics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 5 saved: Correlation Heatmap")

# --- Chart 6: Customer Segments Pie Chart ---
fig, ax = plt.subplots(figsize=(8, 6))
seg_data = df["Customer_Segment"].value_counts()
seg_colors = ["#4da6ff", "#ff9a3c", "#ff6b6b"]
ax.pie(seg_data.values, labels=seg_data.index, autopct="%1.1f%%",
       colors=seg_colors, startangle=90, textprops={"fontsize": 12},
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Customer Segmentation Distribution", pad=20, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart6_customer_segments.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 6 saved: Customer Segments")

# --- Chart 7: Cart Abandonment Factors ---
fig, ax = plt.subplots(figsize=(10, 6))
abandon_top = df["Cart_Abandonment_Factors"].value_counts().head(8)
colors_ab = plt.cm.Reds(np.linspace(0.4, 0.85, len(abandon_top)))
bars = ax.barh(abandon_top.index, abandon_top.values,
               color=colors_ab, edgecolor="white")
ax.set_xlabel("Number of Customers", fontsize=12)
ax.set_title("Top Cart Abandonment Factors", fontsize=14, fontweight="bold")
for bar, val in zip(bars, abandon_top.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig("chart7_cart_abandonment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 7 saved: Cart Abandonment")
 
 # --- Chart 8: Age Distribution ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["age"], bins=20, color="#4da6ff", edgecolor="white", linewidth=0.5)
ax.axvline(df["age"].mean(), color="red", linestyle="--",
           linewidth=2, label=f"Mean Age: {df['age'].mean():.1f}")
ax.axvline(df["age"].median(), color="orange", linestyle="--",
           linewidth=2, label=f"Median Age: {df['age'].median():.1f}")
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.set_title("Age Distribution of eBay Customers", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig("chart8_age_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 8 saved: Age Distribution")
 
print("\nAll 8 charts saved!")
print("\nTask 5 DONE!")

# ============================================================
# FINAL INSIGHTS SUMMARY
# ============================================================
 
print("\n" + "=" * 60)
print("  KEY INSIGHTS FROM eBay ANALYSIS")
print("=" * 60)
 
print("""
INSIGHT 1 - DEMOGRAPHICS:
  Most eBay customers are between 20-55 years old.
  Female customers are the majority shoppers.
 
INSIGHT 2 - PURCHASE BEHAVIOR:
  Most customers shop multiple times a week or once a month.
  Groceries, Beauty and Fashion are the top categories.
 
INSIGHT 3 - CART ABANDONMENT:
  High shipping costs and finding better prices elsewhere
  are the main reasons customers abandon their carts.
 
INSIGHT 4 - CUSTOMER SATISFACTION:
  Average satisfaction is moderate (around 3/5).
  Rating accuracy is trusted more than recommendations.
 
INSIGHT 5 - CUSTOMER SEGMENTS:
  3 segments found: Frequent Buyers, Occasional Shoppers,
  and At-Risk Customers who need special attention.
 
INSIGHT 6 - RECOMMENDATIONS:
  Customers who find recommendations helpful tend to have
  higher shopping satisfaction scores.
 
RECOMMENDATIONS FOR eBay:
  1. Reduce shipping costs to decrease cart abandonment.
  2. Improve recommendation system for At-Risk customers.
  3. Target Frequent Buyers with loyalty programs.
  4. Focus marketing on Groceries and Fashion categories.
""")
 
print("=" * 60)
print("  ANALYSIS COMPLETE!")
print("  All charts saved in current folder.")
print("=" * 60)
 
