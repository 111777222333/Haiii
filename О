import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Деректерді жүктеу (3-тапсырма)
url = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
df = pd.read_csv(url)

# CSV, Excel және JSON форматтарында сақтау
df.to_csv("titanic_data.csv", index=False)
df.to_excel("titanic_data.xlsx", index=False)
df.to_json("titanic_data.json", orient="records", indent=2, force_ascii=False)

# 2. Белгілер және мақсатты айнымалы (4-тапсырма)
target = "survived"
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

# 3. Деректерді тазалау (5-тапсырма)
print("Тазалауға дейінгі бос мәндер:\n", df[features + [target]].isnull().sum())
print("Қайталанған жолдар саны:", df.duplicated().sum())

df_clean = df.drop_duplicates().copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
df_clean["embarked"] = df_clean["embarked"].fillna(
    df_clean["embarked"].mode()[0]
)

print("Тазалаудан кейінгі бос мәндер:\n", df_clean[features + [target]].isnull().sum())

# 4. EDA: 5 график (6-тапсырма)

# 1-график: Таралу
plt.figure(figsize=(8, 4))
sns.histplot(df_clean["age"], bins=25, kde=True)
plt.title("Жолаушылардың жас бойынша таралуы")
plt.xlabel("Жасы, жыл")
plt.ylabel("Жолаушылар саны, адам")
plt.tight_layout()
plt.show()

# 2-график: Корреляция
plt.figure(figsize=(8, 5))
num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
sns.heatmap(df_clean[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Сандық көрсеткіштердің корреляция матрицасы")
plt.xlabel("Айнымалылар")
plt.ylabel("Айнымалылар")
plt.tight_layout()
plt.show()

# 3-график: Топтарды салыстыру (жынысы бойынша)
plt.figure(figsize=(7, 4))
sns.barplot(data=df_clean, x="sex", y="survived", errorbar=None)
plt.title("Жыныс бойынша аман қалу үлесі")
plt.xlabel("Жынысы")
plt.ylabel("Аман қалу үлесі (үлес)")
plt.tight_layout()
plt.show()

# 4-график: Топтарды салыстыру (билет класы бойынша)
plt.figure(figsize=(7, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", errorbar=None)
plt.title("Билет класы бойынша аман қалу үлесі")
plt.xlabel("Билет класы (1, 2, 3)")
plt.ylabel("Аман қалу үлесі (үлес)")
plt.tight_layout()
plt.show()

# 5-график: Билет құнының кластар бойынша таралуы
plt.figure(figsize=(8, 4))
sns.boxplot(data=df_clean, x="pclass", y="fare")
plt.title("Билет құнының кластар бойынша таралуы")
plt.xlabel("Билет класы")
plt.ylabel("Билет құны, £")
plt.tight_layout()
plt.show()
