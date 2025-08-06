import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

summary_stats = df.describe(include='all')
median_values = df.median(numeric_only=True)
std_values = df.std(numeric_only=True)


plt.figure(figsize=(12, 5))
sns.histplot(df['Age'].dropna(), kde=True)
plt.title("Distribution of Age")
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare by Passenger Class")
plt.show()

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()

plt.figure(figsize=(8, 6))
corr = df[['Survived', 'Pclass', 'Age', 'Fare']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


fig = px.box(df, x='Sex', y='Age', color='Sex', title='Age Distribution by Gender')
fig.show()

fig = px.histogram(df, x='Fare', nbins=50, title='Fare Distribution')
fig.show()


survival_by_sex = df.groupby('Sex')['Survived'].mean()
print("\n💡 Survival rate by Sex:\n", survival_by_sex)

plt.figure(figsize=(10, 4))
sns.violinplot(x='Survived', y='Age', data=df)
plt.title("Age Distribution vs Survival")
plt.show()