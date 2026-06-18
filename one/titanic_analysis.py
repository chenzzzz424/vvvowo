import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
PALETTE = sns.color_palette("Set2")

df = pd.read_csv('titanic.csv')

# ---------- Data Cleaning ----------
# Fill missing Age with median by Pclass+Sex group
df['Age'] = df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Drop Cabin (too many missing), keep HasCabin as feature
df['HasCabin'] = df['Cabin'].notna().astype(int)
df = df.drop(columns=['Cabin', 'Ticket'])

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
def age_group(a):
    if a < 12: return 'Child'
    elif a < 18: return 'Teen'
    elif a < 60: return 'Adult'
    else: return 'Senior'
df['AgeGroup'] = df['Age'].apply(age_group)
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
rare = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare, 'Other')

# Remove extreme fare outliers (>99th percentile) for Fare distribution plot only -> keep in df, just flag
print(df.describe(include='all'))
print(df['AgeGroup'].value_counts())
print(df['Title'].value_counts())

df.to_csv('titanic_clean.csv', index=False)

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30,
              palette={0:'#fc8d62', 1:'#66c2a5'}, ax=ax)
ax.set_title('乘客年龄分布（按是否生还分层）', fontsize=14, fontweight='bold')
ax.set_xlabel('年龄')
ax.set_ylabel('人数')
handles = [plt.Rectangle((0,0),1,1,color='#fc8d62'), plt.Rectangle((0,0),1,1,color='#66c2a5')]
ax.legend(handles, ['未生还','生还'], title='生还情况')
plt.tight_layout()
plt.savefig('fig1_age_dist.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Pclass', palette='viridis',
                 alpha=0.7, s=50, ax=ax)
ax.set_title('票价与年龄关系（按舱位等级分组）', fontsize=14, fontweight='bold')
ax.set_xlabel('年龄')
ax.set_ylabel('票价（英镑）')
ax.legend(title='舱位等级')
plt.tight_layout()
plt.savefig('fig2_fare_age_scatter.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
grp = df.groupby(['Pclass','Sex'])['Survived'].mean().reset_index()
sns.barplot(data=grp, x='Pclass', y='Survived', hue='Sex', palette='Set2', ax=ax)
ax.set_title('不同舱位等级与性别的生还率对比', fontsize=14, fontweight='bold')
ax.set_xlabel('舱位等级')
ax.set_ylabel('生还率')
ax.set_ylim(0,1)
ax.legend(title='性别')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
plt.tight_layout()
plt.savefig('fig3_survival_bar.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,6))
num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare','FamilySize','HasCabin']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('数值变量相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_corr_heatmap.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
grp2 = df.groupby(['Embarked','AgeGroup'])['Survived'].mean().reset_index()
order = ['Child','Teen','Adult','Senior']
sns.barplot(data=grp2, x='Embarked', y='Survived', hue='AgeGroup', hue_order=order,
            palette='coolwarm', ax=ax)
ax.set_title('登船港口与年龄段对生还率的影响', fontsize=14, fontweight='bold')
ax.set_xlabel('登船港口 (C=Cherbourg, Q=Queenstown, S=Southampton)')
ax.set_ylabel('生还率')
ax.set_ylim(0,1)
ax.legend(title='年龄段')
plt.tight_layout()
plt.savefig('fig5_embark_agegroup.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
df_box = df[df['Fare'] < df['Fare'].quantile(0.99)]  # remove extreme outliers for visual clarity
sns.boxplot(data=df_box, x='Pclass', y='Fare', palette='Set3', ax=ax)
ax.set_title('各舱位等级票价分布箱线图（剔除前1%极端值）', fontsize=14, fontweight='bold')
ax.set_xlabel('舱位等级')
ax.set_ylabel('票价（英镑）')
plt.tight_layout()
plt.savefig('fig6_fare_box.png', dpi=150)
plt.close()

print("All figures saved.")
