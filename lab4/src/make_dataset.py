import pandas as pd
from catboost.datasets import titanic

train_df, test_df = titanic()
print("Информация об исходном датасете:")
print(train_df.info())

# Заполнение пропущенных значений в поле "Age" средним значением
mean_age = train_df['Age'].mean()
print(f"Средний возраст: {mean_age}")

train_df['Age'] = train_df['Age'].fillna(mean_age)
test_df['Age'] = test_df['Age'].fillna(mean_age)

print("Информация о датасете после замены пропущенных значений:")
print(train_df.info())

try:
    # Сохранение обновленных датасетов в CSV
    train_df.to_csv('../datasets/train.csv', index=False)
    test_df.to_csv('../datasets/test.csv', index=False)
    print("Датасеты успешно сохранены в каталоге ../dataset под именами train.csv и test.csv.")
except Exception as e:
    print("Произошла ошибка при сохранении датасетов:", e)

# print("Train dataset:")
# print(train_df.head())

#print(test_df.info())
# print("\nTest dataset:")
# print(test_df.head())