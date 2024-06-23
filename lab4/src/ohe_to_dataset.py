import pandas as pd

# Прочитаем датасет из файлов CSV
train_df = pd.read_csv('../datasets/train.csv')
test_df = pd.read_csv('../datasets/test.csv')

print("Информация об исходном датасете:")
print(train_df.info())

# Применяем One-Hot-Encoding к столбцу "Sex"
train_df_sex_encoded = pd.get_dummies(train_df['Sex'], prefix='Sex')
test_df_sex_encoded = pd.get_dummies(test_df['Sex'], prefix='Sex')

# Соединяем закодированные данные с исходным DataFrame
train_df = pd.concat([train_df, train_df_sex_encoded], axis=1)
test_df = pd.concat([test_df, test_df_sex_encoded], axis=1)
print("Информация о датасете после замены пропущенных значений:")
print(train_df.info())

try:
    # Сохранение обновленных датасетов в CSV
    train_df.to_csv('../datasets/train.csv', index=False)
    test_df.to_csv('../datasets/test.csv', index=False)
    print("Датасеты успешно сохранены в каталоге ../dataset под именами train.csv и test.csv.")
except Exception as e:
    print("Произошла ошибка при сохранении датасетов:", e)

#print("Train dataset:")
#print(train_df.head())

#print(test_df.info())
# print("\nTest dataset:")
# print(test_df.head())