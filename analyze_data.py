import pandas as pd
import numpy as np

# Загружаем данные
df = pd.read_csv('hand_data.csv')

print("=== АНАЛИЗ ДАННЫХ ТРЕКИНГА ===")
print(f"Общий размер данных: {df.shape}")
print(f"Уникальные руки: {df['hand'].unique()}")
print(f"Количество записей по рукам:")
print(df['hand'].value_counts())

print("\n=== ДАННЫЕ ЛЕВОЙ РУКИ ===")
left_data = df[df['hand'] == 'left']
print(f"Количество записей: {len(left_data)}")
if len(left_data) > 0:
    print("Записи левой руки:")
    print(left_data[['frame', 'timestamp', 'hand']].to_string())
    print(f"\nВременной диапазон: {left_data['timestamp'].min():.2f}с - {left_data['timestamp'].max():.2f}с")
    print(f"Длительность: {left_data['timestamp'].max() - left_data['timestamp'].min():.2f}с")
else:
    print("Нет данных для левой руки!")

print("\n=== ДАННЫЕ ПРАВОЙ РУКИ ===")
right_data = df[df['hand'] == 'right']
print(f"Количество записей: {len(right_data)}")
if len(right_data) > 0:
    print(f"Временной диапазон: {right_data['timestamp'].min():.2f}с - {right_data['timestamp'].max():.2f}с")
    print(f"Длительность: {right_data['timestamp'].max() - right_data['timestamp'].min():.2f}с")

print("\n=== ПРОБЛЕМА ===")
print("Для левой руки недостаточно данных для корректного анализа движения.")
print("Нужно улучшить алгоритм детекции рук в hand_tracker.py")

