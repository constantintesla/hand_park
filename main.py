import argparse
from hand_tracker import analyze_video
from hand_analysis import HandMovementAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Полный анализ движения кисти: трекинг и оценка выполнения пробы')
    parser.add_argument('--input', required=True, help='Путь к входному видеофайлу')
    parser.add_argument('--fps', type=float, default=30, help='Кадровая частота видео')
    parser.add_argument('--no-display', action='store_true', help='Не показывать видео в реальном времени')
    parser.add_argument('--tracking-output', default="hand_data.csv", 
                       help='Путь для сохранения данных трекинга (по умолчанию: hand_data.csv)')
    parser.add_argument('--analysis-output', default="hand_analysis_results.csv", 
                       help='Путь для сохранения результатов анализа (по умолчанию: hand_analysis_results.csv)')
    
    args = parser.parse_args()
    
    # Шаг 1: Трекинг руки и сохранение данных
    print("Запуск трекинга руки...")
    analyze_video(
        video_path=args.input,
        output_csv=args.tracking_output,
        show_video=not args.no_display
    )
    
    # Шаг 2: Анализ движения
    print("\nАнализ движения руки...")
    analyzer = HandMovementAnalyzer(args.tracking_output, args.fps)
    results = analyzer.analyze()
    
    # Генерация и вывод отчета
    report = analyzer.generate_report()
    if report:
        print(report)
    
    # Сохранение результатов анализа
    analyzer.save_to_csv(args.analysis_output)
    print(f"\nРезультаты анализа сохранены в {args.analysis_output}")

if __name__ == "__main__":
    main()