import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from enum import Enum
import os

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

class MovementPhase(Enum):
    OPENING = "opening"
    CLOSING = "closing"

class HandMovementAnalyzer:
    def __init__(self, csv_path: str, fps: float = 30):
        """
        Инициализация анализатора
        
        :param csv_path: путь к CSV с данными трекинга
        :param fps: кадровая частота видео (для временных расчетов)
        """
        self.data = pd.read_csv(csv_path)
        self.fps = fps
        self.hand_results = {Hand.LEFT: None, Hand.RIGHT: None}
        
        # Пороговые значения для классификации
        self.AMPLITUDE_THRESHOLD = 0.7
        self.SPEED_THRESHOLD = 0.5
        self.RHYTHM_VARIABILITY = 0.3

    def _detect_hand(self, landmarks: np.ndarray) -> Hand:
        """Определение левой/правой руки по положению landmarks"""
        wrist = landmarks[0]
        thumb_base = landmarks[1]
        pinky_base = landmarks[17]
        
        vec_thumb = thumb_base - wrist
        vec_pinky = pinky_base - wrist
        
        cross_z = vec_thumb[0] * vec_pinky[1] - vec_thumb[1] * vec_pinky[0]
        
        return Hand.LEFT if cross_z > 0 else Hand.RIGHT

    def _normalize_landmarks(self, frame_landmarks: np.ndarray) -> np.ndarray:
        """Нормализация landmarks относительно запястья"""
        wrist = frame_landmarks[0]
        return frame_landmarks - wrist

    def _calculate_fist_amplitude(self, landmarks: np.ndarray) -> float:
        """Расчет степени сжатия кулака (0-1)"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        dists = [
            np.linalg.norm(thumb_tip - wrist),
            np.linalg.norm(index_tip - wrist),
            np.linalg.norm(middle_tip - wrist),
            np.linalg.norm(ring_tip - wrist),
            np.linalg.norm(pinky_tip - wrist)
        ]
        
        amplitude = 1 - np.mean(dists) / np.linalg.norm(landmarks[5] - wrist)
        return np.clip(amplitude, 0, 1)

    def _detect_movement_phases(self, amplitudes: np.ndarray) -> List[Tuple[MovementPhase, int, int]]:
        """Детекция фаз движения (сжатие/разжатие)"""
        phases = []
        direction = None
        start = 0
        
        for i in range(1, len(amplitudes)):
            current_dir = MovementPhase.CLOSING if amplitudes[i] > amplitudes[i-1] else MovementPhase.OPENING
            
            if direction is None:
                direction = current_dir
            elif current_dir != direction:
                phases.append((direction, start, i-1))
                direction = current_dir
                start = i
        
        if direction is not None:
            phases.append((direction, start, len(amplitudes)-1))
            
        return phases

    def _analyze_phases(self, phases: List[Tuple[MovementPhase, int, int]]) -> Dict:
        """Анализ характеристик фаз движения"""
        if not phases:
            return {}
        
        phase_durations = [(end-start+1)/self.fps for _, start, end in phases]
        avg_duration = np.mean(phase_durations)
        
        rhythm_variability = np.std(phase_durations) / avg_duration
        
        closing_phases = [p for p in phases if p[0] == MovementPhase.CLOSING]
        full_amplitude_ratio = len(closing_phases) / (len(phases)/2) if phases else 0
        
        return {
            'rhythm_consistency': 1 - min(rhythm_variability / self.RHYTHM_VARIABILITY, 1),
            'speed_consistency': 1 - min(np.std(phase_durations[:4]) / avg_duration, 1) if len(phase_durations) >=4 else 0,
            'full_amplitude_ratio': full_amplitude_ratio,
            'movement_completeness': min(full_amplitude_ratio / self.AMPLITUDE_THRESHOLD, 1),
            'phase_durations': phase_durations,
            'num_cycles': len(phases)//2
        }

    def _classify_performance(self, analysis: Dict) -> int:
        """Классификация качества выполнения пробы (0-4)"""
        if not analysis:
            return 0
        
        criteria = [
            analysis['movement_completeness'] >= 0.9,
            analysis['full_amplitude_ratio'] >= 0.8,
            analysis['rhythm_consistency'] >= 0.7,
            analysis['speed_consistency'] >= 0.6,
            analysis['num_cycles'] >= 4
        ]
        
        return sum(criteria)

    def analyze(self):
        """Основной метод анализа данных"""
        if self.data.empty:
            return self.hand_results
            
        landmarks_per_frame = []
        for i in range(len(self.data)):
            frame_landmarks = []
            for j in range(21):
                x = self.data[f'hand_{j}_x'].values[i]
                y = self.data[f'hand_{j}_y'].values[i]
                z = self.data[f'hand_{j}_z'].values[i] if f'hand_{j}_z' in self.data.columns else 0
                frame_landmarks.append(np.array([x, y, z]))
            landmarks_per_frame.append(np.array(frame_landmarks))
        
        if not landmarks_per_frame:
            return self.hand_results
        
        try:
            hand = self._detect_hand(landmarks_per_frame[0])
            
            normalized_landmarks = [self._normalize_landmarks(lm) for lm in landmarks_per_frame]
            amplitudes = np.array([self._calculate_fist_amplitude(lm) for lm in normalized_landmarks])
            
            phases = self._detect_movement_phases(amplitudes)
            phase_analysis = self._analyze_phases(phases)
            score = self._classify_performance(phase_analysis)
            
            self.hand_results[hand] = {
                'score': score,
                'phases': phases,
                'amplitudes': amplitudes,
                'analysis': phase_analysis,
                'timestamps': self.data['timestamp'].values
            }
        except Exception as e:
            print(f"Ошибка при анализе: {str(e)}")
        
        return self.hand_results

    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        if not any(self.hand_results.values()):
            return ""
        
        report = []
        for hand, result in self.hand_results.items():
            if result is None:
                continue
                
            if hand == Hand.RIGHT:
                report.append(f"\nАнализ {hand.value} руки:")
                report.append(f"Оценка выполнения: {result['score']}/4")
                
                if result['score'] == 4:
                    report.append("Отличное выполнение: полная амплитуда, стабильный ритм, высокая скорость")
                elif result['score'] >= 2:
                    report.append("Удовлетворительное выполнение: небольшие отклонения в амплитуде или ритме")
                else:
                    report.append("Низкое качество выполнения: выраженные нарушения амплитуды или ритма")
                
                report.append("\nДетальная информация:")
                report.append(f"- Полнота амплитуды: {result['analysis'].get('movement_completeness', 0)*100:.1f}%")
                report.append(f"- Согласованность ритма: {result['analysis'].get('rhythm_consistency', 0)*100:.1f}%")
                report.append(f"- Количество циклов: {result['analysis'].get('num_cycles', 0)}")
        
        return "\n".join(report)

    def save_to_csv(self, output_path: str):
        """Сохранение результатов анализа в CSV"""
        if not any(self.hand_results.values()):
            return
        
        results = []
        for hand, result in self.hand_results.items():
            if result is not None:
                row = {
                    'hand': hand.value,
                    'score': result['score'],
                    'movement_completeness': result['analysis'].get('movement_completeness', 0),
                    'rhythm_consistency': result['analysis'].get('rhythm_consistency', 0),
                    'num_cycles': result['analysis'].get('num_cycles', 0),
                    'full_amplitude_ratio': result['analysis'].get('full_amplitude_ratio', 0),
                    'speed_consistency': result['analysis'].get('speed_consistency', 0)
                }
                results.append(row)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ функциональной пробы кисти')
    parser.add_argument('--input', required=True, help='Путь к CSV с данными трекинга')
    parser.add_argument('--fps', type=float, default=30, help='Кадровая частота исходного видео')
    parser.add_argument('--output', default="hand_analysis_results.csv", help='Путь для сохранения результатов анализа')
    
    args = parser.parse_args()
    
    analyzer = HandMovementAnalyzer(args.input, args.fps)
    analyzer.analyze()
    
    # Вывод только указанной информации
    report = analyzer.generate_report()
    if report:
        print(report)
    
    # Сохранение полных результатов в CSV
    analyzer.save_to_csv(args.output)