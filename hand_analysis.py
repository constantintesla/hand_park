import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from enum import Enum
import os
import matplotlib.pyplot as plt

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

        # Параметры детекции сжатий (антидребезг)
        # Гистерезис по амплитуде: вход в "кулак" и выход из него разделены
        self.FIST_ENTER_THRESHOLD = 0.65
        self.FIST_EXIT_THRESHOLD = 0.45
        # Минимальная длительность стабильного состояния (в секундах)
        self.MIN_STATE_DURATION_S = 0.10
        # Рефрактерный интервал между срабатываниями (в секундах)
        self.REFRACTORY_S = 0.20

    def _detect_hand(self, landmarks: np.ndarray) -> Hand:
        """Определение левой/правой руки по положению landmarks"""
        wrist = landmarks[0]
        thumb_base = landmarks[1]
        pinky_base = landmarks[17]
        
        vec_thumb = thumb_base - wrist
        vec_pinky = pinky_base - wrist
        
        cross_z = vec_thumb[0] * vec_pinky[1] - vec_thumb[1] * vec_pinky[0]
        
        return Hand.LEFT if cross_z > 0 else Hand.RIGHT

    def _hand_from_label(self, label: str) -> Hand:
        label = (label or '').strip().lower()
        return Hand.LEFT if label == 'left' else Hand.RIGHT

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

    def _smooth(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """Скользящее среднее для уменьшения шума."""
        if window <= 1 or len(values) == 0:
            return values
        window = min(window, max(1, len(values)))
        kernel = np.ones(window, dtype=float) / window
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(values)]

    def _compute_thresholds(self, amplitudes: np.ndarray) -> Tuple[float, float]:
        """Возвращает (enter, exit) пороги. Если максимум ниже базового порога
        или размах мал, использует относительные пороги от наблюдаемого диапазона."""
        if len(amplitudes) == 0:
            return self.FIST_ENTER_THRESHOLD, self.FIST_EXIT_THRESHOLD
        observed_min = float(np.min(amplitudes))
        observed_max = float(np.max(amplitudes))
        observed_range = observed_max - observed_min
        enter_thr = self.FIST_ENTER_THRESHOLD
        exit_thr = self.FIST_EXIT_THRESHOLD
        if observed_max < self.FIST_ENTER_THRESHOLD or observed_range < 0.25:
            rel_enter = 0.75
            rel_exit = 0.55
            enter_thr = observed_min + rel_enter * observed_range
            exit_thr = observed_min + rel_exit * observed_range
            if enter_thr <= exit_thr:
                enter_thr = exit_thr + max(0.02, 0.1 * observed_range)
        return enter_thr, exit_thr

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

    def _count_fist_closures(self, amplitudes: np.ndarray, enter_thr: float, exit_thr: float) -> Tuple[int, List[int]]:
        """Подсчет реальных сжатий кулака по амплитуде с гистерезисом и антидребезгом.

        Возвращает количество срабатываний и индексы кадров, где зафиксированы сжатия.
        """
        if len(amplitudes) == 0:
            return 0, []

        min_state_frames = max(1, int(self.MIN_STATE_DURATION_S * self.fps))
        refractory_frames = max(1, int(self.REFRACTORY_S * self.fps))

        state_is_fist = False
        state_start_idx = 0
        last_trigger_idx = -10**9
        closures = 0
        closure_indices: List[int] = []

        # Используем скользящее состояние: open -> fist при превышении ENTER,
        # fist -> open при падении ниже EXIT. Требуем минимальную длительность состояния.
        for i, amp in enumerate(amplitudes):
            if not state_is_fist:
                # Переход в сжатие
                if amp >= enter_thr:
                    # проверим устойчивость: предыдущее состояние должно длиться минимум min_state_frames
                    if i - state_start_idx >= min_state_frames:
                        # проверим рефрактерность
                        if i - last_trigger_idx >= refractory_frames:
                            state_is_fist = True
                            state_start_idx = i
                            closures += 1
                            last_trigger_idx = i
                            closure_indices.append(i)
                        else:
                            # игнорируем из-за рефрактерного окна
                            pass
                # иначе остаемся в open
            else:
                # Переход в раскрытие
                if amp <= exit_thr:
                    if i - state_start_idx >= min_state_frames:
                        state_is_fist = False
                        state_start_idx = i

        return closures, closure_indices

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

    def _clinical_score(self, technical_score: int) -> int:
        """Инвертированная клиническая шкала: 4 (хуже) .. 0 (лучше)."""
        return max(0, min(4, 4 - technical_score))

    def _detect_incomplete_and_localize(self, amplitudes: np.ndarray, closures: List[int], enter_thr: float, exit_thr: float) -> Dict:
        """Оценка неполных циклов и их распределение по третям времени теста."""
        n = len(amplitudes)
        if n == 0:
            return {'incomplete_ratio': 0.0, 'violations_early': 0.0, 'violations_mid': 0.0, 'violations_late': 0.0}
        # Неполный цикл: пики ниже enter_thr или низ не опускается ниже exit_thr между срабатываниями
        incomplete_flags = []
        last_idx = 0
        for c in closures:
            seg = amplitudes[last_idx:c+1]
            peak = float(np.max(seg)) if len(seg) else 0.0
            trough = float(np.min(seg)) if len(seg) else 1.0
            is_incomplete = peak < enter_thr or trough > exit_thr
            incomplete_flags.append((c, is_incomplete))
            last_idx = c
        # Локализация по третям времени
        thirds = [int(n/3), int(2*n/3), n]
        counts = {'early': 0, 'mid': 0, 'late': 0}
        viols = {'early': 0, 'mid': 0, 'late': 0}
        for idx, flag in incomplete_flags:
            bucket = 'late' if idx >= thirds[1] else ('mid' if idx >= thirds[0] else 'early')
            counts[bucket] += 1
            if flag:
                viols[bucket] += 1
        total = max(1, len(incomplete_flags))
        return {
            'incomplete_ratio': sum(1 for _, f in incomplete_flags if f) / total,
            'violations_early': (viols['early'] / max(1, counts['early'])) if counts['early'] else 0.0,
            'violations_mid': (viols['mid'] / max(1, counts['mid'])) if counts['mid'] else 0.0,
            'violations_late': (viols['late'] / max(1, counts['late'])) if counts['late'] else 0.0
        }

    def _frequency_and_trends(self, timestamps: np.ndarray, closures: List[int], amplitudes: np.ndarray) -> Dict:
        """Частота (Гц), средняя/медианная длительность цикла, линейные тренды амплитуды и скорости."""
        result = {
            'frequency_hz': 0.0,
            'cycle_duration_mean_s': 0.0,
            'cycle_duration_median_s': 0.0,
            'amplitude_trend_per_s': 0.0,
            'speed_trend_per_s': 0.0
        }
        if timestamps is None or len(timestamps) == 0:
            return result
        if len(closures) >= 2:
            times = timestamps[closures]
            intervals = np.diff(times)
            result['cycle_duration_mean_s'] = float(np.mean(intervals))
            result['cycle_duration_median_s'] = float(np.median(intervals))
            result['frequency_hz'] = 1.0 / result['cycle_duration_mean_s'] if result['cycle_duration_mean_s'] > 0 else 0.0
        # Тренд амплитуды
        t = timestamps - timestamps[0]
        if len(t) >= 2:
            A = np.vstack([t, np.ones_like(t)]).T
            k_amp, _ = np.linalg.lstsq(A, amplitudes, rcond=None)[0]
            result['amplitude_trend_per_s'] = float(k_amp)
            # Скорость как |delta amp| * fps (грубая оценка)
            speed = np.abs(np.diff(amplitudes)) * self.fps
            t_mid = (t[1:] + t[:-1]) / 2.0
            if len(t_mid) >= 2:
                A2 = np.vstack([t_mid, np.ones_like(t_mid)]).T
                k_speed, _ = np.linalg.lstsq(A2, speed, rcond=None)[0]
                result['speed_trend_per_s'] = float(k_speed)
        return result

    def analyze(self):
        """Основной метод анализа данных"""
        if self.data.empty:
            return self.hand_results
            
        # Если в CSV есть столбец 'hand', анализируем по группам; иначе одна рука как раньше
        if 'hand' in self.data.columns:
            groups = self.data.groupby('hand')
        else:
            groups = [(None, self.data)]

        for hand_label, df in groups:
            try:
                landmarks_per_frame = []
                for i in range(len(df)):
                    frame_landmarks = []
                    for j in range(21):
                        x = df[f'hand_{j}_x'].values[i]
                        y = df[f'hand_{j}_y'].values[i]
                        z = df[f'hand_{j}_z'].values[i] if f'hand_{j}_z' in df.columns else 0
                        frame_landmarks.append(np.array([x, y, z]))
                    landmarks_per_frame.append(np.array(frame_landmarks))
                if not landmarks_per_frame:
                    continue

                # Определяем руку
                hand = self._hand_from_label(hand_label) if hand_label is not None else self._detect_hand(landmarks_per_frame[0])

                normalized_landmarks = [self._normalize_landmarks(lm) for lm in landmarks_per_frame]
                amplitudes = np.array([self._calculate_fist_amplitude(lm) for lm in normalized_landmarks])
                amplitudes = self._smooth(amplitudes, window=max(3, int(self.fps // 10)))

                phases = self._detect_movement_phases(amplitudes)
                used_enter_thr, used_exit_thr = self._compute_thresholds(amplitudes)
                num_closures, closure_indices = self._count_fist_closures(amplitudes, used_enter_thr, used_exit_thr)
                phase_analysis = self._analyze_phases(phases)
                phase_analysis['num_cycles'] = num_closures
                score = self._classify_performance(phase_analysis)

                timestamps = df['timestamp'].values
                extra = self._detect_incomplete_and_localize(amplitudes, closure_indices, used_enter_thr, used_exit_thr)
                trends = self._frequency_and_trends(timestamps, closure_indices, amplitudes)

                self.hand_results[hand] = {
                    'score': score,
                    'clinical_score': self._clinical_score(score),
                    'phases': phases,
                    'amplitudes': amplitudes,
                    'analysis': {**phase_analysis, **extra, **trends},
                    'timestamps': timestamps,
                    'closure_indices': closure_indices,
                    'used_enter_threshold': used_enter_thr,
                    'used_exit_threshold': used_exit_thr
                }
            except Exception as e:
                print(f"Ошибка при анализе ({hand_label}): {str(e)}")

        return self.hand_results

    def plot_debug(self):
        """Отладочный график амплитуды с порогами и отметками сжатий"""
        if not any(self.hand_results.values()):
            print("Нет результатов для визуализации")
            return
        # Покажем правую руку, если есть; иначе первую доступную
        result = None
        if self.hand_results.get(Hand.RIGHT):
            result = self.hand_results[Hand.RIGHT]
        else:
            for v in self.hand_results.values():
                if v is not None:
                    result = v
                    break
        if result is None:
            print("Нет данных для графика")
            return

        amplitudes = result['amplitudes']
        timestamps = result.get('timestamps')
        if timestamps is None or len(timestamps) != len(amplitudes):
            t = np.arange(len(amplitudes)) / self.fps
        else:
            t = timestamps
        closures = result.get('closure_indices', [])

        plt.figure(figsize=(12, 5))
        plt.plot(t, amplitudes, label='Амплитуда (0-1)')
        # Покажем использованные пороги (адаптивные или базовые)
        used_enter = result.get('used_enter_threshold', self.FIST_ENTER_THRESHOLD)
        used_exit = result.get('used_exit_threshold', self.FIST_EXIT_THRESHOLD)
        plt.axhline(used_enter, color='red', linestyle='--', label='Порог входа (исп.)')
        plt.axhline(used_exit, color='orange', linestyle='--', label='Порог выхода (исп.)')
        if closures:
            plt.scatter([t[i] for i in closures], [amplitudes[i] for i in closures],
                        color='green', s=30, zorder=5, label='Срабатывания (сжатия)')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда сжатия')
        plt.title('Отладочный график амплитуды и детекции сжатий')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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
                report.append(f"Оценка выполнения (техн.): {result['score']}/4 | Клиническая: {result.get('clinical_score', 0)}/4")
                
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
                report.append(f"- Частота: {result['analysis'].get('frequency_hz', 0):.2f} Гц")
                report.append(f"- Длительность цикла: ср. {result['analysis'].get('cycle_duration_mean_s', 0):.2f} с, мед. {result['analysis'].get('cycle_duration_median_s', 0):.2f} с")
                report.append(f"- Неполные циклы: {result['analysis'].get('incomplete_ratio', 0)*100:.1f}% (early {result['analysis'].get('violations_early', 0)*100:.0f}%, mid {result['analysis'].get('violations_mid', 0)*100:.0f}%, late {result['analysis'].get('violations_late', 0)*100:.0f}%)")
                report.append(f"- Тренд амплитуды: {result['analysis'].get('amplitude_trend_per_s', 0):+.3f}/с; тренд скорости: {result['analysis'].get('speed_trend_per_s', 0):+.3f}/с")
        
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
    parser.add_argument('--plot', action='store_true', help='Показать отладочные графики')
    
    args = parser.parse_args()
    
    analyzer = HandMovementAnalyzer(args.input, args.fps)
    analyzer.analyze()
    
    # Вывод только указанной информации
    report = analyzer.generate_report()
    if report:
        print(report)
    
    # Сохранение полных результатов в CSV
    analyzer.save_to_csv(args.output)
    
    # Отладочный график
    if args.plot:
        analyzer.plot_debug()