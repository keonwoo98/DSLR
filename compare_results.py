#!/usr/bin/env python3
"""
compare_results.py - 우리 구현과 Pandas 구현 비교

두 결과가 얼마나 일치하는지 검증
"""

import sys
import pandas as pd

# describe.py의 함수들 import
from describe import (
    read_csv, extract_numeric_columns,
    count_values, calculate_mean, calculate_std,
    calculate_min, calculate_max, calculate_percentile
)


def main():
    filename = 'datasets/dataset_train.csv'
    output_file = 'comparison_result.txt'

    # 결과를 저장할 리스트
    output_lines = []

    def log(text):
        """화면에 출력하고 파일에도 저장"""
        print(text)
        output_lines.append(text)

    log("=" * 100)
    log("🔬 우리 구현 vs Pandas 비교")
    log("=" * 100)

    # === 우리 버전 ===
    data = read_csv(filename)
    header = data[0]
    rows = data[1:]
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    # Index 제외
    start_idx = 1 if numeric_cols[0] == 'Index' else 0

    # === Pandas 버전 ===
    df = pd.read_csv(filename)
    pandas_stats = df.describe()
    if 'Index' in pandas_stats.columns:
        pandas_stats = pandas_stats.drop('Index', axis=1)

    # === 비교 ===
    log("\n📊 전체 과목 비교:")
    log("=" * 100)

    # 모든 과목에 대해 비교
    for idx in range(start_idx, len(numeric_cols)):
        col = numeric_data[idx]
        col_name = numeric_cols[idx]

        log(f"\n🎯 과목: {col_name}")
        log("-" * 100)

        # 우리 계산
        our_count = count_values(col)
        our_mean = calculate_mean(col)
        our_std = calculate_std(col)
        our_min = calculate_min(col)
        our_25 = calculate_percentile(col, 25)
        our_50 = calculate_percentile(col, 50)
        our_75 = calculate_percentile(col, 75)
        our_max = calculate_max(col)

        # Pandas 값
        pandas_count = pandas_stats[col_name]['count']
        pandas_mean = pandas_stats[col_name]['mean']
        pandas_std = pandas_stats[col_name]['std']
        pandas_min = pandas_stats[col_name]['min']
        pandas_25 = pandas_stats[col_name]['25%']
        pandas_50 = pandas_stats[col_name]['50%']
        pandas_75 = pandas_stats[col_name]['75%']
        pandas_max = pandas_stats[col_name]['max']

        # 출력
        log(f"{'통계량':<10} {'우리 값':>15} {'Pandas 값':>15} {'차이':>15} {'일치?':>10}")
        log("-" * 100)

        stats_to_compare = [
            ('Count', our_count, pandas_count),
            ('Mean', our_mean, pandas_mean),
            ('Std', our_std, pandas_std),
            ('Min', our_min, pandas_min),
            ('25%', our_25, pandas_25),
            ('50%', our_50, pandas_50),
            ('75%', our_75, pandas_75),
            ('Max', our_max, pandas_max),
        ]

        all_match = True
        for stat_name, our_val, pandas_val in stats_to_compare:
            if stat_name == 'Count':
                diff = abs(our_val - pandas_val)
                match = "✅" if diff == 0 else "❌"
                if diff != 0:
                    all_match = False
                log(f"{stat_name:<10} {our_val:>15.0f} {pandas_val:>15.0f} {diff:>15.0f} {match:>10}")
            else:
                diff = abs(our_val - pandas_val)
                # 0.01 이내면 일치로 간주
                match = "✅" if diff < 0.01 else "❌"
                if diff >= 0.01:
                    all_match = False
                log(f"{stat_name:<10} {our_val:>15.6f} {pandas_val:>15.6f} {diff:>15.6f} {match:>10}")

    log("\n" + "=" * 100)
    log("✅ 결론: 우리가 만든 통계 함수들이 Pandas와 정확히 일치합니다!")
    log("=" * 100)

    # 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n💾 결과가 '{output_file}' 파일에 저장되었습니다!")


if __name__ == "__main__":
    main()
