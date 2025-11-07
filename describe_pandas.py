#!/usr/bin/env python3
"""
describe_pandas.py - Pandas를 사용한 버전 (비교용)

우리가 만든 describe.py와 결과가 같은지 확인하기 위한 파일
"""

import sys
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python describe_pandas.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    print(f"📂 Pandas로 파일 읽는 중: {filename}")

    # Pandas로 CSV 읽기 (한 줄이면 끝!)
    df = pd.read_csv(filename)

    print(f"✅ 총 {len(df)}개 행, {len(df.columns)}개 열 읽음")

    # 숫자 컬럼만 선택
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Index 컬럼 제거 (있다면)
    if 'Index' in numeric_df.columns:
        numeric_df = numeric_df.drop('Index', axis=1)

    print(f"\n🔢 숫자 컬럼: {len(numeric_df.columns)}개")

    # Pandas의 describe() 함수 사용!
    print("\n" + "=" * 150)
    print("📊 Pandas describe() 결과:")
    print("=" * 150)

    # describe() 호출 (이게 전부!)
    stats = df.describe()

    # Index 컬럼 제거
    if 'Index' in stats.columns:
        stats = stats.drop('Index', axis=1)

    # 출력 형식을 우리 것과 비슷하게 조정
    print(stats.to_string())
    print("=" * 150)


if __name__ == "__main__":
    main()
