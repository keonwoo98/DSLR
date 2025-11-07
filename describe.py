#!/usr/bin/env python3
"""
describe.py - 데이터셋의 통계 정보를 출력하는 프로그램

pandas의 describe() 함수와 유사하게 동작:
- Count (개수)
- Mean (평균)
- Std (표준편차)
- Min (최소값)
- 25% (1사분위수)
- 50% (중앙값)
- 75% (3사분위수)
- Max (최대값)
"""

import sys


# ==================== 통계 함수들 ====================

def count_values(column):
    """
    None이 아닌 값의 개수를 세기

    Args:
        column: 숫자 리스트 (None 포함 가능)

    Returns:
        int: 유효한 값의 개수
    """
    count = 0
    for val in column:
        if val is not None:
            count += 1
    return count


def calculate_mean(column):
    """
    평균 계산
    공식: (모든 값의 합) / (값의 개수)

    Args:
        column: 숫자 리스트

    Returns:
        float: 평균값 (값이 없으면 None)
    """
    # None이 아닌 값만 추출
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    # 합계 구하기
    total = 0
    for val in valid_values:
        total += val

    # 평균 = 합계 / 개수
    mean = total / len(valid_values)
    return mean


def calculate_min(column):
    """
    최소값 찾기

    Args:
        column: 숫자 리스트

    Returns:
        float: 최소값
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    min_val = valid_values[0]
    for val in valid_values:
        if val < min_val:
            min_val = val

    return min_val


def calculate_max(column):
    """
    최대값 찾기

    Args:
        column: 숫자 리스트

    Returns:
        float: 최대값
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    max_val = valid_values[0]
    for val in valid_values:
        if val > max_val:
            max_val = val

    return max_val


def calculate_std(column):
    """
    표준편차 계산

    공식:
    1. 각 값과 평균의 차이를 제곱 → (x - mean)²
    2. 모든 제곱의 평균 → variance (분산)
    3. 분산의 제곱근 → std (표준편차)

    Args:
        column: 숫자 리스트

    Returns:
        float: 표준편차
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) < 2:
        return None

    # 1단계: 평균 구하기
    mean = calculate_mean(column)

    # 2단계: 각 값과 평균의 차이를 제곱
    squared_diffs = []
    for val in valid_values:
        diff = val - mean
        squared_diff = diff * diff  # diff ** 2
        squared_diffs.append(squared_diff)

    # 3단계: 제곱의 평균 (분산)
    # Pandas와 동일하게 샘플 표준편차 사용 (n-1로 나눔)
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)

    # 4단계: 제곱근 (표준편차)
    std = variance ** 0.5  # sqrt(variance)

    return std


def calculate_percentile(column, percentile):
    """
    백분위수 계산 (예: 25%, 50%, 75%)

    백분위수: 데이터를 정렬했을 때 특정 위치의 값
    예: 25% = 하위 25% 위치의 값

    Args:
        column: 숫자 리스트
        percentile: 0~100 사이의 숫자 (예: 25, 50, 75)

    Returns:
        float: 백분위수 값
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    # 1단계: 정렬 (작은 것부터 큰 순서로)
    sorted_values = sorted(valid_values)

    # 2단계: 위치 계산
    # 예: 100개 데이터의 25% = 25번째 위치
    index = (percentile / 100) * (len(sorted_values) - 1)

    # 3단계: 정수가 아니면 보간(interpolation)
    if index == int(index):
        return sorted_values[int(index)]
    else:
        # 예: 25.5번째 → 25번째와 26번째의 평균
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        weight = index - lower_idx

        return sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight


def print_stats_table(feature_names, stats):
    """
    통계를 pandas.describe()처럼 표 형태로 출력

    Args:
        feature_names: 컬럼명 리스트
        stats: 통계 딕셔너리
    """
    print("\n" + "=" * 150)

    # 헤더 출력 (첫 번째 열은 통계명)
    header = f"{'':15}"  # 첫 열은 비워둠
    for name in feature_names:
        # 컬럼명이 길면 줄임
        short_name = name[:12] + '...' if len(name) > 15 else name
        header += f"{short_name:>15}"
    print(header)
    print("-" * 150)

    # 각 통계 행 출력
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    for stat_name in stat_names:
        row = f"{stat_name:15}"
        for value in stats[stat_name]:
            if value is None:
                row += f"{'NaN':>15}"
            elif stat_name == 'Count':
                # Count는 정수로
                row += f"{int(value):>15}"
            else:
                # 나머지는 소수점 2자리
                row += f"{value:>15.2f}"
        print(row)

    print("=" * 150)


def read_csv(filename):
    """
    CSV 파일을 읽어서 2차원 리스트로 반환

    Args:
        filename: CSV 파일 경로

    Returns:
        list: [헤더, 데이터행1, 데이터행2, ...]
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # 각 줄을 콤마로 분리
        data = []
        for line in lines:
            # 줄바꿈 제거 후 콤마로 split
            row = line.strip().split(',')
            data.append(row)

        return data

    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: 파일을 읽는 중 오류 발생: {e}")
        sys.exit(1)


def is_numeric_column(values):
    """
    컬럼이 숫자 데이터인지 확인

    Args:
        values: 컬럼의 값들 (리스트)

    Returns:
        bool: 숫자 컬럼이면 True
    """
    numeric_count = 0
    total_count = 0

    for val in values:
        if val == '' or val == 'nan':  # 빈 값이나 nan은 건너뛰기
            continue
        total_count += 1
        try:
            float(val)  # 숫자로 변환 시도
            numeric_count += 1
        except ValueError:
            pass

    # 50% 이상이 숫자면 숫자 컬럼으로 판단
    if total_count == 0:
        return False
    return (numeric_count / total_count) > 0.5


def extract_numeric_columns(header, rows):
    """
    숫자 데이터를 가진 컬럼들만 추출

    Args:
        header: 컬럼명 리스트
        rows: 데이터 행들

    Returns:
        tuple: (숫자 컬럼명 리스트, 숫자 데이터 2차원 리스트)
    """
    numeric_cols = []
    numeric_indices = []

    # 각 컬럼을 확인해서 숫자 컬럼 찾기
    for col_idx in range(len(header)):
        # 해당 컬럼의 모든 값들 추출
        column_values = [row[col_idx] for row in rows]

        if is_numeric_column(column_values):
            numeric_cols.append(header[col_idx])
            numeric_indices.append(col_idx)

    # 숫자 데이터만 추출
    numeric_data = []
    for col_idx in numeric_indices:
        column = []
        for row in rows:
            val = row[col_idx]
            # 빈 값이나 빈 문자열은 None으로 처리
            if val == '' or val == 'nan':
                column.append(None)
            else:
                try:
                    column.append(float(val))
                except ValueError:
                    column.append(None)
        numeric_data.append(column)

    return numeric_cols, numeric_data


def main():
    # 사용법 확인
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    # 1단계: 파일 읽기
    print(f"📂 파일 읽는 중: {filename}")
    data = read_csv(filename)

    # 헤더와 데이터 분리
    header = data[0]
    rows = data[1:]

    print(f"✅ 총 {len(rows)}개 행, {len(header)}개 열 읽음")

    # 2단계: 숫자 컬럼만 추출
    print(f"\n🔢 숫자 컬럼 추출 중...")
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    print(f"✅ {len(numeric_cols)}개 숫자 컬럼 발견:")
    for col_name in numeric_cols:
        print(f"   - {col_name}")

    # 3단계: 각 컬럼의 통계 계산
    print(f"\n📊 통계 계산 중...")

    # Index 컬럼은 제외 (의미 없는 숫자)
    start_idx = 1 if numeric_cols[0] == 'Index' else 0

    # 모든 컬럼의 통계 저장
    stats = {
        'Count': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        '25%': [],
        '50%': [],
        '75%': [],
        'Max': []
    }

    # 각 컬럼별로 통계 계산
    feature_names = []
    for idx in range(start_idx, len(numeric_cols)):
        col = numeric_data[idx]
        col_name = numeric_cols[idx]
        feature_names.append(col_name)

        stats['Count'].append(count_values(col))
        stats['Mean'].append(calculate_mean(col))
        stats['Std'].append(calculate_std(col))
        stats['Min'].append(calculate_min(col))
        stats['25%'].append(calculate_percentile(col, 25))
        stats['50%'].append(calculate_percentile(col, 50))
        stats['75%'].append(calculate_percentile(col, 75))
        stats['Max'].append(calculate_max(col))

    # 4단계: 예쁜 표로 출력
    print_stats_table(feature_names, stats)


if __name__ == "__main__":
    main()
