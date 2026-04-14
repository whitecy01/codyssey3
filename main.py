import json
import os
import time
from typing import Dict, List, Optional, Tuple


Matrix = List[List[float]]
EPSILON = 1e-9
REPEAT_COUNT = 10
STANDARD_LABELS = ("Cross", "X")


class UserExit(Exception):
    pass


def generate_cross_pattern(size: int) -> Matrix:
    center = size // 2
    return [
        [1.0 if row == center or col == center else 0.0 for col in range(size)]
        for row in range(size)
    ]


def generate_x_pattern(size: int) -> Matrix:
    return [
        [1.0 if row == col or row + col == size - 1 else 0.0 for col in range(size)]
        for row in range(size)
    ]


def normalize_expected_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }
    if normalized not in mapping:
        raise ValueError(f"지원하지 않는 expected 라벨입니다: {raw_label}")
    return mapping[normalized]


def normalize_filter_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    mapping = {
        "cross": "Cross",
        "x": "X",
    }
    if normalized not in mapping:
        raise ValueError(f"지원하지 않는 필터 라벨입니다: {raw_label}")
    return mapping[normalized]


def validate_matrix(matrix: Matrix, size: int) -> None:
    if len(matrix) != size:
        raise ValueError(f"행 수 불일치: expected {size}, got {len(matrix)}")
    for row in matrix:
        if len(row) != size:
            raise ValueError(f"열 수 불일치: expected {size}, got {len(row)}")


def mac_operation(pattern: Matrix, filt: Matrix) -> float:
    size = len(pattern)
    total = 0.0
    for row in range(size):
        for col in range(size):
            total += pattern[row][col] * filt[row][col]
    return total


def classify_scores(score_cross: float, score_x: float) -> str:
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def average_mac_time(pattern: Matrix, filters: Dict[str, Matrix], repeat: int = REPEAT_COUNT) -> float:
    elapsed_values = []
    for _ in range(repeat):
        start = time.perf_counter()
        mac_operation(pattern, filters["Cross"])
        mac_operation(pattern, filters["X"])
        end = time.perf_counter()
        elapsed_values.append((end - start) * 1000.0)
    return sum(elapsed_values) / len(elapsed_values)


def read_matrix_from_console(size: int, title: str) -> Matrix:
    print(title)
    matrix: Matrix = []
    row_index = 0
    while row_index < size:
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt) as error:
            raise UserExit() from error
        parts = raw.split()
        if len(parts) != size:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        try:
            row = [float(value) for value in parts]
        except ValueError:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        matrix.append(row)
        row_index += 1
    return matrix


def print_header(title: str) -> None:
    print("\n#----------------------------------------")
    print(title)
    print("#----------------------------------------")


def load_json_data(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_size_from_pattern_key(pattern_key: str) -> int:
    parts = pattern_key.split("_")
    if len(parts) < 3 or parts[0] != "size":
        raise ValueError("패턴 키 형식 오류")
    return int(parts[1])


def pattern_sort_key(pattern_key: str) -> Tuple[int, int]:
    parts = pattern_key.split("_")
    if len(parts) < 3 or parts[0] != "size":
        raise ValueError("패턴 키 형식 오류")
    size = int(parts[1])
    index = int(parts[2])
    return size, index


def normalize_filters(raw_filters: Dict) -> Dict[int, Dict[str, Matrix]]:
    normalized: Dict[int, Dict[str, Matrix]] = {}
    for size_key, filter_entry in raw_filters.items():
        if not size_key.startswith("size_"):
            raise ValueError(f"필터 키 형식 오류: {size_key}")
        size = int(size_key.split("_")[1])
        if not isinstance(filter_entry, dict): # filter_entry가 dict 타입인지 확인하는 코드
            raise ValueError(f"필터 스키마 오류: {size_key}")

        normalized[size] = {}
        for raw_label, matrix in filter_entry.items():
            label = normalize_filter_label(raw_label) # cross -> Cross, x -> X
            validate_matrix(matrix, size) # size x size 크기인지 검사
            normalized[size][label] = matrix # normalized[5]["Cross"] = ... 같은 형태

        if set(normalized[size].keys()) != set(STANDARD_LABELS):
            raise ValueError(f"{size_key} 필터는 cross/x 두 종류를 모두 포함해야 합니다.")
    return normalized


def analyze_json_patterns(file_path: str) -> None:
    print_header("[1] 필터 로드")
    payload = load_json_data(file_path)
    filters_by_size = normalize_filters(payload.get("filters", {}))
    for size in sorted(filters_by_size):
        print(f"✓ size_{size:<2} 필터 로드 완료 (Cross, X)")

    print_header("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = payload.get("patterns", {})
    failures: List[Tuple[str, str]] = []
    passed = 0
    total = 0

    for case_id in sorted(patterns, key=pattern_sort_key):
        total += 1
        case = patterns[case_id]
        print(f"--- {case_id} ---")
        try:
            size = extract_size_from_pattern_key(case_id)
            pattern = case["input"]
            expected = normalize_expected_label(case["expected"])
            validate_matrix(pattern, size)

            if size not in filters_by_size:
                raise ValueError(f"size_{size} 필터가 없습니다.")

            selected_filters = filters_by_size[size]
            validate_matrix(selected_filters["Cross"], size)
            validate_matrix(selected_filters["X"], size)

            score_cross = mac_operation(pattern, selected_filters["Cross"])
            score_x = mac_operation(pattern, selected_filters["X"])
            decision = classify_scores(score_cross, score_x)
            result = "PASS" if decision == expected else "FAIL"
            print(f"Cross 점수: {score_cross}")
            print(f"X 점수: {score_x}")
            print(f"판정: {decision} | expected: {expected} | {result}")

            if result == "PASS":
                passed += 1
            else:
                if decision == "UNDECIDED":
                    failures.append((case_id, "동점(UNDECIDED) 처리 규칙에 따라 FAIL"))
                else:
                    failures.append((case_id, "예상 라벨과 판정 결과가 다릅니다."))
        except KeyError as error:
            message = f"스키마 오류: 누락된 키 {error}"
            failures.append((case_id, message))
            print(f"FAIL: {message}")
        except (TypeError, ValueError) as error:
            failures.append((case_id, str(error)))
            print(f"FAIL: {error}")
        print()

    print_header(f"[3] 성능 분석 (평균/{REPEAT_COUNT}회)")
    performance_rows = build_performance_rows(filters_by_size)
    print_performance_table(performance_rows)

    print_header("[4] 결과 요약")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failures)}개")
    if failures:
        print("\n실패 케이스:")
        for case_id, reason in failures:
            print(f"- {case_id}: {reason}")


def build_performance_rows(filters_by_size: Optional[Dict[int, Dict[str, Matrix]]] = None) -> List[Tuple[int, float, int]]:
    rows: List[Tuple[int, float, int]] = []
    for size in (3, 5, 13, 25):
        if filters_by_size and size in filters_by_size:
            filters = filters_by_size[size]
        else:
            filters = {
                "Cross": generate_cross_pattern(size),
                "X": generate_x_pattern(size),
            }
        pattern = generate_cross_pattern(size)
        average_ms = average_mac_time(pattern, filters)
        rows.append((size, average_ms, size * size))
    return rows


def print_performance_table(rows: List[Tuple[int, float, int]]) -> None:
    print("크기       평균 시간(ms)    연산 횟수")
    print("-------------------------------------")
    for size, average_ms, operations in rows:
        print(f"{size}x{size:<5} {average_ms:<16.6f} {operations}")


def run_console_mode() -> None:
    print_header("[1] 필터 입력")
    filter_a = read_matrix_from_console(3, "필터 A (3줄 입력, 공백 구분)")
    filter_b = read_matrix_from_console(3, "\n필터 B (3줄 입력, 공백 구분)")

    print_header("[2] 패턴 입력")
    pattern = read_matrix_from_console(3, "패턴 (3줄 입력, 공백 구분)")

    filters = {"Cross": filter_a, "X": filter_b}
    score_a = mac_operation(pattern, filter_a)
    score_b = mac_operation(pattern, filter_b)
    average_ms = average_mac_time(pattern, filters)
    decision = classify_scores(score_a, score_b)

    print_header("[3] MAC 결과")
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{REPEAT_COUNT}회): {average_ms:.6f} ms")
    if decision == "UNDECIDED":
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"판정: {'A' if decision == 'Cross' else 'B'}")


def choose_mode() -> str:
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    while True:
        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt) as error:
            raise UserExit() from error
        if choice in {"1", "2"}:
            return choice
        print("입력 오류: 1 또는 2를 선택하세요.")


def main() -> None:
    print("=== Mini NPU Simulator ===\n")
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    while True:
        try:
            choice = choose_mode()
            if choice == "1":
                run_console_mode()
            else:
                analyze_json_patterns(data_file)
            print()
        except UserExit:
            print("\n입력이 중단되어 프로그램을 안전하게 종료합니다.")
            break
        except FileNotFoundError:
            print(f"오류: data.json 파일을 찾을 수 없습니다. 경로: {data_file}")
            break
        except json.JSONDecodeError as error:
            print(f"오류: data.json 파싱에 실패했습니다. {error}")
            break
        except ValueError as error:
            print(f"오류: data.json 스키마가 올바르지 않습니다. {error}")
            break


if __name__ == "__main__":
    main()
