from pathlib import Path

data_valid = set([f.name for f in Path('Data/valid/images').glob('*.jpg')])
polygon_valid = set([f.name for f in Path('Data_polygon/valid/images').glob('*.jpg')])

deleted_files = data_valid - polygon_valid

print(f"삭제된 파일 ({len(deleted_files)}개):")
for f in sorted(deleted_files):
    print(f"  {f}")
