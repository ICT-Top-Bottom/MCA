import os
from collections import Counter

# e_ver2 파일 찾기
e_files = [f for f in os.listdir('images') if f.startswith('e_') and 'ver2' in f]

# 각 파일의 클래스 확인
classes = []
for img_file in e_files:
    label_file = img_file.replace('.jpg', '.txt')
    label_path = os.path.join('labels', label_file)

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    classes.append(cls)
                    break  # 첫 번째 객체만

class_counter = Counter(classes)

print(f"e_XX_ver2.jpg files: {len(e_files)}")
print(f"\nClass distribution:")
print(f"  Class 0 (fully): {class_counter.get(0, 0)}")
print(f"  Class 1 (empty): {class_counter.get(1, 0)}")
print(f"  Class 2 (combined): {class_counter.get(2, 0)}")

print(f"\nAll e_ver2 files ({len(e_files)}):")
for f in sorted(e_files):
    print(f"  {f}")
