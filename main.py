"""
Shopping Cart Detection - Real-time Video Analysis with GUI
YOLO11n CCTV Augmented 모델 사용

사용법:
1. 프로그램 실행
2. GUI에서 동영상 선택 또는 웹캠 시작
3. 실시간 탐지 결과 확인
"""
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
from pathlib import Path
import time
import threading
from PIL import Image, ImageTk

class ShoppingCartDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shopping Cart Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')

        # 모델 로드 (세그멘테이션 모델)
        print("세그멘테이션 모델 로딩...")
        self.model = YOLO('yolo11s_v9_return_of_1280px/best.pt')
        print("[OK] 모델 로드 완료")

        # 클래스 정보 (Roboflow 형식)
        self.class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
        self.class_colors = {
            0: (255, 0, 0),    # combined: 파랑
            1: (0, 0, 255),    # empty: 빨강
            2: (0, 255, 0)     # fully: 초록
        }

        # 상태 변수
        self.is_running = False
        self.is_paused = False
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.total_frames = 0
        self.total_detections = {'fully': 0, 'empty': 0, 'combined': 0}
        self.current_detections = {'fully': 0, 'empty': 0, 'combined': 0}
        self.fps = 0
        self.seeking = False  # 슬라이더로 이동 중인지

        self.setup_gui()

    def setup_gui(self):
        """GUI 구성"""
        # 상단 제목
        title_frame = tk.Frame(self.root, bg='#1e1e1e', height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🛒 나는 존나 짱이야",
            font=("Arial", 20, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack(pady=10)

        # 메인 컨테이너
        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 좌측: 비디오 디스플레이
        left_frame = tk.Frame(main_container, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.video_label = tk.Label(left_frame, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        # 재생 컨트롤 (슬라이더 + 시간)
        playback_frame = tk.Frame(left_frame, bg='#1e1e1e', height=60)
        playback_frame.pack(fill=tk.X, padx=5, pady=5)
        playback_frame.pack_propagate(False)

        # 시간 표시
        time_frame = tk.Frame(playback_frame, bg='#1e1e1e')
        time_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        self.time_label = tk.Label(
            time_frame,
            text="00:00 / 00:00",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.time_label.pack(side=tk.LEFT)


        # 재생 바 (슬라이더)
        self.progress_slider = tk.Scale(
            playback_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg='#1e1e1e',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#444444',
            activebackground='#2196F3',
            showvalue=False,
            cursor='hand2'
        )
        self.progress_slider.pack(fill=tk.X, padx=5, pady=(0, 5))

        # 슬라이더 이벤트 바인딩
        self.progress_slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # 우측: 컨트롤 패널
        right_frame = tk.Frame(main_container, bg='#1e1e1e', width=300, relief=tk.RIDGE, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # 파일 선택 섹션
        file_section = tk.LabelFrame(
            right_frame,
            text="📁 파일 선택",
            font=("Arial", 12, "bold"),
            bg='#1e1e1e',
            fg='#ffffff',
            padx=10,
            pady=10
        )
        file_section.pack(fill=tk.X, padx=10, pady=10)

        self.select_btn = tk.Button(
            file_section,
            text="동영상 선택",
            command=self.select_video,
            font=("Arial", 10),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        self.select_btn.pack(fill=tk.X, pady=5)

        self.file_label = tk.Label(
            file_section,
            text="선택된 파일 없음",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#888888',
            wraplength=250
        )
        self.file_label.pack(pady=5)

        # 컨트롤 섹션
        control_section = tk.LabelFrame(
            right_frame,
            text="🎮 컨트롤",
            font=("Arial", 12, "bold"),
            bg='#1e1e1e',
            fg='#ffffff',
            padx=10,
            pady=10
        )
        control_section.pack(fill=tk.X, padx=10, pady=10)

        # 시작/정지 버튼
        btn_frame1 = tk.Frame(control_section, bg='#1e1e1e')
        btn_frame1.pack(fill=tk.X, pady=5)

        self.start_btn = tk.Button(
            btn_frame1,
            text="▶ 시작",
            command=self.start_detection,
            font=("Arial", 10),
            bg='#2196F3',
            fg='white',
            activebackground='#0b7dda',
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED,
            width=12,
            pady=8
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = tk.Button(
            btn_frame1,
            text="⏹ 정지",
            command=self.stop_detection,
            font=("Arial", 10),
            bg='#f44336',
            fg='white',
            activebackground='#da190b',
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED,
            width=12,
            pady=8
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # 일시정지 버튼
        self.pause_btn = tk.Button(
            control_section,
            text="⏸ 일시정지",
            command=self.toggle_pause,
            font=("Arial", 10),
            bg='#FF9800',
            fg='white',
            activebackground='#e68900',
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED,
            pady=8
        )
        self.pause_btn.pack(fill=tk.X, pady=5)


        # 통계 섹션
        stats_section = tk.LabelFrame(
            right_frame,
            text="📊 탐지 통계",
            font=("Arial", 12, "bold"),
            bg='#1e1e1e',
            fg='#ffffff',
            padx=10,
            pady=10
        )
        stats_section.pack(fill=tk.X, padx=10, pady=10)

        # 현재 프레임
        current_frame_label = tk.Label(
            stats_section,
            text="현재 프레임:",
            font=("Arial", 10, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        current_frame_label.pack(anchor=tk.W)

        self.fully_current_label = tk.Label(
            stats_section,
            text="🟢 Fully: 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#00ff00'
        )
        self.fully_current_label.pack(anchor=tk.W, pady=2)

        self.empty_current_label = tk.Label(
            stats_section,
            text="🔴 Empty: 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#ff0000'
        )
        self.empty_current_label.pack(anchor=tk.W, pady=2)

        self.combined_current_label = tk.Label(
            stats_section,
            text="🔵 Combined: 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#0000ff'
        )
        self.combined_current_label.pack(anchor=tk.W, pady=2)

        # 구분선
        separator = tk.Frame(stats_section, height=2, bg='#444444')
        separator.pack(fill=tk.X, pady=10)

        # 총 누적
        total_frame_label = tk.Label(
            stats_section,
            text="총 누적:",
            font=("Arial", 10, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        total_frame_label.pack(anchor=tk.W)

        self.fully_total_label = tk.Label(
            stats_section,
            text="🟢 Fully: 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#00ff00'
        )
        self.fully_total_label.pack(anchor=tk.W, pady=2)

        self.empty_total_label = tk.Label(
            stats_section,
            text="🔴 Empty: 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#ff0000'
        )
        self.empty_total_label.pack(anchor=tk.W, pady=2)

        self.combined_total_label = tk.Label(
            stats_section,
            text="🔵 Combined: 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#0000ff'
        )
        self.combined_total_label.pack(anchor=tk.W, pady=2)

        # 초기화 버튼
        reset_btn = tk.Button(
            stats_section,
            text="🔄 통계 초기화",
            command=self.reset_stats,
            font=("Arial", 9),
            bg='#607D8B',
            fg='white',
            activebackground='#4d6370',
            cursor='hand2',
            relief=tk.FLAT,
            pady=5
        )
        reset_btn.pack(fill=tk.X, pady=(10, 0))

        # 정보 섹션
        info_section = tk.LabelFrame(
            right_frame,
            text="ℹ️ 정보",
            font=("Arial", 12, "bold"),
            bg='#1e1e1e',
            fg='#ffffff',
            padx=10,
            pady=10
        )
        info_section.pack(fill=tk.X, padx=10, pady=10)

        self.fps_label = tk.Label(
            info_section,
            text="FPS: 0.0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.fps_label.pack(anchor=tk.W)

        self.frame_label = tk.Label(
            info_section,
            text="프레임: 0 / 0",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.frame_label.pack(anchor=tk.W, pady=2)

        self.progress_label = tk.Label(
            info_section,
            text="진행률: 0%",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.progress_label.pack(anchor=tk.W, pady=2)

        # 하단 상태바
        status_frame = tk.Frame(self.root, bg='#1e1e1e', height=30)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="준비",
            font=("Arial", 9),
            bg='#1e1e1e',
            fg='#888888'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

    def select_video(self):
        """동영상 선택"""
        file_path = filedialog.askopenfilename(
            title="동영상 파일 선택",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.video_path = file_path
            filename = Path(file_path).name
            self.file_label.config(text=filename, fg='#00ff00')
            self.start_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"파일 선택됨: {filename}")

    def start_detection(self):
        """탐지 시작"""
        if not self.video_path:
            messagebox.showwarning("경고", "동영상 파일을 먼저 선택하세요.")
            return

        self.is_running = True
        self.is_paused = False

        # 버튼 상태 변경
        self.select_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)

        self.status_label.config(text="탐지 중...", fg='#00ff00')

        # 백그라운드 스레드로 실행
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()

    def stop_detection(self):
        """탐지 정지"""
        self.is_running = False
        self.is_paused = False

        if self.cap:
            self.cap.release()

        # 버튼 상태 변경
        self.select_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ 일시정지")

        self.status_label.config(text="정지됨", fg='#ff0000')

    def toggle_pause(self):
        """일시정지/재생 토글"""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.config(text="▶ 재생")
            self.status_label.config(text="일시정지됨", fg='#FF9800')
        else:
            self.pause_btn.config(text="⏸ 일시정지")
            self.status_label.config(text="탐지 중...", fg='#00ff00')

    def reset_stats(self):
        """통계 초기화"""
        self.total_detections = {'fully': 0, 'empty': 0, 'combined': 0}
        self.update_stats_display()

    def on_slider_press(self, event):
        """슬라이더 누르기 시작"""
        self.seeking = True

    def on_slider_release(self, event):
        """슬라이더 놓을 때 - 프레임 이동"""
        if not self.cap or not self.is_running:
            self.seeking = False
            return

        target_frame = int(self.progress_slider.get())
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.frame_count = target_frame
        self.seeking = False

    def format_time(self, seconds):
        """초를 MM:SS 형식으로 변환"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def update_stats_display(self):
        """통계 화면 업데이트"""
        # 현재 프레임
        self.fully_current_label.config(text=f"🟢 Fully: {self.current_detections['fully']}")
        self.empty_current_label.config(text=f"🔴 Empty: {self.current_detections['empty']}")
        self.combined_current_label.config(text=f"🔵 Combined: {self.current_detections['combined']}")

        # 총 누적
        self.fully_total_label.config(text=f"🟢 Fully: {self.total_detections['fully']}")
        self.empty_total_label.config(text=f"🔴 Empty: {self.total_detections['empty']}")
        self.combined_total_label.config(text=f"🔵 Combined: {self.total_detections['combined']}")

    def process_video(self):
        """동영상 처리"""
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            messagebox.showerror("오류", "동영상을 열 수 없습니다.")
            self.stop_detection()
            return

        # 동영상 정보
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.frame_count = 0
        frame_times = []

        # 슬라이더 최대값 설정
        self.progress_slider.config(to=self.total_frames)

        # YOLO 추론 주기 (매 N 프레임마다) - 속도 향상
        detect_interval = 10  # 10프레임마다 탐지 (30fps -> 3fps 탐지)

        last_detections = {'fully': 0, 'empty': 0, 'combined': 0}
        last_annotated = None

        while self.is_running:
            if not self.is_paused and not self.seeking:
                ret, frame = self.cap.read()

                if not ret:
                    self.stop_detection()
                    messagebox.showinfo("완료", "동영상 처리가 완료되었습니다.")
                    break

                self.frame_count += 1

                # YOLO 탐지 (간격을 두고 실행)
                if self.frame_count % detect_interval == 0:
                    start_time = time.time()

                    # 세그멘테이션 추론
                    results = self.model.predict(
                        frame,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        retina_masks=True
                    )
                    annotated_frame, detections = self.draw_detections(frame, results)

                    last_detections = detections
                    last_annotated = annotated_frame

                    # 누적 통계 업데이트
                    for cls_name in detections:
                        self.total_detections[cls_name] += detections[cls_name]

                    # FPS 계산
                    end_time = time.time()
                    process_time = end_time - start_time
                    frame_times.append(process_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                else:
                    # 이전 탐지 결과 재사용 (빠른 재생)
                    annotated_frame = last_annotated if last_annotated is not None else frame
                    detections = last_detections

                # 현재 탐지 개수 업데이트
                self.current_detections = detections

                # 처리 FPS 계산
                if frame_times:
                    current_fps = 1.0 / (sum(frame_times) / len(frame_times))
                else:
                    current_fps = 0

                # GUI 업데이트 (간격을 두고)
                if self.frame_count % 5 == 0:  # 5프레임마다 화면 업데이트
                    # 프레임 크기 조정 (GUI에 맞게)
                    display_frame = cv2.resize(annotated_frame, (800, 600))

                    # BGR -> RGB 변환
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    # PIL Image로 변환
                    img = Image.fromarray(display_frame)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # GUI 업데이트
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    # 슬라이더 업데이트 (seeking 중이 아닐 때만)
                    if not self.seeking:
                        self.progress_slider.set(self.frame_count)

                    # 시간 정보 업데이트
                    current_time = self.frame_count / self.fps if self.fps > 0 else 0
                    time_text = f"{self.format_time(current_time)} / {self.format_time(total_duration)}"
                    self.time_label.config(text=time_text)

                    # 정보 업데이트
                    self.fps_label.config(text=f"FPS: {current_fps:.1f}")
                    self.frame_label.config(text=f"프레임: {self.frame_count} / {self.total_frames}")

                    progress = (self.frame_count / self.total_frames) * 100
                    self.progress_label.config(text=f"진행률: {progress:.1f}%")

                    # 통계 업데이트
                    self.update_stats_display()

                # 딜레이 없음 - 최대 속도로 실행
            else:
                # 일시정지 중이거나 seeking 중
                time.sleep(0.01)

    def draw_detections(self, frame, results):
        """프레임에 탐지 결과 그리기 (세그멘테이션 마스크 + 바운딩 박스)"""
        annotated_frame = frame.copy()
        detections = {'fully': 0, 'empty': 0, 'combined': 0}

        # 마스크 오버레이용
        mask_overlay = annotated_frame.copy()

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    color = self.class_colors[cls]

                    # 세그멘테이션 마스크 그리기
                    if masks is not None and idx < len(masks):
                        mask = masks[idx].data[0].cpu().numpy()
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_bool = mask_resized > 0.5
                        mask_overlay[mask_bool] = color

                    # 바운딩 박스
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # 라벨
                    label = f"{self.class_names[cls]} {conf:.2f}"

                    # 배경
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)

                    # 텍스트
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 통계
                    detections[self.class_names[cls]] += 1

        # 마스크 오버레이 합성 (투명도 40%)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.6, mask_overlay, 0.4, 0)

        return annotated_frame, detections

def main():
    root = tk.Tk()
    app = ShoppingCartDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
