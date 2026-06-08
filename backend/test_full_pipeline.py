import os
from shot_detector_video import BasketballShotDetector
from video_processor import VideoProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
TEST_VIDEO = os.path.join(BASE_DIR, 'test_files', 'video_test_2.mp4')
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
ANNOTATED_PATH = os.path.join(OUTPUT_DIR, 'video_test_2_annotated.mp4')
HIGHLIGHT_PATH = os.path.join(OUTPUT_DIR, 'basketball_highlight.mp4')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 步骤1: 检测进球（开启标注并输出标注视频）
print("步骤1: 检测进球时刻并生成标注视频...")
detector = BasketballShotDetector(model_path=MODEL_PATH)
result = detector.detect_shots_with_clips(
    TEST_VIDEO,
    before_seconds=3,
    after_seconds=1,
    annotate=True,
    annotated_output_path=ANNOTATED_PATH,
)

print(f"\n检测结果:")
print(f"  总投篮: {result['stats']['total_attempts']}")
print(f"  进球数: {result['stats']['total_makes']}")
print(f"  命中率: {result['stats']['accuracy']}%")
if result.get('annotated_video'):
    print(f"  标注视频: {result['annotated_video']}")
elif os.path.exists(ANNOTATED_PATH):
    print(f"  标注视频: {ANNOTATED_PATH}")

# 步骤2: 生成集锦视频（从原视频剪辑，保留原始音频）
print("\n步骤2: 生成集锦视频（使用原视频作为剪辑源）...")
processor = VideoProcessor()

# 注意：output_path 必须是完整的文件路径（包含文件名），不能只是目录
output = processor.process_video_full_pipeline(
    video_path=TEST_VIDEO,
    timestamps=result['made_shots'],
    output_path=HIGHLIGHT_PATH,
    before=3,
    after=1
)

if output['success']:
    print(f"\n🎉 集锦生成成功!")
    print(f"📁 输出文件: {output['output_file']}")
    
    # 显示文件大小
    if os.path.exists(output['output_file']):
        size_mb = os.path.getsize(output['output_file']) / (1024 * 1024)
        print(f"📦 文件大小: {size_mb:.2f} MB")
else:
    print(f"\n❌ 失败: {output['error']}")
