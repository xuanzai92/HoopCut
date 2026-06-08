"""
独立测试进球检测功能（不涉及 Flask）
"""
import sys
import os
import json

# 获取当前脚本所在目录的父目录（即backend目录）
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将backend目录添加到Python搜索路径
sys.path.append(backend_dir)

from shot_detector_video import BasketballShotDetector

def parse_target_box(raw_box):
    if not raw_box:
        return None

    try:
        parsed = json.loads(raw_box)
    except json.JSONDecodeError as error:
        raise ValueError(f'无法解析 targetPlayerBox JSON: {error}') from error

    if not isinstance(parsed, dict):
        raise ValueError('targetPlayerBox 必须是一个 JSON 对象')

    return parsed

def test_detection(video_path, target_box=None):
    """测试视频检测"""
    print("="*60)
    print("🏀 篮球进球检测测试")
    print("="*60)
    
    # 检查文件
    if not os.path.exists(video_path):
        print(f"❌ 错误: 视频文件不存在: {video_path}")
        return
    
    print(f"📹 视频文件: {video_path}")
    print(f"📦 文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print()
    
    try:
        # 创建检测器
        print("🔧 初始化检测器...")
        model_path = os.path.join(backend_dir, 'best.pt')  # 利用已获取的backend_dir拼接绝对路径
        detector = BasketballShotDetector(model_path=model_path)
        print("✅ 检测器初始化成功")
        print()
        
        # 执行检测
        print("🎯 开始检测进球...")
        print("-"*60)
        results, tracking = detector.detect_shots(video_path, target_player_box=target_box)
        print("-"*60)
        print()
        
        # 显示结果
        print("📊 检测结果:")
        print(f"   检测到投篮: {len(results)} 个")
        if tracking.get('enabled'):
            print(f"   目标球员覆盖率: {tracking['coverage'] * 100:.1f}%")
        elif tracking.get('error'):
            print(f"   目标球员跟踪未启用: {tracking['error']}")
        
        if results:
            print("\n   投篮时间点:")
            for i, shot in enumerate(results, 1):
                owner = shot.get('owner', 'unknown')
                confidence = shot.get('owner_confidence', 0.0)
                print(
                    f"   {i}. {shot['timestamp']:.2f}s "
                    f"(第 {shot['frame']} 帧, {'进球' if shot['made'] else '未进'}, "
                    f"owner={owner}, conf={confidence:.2f})"
                )
        else:
            print("   ⚠️  未检测到投篮")
        
        print("\n✅ 测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_detector_only.py <video_path>")
        print("示例: python test_detector_only.py test_video.mp4 '{\"x\":100,\"y\":80,\"width\":120,\"height\":280,\"frameWidth\":1280,\"frameHeight\":720}'")
        sys.exit(1)

    raw_target_box = sys.argv[2] if len(sys.argv) > 2 else None
    test_detection(sys.argv[1], parse_target_box(raw_target_box))
