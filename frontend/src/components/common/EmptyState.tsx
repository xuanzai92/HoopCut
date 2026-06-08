/**
 * 空状态组件
 * 用于显示各种空状态和占位内容
 */
import React from 'react';
import { Button, Card } from '@heroui/react';
import { FileX, History, Inbox, Search, Video } from 'lucide-react';

interface EmptyStateProps {
  type?: 'default' | 'video' | 'history' | 'search' | 'upload';
  title?: string;
  description?: string;
  actionText?: string;
  onAction?: () => void;
  className?: string;
  image?: React.ReactNode;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  type = 'default',
  title,
  description,
  actionText,
  onAction,
  className = '',
  image,
}) => {
  const getEmptyConfig = () => {
    switch (type) {
      case 'video':
        return {
          icon: <Video size={36} className="text-slate-400" />,
          title: title || '暂无视频',
          description: description || '还没有上传任何视频文件',
          actionText: actionText || '上传视频',
        };

      case 'history':
        return {
          icon: <History size={36} className="text-slate-400" />,
          title: title || '暂无历史记录',
          description: description || '还没有处理过任何视频',
          actionText: actionText || '开始处理',
        };

      case 'search':
        return {
          icon: <Search size={36} className="text-slate-400" />,
          title: title || '未找到相关内容',
          description: description || '请尝试调整搜索条件',
          actionText: actionText || '重新搜索',
        };

      case 'upload':
        return {
          icon: <Inbox size={36} className="text-slate-400" />,
          title: title || '拖拽文件到此处',
          description: description || '或点击选择文件上传',
          actionText: actionText || '选择文件',
        };

      case 'default':
      default:
        return {
          icon: <FileX size={36} className="text-slate-400" />,
          title: title || '暂无数据',
          description: description || '当前没有可显示的内容',
          actionText: actionText || '刷新',
        };
    }
  };

  const config = getEmptyConfig();

  return (
    <div className={`flex items-center justify-center p-8 ${className}`}>
      <Card className="w-full max-w-md border border-white/40 bg-white/82 p-8 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-100">
            {image || config.icon}
          </div>
          <div className="mt-5 text-lg font-medium text-slate-700">{config.title}</div>
          <div className="mt-2 text-sm leading-6 text-slate-400">{config.description}</div>
          {onAction ? (
            <div className="mt-5">
              <Button variant="primary" onClick={onAction}>
                {config.actionText}
              </Button>
            </div>
          ) : null}
        </div>
      </Card>
    </div>
  );
};

export default EmptyState;
