/**
 * 全局加载组件
 * 用于显示全屏加载状态
 */
import React from 'react';
import { Card, Spinner } from '@heroui/react';

interface GlobalLoadingProps {
  loading?: boolean;
  tip?: string;
  size?: 'small' | 'default' | 'large';
  className?: string;
}

const GlobalLoading: React.FC<GlobalLoadingProps> = ({
  loading = true,
  tip = '加载中...',
  size = 'large',
  className = '',
}) => {
  if (!loading) return null;

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center bg-white/72 backdrop-blur-sm ${className}`}>
      <Card className="border border-white/40 bg-white/82 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.10)]">
        <div className="text-center">
          <Spinner size={size === 'large' ? 'lg' : size === 'small' ? 'sm' : 'md'} />
          <div className="mt-4 text-sm text-slate-600">{tip}</div>
        </div>
      </Card>
    </div>
  );
};

export default GlobalLoading;
