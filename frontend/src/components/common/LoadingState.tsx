import React from 'react';
import { Card, Spinner } from '@heroui/react';

interface LoadingStateProps {
  type?: 'spin' | 'skeleton' | 'card' | 'page';
  loading?: boolean;
  tip?: string;
  size?: 'small' | 'default' | 'large';
  children?: React.ReactNode;
  className?: string;
}

const spinnerSizeMap = {
  small: 'sm',
  default: 'md',
  large: 'lg',
} as const;

const LoadingState: React.FC<LoadingStateProps> = ({
  type = 'spin',
  loading = true,
  tip = '加载中...',
  size = 'default',
  children,
  className = '',
}) => {
  if (!loading && children) {
    return <>{children}</>;
  }

  if (!loading) {
    return null;
  }

  const content = (
    <div className="flex flex-col items-center justify-center gap-4 p-8 text-center">
      <Spinner size={spinnerSizeMap[size]} />
      <div className="text-sm text-slate-500">{tip}</div>
    </div>
  );

  if (type === 'page') {
    return (
      <div className={`flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)] px-4 ${className}`}>
        <Card className="w-full max-w-md border border-white/40 bg-white/80 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur">
          {content}
        </Card>
      </div>
    );
  }

  if (type === 'card' || type === 'skeleton') {
    return (
      <Card className={`border border-white/40 bg-white/80 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur ${className}`}>
        {content}
      </Card>
    );
  }

  return <div className={className}>{content}</div>;
};

export default LoadingState;
