/**
 * 错误提示组件
 * 用于显示各种错误状态和提示信息
 */
import React from 'react';
import { Alert, Button } from '@heroui/react';
import { Home, RefreshCcw, TriangleAlert } from 'lucide-react';

interface ErrorAlertProps {
  title?: string;
  message?: string;
  type?: 'error' | 'warning' | 'info';
  showIcon?: boolean;
  closable?: boolean;
  onRetry?: () => void;
  onGoHome?: () => void;
  onClose?: () => void;
  retryText?: string;
  homeText?: string;
  className?: string;
}

const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = '操作失败',
  message = '请稍后重试或联系技术支持',
  type = 'error',
  showIcon = true,
  closable = true,
  onRetry,
  onGoHome,
  onClose,
  retryText = '重试',
  homeText = '返回首页',
  className = '',
}) => {
  const status = type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'accent';
  const actions: React.ReactNode[] = [];

  if (onRetry) {
    actions.push(
      <Button
        key="retry"
        size="sm"
        variant={type === 'error' ? 'danger' : 'secondary'}
        onClick={onRetry}
      >
        <span className="inline-flex items-center gap-2">
          <RefreshCcw size={14} />
          {retryText}
        </span>
      </Button>
    );
  }

  if (onGoHome) {
    actions.push(
      <Button
        key="home"
        size="sm"
        variant="ghost"
        onClick={onGoHome}
      >
        <span className="inline-flex items-center gap-2">
          <Home size={14} />
          {homeText}
        </span>
      </Button>
    );
  }

  return (
    <Alert status={status} className={className}>
      {showIcon ? <TriangleAlert size={18} className="shrink-0" /> : null}
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="font-medium text-current">{title}</div>
        <div className="text-sm text-current/80">{message}</div>
        {actions.length > 0 ? <div className="flex flex-wrap gap-2 pt-1">{actions}</div> : null}
      </div>
      {closable ? (
        <button
          type="button"
          onClick={onClose}
          className="ml-2 text-current/60 transition hover:text-current"
          aria-label="关闭错误提示"
        >
          ×
        </button>
      ) : null}
    </Alert>
  );
};

export default ErrorAlert;
