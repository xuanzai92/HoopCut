/**
 * 网络状态检测组件
 * 监控网络连接状态并显示相应提示
 */
import React, { useState, useEffect } from 'react';
import { Alert, Button } from '@heroui/react';
import { RefreshCcw, WifiOff } from 'lucide-react';

interface NetworkStatusProps {
  onRetry?: () => void;
}

const NetworkStatus: React.FC<NetworkStatusProps> = ({ onRetry }) => {
  const [showOfflineAlert, setShowOfflineAlert] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setShowOfflineAlert(false);
    };

    const handleOffline = () => {
      setShowOfflineAlert(true);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // 初始检查
    if (!navigator.onLine) {
      setShowOfflineAlert(true);
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRetry = () => {
    if (onRetry) {
      onRetry();
    } else {
      window.location.reload();
    }
  };

  if (!showOfflineAlert) return null;

  return (
    <div className="fixed left-1/2 top-4 z-50 w-full max-w-md -translate-x-1/2 px-4">
      <Alert status="danger" className="shadow-lg">
        <WifiOff size={18} className="shrink-0" />
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="font-medium text-current">网络连接已断开</div>
          <div className="text-sm text-current/80">请检查您的网络连接，然后重试。</div>
          <div>
            <Button variant="danger-soft" onClick={handleRetry}>
              <span className="inline-flex items-center gap-2">
                <RefreshCcw size={14} />
                重试
              </span>
            </Button>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowOfflineAlert(false)}
          className="ml-2 text-current/60 transition hover:text-current"
          aria-label="关闭网络提示"
        >
          ×
        </button>
      </Alert>
    </div>
  );
};

export default NetworkStatus;
