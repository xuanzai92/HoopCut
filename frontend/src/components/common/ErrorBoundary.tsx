/**
 * 错误边界组件
 * 用于捕获和处理React组件树中的JavaScript错误
 */
import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Button, Card } from '@heroui/react';
import { AlertTriangle, Home, RefreshCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // 更新 state 使下一次渲染能够显示降级后的 UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // 记录错误信息
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // 可以将错误日志上报给服务
    this.logErrorToService(error, errorInfo);
  }

  logErrorToService = (error: Error, errorInfo: ErrorInfo) => {
    // 这里可以集成错误监控服务，如 Sentry
    try {
      const errorData = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      };
      
      // 发送到错误监控服务
      console.log('Error logged:', errorData);
    } catch (logError) {
      console.error('Failed to log error:', logError);
    }
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      // 如果有自定义的 fallback UI，使用它
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // 默认的错误 UI
      return (
        <div className="flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)] p-4">
          <div className="max-w-xl w-full">
            <Card className="border border-white/40 bg-white/82 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur">
              <div className="text-center">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-rose-50 text-rose-500">
                  <AlertTriangle size={34} />
                </div>
                <h1 className="mt-5 text-3xl font-semibold tracking-tight text-slate-950">页面出现错误</h1>
                <p className="mt-3 text-sm leading-6 text-slate-500">
                  抱歉，页面遇到了一些问题。您可以尝试重试、刷新页面或返回首页。
                </p>
                <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
                  <Button variant="primary" onClick={this.handleRetry}>
                    <span className="inline-flex items-center gap-2">
                      <RefreshCcw size={16} />
                      重试
                    </span>
                  </Button>
                  <Button variant="secondary" onClick={this.handleReload}>
                    <span className="inline-flex items-center gap-2">
                      <RefreshCcw size={16} />
                      刷新页面
                    </span>
                  </Button>
                  <Button variant="ghost" onClick={this.handleGoHome}>
                    <span className="inline-flex items-center gap-2">
                      <Home size={16} />
                      返回首页
                    </span>
                  </Button>
                </div>
              </div>
            </Card>

            {/* 开发环境下显示详细错误信息 */}
            {import.meta.env.DEV && this.state.error && (
              <div className="mt-8 rounded-lg border border-red-200 bg-red-50 p-4">
                <h3 className="text-red-800 font-semibold mb-2">错误详情（仅开发环境显示）</h3>
                <div className="text-sm text-red-700 space-y-2">
                  <div>
                    <strong>错误信息：</strong>
                    <pre className="mt-1 text-xs bg-red-100 p-2 rounded overflow-auto">
                      {this.state.error.message}
                    </pre>
                  </div>
                  {this.state.error.stack && (
                    <div>
                      <strong>错误堆栈：</strong>
                      <pre className="mt-1 text-xs bg-red-100 p-2 rounded overflow-auto max-h-40">
                        {this.state.error.stack}
                      </pre>
                    </div>
                  )}
                  {this.state.errorInfo?.componentStack && (
                    <div>
                      <strong>组件堆栈：</strong>
                      <pre className="mt-1 text-xs bg-red-100 p-2 rounded overflow-auto max-h-40">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
