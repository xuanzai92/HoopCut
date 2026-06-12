import React, { useEffect, useState } from 'react';
import { Button, Card, Chip } from '@heroui/react';
import { Sparkles } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import ErrorAlert from '@/components/common/ErrorAlert';
import { ManualWorkflow } from '@/components/workflows/ManualWorkflow';
import { AutoWorkflow } from '@/components/workflows/AutoWorkflow';
import { ApiService } from '@/services';
import type { ProcessingMode, ReusableVideoSource } from '@/types';
import { useErrorHandler } from '@/hooks';

const MODE_OPTIONS: Array<{
  key: ProcessingMode;
  label: string;
  title: string;
  description: string;
}> = [
  {
    key: 'manual',
    label: '默认主流程',
    title: '自己选时间点剪片',
    description: '适合你已经知道要剪哪些球，想要更快、更可控、更少干扰。',
  },
  {
    key: 'auto',
    label: '可选高级模式',
    title: 'AI 自动找目标片段',
    description: '只有你不想手动找时间点时，再让系统跑人物锁定、找球和归因。',
  },
];

export const Home: React.FC = () => {
  const [activeMode, setActiveMode] = useState<ProcessingMode>('manual');
  const [reusableSource, setReusableSource] = useState<ReusableVideoSource | null>(null);
  const [isLoadingReusableSource, setIsLoadingReusableSource] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const reuseTaskId = searchParams.get('reuseTaskId')?.trim() || '';
  const requestedMode = searchParams.get('mode')?.trim() as ProcessingMode | null;
  const { error, hasError, clearError, handleError } = useErrorHandler();

  useEffect(() => {
    if (requestedMode === 'manual' || requestedMode === 'auto') {
      setActiveMode(requestedMode);
    }
  }, [requestedMode]);

  useEffect(() => {
    let cancelled = false;

    if (!reuseTaskId) {
      setReusableSource(null);
      setIsLoadingReusableSource(false);
      return () => {
        cancelled = true;
      };
    }

    setIsLoadingReusableSource(true);
    clearError();

    void ApiService.getReusableSource(reuseTaskId)
      .then((source) => {
        if (cancelled) {
          return;
        }

        setReusableSource(source);
        if (source.processingMode === 'auto' || source.processingMode === 'manual') {
          setActiveMode(source.processingMode);
        }
      })
      .catch((nextError) => {
        if (cancelled) {
          return;
        }

        setReusableSource(null);
        handleError(nextError instanceof Error ? nextError : new Error('加载源视频失败'));
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingReusableSource(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [reuseTaskId, clearError, handleError]);

  const updateMode = (nextMode: ProcessingMode) => {
    setActiveMode(nextMode);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('mode', nextMode);
    setSearchParams(nextParams, { replace: true });
  };

  const clearReusableMode = () => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete('reuseTaskId');
    setSearchParams(nextParams, { replace: true });
    setReusableSource(null);
  };

  return (
    <div className="home-page min-h-screen overflow-x-hidden bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)]">
      {hasError && error ? (
        <div className="fixed left-1/2 top-4 z-50 w-full max-w-md -translate-x-1/2 px-4">
          <ErrorAlert
            title="加载失败"
            message={error.message}
            type="error"
            showIcon
            closable
            onClose={clearError}
          />
        </div>
      ) : null}

      <div className="relative isolate">
        <div className="absolute left-1/2 top-0 -z-10 h-[480px] w-[480px] -translate-x-1/2 rounded-full bg-orange-300/20 blur-3xl" />
        <div className="absolute right-[-120px] top-[120px] -z-10 h-[340px] w-[340px] rounded-full bg-indigo-300/18 blur-3xl" />

        <div className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-4 pb-12 pt-8 sm:px-6 lg:px-8 lg:pt-12">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-950 text-lg font-semibold text-white shadow-lg shadow-orange-500/20">
                H
              </div>
              <div>
                <p className="text-lg font-semibold tracking-tight text-slate-950">HoopCut</p>
                <p className="text-sm text-slate-500">篮球高光剪片工具</p>
              </div>
            </div>
            <Chip variant="soft" color="warning" className="hidden md:flex">
              本地视频处理
            </Chip>
          </header>

          <section className="grid gap-8 lg:grid-cols-[minmax(0,1.05fr)_360px] lg:items-end">
            <div className="space-y-6">
              <div className="flex flex-wrap gap-2">
                <Chip variant="soft" color="warning">默认手动时间点</Chip>
                <Chip variant="soft" color="accent">AI 自动模式可选</Chip>
              </div>

              <div className="max-w-4xl space-y-4">
                <h1 className="font-serif text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl lg:text-7xl">
                  直接自己选时间点，
                  <br />
                  把片段切出来。
                </h1>
                <p className="max-w-2xl text-base leading-8 text-slate-600 sm:text-lg">
                  默认主流程直接让你上传视频、拖到关键时刻、加入待剪列表并导出，不再先塞 AI 推荐画面和一堆解释。只有你不想自己找时间点时，再切到自动模式。
                </p>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row">
                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => document.getElementById('workflow-start')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
                  className="rounded-full px-6"
                >
                  直接开始剪片
                </Button>
                <div className="flex items-center text-sm text-slate-500">
                  手动模式更快，也更符合你现在的使用方式。
                </div>
              </div>
            </div>

            <Card className="border border-white/15 bg-slate-950 text-white shadow-[0_24px_80px_rgba(15,23,42,0.30)]">
              <div className="space-y-4 p-6">
                <div className="flex items-center justify-between">
                  <span className="text-sm uppercase tracking-[0.24em] text-slate-400">模式</span>
                  <Sparkles size={16} className="text-orange-300" />
                </div>

                {MODE_OPTIONS.map((mode) => {
                  const active = activeMode === mode.key;
                  return (
                    <button
                      key={mode.key}
                      type="button"
                      className={`w-full rounded-2xl border p-4 text-left transition ${
                        active
                          ? 'border-orange-300 bg-white/12'
                          : 'border-white/10 bg-white/5 hover:border-white/20 hover:bg-white/8'
                      }`}
                      onClick={() => updateMode(mode.key)}
                    >
                      <div className="mb-2 text-xs font-semibold tracking-[0.22em] text-orange-300">
                        {mode.label}
                      </div>
                      <div className="text-base font-semibold text-white">{mode.title}</div>
                      <div className="mt-1 text-sm leading-6 text-slate-400">{mode.description}</div>
                    </button>
                  );
                })}
              </div>
            </Card>
          </section>
        </div>
      </div>

      <div id="workflow-start" className="mx-auto w-full max-w-7xl px-4 pb-16 sm:px-6 lg:px-8">
        {isLoadingReusableSource ? (
          <Card className="border border-white/40 bg-white/82 p-8 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
            <div className="text-sm text-slate-600">正在载入上一次处理的视频...</div>
          </Card>
        ) : activeMode === 'manual' ? (
          <ManualWorkflow
            initialSource={reusableSource}
            onExitReusableSource={clearReusableMode}
          />
        ) : (
          <AutoWorkflow
            initialSource={reusableSource}
            onExitReusableSource={clearReusableMode}
          />
        )}
      </div>
    </div>
  );
};

export default Home;
