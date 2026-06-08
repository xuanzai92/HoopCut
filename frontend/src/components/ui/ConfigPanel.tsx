/**
 * 参数设置面板组件
 */
import React from 'react';
import { Button, Card, Chip } from '@heroui/react';
import { RotateCcw, Settings2 } from 'lucide-react';
import type { ProcessingConfig } from '@/types';

interface ConfigPanelProps {
  config: ProcessingConfig;
  onChange: (config: ProcessingConfig) => void;
  disabled?: boolean;
  className?: string;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  config,
  onChange,
  disabled = false,
  className = '',
}) => {
  const handleConfigChange = (field: keyof ProcessingConfig, value: number) => {
    const newConfig = { ...config, [field]: value };
    onChange(newConfig);
  };

  // 重置为默认配置
  const resetToDefaults = () => {
    const defaultConfig: ProcessingConfig = {
      beforeSeconds: 3,
      afterSeconds: 1,
    };
    onChange(defaultConfig);
  };

  return (
    <Card className={`border border-white/12 bg-white/80 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur ${className}`}>
      <div className="flex items-start justify-between gap-4 border-b border-slate-200/80 px-6 py-5">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-slate-900">
            <Settings2 size={18} className="text-orange-500" />
            <span className="text-base font-semibold">剪辑参数</span>
          </div>
          <p className="text-sm text-slate-500">
            控制每个进球片段前后的保留时间，决定最后成片的节奏感。
          </p>
        </div>
        <Button
          onClick={resetToDefaults}
          isDisabled={disabled}
          size="sm"
          variant="ghost"
        >
          <span className="inline-flex items-center gap-2">
            <RotateCcw size={14} />
            重置默认
          </span>
        </Button>
      </div>

      <div className="space-y-6 px-6 py-6">
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-slate-900">进球前保留时间</p>
              <p className="text-sm text-slate-500">保留进攻发起、跑位和出手前的关键片段。</p>
            </div>
            <Chip color="warning" variant="soft">
              {config.beforeSeconds.toFixed(1)}s
            </Chip>
          </div>
          <input
            type="range"
            min={1}
            max={15}
            step={0.5}
            value={config.beforeSeconds}
            disabled={disabled}
            onChange={(event) => handleConfigChange('beforeSeconds', Number(event.target.value))}
            className="hero-range"
          />
          <div className="flex justify-between text-xs text-slate-400">
            <span>1s</span>
            <span>15s</span>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-slate-900">进球后保留时间</p>
              <p className="text-sm text-slate-500">保留庆祝、回防和镜头收尾，避免成片太硬切。</p>
            </div>
            <Chip color="success" variant="soft">
              {config.afterSeconds.toFixed(1)}s
            </Chip>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={config.afterSeconds}
            disabled={disabled}
            onChange={(event) => handleConfigChange('afterSeconds', Number(event.target.value))}
            className="hero-range"
          />
          <div className="flex justify-between text-xs text-slate-400">
            <span>1s</span>
            <span>10s</span>
          </div>
        </div>

        <div className="rounded-2xl border border-orange-100 bg-orange-50/70 p-4 text-sm text-slate-600">
          默认配置适合大多数半场和全场比赛素材。如果原视频节奏更快，建议适当缩短前置时间。
        </div>
      </div>
    </Card>
  );
};
