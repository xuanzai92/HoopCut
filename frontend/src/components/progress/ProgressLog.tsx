import React from 'react';
import { Card } from '@heroui/react';

interface ProgressLogProps {
  logs: string[];
  className?: string;
}

export const ProgressLog: React.FC<ProgressLogProps> = ({ logs, className = '' }) => {
  const items = logs.slice(-6);

  return (
    <Card className={`border border-white/40 bg-white/82 shadow-[0_16px_50px_rgba(15,23,42,0.06)] ${className}`}>
      <div className="border-b border-slate-200/80 px-5 py-4">
        <h3 className="text-base font-semibold text-slate-900">实时日志</h3>
      </div>
      <div className="max-h-52 overflow-y-auto px-5 py-4">
        {items.length === 0 ? (
          <div className="text-sm text-slate-500">暂无日志</div>
        ) : (
          <ul className="space-y-2">
            {items.map((line, idx) => (
              <li key={`${idx}-${line}`} className="rounded-xl bg-slate-50 px-3 py-2 text-sm text-slate-700">
                {line}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
};

export default ProgressLog;
