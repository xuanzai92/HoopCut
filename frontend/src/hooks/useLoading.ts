/**
 * 加载状态管理 Hook
 * 用于管理组件的加载状态
 */
import { useState, useCallback } from 'react';

interface UseLoadingReturn {
  loading: boolean;
  setLoading: (loading: boolean) => void;
  withLoading: <TArgs extends unknown[], R>(
    fn: (...args: TArgs) => Promise<R>
  ) => (...args: TArgs) => Promise<R>;
}

export const useLoading = (initialLoading = false): UseLoadingReturn => {
  const [loading, setLoading] = useState(initialLoading);

  const withLoading = useCallback(
    <TArgs extends unknown[], R>(fn: (...args: TArgs) => Promise<R>) => {
      return async (...args: TArgs): Promise<R> => {
        setLoading(true);
        try {
          const result = await fn(...args);
          return result;
        } finally {
          setLoading(false);
        }
      };
    },
    []
  );

  return {
    loading,
    setLoading,
    withLoading,
  };
};

export default useLoading;
