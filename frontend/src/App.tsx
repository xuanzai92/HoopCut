import { Suspense, lazy } from 'react';
import { RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toast } from '@heroui/react';
import { router } from '@/router';
import ErrorBoundary from '@/components/common/ErrorBoundary';
import './index.css';
import './App.css';

// 创建 QueryClient 实例
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5分钟
    },
    mutations: {
      onError: (error) => {
        console.error('Mutation error:', error);
        Toast.toast.danger(error instanceof Error ? error.message : '请稍后重试');
      },
    },
  },
});

const showQueryDevtools =
  import.meta.env.DEV && import.meta.env.VITE_SHOW_QUERY_DEVTOOLS === 'true';

const QueryDevtools = showQueryDevtools
  ? lazy(() =>
      import('@tanstack/react-query-devtools').then((module) => ({
        default: module.ReactQueryDevtools,
      })),
    )
  : null;

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
        <Toast.Provider placement="top" />
        {QueryDevtools ? (
          <Suspense fallback={null}>
            <QueryDevtools initialIsOpen={false} />
          </Suspense>
        ) : null}
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
