/**
 * 路由配置
 */
import { Suspense, lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import LoadingState from '@/components/common/LoadingState';

const Home = lazy(() =>
  import('@/pages/Home').then((module) => ({ default: module.Home })),
);
const Progress = lazy(() => import('@/pages/Progress'));
const Result = lazy(() => import('@/pages/Result'));

const withPageSuspense = (element: React.ReactNode) => (
  <Suspense fallback={<LoadingState type="page" tip="页面加载中..." size="large" />}>
    {element}
  </Suspense>
);

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: withPageSuspense(<Home />),
      },
      {
        path: 'progress/:fileId',
        element: withPageSuspense(<Progress />),
      },
      {
        path: 'result/:fileId',
        element: withPageSuspense(<Result />),
      },
      {
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);

export default router;
