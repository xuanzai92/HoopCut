import React from 'react';
import { Outlet } from 'react-router-dom';

export const AppLayout: React.FC = () => {
  return (
    <main className="min-h-screen">
      <Outlet />
    </main>
  );
};
