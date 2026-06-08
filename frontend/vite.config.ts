import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendTarget = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:5050'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined
          }

          if (
            id.includes('/react/') ||
            id.includes('/react-dom/') ||
            id.includes('scheduler')
          ) {
            return 'react-core'
          }

          if (id.includes('react-router')) {
            return 'router'
          }

          if (id.includes('@tanstack')) {
            return 'query'
          }

          if (
            id.includes('@heroui') ||
            id.includes('@react-aria') ||
            id.includes('react-aria-components') ||
            id.includes('@react-stately') ||
            id.includes('@internationalized')
          ) {
            return 'heroui-ui'
          }

          if (
            id.includes('socket.io-client') ||
            id.includes('engine.io-client') ||
            id.includes('socket.io-parser')
          ) {
            return 'socket'
          }

          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/socket.io': {
        target: backendTarget,
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
