/**
 * HTTP 客户端配置
 */
import axios from 'axios';
import type {
  AxiosError,
  AxiosInstance,
  AxiosProgressEvent,
  AxiosRequestConfig,
  AxiosResponse,
} from 'axios';
import { API_CONFIG, ERROR_MESSAGES } from '@/utils/constants';

export class HttpRequestError extends Error {
  status?: number;
  code?: string;

  constructor(message: string, options?: { status?: number; code?: string }) {
    super(message);
    this.name = 'HttpRequestError';
    this.status = options?.status;
    this.code = options?.code;
  }
}

// 创建 axios 实例
const createHttpClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_CONFIG.BASE_URL,
    timeout: API_CONFIG.TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // 请求拦截器
  client.interceptors.request.use(
    (config) => {
      // 添加请求时间戳
      config.metadata = { startTime: Date.now() };
      
      // 如果是文件上传，设置正确的 Content-Type
      if (config.data instanceof FormData) {
        config.headers['Content-Type'] = 'multipart/form-data';
      }
      
      console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    },
    (error) => {
      console.error('❌ Request Error:', error);
      return Promise.reject(error);
    }
  );

  // 响应拦截器
  client.interceptors.response.use(
    (response: AxiosResponse) => {
      const duration = Date.now() - (response.config.metadata?.startTime || 0);
      console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`);
      
      return response;
    },
    (error: AxiosError) => {
      const duration = Date.now() - (error.config?.metadata?.startTime || 0);
      console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`, error);
      
      return Promise.reject(handleApiError(error));
    }
  );

  return client;
};

const extractMessageFromResponse = (data: unknown): string | undefined => {
  if (!data || typeof data !== 'object') {
    return undefined;
  }

  const possibleMessage = (data as { message?: unknown }).message;
  if (typeof possibleMessage === 'string' && possibleMessage.trim()) {
    return possibleMessage;
  }

  const possibleError = (data as { error?: unknown }).error;
  if (typeof possibleError === 'string' && possibleError.trim()) {
    return possibleError;
  }

  return undefined;
};

// 错误处理函数
const handleApiError = (error: AxiosError): Error => {
  if (error.response) {
    // 服务器响应错误
    const { status, data } = error.response;
    const message = extractMessageFromResponse(data) || ERROR_MESSAGES.UNKNOWN_ERROR;
    
    switch (status) {
      case 400:
        return new HttpRequestError(message || '请求参数错误', { status });
      case 401:
        return new HttpRequestError('未授权访问', { status });
      case 403:
        return new HttpRequestError('访问被拒绝', { status });
      case 404:
        return new HttpRequestError('请求的资源不存在', { status });
      case 413:
        return new HttpRequestError(ERROR_MESSAGES.FILE_TOO_LARGE, { status });
      case 422:
        return new HttpRequestError(message || '请求数据验证失败', { status });
      case 429:
        return new HttpRequestError('请求过于频繁，请稍后再试', { status });
      case 500:
        return new HttpRequestError('服务器内部错误', { status });
      case 502:
        return new HttpRequestError('网关错误', { status });
      case 503:
        return new HttpRequestError('服务暂时不可用', { status });
      case 504:
        return new HttpRequestError('网关超时', { status });
      default:
        return new HttpRequestError(message || `服务器错误 (${status})`, { status });
    }
  } else if (error.request) {
    // 网络错误
    if (error.code === 'ECONNABORTED') {
      return new HttpRequestError('请求超时，请检查网络连接', { code: error.code });
    }
    return new HttpRequestError(ERROR_MESSAGES.NETWORK_ERROR, { code: error.code });
  } else {
    // 其他错误
    return new HttpRequestError(error.message || ERROR_MESSAGES.UNKNOWN_ERROR, { code: error.code });
  }
};

const shouldRetryError = (error: Error): boolean => {
  const status = (error as HttpRequestError).status;

  if (typeof status === 'number' && status >= 400 && status < 500 && status !== 408 && status !== 429) {
    return false;
  }

  return true;
};

// 创建 HTTP 客户端实例
export const httpClient = createHttpClient();

// 通用请求方法
export class HttpService {
  // GET 请求
  static async get<T = unknown>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await httpClient.get<T>(url, config);
    return response.data;
  }

  // POST 请求
  static async post<T = unknown, D = unknown>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<T> {
    const response = await httpClient.post<T>(url, data, config);
    return response.data;
  }

  // PUT 请求
  static async put<T = unknown, D = unknown>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<T> {
    const response = await httpClient.put<T>(url, data, config);
    return response.data;
  }

  // DELETE 请求
  static async delete<T = unknown>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await httpClient.delete<T>(url, config);
    return response.data;
  }

  // 文件上传
  static async upload<T = unknown>(
    url: string,
    formData: FormData,
    onUploadProgress?: (progressEvent: AxiosProgressEvent) => void
  ): Promise<T> {
    const response = await httpClient.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
    return response.data;
  }

  // 文件下载
  static async download(
    url: string,
    filename?: string,
    onDownloadProgress?: (progressEvent: AxiosProgressEvent) => void
  ): Promise<Blob> {
    const response = await httpClient.get(url, {
      responseType: 'blob',
      onDownloadProgress,
    });
    
    // 如果提供了文件名，自动下载
    if (filename) {
      const blob = new Blob([response.data]);
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    }
    
    return response.data;
  }

  static async postDownload<D = unknown>(
    url: string,
    data: D,
    filename: string
  ): Promise<Blob> {
    const response = await httpClient.post(url, data, {
      responseType: 'blob',
    });
    const blob = new Blob([response.data]);
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
    return blob;
  }
}

// 重试机制
export const withRetry = async <T>(
  operation: () => Promise<T>,
  maxAttempts: number = API_CONFIG.RETRY_ATTEMPTS,
  delay: number = API_CONFIG.RETRY_DELAY
): Promise<T> => {
  let lastError: Error;
  
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;

      if (!shouldRetryError(lastError)) {
        throw lastError;
      }
      
      if (attempt === maxAttempts) {
        throw lastError;
      }
      
      // 指数退避
      const waitTime = delay * Math.pow(2, attempt - 1);
      console.warn(`请求失败，${waitTime}ms 后重试 (${attempt}/${maxAttempts}):`, error);
      
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }
  
  throw lastError!;
};

// 扩展 AxiosRequestConfig 类型以支持 metadata
declare module 'axios' {
  interface AxiosRequestConfig {
    metadata?: {
      startTime: number;
    };
  }
}
