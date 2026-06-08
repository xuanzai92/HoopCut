// 导出所有工具函数
export * from './format';
export * from './validation';
export * from './constants';

// 通用工具函数
export const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

// 防抖函数
export const debounce = <TArgs extends unknown[]>(
  func: (...args: TArgs) => void,
  wait: number
): ((...args: TArgs) => void) => {
  let timeout: ReturnType<typeof setTimeout>;
  
  return (...args: TArgs) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// 节流函数
export const throttle = <TArgs extends unknown[]>(
  func: (...args: TArgs) => void,
  limit: number
): ((...args: TArgs) => void) => {
  let inThrottle = false;
  
  return (...args: TArgs) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
};

// 生成唯一ID
export const generateId = (): string => {
  return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
};

type PlainObject = Record<string, unknown>;

// 深拷贝
export const deepClone = <T>(obj: T): T => {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as T;
  if (Array.isArray(obj)) return obj.map((item) => deepClone(item)) as T;
  if (typeof obj === 'object') {
    const clonedObj: PlainObject = {};
    for (const [key, value] of Object.entries(obj as PlainObject)) {
      clonedObj[key] = deepClone(value);
    }
    return clonedObj as T;
  }
  return obj;
};

// 对象合并
export const mergeObjects = <T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T => {
  return Object.assign({}, target, ...sources);
};

// 数组去重
export const uniqueArray = <T>(array: T[], key?: keyof T): T[] => {
  if (!key) {
    return [...new Set(array)];
  }
  
  const seen = new Set();
  return array.filter(item => {
    const value = item[key];
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
};

// 获取嵌套对象属性
export const getNestedValue = (
  obj: unknown,
  path: string,
  defaultValue?: unknown
): unknown => {
  const keys = path.split('.');
  let result: unknown = obj;
  
  for (const key of keys) {
    if (
      result === null ||
      result === undefined ||
      typeof result !== 'object' ||
      !(key in result)
    ) {
      return defaultValue;
    }
    result = (result as PlainObject)[key];
  }
  
  return result;
};

// 设置嵌套对象属性
export const setNestedValue = (obj: PlainObject, path: string, value: unknown): void => {
  const keys = path.split('.');
  const lastKey = keys.pop()!;
  let current: PlainObject = obj;
  
  for (const key of keys) {
    if (!(key in current) || typeof current[key] !== 'object' || current[key] === null) {
      current[key] = {};
    }
    current = current[key] as PlainObject;
  }
  
  current[lastKey] = value;
};

// 检查是否为空值
export const isEmpty = (value: unknown): boolean => {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') return value.trim() === '';
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
};

// 类名合并工具
export const classNames = (...classes: (string | undefined | null | false)[]): string => {
  return classes.filter(Boolean).join(' ');
};

// 下载文件
export const downloadFile = (url: string, filename: string): void => {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// 复制到剪贴板
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      // 降级方案
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const result = document.execCommand('copy');
      document.body.removeChild(textArea);
      return result;
    }
  } catch (error) {
    console.error('复制到剪贴板失败:', error);
    return false;
  }
};
