/**
 * 发送 POST 请求
 * @param {string} url - 请求地址
 * @param {any} body - 请求数据
 * @param {Object} options - 额外配置选项
 * @returns {Promise} - 返回 Promise 对象
 */
export async function post(url, body, options = {}, flag = 'json') {
  try {
    let requestBody = body;

    if (!options.headers || options.headers == undefined) {
      options.headers = {};
    }

    if (flag === 'json') {
      requestBody = JSON.stringify(body);
      options.headers['Content-Type'] = 'application/json';
    }

    const defaultOptions = {
      method: 'POST',
      body: requestBody,
      ...options, // 其他配置 headers 等
    };

    console.log('POST request options:', defaultOptions);

    const response = await fetch(url, defaultOptions);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    }

    return await response.text();
  } catch (error) {
    console.error('POST request failed:', error);
    throw error; // 抛出错误，让调用者处理
  }
}

/**
 * 发送 GET 请求
 * @param {string} url - 请求地址
 * @param {Object} options - 额外配置选项
 * @returns {Promise} - 返回 Promise 对象
 */
export async function get(url, options = {}) {
  try {
    // 提取 headers，避免被 ...options 覆盖
    const { headers, ...otherOptions } = options;
    
    const defaultOptions = {
      method: 'GET',
      ...(headers && { headers }),
      ...otherOptions, // 其他配置：credentials, cache, redirect, signal, mode 等
    };

    // GET 请求不应该有 body（如果有的话会被忽略）
    delete defaultOptions.body;

    const response = await fetch(url, defaultOptions);

    if (!response.ok) {
      throw new Error(`HTTP  error! status: ${response.status}`);
    }
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    }

    return await response.text();
  } catch (error) {
    console.error('GET request failed:', error);
    throw error; // 抛出错误，让调用者处理
  }
}

import authStorage from "@/store/authStorage";

/**
 * 请求拦截器 - 自动添加认证头信息
 * @param {Object} options - 原始请求选项
 * @returns {Object} - 添加认证头后的请求选项
 */
export function requestInterceptor(options = {}) {
    const token = authStorage.getAccessToken();
    const userInfo = authStorage.getUserInfo();
    console.log("requestInterceptor....")

    // 如果存在认证信息，自动添加到请求头
    if (token && userInfo) {
        return {
            ...options,
            headers: {
                // 'Authorization': `Bearer ${userStore.access_token}`  // OAuth2 认证
                'Authorization': token,
                'X-User-ID': userInfo.id.toString(),
                ...options.headers
            }
        };
    }

    return options;
}
