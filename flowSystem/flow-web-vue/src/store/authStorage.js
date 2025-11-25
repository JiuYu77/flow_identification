const ACCESS_TOKEN_KEY = 'access_token'
const ACCESS_TOKEN_EXPIRES_KEY = 'access_token_expires'
const USER_INFO_KEY = 'user_info'
const PRE_AUTH_PATH_KEY = 'preAuthPath'

const authStorage = {
    // 存储 Access Token (内存 + sessionStorage + localStorage), 刷新页面时不丢失, 新标签页/新窗口也能访问
    // expiresIn 有效时长, 单位：秒
    setAccessToken(token, expiresIn) {
        // 存储access_token
        window._accessToken = token // 内存存储（主要使用）
        sessionStorage.setItem(ACCESS_TOKEN_KEY, token) // sessionStorage 作为备份
        localStorage.setItem(ACCESS_TOKEN_KEY, token)   // localStorage 作为备份, 存储在 外部存储设备 如 硬盘

        // 存储过期时间（单位：毫秒 milliseconds）
        const expiresAt = Date.now() + expiresIn * 1000
        sessionStorage.setItem(ACCESS_TOKEN_EXPIRES_KEY, expiresAt)
        localStorage.setItem(ACCESS_TOKEN_EXPIRES_KEY, expiresAt)
    },
    // 获取 Access Token
    getAccessToken() {
        return window._accessToken || sessionStorage.getItem(ACCESS_TOKEN_KEY) || localStorage.getItem(ACCESS_TOKEN_KEY)
    },
    // 获取 Access Token 过期时间
    getAccessTokenExpires() {
        return sessionStorage.getItem(ACCESS_TOKEN_EXPIRES_KEY) || localStorage.getItem(ACCESS_TOKEN_EXPIRES_KEY)
    },
    // 检查 Access Token 是否过期
    // 返回值：true 表示已过期，false 表示未过期
    isAccessTokenExpired() {
        const expiresAt = this.getAccessTokenExpires()
        if (!expiresAt) {
            return true // 如果没有过期时间，认为已过期
        }
        return Date.now() >= parseInt(expiresAt, 10)
    },
    // 存储用户信息
    setUserInfo(userInfo) {
        const safeUserInfo = {
            ...userInfo, // 保留其他字段
            // 不存储敏感信息（安全考虑）
            password: undefined, // 移除密码
        }
        sessionStorage.setItem(USER_INFO_KEY, JSON.stringify(safeUserInfo))
        localStorage.setItem(USER_INFO_KEY, JSON.stringify(safeUserInfo))
    },
    // 获取用户信息
    getUserInfo() {
        const userInfo = sessionStorage.getItem(USER_INFO_KEY) || localStorage.getItem(USER_INFO_KEY)
        return  userInfo ? JSON.parse(userInfo) : null
    },
    // 清除所有认证信息
    clearAuth() {
        delete window._accessToken
        sessionStorage.removeItem(ACCESS_TOKEN_KEY)
        sessionStorage.removeItem(USER_INFO_KEY)
        sessionStorage.removeItem(PRE_AUTH_PATH_KEY)
        sessionStorage.removeItem(ACCESS_TOKEN_EXPIRES_KEY)

        localStorage.removeItem(ACCESS_TOKEN_KEY)
        localStorage.removeItem(USER_INFO_KEY)
        localStorage.removeItem(ACCESS_TOKEN_EXPIRES_KEY)
    },
    // 保存刷新前的路径
    setPreAuthPath(path) {
        sessionStorage.setItem(PRE_AUTH_PATH_KEY, path)
    },
    // 获取刷新前的路径
    getPreAuthPath() {
        return sessionStorage.getItem(PRE_AUTH_PATH_KEY) || '/'
    },
}

export default authStorage;
