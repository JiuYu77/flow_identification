import { defineStore } from "pinia";
import authStorage from "./authStorage.js";
import router from "@/router/index.js";
import { ElMessage } from 'element-plus';
import { get, post, requestInterceptor } from "@/assets/js/http.js";


export const Admin_Identity = 1; // 1 是管理员

export const useUserStore = defineStore('user', {
    state: () => ({
        password: '',
        id: 0,
        username: '',
        email: '',
        nickname: '',
        avatar: '',
        identity: 0, // 1 管理员、2 普通用户
        created_at: '',
        updated_at: '',
        token: '',

        access_token: '',
	    refresh_token: '',
        expires_in: 0,

        isLoggedIn: false,

        loading: false,
    }),
    getters: {
        getEmail(state) {
            return state.email;
        },
        isAuthenticated(state) {
            return state.isLoggedIn;
        },
        getIdentity(){
            return this.identity;
        },
        gets(){
            return {
                id: this.id,
                username: this.username,
                email: this.email,
                nickname: this.nickname,
                avatar: this.avatar,
                identity: this.identity,
                created_at: this.created_at,
                updated_at: this.updated_at
            }
        }
    },
    actions: {
        setEmail(email) {
            this.email = email;
        },
        setLoginStatus(status) {
            this.isLoggedIn = status;
        },
        setToken(token) {
            this.token = token;
        },
        initUser(){
            const userInfo = authStorage.getUserInfo()
            const access_token = authStorage.getAccessToken()
            const access_token_expires_in = authStorage.getAccessTokenExpires()

            if(!userInfo){
                console.log('用户信息不存在')
                return
            }
            if(!access_token){
                console.log('访问令牌不存在')
                return
            }

            this.id = userInfo.id;
            this.email = userInfo.email;
            this.nickname = userInfo.nickname;
            this.avatar = userInfo.avatar;
            this.identity = userInfo.identity;
            this.access_token = access_token;
            this.expires_in = access_token_expires_in;
        },
        // 初始化认证状态
        async initializeAuth() {
            console.log('initializeAuth() 开始执行...');
            try{
                this.initUser()

                // 1. 检查本地是否有 accessToken
                const accessToken = authStorage.getAccessToken()
                if(!accessToken){
                    this.setLoginStatus(false)
                    console.log('initializeAuth: No access token found')
                    return
                }

                // 2. 验证 token 有效性（调用验证接口）
                const isValid = await this.validateToken()
                
                console.log('Token validation preAuthPath:', authStorage.getPreAuthPath())

                if(isValid){
                    if(await this.fetchUserInfo()){ // 3. 获取用户信息
                        this.setLoginStatus(true)
                        // 保持在当前页面
                        console.log('保持在当前页面11', router.currentRoute.value.path)
                        router.replace(authStorage.getPreAuthPath())
                        return
                    }
                }else{
                    // 4. Token 无效，尝试刷新 token
                    const refreshed = await this.refreshToken()
                    if(refreshed){
                        if(await this.fetchUserInfo()){
                            this.setLoginStatus(true)
                            // 保持在当前页面
                            console.log('保持在当前页面22', router.currentRoute.value.path)
                            router.replace(authStorage.getPreAuthPath())
                            return
                        }
                    } else{
                        // 5. 认证失败，注销并跳转登录页
                        this.logout()
                    }
                }
            }catch(error){
                console.error('Auth initialization failed:', error)
                this.logout()
            }finally{}
        },
        async login() {
            let userInfo, access_token, access_token_expires_in

            const res = await post('/api/user/login', {
                username: this.username,
                password: this.password
            });

            console.log('login', res);
            if(res.code == 0){
                ElMessage.success({message: "登录成功", plain: true});

                userInfo = res.data.user
                access_token =  res.data.access_token
                access_token_expires_in = res.data.expires_in
                console.log('login',"userInfo:", userInfo, "access_token:", access_token);

                authStorage.setUserInfo(userInfo)
                authStorage.setAccessToken(access_token, access_token_expires_in)
                this.initUser()
                this.setLoginStatus(true); // 更新用户状态为已登录

                router.push('/'); // 跳转到管理页面
                this.loading = false;
            } else {
                this.loading = false;
                ElMessage.error({
                    message: '登录失败，请检查用户名和密码',
                    plain: true
                });
            }
        },
        async logout() {
            try{
                const res = await post('/api/user/logout', {
                    id: this.id,
                    Authorization: this.access_token,
                    token_type: 'access'
                }, requestInterceptor());

                console.log('logout', res);
                if(res.code == 0){
                    ElMessage.success({message: "已退出登录", plain: true});
                } else {
                    console.log("...logout:", res.msg);
                    ElMessage.error({message: "需要登录", plain: true});
                }
            }catch(e){
                console.error('退出登录失败:', e)
                ElMessage.error({message: "退出登录失败", plain: true});
            }finally{
                this.reset()
                router.push('/login')
            }
        },
        reset(){
            this.id = 0;
            this.email = '';
            this.password = '';
            this.nickname = '';
            this.username = '';
            this.avatar = '';
            this.identity = 0;
            this.token = '';
            this.access_token = '';
            this.refresh_token = '';
            this.expires_in = 0;
            this.isLoggedIn = false,

            authStorage.clearAuth()
        },
        // 更新用户信息
        updateUserInfo(userInfo) {
            authStorage.setUserInfo(userInfo)
            this.initUser()
        },
        // 验证 Token 有效性
        async validateToken(){
            const res = await post('/api/auth/validate', {}, requestInterceptor());

            console.log('validateToken', res);
            if (res.code === 0){
                return true
            }else{
                return false
            }
        },
        async refreshToken(){
            const res = await post('/api/auth/refresh', {}, requestInterceptor());
            console.log('refreshToken', res);

            if (res.code === 0){
                authStorage.setAccessToken(res.data.access_token, res.data.expires_in)
                return true
            }else{
                return false
            }
        },
        // 获取当前登录用户信息
        async fetchUserInfo(){
            const res = await get('/api/user/me', requestInterceptor());
            console.log('fetchUserInfo:', res);

            if (res.code === 0){
                this.updateUserInfo(res.data)
                return true
            }else{
                return false
            }
        }
    }
})
